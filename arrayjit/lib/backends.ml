open Base
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type 'context routine = {
  context : 'context;
  schedule : unit -> Tnode.work;
  bindings : Indexing.lowered_bindings;
  name : string;
}
[@@deriving sexp_of]

module type No_device_backend = sig
  type code [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool

  val init : label:string -> context
  (** [label] is usually the backend name concatenated with the device number. *)

  val finalize : context -> unit
  (** Finalizes (just) the context. *)

  val compile : ?shared:bool -> ?name:string -> Indexing.unit_bindings -> Assignments.t -> code
  (** If [~shared:true] (default [false]), the backend should prefer to do more compile work in a
      device-agnostic way. If [~shared:false], the backend can opt to postpone compiling altogether until
      [link] is called, to benefit from more optimizations. *)

  val compile_batch :
    ?shared:bool -> ?names:string array -> Indexing.unit_bindings -> Assignments.t array -> code array
  (** Unlike the [~shared] parameter, [compile_batch] vs. [compile] is mostly about improving the compile time
      and debugging convenience by generating fewer files -- ideally does not affect execution, but there can
      be backend-specific differences. *)

  val link : context -> code -> routine

  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  (** Cleans up all work on a backend. If [~unsafe_shutdown:true], releases resources, potentially making the
      backend unusable. *)

  val from_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from host to context and returns true. *)

  val to_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from context to host and returns true. *)

  val merge : ?name_prefix:string -> Tnode.t -> accum:Ops.binop -> src:context -> code option
  (** Merges the array from the source context into the destination context: [dst =: dst accum src]. If the
      array is hosted, its state on host is undefined after this operation. (A backend may choose to use the
      host array as a buffer, if that is beneficial.) [name_prefix] is prepended to the jitted function's
      name. Returns [None] if the array is not in the context. *)

  val merge_batch :
    ?name_prefixes:string array ->
    occupancy:(Tnode.t -> src_n:int -> src:context -> Utils.requirement) ->
    Tnode.t list ->
    accum:Ops.binop ->
    srcs:context array ->
    (Tnode.t, code option array) Hashtbl.t
  (** [merge_batch] vs. [merge] is purely about improving the compile time (does not affect execution). *)
end

module type Backend = sig
  include No_device_backend

  type device

  val init : device -> context
  val await : device -> unit
  val sexp_of_device : device -> Sexp.t
  val num_devices : unit -> int
  val get_device : ordinal:int -> device
  val get_ctx_device : context -> device
  val to_ordinal : device -> int
  val get_all_devices : unit -> device array
end

let forget_printbox (module Runtime : Minidebug_runtime.PrintBox_runtime) =
  (module Runtime : Minidebug_runtime.Debug_runtime)

module Multicore_backend (Backend : No_device_backend) : Backend = struct
  module Domain = Domain [@warning "-3"]

  type device = {
    next_task : (Tnode.work option ref[@sexp.opaque]);
    keep_spinning : bool ref;
    wait_for_device : (Utils.waiter[@sexp.opaque]);
    wait_for_work : (Utils.waiter[@sexp.opaque]);
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type code = Backend.code [@@deriving sexp_of]
  type context = { device : device; ctx : Backend.context } [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  let name = "multicore " ^ Backend.name
  let init device = { device; ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal) }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized

  let await device =
    assert (Domain.is_main_domain ());
    while !(device.keep_spinning) && Option.is_some !(device.next_task) do
      device.wait_for_device.await ()
    done

  let finalize { device; ctx } =
    await device;
    Backend.finalize ctx

  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link { ctx; device } code : routine =
    let task = Backend.link ctx code in
    let%diagn_sexp schedule () =
      [%log_result "Scheduling", task.name];
      let task = task.schedule () in
      let%diagn_rt_sexp work () =
        await device;
        if not !(device.keep_spinning) then invalid_arg "Multicore_backend.jit: device not available";
        device.next_task := Some task;
        device.wait_for_work.release ()
      in
      Tnode.Work work
    in
    { task with context = { ctx = task.context; device }; schedule }

  let from_host { device; ctx } =
    await device;
    Backend.from_host ctx

  let to_host { device; ctx } =
    await device;
    Backend.to_host ctx

  let merge ?name_prefix tn ~accum ~src =
    let name_prefix : string = Option.value ~default:"" @@ Option.map name_prefix ~f:(fun s -> s ^ "_") in
    let ord ctx = ctx.device.ordinal in
    let name_prefix : string = [%string "%{name_prefix}from_dev_%{ord src#Int}"] in
    Backend.merge ~name_prefix tn ~accum ~src:src.ctx

  let merge_batch ?name_prefixes ~occupancy tns ~accum ~srcs =
    let ord ctx = ctx.device.ordinal in
    let name_prefixes =
      let f prefix src = [%string "%{prefix}from_dev_%{ord src#Int}"] in
      match name_prefixes with
      | None -> Array.map srcs ~f:(f "")
      | Some prefixes -> Array.map2_exn prefixes srcs ~f:(fun p -> f @@ p ^ "_")
    in
    let devices, srcs = Array.unzip (Array.map srcs ~f:(fun s -> (s.device, s.ctx))) in
    let occupancy tn ~src_n ~src = occupancy tn ~src_n ~src:{ device = devices.(src_n); ctx = src } in
    Backend.merge_batch ~name_prefixes ~occupancy tns ~accum ~srcs

  let num_devices () = Domain.recommended_domain_count () - 1
  let global_run_no = ref 0

  let%debug_sexp spinup_device ~(ordinal : int) : device =
    Int.incr global_run_no;
    let next_task = ref None in
    let keep_spinning = ref true in
    let wait_for_device = Utils.waiter () in
    let wait_for_work = Utils.waiter () in
    let run_no = !global_run_no in
    let debug_runtime = Utils.get_debug ("dev-" ^ Int.to_string ordinal ^ "-run-" ^ Int.to_string run_no) in
    (* For detailed debugging of a worker operation, set OCANNL_SNAPSHOT_EVERY_SEC and uncomment: *)
    (* let%track_rt_sexp worker (() : unit) : unit = *)
    let worker () =
      while !keep_spinning do
        while !keep_spinning && Option.is_none !next_task do
          wait_for_work.await ()
        done;
        if !keep_spinning then (
          Tnode.run debug_runtime @@ Option.value_exn !next_task;
          next_task := None;
          wait_for_device.release ())
      done
    in
    {
      next_task;
      keep_spinning;
      wait_for_device;
      wait_for_work;
      ordinal;
      (* For detailed debugging of a worker operation, uncomment: *)
      (* domain = Domain.spawn (worker _debug_runtime); *)
      domain = Domain.spawn worker;
    }

  let devices = Array.init (num_devices ()) ~f:(fun ordinal -> spinup_device ~ordinal)
  let get_all_devices () = devices

  let%track_sexp unsafe_cleanup ?(unsafe_shutdown = false) () =
    assert (Domain.is_main_domain ());
    let cleanup ordinal device =
      await device;
      device.keep_spinning := false;
      (* Important to release after setting to not keep spinning. *)
      device.wait_for_work.release ();
      Domain.join device.domain;
      device.wait_for_device.finalize ();
      device.wait_for_work.finalize ();
      if not unsafe_shutdown then devices.(ordinal) <- spinup_device ~ordinal
    in
    Array.iteri devices ~f:cleanup;
    Backend.unsafe_cleanup ~unsafe_shutdown ()

  let get_device ~ordinal = devices.(ordinal)
  let get_ctx_device { device; _ } = device
  let to_ordinal device = device.ordinal
end

let lower_assignments ?name bindings asgns =
  let name = Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name (Indexing.bound_symbols bindings)
      asgns )

let lower_batch_assignments ?names bindings asgns_l =
  let names =
    Option.value_or_thunk names ~default:(fun () ->
        Array.map asgns_l ~f:(fun asgns -> Assignments.get_name_exn asgns))
  in
  let prefix_name = String.(strip ~drop:(equal_char '_') @@ common_prefix @@ Array.to_list names) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".cd") in
  let bound = Indexing.bound_symbols bindings in
  ( names,
    Array.map2_exn names asgns_l ~f:(fun name asgns ->
        Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns) )

module type Simple_backend = sig
  type context [@@deriving sexp_of]
  type procedure [@@deriving sexp_of]
  type ctx_arrays [@@deriving sexp_of]

  val ctx_arrays : context -> ctx_arrays

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    (unit -> unit) Indexing.bindings ->
    Low_level.optimized ->
    procedure

  val compile_batch :
    names:string array ->
    opt_ctx_arrays:ctx_arrays option ->
    (unit -> unit) Indexing.bindings ->
    Low_level.optimized array ->
    procedure array

  val link_compiled :
    context -> procedure -> context * Indexing.lowered_bindings * (unit -> Tnode.work) * string

  val merge :
    ?name_prefix:string ->
    Tnode.t ->
    accum:Ops.binop ->
    src:context ->
    (unit -> unit) Indexing.bindings ->
    procedure option

  val merge_batch :
    ?name_prefixes:string array ->
    occupancy:(Tnode.t -> src_n:int -> src:context -> Utils.requirement) ->
    Tnode.t list ->
    accum:Ops.binop ->
    srcs:context array ->
    (unit -> unit) Indexing.bindings ->
    (Tnode.t, procedure option array) Base.Hashtbl.t

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool
  val init : label:string -> context
  val finalize : context -> unit
  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  val from_host : context -> Tnode.t -> bool
  val to_host : context -> Tnode.t -> bool
end

module Simple_no_device_backend (Backend : Simple_backend) : No_device_backend = struct
  type code =
    | Postponed of { lowered : Low_level.optimized; bindings : Indexing.unit_bindings; name : string }
    | Compiled of Backend.procedure
  [@@deriving sexp_of]

  include Backend

  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?(shared = false) ?name bindings asgns =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ?names bindings asgns_l =
    let names, lowereds = lower_batch_assignments ?names bindings asgns_l in
    if shared then
      let routines = compile_batch ~names ~opt_ctx_arrays:None bindings lowereds in
      Array.map routines ~f:(fun routine -> Compiled routine)
    else Array.map2_exn names lowereds ~f:(fun name lowered -> Postponed { lowered; bindings; name })

  let link (old_context : context) code =
    let context, bindings, schedule, name =
      match code with
      | Postponed { lowered; bindings; name } ->
          let code = Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowered in
          link_compiled old_context code
      | Compiled code -> link_compiled old_context code
    in
    { context; schedule; bindings; name }

  let merge ?name_prefix tn ~accum ~(src : context) =
    let bindings = Indexing.Empty in
    merge ?name_prefix tn ~accum ~src bindings |> Option.map ~f:(fun routine -> Compiled routine)

  let merge_batch ?name_prefixes ~occupancy tns ~accum ~(srcs : context array) =
    let bindings = Indexing.Empty in
    merge_batch ?name_prefixes ~occupancy tns ~accum ~srcs bindings
    |> Hashtbl.map ~f:(fun cds -> Array.map cds ~f:(Option.map ~f:(fun routine -> Compiled routine)))
end

module Gccjit_device : No_device_backend = Simple_no_device_backend ((
  Gccjit_backend : Simple_backend with type context = Gccjit_backend.context))

module Gccjit_backend = Multicore_backend (Gccjit_device)

module Cuda_backend : Backend with type context = Cuda_backend.context = struct
  type code = Cuda_backend.code [@@deriving sexp_of]
  type context = Cuda_backend.context [@@deriving sexp_of]
  type device = Cuda_backend.device
  type nonrec routine = context routine [@@deriving sexp_of]

  open Cuda_backend

  let name = "cuda"
  let initialize = initialize
  let is_initialized = is_initialized
  let unsafe_cleanup ?unsafe_shutdown () = unsafe_cleanup ?unsafe_shutdown ()
  let init = init
  let finalize = finalize
  let sexp_of_context = sexp_of_context
  let sexp_of_device = sexp_of_device

  let compile ?shared:_ ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    compile ~name bindings lowered

  let compile_batch ?shared:_ ?names:_ _bindings _asgns_batch = failwith "NOT IMPLEMENTED YET"

  let link context code =
    let context, bindings, schedule = link context code in
    { context; schedule; bindings; name }

  let from_host = from_host
  let to_host = to_host

  let merge ?name_prefix tn ~accum ~(src : context) =
    let bindings = Indexing.Empty in
    ignore (bindings, name_prefix, tn, accum, src);
    failwith "NOT IMPLEMENTED YET"

  let merge_batch ?name_prefixes:_ ~occupancy:_ _tns ~accum:_ ~srcs:_ = failwith "NOT IMPLEMENTED YET"
  let await = await
  let num_devices = num_devices
  let get_device = get_device
  let get_ctx_device = get_ctx_device
  let to_ordinal = to_ordinal
  let get_all_devices () = Array.init (num_devices ()) ~f:(fun ordinal -> get_device ~ordinal)
end

let reinitialize (module Backend : Backend) =
  if not @@ Backend.is_initialized () then Backend.initialize ()
  else (
    Core.Gc.full_major ();
    Backend.unsafe_cleanup ())
