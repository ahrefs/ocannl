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
  type code_batch [@@deriving sexp_of]
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
    ?shared:bool ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.t array ->
    code_batch
  (** Unlike the [~shared] parameter, [compile_batch] vs. [compile] is mostly about improving the compile time
      and debugging convenience by generating fewer files -- ideally does not affect execution, but there can
      be backend-specific differences. Only array entries for which [occupancy] returns true are included. *)

  val link : context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. See also
      [supports_merge_buffers]. *)

  val link_batch : context -> code_batch -> routine option array
  (** Returns the routines for the procedures included in the code batch. All returned routines share the same
      new context. See also [supports_merge_buffers]. *)

  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  (** Cleans up all work on a backend. If [~unsafe_shutdown:true], releases resources, potentially making the
      backend unusable. *)

  val from_host_callback : context -> Tnode.t -> Tnode.work option
  val to_host_callback : context -> Tnode.t -> Tnode.work option
  val merge_callback : Tnode.t -> accum:Ops.binop -> dst:context -> src:context -> Tnode.work option

  val compile_merges :
    ?name_prefixes:string array ->
    occupancy:(Tnode.t -> dst_n:int -> dst:context -> src_n:int -> src:context -> Utils.requirement) ->
    Tnode.t list ->
    accum:Ops.binop ->
    dsts:context array ->
    srcs:context array ->
    unit
  (** Compiles the code, if any, required by [merge], as if applying [compile_batch] to [srcs] for each of
      [dsts]. Includes only the merges for for which [occupancy] does not return [Skip], and the node is
      present in both the destination and the source. *)

  val supports_merge_buffers : bool
  (** If [false], using [Ops.Merge_buffer_unsafe] in assignments will fail during {!compile} or {!link}, resp.
      {!compile_batch} or {!link_batch}. If [true], each device maintains (at most) one merge buffer, that is
      grown by calls to [compile_merges], to ensure that the merge buffer can fit the data arriving for
      {!merge}, {!merge_unsafe} tasks. *)
end

module type Backend = sig
  include No_device_backend

  val from_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from host to context and returns true. This function
      is synchronous in the sense that when it returns, the host data is no longer needed. Beware that
      depending on the backend, calling [from_host] might even synchronize all devices of the backend. *)

  val to_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from context to host and returns true. This function
      is synchronous: returns when fully complete. Beware that depending on the backend, calling [to_host]
      might even synchronize all devices of the backend. *)

  val from_host_async : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from host to context and returns true. NOTE: this
      function returns once it books (schedules) the copying task and it's the caller's responsibility to
      synchronize the device before the host's data is overwritten. *)

  val to_host_async : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from context to host and returns true. NOTE: this
      function returns once it books (schedules) the copying task and it's the caller's responsibility to
      synchronize the device before the host's data is read. *)

  val merge : Tnode.t -> accum:Ops.binop -> dst:context -> src:context -> bool
  (** Merges the array from the source context into the destination context: [dst =: dst accum src]. If the
      array is hosted, its state on host is undefined after this operation. (A backend may choose to use the
      host array as a buffer, if that is beneficial.) The corresponding tuple of [tnode, accum, dst, src] must
      be in the scope of an earlier {!compile_merges} call; otherwise, [merge] will fail, even if it would
      otherwise return [false]. Returns [false] if the array is not in both the [dst] and [src] contexts.
      Otherwise, it synchronizes the [src] device: calls [Backend.await src.device], {i then} it schedules the
      merge task on the [dst] device, then it returns [true]. NOTE: [merge] is asynchronous, it's the caller's
      responsibility to not overwrite source before the merge completes. *)

  val merge_unsafe : Tnode.t -> accum:Ops.binop -> dst:context -> src:context -> bool
  (** Merges the array from the source context into the destination context: see {!merge} for details.
      [merge_unsafe] starts with a [Backend.acknowledge src.device] call (blocking until the latest task on
      [src] starts), {i then} it schedules the merge task on the [dst] device. NOTE: [merge_unsafe] is likely
      to cause {i read before intended write} bugs. *)

  type device

  val init : device -> context

  val await : device -> unit
  (** Blocks till the device becomes idle, i.e. synchronizes the device. *)

  val acknowledge : device -> unit
  (** Blocks till the device becomes not booked (its queue becomes empty). *)

  val is_idle : device -> bool
  (** The device is currently waiting for work. See also {!is_booked}: if a device is idle, then it's not
      booked; but a device might be both not idle and not booked. *)

  val is_booked : device -> bool
  (* The device queue is not empty: the next task is waiting to start execution. See also {!is_idle}. *)

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
    is_idle : bool ref;
    next_task : (Tnode.work[@sexp.opaque]) option ref;
    keep_spinning : bool ref;
    wait_for_ackn : (Utils.waiter[@sexp.opaque]);
    wait_for_work : (Utils.waiter[@sexp.opaque]);
    wait_for_idle : (Utils.waiter[@sexp.opaque]);
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]
  type context = { device : device; ctx : Backend.context } [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  let name = "multicore " ^ Backend.name
  let init device = { device; ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal) }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let supports_merge_buffers = Backend.supports_merge_buffers
  let is_idle device = !(device.is_idle)
  let is_booked device = Option.is_some !(device.next_task)

  let await device =
    assert (Domain.is_main_domain ());
    while !(device.keep_spinning) && not !(device.is_idle) do
      device.wait_for_idle.await ()
    done

  let acknowledge device =
    assert (Domain.is_main_domain ());
    while !(device.keep_spinning) && Option.is_some !(device.next_task) do
      device.wait_for_ackn.await ()
    done

  let finalize { device; ctx } =
    await device;
    Backend.finalize ctx

  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let make_work device task =
    let%diagn_rt_sexp work () =
      acknowledge device;
      if not !(device.keep_spinning) then invalid_arg "Multicore_backend: device not available";
      device.next_task := Some task;
      device.wait_for_work.release ()
    in
    Tnode.Work work

  let link { ctx; device } code =
    let task = Backend.link ctx code in
    let%diagn_sexp schedule () =
      [%log_result "Scheduling", task.name];
      make_work device @@ task.schedule ()
    in
    { task with context = { ctx = task.context; device }; schedule }

  let link_batch { ctx; device } code_batch =
    Array.map (Backend.link_batch ctx code_batch)
      ~f:
        (Option.map ~f:(fun task ->
             let%diagn_sexp schedule () =
               [%log_result "Scheduling from batch", task.name];
               make_work device @@ task.schedule ()
             in
             { task with context = { ctx = task.context; device }; schedule }))

  let from_host { device; ctx } tn =
    match Backend.from_host_callback ctx tn with
    | None -> false
    | Some task ->
        await device;
        Tnode.run (module Debug_runtime) task;
        true

  let to_host { device; ctx } tn =
    match Backend.to_host_callback ctx tn with
    | None -> false
    | Some task ->
        await device;
        Tnode.run (module Debug_runtime) task;
        true

  let from_host_callback { device; ctx } tn =
    Option.map (Backend.from_host_callback ctx tn) ~f:(fun task -> make_work device task)

  let to_host_callback { device; ctx } tn =
    Option.map (Backend.to_host_callback ctx tn) ~f:(fun task -> make_work device task)

  let from_host_async context tn =
    match from_host_callback context tn with
    | None -> false
    | Some task ->
        Tnode.run (module Debug_runtime) task;
        true

  let to_host_async context tn =
    match to_host_callback context tn with
    | None -> false
    | Some task ->
        Tnode.run (module Debug_runtime) task;
        true

  let merge_callback tn ~accum ~dst ~src =
    Option.map (Backend.merge_callback tn ~accum ~dst:dst.ctx ~src:src.ctx) ~f:(fun task ->
        make_work dst.device task)

  let merge tn ~accum ~dst ~src =
    match merge_callback tn ~accum ~dst ~src with
    | None -> false
    | Some task ->
        await src.device;
        Tnode.run (module Debug_runtime) task;
        true

  let merge_unsafe tn ~accum ~dst ~src =
    match merge_callback tn ~accum ~dst ~src with
    | None -> false
    | Some task ->
        acknowledge src.device;
        Tnode.run (module Debug_runtime) task;
        true

  let compile_merges ?name_prefixes ~occupancy tns ~accum ~dsts ~srcs =
    let ctxs = Array.map ~f:(fun ctx -> ctx.ctx) in
    let occupancy tn ~dst_n ~dst:_ ~src_n ~src:_ =
      let dst = dsts.(dst_n) in
      let src = srcs.(src_n) in
      occupancy tn ~dst_n ~dst ~src_n ~src
    in
    Backend.compile_merges ?name_prefixes ~occupancy tns ~accum ~dsts:(ctxs dsts) ~srcs:(ctxs srcs)

  let num_devices () = Domain.recommended_domain_count () - 1
  let global_run_no = ref 0

  let%debug_sexp spinup_device ~(ordinal : int) : device =
    Int.incr global_run_no;
    let next_task = ref None in
    let is_idle = ref true in
    let keep_spinning = ref true in
    let wait_for_idle = Utils.waiter () in
    let wait_for_ackn = Utils.waiter () in
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
          let task = Option.value_exn !next_task in
          next_task := None;
          wait_for_ackn.release ();
          is_idle := false;
          Tnode.run debug_runtime task;
          is_idle := true;
          wait_for_idle.release ())
      done
    in
    {
      next_task;
      is_idle;
      keep_spinning;
      wait_for_idle;
      wait_for_ackn;
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
      device.wait_for_idle.finalize ();
      device.wait_for_ackn.finalize ();
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

let lower_batch_assignments ?names ?occupancy bindings asgns_l =
  let names =
    Option.value_or_thunk names ~default:(fun () ->
        Array.map asgns_l ~f:(fun asgns -> Assignments.get_name_exn asgns))
  in
  let prefix_name = String.(strip ~drop:(equal_char '_') @@ common_prefix @@ Array.to_list names) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(prefix_name ^ ".cd") in
  let bound = Indexing.bound_symbols bindings in
  let occupancy = Option.value occupancy ~default:(fun ~name:_ ~src_n:_ -> true) in
  Array.unzip
  @@ Array.mapi names ~f:(fun src_n name ->
         let asgns = asgns_l.(src_n) in
         if occupancy ~name ~src_n then
           ( Some name,
             Some (Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns) )
         else (None, None))

module type Simple_backend = sig
  type context [@@deriving sexp_of]
  type procedure [@@deriving sexp_of]
  type ctx_arrays [@@deriving sexp_of]

  val ctx_arrays : context -> ctx_arrays

  val compile :
    name:string ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized ->
    procedure

  val compile_batch :
    names:string option array ->
    opt_ctx_arrays:ctx_arrays option ->
    Indexing.unit_bindings ->
    Low_level.optimized option array ->
    procedure option array

  val link_compiled :
    context -> procedure -> context * Indexing.lowered_bindings * (unit -> Tnode.work) * string

  val merge :
    ?name_prefix:string ->
    Tnode.t ->
    accum:Ops.binop ->
    src:context ->
    Indexing.unit_bindings ->
    procedure option

  val merge_batch :
    ?name_prefixes:string array ->
    occupancy:(Tnode.t -> src_n:int -> src:context -> Utils.requirement) ->
    Tnode.t list ->
    accum:Ops.binop ->
    srcs:context array ->
    Indexing.unit_bindings ->
    (Tnode.t, procedure option array) Base.Hashtbl.t

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool
  val init : label:string -> context
  val finalize : context -> unit
  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  val from_host_callback : context -> Tnode.t -> Tnode.work option
  val to_host_callback : context -> Tnode.t -> Tnode.work option
end

module Simple_no_device_backend (Backend : Simple_backend) : No_device_backend = struct
  type code =
    | Postponed of { lowered : Low_level.optimized; bindings : Indexing.unit_bindings; name : string }
    | Compiled of Backend.procedure
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of Backend.procedure option array
  [@@deriving sexp_of]

  include Backend

  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?(shared = false) ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ?names ?occupancy bindings asgns_l : code_batch =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    if shared then Compiled (compile_batch ~names ~opt_ctx_arrays:None bindings lowereds)
    else Postponed { lowereds; bindings; names }

  let link (old_context : context) (code : code) =
    let context, bindings, schedule, name =
      match code with
      | Postponed { lowered; bindings; name } ->
          let proc = Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowered in
          link_compiled old_context proc
      | Compiled code -> link_compiled old_context code
    in
    { context; schedule; bindings; name }

  let link_batch (old_context : context) (code_batch : code_batch) : routine option array =
    let procs =
      match code_batch with
      | Postponed { lowereds; bindings; names } ->
          Backend.compile_batch ~names ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowereds
      | Compiled procs -> procs
    in
    Array.map procs
      ~f:
        (Option.map ~f:(fun proc ->
             let context, bindings, schedule, name = link_compiled old_context proc in
             { context; schedule; bindings; name }))

  let merge ?name_prefix tn ~accum ~(src : context) : code option =
    let bindings = Indexing.Empty in
    merge ?name_prefix tn ~accum ~src bindings |> Option.map ~f:(fun routine : code -> Compiled routine)

  let merge_batch ?name_prefixes ~occupancy tns ~accum ~(srcs : context array) =
    let bindings = Indexing.Empty in
    merge_batch ?name_prefixes ~occupancy tns ~accum ~srcs bindings
    |> Hashtbl.map ~f:(fun procs ->
           Array.map procs ~f:(Option.map ~f:(fun routine : code -> Compiled routine)))
end

module Gccjit_device : No_device_backend = Simple_no_device_backend ((
  Gccjit_backend : Simple_backend with type context = Gccjit_backend.context))

module Gccjit_backend = Multicore_backend (Gccjit_device)

module Cuda_backend : Backend with type context = Cuda_backend.context = struct
  type code = Cuda_backend.code [@@deriving sexp_of]
  type code_batch = Cuda_backend.code_batch [@@deriving sexp_of]
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

  let compile_batch ?shared:_ ?names ?occupancy bindings asgns_l =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    compile_batch ~names bindings lowereds

  let link context code =
    let context, bindings, schedule = link context code in
    { context; schedule; bindings; name }

  let link_batch context code_batch : routine option array =
    let context, bindings, schedules = link_batch context code_batch in
    Array.map schedules ~f:(Option.map ~f:(fun schedule -> { context; schedule; bindings; name }))

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
