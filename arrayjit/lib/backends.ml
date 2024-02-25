open Base
module Debug_runtime = Utils.Debug_runtime

type 'context jitted = {
  context : 'context;
  run : (module Minidebug_runtime.Debug_runtime) -> unit -> unit;
  bindings : Indexing.jitted_bindings;
}
[@@deriving sexp_of]

module type No_device_backend = sig
  type context
  type nonrec jitted = context jitted

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool

  val init : label:string -> context
  (** [label] is usually the backend name concatenated with the device number. *)

  val finalize : context -> unit
  (** Finalizes (just) the context. *)

  val sexp_of_context : context -> Sexp.t
  val sexp_of_jitted : jitted -> Sexp.t
  val jit : context -> ?name:string -> unit Indexing.bindings -> Assignments.t -> jitted

  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  (** Cleans up all work on a backend.
      If [~unsafe_shutdown:true], releases resources, potentially making the backend unusable. *)

  val from_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from host to context and returns true. *)

  val to_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, copies from context to host and returns true. *)

  val merge : ?name_suffix:string -> Tnode.t -> dst:context -> accum:Ops.binop -> src:context -> jitted option
  (** Merges the array from the source context into the destination context: [dst =: dst accum src].
      If the array is hosted, its state on host is undefined after this operation. (A backend may choose
      to use the host array as a buffer, if that is beneficial.) [name_suffix] is appended to
      the jitted function's name. Returns [None] if the array is not in the context. *)
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
    next_task : (((module Minidebug_runtime.Debug_runtime) -> unit -> unit) option ref[@sexp.opaque]);
    keep_spinning : bool ref;
    wait_for_device : (Utils.waiter[@sexp.opaque]);
    wait_for_work : (Utils.waiter[@sexp.opaque]);
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type context = { device : device; ctx : Backend.context } [@@deriving sexp_of]
  type nonrec jitted = context jitted [@@deriving sexp_of]

  let name = "multicore " ^ Backend.name
  let init device = { device; ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal) }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized

  let await device =
    while !(device.keep_spinning) && Option.is_some !(device.next_task) do
      device.wait_for_device.await ()
    done

  let finalize { device; ctx } =
    await device;
    Backend.finalize ctx

  let jit { ctx; device } ?name bindings code : jitted =
    let result = Backend.jit ctx ?name bindings code in
    let%track_rt_sexp run () =
      assert (Domain.is_main_domain ());
      await device;
      if not !(device.keep_spinning) then invalid_arg "Multicore_backend.jit: device not available";
      device.next_task := Some result.run;
      device.wait_for_work.release ()
    in
    { context = { ctx = result.context; device }; run; bindings = result.bindings }

  let from_host { ctx; _ } = Backend.from_host ctx
  let to_host { ctx; _ } = Backend.to_host ctx

  let%track_sexp merge ?name_suffix la ~dst ~accum ~src =
    let name_suffix : string = Option.value name_suffix ~default:"" in
    let ord ctx = ctx.device.ordinal in
    let name_suffix : string =
      [%string "_on_dev_%{ord dst#Int}_from_dev_%{ord src#Int}_%{name_suffix}"]
    in
    Option.map (Backend.merge ~name_suffix la ~dst:dst.ctx ~accum ~src:src.ctx) ~f:(fun result ->
        let device = dst.device in
        let%track_rt_sexp run () =
          assert (Domain.is_main_domain ());
          await device;
          if not !(device.keep_spinning) then invalid_arg "Multicore_backend.merge: device not available";
          device.next_task := Some result.run;
          device.wait_for_work.release ()
        in
        { context = { ctx = result.context; device }; run; bindings = result.bindings })

  let num_devices () = Domain.recommended_domain_count () - 1

  let%debug_sexp spinup_device ~(ordinal : int) =
    let next_task = ref None in
    let keep_spinning = ref true in
    let wait_for_device = Utils.waiter () in
    let wait_for_work = Utils.waiter () in
    let runtime = Utils.get_debug ("dev-" ^ Int.to_string ordinal) in
    let worker () =
      let module Debug_runtime = (val runtime) in
      [%log "spinup-dev", (ordinal : int)];
      while !keep_spinning do
        while Option.is_none !next_task do
          wait_for_work.await ()
        done;
        Option.value_exn !next_task runtime ();
        next_task := None;
        wait_for_device.release ()
      done
    in
    { next_task; keep_spinning; wait_for_device; wait_for_work; ordinal; domain = Domain.spawn worker }

  let devices = Array.init (num_devices ()) ~f:(fun ordinal -> spinup_device ~ordinal)
  let get_all_devices () = devices

  let%track_sexp unsafe_cleanup ?(unsafe_shutdown = false) () =
    assert (Domain.is_main_domain ());
    let cleanup ordinal device =
      await device;
      device.keep_spinning := false;
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

module Gccjit_device : No_device_backend with type context = Gccjit_backend.context = struct
  type context = Gccjit_backend.context [@@deriving sexp_of]
  type nonrec jitted = context jitted [@@deriving sexp_of]

  open Gccjit_backend

  let name = "gccjit"
  let initialize = initialize
  let is_initialized = is_initialized
  let unsafe_cleanup ?unsafe_shutdown () = Gccjit_backend.unsafe_cleanup ?unsafe_shutdown ()
  let init = init
  let finalize = finalize
  let sexp_of_context = sexp_of_context

  let jit context ?name bindings code =
    let name = Option.value name ~default:(Assignments.get_name code) in
    let context, bindings, run =
      jit context ~name bindings @@ Assignments.compile_proc ~name (Indexing.bound_symbols bindings) code
    in
    { context; run; bindings }

  let from_host = from_host
  let to_host = to_host

  let merge ?name_suffix la ~dst ~accum ~(src : context) =
    let bindings = Indexing.Empty in
    merge ?name_suffix la ~dst ~accum ~src bindings
    |> Option.map ~f:(fun (context, bindings, run) -> { context; run; bindings })
end

module Gccjit_backend = Multicore_backend (Gccjit_device)

module Dummy_device : No_device_backend with type context = Dummy_backend.context = struct
  type context = Dummy_backend.context [@@deriving sexp_of]
  type nonrec jitted = context jitted [@@deriving sexp_of]

  open Dummy_backend

  let name = "dummy"
  let initialize = initialize
  let is_initialized = is_initialized
  let unsafe_cleanup ?unsafe_shutdown () = Dummy_backend.unsafe_cleanup ?unsafe_shutdown ()
  let init = init
  let finalize = finalize
  let sexp_of_context = sexp_of_context

  let jit context ?name bindings code =
    let name = Option.value name ~default:(Assignments.get_name code) in
    let context, bindings, run =
      jit context ~name bindings @@ Assignments.compile_proc ~name (Indexing.bound_symbols bindings) code
    in
    { context; run; bindings }

  let from_host = from_host
  let to_host = to_host

  let merge ?name_suffix la ~dst ~accum ~(src : context) =
    let bindings = Indexing.Empty in
    merge ?name_suffix la ~dst ~accum ~src bindings
    |> Option.map ~f:(fun (context, bindings, run) -> { context; run; bindings })
end

module Dummy_backend = Multicore_backend (Dummy_device)

module Cuda_backend : Backend with type context = Cuda_backend.context = struct
  type context = Cuda_backend.context [@@deriving sexp_of]
  type device = Cuda_backend.device
  type nonrec jitted = context jitted [@@deriving sexp_of]

  open Cuda_backend

  let name = "cuda"
  let initialize = initialize
  let is_initialized = is_initialized
  let unsafe_cleanup ?unsafe_shutdown () = Cuda_backend.unsafe_cleanup ?unsafe_shutdown ()
  let init = init
  let finalize = finalize
  let sexp_of_context = sexp_of_context
  let sexp_of_device = sexp_of_device

  let jit context ?name bindings code =
    let name = Option.value name ~default:(Assignments.get_name code) in
    let context, bindings, run =
      jit context ~name bindings @@ Assignments.compile_proc ~name (Indexing.bound_symbols bindings) code
    in
    { context; run; bindings }

  let from_host = from_host
  let to_host = to_host

  let merge ?name_suffix la ~dst ~accum ~src =
    let bindings = Indexing.Empty in
    merge ?name_suffix la ~dst ~accum ~src bindings
    |> Option.map ~f:(fun (context, run) -> { context; run; bindings = [] })

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
