open Base
open Backend_utils.Types
module Debug_runtime = Utils.Debug_runtime

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module type No_device_backend = sig
  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type buffer_ptr [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  val name : string

  val initialize : config -> unit
  (** Initializes a backend before first use or (on some backends) after {!unsafe_cleanup}. Does
      nothing if the backend is already initialized. *)

  val is_initialized : unit -> bool
  (** Returns false if there was no previous {!initialize} call, or, on some backends, the most
      recent call was followed by {!unsafe_cleanup}. If it returns false, one must call
      {!initialize} before using the backend. *)

  val init : label:string -> context
  (** [label] is usually the backend name concatenated with the device number. *)

  val finalize : context -> unit
  (** Finalizes (just) the context. *)

  val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> unit -> buffer_ptr
  val expected_merge_node : code -> Tnode.t option
  val expected_merge_nodes : code_batch -> Tnode.t option array

  val compile : ?shared:bool -> ?name:string -> Indexing.unit_bindings -> Assignments.t -> code
  (** If [~shared:true] (default [false]), the backend should prefer to do more compile work in a
      device-agnostic way. If [~shared:false], the backend can opt to postpone compiling altogether
      until [link] is called, to benefit from more optimizations. *)

  val compile_batch :
    ?shared:bool ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.t array ->
    code_batch
  (** Unlike the [~shared] parameter, [compile_batch] vs. [compile] is mostly about improving the
      compile time and debugging convenience by generating fewer files -- ideally does not affect
      execution, but there can be backend-specific differences. Only array entries for which
      [occupancy] returns true are included. *)

  val link :
    ?from_prior_context:Tnode.t list ->
    merge_buffer:(buffer_ptr * Tnode.t) option ref ->
    context ->
    code ->
    routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context.

      The [from_prior_context] nodes must not be added to the resulting context -- if needed in
      context, they must be part of the given context. *)

  val link_batch :
    ?from_prior_context:Tnode.t list ->
    merge_buffer:(buffer_ptr * Tnode.t) option ref ->
    context ->
    code_batch ->
    context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines (in particular, the routines' contexts are not
      independent).

      The [from_prior_context] nodes must not be added to the resulting context -- if needed in
      context, they must be part of the given context. *)

  val unsafe_cleanup : unit -> unit
  (** Cleans up all work on a backend, releases resources. All previously retrieved values
      (contexts, virtual and physical devices) become invalid. The backend needs to be initialized
      again to be used again. *)

  val to_buffer : Tnode.t -> dst:buffer_ptr -> src:context -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
  val get_buffer : Tnode.t -> context -> buffer_ptr option
end

module type Backend = sig
  include No_device_backend

  val link : ?from_prior_context:Tnode.t list -> context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch :
    ?from_prior_context:Tnode.t list -> context -> code_batch -> context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  val from_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, schedules a copy from host to context and returns
      true, otherwise returns false. NOTE: when run for a device, it's the caller's responsibility
      to synchronize the device before the host's data is overwritten. *)

  val to_host : context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, schedules a copy from context to host and returns
      true, otherwise returns false. NOTE: when run for a device, it's the caller's responsibility
      to synchronize the device before the host's data is read. *)

  val device_to_device :
    Tnode.t -> into_merge_buffer:merge_buffer_use -> dst:context -> src:context -> bool
  (** If the node is absent from the [src] context and either it is present in the [dst] context or
      [~into_merge_buffer] is different from [No]: raises an error.

      If [~into_merge_buffer:No]: If the node is present in the [dst] context, schedules a copy of
      the tensor node from the device of [src] to the device of [dst] and returns true, otherwise
      returns false.

      If [~into_merge_buffer] is different from [No]: schedules the following task and returns true.

      The merge-buffer task sets on [dst] the merge buffer source to the given node. If
      [~into_merge_buffer:Streaming], remembers the buffer pointer of the source node to use for
      streaming, without blocking. If [~into_merge_buffer:Copy], copies from [src] to the merge
      buffer of [dst]'s device.

      If the [dst] context resulted from a compilation with [Streaming] or [Copy] specific merge
      buffer code, the [device_to_device] call should fail immediately if there's a mismatch with
      [~into_merge_buffer].

      NOTE: it's the caller's responsibility to synchronize the [src] device, if needed, {i before}
      calling [device_to_device], and if [~into_merge_buffer:Streaming], the [dst] device
      {i afterward}, before any computations on the [src] device overwrite the node. *)

  type physical_device
  type device

  val init : device -> context
  val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> device -> buffer_ptr

  val await : device -> unit
  (** Blocks till the device becomes idle, i.e. synchronizes the device. *)

  val is_idle : device -> bool
  (** Whether the device is currently waiting for work. *)

  val sexp_of_device : device -> Sexp.t
  val get_device : ordinal:int -> physical_device
  val num_physical_devices : unit -> int

  val suggested_num_virtual_devices : physical_device -> int
  (** The optimal number of virtual devices for the given physical device to follow the
      {!Backend_types.config} strategy passed to {!No_device_backend.initialize}. *)

  val new_virtual_device : physical_device -> device
  val get_ctx_device : context -> device
  val get_physical_device : device -> physical_device
  val to_ordinal : physical_device -> int
  val to_subordinal : device -> int
  val get_name : device -> string
end

module Multicore_backend (Backend : No_device_backend) : Backend = struct
  module Domain = Domain [@warning "-3"]

  type task_list = Tnode.task Utils.mutable_list [@@deriving sexp_of]

  module Mut = Stdlib.Mutex
  module Queue = Saturn_lockfree.Single_prod_single_cons_queue

  type task_queue = Tnode.task Queue.t

  let sexp_of_task_queue q =
    Sexp.(List [ Atom "task_queue_of_size"; Atom (Int.to_string @@ Queue.size q) ])

  type device_state = {
    mutable keep_spinning : bool;
    mutable device_error : exn option;
    queue : task_queue;
    mut : (Mut.t[@sexp.opaque]);
    host_wait_for_idle : (Stdlib.Condition.t[@sexp.opaque]);
    dev_wait_for_work : (Stdlib.Condition.t[@sexp.opaque]);
    mutable is_ready : bool;
  }
  [@@deriving sexp_of]

  type buffer_ptr = Backend.buffer_ptr [@@deriving sexp_of]

  type device = {
    state : device_state;
    merge_buffer : (buffer_ptr * Tnode.t) option ref;
    mutable allocated_buffer : (buffer_ptr * int) option;
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  let alloc_buffer ?old_buffer ~size_in_bytes _device =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  type physical_device = device [@@deriving sexp_of]
  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]

  let expected_merge_node (code : code) = Backend.expected_merge_node code
  let expected_merge_nodes (codes : code_batch) = Backend.expected_merge_nodes codes
  let is_dev_queue_empty state = Queue.size state.queue = 0
  let is_idle device = is_dev_queue_empty device.state && device.state.is_ready
  let name = "multicore " ^ Backend.name

  let%track3_l_sexp await device =
    assert (Domain.is_main_domain ());
    let d = device.state in
    if (not @@ is_idle device) && d.keep_spinning then (
      Mut.lock d.mut;
      if (not @@ is_idle device) && d.keep_spinning then (
        while (not @@ is_idle device) && d.keep_spinning do
          Stdlib.Condition.wait d.host_wait_for_idle d.mut
        done);
      Mut.unlock d.mut;
      Option.iter d.device_error ~f:(fun e ->
          Exn.reraise e @@ name ^ " device " ^ Int.to_string device.ordinal))

  let%track3_l_sexp schedule_task device task =
    assert (Domain.is_main_domain ());
    [%log_result "schedule_task", Tnode.describe task, "device", (device.ordinal : int)];
    let d = device.state in
    Option.iter d.device_error ~f:(fun e ->
        Exn.reraise e @@ name ^ " device " ^ Int.to_string device.ordinal);
    if not d.keep_spinning then invalid_arg "Multicore_backend: device not available";
    if not @@ Queue.try_push d.queue task then (
      await device;
      Queue.push_exn d.queue task);
    if d.is_ready then (
      Mut.lock d.mut;
      Stdlib.Condition.broadcast d.dev_wait_for_work;
      Mut.unlock d.mut)

  let global_run_no = ref 0

  let%track3_l_sexp spinup_device ~(ordinal : int) : device =
    Int.incr global_run_no;
    let state =
      {
        keep_spinning = true;
        device_error = None;
        queue = Queue.create ~size_exponent:12;
        mut = Mut.create ();
        is_ready = false;
        host_wait_for_idle = Stdlib.Condition.create ();
        dev_wait_for_work = Stdlib.Condition.create ();
      }
    in
    let%track3_l_sexp worker (() : unit) : unit =
      assert (not @@ Domain.is_main_domain ());
      try
        while state.keep_spinning do
          match Queue.pop_opt state.queue with
          | None ->
              Mut.lock state.mut;
              if is_dev_queue_empty state && state.keep_spinning then (
                state.is_ready <- true;
                while is_dev_queue_empty state && state.keep_spinning do
                  Stdlib.Condition.broadcast state.host_wait_for_idle;
                  Stdlib.Condition.wait state.dev_wait_for_work state.mut
                done;
                state.is_ready <- false);
              Mut.unlock state.mut
          | Some task -> Tnode.run task
        done
      with e ->
        state.device_error <- Some e;
        state.keep_spinning <- false;
        [%log1 "Device", (ordinal : int), "exception", Exn.to_string e];
        (* TODO: we risk raising this error multiple times because await and schedule_task raise
           device_error. But this is fine if we assume all exceptions are fatal. *)
        raise e
    in
    {
      state;
      ordinal;
      domain = Domain.spawn worker;
      merge_buffer = ref None;
      allocated_buffer = None;
    }

  let%track3_l_sexp make_work device (Tnode.Task { description; _ } as task) =
    [%log_result "make_work", description, "device", (device.ordinal : int)];
    let work () = schedule_task device task in
    Tnode.Task
      {
        context_lifetime = task;
        description = "schedules {" ^ description ^ "} on device " ^ Int.to_string device.ordinal;
        work;
      }

  type context = { device : device; ctx : Backend.context; expected_merge_node : Tnode.t option }
  [@@deriving sexp_of]

  type nonrec routine = context routine [@@deriving sexp_of]

  let init device =
    {
      device;
      ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal);
      expected_merge_node = None;
    }

  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized

  let finalize { device; ctx; expected_merge_node = _ } =
    await device;
    Backend.finalize ctx

  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link ?from_prior_context { ctx; device; expected_merge_node = _ } code =
    let task = Backend.link ?from_prior_context ~merge_buffer:device.merge_buffer ctx code in
    {
      task with
      context =
        { ctx = task.context; device; expected_merge_node = Backend.expected_merge_node code };
      schedule = make_work device task.schedule;
    }

  let link_batch ?from_prior_context { ctx; device; expected_merge_node } code_batch =
    let ctx, routines =
      Backend.link_batch ?from_prior_context ~merge_buffer:device.merge_buffer ctx code_batch
    in
    let merge_nodes = Backend.expected_merge_nodes code_batch in
    ( { ctx; device; expected_merge_node },
      Array.mapi routines ~f:(fun i ->
          Option.map ~f:(fun task ->
              {
                task with
                context = { ctx = task.context; device; expected_merge_node = merge_nodes.(i) };
                schedule = make_work device task.schedule;
              })) )

  let from_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let%diagn_l_sexp work () =
                 Backend.host_to_buffer h_arr ~dst:c_arr;
                 [%diagn_sexp
                   [%log_block
                     "from_host " ^ Tnode.debug_name tn;
                     [%log "copied", Tnode.debug_name tn, "from host"];
                     [%log2_printbox
                       let indices =
                         Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                       in
                       Ndarray.render_array ~indices h_arr]]]
               in
               schedule_task context.device
                 (Tnode.Task
                    {
                      context_lifetime = context;
                      description =
                        "from_host " ^ Tnode.debug_name tn ^ " dst "
                        ^ Int.to_string context.device.ordinal;
                      work;
                    });
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_block
                   "nothing to copy from host";
                   [%log "for", Tnode.debug_name tn]]];
               false)

  let to_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let%diagn_l_sexp work () =
                 Backend.buffer_to_host h_arr ~src:c_arr;
                 [%diagn_sexp
                   [%log_block
                     "to_host " ^ Tnode.debug_name tn;
                     [%log "copied to host"];
                     [%log2_printbox
                       let indices =
                         Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                       in
                       Ndarray.render_array ~indices h_arr]]]
               in
               schedule_task context.device
                 (Tnode.Task
                    {
                      context_lifetime = context;
                      description =
                        "from_host " ^ Tnode.debug_name tn ^ " dst "
                        ^ Int.to_string context.device.ordinal;
                      work;
                    });
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_block
                   "nothing to copy to host";
                   [%log "for", Tnode.debug_name tn]]];
               false)

  let device_to_device tn ~into_merge_buffer ~dst ~src =
    let dev = dst.device in
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.debug_name tn);
    let schedule dst =
      let work =
        (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
        match into_merge_buffer with
        | No -> fun () -> Backend.to_buffer tn ~dst ~src:src.ctx
        | Streaming ->
            fun () ->
              dev.merge_buffer :=
                Option.map ~f:(fun ptr -> (ptr, tn)) @@ Backend.get_buffer tn src.ctx
        | Copy ->
            fun () ->
              let size_in_bytes = Tnode.size_in_bytes tn in
              let allocated_capacity =
                Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
              in
              if allocated_capacity < size_in_bytes then
                dev.allocated_buffer <-
                  Some
                    ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                      size_in_bytes );
              let merge_ptr = fst @@ Option.value_exn dev.allocated_buffer in
              dev.merge_buffer := Some (merge_ptr, tn);
              Backend.to_buffer tn ~dst:merge_ptr ~src:src.ctx
      in
      let description =
        "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ Int.to_string dev.ordinal ^ " src "
        ^ Int.to_string src.device.ordinal
      in
      schedule_task dev (Tnode.Task { context_lifetime = (src, dst); description; work })
    in
    match (Backend.get_buffer tn dst.ctx, Backend.get_buffer tn src.ctx) with
    | Some dst, Some _ ->
        schedule dst;
        true
    | _ -> false

  let num_physical_devices () = Domain.recommended_domain_count () - 1
  let suggested_num_virtual_devices _device = 1
  let devices = Array.create ~len:(num_physical_devices ()) None

  let%track2_sexp unsafe_cleanup () =
    assert (Domain.is_main_domain ());
    let wait_for_finish device =
      await device;
      device.state.keep_spinning <- false;
      Stdlib.Condition.broadcast device.state.dev_wait_for_work
    in
    Array.iter devices ~f:(Option.iter ~f:wait_for_finish);
    let cleanup ordinal device =
      Domain.join device.domain;
      devices.(ordinal) <- None
    in
    Array.iteri devices ~f:(fun ordinal -> Option.iter ~f:(cleanup ordinal));
    Backend.unsafe_cleanup ()

  let get_device ~ordinal =
    Option.value_or_thunk devices.(ordinal) ~default:(fun () ->
        let dev = spinup_device ~ordinal in
        devices.(ordinal) <- Some dev;
        dev)

  let new_virtual_device device = device
  let get_physical_device device = device
  let get_ctx_device { device; _ } = device
  let get_name device = Int.to_string device.ordinal
  let to_ordinal { ordinal; _ } = ordinal
  let to_subordinal _ = 0
  let to_buffer tn ~dst ~src = Backend.to_buffer tn ~dst ~src:src.ctx
  let host_to_buffer = Backend.host_to_buffer
  let buffer_to_host = Backend.buffer_to_host
  let get_buffer tn context = Backend.get_buffer tn context.ctx
end

module Pipes_multicore_backend (Backend : No_device_backend) : Backend = struct
  module Domain = Domain [@warning "-3"]

  type task_list = Tnode.task Utils.mutable_list [@@deriving sexp_of]

  type device_state = {
    mutable keep_spinning : bool;
    mutable device_error : exn option;
    mutable host_pos : task_list;
    mutable dev_pos : task_list;
    mutable dev_previous_pos : task_list;
    dev_wait : (Utils.waiter[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type buffer_ptr = Backend.buffer_ptr [@@deriving sexp_of]

  type device = {
    state : device_state;
    host_wait_for_idle : (Utils.waiter[@sexp.opaque]);
    merge_buffer : (buffer_ptr * Tnode.t) option ref;
    mutable allocated_buffer : (buffer_ptr * int) option;
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  let alloc_buffer ?old_buffer ~size_in_bytes _device =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  type physical_device = device [@@deriving sexp_of]
  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]

  let expected_merge_node (code : code) = Backend.expected_merge_node code
  let expected_merge_nodes (codes : code_batch) = Backend.expected_merge_nodes codes
  let is_dev_queue_empty state = Utils.(is_empty @@ tl_exn state.dev_previous_pos)
  let is_idle device = is_dev_queue_empty device.state && device.state.dev_wait.is_waiting ()
  let name = "multicore " ^ Backend.name

  let await device =
    assert (Domain.is_main_domain ());
    let d = device.state in
    let keep_waiting () =
      if d.keep_spinning && not (is_dev_queue_empty d) then (
        ignore (d.dev_wait.release_if_waiting () : bool);
        true)
      else d.keep_spinning && not (d.dev_wait.is_waiting ())
    in
    while not (is_dev_queue_empty d) do
      ignore (device.host_wait_for_idle.await ~keep_waiting () : bool)
    done;
    Option.iter d.device_error ~f:(fun e ->
        Exn.reraise e @@ name ^ " device " ^ Int.to_string device.ordinal)

  let%track2_sexp schedule_task device task =
    assert (Domain.is_main_domain ());
    let d = device.state in
    Option.iter d.device_error ~f:(fun e ->
        Exn.reraise e @@ name ^ " device " ^ Int.to_string device.ordinal);
    if not d.keep_spinning then invalid_arg "Multicore_backend: device not available";
    d.host_pos <- Utils.insert ~next:task d.host_pos;
    ignore (d.dev_wait.release_if_waiting () : bool)

  let global_run_no = ref 0

  let spinup_device ~(ordinal : int) : device =
    Int.incr global_run_no;
    let init_pos =
      Utils.Cons
        {
          hd =
            Tnode.Task
              { context_lifetime = (); description = "root of task queue"; work = (fun () -> ()) };
          tl = Empty;
        }
    in
    let state =
      {
        keep_spinning = true;
        device_error = None;
        host_pos = init_pos;
        dev_pos = Empty;
        dev_previous_pos = init_pos;
        dev_wait = Utils.waiter ~name:"dev" ();
      }
    in
    let host_wait_for_idle = Utils.waiter ~name:"host" () in
    let keep_waiting () =
      state.keep_spinning && is_dev_queue_empty state && not (host_wait_for_idle.is_waiting ())
    in
    let wait_by_dev = state.dev_wait.await ~keep_waiting in
    let%diagn_l_sexp worker (() : unit) : unit =
      try
        while state.keep_spinning do
          match state.dev_pos with
          | Empty ->
              let _host_released : bool = host_wait_for_idle.release_if_waiting () in
              let _could_wait : bool = wait_by_dev () in
              (* not _host_released && not _could_wait: we busy-loop until host processes its
                 release. *)
              state.dev_pos <- Utils.tl_exn state.dev_previous_pos
          | Cons { hd; tl } ->
              Tnode.run hd;
              state.dev_previous_pos <- state.dev_pos;
              state.dev_pos <- tl
        done
      with e ->
        state.device_error <- Some e;
        state.keep_spinning <- false;
        [%log "Device", (ordinal : int), "exception", Exn.to_string e];
        ignore (host_wait_for_idle.release_if_waiting () : bool);
        (* TODO: we risk raising this error multiple times because await and schedule_task raise
           device_error. But this is fine if we assume all exceptions are fatal. *)
        raise e
    in
    {
      state;
      host_wait_for_idle;
      ordinal;
      domain = Domain.spawn worker;
      merge_buffer = ref None;
      allocated_buffer = None;
    }

  let%diagn_sexp make_work device (Tnode.Task { context_lifetime; description; _ } as task) =
    let%diagn_l_sexp work () = schedule_task device task in
    Tnode.Task
      {
        context_lifetime;
        description = "schedules {" ^ description ^ "} on device " ^ Int.to_string device.ordinal;
        work;
      }

  type context = { device : device; ctx : Backend.context; expected_merge_node : Tnode.t option }
  [@@deriving sexp_of]

  type nonrec routine = context routine [@@deriving sexp_of]

  let init device =
    {
      device;
      ctx = Backend.init ~label:(name ^ " " ^ Int.to_string device.ordinal);
      expected_merge_node = None;
    }

  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized

  let finalize { device; ctx; expected_merge_node = _ } =
    await device;
    Backend.finalize ctx

  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link ?from_prior_context { ctx; device; expected_merge_node = _ } code =
    let task = Backend.link ?from_prior_context ~merge_buffer:device.merge_buffer ctx code in
    {
      task with
      context =
        { ctx = task.context; device; expected_merge_node = Backend.expected_merge_node code };
      schedule = make_work device task.schedule;
    }

  let link_batch ?from_prior_context { ctx; device; expected_merge_node } code_batch =
    let ctx, routines =
      Backend.link_batch ?from_prior_context ~merge_buffer:device.merge_buffer ctx code_batch
    in
    let merge_nodes = Backend.expected_merge_nodes code_batch in
    ( { ctx; device; expected_merge_node },
      Array.mapi routines ~f:(fun i ->
          Option.map ~f:(fun task ->
              {
                task with
                context = { ctx = task.context; device; expected_merge_node = merge_nodes.(i) };
                schedule = make_work device task.schedule;
              })) )

  let from_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let%diagn_l_sexp work () =
                 Backend.host_to_buffer h_arr ~dst:c_arr;
                 [%log_block
                   "from_host " ^ Tnode.debug_name tn;
                   [%log "copied", Tnode.debug_name tn, "from host"];
                   [%log2_printbox
                     let indices =
                       Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                     in
                     Ndarray.render_array ~indices h_arr]]
               in
               schedule_task context.device
                 (Tnode.Task
                    {
                      context_lifetime = context;
                      description =
                        "from_host " ^ Tnode.debug_name tn ^ " dst "
                        ^ Int.to_string context.device.ordinal;
                      work;
                    });
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_block
                   "nothing to copy from host";
                   [%log "for", Tnode.debug_name tn]]];
               false)

  let to_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let%diagn_l_sexp work () =
                 Backend.buffer_to_host h_arr ~src:c_arr;
                 [%log_block
                   "to_host " ^ Tnode.debug_name tn;
                   [%log "copied to host"];
                   [%log2_printbox
                     let indices =
                       Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                     in
                     Ndarray.render_array ~indices h_arr]]
               in
               schedule_task context.device
                 (Tnode.Task
                    {
                      context_lifetime = context;
                      description =
                        "from_host " ^ Tnode.debug_name tn ^ " dst "
                        ^ Int.to_string context.device.ordinal;
                      work;
                    });
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_block
                   "nothing to copy to host";
                   [%log "for", Tnode.debug_name tn]]];
               false)

  let device_to_device tn ~into_merge_buffer ~dst ~src =
    let dev = dst.device in
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.debug_name tn);
    let schedule dst_ptr =
      let work =
        match into_merge_buffer with
        | No -> fun () -> Backend.to_buffer tn ~dst:dst_ptr ~src:src.ctx
        | Streaming ->
            fun () ->
              dev.merge_buffer :=
                Option.map ~f:(fun ptr -> (ptr, tn)) @@ Backend.get_buffer tn src.ctx
        | Copy ->
            fun () ->
              let size_in_bytes = Tnode.size_in_bytes tn in
              let allocated_capacity =
                Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
              in
              if allocated_capacity < size_in_bytes then
                dev.allocated_buffer <-
                  Some
                    ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                      size_in_bytes );
              let merge_ptr = fst @@ Option.value_exn dev.allocated_buffer in
              dev.merge_buffer := Some (merge_ptr, tn);
              Backend.to_buffer tn ~dst:merge_ptr ~src:src.ctx
      in
      schedule_task dev
        (Tnode.Task
           {
             context_lifetime = (dst, src);
             description =
               "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ Int.to_string dev.ordinal
               ^ " src " ^ Int.to_string src.device.ordinal;
             work;
           })
    in
    match (Backend.get_buffer tn dst.ctx, Backend.get_buffer tn src.ctx) with
    | Some dst, Some _ ->
        schedule dst;
        true
    | _ -> false

  let num_physical_devices () = Domain.recommended_domain_count () - 1
  let suggested_num_virtual_devices _device = 1
  let devices = Array.create ~len:(num_physical_devices ()) None

  let%track2_sexp unsafe_cleanup () =
    assert (Domain.is_main_domain ());
    let wait_for_finish device =
      await device;
      device.state.keep_spinning <- false;
      ignore (device.state.dev_wait.release_if_waiting () : bool)
    in
    Array.iter devices ~f:(Option.iter ~f:wait_for_finish);
    let cleanup ordinal device =
      Domain.join device.domain;
      device.host_wait_for_idle.finalize ();
      device.state.dev_wait.finalize ();
      devices.(ordinal) <- None
    in
    Array.iteri devices ~f:(fun ordinal -> Option.iter ~f:(cleanup ordinal));
    Backend.unsafe_cleanup ()

  let get_device ~ordinal =
    Option.value_or_thunk devices.(ordinal) ~default:(fun () ->
        let dev = spinup_device ~ordinal in
        devices.(ordinal) <- Some dev;
        dev)

  let new_virtual_device device = device
  let get_physical_device device = device
  let get_ctx_device { device; _ } = device
  let get_name device = Int.to_string device.ordinal
  let to_ordinal { ordinal; _ } = ordinal
  let to_subordinal _ = 0
  let to_buffer tn ~dst ~src = Backend.to_buffer tn ~dst ~src:src.ctx
  let host_to_buffer = Backend.host_to_buffer
  let buffer_to_host = Backend.buffer_to_host
  let get_buffer tn context = Backend.get_buffer tn context.ctx
end

(** For debugging, allow [Sync_backend(...).suggested_num_virtual_devices] calls to return >1
    numbers. *)
let sync_suggested_num_virtual_devices = ref 1

(** A minimalisitc wrapper creating backends where all calls run synchronously on the main thread.
    There is only one physical device, but an arbitrary number of virtual devices. *)
module Sync_backend (Backend : No_device_backend) (* : Backend *) = struct
  type buffer_ptr = Backend.buffer_ptr [@@deriving sexp_of]

  type device = {
    subordinal : int;
    merge_buffer : (buffer_ptr * Tnode.t) option ref;
    mutable allocated_buffer : (buffer_ptr * int) option;
  }
  [@@deriving sexp_of]

  let alloc_buffer ?old_buffer ~size_in_bytes _device =
    Backend.alloc_buffer ?old_buffer ~size_in_bytes ()

  type physical_device = CPU [@@deriving sexp_of]
  type code = Backend.code [@@deriving sexp_of]
  type code_batch = Backend.code_batch [@@deriving sexp_of]

  let expected_merge_node (code : code) = Backend.expected_merge_node code
  let expected_merge_nodes (codes : code_batch) = Backend.expected_merge_nodes codes
  let is_idle _device = true
  let name = "sync " ^ Backend.name
  let await _device = ()
  let global_run_no = ref 0

  type context = { device : device; ctx : Backend.context; expected_merge_node : Tnode.t option }
  [@@deriving sexp_of]

  type nonrec routine = context routine [@@deriving sexp_of]

  let init device = { device; ctx = Backend.init ~label:name; expected_merge_node = None }
  let initialize = Backend.initialize
  let is_initialized = Backend.is_initialized
  let finalize { device = _; ctx; expected_merge_node = _ } = Backend.finalize ctx
  let compile = Backend.compile
  let compile_batch = Backend.compile_batch

  let link ?from_prior_context { ctx; device; expected_merge_node = _ } code =
    let task = Backend.link ?from_prior_context ~merge_buffer:device.merge_buffer ctx code in
    {
      task with
      context =
        { ctx = task.context; device; expected_merge_node = Backend.expected_merge_node code };
    }

  let link_batch ?from_prior_context { ctx; device; expected_merge_node } code_batch =
    let ctx, routines =
      Backend.link_batch ?from_prior_context ~merge_buffer:device.merge_buffer ctx code_batch
    in
    let merge_nodes = Backend.expected_merge_nodes code_batch in
    ( { ctx; device; expected_merge_node },
      Array.mapi routines ~f:(fun i ->
          Option.map ~f:(fun task ->
              {
                task with
                context = { ctx = task.context; device; expected_merge_node = merge_nodes.(i) };
              })) )

  let get_name device = Int.to_string device.subordinal

  let from_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               Backend.host_to_buffer h_arr ~dst:c_arr;
               [%diagn2_l_sexp
                 [%log_block
                   "from_host for " ^ Tnode.debug_name tn;
                   [%log "copied", Tnode.debug_name tn, "from host to", get_name context.device];
                   [%log3_printbox
                     let indices =
                       Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                     in
                     Ndarray.render_array ~indices h_arr]]];
               true
           | (lazy None) ->
               [%diagn2_l_sexp
                 [%log_block
                   "nothing to copy from host for " ^ Tnode.debug_name tn;
                   [%log "to", get_name context.device]]];
               false)

  let to_host (context : context) (tn : Tnode.t) =
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               Backend.buffer_to_host h_arr ~src:c_arr;
               [%diagn2_l_sexp
                 [%log_block
                   "to_host for " ^ Tnode.debug_name tn;
                   [%log "copied to host from", get_name context.device];
                   [%log3_printbox
                     let indices =
                       Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                     in
                     Ndarray.render_array ~indices h_arr]]];
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_block
                   "nothing to copy to host for " ^ Tnode.debug_name tn;
                   [%log "from", get_name context.device]]];
               false)

  let device_to_device tn ~into_merge_buffer ~dst ~src =
    let dev = dst.device in
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.debug_name tn);
    (* TODO: log the operation if [Utils.settings.with_log_level > 0]. *)
    match (Backend.get_buffer tn dst.ctx, Backend.get_buffer tn src.ctx) with
    | None, _ | _, None -> false
    | Some dst, Some _ ->
        (match into_merge_buffer with
        | No -> Backend.to_buffer tn ~dst ~src:src.ctx
        | Streaming ->
            dev.merge_buffer :=
              Option.map ~f:(fun ptr -> (ptr, tn)) @@ Backend.get_buffer tn src.ctx
        | Copy ->
            let size_in_bytes = Tnode.size_in_bytes tn in
            let allocated_capacity =
              Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
            in
            if allocated_capacity < size_in_bytes then
              dev.allocated_buffer <-
                Some
                  ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                    size_in_bytes );
            let merge_ptr = fst @@ Option.value_exn dev.allocated_buffer in
            dev.merge_buffer := Some (merge_ptr, tn);
            Backend.to_buffer tn ~dst:merge_ptr ~src:src.ctx);
        true

  let num_physical_devices () = 1
  let suggested_num_virtual_devices _device = !sync_suggested_num_virtual_devices
  let next_virtual_device_id = ref 0

  let unsafe_cleanup () =
    next_virtual_device_id := 0;
    Backend.unsafe_cleanup ()

  let get_device ~ordinal =
    if ordinal <> 0 then invalid_arg "Sync_backend backends only have physical device number 0";
    CPU

  let new_virtual_device CPU =
    let result =
      { subordinal = !next_virtual_device_id; merge_buffer = ref None; allocated_buffer = None }
    in
    result

  let get_physical_device _device = CPU
  let get_ctx_device { device; _ } = device
  let to_ordinal _ = 0
  let to_subordinal device = device.subordinal
  let to_buffer tn ~dst ~src = Backend.to_buffer tn ~dst ~src:src.ctx
  let host_to_buffer = Backend.host_to_buffer
  let buffer_to_host = Backend.buffer_to_host
  let get_buffer tn context = Backend.get_buffer tn context.ctx
end

let lower_assignments ?name bindings asgns =
  let name = Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns) in
  let unoptim_ll_source = Utils.get_debug_formatter ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.get_debug_formatter ~fname:(name ^ ".ll") in
  let cd_source = Utils.get_debug_formatter ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

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
             Some
               (Assignments.lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns)
           )
         else (None, None))

module type Simple_backend = sig
  type context [@@deriving sexp_of]
  type procedure [@@deriving sexp_of]
  type ctx_array [@@deriving sexp_of]
  type buffer_ptr [@@deriving sexp_of]
  type ctx_arrays = ctx_array Map.M(Tnode).t [@@deriving sexp_of]

  val buffer_ptr : ctx_array -> buffer_ptr
  val ctx_arrays : context -> ctx_arrays
  val alloc_buffer : ?old_buffer:buffer_ptr * int -> size_in_bytes:int -> unit -> buffer_ptr
  val expected_merge_node : procedure -> Tnode.t option

  val is_in_context : Low_level.traced_array -> bool
  (** If true, the node is required to be in the contexts linked with code that uses it.

      Should return false for nodes that are virtual, local, or which the backend prefers to access
      directly from the host. *)

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
    ctx_arrays option * procedure option array

  val link_compiled :
    merge_buffer:(buffer_ptr * Tnode.t) option ref ->
    context ->
    procedure ->
    context * Indexing.lowered_bindings * Tnode.task * string

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool
  val init : label:string -> context
  val finalize : context -> unit
  val unsafe_cleanup : unit -> unit
  val to_buffer : Tnode.t -> dst:buffer_ptr -> src:context -> unit
  val host_to_buffer : Ndarray.t -> dst:buffer_ptr -> unit
  val buffer_to_host : Ndarray.t -> src:buffer_ptr -> unit
end

let verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context traced_stores
    =
  let olds = ctx_arrays prior_context in
  List.iter from_prior_context ~f:(fun tn ->
      let node = Array.find_map traced_stores ~f:(fun store -> Hashtbl.find store tn) in
      if
        Option.value_map node ~default:false ~f:(fun node ->
            is_in_context node && not (Map.mem olds tn))
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

module Simple_no_device_backend (Backend : Simple_backend) : No_device_backend = struct
  include Backend

  type code =
    | Postponed of {
        lowered : Low_level.optimized;
        bindings : Indexing.unit_bindings;
        name : string;
      }
    | Compiled of Low_level.optimized * Backend.procedure
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of
        Low_level.optimized option array * (ctx_arrays option * Backend.procedure option array)
  [@@deriving sexp_of]

  let global_config = ref Physical_devices_only

  let initialize config =
    global_config := config;
    initialize ()

  type nonrec routine = context routine [@@deriving sexp_of]

  let expected_merge_node : code -> _ = function
    | Postponed { lowered = Low_level.{ merge_node; _ }; _ } -> merge_node
    | Compiled (_, proc) -> Backend.expected_merge_node proc

  let expected_merge_nodes : code_batch -> _ = function
    | Postponed { lowereds; _ } ->
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.merge_node)))
    | Compiled (_, (_, procs)) ->
        Array.map ~f:(function Some proc -> Backend.expected_merge_node proc | _ -> None) procs

  let get_traced_store : code -> _ = function
    | Postponed { lowered = Low_level.{ traced_store; _ }; _ }
    | Compiled (Low_level.{ traced_store; _ }, _) ->
        traced_store

  let get_traced_stores : code_batch -> _ = function
    | Postponed { lowereds; _ } ->
        Array.filter_map lowereds ~f:(fun lowered ->
            Option.map lowered ~f:(fun optim -> optim.traced_store))
    | Compiled (lowereds, _) ->
        Array.filter_map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store))

  let compile ?(shared = false) ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (lowered, Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ?names ?occupancy bindings asgns_l : code_batch =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    if shared then Compiled (lowereds, compile_batch ~names ~opt_ctx_arrays:None bindings lowereds)
    else Postponed { lowereds; bindings; names }

  let link ?(from_prior_context = []) ~merge_buffer (prior_context : context) (code : code) =
    Backend.(
      verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context
        [| get_traced_store code |]);
    let context, bindings, schedule, name =
      match code with
      | Postponed { lowered; bindings; name } ->
          let proc =
            Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays prior_context)) bindings lowered
          in
          link_compiled ~merge_buffer prior_context proc
      | Compiled (_, code) -> link_compiled ~merge_buffer prior_context code
    in
    { context; schedule; bindings; name }

  let link_batch ?(from_prior_context = []) ~merge_buffer (prior_context : context)
      (code_batch : code_batch) =
    Backend.(
      verify_prior_context ~ctx_arrays ~is_in_context ~prior_context ~from_prior_context
      @@ get_traced_stores code_batch);
    let _opt_ctx_arrays, procs =
      match code_batch with
      | Postponed { lowereds; bindings; names } ->
          Backend.compile_batch ~names
            ~opt_ctx_arrays:(Some (ctx_arrays prior_context))
            bindings lowereds
      | Compiled (_, procs) -> procs
    in
    Array.fold_map procs ~init:prior_context ~f:(fun context -> function
      | Some proc ->
          let context, bindings, schedule, name = link_compiled ~merge_buffer context proc in
          (context, Some { context; schedule; bindings; name })
      | None -> (context, None))

  let to_buffer tn ~dst ~src = Backend.to_buffer tn ~dst ~src
  let host_to_buffer = Backend.host_to_buffer
  let buffer_to_host = Backend.buffer_to_host

  let get_buffer tn context =
    Map.find (Backend.ctx_arrays context) tn |> Option.map ~f:Backend.buffer_ptr
end

module C_device : No_device_backend = Simple_no_device_backend ((
  Cc_backend : Simple_backend with type context = Cc_backend.context))

module Cc_backend = Multicore_backend (C_device)
module Sync_cc_backend = Sync_backend (C_device)
module Pipes_cc_backend = Pipes_multicore_backend (C_device)

module Gccjit_device : No_device_backend = Simple_no_device_backend ((
  Gcc_backend : Simple_backend with type context = Gcc_backend.context))

module Gccjit_backend = Multicore_backend (Gccjit_device)
module Sync_gccjit_backend = Sync_backend (Gccjit_device)
module Pipes_gccjit_backend = Pipes_multicore_backend (Gccjit_device)

module Cuda_backend : Backend = struct
  include Cuda_backend

  type nonrec code = {
    traced_store : Low_level.traced_store;
    code : code;
    expected_merge_node : Tnode.t option;
  }
  [@@deriving sexp_of]

  type nonrec code_batch = {
    traced_stores : Low_level.traced_store array;
    code_batch : code_batch;
    expected_merge_nodes : Tnode.t option array;
  }
  [@@deriving sexp_of]

  let expected_merge_node code = code.expected_merge_node
  let expected_merge_nodes code_batch = code_batch.expected_merge_nodes
  let name = "cuda"

  type nonrec context = { ctx : context; expected_merge_node : Tnode.t option } [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?shared:_ ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    {
      traced_store = lowered.traced_store;
      code = compile ~name bindings lowered;
      expected_merge_node = lowered.Low_level.merge_node;
    }

  let compile_batch ?shared:_ ?names ?occupancy bindings asgns_l =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    {
      traced_stores =
        Array.filter_map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store));
      code_batch = compile_batch ~names bindings lowereds;
      expected_merge_nodes =
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.Low_level.merge_node)));
    }

  let link ?(from_prior_context = []) context code =
    verify_prior_context ~ctx_arrays ~is_in_context ~prior_context:context.ctx ~from_prior_context
      [| code.traced_store |];
    let ctx, bindings, schedule = link context.ctx code.code in
    { context = { ctx; expected_merge_node = code.expected_merge_node }; schedule; bindings; name }

  let link_batch ?(from_prior_context = []) context code_batch =
    verify_prior_context ~ctx_arrays ~is_in_context ~prior_context:context.ctx ~from_prior_context
      code_batch.traced_stores;
    let ctx, bindings, schedules = link_batch context.ctx code_batch.code_batch in
    ( { ctx; expected_merge_node = context.expected_merge_node },
      Array.mapi schedules ~f:(fun i ->
          Option.map ~f:(fun schedule ->
              {
                context = { ctx; expected_merge_node = code_batch.expected_merge_nodes.(i) };
                schedule;
                bindings;
                name;
              })) )

  let init device = { ctx = init device; expected_merge_node = None }
  let get_ctx_device context = get_ctx_device context.ctx
  let finalize context = finalize context.ctx
  let to_buffer tn ~dst ~src = to_buffer tn ~dst ~src:src.ctx
  let get_buffer tn context = get_buffer tn context.ctx
  let from_host context tn = from_host context.ctx tn
  let to_host context tn = to_host context.ctx tn

  let device_to_device tn ~into_merge_buffer ~dst ~src =
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.debug_name tn);
    device_to_device tn ~into_merge_buffer ~dst:dst.ctx ~src:src.ctx
end

let reinitialize (module Backend : Backend) config =
  if not @@ Backend.is_initialized () then Backend.initialize config
  else (
    Core.Gc.full_major ();
    Backend.unsafe_cleanup ();
    Backend.initialize config)

(** Reinitializes and returns a backend corresponding to [backend_name], or if omitted, selected via
    the global [backend] setting. *)
let fresh_backend ?backend_name ?(config = Physical_devices_only) () =
  let backend =
    match
      Option.value_or_thunk backend_name ~default:(fun () ->
          Utils.get_global_arg ~arg_name:"backend" ~default:"pipes_cc")
      |> String.lowercase
    with
    | "cc" -> (module Cc_backend : Backend)
    | "gccjit" -> (module Gccjit_backend : Backend)
    | "sync_cc" -> (module Sync_cc_backend : Backend)
    | "sync_gccjit" -> (module Sync_gccjit_backend : Backend)
    | "pipes_cc" -> (module Pipes_cc_backend : Backend)
    | "pipes_gccjit" -> (module Pipes_gccjit_backend : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Backends.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend config;
  backend
