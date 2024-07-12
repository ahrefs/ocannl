open Base
open Backend_utils.Types
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module type No_device_backend = sig
  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]
  type buffer_ptr [@@deriving sexp_of]
  type context [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  val name : string
  val initialize : config -> unit
  val is_initialized : unit -> bool

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

  val link : merge_buffer:buffer_ptr option ref -> context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch :
    merge_buffer:buffer_ptr option ref -> context -> code_batch -> context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines (in particular, the routines' contexts are not
      independent). *)

  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit
  (** Cleans up all work on a backend. If [~unsafe_shutdown:true], releases resources, potentially
      making the backend unusable. *)

  val to_buffer :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Tnode.t -> dst:buffer_ptr -> src:context -> unit

  val host_to_buffer :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> dst:buffer_ptr -> unit

  val buffer_to_host :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> src:buffer_ptr -> unit

  val get_buffer : Tnode.t -> context -> buffer_ptr option
end

module type Backend = sig
  include No_device_backend

  val link : context -> code -> routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context. *)

  val link_batch : context -> code_batch -> context * routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  val from_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, schedules a copy from host to context and returns
      true, otherwise returns false. NOTE: when run for a device, it's the caller's responsibility
      to synchronize the device before the host's data is overwritten. *)

  val to_host : ?rt:(module Minidebug_runtime.Debug_runtime) -> context -> Tnode.t -> bool
  (** If the array is both hosted and in-context, schedules a copy from context to host and returns
      true, otherwise returns false. NOTE: when run for a device, it's the caller's responsibility
      to synchronize the device before the host's data is read. *)

  val device_to_device :
    ?rt:(module Minidebug_runtime.Debug_runtime) ->
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst:context ->
    src:context ->
    bool
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

let forget_printbox (module Runtime : Minidebug_runtime.PrintBox_runtime) =
  (module Runtime : Minidebug_runtime.Debug_runtime)

module Multicore_backend (Backend : No_device_backend) : Backend = struct
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

  let alloc_buffer = Backend.alloc_buffer

  type device = {
    state : device_state;
    host_wait_for_idle : (Utils.waiter[@sexp.opaque]);
    merge_buffer_ptr : buffer_ptr option ref;
    (* mutable *) merge_node : Tnode.t option;
    mutable allocated_buffer : (buffer_ptr * int) option;
    ordinal : int;
    domain : (unit Domain.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

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

  let%track_sexp schedule_task device task =
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
        { hd = Tnode.{ description = "root of task queue"; work = (fun _rt () -> ()) }; tl = Empty }
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
    let run_no = !global_run_no in
    let debug_runtime =
      Utils.get_debug ("dev-multicore-" ^ Int.to_string ordinal ^ "-run-" ^ Int.to_string run_no)
    in
    let%diagn_rt_sexp worker (() : unit) : unit =
      try
        while state.keep_spinning do
          match state.dev_pos with
          | Empty ->
              let _host_released : bool = host_wait_for_idle.release_if_waiting () in
              let _could_wait : bool = wait_by_dev () in
              (* not _host_released && not _could_wait: we busy-loop until host processes its release. *)
              (* [%log "WORK WHILE LOOP: EMPTY AFTER WAIT -- dev pos:", (state.dev_pos : task_list)]; *)
              state.dev_pos <- Utils.tl_exn state.dev_previous_pos
          | Cons { hd; tl } ->
              Tnode.run _debug_runtime hd;
              (* [%log "WORK WHILE LOOP: AFTER WORK"]; *)
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
      domain = Domain.spawn (worker debug_runtime);
      merge_buffer_ptr = ref None;
      merge_node = None;
      allocated_buffer = None;
    }

  let%diagn_sexp make_work device task =
    let%diagn_rt_sexp work () = schedule_task device task in
    Tnode.
      {
        description =
          "schedules {" ^ task.description ^ "} on device " ^ Int.to_string device.ordinal;
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

  let link { ctx; device; expected_merge_node = _ } (code : code) =
    let task = Backend.link ~merge_buffer:device.merge_buffer_ptr ctx code in
    {
      task with
      context =
        { ctx = task.context; device; expected_merge_node = Backend.expected_merge_node code };
      schedule = make_work device task.schedule;
    }

  let link_batch { ctx; device; expected_merge_node } (code_batch : code_batch) =
    let ctx, routines = Backend.link_batch ~merge_buffer:device.merge_buffer_ptr ctx code_batch in
    let merge_nodes = Backend.expected_merge_nodes code_batch in
    ( { ctx; device; expected_merge_node },
      Array.mapi routines ~f:(fun i ->
          Option.map ~f:(fun task ->
              {
                task with
                context = { ctx = task.context; device; expected_merge_node = merge_nodes.(i) };
                schedule = make_work device task.schedule;
              })) )

  let from_host ?rt (context : context) (tn : Tnode.t) =
    if Option.is_some rt then
      raise
      @@ Utils.User_error "Multicore_backend.from_host: backend cannot be nested in another runtime";
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let work rt () =
                 Backend.host_to_buffer ~rt h_arr ~dst:c_arr;
                 if Utils.settings.with_debug_level > 0 then
                   let module Debug_runtime = (val rt) in
                   [%diagn_sexp
                     [%log_entry
                       "from_host " ^ Tnode.get_debug_name tn;
                       [%log "copied", Tnode.label tn, Tnode.name tn, "from host"];
                       if Utils.settings.with_debug_level > 1 then
                         [%log_printbox
                           let indices =
                             Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                           in
                           Ndarray.render_array ~indices h_arr]]]
               in
               schedule_task context.device
                 Tnode.
                   {
                     description =
                       "from_host " ^ Tnode.get_debug_name tn ^ " dst "
                       ^ Int.to_string context.device.ordinal;
                     work;
                   };
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_entry
                   "from_host empty " ^ Tnode.get_debug_name tn;
                   [%log "nothing to copy", Tnode.label tn, Tnode.name tn, "from host"]]];
               false)

  let to_host ?rt (context : context) (tn : Tnode.t) =
    if Option.is_some rt then
      raise
      @@ Utils.User_error "Multicore_backend.to_host: backend cannot be nested in another runtime";
    Option.value ~default:false
    @@ Option.map (Backend.get_buffer tn context.ctx) ~f:(fun c_arr ->
           match tn.Tnode.array with
           | (lazy (Some h_arr)) ->
               let work rt () =
                 Backend.buffer_to_host ~rt h_arr ~src:c_arr;
                 if Utils.settings.with_debug_level > 0 then
                   let module Debug_runtime = (val rt) in
                   [%diagn_sexp
                     [%log_entry
                       "to_host " ^ Tnode.get_debug_name tn;
                       [%log "copied", Tnode.label tn, Tnode.name tn, "to host"];
                       if Utils.settings.with_debug_level > 1 then
                         [%log_printbox
                           let indices =
                             Array.init (Array.length @@ Lazy.force tn.dims) ~f:(fun i -> i - 5)
                           in
                           Ndarray.render_array ~indices h_arr]]]
               in
               schedule_task context.device
                 Tnode.
                   {
                     description =
                       "from_host " ^ Tnode.get_debug_name tn ^ " dst "
                       ^ Int.to_string context.device.ordinal;
                     work;
                   };
               true
           | (lazy None) ->
               [%diagn_sexp
                 [%log_entry
                   "to_host empty " ^ Tnode.get_debug_name tn;
                   [%log "nothing to copy", Tnode.label tn, Tnode.name tn, "to host"]]];
               false)

  let device_to_device ?rt tn ~into_merge_buffer ~dst ~src =
    if Option.is_some rt then
      raise
      @@ Utils.User_error
           "Multicore_backend.device_to_device: backend cannot be nested in another runtime";
    let dev = dst.device in
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.get_debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.get_debug_name tn);
    let schedule dst =
      let work =
        match into_merge_buffer with
        | No -> fun rt () -> Backend.to_buffer ~rt tn ~dst ~src:src.ctx
        | Streaming -> fun _rt () -> dev.merge_buffer_ptr := Backend.get_buffer tn src.ctx
        | Copy ->
            fun rt () ->
              let size_in_bytes = Tnode.size_in_bytes tn in
              let allocated_capacity =
                Option.value ~default:0 @@ Option.map dev.allocated_buffer ~f:snd
              in
              if allocated_capacity < size_in_bytes then
                dev.allocated_buffer <-
                  Some
                    ( Backend.alloc_buffer ?old_buffer:dev.allocated_buffer ~size_in_bytes (),
                      size_in_bytes );
              dev.merge_buffer_ptr := Option.map ~f:fst dev.allocated_buffer;
              Backend.to_buffer ~rt tn
                ~dst:(Option.value_exn ~here:[%here] !(dev.merge_buffer_ptr))
                ~src:src.ctx
      in
      schedule_task dev
        Tnode.
          {
            description =
              "device_to_device " ^ Tnode.get_debug_name tn ^ " dst " ^ Int.to_string dev.ordinal
              ^ " src " ^ Int.to_string src.device.ordinal;
            work;
          }
    in
    match (Backend.get_buffer tn dst.ctx, Backend.get_buffer tn src.ctx) with
    | Some dst, Some _ ->
        schedule dst;
        true
    | _ -> false

  let num_physical_devices () = Domain.recommended_domain_count () - 1
  let suggested_num_virtual_devices _device = 1
  let devices = Array.init (num_physical_devices ()) ~f:(fun ordinal -> spinup_device ~ordinal)

  let%track_sexp unsafe_cleanup ?(unsafe_shutdown = false) () =
    assert (Domain.is_main_domain ());
    let wait_for_finish device =
      await device;
      device.state.keep_spinning <- false;
      ignore (device.state.dev_wait.release_if_waiting () : bool)
    in
    Array.iter devices ~f:wait_for_finish;
    let cleanup ordinal device =
      Domain.join device.domain;
      device.host_wait_for_idle.finalize ();
      device.state.dev_wait.finalize ();
      if not unsafe_shutdown then devices.(ordinal) <- spinup_device ~ordinal
    in
    Array.iteri devices ~f:cleanup;
    Backend.unsafe_cleanup ~unsafe_shutdown ()

  let get_device ~ordinal = devices.(ordinal)
  let new_virtual_device device = device
  let get_physical_device device = device
  let get_ctx_device { device; _ } = device
  let get_name device = Int.to_string device.ordinal
  let to_ordinal { ordinal; _ } = ordinal
  let to_subordinal _ = 0
  let to_buffer ?rt tn ~dst ~src = Backend.to_buffer ?rt tn ~dst ~src:src.ctx
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
    merge_buffer:buffer_ptr option ref ->
    context ->
    procedure ->
    context * Indexing.lowered_bindings * Tnode.task * string

  val name : string
  val initialize : unit -> unit
  val is_initialized : unit -> bool
  val init : label:string -> context
  val finalize : context -> unit
  val unsafe_cleanup : ?unsafe_shutdown:bool -> unit -> unit

  val to_buffer :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Tnode.t -> dst:buffer_ptr -> src:context -> unit

  val host_to_buffer :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> dst:buffer_ptr -> unit

  val buffer_to_host :
    ?rt:(module Minidebug_runtime.Debug_runtime) -> Ndarray.t -> src:buffer_ptr -> unit
end

module Simple_no_device_backend (Backend : Simple_backend) : No_device_backend = struct
  include Backend

  type code =
    | Postponed of {
        lowered : Low_level.optimized;
        bindings : Indexing.unit_bindings;
        name : string;
      }
    | Compiled of Backend.procedure
  [@@deriving sexp_of]

  type code_batch =
    | Postponed of {
        lowereds : Low_level.optimized option array;
        bindings : Indexing.unit_bindings;
        names : string option array;
      }
    | Compiled of (ctx_arrays option * Backend.procedure option array)
  [@@deriving sexp_of]

  let global_config = ref Physical_devices_only

  let initialize config =
    global_config := config;
    initialize ()

  type nonrec routine = context routine [@@deriving sexp_of]

  let expected_merge_node : code -> _ = function
    | Postponed { lowered = Low_level.{ merge_node; _ }; _ } -> merge_node
    | Compiled proc -> Backend.expected_merge_node proc

  let expected_merge_nodes : code_batch -> _ = function
    | Postponed { lowereds; _ } ->
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.merge_node)))
    | Compiled (_, procs) ->
        Array.map ~f:(fun proc -> Option.(join @@ map proc ~f:Backend.expected_merge_node)) procs

  let compile ?(shared = false) ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    if shared then Compiled (Backend.compile ~name ~opt_ctx_arrays:None bindings lowered)
    else Postponed { lowered; bindings; name }

  let compile_batch ?(shared = false) ?names ?occupancy bindings asgns_l : code_batch =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    if shared then Compiled (compile_batch ~names ~opt_ctx_arrays:None bindings lowereds)
    else Postponed { lowereds; bindings; names }

  let link ~merge_buffer (old_context : context) (code : code) =
    let context, bindings, schedule, name =
      match code with
      | Postponed { lowered; bindings; name } ->
          let proc =
            Backend.compile ~name ~opt_ctx_arrays:(Some (ctx_arrays old_context)) bindings lowered
          in
          link_compiled ~merge_buffer old_context proc
      | Compiled code -> link_compiled ~merge_buffer old_context code
    in
    { context; schedule; bindings; name }

  let link_batch ~merge_buffer (old_context : context) (code_batch : code_batch) =
    let _opt_ctx_arrays, procs =
      match code_batch with
      | Postponed { lowereds; bindings; names } ->
          Backend.compile_batch ~names
            ~opt_ctx_arrays:(Some (ctx_arrays old_context))
            bindings lowereds
      | Compiled procs -> procs
    in
    Array.fold_map procs ~init:old_context ~f:(fun context -> function
      | Some proc ->
          let context, bindings, schedule, name = link_compiled ~merge_buffer context proc in
          (context, Some { context; schedule; bindings; name })
      | None -> (context, None))

  let to_buffer ?rt tn ~dst ~src = Backend.to_buffer ?rt tn ~dst ~src
  let host_to_buffer = Backend.host_to_buffer
  let buffer_to_host = Backend.buffer_to_host

  let get_buffer tn context =
    Map.find (Backend.ctx_arrays context) tn |> Option.map ~f:Backend.buffer_ptr
end

module C_device : No_device_backend = Simple_no_device_backend ((
  Cc_backend : Simple_backend with type context = Cc_backend.context))

module Cc_backend = Multicore_backend (C_device)

module Gccjit_device : No_device_backend = Simple_no_device_backend ((
  Gcc_backend : Simple_backend with type context = Gcc_backend.context))

module Gccjit_backend = Multicore_backend (Gccjit_device)

module Cuda_backend : Backend = struct
  include Cuda_backend

  type nonrec code = { code : code; expected_merge_node : Tnode.t option } [@@deriving sexp_of]

  type nonrec code_batch = { code_batch : code_batch; expected_merge_nodes : Tnode.t option array }
  [@@deriving sexp_of]

  let expected_merge_node code = code.expected_merge_node
  let expected_merge_nodes code_batch = code_batch.expected_merge_nodes
  let name = "cuda"

  type nonrec context = { ctx : context; expected_merge_node : Tnode.t option } [@@deriving sexp_of]
  type nonrec routine = context routine [@@deriving sexp_of]

  let compile ?shared:_ ?name bindings asgns : code =
    let name, lowered = lower_assignments ?name bindings asgns in
    { code = compile ~name bindings lowered; expected_merge_node = lowered.Low_level.merge_node }

  let compile_batch ?shared:_ ?names ?occupancy bindings asgns_l =
    let names, lowereds = lower_batch_assignments ?names ?occupancy bindings asgns_l in
    {
      code_batch = compile_batch ~names bindings lowereds;
      expected_merge_nodes =
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.Low_level.merge_node)));
    }

  let link context code =
    let ctx, bindings, schedule = link context.ctx code.code in
    { context = { ctx; expected_merge_node = code.expected_merge_node }; schedule; bindings; name }

  let link_batch context code_batch =
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
  let to_buffer ?rt tn ~dst ~src = to_buffer ?rt tn ~dst ~src:src.ctx
  let get_buffer tn context = get_buffer tn context.ctx
  let from_host ?rt context tn = from_host ?rt context.ctx tn
  let to_host ?rt context tn = to_host ?rt context.ctx tn

  let device_to_device ?rt tn ~into_merge_buffer ~dst ~src =
    if
      (not (equal_merge_buffer_use into_merge_buffer No))
      && not (Option.equal Tnode.equal (Some tn) dst.expected_merge_node)
    then
      raise
      @@ Utils.User_error
           ("Multicore_backend.device_to_device: merge node mismatch, expected "
           ^ Option.(value ~default:"none" @@ map ~f:Tnode.get_debug_name dst.expected_merge_node)
           ^ ", actual " ^ Tnode.get_debug_name tn);
    device_to_device ?rt tn ~into_merge_buffer ~dst:dst.ctx ~src:src.ctx
end

let reinitialize (module Backend : Backend) config =
  if not @@ Backend.is_initialized () then Backend.initialize config
  else (
    Core.Gc.full_major ();
    Backend.unsafe_cleanup ())
