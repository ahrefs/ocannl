(* gh-ocannl-1747: what a dropped graph's buffers come back through. gh-ocannl-1745 made the session
   release the OCaml graph (dropped_graph_reclaimed); this test follows the memory behind it.

   - A lineage the caller releases, leaves first, gives back its buffers: the pool table's working
   pools and the device's used-memory gauge return to the baseline (on cc a GC finalizer frees the
   host buffer once the table lets go of it, so there the gauge is the GC's own witness), every pool
   the window allocated is counted freed, and the graph's nodes and the parameter's host-init array
   are reclaimed. Before gh-ocannl-1747 the device's writer-event table ([updating_for]) held every
   node a routine or an upload ever wrote, for the life of the device, so no dropped graph's nodes
   -- nor, through them, its host-init data -- were ever collected. - The negative control: an
   unreleased lineage keeps its pools in the pool table, exactly their bytes and nothing else
   (gh-ocannl-550: release is the only path back), though no longer its nodes. - gh-ocannl-1762:
   [Train.forward_once]'s initialization context, which the caller never holds, is released with the
   context it returns (which adopted it), so the parameter's pool and every context the call minted
   come back; an upload fork of the returned context holds it too, so releasing the chain alone
   keeps the fork usable and the pool live, and releasing the fork gives it back. - A fresh graph's
   host literal lives in a device-wide constant pool that every release unit binding it holds,
   across roots: releasing the last binder frees the pool and drops the cache's strong key, so the
   literal's node and host array are reclaimed. A second root's binding keeps the pool, and its
   routine keeps computing, after the first root's release; an unreleased binder keeps the pool and
   the node (the control).

   Witnesses are [Gc_witness] observers taken in [@inline never] builders that return no other
   reference to the graph; every reading clears the bounded analysis cache and follows
   [Gc.full_major] twice and [Gc.compact], a "reclaimed" claim retries up to three such rounds, and
   the literal's "kept" control requires its node to survive three. A graph's writer events are
   pruned at the next write, so each case runs one more graph before it reads.

   Every memory claim reads the process's own accounting ([Alloc_census], the backend's used gauge),
   never the driver's free bytes (gh-ocannl-1798): those are device-wide, so any other process on
   the GPU -- the suite's concurrent GPU tests included -- moves them between two readings. They are
   printed on stderr, as information. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Census = Ir.Alloc_census

let n = 1 lsl 21
let graphs = 3

type reading = {
  contexts : int;
  working : int;
  working_bytes : int;
  constant_bytes : int;
  pool_bytes : int;
  allocated : int;
  freed : int;
  used : int;
  free : int option;
  host : int;  (** The live host arrays' bytes, {!Ir.Ndarray.get_used_memory}'s gauge. *)
}

(* The analysis cache keeps the last eight compiles' lowered code, hence their nodes: a bounded
   retention by design ([Low_level.clear_analysis_cache]), cleared so the readings see past it. *)
let read () =
  Ir.Low_level.clear_analysis_cache ();
  Gc_witness.settle ();
  let c = Census.snapshot () in
  (* Released again, so the probes leave the context count where they found it. *)
  let probe = Context.auto () in
  let used = Context.get_used_memory probe in
  let free = Option.map (Context.get_free_and_total_memory probe) ~f:fst in
  Context.release probe;
  {
    contexts = Census.unreleased_contexts c;
    working = c.live_working_pools;
    working_bytes = c.live_working_bytes;
    constant_bytes = c.live_constant_bytes;
    pool_bytes = Census.live_pool_bytes c;
    allocated = c.working_pools_allocated + c.constant_pools_allocated;
    freed = c.pools_freed;
    used;
    free;
    host = Ir.Ndarray.get_used_memory ();
  }

let show name r =
  Stdio.eprintf
    "%s: unreleased contexts %d, working pools %d (%d bytes), constant %d bytes, pools allocated \
     %d freed %d, used %d, driver free %s (not part of the golden)\n\
     %!"
    name r.contexts r.working r.working_bytes r.constant_bytes r.allocated r.freed r.used
    (Option.value_map r.free ~default:"-" ~f:Int.to_string)

(* How far the driver's free bytes moved from [before] to [after]: device-wide, so information
   only. *)
let show_driver name ~before ~after =
  Option.iter (Option.both before.free after.free) ~f:(fun (b, a) ->
      Stdio.eprintf "%s: driver free bytes moved by %d (not part of the golden)\n%!" name (a - b))

let backend = Context.backend_name (Context.auto ())

type built = { nodes : Gc_witness.t list; host_init : Gc_witness.t; bytes : int; computed : bool }

let value i j = Float.of_int ((i + j) % 7)

let param i =
  TDSL.param ~values:(Array.init n ~f:(value i)) (Printf.sprintf "dgb_w%d" i) ~output_dims:[ n ] ()

let doubles i got =
  Array.length got = n && Array.for_alli got ~f:(fun j v -> Float.equal v (2. *. value i j))

let host_init (w : Tensor.t) =
  Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find w.value))

(* [t = w *. 2.] over a fresh parameter, initialized and run through the explicit lineage. *)
let[@inline never] lineage ~release i =
  let w = param i in
  let%op t = w *. 2. in
  let root = Context.auto () in
  let c1 = Train.init_params root IDX.empty t in
  let c2, r = Train.to_routine c1 IDX.empty (Train.forward t) in
  let c2 = Context.run c2 r in
  let got = Context.get_values c2 t.value in
  let nd = host_init w in
  if release then (
    Context.release c2;
    Context.release c1;
    Context.release root);
  {
    nodes = [ Gc_witness.witness w.value; Gc_witness.witness t.value ];
    host_init = Gc_witness.witness nd;
    bytes = Ir.Ndarray.size_in_bytes nd;
    computed = doubles i got;
  }

(* [t = w *. 2.] through [Train.forward_once], whose returned context is all the caller holds. *)
let[@inline never] forward_once_released i =
  let w = param i in
  let%op t = w *. 2. in
  let root = Context.auto () in
  let ctx = Train.forward_once root t in
  let computed = doubles i (Context.get_values ctx t.value) in
  let bytes = Ir.Ndarray.size_in_bytes (host_init w) in
  Context.release ctx;
  Context.release root;
  (computed, bytes)

(* A node the returned context lacks: uploading it continues the context's linear chain the first
   time, and forks it the second time. *)
let fork_node = Ll_test.node_factory ~first_id:17620 ~dims:[| 4 |] ()
let fork_seed = [| 1.; 2.; 3.; 4. |]

type forked = {
  fork_reads : bool;  (** The fork still read the parameter after the chain's release. *)
  pool_live : bool;  (** The parameter's pool was still live then. *)
  fk_bytes : int;
}

(* An upload fork of [forward_once]'s result binds the parameter its initialization context owns, so
   it holds that context too: the chain's release must leave it -- a released lifecycle ancestor
   would refuse the fork -- and the fork's release gives it back. *)
let[@inline never] forward_once_forked i =
  let w = param i in
  let%op t = w *. 2. in
  let root = Context.auto () in
  let ctx = Train.forward_once root t in
  let y = fork_node "dgb_fork" in
  Ll_test.materialize y;
  let chain = Context.set_values ctx y fork_seed in
  let fork = Context.set_values ctx y fork_seed in
  let bytes = Ir.Ndarray.size_in_bytes (host_init w) in
  let mid = Census.snapshot () in
  Context.release chain;
  let after_chain = Census.snapshot () in
  let fork_reads =
    match Context.get_values fork w.value with
    | got -> Array.length got = n && Array.for_alli got ~f:(fun j v -> Float.equal v (value i j))
    | exception _ -> false
  in
  let pool_live = after_chain.live_working_bytes - mid.live_working_bytes > -bytes in
  Context.release fork;
  Context.release root;
  { fork_reads; pool_live; fk_bytes = bytes }

let literal i =
  TDSL.ndarray
    (Array.init n ~f:(value i))
    ~label:[ Printf.sprintf "dgb_x%d" i ]
    ~output_dims:[ n ] ()

type literal_built = {
  lit_node : Gc_witness.t;
  lit_host : Gc_witness.t;
  lit_bytes : int;
  lit_computed : bool;
}

(* [t = x *. 2.] over a fresh host literal, released ([release]) or not. *)
let[@inline never] literal_graph ~release i =
  let x = literal i in
  let%op t = x *. 2. in
  let root = Context.auto () in
  let ctx, r = Train.to_routine root IDX.empty (Train.forward t) in
  let ctx = Context.run ctx r in
  let lit_computed = doubles i (Context.get_values ctx t.value) in
  let nd = host_init x in
  if release then (
    Context.release ctx;
    Context.release root);
  {
    lit_node = Gc_witness.witness x.value;
    lit_host = Gc_witness.witness nd;
    lit_bytes = Ir.Ndarray.size_in_bytes nd;
    lit_computed;
  }

type shared = {
  first_computed : bool;
  second_computed : bool;  (** The second root, rerun after the first root's release. *)
  between : int;  (** Constant bytes after the first root's release, less the window's start. *)
  sh_node : Gc_witness.t;
  sh_bytes : int;
}

(* One host literal bound by two independently created roots: one device buffer, held by both. *)
let[@inline never] literal_two_roots ~start i =
  let x = literal i in
  let%op t = x *. 2. in
  let%op u = x *. 2. in
  let root_a = Context.auto () and root_b = Context.auto () in
  let ctx_a, r_a = Train.to_routine root_a IDX.empty (Train.forward t) in
  let ctx_a = Context.run ctx_a r_a in
  let ctx_b, r_b = Train.to_routine root_b IDX.empty (Train.forward u) in
  let ctx_b = Context.run ctx_b r_b in
  let first_computed = doubles i (Context.get_values ctx_a t.value) in
  Context.release ctx_a;
  Context.release root_a;
  let between = (Census.snapshot ()).live_constant_bytes - start in
  let ctx_b = Context.run ctx_b r_b in
  let second_computed = doubles i (Context.get_values ctx_b u.value) in
  let sh_bytes = Ir.Ndarray.size_in_bytes (host_init x) in
  Context.release ctx_b;
  Context.release root_b;
  { first_computed; second_computed; between; sh_node = Gc_witness.witness x.value; sh_bytes }

let builds ~from f = List.init graphs ~f:(fun i -> f (from + i))

(* One more released graph, whose writes prune the events of the graphs before it. *)
let[@inline never] one_more i = ignore (lineage ~release:true i : built)

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend;
  (* Warm-up: the device, its merge buffer and the code caches. *)
  one_more 0;
  case "a released lineage gives its buffers back" (fun () ->
      let before = read () in
      show "released: before" before;
      let built = builds ~from:1 (lineage ~release:true) in
      one_more 9;
      let after = read () in
      show "released: after" after;
      p_all "released: every graph computed t = 2 w" built ~f:(fun b -> b.computed);
      p "released: the live working pools are back to the baseline"
        (after.working = before.working && after.working_bytes = before.working_bytes);
      p "released: the device's used-memory gauge is back to the baseline" (after.used = before.used);
      p "released: every context the window created was released" (after.contexts = before.contexts);
      show_driver "released" ~before ~after;
      p "released: every pool the window allocated was freed, and the pool bytes are back"
        (after.allocated > before.allocated
        && after.freed - before.freed = after.allocated - before.allocated
        && after.pool_bytes = before.pool_bytes);
      p_all "released: every graph's nodes are reclaimed" built ~f:(fun b ->
          List.for_all b.nodes ~f:Gc_witness.reclaimed);
      p_all "released: every parameter's host-init array is reclaimed" built ~f:(fun b ->
          Gc_witness.reclaimed b.host_init));
  case "an unreleased lineage keeps its pools, not its graph" (fun () ->
      let before = read () in
      show "unreleased: before" before;
      let built = builds ~from:11 (lineage ~release:false) in
      one_more 19;
      let after = read () in
      show "unreleased: after" after;
      let bytes = (List.hd_exn built).bytes in
      p_all "unreleased: every graph computed t = 2 w" built ~f:(fun b -> b.computed);
      p "unreleased: two working pools per graph stay in the pool table (gh-ocannl-550)"
        (after.working - before.working = 2 * graphs
        && after.working_bytes - before.working_bytes = 2 * graphs * bytes);
      show_driver "unreleased" ~before ~after;
      p "unreleased: exactly those pools are left unfreed, and the pool bytes grew by theirs alone"
        (after.allocated - before.allocated - (after.freed - before.freed) = 2 * graphs
        && after.pool_bytes - before.pool_bytes = 2 * graphs * bytes);
      p_all "unreleased: every graph's nodes are reclaimed all the same" built ~f:(fun b ->
          List.for_all b.nodes ~f:Gc_witness.reclaimed));
  case "gh-ocannl-1762: forward_once's initialization context is released with its result"
    (fun () ->
      let before = read () in
      let built = builds ~from:21 forward_once_released in
      one_more 29;
      let after = read () in
      show "forward_once: after" after;
      p_all "forward_once: every graph computed t = 2 w" built ~f:fst;
      p "forward_once: the live working pools are back to the baseline"
        (after.working = before.working && after.working_bytes = before.working_bytes);
      p "forward_once: every context the calls minted was released, initialization included"
        (after.contexts = before.contexts);
      p "forward_once: the device's used-memory gauge is back to the baseline"
        (after.used = before.used);
      show_driver "forward_once" ~before ~after;
      p "forward_once: every pool the window allocated was freed"
        (after.allocated > before.allocated
        && after.freed - before.freed = after.allocated - before.allocated));
  case "gh-ocannl-1762: an upload fork of forward_once's result holds its initialization context"
    (fun () ->
      let before = read () in
      let built = builds ~from:41 forward_once_forked in
      one_more 49;
      let after = read () in
      show "fork: after" after;
      p_all "fork: after the chain's release the fork still reads the parameter" built ~f:(fun b ->
          b.fork_reads);
      p_all "fork: after the chain's release the parameter's pool is still live" built ~f:(fun b ->
          b.pool_live && b.fk_bytes > 0);
      p "fork: releasing the fork gives the pools, the contexts and the used memory back"
        (after.working = before.working
        && after.working_bytes = before.working_bytes
        && after.contexts = before.contexts && after.used = before.used));
  case "gh-ocannl-1762: a released graph's host literal gives its constant pool back" (fun () ->
      let before = read () in
      let built = builds ~from:31 (literal_graph ~release:true) in
      one_more 39;
      let after = read () in
      show "literal: after" after;
      p_all "literal: every graph computed t = 2 x" built ~f:(fun b -> b.lit_computed);
      p "literal: the constant bytes are back to the baseline"
        (after.constant_bytes = before.constant_bytes);
      p "literal: the device's used-memory gauge is back to the baseline" (after.used = before.used);
      show_driver "literal" ~before ~after;
      p "literal: every pool the window allocated was freed"
        (after.allocated > before.allocated
        && after.freed - before.freed = after.allocated - before.allocated);
      p_all "literal: every literal's node is reclaimed" built ~f:(fun b ->
          Gc_witness.reclaimed b.lit_node);
      p_all "literal: every literal's host array is reclaimed" built ~f:(fun b ->
          Gc_witness.reclaimed b.lit_host);
      p "literal: the live host bytes are back to the baseline" (after.host = before.host));
  case "gh-ocannl-1762: a literal two roots bind is freed with the second binder" (fun () ->
      let before = read () in
      let start = (Census.snapshot ()).live_constant_bytes in
      let built = builds ~from:51 (literal_two_roots ~start) in
      one_more 59;
      let after = read () in
      show "two roots: after" after;
      p_all "two roots: the first root computed t = 2 x" built ~f:(fun b -> b.first_computed);
      p_all "two roots: the second root still computes after the first root's release" built
        ~f:(fun b -> b.second_computed);
      p_all "two roots: the first root's release left the literal's one constant buffer in place"
        built ~f:(fun b -> b.sh_bytes > 0 && b.between = b.sh_bytes);
      p "two roots: the second root's release gave every constant pool and its used memory back"
        (after.constant_bytes = before.constant_bytes && after.used = before.used);
      p_all "two roots: every literal's node is reclaimed" built ~f:(fun b ->
          Gc_witness.reclaimed b.sh_node));
  case "gh-ocannl-1762: an unreleased literal binder keeps its constant pool and node" (fun () ->
      let before = read () in
      let built = builds ~from:61 (literal_graph ~release:false) in
      one_more 69;
      let after = read () in
      show "unreleased literal: after" after;
      p "unreleased literal: the constant bytes grew by each graph's literal"
        (after.constant_bytes - before.constant_bytes
        = List.sum (module Int) built ~f:(fun b -> b.lit_bytes));
      p "unreleased literal: the live host bytes grew by each graph's literal"
        (after.host - before.host = List.sum (module Int) built ~f:(fun b -> b.lit_bytes));
      p_all "unreleased literal: every literal's node is kept by the device's constant cache" built
        ~f:(fun b -> Gc_witness.kept b.lit_node))
