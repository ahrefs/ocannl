(* Regression test for gh-ocannl-1125: host uploads of nodes no routine has linked yet share pools.

   Loading parameters with [Context.set_values] before the first compile -- restoring a checkpoint
   into a fresh context, or a benchmark injecting fixture weights -- reaches
   [Backend.init_from_host] for every node, because none of them is in the context yet. That used to
   give each node a pool of its own, so a routine reading more of them than Metal binds per kernel
   ([metal_max_pools] = 16) could not link at all: [bench_gpt]'s training step failed with "routine
   needs 20 distinct pools". They are now bump-packed into the context lifecycle's upload arenas,
   whose capacities double.

   [n] parameters of one size are uploaded one [set_values] at a time, then one routine reads all of
   them. The claims, on every backend: - the uploads took at most [1 + ceil(log2 n)] working pools
   (the doubling policy's bound; the regression is [n] of them), holding at most twice the bytes
   uploaded; - the routine computes the sum of all of them, and every parameter reads back what was
   uploaded (distinct per parameter and per element, so an overlap or a wrong offset shows); -
   pool-mates outlive a released child: a context compiled from the uploads' context gets arenas of
   its own for ITS uploads, so releasing it frees nothing the parent's routine reads. And on Metal
   only, where the binding budget exists: the routine linked. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let () =
  Stdio.eprintf "set_values_pool_coalescing: backend=%s (not part of the golden)\n%!" backend_name

(* Over Metal's 16-pool binding budget with room to spare. *)
let n = 24
let len = 64

(* Distinct per parameter and per element, exact in single precision, and every partial sum of the
   routine below stays an integer far below 2^24. *)
let value k i = Float.of_int ((k * 1000) + i + 1)
let values k = Array.init len ~f:(value k)
let ceil_log2 m = if m <= 1 then 0 else Int.ceil_log2 m

let () =
  Tensor.unsafe_reinitialize ();
  let zeros = Ir.Ndarray.init_array ~debug:"svpc" Ir.Ops.single ~dims:[| len |] ~padding:None in
  (* Parameters the way [bench_gpt] builds its fixture-backed weights: a host init, no init code. *)
  let params =
    List.init n ~f:(fun k ->
        TDSL.wrap_param ~l:(Printf.sprintf "svpc_p%d" k) ~o:[ len ] (zeros ~f:(fun _ -> 0.)) ())
  in
  let%op sum = List.reduce_exn params ~f:(fun a b -> a + b) in
  Train.set_materialized sum.Tensor.value;
  let ctx = Context.auto () in
  let before = Ir.Alloc_census.snapshot () in
  let ctx =
    List.foldi params ~init:ctx ~f:(fun k ctx p -> Context.set_values ctx p.Tensor.value (values k))
  in
  let after = Ir.Alloc_census.snapshot () in
  let pools = after.live_working_pools - before.live_working_pools in
  let bytes = after.live_working_bytes - before.live_working_bytes in
  let uploaded =
    List.sum
      (module Int)
      params
      ~f:(fun p -> Ir.Tnode.num_elems p.Tensor.value * Ir.Ops.prec_in_bytes Ir.Ops.single)
  in
  Stdio.eprintf "uploads: %d pools, %d bytes for %d uploaded (not part of the golden)\n%!" pools
    bytes uploaded;
  pf "%d uploads took at most 1 + ceil(log2 %d) working pools" n n (pools <= 1 + ceil_log2 n);
  p "the uploads' pools hold at most twice the bytes uploaded" (bytes <= 2 * uploaded);
  let linked =
    match Context.compile ~name:"svpc_sum" ctx (Train.forward sum) Ir.Indexing.Empty with
    | linked -> Ok linked
    | exception Utils.User_error msg -> Error msg
  in
  gated
    ~when_:(String.equal backend_name "metal")
    ~on:backend_name "the routine reading every upload linked within Metal's pool binding budget"
    (Result.is_ok linked);
  let ctx, routine =
    match linked with Ok linked -> linked | Error msg -> failwith ("svpc_sum did not link: " ^ msg)
  in
  let ctx = Context.run ctx routine in
  let expected_sum =
    Array.init len ~f:(fun i ->
        List.sum (module Float) (List.init n ~f:Fn.id) ~f:(fun k -> value k i))
  in
  p_all2 "the routine sums every upload"
    (Context.get_values ctx sum.Tensor.value)
    expected_sum ~f:Float.equal;
  let reads_back ctx k p =
    Array.equal Float.equal (Context.get_values ctx p.Tensor.value) (values k)
  in
  p_alli "every parameter reads back its own upload" params ~f:(reads_back ctx);
  (* A child lifecycle uploading a node of its own: that upload must land in the child's arena, not
     in a parent arena the child's release would then free under the parameters. *)
  let extra = TDSL.wrap_param ~l:"svpc_extra" ~o:[ len ] (zeros ~f:(fun _ -> 0.)) () in
  let%op probe = sum + 1 in
  let child_ctx, _probe =
    Context.compile ~name:"svpc_probe" ctx (Train.forward probe) Ir.Indexing.Empty
  in
  let child_ctx = Context.set_values child_ctx extra.Tensor.value (values n) in
  p "the child reads back its own upload"
    (Array.equal Float.equal (Context.get_values child_ctx extra.Tensor.value) (values n));
  Context.release child_ctx;
  let ctx = Context.run ctx routine in
  p_all2 "after the child's release the routine still sums every upload"
    (Context.get_values ctx sum.Tensor.value)
    expected_sum ~f:Float.equal;
  p_alli "after the child's release every parameter still reads back its own upload" params
    ~f:(reads_back ctx)

(* Sibling values of one lifecycle (review round 1): [a] extends a fresh root with three uploads,
   the third minting an arena with room to spare, and [b] uploads one node into the same ROOT. [b]
   does not hold [a]'s tenants, so it must not bump into [a]'s arena: releasing [a] frees the arenas
   its uploads live in, and [b]'s own upload has to survive that. *)
let () =
  let zeros = Ir.Ndarray.init_array ~debug:"svpc_sib" Ir.Ops.single ~dims:[| len |] ~padding:None in
  let fresh l = TDSL.wrap_param ~l ~o:[ len ] (zeros ~f:(fun _ -> 0.)) () in
  let xs = List.init 3 ~f:(fun k -> fresh (Printf.sprintf "svpc_sib_x%d" k)) in
  let y = fresh "svpc_sib_y" in
  let root = Context.auto () in
  let a =
    List.foldi xs ~init:root ~f:(fun k ctx x -> Context.set_values ctx x.Tensor.value (values k))
  in
  let b = Context.set_values root y.Tensor.value (values 7) in
  Context.release a;
  p "a sibling's upload survives the other sibling's release"
    (Array.equal Float.equal (Context.get_values b y.Tensor.value) (values 7))
