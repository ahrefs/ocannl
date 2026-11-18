(* gh-ocannl-2311: the candidate shape for a renderer-dropped index guard, reached from user-level
   lowering, and why it reads in range anyway.

   Under a select whose condition reads a scope local, [Scope_order.arms] keeps on an inner
   definition only the guards evaluable anywhere ([Scope_order.evaluable_anywhere]: no division,
   modulo, memory or local read), so a definition under a Div-, Mod- or memory-bearing index guard
   renders ahead of that guard's test wherever the outer select short-circuits (on Metal, CUDA and
   HIP always; a backend that blends it keeps the inner guard instead). The question was whether
   lowering can make such a guard the ONLY protection of a read that leaves its node.

   The program: [out = where (tot - 6) (where g pool x) 0.25], [g = (r / 3 < 0.5) + (1.9 < r / 3)]
   (true at [i] in [{0, 1, 6, 7}]), [tot] a row sum of [s] (inlined: a [Local_scope] in the outer
   condition), [r] the iteration index ([range_of_shape], inlined as an [Embed_index]), and [pool] a
   padded max window over [x] (inlined: a [Local_scope] in the inner select's then-arm), whose
   window reads [x] off both ends. Lowering reaches the shape -- an outer condition reading a local,
   an inner index guard that is not evaluable anywhere (the division), and a definition under it
   reading [x] off its node -- but every read of [x] carries its own clamp, a [Where] on affine
   index comparisons that is evaluable anywhere: lowering builds a range guard at the read it
   protects ([Assignments.clamp_conds]), so the dropped guard was never the read's protection.

   Pinned: (1) the optimized IR has the shape; a divisionless twin, [g = (r < 1.5) + (5.7 < r)] (the
   same iterations), does not, and has it over a guard evaluable anywhere instead: the predicate
   tells the two guards apart (its control). (2) Every read of [x] at a window offset sits under an
   arm of an anywhere index guard. (3) The values equal the materialized reference's and the closed
   form's. The structural claim (2) does not check a clamp's bounds; the values do: [x] peaks at
   each end tap the selected windows reach ([x.(0)] at [i = 0], [x.(7)] at [i = 7]), so a clamp
   excluding a valid edge tap changes a selected cell, and one admitting an off-node tap reads
   outside the buffer. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let n = 8
let xv = [| 30.; 20.; 40.; 7.; 13.; 50.; 21.; 29. |]

let sv =
  Array.init (n * 2) ~f:(Ll_test.cycle_flat ~dims:[| n; 2 |] ~modulus:7 ~offset:0. ~stride:1.)

let threshold = 6.

let selected i =
  let g = Float.of_int i /. 3. in
  Float.(g < 0.5 || 1.9 < g)

(* The closed form: [tot] is [1 5 9 6 3 7 11 1], so only [i = 3] takes the outer else-arm, and a
   selected [i] takes the window max of [x] at [i - 1 .. i + 1], clipped at both ends. *)
let expected =
  Array.init n ~f:(fun i ->
      let tot = sv.(2 * i) +. sv.((2 * i) + 1) in
      let pool =
        List.filter_map
          [ i - 1; i; i + 1 ]
          ~f:(fun j -> if j >= 0 && j < n then Some xv.(j) else None)
        |> List.reduce_exn ~f:Float.max
      in
      if Float.(tot - threshold <> 0.) then if selected i then pool else xv.(i) else 0.25)

(* The statement-level scalars of [code], scope bodies not entered. [unclamped] needs them, and an
   arm-versus-condition context: [Ll_test.walk] cannot give that, since its [on_exit] fires for
   statements only and a scalar's subtree has no exit hook. Both matches are exhaustive, so a new
   [Low_level] constructor fails to compile here rather than going unvisited. *)
let rec stmt_scalars (code : LL.t) : LL.scalar_t list =
  match code with
  | LL.Seq (a, b) -> stmt_scalars a @ stmt_scalars b
  | LL.For_loop { body; _ } | LL.Tile_mma { fallback = body; _ } -> stmt_scalars body
  | LL.Scan_loop { carried; body; _ } ->
      List.map carried ~f:(fun (c : LL.carried) -> c.init) @ stmt_scalars body
  | LL.If { cond = c, _; body } -> c :: stmt_scalars body
  | LL.Set { llsc; _ } | LL.Set_local (_, llsc) -> [ llsc ]
  | LL.Set_from_vec { arg = a, _; _ } -> [ a ]
  | LL.Set_dynamic { dyn_value = v, _; llsc; _ } -> [ v; llsc ]
  | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Zero_out _ | LL.Declare_local _
  | LL.Workgroup_barrier ->
      []

(* [s] and its operands, transitively ([Ll_test.walk_scalar]); [deep] also enters scope bodies. *)
let subterms ~deep (s : LL.scalar_t) : LL.scalar_t list =
  let acc = ref [] in
  Ll_test.walk_scalar ~in_scopes:deep ~on_scalar:(fun t -> acc := t :: !acc) s;
  List.rev !acc

(* Every scalar of [llc], scope bodies included. *)
let all_scalars (llc : LL.t) : LL.scalar_t list =
  let acc = ref [] in
  Ll_test.walk ~on_scalar:(fun t -> acc := t :: !acc) llc;
  List.rev !acc

let shallow_exists c ~f = List.exists (subterms ~deep:false c) ~f
let mentions_index c = shallow_exists c ~f:(function LL.Embed_index _ -> true | _ -> false)

let reads_local c =
  shallow_exists c ~f:(function LL.Local_scope _ | LL.Get_local _ -> true | _ -> false)

let index_guard c = mentions_index c && not (reads_local c)

let reads_node x s =
  List.exists (subterms ~deep:true s) ~f:(function LL.Get (tn, _) -> Tn.equal tn x | _ -> false)

(* A read of [x] at a window offset: the reads that can leave its node (the in-range else-arm read
   [x[i]] has none). *)
let window_read x = function
  | LL.Get (tn, [| Ir.Indexing.Affine { offset; _ } |]) -> Tn.equal tn x && offset <> 0
  | _ -> false

let arms_of = function
  | LL.Ternop (Ir.Ops.Where, (c, _), (a, _), (b, _)) ->
      Some (c, subterms ~deep:false a @ subterms ~deep:false b)
  | _ -> None

(* The issue's shape, rooted at [outer]: a select whose condition reads a local, over an index guard
   not evaluable anywhere, over a definition reading [x]; with [~anywhere:true], the same shape over
   a guard evaluable anywhere (the control's). *)
let shape_root ~anywhere x outer =
  match arms_of outer with
  | Some (c_out, inside) when reads_local c_out ->
      List.exists inside ~f:(fun inner ->
          match arms_of inner with
          | Some (c_in, held)
            when index_guard c_in && Bool.equal anywhere (LL.Scope_order.evaluable_anywhere c_in) ->
              List.exists held ~f:(function
                | LL.Local_scope _ as def -> reads_node x def
                | _ -> false)
          | _ -> false)
  | _ -> false

(* Whether [s] holds a {!window_read} of [x] outside every arm of an index guard evaluable
   anywhere. *)
let rec unclamped x ~clamped (s : LL.scalar_t) =
  match s with
  | LL.Get _ -> (not clamped) && window_read x s
  | LL.Ternop (Ir.Ops.Where, (c, _), (a, _), (b, _)) ->
      let clamped' = clamped || (index_guard c && LL.Scope_order.evaluable_anywhere c) in
      unclamped x ~clamped c || unclamped x ~clamped:clamped' a || unclamped x ~clamped:clamped' b
  | LL.Local_scope { body; _ } -> List.exists (stmt_scalars body) ~f:(unclamped x ~clamped)
  | LL.Get_dynamic { dyn_value = v, _; _ } | LL.Unop (_, (v, _)) -> unclamped x ~clamped v
  | LL.Binop (_, (a, _), (b, _)) -> unclamped x ~clamped a || unclamped x ~clamped b
  | LL.Ternop (_, (a, _), (b, _), (c, _)) ->
      unclamped x ~clamped a || unclamped x ~clamped b || unclamped x ~clamped c
  | LL.Constant _ | LL.Constant_bits _ | LL.Embed_index _ | LL.Get_local _ | LL.Get_merge_buffer _
    ->
      false

(* One run: [materialize_all] keeps every intermediate in memory (the value reference);
   [divisionless] spells the inner guard without the division, selecting the same iterations (the
   shape predicate's control). *)
let run ?(divisionless = false) ~materialize_all () =
  Tensor.unsafe_reinitialize ();
  let x = TDSL.ndarray xv ~label:[ "vsm_x" ] ~output_dims:[ n ] () in
  let s = TDSL.ndarray sv ~label:[ "vsm_s" ] ~output_dims:[ n; 2 ] () in
  let%op pool = x @^+ "o=+w; w => o" [ "w" ] (stretch 0.0) in
  Shape.set_dim w 3;
  let%op r = TDSL.range_of_shape ~output_dims:[ n ] () in
  let par =
    if divisionless then
      let%op par = (r < 1.5) + (5.7 < r) in
      par
    else
      let%op par = (r /. 3.0 < 0.5) + (1.9 < r /. 3.0) in
      par
  in
  let%op tot = s ++ "ok => o" in
  let%op out = where (tot - 6.0) (where par pool x) (0.25 + 0.0) in
  let ctx = Context.auto () in
  List.iter [ x; s; out ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  if materialize_all then
    List.iter [ pool; r; par; tot ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let observed = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o ->
        observed := Some o.LL.llc;
        [ o ])
      ctx
      (let comp = Train.forward out in
       let name =
         if materialize_all then "vsd_materialized"
         else if divisionless then "vsd_divisionless"
         else "vsd_inlined"
       in
       { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) })
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (x.Tensor.value, Option.value_exn ~here:[%here] !observed, Context.get_values ctx out.Tensor.value)

let () =
  Stdio.eprintf "value_select_division_guard: backend %s (not part of the golden)\n%!"
    (Utils.get_global_arg ~arg_name:"backend" ~default:"");
  let x, llc, got = run ~materialize_all:false () in
  let x_c, llc_c, got_c = run ~divisionless:true ~materialize_all:false () in
  let _, _, want = run ~materialize_all:true () in
  p_exists
    "the inlined program has the shape: a local-reading select over a division guard over a \
     definition reading x"
    (all_scalars llc) ~f:(shape_root ~anywhere:false x);
  p_none "the divisionless control does not (the shape predicate's control)" (all_scalars llc_c)
    ~f:(shape_root ~anywhere:false x_c);
  p_exists "the divisionless control has the same shape over a guard evaluable anywhere"
    (all_scalars llc_c) ~f:(shape_root ~anywhere:true x_c);
  p_all2 "the divisionless control's values equal the materialized reference's" got_c want
    ~f:Float.equal;
  p_exists "the inlined program reads x below its node (the window's [-1] offset)" (all_scalars llc)
    ~f:(function
    | LL.Get (tn, [| Ir.Indexing.Affine { offset; _ } |]) -> Tn.equal tn x && offset < 0
    | _ -> false);
  p_none
    "every read of x at a window offset sits in an arm of an index guard evaluable anywhere, its \
     own clamp"
    (stmt_scalars llc) ~f:(unclamped x ~clamped:false);
  p_all2 "the inlined values equal the materialized reference's" got want ~f:Float.equal;
  p_all2 "the values equal the closed form's" got expected ~f:Float.equal
