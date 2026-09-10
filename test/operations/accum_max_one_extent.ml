(* gh-ocannl-878: gh-490's runtime-extent guard is emitted per product ITERATOR, so an axis that has
   no iterator gets no guard. A maximum-one symbolic axis is exactly that shape -- gh-ocannl-817
   retains its extent metadata under [None] precisely so a consumer can refuse it -- and this
   settles that [Accum_op] could reach it, that the consequence is a wrong VALUE rather than a
   merely-untidy write, and that lowering now refuses it loudly.

   The reachability leg is ordinary [%op] code, not hand-built IR: reducing over a symbolic axis is
   what [test/operations/symbolic_extent_launch.ml] already does at range 6, where extent 0 sums to
   0.0. Repeating it at range 1 is the whole defect. The range-2 twin is the control: same graph,
   same binding, an iterator to guard -- so a difference between the two legs can only come from the
   missing guard.

   The executed leg reconstructs the pre-boundary program deliberately: after the refusal the bound
   path no longer lowers, so the wrong value is shown through the lowering [static_indices] never
   reached (there is no loop for them to guard here -- the [If] census pins that), compiled with the
   launch binding still in place so the extent is genuinely bound to zero at run time. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let rec find_accum = function
  | Asgns.Accum_op _ as a -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_accum a) (find_accum b)
  | Asgns.Block_comment (_, body) -> find_accum body
  | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

let rejection f = match f () with () -> None | exception Utils.User_error msg -> Some msg

(* Harness misuse, which [Ll_test] reports as [Invalid_argument] rather than as a user error. *)
let misuse f = match f () with () -> None | exception Invalid_argument msg -> Some msg

let says rejection substring =
  Option.value_map rejection ~default:false ~f:(String.is_substring ~substring)

let count_ifs llc = Ll_test.count_stmt llc ~f:(function LL.If _ -> true | _ -> false)

type case = {
  extent : Idx.static_symbol;
  bindings : Idx.unit_bindings;
  x : Tensor.t;
  y : Tensor.t;
  fwd : Asgns.comp;
  accum : Asgns.t;
  projections : Idx.projections;
}

(* The reduction [x ++ "s=>0"] over a symbolic axis [s], its extent declared with maximum [range]
   and bound as a launch parameter. The result [y] is a CONCRETE one-element tensor -- the [=>0]
   result axis is not the symbolic one -- which is what makes a wrong value there an answer rather
   than an untouched margin. *)
let sum_over_symbolic_axis ~range =
  Tensor.unsafe_reinitialize ();
  let extent, bindings = Idx.get_static_symbol ~static_range:range Idx.Empty in
  let%op x = { x = 0.5 } in
  let%op y = x ++ "s=>0" [ "s" ] in
  Shape.set_sym_dim s extent;
  Train.set_materialized y.Tensor.value;
  let fwd = Train.forward y in
  let accum = Option.value_exn ~here:[%here] (find_accum fwd.Asgns.asgns) in
  let projections =
    match accum with
    | Asgns.Accum_op { projections; _ } -> Lazy.force projections
    | _ -> failwith "expected the reduction to lower as Accum_op"
  in
  { extent; bindings; x; y; fwd; accum; projections }

let () =
  (* {1 The control: a maximum-TWO extent, which has an iterator to guard.} *)
  let two = sum_over_symbolic_axis ~range:2 in
  let bound2 = Idx.bound_symbols two.bindings in
  p "a maximum-two symbolic reduction axis carries a product iterator"
    (List.exists two.projections.Idx.extent_syms ~f:(fun (iter, sym) ->
         Option.is_some iter && Idx.equal_static_symbol sym two.extent));
  p "its bound lowering emits the gh-490 extent guard"
    (count_ifs (Asgns.to_low_level ~static_indices:bound2 two.fwd.Asgns.asgns) > 0);
  let ctx2 = Train.init_params (Context.auto ()) two.bindings two.y in
  let ctx2, routine2 = Train.to_routine ctx2 two.bindings two.fwd in
  let ctx2 = Context.set_values ctx2 two.x.Tensor.value [| 37.; 37. |] in
  Idx.find_exn routine2.Context.bindings two.extent := 0;
  let ctx2 = Context.run ctx2 routine2 in
  let guarded = Context.get_values ctx2 two.y.Tensor.value in
  p "the guarded reduction of 37.0 over an extent-0 axis is the empty sum 0.0"
    (Array.equal Float.equal guarded [| 0. |]);

  (* {1 Reachability: the same graph at maximum ONE has no iterator to guard.} *)
  let one = sum_over_symbolic_axis ~range:1 in
  let bound1 = Idx.bound_symbols one.bindings in
  p "an ordinary %op reduction reaches Accum_op with a maximum-one extent and no iterator"
    (List.exists one.projections.Idx.extent_syms ~f:(fun (iter, sym) ->
         Option.is_none iter && Idx.equal_static_symbol sym one.extent));
  (* Over the extent symbols the projection carries: the product space is what their iterators span,
     so an empty one is only a claim where there is an extent to have no iterator. *)
  p_empty "its product space is empty, so no loop exists for an extent guard to sit in"
    ~over:one.projections.Idx.extent_syms
    (Array.to_list one.projections.Idx.components);
  p_all "both sides of the reduction are indexed at element zero"
    (Array.to_list one.projections.Idx.project_lhs
    @ List.concat_map (Array.to_list one.projections.Idx.project_rhs) ~f:Array.to_list)
    ~f:(Idx.equal_axis_index (Idx.Fixed_idx 0));
  p "zero is a legal runtime binding for a maximum-one extent"
    (Result.is_ok (Result.try_with (fun () -> Idx.validate_bound_value one.extent 0)));

  (* {1 The boundary: bound lowering of that assignment is refused.} *)
  let refusal =
    rejection (fun () -> ignore (Asgns.to_low_level ~static_indices:bound1 one.accum : LL.t))
  in
  Option.iter refusal ~f:(fun msg ->
      Stdlib.prerr_endline ("gh-878 refusal (not part of the golden): " ^ msg));
  p "Accum_op lowering refuses a bound maximum-one symbolic extent" (Option.is_some refusal);
  p "the refusal names Accum_op, the absent iterator and the element-zero write"
    (says refusal "Accum_op" && says refusal "no iterator" && says refusal "element zero");
  p "the user-facing compile path refuses it too"
    (Option.is_some
       (rejection (fun () ->
            let ctx = Train.init_params (Context.auto ()) one.bindings one.y in
            ignore (Train.to_routine ctx one.bindings one.fwd : Context.t * Context.routine))));
  p "the same extent left unbound still lowers, retaining maximum-shape semantics"
    (Option.is_none (rejection (fun () -> ignore (Asgns.to_low_level one.accum : LL.t))));

  (* {1 What the boundary prevents, executed.} *)
  let legacy = Asgns.to_low_level one.fwd.Asgns.asgns in
  p "that lowering is what the bound path emitted: it contains no guard for static indices to fill"
    (count_ifs legacy = 0);
  Ll_test.materialize one.x.Tensor.value;
  Ll_test.materialize one.y.Tensor.value;
  let optimized =
    Ll_test.optimize
      ~materialized:[ one.x.Tensor.value; one.y.Tensor.value ]
      ~name:"accum_max_one_preboundary" legacy
  in
  (* The extent is bound to zero explicitly. This program has no guard to read it -- that is the
     defect -- so the harness, not the program, is what keeps the value from defaulting: a launch
     parameter left unnamed would run at the backend's [ref 0] whether or not a test meant zero,
     which in a case ABOUT what an extent-0 launch computes is exactly the reading that must not be
     reachable by accident. Both refusals are checked before anything is compiled. *)
  let refused launch =
    Option.is_some
      (misuse (fun () ->
           ignore
             (Ll_test.execute ~bindings:one.bindings ~launch ~name:"accum_max_one_misuse" optimized
                ~seed:[] ~read:[]
               : float array list)))
  in
  p_all ~min:3 "the harness refuses every ~launch that does not value the parameter exactly once"
    [
      ("left unset", []);
      ("valued twice", [ (one.extent, 0); (one.extent, 1) ]);
      ("naming a symbol the bindings do not bind", [ (one.extent, 0); (two.extent, 0) ]);
    ]
    ~f:(fun (_, launch) -> refused launch);
  let got =
    List.hd_exn
      (Ll_test.execute ~bindings:one.bindings
         ~launch:[ (one.extent, 0) ]
         ~name:"accum_max_one_preboundary" optimized
         ~seed:[ (one.x.Tensor.value, [| 37. |]); (one.y.Tensor.value, Ll_test.blank 1) ]
         ~read:[ one.y.Tensor.value ])
  in
  p "the refused program would have returned its operand where the empty sum is 0.0"
    (Array.equal Float.equal got [| 37. |]);
  p "so the refusal guards a wrong VALUE in a concrete output cell, not a beyond-extent margin"
    (Array.equal Int.equal (Lazy.force one.y.Tensor.value.Tn.dims) [| 1 |]
    && not (Array.equal Float.equal got guarded))
