(* gh-ocannl-929: the gradient half of the gh-ocannl-928 contract ([docs/shape_inference.md],
   "Symbolic extents: cells past a bound extent"). Every earlier gh-490 test ran forward code only,
   so nothing pinned that a gradient through a bound symbolic extent is guarded the way the forward
   pass is, and the [Rev_sides] half of gh-ocannl-878's refusal had no executed leg.

   Backprop accumulations are NON-initializing ([=+]), so the contract says they leave the cells
   past the extent untouched; the [zero_grads] fills that precede them are [Fetch]es, which cover
   the materialized buffer whatever the extent. Three legs, all through the real compile path
   ([Train.to_routine], the backend's default schedule) except where a refusal has to be shown:

   - one [Train.grad_update] plus [Train.sgd_update] step at a bound extent: in-extent loss,
   gradient and update exact; the gradient past the extent is the [zero_grads] fill; the parameter
   past the extent keeps its poison, untouched by the update; - gradient accumulation across
   micro-steps ([~accum_steps], which is where untouched gradient cells matter: the parameter
   gradients are not re-zeroed between micro-steps), first with a sentinel past the extent that must
   survive, then across two DIFFERENT extents, where each row must hold exactly the micro-steps it
   was in extent for; - a concatenation along a non-symbolic axis, whose gradient is the [Rev_sides]
   scatter: executed at a maximum-two extent (guarded, untouched past the extent), and refused at a
   maximum-one extent in both roles, with the pre-boundary [Rev_sides] lowering executed to show the
   write the refusal prevents.

   Values are integers or halves of integers throughout, so every comparison is exact on every
   backend; the in-extent cells are [1 + 10 s + a] and the cells past the extent a poison three
   orders of magnitude larger, so a read past the extent shows as a jump. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let rec find_accum ~f = function
  | Asgns.Accum_op { rhs; _ } as a when f rhs -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_accum ~f a) (find_accum ~f b)
  | Asgns.Block_comment (_, body) -> find_accum ~f body
  | Asgns.Accum_op _ | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

let rejection f = match f () with () -> None | exception Utils.User_error msg -> Some msg

let says rejection substring =
  Option.value_map rejection ~default:false ~f:(String.is_substring ~substring)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* A [rows x width] array, row-major, each row given by [row s]. *)
let rows ~rows:n ~width row = Array.concat (List.init n ~f:(fun s -> Array.init width ~f:(row s)))
let cell s a = Float.of_int (1 + (10 * s) + a)
let poison s a = Float.of_int (1000 + (10 * s) + a)
let sentinel s a = -1. -. Float.of_int ((10 * s) + a)

(* [range x width] cells: [cell] for rows in extent [e], [past] for the rest. *)
let seeded ~range ~width ~e ~past =
  rows ~rows:range ~width (fun s a -> if s < e then cell s a else past s a)

let sub_rows ~width v ~from ~upto = Array.sub v ~pos:(from * width) ~len:((upto - from) * width)
let range = 4
let width = 3

type graph = {
  extent : Idx.static_symbol;
  bindings : Idx.unit_bindings;
  x : Tensor.t;
  loss : Tensor.t;
}

(* [loss = sum (x *. x)] over a parameter [x : s a], [s] a symbolic extent of maximum [range]: [d
   loss / d x = 2 x], cell by cell. *)
let squares () =
  Tensor.unsafe_reinitialize ();
  let extent, bindings = Idx.get_static_symbol ~static_range:range Idx.Empty in
  let%op x = { x = 0.5 } in
  let%op xs = x ++ "s a => s a" [ "s"; "a" ] in
  Shape.set_sym_dim s extent;
  Shape.set_dim a width;
  let%op loss = (xs *. xs) ++ "... => 0" in
  { extent; bindings; x; loss }

let () =
  (* {1 One training step at a bound extent.} *)
  let g = squares () in
  let e = 2 in
  let step =
    Asgns.sequence
      [ Train.grad_update g.loss; Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.25) g.loss ]
  in
  let ctx = Train.init_params (Context.auto ()) g.bindings g.loss in
  let ctx, routine = Train.to_routine ctx g.bindings step in
  Idx.find_exn routine.Context.bindings g.extent := e;
  let x0 = seeded ~range ~width ~e ~past:poison in
  let gx = grad_of g.x in
  let ctx = Context.set_values ctx g.x.Tensor.value x0 in
  let ctx = Context.set_values ctx gx (rows ~rows:range ~width sentinel) in
  let ctx = Context.run ctx routine in
  let x1 = Context.get_values ctx g.x.Tensor.value and dx = Context.get_values ctx gx in
  let in_extent v = sub_rows ~width v ~from:0 ~upto:e
  and past v = sub_rows ~width v ~from:e ~upto:range in
  p "the loss sums the squares of exactly the in-extent cells"
    (Float.equal
       (Context.get_values ctx g.loss.Tensor.value).(0)
       (Array.sum (module Float) (in_extent x0) ~f:(fun v -> v *. v)));
  p_all2 "the in-extent gradient is 2x" (in_extent dx)
    (Array.map (in_extent x0) ~f:(fun v -> 2. *. v))
    ~f:Float.equal;
  p_all "past the extent the gradient holds the zero_grads fill, not the backprop's"
    (Array.to_list (past dx))
    ~f:(Float.equal 0.);
  p_all2 "the update moves the in-extent parameter cells by learning rate times gradient"
    (in_extent x1)
    (Array.map (in_extent x0) ~f:(fun v -> v -. (0.25 *. 2. *. v)))
    ~f:Float.equal;
  p_all2 "the update leaves the parameter cells past the extent untouched" (past x1) (past x0)
    ~f:Float.equal;

  (* {1 Gradient accumulation: the parameter gradient is not re-zeroed between micro-steps.} *)
  let g = squares () in
  let gx = grad_of g.x in
  let micro = Train.grad_update ~accum_steps:2 g.loss in
  let zero = Train.zero_params_grads g.loss in
  let ctx = Train.init_params (Context.auto ()) g.bindings g.loss in
  let ctx, zero_r = Train.to_routine ctx g.bindings zero in
  let ctx, micro_r = Train.to_routine ctx g.bindings micro in
  (* Every row in extent at the largest extent, so each micro-step's contribution is [0.5 * 2x]. *)
  let x_all = seeded ~range ~width ~e:range ~past:poison in
  let ctx = Context.set_values ctx g.x.Tensor.value x_all in
  let micro_step ctx e =
    Idx.find_exn micro_r.Context.bindings g.extent := e;
    Context.run ctx micro_r
  in
  (* The zeroing routine takes the extent too; the planted gradient below overwrites what it wrote,
     so it runs at the micro-steps' extent for want of any other. *)
  Idx.find_exn zero_r.Context.bindings g.extent := e;
  let ctx = Context.run ctx zero_r in
  let planted = rows ~rows:range ~width (fun s a -> if s < e then 0. else sentinel s a) in
  let ctx = Context.set_values ctx gx planted in
  let ctx = micro_step ctx e in
  let dx = Context.get_values ctx gx in
  p_all2 "a micro-step at extent 2 accumulates x into each in-extent gradient cell"
    (sub_rows ~width dx ~from:0 ~upto:e)
    (sub_rows ~width x_all ~from:0 ~upto:e)
    ~f:Float.equal;
  p_all2 "and leaves every gradient cell past the extent exactly as it found it"
    (sub_rows ~width dx ~from:e ~upto:range)
    (sub_rows ~width planted ~from:e ~upto:range)
    ~f:Float.equal;
  let ctx = Context.run ctx zero_r in
  let ctx = micro_step ctx e in
  let ctx = micro_step ctx range in
  let dx = Context.get_values ctx gx in
  p_all2 "across micro-steps at extents 2 then 4, each row holds the steps it was in extent for" dx
    (rows ~rows:range ~width (fun s a -> if s < e then 2. *. cell s a else cell s a))
    ~f:Float.equal;

  (* {1 The [Rev_sides] scatter: a concatenation along a non-symbolic axis.} *)
  let concat ~range =
    Tensor.unsafe_reinitialize ();
    let extent, bindings = Idx.get_static_symbol ~static_range:range Idx.Empty in
    let%op u = { u = 0.5 } in
    let%op v = { v = 0.5 } in
    let%op cat = (u, v) ++^ "s, a; s, b => s, a^b" [ "s"; "a"; "b" ] in
    Shape.set_sym_dim s extent;
    Shape.set_dim a 2;
    Shape.set_dim b 1;
    let%op loss = (cat *. cat) ++ "... => 0" in
    (extent, bindings, u, v, cat, loss)
  in
  let is_block = function Asgns.Block _ -> true | _ -> false in
  let is_rev_sides = function Asgns.Rev_sides _ -> true | _ -> false in
  (* Maximum two, bound to one: the guarded scatter, executed. *)
  let extent, bindings, u, v, _cat, loss = concat ~range:2 in
  let micro = Train.grad_update ~accum_steps:1 loss in
  p "the concatenation's gradient lowers through a Rev_sides scatter"
    (Option.is_some (find_accum ~f:is_rev_sides micro.Asgns.asgns));
  let ctx = Train.init_params (Context.auto ()) bindings loss in
  let ctx, zero_r = Train.to_routine ctx bindings (Train.zero_params_grads loss) in
  let ctx, micro_r = Train.to_routine ctx bindings micro in
  let e = 1 in
  Idx.find_exn micro_r.Context.bindings extent := e;
  let u0 = seeded ~range:2 ~width:2 ~e ~past:poison
  and v0 = seeded ~range:2 ~width:1 ~e ~past:poison in
  let ctx = Context.set_values ctx u.Tensor.value u0 in
  let ctx = Context.set_values ctx v.Tensor.value v0 in
  Idx.find_exn zero_r.Context.bindings extent := e;
  let ctx = Context.run ctx zero_r in
  let planted width = rows ~rows:2 ~width (fun s a -> if s < e then 0. else sentinel s a) in
  let ctx = Context.set_values ctx (grad_of u) (planted 2) in
  let ctx = Context.set_values ctx (grad_of v) (planted 1) in
  let ctx = Context.run ctx micro_r in
  List.iter
    [ ("u", u, u0, 2); ("v", v, v0, 1) ]
    ~f:(fun (name, t, t0, width) ->
      let d = Context.get_values ctx (grad_of t) in
      pf_all2 "Rev_sides at extent 1 of 2: %s's in-extent gradient is 2%s" name name
        (sub_rows ~width d ~from:0 ~upto:e)
        (Array.map (sub_rows ~width t0 ~from:0 ~upto:e) ~f:(fun x -> 2. *. x))
        ~f:Float.equal;
      pf_all2 "Rev_sides at extent 1 of 2: %s's gradient past the extent is untouched" name
        (sub_rows ~width d ~from:e ~upto:2)
        (sub_rows ~width (planted width) ~from:e ~upto:2)
        ~f:Float.equal);

  (* Maximum one: no iterator to guard, refused in BOTH roles (gh-ocannl-878). *)
  let extent, bindings, u, v, cat, loss = concat ~range:1 in
  let static_indices = Idx.bound_symbols bindings in
  let micro = Train.grad_update ~accum_steps:1 loss in
  let block = Option.value_exn ~here:[%here] (find_accum ~f:is_block micro.Asgns.asgns) in
  let rev_sides = Option.value_exn ~here:[%here] (find_accum ~f:is_rev_sides micro.Asgns.asgns) in
  let refused accum =
    rejection (fun () -> ignore (Asgns.to_low_level ~static_indices accum : LL.t))
  in
  let block_refusal = refused block and rev_refusal = refused rev_sides in
  Option.iter rev_refusal ~f:(fun msg ->
      Stdio.prerr_endline ("gh-929 Rev_sides refusal (not part of the golden): " ^ msg));
  p "the forward concatenation over a maximum-one extent is refused in its Block role"
    (says block_refusal "Block lowering" && says block_refusal "no iterator");
  p "its gradient scatter is refused in its Rev_sides role, for the same reason"
    (says rev_refusal "Rev_sides lowering" && says rev_refusal "no iterator");
  p "the user-facing compile of the training step refuses it"
    (Option.is_some
       (rejection (fun () ->
            let ctx = Train.init_params (Context.auto ()) bindings loss in
            ignore (Train.to_routine ctx bindings micro : Context.t * Context.routine))));
  (* What the Rev_sides refusal prevents, executed: the extent left unbound is gh-490's
     maximum-shape lowering, which has no guard, launched with the extent bound to ZERO -- so no
     gradient cell is in extent, and every one must stay untouched. *)
  let gu = grad_of u and gcat = grad_of cat in
  let gv = grad_of v in
  List.iter [ gu; gv; gcat ] ~f:Ll_test.materialize;
  let legacy = Asgns.to_low_level rev_sides in
  let optimized =
    Ll_test.optimize ~materialized:[ gu; gv; gcat ] ~name:"sym_backward_rev_preboundary" legacy
  in
  let got =
    List.hd_exn
      (Ll_test.execute ~bindings
         ~launch:[ (extent, 0) ]
         ~name:"sym_backward_rev_preboundary" optimized
         ~seed:[ (gcat, [| 5.; 7.; 11. |]); (gu, Ll_test.blank 2); (gv, Ll_test.blank 1) ]
         ~read:[ gu ])
  in
  p "the refused scatter would have written u's gradient at element zero of an extent-0 axis"
    (not (Array.equal Float.equal got (Ll_test.blank 2)))
