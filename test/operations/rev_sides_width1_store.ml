(* The [Rev_sides] scatter (a [++^] concatenation's gradient) attributes a width-1 concatenation
   segment's target to that segment, so an initializing scatter writes it with a plain store, the
   form its wider sibling already gets.

   A dimension-1 segment iterates its symbol over [0] alone, so the target's projection indexes that
   axis at [Fixed_idx 0] and never mentions the symbol. [loop_accum_rev]'s selector used to drop
   such a target: with no attributed view, [Affine.is_injective] was never asked, and the target got
   the read-add-write form plus an initialization pass even where every cell is written exactly
   once.

   The concatenation is [(u, v) ++^ "s, b; s, c => s, b^c"] with [b = 2] and [c = 1], so [v] is the
   width-1 operand. The front end emits its gradient scatter only NON-initializing (a backprop
   [=+]), so the initializing flavour is that record with [initialize_neutral] set: same
   projections. Two cases: - [s] concrete: both targets are proven injective and surjective, so
   neither gets an init and both get a plain store; - [s] a bound gh-490 symbolic extent: no
   projection carrying a symbolic extent is proven surjective, so both keep their (guarded) init,
   but each is still proven injective and gets a plain store.

   Each case is executed three ways, all seeded with the same [cat] gradient, whose every in-extent
   cell [1 + 10 s + j] varies with every symbol of the iteration, and with a per-cell negative
   sentinel in both targets (distinct from every result, so a cell the plain store missed stays
   visible): - the REFERENCE: an explicit zero [Fetch] of both targets ahead of the NON-initializing
   record (the read-add-write form), lowered and linked with every node materialized and no
   schedule; - the initializing record, the same way (unscheduled); - the initializing record
   through the real compile path ([Context.compile], the backend's default schedule). Every
   in-extent target cell of the two initializing runs equals the reference's, which equals the
   target's slice of [cat]'s gradient. *)

open Base
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let b_width = 2
let cat_width = 3

let rec find_rev_sides = function
  | Asgns.Accum_op { rhs = Asgns.Rev_sides _; _ } as a -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_rev_sides a) (find_rev_sides b)
  | Asgns.Block_comment (_, body) -> find_rev_sides body
  | Asgns.Accum_op _ | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

(* The statements of raw lowered code that write [tn], by form: an initialization stores a constant
   (or zeroes the node), a read-add-write reads the cell it writes, and a plain store does
   neither. *)
type writes = { inits : int; read_add_writes : int; plain_stores : int }

let writes_of ~tn llc =
  let form : LL.t -> [ `Init | `Read_add_write | `Plain_store ] option = function
    | LL.Zero_out t when Tn.equal t tn -> Some `Init
    | LL.Set { tn = t; llsc = LL.Constant _; _ } when Tn.equal t tn -> Some `Init
    | LL.Set { tn = t; llsc; _ } when Tn.equal t tn ->
        Some (if LL.scalar_touches_tn tn llsc then `Read_add_write else `Plain_store)
    | _ -> None
  in
  let count kind =
    Ll_test.count_stmt llc ~f:(fun stmt ->
        Option.value_map (form stmt) ~default:false ~f:(fun k -> Poly.equal k kind))
  in
  {
    inits = count `Init;
    read_add_writes = count `Read_add_write;
    plain_stores = count `Plain_store;
  }

let in_extent_cell s j = Float.of_int (1 + (10 * s) + j)
let poison s j = Float.of_int (1000 + (10 * s) + j)
let prior n = Array.init n ~f:(fun i -> -1. -. Float.of_int i)

let case ~label ~rows ~symbolic =
  (* [rows] is [s]'s dimension, the declared maximum for a symbolic extent. *)
  let extent_bindings =
    if symbolic then Some (Idx.get_static_symbol ~static_range:rows Idx.Empty) else None
  in
  let bindings = Option.value_map extent_bindings ~default:Idx.Empty ~f:snd in
  let static_indices = Idx.bound_symbols bindings in
  let%op u = { u = 0.5 } in
  let%op v = { v = 0.5 } in
  let%op cat = (u, v) ++^ "s, b; s, c => s, b^c" [ "s"; "b"; "c" ] in
  (match extent_bindings with
  | Some (extent, _) -> Shape.set_sym_dim s extent
  | None -> Shape.set_dim s rows);
  Shape.set_dim b b_width;
  Shape.set_dim c 1;
  let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  let gu = grad_of u and gv = grad_of v and gcat = grad_of cat in
  let non_initializing, initializing, projections =
    match
      find_rev_sides (Option.value_exn ~here:[%here] cat.Tensor.diff).Tensor.backprop.Asgns.asgns
    with
    | Some (Asgns.Accum_op r as a) ->
        p
          (label ^ ": the concatenation's gradient scatter is emitted non-initializing")
          (not r.initialize_neutral);
        (a, Asgns.Accum_op { r with initialize_neutral = true }, Lazy.force r.projections)
    | _ -> failwith "expected the concatenation's Rev_sides gradient"
  in
  (* The premise: [v]'s projection indexes the concatenation axis at [Fixed_idx 0], mentioning no
     symbol of the concatenation's component. *)
  let concat_syms =
    Array.find_map projections.project_lhs ~f:(function Idx.Concat syms -> Some syms | _ -> None)
    |> Option.value_exn ~here:[%here]
  in
  let v_proj = projections.project_rhs.(1) in
  p_exists (label ^ ": v's projection indexes its width-1 concatenation axis at Fixed_idx 0")
    (Array.to_list v_proj) ~f:(function
    | Idx.Fixed_idx 0 -> true
    | _ -> false);
  p_none (label ^ ": v's projection mentions no symbol of the concatenation's component")
    (Array.to_list v_proj) ~f:(fun idx ->
      List.exists concat_syms ~f:(fun s -> Idx.axis_index_mentions_symbol s idx));
  (* {1 Structural: the form each target is written in.} *)
  let llc = Asgns.to_low_level ~static_indices initializing in
  List.iter
    [ ("u", gu); ("v", gv) ]
    ~f:(fun (name, g) ->
      let w = writes_of ~tn:g llc in
      p
        (Printf.sprintf
           "%s: the initializing scatter writes %s with a plain store, never reading it" label name)
        (w.plain_stores > 0 && w.read_add_writes = 0);
      if symbolic then
        p
          (Printf.sprintf "%s: %s keeps an init pass (a symbolic extent is never proven surjective)"
             label name)
          (w.inits > 0)
      else p (Printf.sprintf "%s: %s gets no init pass" label name) (w.inits = 0));
  (* {1 Executed: both initializing runs equal the materialized, unscheduled reference.} *)
  List.iter [ gu; gv; gcat ] ~f:Ll_test.materialize;
  let materialized = [ gu; gv; gcat ] in
  let link ~name asgns =
    let llc = Asgns.to_low_level ~static_indices asgns in
    Ll_test.link ~bindings ~name (Ll_test.optimize ~materialized ~static_indices ~name llc)
  in
  let zero tn = Asgns.Fetch { array = tn; fetch_op = Asgns.Constant 0.; dims = tn.Tn.dims } in
  let reference =
    link ~name:(label ^ "_reference") (Asgns.Seq (zero gu, Asgns.Seq (zero gv, non_initializing)))
  in
  let unscheduled = link ~name:(label ^ "_unscheduled") initializing in
  (* The real compile path links against a context already holding the three buffers, as a training
     step's context does; [run] reseeds them before every launch. *)
  let scheduled =
    let ctx =
      List.fold materialized ~init:(Context.auto ()) ~f:(fun ctx tn ->
          Context.set_values ctx tn (Array.create ~len:(Tn.num_elems tn) 0.))
    in
    Context.compile ~name:(label ^ "_scheduled") ctx (Asgns.to_comp initializing) bindings
  in
  let gcat_cells e =
    Array.init (rows * cat_width) ~f:(fun i ->
        let s = i / cat_width and j = i % cat_width in
        if s < e then in_extent_cell s j else poison s j)
  in
  let seed = [ (gu, prior (rows * b_width)); (gv, prior rows) ] in
  let run (ctx, routine) ~e =
    Option.iter extent_bindings ~f:(fun (extent, _) ->
        Idx.find_exn routine.Context.bindings extent := e);
    let ctx = Ll_test.run_linked (ctx, routine) ~seed:((gcat, gcat_cells e) :: seed) in
    (Context.get_values ctx gu, Context.get_values ctx gv)
  in
  let extents = if symbolic then [ 0; 2; rows ] else [ rows ] in
  List.iter extents ~f:(fun e ->
      let ref_u, ref_v = run reference ~e in
      List.iter
        [ ("u", ref_u, b_width, 0); ("v", ref_v, 1, b_width) ]
        ~f:(fun (name, got, g_width, offset) ->
          let len = e * g_width in
          if e > 0 then
            pf_all2 "%s at extent %d: the reference %s gradient is its slice of cat's" label e name
              (Array.sub got ~pos:0 ~len)
              (Array.init len ~f:(fun i -> in_extent_cell (i / g_width) (offset + (i % g_width))))
              ~f:Float.equal);
      List.iter
        [ ("unscheduled", unscheduled); ("scheduled", scheduled) ]
        ~f:(fun (how, linked) ->
          let got_u, got_v = run linked ~e in
          List.iter
            [ ("u", got_u, ref_u, b_width); ("v", got_v, ref_v, 1) ]
            ~f:(fun (name, got, reference, g_width) ->
              let len = e * g_width in
              if e > 0 then
                pf_all2 "%s at extent %d, %s initializing scatter: %s equals the reference" label e
                  how name (Array.sub got ~pos:0 ~len) (Array.sub reference ~pos:0 ~len)
                  ~f:Float.equal)))

let () =
  Tensor.unsafe_reinitialize ();
  case ~label:"concrete" ~rows:3 ~symbolic:false;
  case ~label:"symbolic" ~rows:4 ~symbolic:true
