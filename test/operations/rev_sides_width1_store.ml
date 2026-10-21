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
   projections. With [s] concrete, both targets are proven injective and surjective, so neither gets
   an init and both get a plain store. With [s] a bound gh-490 symbolic extent, no projection
   carrying a symbolic extent is proven surjective, so both keep their (guarded) init, but each is
   still proven injective and gets a plain store.

   Each case is executed three ways, all seeded with the same [cat] gradient, whose every in-extent
   cell [1 + 10 s + j] varies with every symbol of the iteration, and with a per-cell negative
   sentinel in both targets (distinct from every result, so a cell the plain store missed stays
   visible). The REFERENCE is an explicit zero [Fetch] of both targets ahead of the NON-initializing
   record (the read-add-write form), lowered and linked with every node materialized and no
   schedule. The initializing record runs the same way (unscheduled), and through the real compile
   path ([Context.compile], the backend's default schedule). Every in-extent target cell of the two
   initializing runs equals the reference's, which equals the target's slice of [cat]'s gradient.

   The last cases, on hand-built records, keep the attribution from overreaching: two width-1
   targets naming ONE storage (one node, or two slice-alias views of one parent) must still
   accumulate into the parent's slice, and a padded width-1 target whose init is elided must still
   get its margin; both are executed. *)

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

(* The statements of raw lowered code that write [tn], by form: an initialization stores the
   accumulation's neutral zero (or zeroes the node), a margin reset stores the padding value (never
   zero in this file), a read-add-write reads the cell it writes, and a plain store does neither. *)
type writes = { inits : int; margins : int; read_add_writes : int; plain_stores : int }

let writes_of ~tn llc =
  let form : LL.t -> [ `Init | `Margin | `Read_add_write | `Plain_store ] option = function
    | LL.Zero_out t when Tn.equal t tn -> Some `Init
    | LL.Set { tn = t; llsc = LL.Constant v; _ } when Tn.equal t tn ->
        Some (if Float.equal v 0. then `Init else `Margin)
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
    margins = count `Margin;
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

(* {1 Hand-built targets.} The front end cannot emit the records below; each is the gradient record
   of [(a, b) ++^ "s, c; s, d => s, c^d"] with [c = d = 1], flipped to initializing, its targets
   replaced by [targets ~ga ~gb]. *)
let rows = 3

let width1_pair ~targets =
  let%op a = { a = 0.5 } in
  let%op b = { b = 0.5 } in
  let%op cat = (a, b) ++^ "s, c; s, d => s, c^d" [ "s"; "c"; "d" ] in
  Shape.set_dim s rows;
  Shape.set_dim c 1;
  Shape.set_dim d 1;
  let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  let ga = grad_of a and gb = grad_of b and gcat = grad_of cat in
  match
    find_rev_sides (Option.value_exn ~here:[%here] cat.Tensor.diff).Tensor.backprop.Asgns.asgns
  with
  | Some (Asgns.Accum_op ({ rhs = Asgns.Rev_sides { op; _ }; _ } as r)) ->
      let rhs =
        Asgns.Rev_sides { op; lhses = Array.map (targets ~ga ~gb) ~f:(fun t -> Asgns.Node t) }
      in
      ( ga,
        gcat,
        Asgns.Accum_op { r with rhs },
        Asgns.Accum_op { r with rhs; initialize_neutral = true } )
  | _ -> failwith "expected the concatenation's Rev_sides gradient"

let next_id = ref 31000

let node ?(padding = None) dims label =
  Int.incr next_id;
  Tn.create (Tn.Specified Ir.Ops.single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy padding)
    ()

(* {2 Two width-1 targets naming one node.} [a]'s gradient twice, the targets the front end emits
   for [(w, w) ++^ ...] (executed end to end in self_concat). Each target alone is injective and
   surjective, but both write every cell: plain stores would keep only the later column, so a node
   named by more than one target keeps the read-add-write form and its init. *)
let shared_case () =
  let label = "shared" in
  let ga, gcat, non_initializing, initializing =
    width1_pair ~targets:(fun ~ga ~gb:_ -> [| ga; ga |])
  in
  let wr = writes_of ~tn:ga (Asgns.to_low_level initializing) in
  p
    (label ^ ": the initializing scatter accumulates into the shared node, with no plain store")
    (wr.read_add_writes > 0 && wr.plain_stores = 0);
  p (label ^ ": the shared node keeps an init pass") (wr.inits > 0);
  List.iter [ ga; gcat ] ~f:Ll_test.materialize;
  let materialized = [ ga; gcat ] in
  let link ~name asgns =
    Ll_test.link ~name (Ll_test.optimize ~materialized ~name (Asgns.to_low_level asgns))
  in
  let zero = Asgns.Fetch { array = ga; fetch_op = Asgns.Constant 0.; dims = ga.Tn.dims } in
  let run linked =
    let ctx =
      Ll_test.run_linked linked
        ~seed:
          [
            (gcat, Array.init (rows * 2) ~f:(fun i -> in_extent_cell (i / 2) (i % 2)));
            (ga, prior rows);
          ]
    in
    Context.get_values ctx ga
  in
  let reference = run (link ~name:"shared_reference" (Asgns.Seq (zero, non_initializing))) in
  pf_all2 "%s: the reference sums both columns of cat's gradient" label reference
    (Array.init rows ~f:(fun s -> in_extent_cell s 0 +. in_extent_cell s 1))
    ~f:Float.equal;
  pf_all2 "%s: the initializing scatter equals the reference" label
    (run (link ~name:"shared_unscheduled" initializing))
    reference ~f:Float.equal

(* [cat]'s gradient for the hand-built records: [1 + 10 s + j], two columns. *)
let gcat_pair = Array.init (rows * 2) ~f:(fun i -> in_extent_cell (i / 2) (i % 2))

(* {2 Two slice-alias views of one parent.} Distinct nodes whose accesses [resolve_alias] redirects
   to the same parent cells (batch index 1 of 2): sharing is judged on the storage, so both still
   accumulate, and their init reaches the parent's slice through the same resolution. Executed:
   slice 1 holds the sum of both columns, and slice 0 keeps its sentinel. *)
let alias_case () =
  let label = "alias" in
  let parent = node [| 2; rows; 1 |] "alias_parent" in
  let batch_idx, bindings = Idx.get_static_symbol ~static_range:2 Idx.Empty in
  let static_indices = Idx.bound_symbols bindings in
  let views =
    Array.init 2 ~f:(fun k ->
        let view = node [| rows; 1 |] ("alias_view" ^ Int.to_string k) in
        Tn.set_alias_of view ~parent ~batch_idx;
        view)
  in
  let _, gcat, _, initializing = width1_pair ~targets:(fun ~ga:_ ~gb:_ -> views) in
  let llc = Asgns.to_low_level ~static_indices initializing in
  let wr = writes_of ~tn:parent llc in
  p
    (label ^ ": the initializing scatter accumulates into the shared parent, with no plain store")
    (wr.read_add_writes > 0 && wr.plain_stores = 0);
  p_none (label ^ ": no statement of the scatter touches an alias view itself")
    (Array.to_list views) ~f:(fun view -> LL.code_touches_tn view llc);
  List.iter [ parent; gcat ] ~f:Ll_test.materialize;
  let name = "alias_unscheduled" in
  let ctx, routine =
    Ll_test.link ~bindings ~name
      (Ll_test.optimize ~materialized:[ parent; gcat ] ~static_indices ~name llc)
  in
  Idx.find_exn routine.Context.bindings batch_idx := 1;
  let ctx =
    Ll_test.run_linked (ctx, routine) ~seed:[ (gcat, gcat_pair); (parent, prior (2 * rows)) ]
  in
  pf_all2 "%s: slice 0 keeps its sentinel, slice 1 sums both columns" label
    (Context.get_values ctx parent)
    (Array.append
       (Array.sub (prior (2 * rows)) ~pos:0 ~len:rows)
       (Array.init rows ~f:(fun s -> in_extent_cell s 0 +. in_extent_cell s 1)))
    ~f:Float.equal

(* {2 A padded width-1 target.} [b]'s gradient replaced by a node with a one-row top margin holding
   7. Its semantic cells are written exactly once, so its init is elided; the plain stores write
   only those cells, so the scatter still establishes the margin (device buffers are allocated raw),
   as the init it replaces would have. Executed on a physical buffer seeded with sentinels in every
   cell, margin included: the margin holds 7 and the semantic cells [cat]'s second column. *)
let padded_case () =
  let label = "padded" in
  let padded =
    node
      ~padding:(Some ([| Ir.Ops.{ left = 1; right = 0 }; Ir.Ops.{ left = 0; right = 0 } |], 7.))
      [| rows; 1 |] "padded_target"
  in
  let ga, gcat, _, initializing = width1_pair ~targets:(fun ~ga ~gb:_ -> [| ga; padded |]) in
  let llc = Asgns.to_low_level initializing in
  let wr = writes_of ~tn:padded llc in
  p
    (label ^ ": the padded width-1 target gets a plain store and no init")
    (wr.plain_stores > 0 && wr.read_add_writes = 0 && wr.inits = 0);
  p (label ^ ": its margin is still reset to the padding value") (wr.margins > 0);
  List.iter [ ga; padded; gcat ] ~f:Ll_test.materialize;
  let name = "padded_unscheduled" in
  let ctx, routine =
    Ll_test.link ~name (Ll_test.optimize ~materialized:[ ga; padded; gcat ] ~name llc)
  in
  let physical_dims = [| rows + 1; 1 |] in
  let sentinels =
    Ir.Ndarray.create_array ~debug:name Ir.Ops.single ~dims:physical_dims ~padding:None
  in
  for i = 0 to rows do
    Ir.Ndarray.set_from_float sentinels [| i; 0 |] (-100. -. Float.of_int i)
  done;
  let ctx = Context.from_host ctx padded sentinels in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed:[ (gcat, gcat_pair); (ga, prior rows) ] in
  pf_all2 "%s: the physical buffer holds the margin value, then cat's second column" label
    (Ir.Ndarray.retrieve_flat_values (Context.to_host ctx padded))
    (Array.append [| 7. |] (Array.init rows ~f:(fun s -> in_extent_cell s 1)))
    ~f:Float.equal

let () =
  Tensor.unsafe_reinitialize ();
  case ~label:"concrete" ~rows:3 ~symbolic:false;
  case ~label:"symbolic" ~rows:4 ~symbolic:true;
  shared_case ();
  alias_case ();
  padded_case ()
