(* gh-ocannl-1003: the single-pass block fold of the online-softmax attention rewrite
   ([Ir.Online_softmax] under [online_softmax_block = B > 0]).

   The two-pass rewrite (test/operations/online_softmax.ml) turns the softmax normalizer into a
   per-row scan and hoists the probability read of the value contraction, which reads the scores a
   second time. The fold makes it ONE pass: per query row, a scan over the key blocks of [B]
   carrying the row's running max and sum, whose body computes the block's scores into a [B] tile by
   the score contraction and its scale/mask chain, rescales a [d_v] numerator tile and accumulates
   [probabilities * v] into it; the last key block writes the output.

   Every executed leg compares the fold against the SAME model composed and in the two-pass form
   (the session is reinitialized before each build, so the builds mint the same tensor ids and draw
   the same initializations). Device floats stay off the golden: the claims are two-sided tolerance
   comparisons, and the digits go to stderr.

   Legs: 1. parity over block sizes and sequence lengths -- dividing, with tails, a single key
   block, blocks of one -- with the structural pins (one scan per attention, over the key blocks,
   one carried pair; no [seq, seq] node written; the tiles never a routine argument); 2. the head
   width above the recompute cap: still no [seq, seq] buffer, the scores being read once; 3. the
   special values of the masked scores -- masked prefixes, fully masked rows (NaN, as composed),
   finite and lowest-finite fills, NaN and positive-infinity fills; 4. two stacked blocks; 5.
   rectangular attention on leaf tensors, queries and keys of different lengths and unequal head
   widths; 6. the member contract: idempotence, the analysis cache across sibling lowerings, the two
   facts the in-place tiles rest on (never virtual, by a structural rule; the scan body out of the
   schedule ops' reach), and the declines (an active dropout's shape, a reader of the row state the
   fold cannot move, scores that are no contraction), each falling back to the two-pass form; 7.
   training: the fold forward under the composed backward (treatment E of the gh-1002/1003 record)
   and under the fused one (treatment F). *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Online_softmax = Ir.Online_softmax

let fast_math = Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math"
let backend_name = Utils.get_global_arg ~arg_name:"backend" ~default:"cc"
let batch = 2
let d_model = 16
let heads = 2

(* Over (query [s], key [t]): causal, with the first [prefix] keys masked for every query at or past
   [prefix], and the first [dead] query rows masked entirely. *)
let mask ~seq ~prefix ~dead =
  NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
    ~f:(function
      | [| s; t |] -> if s >= dead && s >= t && (t >= prefix || s < prefix) then 1. else 0.
      | _ -> assert false)
    ()

(* [layers] attention blocks over a ramp scaled into [0, 1) (a non-saturated softmax), residually
   stacked. *)
let model ?mask_fill ?(dead = 0) ~seq ~layers ~d_k ~prefix () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let scale = Float.of_int (batch * seq * d_model) in
  let%op x = x /. !.scale in
  let mask = mask ~seq ~prefix ~dead in
  let blocks =
    List.init layers ~f:(fun i ->
        Nn_blocks.multi_head_attention
          ~label:[ "attn" ^ Int.to_string i ]
          ~num_heads:heads ~d_k ~d_v:d_k ?mask_fill ())
  in
  List.fold blocks ~init:x ~f:(fun x block ->
      let%op y = x + block ~train_step:None ~mask x in
      y)

let inspect ?(name = "probe") (asgns : Ir.Assignments.t) : LL.optimized =
  Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
    ~cd_source:None ~name [] asgns

let rec scan_list (llc : LL.t) =
  match llc with
  | LL.Scan_loop { to_; carried; body; _ } -> (to_, List.length carried) :: scan_list body
  | LL.Seq (a, b) -> scan_list a @ scan_list b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> scan_list body
  | _ -> []

let scans_of llc = List.length (scan_list llc)

(* The minted tiles the fold writes: nodes labelled [block_...] with a buffer. *)
let is_tile (tn : Tn.t) =
  match tn.Tn.label with l :: _ -> String.is_prefix l ~prefix:"block_" | [] -> false

let written (llc : LL.t) =
  LL.affine_accesses llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)
  |> Set.of_list (module Tn)

let tiles llc = Set.filter (written llc) ~f:is_tile

(* The written nodes with two axes of extent [extent]: the attention's [seq, seq] intermediates (a
   tile has the block's extents, which a block covering the whole sequence makes [seq, seq] too). *)
let square_buffers ~extent llc =
  Set.filter (written llc) ~f:(fun tn ->
      (not (is_tile tn)) && Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = extent) >= 2)

type run = {
  values : float array;
  optimized : LL.optimized;
  interface : Set.M(Tn).t;  (** The forward routine's arguments. *)
}

let with_gates ~on ~block f =
  Online_softmax.set_enabled (Some on);
  Online_softmax.set_block (Some block);
  Exn.protect ~f ~finally:(fun () ->
      Online_softmax.set_enabled None;
      Online_softmax.set_block None;
      Online_softmax.set_backward_enabled None)

let forward ?mask_fill ?dead ?(layers = 1) ?(d_k = 8) ?(prefix = 0) ~on ~block ~seq () =
  Tensor.unsafe_reinitialize ();
  with_gates ~on ~block (fun () ->
      let t = model ?mask_fill ?dead ~seq ~layers ~d_k ~prefix () in
      Train.set_materialized t.Tensor.value;
      let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty t in
      let captured = ref None in
      let ctx, routine =
        Context.compile ~name:"osb_forward"
          ~lowered_transform:(fun o ->
            captured := Some o;
            [ o ])
          ctx t.Tensor.forward Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      let values = Context.get_values ctx t.Tensor.value in
      {
        values;
        optimized = Option.value_exn !captured;
        interface = Set.union routine.Context.inputs routine.Context.outputs;
      })

let close ~tol g w = Float.(abs (g -. w) <= tol *. max 1. (abs w))

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (Array.to_list (Array.map (Array.sub got ~pos:0 ~len:6) ~f:(Printf.sprintf "%.9g"))))

let ceil_div a b = (a + b - 1) / b

(* Values equal within [tol] where both are numbers, and NaN in the same cells. *)
let agree ~tol a b =
  Bool.equal (Float.is_nan a) (Float.is_nan b) && (Float.is_nan a || close ~tol a b)

let () =
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  printf "--- run under cc_backend_fast_math=%b ---\n" fast_math;
  printf "--- leg 1: parity over block sizes and sequence lengths ---\n";
  List.iter
    [ (24, [ 1; 8; 16; 32; 5 ]); (7, [ 3; 8 ]); (20, [ 8 ]) ]
    ~f:(fun (seq, blocks) ->
      let composed = forward ~on:false ~block:0 ~seq () in
      let two_pass = forward ~on:true ~block:0 ~seq () in
      report (Printf.sprintf "seq %d composed" seq) composed.values;
      p
        (Printf.sprintf "seq %d: the two-pass form has one scan and no tile" seq)
        (scans_of two_pass.optimized.LL.llc = 1 && Set.is_empty (tiles two_pass.optimized.LL.llc));
      List.iter blocks ~f:(fun block ->
          let fold = forward ~on:true ~block ~seq () in
          let what = Printf.sprintf "seq %d, block %d" seq block in
          report what fold.values;
          let b = Int.min block seq in
          p
            (Printf.sprintf "%s: one scan, over the %d key blocks, carrying one pair" what
               (ceil_div seq b))
            (Poly.equal (scan_list fold.optimized.LL.llc) [ (ceil_div seq b - 1, 2) ]);
          p (what ^ ": the fold writes its two tiles") (Set.length (tiles fold.optimized.LL.llc) = 2);
          p_none (what ^ ": no tile is a routine argument") (Set.to_list fold.interface) ~f:is_tile;
          p
            (what ^ ": no seq^2-sized node is written")
            (Set.is_empty (square_buffers ~extent:seq fold.optimized.LL.llc));
          p_all2
            (what ^ ": matches the composed output within 1e-5 relative")
            fold.values composed.values ~f:(close ~tol:1e-5);
          p_all2
            (what ^ ": matches the two-pass output within 1e-5 relative")
            fold.values two_pass.values ~f:(close ~tol:1e-5)));

  printf "--- leg 2: head width above the recompute cap ---\n";
  let seq = 24 and d_k = 32 in
  let composed = forward ~on:false ~block:0 ~seq ~d_k () in
  let two_pass = forward ~on:true ~block:0 ~seq ~d_k () in
  let fold = forward ~on:true ~block:8 ~seq ~d_k () in
  printf "seq^2 buffers: composed %d, two-pass %d, fold %d\n"
    (Set.length (square_buffers ~extent:seq composed.optimized.LL.llc))
    (Set.length (square_buffers ~extent:seq two_pass.optimized.LL.llc))
    (Set.length (square_buffers ~extent:seq fold.optimized.LL.llc));
  p "the two-pass form stores the scores, read twice"
    (Set.length (square_buffers ~extent:seq two_pass.optimized.LL.llc) = 1);
  p "the fold reads the scores once, from its tile: no seq^2 buffer"
    (Set.is_empty (square_buffers ~extent:seq fold.optimized.LL.llc));
  p_all2 "the fold matches the composed output within 1e-5 relative" fold.values composed.values
    ~f:(close ~tol:1e-5);

  printf "--- leg 3: special values of the masked scores, against the composed form ---\n";
  let lowest = -3.4028234663852886e38 in
  List.iter
    [
      ("masked key prefixes", None, 3, 0, `Finite);
      ("fully masked rows at -inf", None, 0, 2, `Nan_rows);
      ("fully masked rows at a finite fill", Some (-1e4), 0, 2, `Finite);
      ("fully masked rows at the lowest finite fill", Some lowest, 2, 2, `Finite);
      ("a NaN fill", Some Float.nan, 3, 0, `Nan_rows);
      ("a positive-infinity fill", Some Float.infinity, 3, 0, `Nan_rows);
    ]
    ~f:(fun (what, mask_fill, prefix, dead, expect) ->
      List.iter [ 8; 5 ] ~f:(fun block ->
          let seq = 20 in
          let composed = forward ?mask_fill ~dead ~prefix ~on:false ~block:0 ~seq () in
          let fold = forward ?mask_fill ~dead ~prefix ~on:true ~block ~seq () in
          let what = Printf.sprintf "%s, block %d" what block in
          report (what ^ " composed") composed.values;
          report (what ^ " fold") fold.values;
          (match expect with
          | `Finite ->
              p_all
                (what ^ ": every composed output is finite")
                (Array.to_list composed.values) ~f:Float.is_finite
          | `Nan_rows ->
              p
                (what ^ ": the composed output has NaN rows")
                (Array.exists composed.values ~f:Float.is_nan));
          p_all2
            (what
           ^ ": the fold is NaN exactly where the composed form is, and within 1e-5 elsewhere")
            fold.values composed.values ~f:(agree ~tol:1e-5)));

  printf "--- leg 4: two stacked blocks ---\n";
  let seq = 20 in
  let composed = forward ~layers:2 ~on:false ~block:0 ~seq () in
  let fold = forward ~layers:2 ~on:true ~block:8 ~seq () in
  p "two scans" (scans_of fold.optimized.LL.llc = 2);
  p "no seq^2-sized node is written"
    (Set.is_empty (square_buffers ~extent:seq fold.optimized.LL.llc));
  p_all2 "matches the composed output within 1e-5 relative" fold.values composed.values
    ~f:(close ~tol:1e-5)

(* --- Leg 5: rectangular attention on leaf tensors. --- *)

let wave salt idcs =
  Float.sin
    (Array.foldi idcs ~init:salt ~f:(fun i acc x ->
         acc +. (Float.of_int ((i + 2) * (x + 1)) *. 0.37)))

let rect ~on ~block ~sq ~sk =
  Tensor.unsafe_reinitialize ();
  with_gates ~on ~block (fun () ->
      let leaf name salt ~b ~o = NTDSL.init ~l:name ~prec:Ir.Ops.single ~b ~o ~f:(wave salt) () in
      let q = leaf "q" 0.1 ~b:[ sq ] ~o:[ 2; 3 ] in
      let k = leaf "k" 0.2 ~b:[ sk ] ~o:[ 2; 3 ] in
      let v = leaf "v" 0.3 ~b:[ sk ] ~o:[ 2; 5 ] in
      let mask =
        NTDSL.init ~l:"rect_mask" ~prec:Ir.Ops.single ~b:[ sq ] ~i:[ sk ] ~o:[]
          ~f:(function [| s; t |] -> if (s + t) % 4 <> 3 then 1. else 0. | _ -> assert false)
          ()
      in
      let%op scores =
        (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
      in
      let%op masked = where mask scores !.Float.neg_infinity in
      let weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () masked in
      let%op o = weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ] in
      Train.set_materialized o.Tensor.value;
      let ctx = Train.forward_once (Context.auto ()) o in
      let values = Context.get_values ctx o.Tensor.value in
      (values, inspect o.Tensor.forward.Ir.Assignments.asgns))

let () =
  printf "--- leg 5: rectangular attention, queries and keys of different lengths ---\n";
  List.iter
    [ (9, 13, 4); (13, 9, 4); (9, 13, 16); (6, 11, 1) ]
    ~f:(fun (sq, sk, block) ->
      let composed, _ = rect ~on:false ~block:0 ~sq ~sk in
      let fold, optimized = rect ~on:true ~block ~sq ~sk in
      let what = Printf.sprintf "%d queries, %d keys, block %d" sq sk block in
      report what fold;
      let b = Int.min block sk in
      p
        (Printf.sprintf "%s: one scan over the %d key blocks" what (ceil_div sk b))
        (Poly.equal (scan_list optimized.LL.llc) [ (ceil_div sk b - 1, 2) ]);
      p_all2
        (what ^ ": matches the composed output within 1e-5 relative")
        fold composed ~f:(close ~tol:1e-5))

(* --- Leg 6: the member contract, and what the fold declines. --- *)

(* [stmts] with [extra] inserted ahead of the first statement satisfying [before]. *)
let insert_before (llc : LL.t) ~before extra =
  let stmts = LL.flat_lines [ llc ] in
  let pos = Option.value_exn (List.findi stmts ~f:(fun _ s -> before s)) |> fst in
  LL.unflat_lines (List.take stmts pos @ extra @ List.drop stmts pos)

let label_is name (tn : Tn.t) = String.equal (Tn.debug_name tn) name

(* Whether [stmt] reads the node named [name] (the value pass reads the probabilities). *)
let reads_node name stmt =
  List.exists (LL.affine_accesses stmt) ~f:(fun (a : Tn.t Ir.Affine.access) ->
      (not a.a_write) && label_is name a.a_tn)

let rec map_gets ~f (s : LL.scalar_t) : LL.scalar_t =
  match s with
  | LL.Get (tn, idcs) -> f tn idcs
  | LL.Unop (op, (a, p)) -> LL.Unop (op, (map_gets ~f a, p))
  | LL.Binop (op, (a, p), (b, q)) -> LL.Binop (op, (map_gets ~f a, p), (map_gets ~f b, q))
  | LL.Ternop (op, (a, p), (b, q), (c, r)) ->
      LL.Ternop (op, (map_gets ~f a, p), (map_gets ~f b, q), (map_gets ~f c, r))
  | other -> other

let rec map_sets ~f (llc : LL.t) : LL.t =
  match llc with
  | LL.Seq (a, b) -> LL.Seq (map_sets ~f a, map_sets ~f b)
  | LL.For_loop fl -> LL.For_loop { fl with body = map_sets ~f fl.body }
  | LL.Set s -> LL.Set { s with llsc = map_gets ~f s.llsc }
  | other -> other

(* A value operand aliasing a node of the score chain, on a model whose sequence length equals its
   head width, so that the masked scores have the values' shape. *)
module Value_alias = struct
  let check ~rewrite =
    Tensor.unsafe_reinitialize ();
    let raw =
      with_gates ~on:true ~block:4 (fun () ->
          let t = model ~seq:8 ~layers:1 ~d_k:8 ~prefix:0 () in
          Ir.Assignments.to_low_level t.Tensor.forward.Ir.Assignments.asgns)
    in
    let where =
      Option.value_exn (Set.find (written raw) ~f:(label_is "where")) ~message:"masked scores"
    in
    let aliased =
      LL.unflat_lines
        (List.map (LL.flat_lines [ raw ]) ~f:(fun stmt ->
             if reads_node "softmax" stmt then
               map_sets stmt ~f:(fun tn i ->
                   if label_is "v" tn then LL.Get (where, i) else LL.Get (tn, i))
             else stmt))
    in
    p "a value operand aliasing the masked scores: no fold"
      (Set.is_empty (tiles (rewrite ~block:4 aliased)));
    (* The same with a materialized exponential: a live row-state definition the fold would move
       behind itself, and so define only after reading it. *)
    let e = Option.value_exn (Set.find (written raw) ~f:(label_is "exp_exp_vals")) in
    Ll_test.materialize e;
    let aliased_live =
      LL.unflat_lines
        (List.map (LL.flat_lines [ raw ]) ~f:(fun stmt ->
             if reads_node "softmax" stmt then
               map_sets stmt ~f:(fun tn i ->
                   if label_is "v" tn then LL.Get (e, i) else LL.Get (tn, i))
             else stmt))
    in
    p "a value operand aliasing a live row-state definition (the exponentials): no fold"
      (Set.is_empty (tiles (rewrite ~block:4 aliased_live)));
    (* A live definition reading a scope local: the census does not see local writes, so the fold
       will not move it (the probabilities materialized, their definition wrapped in a scope). *)
    let probs = Option.value_exn (Set.find (written raw) ~f:(label_is "softmax")) in
    Ll_test.materialize probs;
    let scoped =
      LL.unflat_lines
        (List.map (LL.flat_lines [ raw ]) ~f:(fun stmt ->
             let rec wrap (llc : LL.t) : LL.t =
               match llc with
               | LL.For_loop fl -> LL.For_loop { fl with body = wrap fl.body }
               | LL.Set ({ tn; llsc; _ } as st) when Tn.equal tn probs ->
                   let id = LL.get_scope probs in
                   LL.Set
                     {
                       st with
                       llsc =
                         LL.Local_scope
                           {
                             id;
                             orig_indices = st.idcs;
                             mint = LL.Inlined_computation;
                             body = LL.Set_local (id, llsc);
                           };
                     }
               | other -> other
             in
             wrap stmt))
    in
    p "a live definition reading a scope local: no fold"
      (Set.is_empty (tiles (rewrite ~block:4 scoped)))
end

let () =
  printf "--- leg 6: the member contract, and the declines ---\n";
  let module B = Ll_test in
  let seq = 20 in
  Tensor.unsafe_reinitialize ();
  let t, raw =
    with_gates ~on:true ~block:8 (fun () ->
        let t = model ~seq ~layers:1 ~d_k:8 ~prefix:0 () in
        Train.set_materialized t.Tensor.value;
        let _ctx = Train.forward_once (Context.auto ()) t in
        (t, Ir.Assignments.to_low_level t.Tensor.forward.Ir.Assignments.asgns))
  in
  let rewrite ?(block = 8) llc =
    with_gates ~on:true ~block (fun () -> Online_softmax.rewrite llc)
  in
  let once = rewrite raw in
  p "the pass changes the raw lowering" (not (LL.equal once raw));
  p "the pass is idempotent on its own output" (LL.equal (rewrite once) once);
  p "block 0 is the two-pass form: one scan, no tile"
    (let two = rewrite ~block:0 raw in
     scans_of two = 1 && Set.is_empty (tiles two));
  with_gates ~on:true ~block:8 (fun () ->
      ignore (inspect t.Tensor.forward.Ir.Assignments.asgns : LL.optimized);
      let h1, m1 = LL.analysis_cache_stats () in
      ignore (inspect t.Tensor.forward.Ir.Assignments.asgns : LL.optimized);
      let h2, m2 = LL.analysis_cache_stats () in
      p "re-lowering the folded forward is an analysis-cache hit, not a miss"
        (h2 = h1 + 1 && m2 = m1));
  (* The two facts the fold's in-place tiles rest on, pinned on the optimized forward: a tile is
     never a virtualization candidate, so it is never inlined across the scan's iterations, and a
     schedule op cannot reach a loop inside the scan body, so nothing reorders the rescale against
     the accumulation. The tiles request no memory mode: the refusal is the optimizer's own, and by
     a structural rule, not a policy cap. Under the default caps a cap is recorded first (the
     recompute cap on the value update's key loop, then the visit cap), and a refused node is never
     examined again; with the caps lifted, the tile's first setter -- its whole-tile init ahead of
     the scan -- is refused as a write the enclosing row loops repeat ([147]), and every setter in
     the scan body would be by the scan's rule ([148]). *)
  let under_defaults =
    with_gates ~on:true ~block:8 (fun () ->
        inspect ~name:"osb_sound_default" t.Tensor.forward.Ir.Assignments.asgns)
  in
  p_all "under the default caps, each tile is non-virtual"
    (Set.to_list (tiles under_defaults.LL.llc))
    ~f:(B.known_non_virtual under_defaults);
  let vs = LL.virtualize_settings in
  let caps = (vs.LL.max_inline_reduction, vs.LL.max_visits, vs.LL.max_inline_fanin) in
  vs.LL.max_inline_reduction <- -1;
  vs.LL.max_visits <- 1_000_000;
  vs.LL.max_inline_fanin <- 1_000_000;
  let o =
    Exn.protect
      ~f:(fun () ->
        with_gates ~on:true ~block:8 (fun () ->
            inspect ~name:"osb_sound" t.Tensor.forward.Ir.Assignments.asgns))
      ~finally:(fun () ->
        let r, v, f = caps in
        vs.LL.max_inline_reduction <- r;
        vs.LL.max_visits <- v;
        vs.LL.max_inline_fanin <- f)
  in
  let tile_nodes = Set.to_list (tiles o.LL.llc) in
  List.iter tile_nodes ~f:(fun tn ->
      eprintf "%s: %s (not part of the golden)\n%!" (Tn.debug_name tn)
        (Option.value_map (B.rejection_code o tn) ~default:"undecided" ~f:Tn.provenance_to_string));
  p "the forward writes the two tiles" (List.length tile_nodes = 2);
  p_all "with the caps lifted, each tile is refused by a structural rule (147 or 148)" tile_nodes
    ~f:(fun tn ->
      Option.exists (B.rejection_code o tn) ~f:(fun prov ->
          let code = Tn.provenance_to_string prov in
          String.is_prefix code ~prefix:"147:" || String.is_prefix code ~prefix:"148:"));
  let rec loop_in_scan = function
    | LL.Scan_loop { body; _ } -> first_loop body
    | LL.Seq (a, b) -> Option.first_some (loop_in_scan a) (loop_in_scan b)
    | LL.For_loop { body; _ } | LL.If { body; _ } -> loop_in_scan body
    | _ -> None
  and first_loop = function
    | LL.For_loop { index; _ } -> Some index
    | LL.Seq (a, b) -> Option.first_some (first_loop a) (first_loop b)
    | LL.If { body; _ } -> first_loop body
    | _ -> None
  in
  let inner = Option.value_exn (loop_in_scan o.LL.llc) ~message:"a loop inside the fold's scan" in
  p "a schedule op naming a loop inside the fold's scan body declines: the body is out of reach"
    (try
       ignore
         (Ir.Schedule.apply [ Ir.Schedule.Unroll { axis = inner; materialize = false } ] o
           : LL.optimized);
       false
     with Invalid_argument msg -> String.is_substring msg ~substring:"no For_loop with index");
  (* The value pass reading the probabilities through one more elementwise step -- an active
     dropout's shape: the fold declines, and the two-pass form (whose hoist follows elementwise
     chains) takes the normalizer. *)
  let dropped =
    let softmax =
      Option.value_exn
        (Set.find (written raw) ~f:(label_is "softmax"))
        ~message:"the probabilities node"
    in
    let dims = Lazy.force softmax.Tn.dims in
    let d = B.node_factory ~first_id:100300 ~dims () "dropped" in
    let syms = Array.map dims ~f:(fun _ -> B.sym ()) in
    let idcs = Array.map syms ~f:B.iter in
    let copy =
      Array.fold_right (Array.zip_exn syms dims)
        ~init:(B.set d idcs (B.mul (B.get softmax idcs) (B.c 1.)))
        ~f:(fun (s, n) body -> B.loop_n s n body)
    in
    let redirect stmt =
      if reads_node "softmax" stmt then
        map_sets stmt ~f:(fun tn i -> if Tn.equal tn softmax then LL.Get (d, i) else LL.Get (tn, i))
      else stmt
    in
    let with_copy = insert_before raw ~before:(reads_node "softmax") [ copy ] in
    LL.unflat_lines
      (List.map (LL.flat_lines [ with_copy ]) ~f:(fun s ->
           if LL.equal s copy then s else redirect s))
  in
  let r = rewrite dropped in
  p "a value pass reading the probabilities through an elementwise step: no fold, the two-pass form"
    (scans_of r = 1 && Set.is_empty (tiles r));
  (* A reader of the row max between the normalizer and the value pass that is no elementwise
     definition the fold could move behind it. *)
  let probe = B.node_factory ~first_id:100400 ~dims:[| 1 |] () "probe" in
  let side_read =
    let max_vals = Option.value_exn (Set.find (written raw) ~f:(label_is "max_vals")) in
    insert_before raw
      ~before:(fun s -> match s with LL.Zero_out tn -> label_is "v" tn | _ -> false)
      [
        B.set_at probe (B.fixed 0)
          (B.add
             (B.get probe [| B.fixed 0 |])
             (B.get max_vals [| B.fixed 0; B.fixed 0; B.fixed 0; B.fixed 0 |]));
      ]
  in
  let r = rewrite side_read in
  p "an accumulation reading the row max before the value pass: no fold, the two-pass form"
    (scans_of r = 1 && Set.is_empty (tiles r));
  (* Scores that are an input, no contraction: nothing to compute a tile from. *)
  let n = 6 and w = 3 in
  let mk = B.node_factory ~first_id:100500 in
  let x = mk ~dims:[| n |] () "x" and m = mk ~dims:[| 1 |] () "m" in
  let nn = mk ~dims:[| n |] () "n" and e = mk ~dims:[| n |] () "e" in
  let l = mk ~dims:[| 1 |] () "l" and pr = mk ~dims:[| n |] () "p" in
  let vals = mk ~dims:[| n; w |] () "vals" and out = mk ~dims:[| w |] () "out" in
  List.iter [ x; vals; out ] ~f:B.materialize;
  let op o args = LL.apply_op o args in
  let cell tn = B.get tn [| B.fixed 0 |] in
  let over f =
    let t = B.sym () in
    B.loop_n t n (f t)
  in
  let plain =
    LL.unflat_lines
      [
        B.set_at m (B.fixed 0) (B.c Float.neg_infinity);
        over (fun t ->
            B.set_at m (B.fixed 0)
              (op (Ir.Ops.Binop Ir.Ops.Max) [| cell m; B.get x [| B.iter t |] |]));
        over (fun t ->
            B.set_at nn (B.iter t)
              (op (Ir.Ops.Binop Ir.Ops.Sub) [| B.get x [| B.iter t |]; cell m |]));
        over (fun t ->
            B.set_at e (B.iter t) (op (Ir.Ops.Unop Ir.Ops.Exp) [| B.get nn [| B.iter t |] |]));
        B.zero l;
        over (fun t ->
            B.set_at l (B.fixed 0)
              (op (Ir.Ops.Binop Ir.Ops.Add) [| cell l; B.get e [| B.iter t |] |]));
        over (fun t ->
            B.set_at pr (B.iter t)
              (op (Ir.Ops.Binop Ir.Ops.Div) [| B.get e [| B.iter t |]; cell l |]));
        B.zero out;
        over (fun t ->
            let j = B.sym () in
            B.loop_n j w
              (B.set out
                 [| B.iter j |]
                 (B.add
                    (B.get out [| B.iter j |])
                    (B.mul (B.get pr [| B.iter t |]) (B.get vals [| B.iter t; B.iter j |])))));
      ]
  in
  let r = rewrite plain in
  p "scores that are an input, not a contraction: no fold, the two-pass form"
    (scans_of r = 1 && Set.is_empty (tiles r) && not (LL.equal r plain));
  (* The value pass writing a slice of its output -- [O] with an extra axis of 2, the pass at its
     index 0: the fold would replace the whole-node zeroing and write the slice only, so it
     declines, and the two-pass form keeps the zeroing. *)
  let o = Option.value_exn (Set.find (written raw) ~f:(label_is "n76")) ~message:"O" in
  let wide =
    B.node_factory ~first_id:100600 ~dims:(Array.append (Lazy.force o.Tn.dims) [| 2 |]) () "o_wide"
  in
  B.materialize wide;
  let slot idcs = Array.append idcs [| B.fixed 0 |] in
  let rec widen (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (a, b) -> LL.Seq (widen a, widen b)
    | LL.For_loop fl -> LL.For_loop { fl with body = widen fl.body }
    | LL.Zero_out tn when Tn.equal tn o -> LL.Zero_out wide
    | LL.Set s ->
        let get tn i = if Tn.equal tn o then LL.Get (wide, slot i) else LL.Get (tn, i) in
        let llsc = map_gets ~f:get s.llsc in
        if Tn.equal s.tn o then LL.Set { s with tn = wide; idcs = slot s.idcs; llsc }
        else LL.Set { s with llsc }
    | other -> other
  in
  let r = rewrite (widen raw) in
  p "a value pass into a slice of its output: no fold, the two-pass form keeps the zeroing"
    (scans_of r = 1
    && Set.is_empty (tiles r)
    && Ll_test.count_stmt r ~f:(function LL.Zero_out tn -> Tn.equal tn wide | _ -> false) = 1);
  (* The value pass ahead of the probabilities' definition, reading those an earlier call left (read
     before write): not the pass over this call's normalizer, so no fold. *)
  let reordered =
    let stmts = LL.flat_lines [ raw ] in
    let is_pv s = reads_node "softmax" s in
    let is_zero_o = function LL.Zero_out tn -> label_is "n76" tn | _ -> false in
    let is_p_def s =
      List.exists (LL.affine_accesses s) ~f:(fun (a : Tn.t Ir.Affine.access) ->
          a.a_write && label_is "softmax" a.a_tn)
    in
    let pv, _ = Option.value_exn (List.findi stmts ~f:(fun _ s -> is_pv s)) in
    let prefix = List.take stmts (pv + 1) and suffix = List.drop stmts (pv + 1) in
    let keep s = not (is_zero_o s || is_pv s || is_p_def s) in
    LL.unflat_lines
      (List.filter prefix ~f:keep @ List.filter prefix ~f:is_zero_o @ List.filter prefix ~f:is_pv
     @ List.filter prefix ~f:is_p_def @ suffix)
  in
  p "a value pass ahead of the probabilities it would consume: no fold"
    (Set.is_empty (tiles (rewrite reordered)));
  (* A value operand aliasing a node of the score chain -- at seq 8 = head width 8 the masked scores
     sign like the values: the fold would remove that node's definition and still read it, so it
     declines. *)
  Value_alias.check ~rewrite:(fun ~block llc -> rewrite ~block llc);
  (* Tiles past the stack threshold would be [On_device] buffers shared by every parallel row: the
     fold declines, and the output is the composed one (executed). *)
  let threshold_key = "stack_threshold_in_bytes" in
  Hashtbl.set Utils.config_file_args ~key:threshold_key ~data:"16";
  let small =
    Exn.protect
      ~f:(fun () -> forward ~on:true ~block:8 ~seq ~d_k:8 ())
      ~finally:(fun () -> Hashtbl.remove Utils.config_file_args threshold_key)
  in
  let reference = forward ~on:false ~block:0 ~seq ~d_k:8 () in
  p "tiles past the stack threshold: no fold" (Set.is_empty (tiles small.optimized.LL.llc));
  p_all2 "tiles past the stack threshold: the output matches the composed within 1e-5 relative"
    small.values reference.values ~f:(close ~tol:1e-5);
  (* Probabilities stored at a precision other than the scores': the composed value pass reads them
     rounded to their node, which the fold never stores, so it declines. *)
  let probs = Option.value_exn (Set.find (written raw) ~f:(label_is "softmax")) in
  let narrow =
    B.node_factory ~prec:Ir.Ops.half ~first_id:100700 ~dims:(Lazy.force probs.Tn.dims) ()
      "softmax_f16"
  in
  let rec renarrow (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (a, b) -> LL.Seq (renarrow a, renarrow b)
    | LL.For_loop fl -> LL.For_loop { fl with body = renarrow fl.body }
    | LL.Set st ->
        let llsc =
          map_gets st.llsc ~f:(fun tn i ->
              if Tn.equal tn probs then LL.Get (narrow, i) else LL.Get (tn, i))
        in
        if Tn.equal st.tn probs then LL.Set { st with tn = narrow; llsc }
        else LL.Set { st with llsc }
    | other -> other
  in
  p "probabilities stored narrower than the scores: no fold"
    (Set.is_empty (tiles (rewrite (renarrow raw))));
  (* Only the row max requested: the fold writes it, and nothing else of the row state -- the
     shifted scores, exponentials and probabilities have no reader, so they go, and with them the
     score chain; only a live definition would move behind the fold. *)
  let max_vals = Option.value_exn (Set.find (written raw) ~f:(label_is "max_vals")) in
  B.materialize max_vals;
  let r = rewrite raw in
  let w = written r in
  p "the row max alone requested: the fold writes it and its tiles"
    (Set.mem w max_vals && not (Set.is_empty (tiles r)));
  p_none "the row max alone requested: no other row-state or score-chain node is written"
    [ "softmax"; "exp_exp_vals"; "where"; "n67" ] ~f:(fun name -> Set.exists w ~f:(label_is name))

(* --- Leg 7: training -- the fold forward under the composed backward and under the fused one.
   --- *)

let rowdot_writes (llc : LL.t) =
  Ll_test.count_stmt llc ~f:(function
    | LL.Set { tn; _ } -> ( match tn.Tn.label with "bwd_rowdot" :: _ -> true | _ -> false)
    | _ -> false)

let training ?mask_fill ?(prefix = 2) ~seq ~d_k ~on ~block ~bwd () =
  Tensor.unsafe_reinitialize ();
  with_gates ~on ~block (fun () ->
      Online_softmax.set_backward_enabled (Some bwd);
      let y = model ?mask_fill ~seq ~layers:1 ~d_k ~prefix () in
      let%op loss = (y *. y) ++ "... | ... => 0" in
      let params =
        Set.to_list y.Tensor.params
        |> List.sort ~compare:(fun a b -> Int.compare a.Tensor.value.Tn.id b.Tensor.value.Tn.id)
      in
      List.iter params ~f:(fun p ->
          Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
      let update = Train.grad_update loss in
      let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
      let captured = ref None in
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun o ->
            captured := Some o;
            [ o ])
          ctx update Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      let grads =
        List.map params ~f:(fun p ->
            ( Tn.debug_name p.Tensor.value,
              Context.get_values ctx (Option.value_exn p.Tensor.diff).Tensor.grad ))
      in
      let loss = Array.fold (Context.get_values ctx loss.Tensor.value) ~init:0. ~f:( +. ) in
      let rewritten =
        Online_softmax.rewrite (Ir.Assignments.to_low_level update.Ir.Assignments.asgns)
      in
      (loss, grads, Option.value_exn !captured, rewritten))

let () =
  printf "--- leg 7: training -- the fold forward under the composed and the fused backward ---\n";
  let seq = 20 and d_k = 8 in
  let loss_c, grads_c, _, _ = training ~seq ~d_k ~on:false ~block:0 ~bwd:false () in
  List.iter
    [ ("treatment E (composed backward)", false); ("treatment F (fused backward)", true) ]
    ~f:(fun (what, bwd) ->
      let loss, grads, optimized, rewritten = training ~seq ~d_k ~on:true ~block:8 ~bwd () in
      eprintf "%s loss: composed %.9g fold %.9g (not part of the golden)\n%!" what loss_c loss;
      p (what ^ ": the step holds the fold") (Set.length (tiles optimized.LL.llc) = 2);
      (* The row-state definitions the backward reads move behind the fold: one writer each, never a
         copy beside the original. *)
      let writers_of name =
        Ll_test.count_stmt rewritten ~f:(function
          | LL.Set { tn; _ } -> label_is name tn
          | _ -> false)
      in
      p
        (what ^ ": the moved probabilities and exponentials have one writer each")
        (writers_of "softmax" = 1 && writers_of "exp_exp_vals" = 1);
      p
        (what ^ if bwd then ": and the fused backward" else ": and no fused backward")
        (rowdot_writes optimized.LL.llc = if bwd then 1 else 0);
      p (what ^ ": the loss agrees within 1e-5 relative") (close ~tol:1e-5 loss loss_c);
      p
        (what ^ ": the same parameters carry gradients")
        (List.equal String.equal (List.map grads ~f:fst) (List.map grads_c ~f:fst));
      List.iter2_exn grads grads_c ~f:(fun (name, gf) (_, gc) ->
          p_all2
            (Printf.sprintf "%s: %s.grad agrees within 1e-4 relative" what name)
            gf gc ~f:(close ~tol:1e-4);
          p
            (Printf.sprintf "%s: %s.grad is not identically zero" what name)
            (Array.exists gc ~f:(fun v -> Float.(v <> 0.)))));
  (* The NaN fill through the whole step: the gradients poisoned exactly where the composed ones
     are. *)
  let _, grads_c, _, _ =
    training ~mask_fill:Float.nan ~prefix:3 ~seq ~d_k ~on:false ~block:0 ~bwd:false ()
  in
  let _, grads_f, _, _ =
    training ~mask_fill:Float.nan ~prefix:3 ~seq ~d_k ~on:true ~block:8 ~bwd:true ()
  in
  List.iter2_exn grads_f grads_c ~f:(fun (name, gf) (_, gc) ->
      p_all2
        ("a NaN fill: " ^ name ^ ".grad is NaN exactly where the composed one is")
        gf gc
        ~f:(fun f c -> Bool.equal (Float.is_nan f) (Float.is_nan c)))

(* --- Leg 8: automatic block size is a device fact before every lowering. --- *)
let () =
  printf "--- leg 8: auto resolves before lowering and preserves explicit sizes ---\n";
  let key = "online_softmax_block" in
  let previous = Hashtbl.find Utils.config_file_args key in
  Hashtbl.set Utils.config_file_args ~key ~data:"auto";
  Online_softmax.set_enabled (Some true);
  Online_softmax.set_backward_enabled (Some false);
  Online_softmax.set_block None;
  Exn.protect
    ~finally:(fun () ->
      (match previous with
      | None -> Hashtbl.remove Utils.config_file_args key
      | Some value -> Hashtbl.set Utils.config_file_args ~key ~data:value);
      Online_softmax.set_enabled None;
      Online_softmax.set_backward_enabled None;
      Online_softmax.set_block None)
    ~f:(fun () ->
      p "auto keeps block 16 where the device says the fold is profitable"
        (Online_softmax.block ~profitable:true () = 16);
      p "auto keeps the two-pass rewrite on an unmeasured device" (Online_softmax.block () = 0);
      Tensor.unsafe_reinitialize ();
      let t = model ~seq:32 ~layers:1 ~d_k:8 ~prefix:0 () in
      Train.set_materialized t.Tensor.value;
      let comp = Train.forward t in
      let lower profitable =
        Ir.Assignments.lower
          ~rewrite_target:{ online_softmax_block_profitable = profitable }
          (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None ~cd_source:None
          ~name:"osb_auto_probe" [] comp.Ir.Assignments.asgns
      in
      let off = lower false and on = lower true in
      p "auto on a declining device lowers to the two-pass form"
        (scans_of off.LL.llc = 1 && Set.is_empty (tiles off.LL.llc));
      p "auto on a profitable device lowers to the block fold"
        (scans_of on.LL.llc = 1 && Set.length (tiles on.LL.llc) = 2);
      let digest opt = Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize opt) in
      p "different resolved programs have different schedule-cache code digests"
        (not (String.equal (digest off) (digest on)));
      List.iter
        [ (0, off); (16, on) ]
        ~f:(fun (size, expected) ->
          Online_softmax.set_block (Some size);
          p
            (Printf.sprintf "explicit block %d bypasses declining device economics" size)
            (String.equal (digest (lower false)) (digest expected));
          p
            (Printf.sprintf "explicit block %d bypasses profitable device economics" size)
            (String.equal (digest (lower true)) (digest expected)));
      Online_softmax.set_block None;
      let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty t in
      let profitable =
        (Context.hardware_limits ctx).Ir.Backend_intf.online_softmax_block_profitable
      in
      p "the backend pins the automatic block policy (HIP off; CUDA/Metal/CPU preserved)"
        (Bool.equal profitable (not (String.equal backend_name "hip")));
      p "backend-free hardware limits conservatively decline the automatic fold"
        (not Ir.Backend_intf.no_hardware_limits.online_softmax_block_profitable);
      let analyzed =
        Context.lowered_for_decisions ~name:"osb_auto_analysis" ctx comp Ir.Indexing.Empty
      in
      (* Linking settles placements after the transform; compare the resolved structure here. *)
      let structural_digest opt =
        Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize ~with_placements:false opt)
      in
      let analyzed_digest = structural_digest analyzed in
      let captured_digest = ref None in
      let captured = ref None in
      let ctx, routine =
        Context.compile ~name:"osb_auto_exec"
          ~lowered_transform:(fun opt ->
            captured := Some opt;
            captured_digest := Some (structural_digest opt);
            [ opt ])
          ctx comp Ir.Indexing.Empty
      in
      let compiled = Option.value_exn !captured in
      p "analyze-only lowering honors the run's device economics"
        (Bool.equal (Set.is_empty (tiles analyzed.LL.llc)) (not profitable));
      p "backend compilation honors the same device economics"
        (Bool.equal (Set.is_empty (tiles compiled.LL.llc)) (not profitable));
      p "analysis and compilation agree on the resolved code structure"
        (String.equal analyzed_digest (Option.value_exn !captured_digest));
      let ctx = Context.run ctx routine in
      let auto_values = Context.get_values ctx t.Tensor.value in
      Online_softmax.set_block (Some (if profitable then 16 else 0));
      let ctx, forced = Context.compile ~name:"osb_auto_forced" ctx comp Ir.Indexing.Empty in
      let ctx = Context.run ctx forced in
      p_all2 "executed auto output matches the explicit resolved mode within 1e-5 relative"
        auto_values
        (Context.get_values ctx t.Tensor.value)
        ~f:(close ~tol:1e-5))
