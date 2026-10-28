(* gh-ocannl-1230: a search's candidate labels tell distinct candidates apart.

   Every search log line, every timing the measurement seam receives and the crowned
   [Autotune.report.best_label] name a candidate by its label, so two candidates that print alike
   cannot be told apart anywhere downstream: which one failed, which one was timed at what, which
   one won. Two gaps made that happen.

   A sketch's label rendered a hand-picked subset of its parameters, and the tuner seeds twins that
   differ in exactly one parameter (the register-tile twins of one CPU tensorized leaf printed
   alike, gh-ocannl-1166). The first part enumerates the fields of [Autotune.sketch_params] from the
   type's derived s-expression -- never from a list here, so a field added to the type is covered
   the day it appears -- and claims that changing any one of them, at two base points that differ in
   every field, changes the label; and that a per-segment candidate's label names its segment, since
   two segments can admit equal parameters.

   Every beam-round candidate printed as [F_saved[N segs]], naming neither the segment it extended
   nor the move (gh-ocannl-1137's probe could not recover which of 635 round candidates failed). The
   second part runs a one-round search on cc with the ranking scripted so a fissioned candidate
   leads the beam, and claims that the round's candidates -- the [W_saved]/[F_saved] attempts, the
   only saved forms a search with no cache attempts -- all print apart, and that each fissioned one
   names a segment of its parent.

   A split-reduce candidate's label printed a site's target name, extents and only the NUMBER of its
   interchange swaps, so two sites on alike-named targets, or with equal-length chains, printed
   alike. Part 1b reads [Autotune.sr_site]'s fields off its derived s-expression and claims that
   changing any one of them changes the label. And the winner a [timing_ctx] search rebuilds in the
   caller's context lost its round move, so its replay -- and a failure it reported -- printed as
   the bare [F_saved[N segs]] its parent's every round candidate shares. Part 3 crowns a round
   candidate and claims its replay is labeled exactly as it was timed.

   Pinned to cc ([Context.cpu]): the scripted ranking is backend-independent, and cc is always
   available. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* --- Part 1: every field of a sketch's parameters reaches its label --- *)

(* The two base points. They differ in every field (claimed below), so each field is changed in both
   directions: from its common value to an unusual one, and back. *)
let lo : Autotune.sketch_params =
  {
    sk_gpu = false;
    sk_mma = false;
    sk_simd = 0;
    sk_bm = 16;
    sk_bn = 16;
    sk_bk = 0;
    sk_tm = 2;
    sk_tn = 2;
    sk_hoist = false;
    sk_grid = false;
    sk_pack_rest = false;
    sk_conv = false;
    sk_epilogue = false;
    sk_batch_grid = false;
    sk_batch_inner = false;
    sk_coalesce = None;
    sk_swizzle = None;
    sk_depth = 1;
    sk_pack_prec = None;
    sk_tile = None;
  }

let hi : Autotune.sketch_params =
  {
    sk_gpu = true;
    sk_mma = true;
    sk_simd = 32;
    sk_bm = 32;
    sk_bn = 64;
    sk_bk = 16;
    sk_tm = 4;
    sk_tn = 8;
    sk_hoist = true;
    sk_grid = true;
    sk_pack_rest = true;
    sk_conv = true;
    sk_epilogue = true;
    sk_batch_grid = true;
    sk_batch_inner = true;
    sk_coalesce = Some Autotune.Coalesce_contraction;
    sk_swizzle = Some Ir.Low_level.Swizzle_b128;
    sk_depth = 2;
    sk_pack_prec = Some Ir.Ops.single;
    sk_tile = Some { Ir.Register_tile.rm = 4; rn = 2; lanes = 4 };
  }

(* The record's fields as the derived s-expression lists them, in declaration order. *)
let fields p =
  match Autotune.sexp_of_sketch_params p with
  | Sexp.List l ->
      List.map l ~f:(function
        | Sexp.List [ Sexp.Atom name; v ] -> (name, v)
        | s -> failwith ("unexpected record field " ^ Sexp.to_string s))
  | s -> failwith ("unexpected record " ^ Sexp.to_string s)

(* [p] with field [name] taken from [donor], rebuilt through the derived reader. *)
let with_field_of ~donor p name =
  let value = List.Assoc.find_exn (fields donor) ~equal:String.equal name in
  Autotune.sketch_params_of_sexp
    (Sexp.List
       (List.map (fields p) ~f:(fun (n, v) ->
            Sexp.List [ Sexp.Atom n; (if String.equal n name then value else v) ])))

let () =
  let names = List.map (fields lo) ~f:fst in
  p_all "the two base points differ in every field" (fields lo) ~f:(fun (name, v) ->
      not (Sexp.equal v (List.Assoc.find_exn (fields hi) ~equal:String.equal name)));
  let label = Autotune.sketch_params_label in
  p_all "changing any one field of the common base point changes its label" names ~f:(fun name ->
      let changed = with_field_of ~donor:hi lo name in
      let differs = not (String.equal (label changed) (label lo)) in
      if not differs then Stdio.eprintf "%s does not reach the label: %s\n" name (label lo);
      differs);
  p_all "changing any one field of the unusual base point changes its label" names ~f:(fun name ->
      let changed = with_field_of ~donor:lo hi name in
      let differs = not (String.equal (label changed) (label hi)) in
      if not differs then Stdio.eprintf "%s does not reach the label: %s\n" name (label hi);
      differs);
  (* The separator of a composite's entries, which the playoff's per-segment comparison splits
     on. *)
  p_none "no label carries a comma"
    (List.concat_map names ~f:(fun name ->
         [
           label lo;
           label hi;
           label (with_field_of ~donor:hi lo name);
           label (with_field_of ~donor:lo hi name);
         ]))
    ~f:(fun l -> String.mem l ',');
  (* The field flips above reach only the two base points' values; a variant field needs each of its
     constructors labelled apart too: a column-coalesced and a contraction-coalesced seed of one
     geometry are different schedules (gh-ocannl-1279). *)
  let layouts = [ None; Some Autotune.Coalesce_columns; Some Autotune.Coalesce_contraction ] in
  let layout_labels = List.map layouts ~f:(fun l -> label { lo with sk_coalesce = l }) in
  p "every coalesced layout labels apart"
    (List.length (List.dedup_and_sort layout_labels ~compare:String.compare) = List.length layouts)

(* Two segments can admit the same parameters (two matmul segments sharing a tile geometry), and
   their one-entry per-segment candidates schedule different kernels: the label names the segment
   too. The keys are the shape the search uses: structural digests of segments, with their first
   units -- and two segments of one structure but different placements share the digest, so their
   keys differ in the units alone (gh-ocannl-1243). *)
let () =
  let digest_a = Stdlib.Digest.(to_hex (string "segment a"))
  and digest_b = Stdlib.Digest.(to_hex (string "segment b")) in
  let keys =
    [
      { Autotune.key_units = [ 0 ]; key_digest = digest_a };
      { key_units = [ 2 ]; key_digest = digest_b };
      { key_units = [ 4 ]; key_digest = digest_a };
      { key_units = [ 0; 4 ]; key_digest = digest_a };
    ]
  in
  p_pairwise_distinct
    "one-entry sketch candidates of distinct segment keys with equal parameters print apart"
    (List.concat_map [ false; true ] ~f:(fun fine ->
         List.map keys ~f:(fun key -> Autotune.fiss_sketch_label ~fine [ (key, lo) ])))
    ~equal:String.equal ~to_string:Fn.id

(* A staffed single (gh-ocannl-1138) is a single of its focus, timed against the running composite's
   choices for the other keys: its label is the bare single's with the staff appended after [" |
   staff "] -- so a reader that splits the focus off still reads the bare label -- and two staffs
   print apart, so the singles of one key swept in two backgrounds never share a label. *)
let () =
  let key_a = { Autotune.key_units = [ 0 ]; key_digest = Stdlib.Digest.(to_hex (string "a")) }
  and key_b = { Autotune.key_units = [ 2 ]; key_digest = Stdlib.Digest.(to_hex (string "b")) } in
  let bare = Autotune.fiss_sketch_label ~fine:false [ (key_a, lo) ] in
  let staffed staff = Autotune.fiss_sketch_label ~staff ~fine:false [ (key_a, lo) ] in
  p "an empty staff prints the bare label" (String.equal (staffed []) bare);
  p "a staffed label is the bare one's focus followed by its staff"
    (String.is_prefix
       (staffed [ (key_b, hi) ])
       ~prefix:(String.chop_suffix_exn bare ~suffix:"]" ^ " | staff "));
  p_pairwise_distinct "staffs of different parameters print apart"
    [ bare; staffed [ (key_b, lo) ]; staffed [ (key_b, hi) ] ]
    ~equal:String.equal ~to_string:Fn.id

(* --- Part 1b: every field of a split-reduce site reaches its label --- *)

(* Two targets of one debug name -- two nodes labeled alike: the label must tell them apart by more
   than the name. *)
let labeled_x () = (TDSL.ndarray [| 1.; 2. |] ~label:[ "x" ] ~output_dims:[ 2 ] ()).Tensor.value
let x1 = labeled_x ()
let x2 = labeled_x ()
let sym i = Ir.Indexing.Symbol i

(* The two base points, differing in every field (claimed below). [sr_swaps] has equal length at
   both, so the chain is claimed to reach the label pair by pair, not by its length. *)
let site_lo : Autotune.sr_site =
  {
    sr_axis = sym 1;
    sr_target = x1;
    sr_red = 64;
    sr_out = 16;
    sr_cost = 1024;
    sr_dynamic = false;
    sr_swaps = [ (sym 2, sym 1) ];
    sr_coalesce = None;
    sr_block_hoist = [];
  }

let site_hi : Autotune.sr_site =
  {
    sr_axis = sym 3;
    sr_target = x2;
    sr_red = 128;
    sr_out = 32;
    sr_cost = 4096;
    sr_dynamic = true;
    sr_swaps = [ (sym 4, sym 3) ];
    sr_coalesce = Some (sym 5, sym 6);
    sr_block_hoist = [ sym 7 ];
  }

(* The fields as the type's derived s-expression lists them, in declaration order. *)
let site_fields s =
  match Autotune.sexp_of_sr_site s with
  | Sexp.List l ->
      List.map l ~f:(function
        | Sexp.List [ Sexp.Atom name; v ] -> (name, v)
        | s -> failwith ("unexpected record field " ^ Sexp.to_string s))
  | s -> failwith ("unexpected record " ^ Sexp.to_string s)

(* One field of [s] taken from [donor]. The type has no reader (a tensor node cannot be rebuilt from
   its s-expression), so the update is written per field here -- and claimed below to name exactly
   the derived field list, so a field added to the type fails this test until it gets an entry. *)
let site_updates : (string * (donor:Autotune.sr_site -> Autotune.sr_site -> Autotune.sr_site)) list
    =
  [
    ("sr_axis", fun ~donor s -> { s with sr_axis = donor.sr_axis });
    ("sr_target", fun ~donor s -> { s with sr_target = donor.sr_target });
    ("sr_red", fun ~donor s -> { s with sr_red = donor.sr_red });
    ("sr_out", fun ~donor s -> { s with sr_out = donor.sr_out });
    ("sr_cost", fun ~donor s -> { s with sr_cost = donor.sr_cost });
    ("sr_dynamic", fun ~donor s -> { s with sr_dynamic = donor.sr_dynamic });
    ("sr_swaps", fun ~donor s -> { s with sr_swaps = donor.sr_swaps });
    ("sr_coalesce", fun ~donor s -> { s with sr_coalesce = donor.sr_coalesce });
    ("sr_block_hoist", fun ~donor s -> { s with sr_block_hoist = donor.sr_block_hoist });
  ]

let () =
  let names = List.map (site_fields site_lo) ~f:fst in
  p "the per-field updates name exactly the site type's fields, in order"
    (List.equal String.equal names (List.map site_updates ~f:fst));
  p_all "the two base sites differ in every field" (site_fields site_lo) ~f:(fun (name, v) ->
      not (Sexp.equal v (List.Assoc.find_exn (site_fields site_hi) ~equal:String.equal name)));
  p "the two base targets share a label name"
    (String.equal (Ir.Tnode.label_name x1) (Ir.Tnode.label_name x2));
  p "the two base chains have equal length"
    (List.length site_lo.sr_swaps = List.length site_hi.sr_swaps);
  let label s = Autotune.split_sites_label [ (s, 32) ] in
  let changes ~which ~base ~donor =
    p_all (Printf.sprintf "changing any one field of the %s base site changes its label" which)
      site_updates ~f:(fun (name, update) ->
        let differs = not (String.equal (label (update ~donor base)) (label base)) in
        if not differs then Stdio.eprintf "%s does not reach the label: %s\n" name (label base);
        differs)
  in
  changes ~which:"first" ~base:site_lo ~donor:site_hi;
  changes ~which:"second" ~base:site_hi ~donor:site_lo;
  p "changing the block count changes the label"
    (not
       (String.equal
          (Autotune.split_sites_label [ (site_lo, 32) ])
          (Autotune.split_sites_label [ (site_lo, 16) ])));
  (* A staffed split single (gh-ocannl-1138) appends its staff after [" | staff "], as a staffed
     sketch single does, so its focus reads as the bare label and two staffs print apart. *)
  let bare = Autotune.split_sites_label [ (site_lo, 32) ] in
  let staffed b = Autotune.split_sites_label ~staff:[ (site_hi, b) ] [ (site_lo, 32) ] in
  p "a staffed split label is the bare one's focus followed by its staff"
    (String.equal (Autotune.split_sites_label ~staff:[] [ (site_lo, 32) ]) bare
    && String.is_prefix (staffed 8) ~prefix:(String.chop_suffix_exn bare ~suffix:"]" ^ " | staff ")
    );
  p_pairwise_distinct "split staffs of different block counts print apart"
    [ bare; staffed 8; staffed 16 ]
    ~equal:String.equal ~to_string:Fn.id;
  p_none "no site label carries a comma"
    (List.concat_map site_updates ~f:(fun (_, update) ->
         [
           label site_lo;
           label site_hi;
           label (update ~donor:site_hi site_lo);
           label (update ~donor:site_lo site_hi);
         ]))
    ~f:(fun l -> String.mem (Option.value_exn (String.chop_prefix l ~prefix:"F_split[")) ',')

(* --- Part 2: round candidates name their segment and move --- *)

let n = 8

(* The fissionable chain of [autotune_fission_sketch]: d = a + b (forced materialized), e = d *.
   d^T; the transposed read keeps the cross-nest edge misaligned, so the chain fissions. *)
let chain () =
  let init ~modulus ~offset ~stride =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus ~offset ~stride)
  in
  let a =
    TDSL.ndarray (init ~modulus:7 ~offset:0. ~stride:0.5) ~label:[ "a" ] ~output_dims:[ n; n ] ()
  in
  let b =
    TDSL.ndarray (init ~modulus:5 ~offset:(-2.) ~stride:1.) ~label:[ "b" ] ~output_dims:[ n; n ] ()
  in
  let%op d = a + b in
  Train.set_materialized d.Tensor.value;
  let%op e = d *. (d ++ "ij=>ji") in
  Train.forward e

let is_saved label =
  List.exists [ "W_saved["; "F_saved["; "F_split_saved[" ] ~f:(fun prefix ->
      String.is_prefix label ~prefix)

(* The segment a fissioned round label names, and its parent's segment count: [F_saved[<segs> segs
   from <tag>]+seg<i>:<move>]. *)
let segment_of label =
  let ( let* ) = Option.( >>= ) in
  let* rest = String.chop_prefix label ~prefix:"F_saved[" in
  let* segs, rest = String.lsplit2 rest ~on:' ' in
  let* _, ext = String.lsplit2 rest ~on:']' in
  let* ext = String.chop_prefix ext ~prefix:"+seg" in
  let* i, move = String.lsplit2 ext ~on:':' in
  let* segs = Int.of_string_opt segs in
  let* i = Int.of_string_opt i in
  if String.is_empty move then None else Some (i, segs)

let () =
  (* The fissioned presets lead the seeds, so the beam holds a fissioned parent; everything else,
     this round's candidates included, is slower, so the search stops after one round. *)
  let (o : Scripted_search.observed) =
    Scripted_search.search ~rounds:1 ~log:false ~progress:false ~tag:"round labels"
      ~measure:(fun w -> if String.is_prefix w.label ~prefix:"F_preset[" then 1. else 1000.)
      (chain ())
  in
  let rounds = List.filter o.attempts ~f:is_saved in
  Stdio.eprintf "round candidates (not part of the golden): %d\n" (List.length rounds);
  List.iter rounds ~f:(fun l -> Stdio.eprintf "  %s\n" l);
  p "the search ran its round" (o.report.Autotune.rounds_run >= 1);
  p_exists "a round candidate extends a segment of a fissioned parent" rounds ~f:(fun l ->
      Option.is_some (segment_of l));
  p_all "every fissioned round candidate names a segment of its parent"
    (List.filter rounds ~f:(String.is_prefix ~prefix:"F_saved["))
    ~f:(fun l -> Option.exists (segment_of l) ~f:(fun (i, segs) -> 0 <= i && i < segs));
  p_all "every round candidate names its move" rounds ~f:(fun l ->
      match String.lsplit2 l ~on:']' with
      | Some (_, ext) -> String.is_prefix ext ~prefix:"+" && String.length ext > 1
      | None -> false);
  p_pairwise_distinct "the round's candidates all print apart" rounds ~equal:String.equal
    ~to_string:Fn.id

(* --- Part 3: a winner replay keeps the label its winner was timed under --- *)

(* With a [timing_ctx] the search times against a scratch lineage and the winner is rebuilt from its
   saved schedule in the caller's context. That rebuilt spec carries no round move, so the replay
   used to print -- and a failed one to be reported -- as the bare [F_saved[N segs]] every round
   candidate of that parent shares. The ranking is scripted so a round candidate wins (the fissioned
   presets lead the seeds, the round's extensions beat them), and the replay's log line is claimed
   to name exactly the crowned label. *)
let () =
  let report = ref None in
  let measure (w : Scripted_search.window) =
    if String.is_prefix w.label ~prefix:"F_preset[" then 1.
    else if is_saved w.label && String.is_substring w.label ~substring:" from " then 0.5
    else 1000.
  in
  let ((), _), lines =
    Stderr_capture.lines (fun () ->
        Scripted_search.scripted ~measure (fun () ->
            let ctx = Context.cpu () in
            let ctx, _routine =
              Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:"" ~log:true
                ~progress:false ~timing_ctx:ctx
                ~report:(fun r -> report := Some r)
                ctx (chain ()) Ir.Indexing.Empty
            in
            Context.release ctx))
  in
  let best = Option.value_map !report ~default:"" ~f:(fun r -> r.Autotune.best_label) in
  Stdio.eprintf "crowned (not part of the golden): %s\n" best;
  p "the crowned candidate is a round candidate, named by its parent and move"
    (is_saved best && String.is_substring best ~substring:" from ");
  let replays =
    List.filter_map lines ~f:(String.chop_prefix ~prefix:"autotune: winner replay ok: ")
  in
  p "the winner replayed once" (List.length replays = 1);
  p_all "the winner replay is labeled as its winner was timed" replays ~f:(String.equal best)
