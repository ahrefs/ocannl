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
    sk_coalesce = false;
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
    sk_coalesce = true;
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
    ~f:(fun l -> String.mem l ',')

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
  let attempts = ref [] and report = ref None in
  let old_attempt = !Autotune.on_candidate_attempt
  and old_measured = !Autotune.on_candidate_measured in
  Exn.protect
    ~finally:(fun () ->
      Autotune.on_candidate_attempt := old_attempt;
      Autotune.on_candidate_measured := old_measured)
    ~f:(fun () ->
      (Autotune.on_candidate_attempt := fun label -> attempts := label :: !attempts);
      (* The fissioned presets lead the seeds, so the beam holds a fissioned parent; everything
         else, this round's candidates included, is slower, so the search stops after one round. *)
      (Autotune.on_candidate_measured :=
         fun ~label ~digest:_ _ms ->
           if String.is_prefix label ~prefix:"F_preset[" then 1. else 1000.);
      let ctx, _routine =
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:"" ~log:false
              ~progress:false
              ~report:(fun r -> report := Some r)
              (Context.cpu ()) (chain ()) Ir.Indexing.Empty)
      in
      Context.release ctx);
  let rounds = List.rev !attempts |> List.filter ~f:is_saved in
  Stdio.eprintf "round candidates (not part of the golden): %d\n" (List.length rounds);
  List.iter rounds ~f:(fun l -> Stdio.eprintf "  %s\n" l);
  p "the search ran its round" (Option.exists !report ~f:(fun r -> r.Autotune.rounds_run >= 1));
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
