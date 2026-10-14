(* gh-ocannl-1274: the flip chain skips a structural twin of a flip that lost, without a budget
   slot.

   On rog's approximate CUDA gpt2_mini cell the first 11 slots of the inline-flip chain were the
   per-layer q/k/v projections, and each set's members lost by the same figure: a budget of 2 never
   reached past layer 1's k and v. The rule: an alternative whose local reading is canonically
   identical ([Autotune.placement_surface]'s [ps_twin_identity] -- the flipped node's
   producer/consumer statements under fresh-name normalization, each node with its dims, precision
   and placements, plus the flip's readings and recompute cost) to one already measured and lost in
   the same chain is skipped as [twin-of] that flip, and spends no budget.

   The fixture has three branches of the same shape, [x = a *. b; y = relu x], each summed into the
   loss: the first two over 8x8 inputs, the third over 4x4. So [x1]/[x2] and [y1]/[y2] are twins,
   while [x3]/[y3] share their structure but not their dims, and must not be twinned (the issue's
   negative acceptance). On cc, with every timing from a synthetic clock
   ([Autotune.on_candidate_measured]): - the surface's identities, printed as twin classes; - a
   chain whose every flip is hopeless: each twin is skipped as [twin-of] its first member, the
   dims-differing branch is measured, the budget counts only measured flips, and a budget of 2
   reaches 2 distinct classes past a skipped twin; - a warm rerun of that chain replays the anchors'
   abandonments and skips the same twins; - a flip searched in full that loses anchors its twin just
   as an abandoned one does; - the control: flips that WIN anchor nothing, so their twins are
   measured. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

(* The abandonment depth (the beam width), and no beam rounds: each search times only its seeds. A
   flip at 100 ms is abandoned when its second attempt starts, so the chain compiles as few
   candidates as the claims allow -- every compiled candidate is a fresh dlopen, which macOS
   scans. *)
let k = 1

(* One [Autotune.on_candidate_measured] script: the [i]-th admitted window measures [script i]. *)
let with_clock script f =
  let measured = ref 0 in
  let old_measured = !Autotune.on_candidate_measured in
  Exn.protect
    ~finally:(fun () -> Autotune.on_candidate_measured := old_measured)
    ~f:(fun () ->
      (Autotune.on_candidate_measured :=
         fun ~label:_ ~digest:_ _ms ->
           Int.incr measured;
           script !measured);
      Autotune.with_uncontended_test_windows f)

let with_stderr_captured f =
  let file = Stdlib.Filename.temp_file "flip_twins" ".stderr" in
  Stdio.Out_channel.flush Stdio.stderr;
  let saved = Unix.dup Unix.stderr in
  let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  Unix.dup2 fd Unix.stderr;
  Unix.close fd;
  let restore () =
    Stdio.Out_channel.flush Stdio.stderr;
    Unix.dup2 saved Unix.stderr;
    Unix.close saved
  in
  let result = Exn.protect ~f ~finally:restore in
  let text = Stdio.In_channel.read_all file in
  Stdlib.Sys.remove file;
  (* Echoed, so nothing the run wrote is hidden. *)
  Stdio.eprintf "%s%!" text;
  (result, String.split_lines text)

(* The value of [key="..."] on an [autotune-progress:] line of the given event. Arm labels carry
   spaces, so the fields are read by their quotes rather than split on spaces. *)
let quoted line key =
  let open String in
  Option.bind
    (substr_index line ~pattern:(key ^ "=\""))
    ~f:(fun i ->
      let start = i + length key + 2 in
      Option.map (index_from line start '"') ~f:(fun j -> sub line ~pos:start ~len:(j - start)))

let events lines name =
  List.filter lines ~f:(fun l ->
      String.is_prefix l ~prefix:"autotune-progress: "
      && String.is_substring l ~substring:(" event=" ^ name ^ " "))

let plain lines name key =
  List.filter_map (events lines name) ~f:(fun l ->
      List.find_map (String.split l ~on:' ') ~f:(fun w -> String.chop_prefix w ~prefix:(key ^ "=")))

type chain = {
  flip_reports : Autotune.report list;
  measured_arms : string list;  (** The flips' [arm_start] labels, in chain order. *)
  twins : (string * string) list;  (** [(arm, twin_of)] per [flip_twin] event, in chain order. *)
  done_measured : string list;
  done_twinned : string list;
  shipped : string list;
}

(* [Train.tune_placements] with arm A's windows at 4 ms, arm B's at 50 ms, and each flip's per
   [flip_script ~flip window] (both 1-based: the flip's ordinal among the measured ones, the
   window's within that flip's search). The searches are told apart by how many reports have been
   delivered when a window is admitted. *)
let run_chain ?(cache_dir = "") ?timing_ctx ~budget comp loss ~flip_script =
  let arm_reports = ref [] and flip_reports = ref [] and shipped = ref [] in
  let delivered () = List.length !arm_reports + List.length !flip_reports in
  let flip_window = ref 0 and flip_seen = ref 0 in
  let script _ =
    match delivered () with
    | 0 -> 4.0
    | 1 -> 50.0
    | d ->
        if d <> !flip_seen then (
          flip_seen := d;
          flip_window := 0);
        Int.incr flip_window;
        flip_script ~flip:(d - 1) !flip_window
  in
  let (), lines =
    with_stderr_captured (fun () ->
        with_clock script (fun () ->
            let _ctx, _routine =
              Train.tune_placements ~beam_width:k ~rounds:0 ~repeats:1 ~cache_dir
                ~placement_store:false
                ~report:(fun r -> arm_reports := r :: !arm_reports)
                ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
                ~on_ship:(fun what -> shipped := what :: !shipped)
                ?timing_ctx ~inline_flips:budget (Context.auto ()) loss comp Ir.Indexing.Empty
            in
            ()))
  in
  let flip_label l = Option.filter (quoted l "arm") ~f:(String.is_prefix ~prefix:"flip ") in
  {
    flip_reports = List.rev !flip_reports;
    measured_arms = List.filter_map (events lines "arm_start") ~f:flip_label;
    twins =
      List.filter_map (events lines "flip_twin") ~f:(fun l ->
          Option.both (quoted l "arm") (quoted l "twin_of"));
    done_measured = plain lines "flips_done" "measured";
    done_twinned = plain lines "flips_done" "twinned";
    shipped = !shipped;
  }

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* An arm label names its node as [flip <reading> <debug name> (cost ...)]. *)
let names (t : Tensor.t) label =
  String.is_substring label ~substring:(" " ^ Tn.debug_name t.Tensor.value ^ " (")

let () =
  let n = 8 and m = 4 in
  let nd label dims modulus offset =
    let d = dims in
    TDSL.ndarray
      (Array.init (d * d) ~f:(Ll_test.cycle_flat ~dims:[| d; d |] ~modulus ~offset ~stride:0.5))
      ~label:[ label ] ~output_dims:[ d; d ] ()
  in
  (* Distinct values per input: the twins share structure, not data. *)
  let a1 = nd "a1" n 7 0. and b1 = nd "b1" n 5 (-2.) in
  let a2 = nd "a2" n 11 1. and b2 = nd "b2" n 3 (-1.) in
  let a3 = nd "a3" m 7 0. and b3 = nd "b3" m 5 (-2.) in
  let%op x1 = a1 *. b1 in
  let%op y1 = relu x1 in
  let%op x2 = a2 *. b2 in
  let%op y2 = relu x2 in
  let%op x3 = a3 *. b3 in
  let%op y3 = relu x3 in
  let%op loss = y1 ++ "ij => 0" + (y2 ++ "ij => 0") + (y3 ++ "ij => 0") in
  let comp = Train.forward loss in
  (* The surface, as the chain reads it. *)
  let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
  let no_chain _ = None and no_tag _ = Some "" in
  let classes = Hashtbl.create (module String) in
  let class_of id =
    match id with
    | None -> "-"
    | Some id ->
        Hashtbl.find_or_add classes id ~default:(fun () ->
            String.of_char (Char.of_int_exn (Char.to_int 'A' + Hashtbl.length classes)))
  in
  Stdio.printf "ranked surface (node, default -> alternative, cost, twin class):\n";
  List.iter surface.Autotune.ps_candidates ~f:(fun fc ->
      List.iter fc.LL.fc_alternatives ~f:(fun fa ->
          Stdio.printf "  %-10s %-11s -> %-11s cost %-5d class %s\n" (Tn.debug_name fc.LL.fc_tn)
            (LL.reading_to_string fc.LL.fc_default)
            (LL.reading_to_string fa.LL.fa_flip)
            fa.LL.fa_recompute_cost
            (class_of (surface.Autotune.ps_twin_identity ~node_tag:no_tag ~chain:no_chain fc fa))));
  let candidate (t : Tensor.t) =
    List.find surface.Autotune.ps_candidates ~f:(fun fc -> Tn.equal fc.LL.fc_tn t.Tensor.value)
  in
  let identity ?(node_tag = no_tag) ?(chain = no_chain) t =
    Option.bind (candidate t) ~f:(fun fc ->
        Option.bind (List.hd fc.LL.fc_alternatives) ~f:(fun fa ->
            surface.Autotune.ps_twin_identity ~node_tag ~chain fc fa))
  in
  let twins s t =
    Option.is_some (identity s) && Option.equal String.equal (identity s) (identity t)
  in
  p_all "every branch node is a flip candidate with an identity" [ x1; y1; x2; y2; x3; y3 ]
    ~f:(fun t -> Option.is_some (identity t));
  p "x1 and x2 are twins: same structure, same dims, same placements" (twins x1 x2);
  p "y1 and y2 are twins" (twins y1 y2);
  p "x3 shares x1's structure but not its dims, and is not its twin" (not (twins x1 x3));
  p "y3 is not y1's twin either" (not (twins y1 y3));
  p "a producer and its consumer are not twins" (not (twins x1 y1));
  (* A reading the chain has committed for a neighbour changes the flip's world: y1 reads x1, so
     materializing x1 changes y1's identity and leaves y2's, which no longer match. *)
  let x1_committed n = if Tn.equal n x1.Tensor.value then Some `Materialize else None in
  p "a chain-committed neighbour reading separates twins"
    (not
       (Option.equal String.equal (identity ~chain:x1_committed y1)
          (identity ~chain:x1_committed y2)));
  (* The caller's tag -- the flip chain's timing lineage -- enters per node too. *)
  let a2_tagged n = Some (if Tn.equal n a2.Tensor.value then ";t-differs" else "") in
  p "a caller tag on a twin's neighbour separates the twins"
    (not
       (Option.equal String.equal (identity ~node_tag:a2_tagged x2)
          (identity ~node_tag:a2_tagged x1)));
  let a2_refused n = if Tn.equal n a2.Tensor.value then None else Some "" in
  p "a caller tag refusing a neighbour leaves no identity"
    (Option.is_some (identity x2) && Option.is_none (identity ~node_tag:a2_refused x2));
  p "a commitment outside the neighbourhood leaves twins twinned"
    (Option.equal String.equal (identity ~chain:x1_committed x2) (identity x2));
  let total_alternatives =
    List.sum
      (module Int)
      surface.Autotune.ps_candidates
      ~f:(fun fc -> List.length fc.LL.fc_alternatives)
  in
  let big_budget = total_alternatives + 1 in
  let hopeless ~flip:_ _ = 100.0 in
  (* Every flip hopeless, budget past the surface: the whole chain walked. *)
  let cache_dir = "autotune_cache_flip_twins" in
  clean_cache cache_dir;
  let cold = run_chain ~cache_dir ~budget:big_budget comp loss ~flip_script:hopeless in
  let twin_of t c =
    List.find_map c.twins ~f:(fun (arm, anchor) -> Option.some_if (names t arm) anchor)
  in
  let measured t c = List.exists c.measured_arms ~f:(names t) in
  p "x2 is skipped as twin-of x1" (Option.exists (twin_of x2 cold) ~f:(names x1));
  p "y2 is skipped as twin-of y1" (Option.exists (twin_of y2 cold) ~f:(names y1));
  p_all "the first of each twin set and the dims-differing branch are measured" [ x1; y1; x3; y3 ]
    ~f:(fun t -> measured t cold && Option.is_none (twin_of t cold));
  p_none "a skipped twin is never also measured" cold.twins ~f:(fun (arm, _) ->
      List.mem cold.measured_arms arm ~equal:String.equal);
  p_all "every twin names an anchor measured before it" cold.twins ~f:(fun (_, anchor) ->
      List.mem cold.measured_arms anchor ~equal:String.equal);
  p "every alternative is either measured or skipped as a twin"
    (List.length cold.measured_arms + List.length cold.twins = total_alternatives);
  p "flips_done counts the measured flips and the twins apart"
    (List.equal String.equal cold.done_measured [ Int.to_string (List.length cold.measured_arms) ]
    && List.equal String.equal cold.done_twinned [ Int.to_string (List.length cold.twins) ]
    && List.length cold.flip_reports = List.length cold.measured_arms);
  p_all "every measured flip was abandoned" cold.flip_reports ~f:(fun r ->
      match r.Autotune.outcome with Autotune.Abandoned _ -> true | _ -> false);
  p "the A/B winner ships" (List.equal String.equal cold.shipped [ "A" ]);
  (* A warm rerun: the anchors replay their abandonments, and the same twins are skipped. *)
  let warm = run_chain ~cache_dir ~budget:big_budget comp loss ~flip_script:hopeless in
  p_all ~min:1 "the warm chain replays every measured flip's abandonment" warm.flip_reports
    ~f:(fun r -> match r.Autotune.outcome with Autotune.Abandonment_replay _ -> true | _ -> false);
  p "the warm chain measures the same flips and skips the same twins"
    (List.equal String.equal warm.measured_arms cold.measured_arms
    && List.equal (fun (a, b) (c, d) -> String.equal a c && String.equal b d) warm.twins cold.twins
    );
  clean_cache cache_dir;
  (* The flips are measured in the timing lineage, so what it brings to a neighbour separates twins
     there: with [y2] preferred inline in the timing context alone, neither [x2] (whose consumer
     [y2] is) nor [y2] itself is a twin any more, while [y3]'s branch and the other pair are
     unaffected. *)
  let timing_ctx = Context.decide_inline (Context.auto ()) [ y2.Tensor.value ] in
  let timed = run_chain ~timing_ctx ~budget:big_budget comp loss ~flip_script:hopeless in
  p "under a timing lineage that differs on y2, x2 and y2 are measured, not twinned"
    (measured x2 timed && measured y2 timed
    && Option.is_none (twin_of x2 timed)
    && Option.is_none (twin_of y2 timed));
  (* Budget 2, the approximate profile's: the first twin pair's second member is skipped without a
     slot, so the chain reaches two distinct classes and not one class twice. *)
  let first_twin_rank =
    List.find_mapi surface.Autotune.ps_candidates ~f:(fun i fc ->
        Option.some_if
          (List.exists cold.twins ~f:(fun (arm, _) ->
               String.is_substring arm ~substring:(" " ^ Tn.debug_name fc.LL.fc_tn ^ " (")))
          i)
  in
  p "the surface ranks a twin among its first three candidates"
    (Option.exists first_twin_rank ~f:(fun i -> i < 3));
  let small = run_chain ~budget:2 comp loss ~flip_script:hopeless in
  p "a budget-2 chain measures two flips" (List.length small.measured_arms = 2);
  p "it skips at least one twin on the way" (not (List.is_empty small.twins));
  p "the two flips it measures are of distinct twin classes"
    (match small.measured_arms with
    | [ f1; f2 ] ->
        let id label =
          List.find_map surface.Autotune.ps_candidates ~f:(fun fc ->
              Option.some_if
                (String.is_substring label ~substring:(" " ^ Tn.debug_name fc.LL.fc_tn ^ " ("))
                fc)
          |> Option.bind ~f:(fun fc ->
              Option.bind (List.hd fc.LL.fc_alternatives) ~f:(fun fa ->
                  surface.Autotune.ps_twin_identity ~node_tag:no_tag ~chain:no_chain fc fa))
        in
        Option.is_some (id f1)
        && Option.is_some (id f2)
        && not (Option.equal String.equal (id f1) (id f2))
    | _ -> false);
  (* A flip searched in full that loses anchors its twin like an abandoned one: within the ratio at
     depth k (5 ms against the incumbent's 4 ms, ratio 1.5625), never faster than arm A. *)
  let losing ~flip:_ _ = 5.0 in
  let lost = run_chain ~budget:2 comp loss ~flip_script:losing in
  p_all ~min:1 "every measured flip is searched in full" lost.flip_reports ~f:(fun r ->
      Poly.equal r.Autotune.outcome Autotune.Searched);
  p "a flip that lost a full search anchors its twin"
    (Option.exists (twin_of y2 lost) ~f:(names y1));
  (* Only clean evidence anchors: a flip whose every window is refused (a NaN reading, refused as a
     contended window is) measured nothing, so its twin is measured in turn. *)
  let refused ~flip:_ _ = Float.nan in
  let unclean = run_chain ~budget:2 comp loss ~flip_script:refused in
  p "a flip whose windows were all refused timed nothing"
    (match unclean.flip_reports with
    | r :: _ -> r.Autotune.candidates_timed = 0 && r.Autotune.timings_contended > 0
    | [] -> false);
  p_empty "a flip that timed nothing anchors no twin" ~over:unclean.measured_arms unclean.twins;
  p "y2 is measured after y1 timed nothing" (measured y1 unclean && measured y2 unclean);
  (* The control: every flip wins, each beating the one accepted before it. A flip that is accepted
     did not lose, so it anchors nothing and its twin is measured. *)
  let winning ~flip _ = 2.0 /. Float.of_int flip in
  let won = run_chain ~budget:2 comp loss ~flip_script:winning in
  p "the winning chain ships a refined placement" (List.equal String.equal won.shipped [ "flip" ]);
  p_empty "no flip is skipped as a twin when the flips win" ~over:won.measured_arms won.twins;
  p "y2 is measured after y1 won" (measured y1 won && measured y2 won);
  (* Two per-node facts, each declared on one member of a still-twinned pair, last since both
     declarations are monotone. The compile-time bounds candidate (the simplifier folds guards on
     it): a proposal on [b2], an operand of [x2]'s producer, separates [x2] from [x1]. *)
  let x_were_twins = twins x1 x2 in
  let b2_unknown = Option.is_none (Tn.bounds_candidate b2.Tensor.value) in
  Tn.propose_bounds ~what:"flip_twins" b2.Tensor.value (Ir.Interval.of_int_range 0 3);
  p "a bounds candidate on x2's operand separates x2 from x1"
    (x_were_twins && b2_unknown && not (twins x1 x2));
  (* Observability (an observable buffer is never aliased): declaring [y2] observable separates it
     from [y1]. *)
  let y_were_twins = twins y1 y2 in
  let was_observable = Tn.is_observable y2.Tensor.value in
  Tn.set_observable y2.Tensor.value;
  p "declaring y2 observable separates it from its twin y1"
    (y_were_twins && (not was_observable) && not (twins y1 y2))
