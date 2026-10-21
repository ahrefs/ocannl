(* gh-ocannl-1440: one wall-clock budget for a whole [Train.tune_placements] call.

   [tune_call_budget_seconds] (here [?call_budget]) runs one clock from the start of the call. Each
   search the call starts gets a share: what is left, divided among the searches still planned --
   the arms not yet searched plus the flip slots ([inline_flips] before the decision surface is
   read, then the surface's alternatives if fewer) -- and is cut at its share by
   [autotune_budget_seconds]' rule: a non-floor attempt starts only while the search's elapsed time
   is below it. Both arms always search; the flip refinement starts no alternative once the call's
   budget is spent.

   Never timed: both clocks are virtual, counted in candidate attempts
   ([Autotune.on_candidate_attempt], whose first attempt in every search is its "baseline" base
   compile). The call's clock ([Autotune.call_budget_elapsed]) reads the attempts since the call
   started, a search's ([Autotune.budget_elapsed]) those since its own baseline. Timings are
   scripted (arm A fast, arm B slow, every flip hopeless), on cc. The claims, for budgets that cut
   inside arm A, inside the flip chain, and nowhere: - every report carries the call's budget, and
   its share is exactly what was left when its search started divided by the searches still planned,
   from the attempts the test counted; - within each search, no non-floor attempt started at or past
   its share; - no flip search started once the call's budget was spent, while both arms always
   reported; - a tighter per-search budget wins over the share; - the shipped routine computes the
   untuned output bit for bit; - and the budget cuts: the tight call attempts fewer candidates and
   measures fewer flips than the unbudgeted control. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
open Verdict.Claims

let backend_name = Context.backend_name (Context.cpu ())

(* The untuned floor a cc search attempts whatever its budget: the default-pipeline seed. *)
let is_floor label =
  String.equal label "baseline"
  || Option.exists (Autotune.default_seed_label ~backend_name) ~f:(String.equal label)

let inline_flips = 2

let n = 8
and m = 4

let nd label d modulus offset =
  TDSL.ndarray
    (Array.init (d * d) ~f:(Ll_test.cycle_flat ~dims:[| d; d |] ~modulus ~offset ~stride:0.5))
    ~label:[ label ] ~output_dims:[ d; d ] ()

(* Two branches of different dims: flip candidates, and no structural twins among them. *)
let a1 = nd "a1" n 7 0.
let b1 = nd "b1" n 5 (-2.)
let a3 = nd "a3" m 7 0.
let b3 = nd "b3" m 5 (-2.)
let%op x1 = a1 *. b1
let%op y1 = relu x1
let%op x3 = a3 *. b3
let%op y3 = relu x3
let%op loss = y1 ++ "ij => 0" + (y3 ++ "ij => 0")
let comp = Train.forward loss

let reference =
  let ctx, routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  values

let total_alternatives =
  let surface = Autotune.placement_surface (Context.cpu ()) comp Ir.Indexing.Empty in
  List.sum
    (module Int)
    surface.Autotune.ps_candidates
    ~f:(fun fc -> List.length fc.LL.fc_alternatives)

let flip_slots = min inline_flips total_alternatives

(* One search as the test saw it: its attempts' labels, the call's attempts before it started, and
   its report. *)
type search = { attempts : string list; started_at : int; report : Autotune.report; flip : bool }
type call = { searches : search list; output : float array; total_attempts : int }

(* Each call searches in a cache of its own: a search must not replay another call's crown. *)
let run ?call_budget ?(settings = []) ~cache_dir tag =
  Scripted_search.clear_cache_dir cache_dir;
  (* Each attempt with the search it belongs to: a "baseline" attempt opens a search. *)
  let attempts = ref [] and n_searches = ref 0 in
  let reports = ref [] and n_arm = ref 0 in
  let script _ = match !n_arm with 0 -> 4.0 | 1 -> 50.0 | _ -> 100.0 in
  let on_attempt label =
    if String.equal label "baseline" then Int.incr n_searches;
    attempts := (!n_searches, label) :: !attempts
  in
  let search_attempts () =
    List.count !attempts ~f:(fun (s, _) -> s = !n_searches) |> Float.of_int
  in
  let (ctx, routine), _record =
    Autotune.Seam.with_value Autotune.on_candidate_attempt on_attempt @@ fun () ->
    Autotune.Seam.with_value Autotune.call_budget_elapsed (fun _real ->
        Float.of_int (List.length !attempts))
    @@ fun () ->
    Autotune.Seam.with_value Autotune.budget_elapsed (fun _real -> search_attempts ()) @@ fun () ->
    Scripted_search.scripted ~settings
      ~measure:(fun _ -> script ())
      (fun () ->
        Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir ~placement_store:false
          ~inline_flips ?call_budget
          ~report:(fun r ->
            Int.incr n_arm;
            reports := (false, r) :: !reports)
          ~flip_report:(fun r -> reports := (true, r) :: !reports)
          (Context.cpu ()) loss comp Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  let output = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  let attempts = List.rev !attempts and reports = List.rev !reports in
  let searches =
    List.mapi reports ~f:(fun i (flip, report) ->
        let own = List.filter_map attempts ~f:(fun (s, l) -> Option.some_if (s = i + 1) l) in
        let started_at = List.count attempts ~f:(fun (s, _) -> s <= i) in
        { attempts = own; started_at; report; flip })
  in
  Stdio.eprintf "%s (not part of the golden): %d attempts; %s\n%!" tag (List.length attempts)
    (String.concat ~sep:" | "
       (List.map searches ~f:(fun s ->
            Printf.sprintf "%s@%d share %s: %s"
              (if s.flip then "flip" else "arm")
              s.started_at
              (Option.value_map s.report.Autotune.call_share_s ~default:"-"
                 ~f:(Printf.sprintf "%.4g"))
              (String.concat ~sep:"," s.attempts))));
  ( { searches; output; total_attempts = List.length attempts },
    List.length attempts = List.length (List.concat_map searches ~f:(fun s -> s.attempts)) )

let arms c = List.filter c.searches ~f:(fun s -> not s.flip)
let flips c = List.filter c.searches ~f:(fun s -> s.flip)

(* The share each search should have had: what was left when it started, over the searches still
   planned then. *)
let expected_shares ~budget c =
  let measured = ref 0 in
  List.mapi c.searches ~f:(fun i s ->
      let left =
        match i with
        | 0 -> 2 + inline_flips
        | 1 -> 1 + inline_flips
        | _ ->
            let l = max 1 (flip_slots - !measured) in
            Int.incr measured;
            l
      in
      (budget -. Float.of_int s.started_at) /. Float.of_int left)

let check ~tag ?per_search ~budget (c, every_attempt_attributed) =
  p (tag ^ ": every attempt belongs to a reported search") every_attempt_attributed;
  p (tag ^ ": both arms searched and reported, in position") (List.length (arms c) = 2);
  p_all (tag ^ ": every report carries the call's budget") c.searches ~f:(fun s ->
      Option.equal Float.equal s.report.Autotune.call_budget_s (Some budget));
  p
    (tag ^ ": each search's share is what was left over the searches still planned")
    (List.equal (Option.equal Float.equal)
       (List.map c.searches ~f:(fun s -> s.report.Autotune.call_share_s))
       (List.map (expected_shares ~budget c) ~f:Option.some));
  p_all (tag ^ ": no non-floor attempt started at or past its search's cut") c.searches ~f:(fun s ->
      (* A report with no share was cut by nothing the call set: the claim fails for it. *)
      Option.value_map s.report.Autotune.call_share_s ~default:false ~f:(fun share ->
          let share = Float.max 0. share in
          let cut = Option.value_map per_search ~default:share ~f:(Float.min share) in
          List.for_alli s.attempts ~f:(fun i label -> is_floor label || Float.(of_int i < cut))));
  p_empty
    (tag ^ ": no flip search started once the call's budget was spent")
    ~over:c.searches
    (List.filter (flips c) ~f:(fun s -> Float.(of_int s.started_at >= budget)));
  p_all2
    (tag ^ ": the shipped routine computes the untuned output bit for bit")
    c.output reference ~f:Float.equal

(* --- the setting --- *)

let () =
  let parse v =
    Scripted_search.with_settings [ ("tune_call_budget_seconds", v) ] Autotune.call_budget_setting
  in
  p "setting: a positive number of seconds is a call budget"
    (Option.equal Float.equal (parse "90") (Some 90.));
  p_all "setting: empty, zero and negative mean none" [ ""; "0"; "-1" ] ~f:(fun v ->
      Option.is_none (parse v));
  p "setting: unparsable text is refused, naming the key"
    (match parse "soon" with
    | _ -> false
    | exception Utils.Invalid_config_value { key; _ } -> String.equal key "tune_call_budget_seconds")

(* --- the control and three budgets --- *)

let control_cache_dir = "autotune_cache_call_budget_control"
let tight_cache_dir = "autotune_cache_call_budget_tight"
let mid_cache_dir = "autotune_cache_call_budget_mid"
let generous_cache_dir = "autotune_cache_call_budget_generous"
let per_search_cache_dir = "autotune_cache_call_budget_per_search"
let control, _ = run ~cache_dir:control_cache_dir "control"

let () =
  p_all "control: no call budget on any report" control.searches ~f:(fun s ->
      Option.is_none s.report.Autotune.call_budget_s);
  p "control: the flip chain measured flips" (not (List.is_empty (flips control)))

(* Spent inside arm A: its share is 3/(2+2) attempts. *)
let tight = run ~call_budget:3. ~cache_dir:tight_cache_dir "tight"

(* Spent inside the flip chain. With 7 attempts, arm A's share of 7/4 admits the baseline, one
   candidate and the floor (3 attempts), arm B's of 4/3 the same (6), and the first flip's of 1/2
   its baseline (7): the budget is spent before the second flip. A menu change that moves these
   counts moves where 7 cuts, which the two "mid" claims below say. *)
let mid_budget = 7.
let mid = run ~call_budget:mid_budget ~cache_dir:mid_cache_dir "mid"
let generous = run ~call_budget:10000. ~cache_dir:generous_cache_dir "generous"

let per_search =
  run ~call_budget:10000.
    ~settings:[ ("autotune_budget_seconds", "1") ]
    ~cache_dir:per_search_cache_dir "per-search"

let () =
  check ~tag:"tight" ~budget:3. tight;
  check ~tag:"mid" ~budget:mid_budget mid;
  check ~tag:"generous" ~budget:10000. generous;
  check ~tag:"per-search" ~per_search:1. ~budget:10000. per_search;
  let tight, _ = tight and mid, _ = mid and generous, _ = generous and per_search, _ = per_search in
  p "tight: the call attempted fewer candidates than the control"
    (tight.total_attempts < control.total_attempts);
  p_empty "tight: the budget was spent before the flips, so none was measured" ~over:tight.searches
    (flips tight);
  p "mid: the chain measured a flip, then stopped short of the control's flips"
    (List.length (flips mid) >= 1 && List.length (flips mid) < List.length (flips control));
  p "generous: a budget never spent changes nothing the control did"
    (generous.total_attempts = control.total_attempts
    && List.length (flips generous) = List.length (flips control));
  p_all "generous: no search was cut" generous.searches ~f:(fun s ->
      not s.report.Autotune.budget_cut);
  p_all "per-search: every report carries the per-search budget, the tighter cut"
    per_search.searches ~f:(fun s -> Option.equal Float.equal s.report.Autotune.budget_s (Some 1.))

let () = Stdio.printf "tune_call_budget: done\n"
