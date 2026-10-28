(* gh-ocannl-1556, gh-ocannl-1604, gh-ocannl-1667: [autotune_budget_seconds] bounds a search's wall
   time, not only when its last attempt starts. A compile cannot be cancelled, so an attempt starts
   only where a compile of the projected cost still ends within the budget with a second one
   reserved for the winner's recompile, which runs after every check; and a non-floor candidate's
   timing is abandoned, unranked, before its first batch past the budget less that reserve. The
   projection is the costliest compile the search has made (the base compile's included: its first
   evidence), except that once a compile has reused a kernel an earlier one published
   ([Ir.Compile_cost.t]'s [kernels_reused], the kernel code cache), the cold compiles before it stop
   counting.

   No real clock: every search reads a virtual one through [Autotune.budget_elapsed]. It advances
   only by what the test charges -- the base compile [base_s], the serial baseline's compile
   [baseline_s], each non-floor candidate compile [compile_s] and the winner's recompile [winner_s]
   (each charged as the compile's record reaches [Autotune.install_compile_observer], inside the
   compile's span), and each budget read made during one chosen candidate's timing [batch_s] (the
   reads the timing check makes, one before the warmup and one before each batch). The floors (the
   untuned default seed) cost nothing: they are exempt from the budget, so the claims bound only
   what the budget governs. The timings are scripted ([Scripted_search]) so the ranking is the
   script's; the crowned routines are still compiled, run and compared with the untuned output.
   Searched on cc, over [Fissioned_chain]'s small menu.

   The kernel code cache is process-wide, so which compiles reuse a kernel is set by what this
   process compiled before. The control fills it for the default chain, so in every later search of
   that chain the baseline's compile already reuses its kernel and the base compile has stopped
   counting before the first candidate. A search that needs a cold baseline runs on a chain of its
   own row count: the third product's extent is in the baseline's one kernel (and the untuned
   default's), not in the first two products' kernels, which every fissioned candidate reuses. *)

open Base
open Verdict.Claims

let () = Fissioned_chain.small_menu ()
let budget_key = "autotune_budget_seconds"
let backend_name = Context.backend_name (Context.cpu ())
let default_label = Autotune.default_seed_label ~backend_name

let is_floor label =
  String.equal label "baseline" || Option.exists default_label ~f:(String.equal label)

(* The baseline slowest, the untuned default next, every other candidate a distinct faster time. *)
let cost label =
  if String.equal label "baseline" then 1000.
  else if Option.exists default_label ~f:(String.equal label) then 100.
  else 2. +. (Float.of_int (Stdlib.Hashtbl.hash label % 1000) /. 100.)

let default_chain = Fissioned_chain.chain_with_output ()

(* The virtual clock and what one search charged to it. [timing_of] is the candidate whose timing
   reads are charged; [in_timing] is set by its pre-dispatch validation and cleared as the next
   attempt starts or its window is admitted. *)
type charges = {
  mutable clock : float;
  mutable compiles : int;  (** Non-floor candidate compiles charged. *)
  mutable recompiles : int;  (** Winner recompiles charged. *)
  mutable timing_reads : int;  (** Budget reads charged during [timing_of]'s timing. *)
  mutable current : string;
  mutable in_timing : bool;
}

(* What a search charges per compile: the base compile, the serial baseline's, a non-floor
   candidate's, the winner's recompile. *)
type rates = {
  base_s : float;
  baseline_s : float;
  compile_s : float;
  winner_s : float;
  by_reuse : (float * float) option;
      (** [Some (cold_s, cached_s)]: instead of the rates above, every compile but the base one --
          the baseline's, each candidate's (the floors' too) and the winner's -- is charged [cold_s]
          when it compiled kernels and reused none, [cached_s] otherwise. *)
}

(* The search in flight's charges and rates, and its compile records in order, each with the clock
   after its charge. *)
let active : (charges * rates) option ref = ref None
let records : (Autotune.compile_record * float) list ref = ref []
let reuses (r : Autotune.compile_record) = r.compile_cost.kernels_reused > 0
let is_cold (r : Autotune.compile_record) = r.compile_cost.kernels_compiled > 0 && not (reuses r)

let () =
  Autotune.install_compile_observer (fun r ->
      (match (!active, r.Autotune.compile_stage) with
      | Some (c, { by_reuse = Some (cold_s, cached_s); _ }), stage
        when not (String.equal stage "base_compile") ->
          if String.equal stage "candidate" && not (is_floor r.compile_attempt) then
            c.compiles <- c.compiles + 1;
          if String.equal stage "winner_compile" then c.recompiles <- c.recompiles + 1;
          c.clock <- (c.clock +. if is_cold r then cold_s else cached_s)
      | Some (c, rates), "candidate" when not (is_floor r.Autotune.compile_attempt) ->
          c.compiles <- c.compiles + 1;
          c.clock <- c.clock +. rates.compile_s
      | Some (c, rates), "base_compile" -> c.clock <- c.clock +. rates.base_s
      | Some (c, rates), "baseline_compile" -> c.clock <- c.clock +. rates.baseline_s
      | Some (c, rates), "winner_compile" ->
          c.recompiles <- c.recompiles + 1;
          c.clock <- c.clock +. rates.winner_s
      | Some _, _ | None, _ -> ());
      Option.iter !active ~f:(fun (c, _) -> records := (r, c.clock) :: !records))

(* One search of [chain] (the default chain's by default) under a budget, charging [base_s] for the
   base compile, [baseline_s] for the serial baseline's, [compile_s] per non-floor candidate
   compile, [winner_s] for the winner's recompile and [batch_s] per budget read during [timing_of]'s
   timing. Returns its compile records too, in order. *)
let search ?(chain = default_chain) ?budget ?(base_s = 0.) ?(baseline_s = 0.) ?(compile_s = 0.)
    ?(winner_s = 0.) ?by_reuse ?(batch_s = 0.) ?timing_of ~tag () =
  let c =
    { clock = 0.; compiles = 0; recompiles = 0; timing_reads = 0; current = ""; in_timing = false }
  in
  records := [];
  let qg, comp = chain in
  let attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  let preflight = Autotune.Seam.get Autotune.on_candidate_preflight in
  let measured = Autotune.Seam.get Autotune.on_candidate_measured in
  let o =
    Exn.protect
      ~finally:(fun () -> active := None)
      ~f:(fun () ->
        active := Some (c, { base_s; baseline_s; compile_s; winner_s; by_reuse });
        Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
            c.current <- label;
            c.in_timing <- false;
            attempt label)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_preflight (fun name ->
            c.in_timing <- Option.exists timing_of ~f:(String.equal c.current);
            preflight name)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_measured (fun ~label ~digest reading ->
            c.in_timing <- false;
            measured ~label ~digest reading)
        @@ fun () ->
        Autotune.Seam.with_value Autotune.budget_elapsed (fun _real ->
            if c.in_timing then (
              c.timing_reads <- c.timing_reads + 1;
              c.clock <- c.clock +. batch_s);
            c.clock)
        @@ fun () ->
        Scripted_search.search
          ~settings:(Option.value_map budget ~default:[] ~f:(fun b -> [ (budget_key, b) ]))
          ~output:qg ~cache_dir:"" ~tag
          ~measure:(fun w -> cost w.Scripted_search.label)
          comp)
  in
  Stdio.eprintf
    "%s (not part of the golden): attempts %s | measured %s | clock %.1f, %d compiles charged, %d \
     recompiles, %d timing reads, budget_cut %b, crown %s\n\
     %!"
    tag
    (String.concat ~sep:"; " o.attempts)
    (String.concat ~sep:"; " o.measured)
    c.clock c.compiles c.recompiles c.timing_reads o.report.Autotune.budget_cut
    o.report.Autotune.best_label;
  (o, c, List.rev !records)

(* The untuned output of [chain], computed AFTER the searches of it that need a cold baseline: the
   untuned default's kernel also holds the third product's extent. *)
let reference chain = Scripted_search.untuned (fst chain) (snd chain)
let is_stage stage (r : Autotune.compile_record) = String.equal r.compile_stage stage

let is_non_floor_candidate (r : Autotune.compile_record) =
  is_stage "candidate" r && not (is_floor r.compile_attempt)

let baseline_compiled_cold records =
  List.exists records ~f:(fun (r, _) -> is_stage "baseline_compile" r)
  && List.for_all records ~f:(fun (r, _) -> (not (is_stage "baseline_compile" r)) || is_cold r)

let non_floor attempts = List.filter attempts ~f:(fun l -> not (is_floor l))

let searched (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Searched -> true | _ -> false

(* --- the control: no budget, every candidate attempted --- *)

let control, _, _ = search ~compile_s:3. ~tag:"control" ()
let default_reference = reference default_chain
let control_non_floor = non_floor control.attempts
let control_measured = non_floor control.measured

let () =
  p "control: the menu offers at least five non-floor attempts" (List.length control_non_floor >= 5);
  p "control: nothing was cut" (searched control && not control.report.Autotune.budget_cut)

(* --- the projection: each non-floor compile and the winner's recompile cost 3 s against a 10 s
   budget --- *)

let () =
  let budget = 10. and compile_s = 3. in
  let o, c, _ = search ~budget:"10" ~compile_s ~winner_s:compile_s ~tag:"projection" () in
  (* At 6 s a third compile would end at 9 s, but the winner's recompile reserved after it would end
     at 12 s: the reserve refuses it, where the projection alone (or a check of the elapsed time
     alone) started it. *)
  p "projection: the winner's recompile was charged once" (c.recompiles = 1);
  p "projection: the search ended within the budget, the winner's recompile included"
    Float.(c.clock <= budget);
  p "projection: as many compiles started as fit in the budget beside the reserve"
    (c.compiles = Float.iround_towards_zero_exn ((budget -. compile_s) /. compile_s));
  p "projection: its attempts are a prefix of the control's, cut by the budget"
    (searched o && o.report.Autotune.budget_cut
    && List.is_prefix control_non_floor ~prefix:(non_floor o.attempts) ~equal:String.equal);
  p_all2 "projection: the crowned routine computes the untuned output bit for bit" o.output
    default_reference ~f:Float.equal

(* --- the base compile: charged, and the first compile evidence while nothing has reused a kernel:
   a 4 s base compile against a 10 s budget, on a chain whose baseline compiles cold --- *)

let () =
  let chain = Fissioned_chain.chain_with_output ~rows:32 () in
  let budget = 10. and base_s = 4. and compile_s = 1. in
  let o, c, records =
    search ~chain ~budget:"10" ~base_s ~compile_s ~winner_s:compile_s ~tag:"base" ()
  in
  (* The baseline's compile reuses no kernel, so the base compile is still the projection at the
     first candidate's check: from 4 s, a compile projected at 4 s and the reserve end at 12 s, so
     no non-floor compile starts. Without the base compile as evidence, the first started
     unprojected. *)
  p "base: the serial baseline's compile compiled its kernels and reused none"
    (baseline_compiled_cold records);
  p "base: the base compile's projection refused every non-floor compile"
    (Float.(base_s +. (2. *. base_s) > budget) && c.compiles = 0);
  p "base: the winner's recompile was charged once" (c.recompiles = 1);
  p "base: the search ended within the budget, the base compile and the winner's recompile included"
    Float.(c.clock <= budget);
  p "base: it was cut by the budget" (searched o && o.report.Autotune.budget_cut);
  p_all2 "base: the crowned routine computes the untuned output bit for bit" o.output
    (reference chain) ~f:Float.equal

(* --- the cache: the cold compiles stop counting once a compile reuses a kernel. Every compile is
   charged by what it reused: 4 s when it compiled kernels and reused none, 1 s otherwise (the
   floors' and the winner's recompile too), against a 16 s budget, on a chain whose baseline
   compiles cold --- *)

let () =
  let chain = Fissioned_chain.chain_with_output ~rows:48 () in
  let budget = 16. and cold_s = 4. and cached_s = 1. in
  let o, c, records = search ~chain ~budget:"16" ~by_reuse:(cold_s, cached_s) ~tag:"cache" () in
  p "cache: the serial baseline's compile compiled its kernels and reused none"
    (baseline_compiled_cold records);
  (* The clock as the first compile that reused a kernel returned, and the non-floor compiles after
     it. Before it, the projection is the cold compiles' 4 s; from it on, the cached compiles' 1 s,
     so non-floor compiles start while the elapsed time plus two of them fits. *)
  let rec split = function
    | [] -> None
    | (r, clock) :: rest -> if reuses r then Some (clock, rest) else split rest
  in
  match split records with
  | None -> p "cache: a compile reused a kernel" false
  | Some (reused_at, after) ->
      let after_non_floor = List.count after ~f:(fun (r, _) -> is_non_floor_candidate r) in
      p_none "cache: no compile after the first reusing one compiled cold" after ~f:(fun (r, _) ->
          is_cold r);
      p "cache: a projection held at the cold compiles' cost would have refused every later compile"
        Float.(reused_at +. (2. *. cold_s) > budget);
      p
        "cache: as many non-floor compiles started after it as the cached compiles' projection \
         admits"
        (after_non_floor
        = Float.iround_towards_zero_exn ((budget -. reused_at -. (2. *. cached_s)) /. cached_s) + 1
        );
      p "cache: the winner's recompile was charged once" (c.recompiles = 1);
      p
        "cache: the search ended within the budget, the cold compiles and the winner's recompile \
         included"
        Float.(c.clock <= budget);
      p "cache: it was cut by the budget" (searched o && o.report.Autotune.budget_cut);
      p_all2 "cache: the crowned routine computes the untuned output bit for bit" o.output
        (reference chain) ~f:Float.equal

(* --- the timing check: the second non-floor candidate the control timed passes a 3 s budget in its
   timing, at 1 s per batch --- *)

let () =
  match List.nth control_measured 1 with
  | None -> p "timing: the control timed a second non-floor candidate" false
  | Some slow ->
      let budget = 3. and batch_s = 1. in
      let o, c, _ = search ~budget:"3" ~batch_s ~timing_of:slow ~tag:"timing" () in
      p "timing: the slow candidate was attempted" (List.mem o.attempts slow ~equal:String.equal);
      p "timing: its timing was abandoned, unranked"
        (not (List.mem o.measured slow ~equal:String.equal));
      p "timing: the search completed, cut by its budget"
        (searched o && o.report.Autotune.budget_cut);
      (* One read before the warmup (1 s), then one per batch: the read that passed 3 s was the
         fourth, so batches ran before the stop and none ran after it. *)
      p "timing: it stopped at the first batch past the budget, not before its warmup"
        (c.timing_reads = Float.iround_towards_zero_exn (budget /. batch_s) + 1);
      p "timing: the clock passed the budget by one batch at most"
        Float.(c.clock > budget && c.clock <= budget +. batch_s);
      p "timing: no non-floor attempt started after it"
        (Option.equal String.equal (List.last (non_floor o.attempts)) (Some slow));
      p_all2 "timing: the crowned routine computes the untuned output bit for bit" o.output
        default_reference ~f:Float.equal

(* --- the reserve in the timing check: the same slow candidate, after 1 s candidate compiles,
   against a 6 s budget at 1 s per batch, with a 1 s winner's recompile --- *)

let () =
  match List.nth control_measured 1 with
  | None -> p "reserve: the control timed a second non-floor candidate" false
  | Some slow ->
      let budget = 6. and compile_s = 1. and batch_s = 1. in
      let o, c, _ =
        search ~budget:"6" ~compile_s ~winner_s:compile_s ~batch_s ~timing_of:slow ~tag:"reserve" ()
      in
      p "reserve: the slow candidate was attempted" (List.mem o.attempts slow ~equal:String.equal);
      p "reserve: its timing was abandoned, unranked"
        (not (List.mem o.measured slow ~equal:String.equal));
      p "reserve: the winner's recompile was charged once" (c.recompiles = 1);
      (* The slow candidate is the second non-floor one: its timing starts at 2 s, after two 1 s
         compiles, and the read that passed 5 s (the budget less the 1 s reserve) was the fourth. *)
      p "reserve: two non-floor compiles preceded its timing" (c.compiles = 2);
      p "reserve: its timing stopped at the first batch past the budget less the reserve"
        (c.timing_reads
        = Float.iround_towards_zero_exn ((budget -. (2. *. compile_s) -. compile_s) /. batch_s) + 1
        );
      p
        "reserve: the search, the winner's recompile included, passed the budget by one batch at \
         most"
        Float.(c.clock <= budget +. batch_s);
      p_all2 "reserve: the crowned routine computes the untuned output bit for bit" o.output
        default_reference ~f:Float.equal
