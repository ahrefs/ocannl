(* A substituted ship reaches the benchmark result line's [tune] object (gh-ocannl-1559).

   [Train.tune_placements] reports, through [on_ship], whether what it shipped is a schedule some
   search measured ([Train.shipped.ship_substituted], gh-ocannl-1480). When it is not -- the
   shipping compile substituted the untuned default for the crown, with nothing left to fall back
   to, or the last resort shipped the caller's own untuned compile -- the cell's step times are the
   untuned default's under a tuned variant, while the arms, [shipped] and [shipped_digest] still
   name the search whose crown was discarded. This drives the one collector every OCANNL runner
   wires to [on_ship] ([Bench_harness.collect_ship]) and the one renderer of the object
   ([Bench_harness.tune_json]) on fabricated reports and ships, and reads the result back as JSON:
   the wiring between the two, which [bench_result_line]'s fabricated [Bench_json] calls cannot
   reach. *)

open Base
open Ocannl
open Verdict.Claims
module H = Bench_harness

let member = Yojson.Safe.Util.member

let searched ~label ~digest =
  {
    (Autotune.no_search_report ~timing:Autotune.Queued) with
    Autotune.outcome = Autotune.Searched;
    best_label = label;
    best_ms = 0.5;
    source_digest = digest;
  }

let ship ?report ~arm ~substituted () =
  {
    Train.ship_arm = arm;
    ship_replayed = false;
    ship_routine = "bss_step";
    ship_report = report;
    ship_compiles = 1;
    ship_substituted = substituted;
  }

(* The [tune] object the harness renders after the arms report and [on_ship] fires (when it
   does). *)
let tune_of ~reports ?shipped () =
  let arms = H.tune_arms () in
  List.iter reports ~f:(H.collect_arm arms);
  Option.iter shipped ~f:(H.collect_ship arms);
  match H.tune_json arms with
  | Some line -> Yojson.Safe.from_string line
  | None -> failwith "bss: the harness rendered no tune object for reported arms"

let () =
  let a = searched ~label:"grid 128" ~digest:"aaaa" in
  let b = searched ~label:"tile 32x32" ~digest:"bbbb" in
  (* Arm B's crown won and its shipping compile substituted the untuned default, with nothing left
     to fall back to: the report still names B's search, so only the new pair says what ran. *)
  let t = tune_of ~reports:[ a; b ] ~shipped:(ship ~report:b ~arm:"B" ~substituted:true ()) () in
  p "a substituted ship is on the tune object"
    (Yojson.Safe.equal (member "ship_substituted" t) (`Bool true));
  p "it names the crown the substitute replaced"
    (Yojson.Safe.equal (member "substituted_from" t) (`String "tile 32x32"));
  p "the rest of the ship's identity still names the discarded crown's search"
    (Yojson.Safe.equal (member "shipped" t) (`String "B")
    && Yojson.Safe.equal (member "shipped_digest" t) (`String "bbbb"));
  (* The last resort: every completed result failed to compile, so the caller's own untuned compile
     shipped as arm A, from no search's report. *)
  let t = tune_of ~reports:[ a; b ] ~shipped:(ship ~arm:"A" ~substituted:true ()) () in
  p "a last-resort ship is substituted and names no crown"
    (Yojson.Safe.equal (member "ship_substituted" t) (`Bool true)
    && Yojson.Safe.equal (member "substituted_from" t) `Null);
  (* The ordinary case: the measured crown shipped. *)
  let t = tune_of ~reports:[ a; b ] ~shipped:(ship ~report:a ~arm:"A" ~substituted:false ()) () in
  p "a measured ship says false and names no crown"
    (Yojson.Safe.equal (member "ship_substituted" t) (`Bool false)
    && Yojson.Safe.equal (member "substituted_from" t) `Null);
  (* A caller that reports arms without wiring [on_ship]: the harness does not know, and says so
     rather than claiming the measured crown shipped. *)
  let t = tune_of ~reports:[ a; b ] () in
  p "without an on_ship record the substitution is unknown (null), not false"
    (Yojson.Safe.equal (member "ship_substituted" t) `Null
    && Yojson.Safe.equal (member "substituted_from" t) `Null)

(* gh-ocannl-1702: the same collector and renderer carry each arm's Grid record -- what the crowned
   schedule asked, the crowned routine's census, and how many ranked rivals asked for a Grid loop
   and were timed with one serial -- from the report fields that hold them. *)
let () =
  let serial_census =
    {
      Ir.C_syntax.loops =
        [
          {
            Ir.C_syntax.kernel = "bss_k";
            loop = "i0";
            site = Ir.C_syntax.Grid_serial (Grid_below_min_updates { updates = 64; floor = 4096 });
          };
        ];
      pool = true;
      pool_rendered = 0;
      serial = 1;
    }
  in
  let flagged =
    {
      (searched ~label:"grid 128" ~digest:"cccc") with
      Autotune.best_grid = Some serial_census;
      grid_serial_timed =
        [ { Autotune.grid_label = "grid 64"; grid_ms = 0.75; grid_record = serial_census } ];
    }
  in
  let plain = searched ~label:"tile 32x32" ~digest:"dddd" in
  let t = tune_of ~reports:[ flagged; plain ] () in
  let arms = Yojson.Safe.Util.to_list (member "arms" t) in
  let grid name =
    member "grid"
      (List.find_exn arms ~f:(fun a -> Yojson.Safe.equal (member "arm" a) (`String name)))
  in
  let census name = member "census" (grid name) in
  p "an arm's Grid ask is the report's schedule's, as Autotune reads it"
    (Yojson.Safe.equal
       (member "asked" (grid "A"))
       (`Bool (Autotune.schedule_asks_grid flagged.Autotune.best_schedule)));
  p "an arm's census verdict is Autotune's own on the crowned record"
    (Yojson.Safe.equal
       (member "rendered_serially" (census "A"))
       (`Bool (Autotune.grid_rendered_serially serial_census))
    && Yojson.Safe.equal (member "serial" (census "A")) (`Int 1));
  p "the census line is the renderer's"
    (Yojson.Safe.equal
       (member "summary" (census "A"))
       (`String (Ir.C_syntax.grid_summary_string serial_census)));
  p "an arm counts its ranked Grid-serial rivals"
    (Yojson.Safe.equal (member "serial_timed" (grid "A")) (`Int 1)
    && Yojson.Safe.equal (member "serial_timed" (grid "B")) (`Int 0));
  p "an arm with no crowned record carries a null census" (Yojson.Safe.equal (census "B") `Null);
  p "a harness that recorded no shipped routines carries a null shipped Grid census"
    (Yojson.Safe.equal (member "grid" t) `Null)
