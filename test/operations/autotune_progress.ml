(* gh-ocannl-1061: the tuner's progress lines, the cost record a killed search leaves behind.

   Runs [Train.tune_placements] (both arms, one beam round each, a one-flip refinement budget) with
   [--ocannl_autotune_progress=true] -- a candidate line as every candidate attempt starts -- while
   stderr is routed into a file, then reads the [autotune-progress:] lines back and checks them
   against what the search reported through its callbacks: the format every line keeps, one
   [search_start]/[search_done] pair per report and in the same order, the outcome and timed count
   of each [search_done] equal to its report's, one candidate line per attempt, each phase's last
   candidate line at [tried=N/N], and the arm and flip framing around the searches, with the device
   memory each arm line reports (gh-ocannl-1424): no arm starts beside anything an earlier one left
   in OCANNL's pools, and the driver's free bytes appear exactly where the backend can read them.
   Timings vary run to run, so only relationships between them are claimed. Everything captured is
   echoed back to stderr afterwards, so nothing the run wrote is hidden and the lines can be read
   there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* One progress line: its event and its [key=value] fields, values unquoted. [None] when the line
   does not have the documented shape. *)
let parse_line line =
  Option.map (Stderr_capture.Progress.parse line) ~f:(fun { event; fields; _ } -> (event, fields))

let field fields key = List.Assoc.find fields key ~equal:String.equal
let float_field fields key = Option.bind (field fields key) ~f:Float.of_string_opt
let int_field fields key = Option.bind (field fields key) ~f:Int.of_string_opt

(* [tried=<done>/<total>], [total] an integer or [?]. *)
let tried fields =
  Option.bind (field fields "tried") ~f:(fun t ->
      match String.lsplit2 t ~on:'/' with
      | Some (d, total) ->
          Option.map (Int.of_string_opt d) ~f:(fun d -> (d, Int.of_string_opt total))
      | None -> None)

(* The lines from each [search_start] to its [search_done]: searches do not nest. *)
let searches events =
  let rec go acc current = function
    | [] -> List.rev acc
    | (("search_start", _) as e) :: rest -> go acc [ e ] rest
    | (("search_done", _) as e) :: rest -> go (List.rev (e :: current) :: acc) [] rest
    | e :: rest -> go acc (if List.is_empty current then current else e :: current) rest
  in
  go [] [] events

let () =
  let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5) in
  let bv = Array.init 32 ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ 4; 8 ] () in
  (* A lone pointwise sum: no sketch family applies and there is no intermediate to materialize, so
     each arm is a handful of candidates (a matmul's arms seed 39 each on cc, a matmul-plus-relu's
     arm B 73 -- minutes of compiles). The flip surface is then empty, which still runs the
     refinement's framing. *)
  let%op c = a + b in
  let comp = Train.forward c in
  p "autotune_progress is on for this run" (Autotune.progress_enabled ());
  (* Every report, arms and flips alike, in arrival order. *)
  let reports = ref [] in
  let capture r = reports := r :: !reports in
  let ctx = Context.auto () in
  let (), lines =
    Stderr_capture.lines (fun () ->
        ignore
          (Train.tune_placements ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir:"" ~report:capture
             ~flip_report:capture ~inline_flips:1 ctx c comp Ir.Indexing.Empty
            : _ * _))
  in
  let progress = List.filter lines ~f:Stderr_capture.Progress.is_progress in
  let reports = List.rev !reports in
  let parsed = List.map progress ~f:parse_line in
  p_all "every progress line has the documented shape" parsed ~f:Option.is_some;
  let events = List.filter_opt parsed in
  let walls = List.filter_map events ~f:(fun (_, f) -> float_field f "wall_s") in
  let consecutive =
    match walls with
    | [] -> []
    | _ :: later -> List.zip_exn (List.take walls (List.length later)) later
  in
  p_all "wall_s never decreases from one line to the next" consecutive ~f:(fun (a, b) ->
      Float.(a <= b));
  let named e = List.filter events ~f:(fun (ev, _) -> String.equal ev e) in
  let searches = searches events in
  (* Each search's closing line against the report it closed, in order. *)
  let dones = named "search_done" in
  p "one search_start and one search_done per report"
    (List.length (named "search_start") = List.length reports
    && List.length dones = List.length reports
    && List.length searches = List.length reports);
  p_all2 "each search_done names its report's outcome and timed count" (Array.of_list dones)
    (Array.of_list reports) ~f:(fun (_, f) (r : Autotune.report) ->
      Option.equal String.equal (field f "outcome") (Some (Autotune.outcome_name r.outcome))
      && Option.equal Int.equal (int_field f "timed") (Some r.candidates_timed));
  p_all2 "each search_done retains its report's retried timing windows" (Array.of_list dones)
    (Array.of_list reports) ~f:(fun (_, f) (r : Autotune.report) ->
      Option.equal Int.equal (int_field f "timings_retried") (Some r.timings_retried));
  let candidates s = List.filter s ~f:(fun (ev, _) -> String.equal ev "candidate") in
  p_all "each search printed one candidate line per attempt it reports" searches ~f:(fun s ->
      let _, done_fields = List.last_exn s in
      Option.equal Int.equal (int_field done_fields "attempts") (Some (List.length (candidates s))));
  (* Every step that can block is named by a line written before it: the first thing a search does
     is its base compile. *)
  p_all "each search names its base compile before starting it" searches ~f:(function
    | _ :: (ev, f) :: _ ->
        String.equal ev "stage" && Option.equal String.equal (field f "stage") (Some "base_compile")
    | _ -> false);
  p_all "every candidate line names the attempt it starts" (List.concat_map searches ~f:candidates)
    ~f:(fun (_, f) -> Option.exists (field f "attempt") ~f:(Fn.non String.is_empty));
  p_all "each search's candidate compile and timing seconds fit within its elapsed time" dones
    ~f:(fun (_, f) ->
      match (float_field f "compile_s", float_field f "timing_s", float_field f "elapsed_s") with
      | Some c, Some t, Some e -> Float.(c + t <= e + 0.2)
      | _ -> false);
  (* gh-ocannl-1314: one [compiled] line as each compile returns, its phases inside its wall. *)
  let compiled s = List.filter s ~f:(fun (ev, _) -> String.equal ev "compiled") in
  p_all "each candidate line is answered by one compiled line for the same attempt" searches
    ~f:(fun s ->
      let attempts_started = List.filter_map (candidates s) ~f:(fun (_, f) -> field f "attempt") in
      let attempts_compiled =
        List.filter_map (compiled s) ~f:(fun (_, f) ->
            if Option.equal String.equal (field f "stage") (Some "candidate") then field f "attempt"
            else None)
      in
      List.equal String.equal attempts_started attempts_compiled);
  p_all "each search's base compile has its compiled line, attempt baseline" searches ~f:(fun s ->
      List.count (compiled s) ~f:(fun (_, f) ->
          Option.equal String.equal (field f "stage") (Some "base_compile")
          && Option.equal String.equal (field f "attempt") (Some "baseline"))
      = 1);
  let all_compiled = compiled events in
  p_exists "some compile reached link" all_compiled ~f:(fun (_, f) ->
      Option.equal String.equal (field f "verdict") (Some "compiled"));
  p_all "every compiled line has a known verdict, and its phases fit within its wall" all_compiled
    ~f:(fun (_, f) ->
      let phase k = float_field f k in
      let device =
        match field f "device_s" with
        | Some "none" -> Some 0.
        | Some v -> Float.of_string_opt v
        | None -> None
      in
      List.mem
        [ "compiled"; "dedup"; "refused"; "failed" ]
        (Option.value (field f "verdict") ~default:"")
        ~equal:String.equal
      &&
      match
        ( float_field f "took_s",
          Option.all
            [ phase "lower_s"; phase "transform_s"; phase "codegen_s"; device; phase "link_s" ] )
      with
      | Some wall, Some phases ->
          List.for_all phases ~f:(fun x -> Float.(x >= -0.001))
          && Float.(List.sum (module Float) phases ~f:Fn.id <= wall + 0.01)
      | _ -> false);
  p_all "a compile renders at least the kernels it compiles, and one that reached link renders some"
    all_compiled ~f:(fun (_, f) ->
      match (int_field f "kernels", int_field f "kernels_compiled") with
      | Some k, Some c ->
          c >= 0 && c <= k
          && ((not (Option.equal String.equal (field f "verdict") (Some "compiled"))) || k >= 1)
      | _ -> false);
  p_all "a candidate decided before codegen (dedup or refused) rendered no kernel" all_compiled
    ~f:(fun (_, f) ->
      (not
         (List.mem [ "dedup"; "refused" ]
            (Option.value (field f "verdict") ~default:"")
            ~equal:String.equal))
      || Option.equal Int.equal (int_field f "kernels") (Some 0));
  p_all2 "each report's compile_cost counts the kernels of its search's compiled lines"
    (Array.of_list searches) (Array.of_list reports) ~f:(fun s (r : Autotune.report) ->
      List.sum
        (module Int)
        (compiled s)
        ~f:(fun (_, f) -> Option.value (int_field f "kernels") ~default:(-1_000_000))
      = r.compile_cost.kernels);
  (* Each phase line announces a total; the phase's candidate lines count up to it. *)
  let phases =
    List.concat_map searches ~f:(fun s ->
        List.filter_mapi s ~f:(fun i (ev, f) ->
            if String.equal ev "phase" then
              let name = field f "phase" in
              (* A candidate's [compiled] line follows it inside the phase. *)
              let rest =
                List.filter
                  (List.drop s (i + 1))
                  ~f:(fun (ev, _) -> not (String.equal ev "compiled"))
              in
              let in_phase =
                List.take_while rest ~f:(fun (ev, g) ->
                    String.equal ev "candidate" && Option.equal String.equal (field g "phase") name)
              in
              Some (int_field f "candidates", List.filter_map in_phase ~f:(fun (_, g) -> tried g))
            else None))
  in
  p_exists "a beam round ran" (List.concat_map searches ~f:Fn.id) ~f:(fun (ev, f) ->
      String.equal ev "phase"
      && Option.value_map (field f "phase") ~default:false ~f:(String.is_prefix ~prefix:"round"));
  p_all "each phase's candidate lines count 1..N against its announced total N" phases
    ~f:(fun (total, tries) ->
      match total with
      | None -> false
      | Some n ->
          List.equal
            (fun (d, t) (d', t') -> d = d' && Option.equal Int.equal t t')
            tries
            (List.init n ~f:(fun i -> (i + 1, Some n))));
  (* The placement framing: an arm_start and arm_done around every search, flips counted. *)
  let arm_starts = named "arm_start" and arm_dones = named "arm_done" in
  p "one arm_start and one arm_done per report"
    (List.length arm_starts = List.length reports && List.length arm_dones = List.length reports);
  p_all "every arm_done says whether its arm succeeded" arm_dones ~f:(fun (_, f) ->
      List.mem [ "ok"; "failed"; "abandoned" ]
        (Option.value (field f "result") ~default:"")
        ~equal:String.equal);
  let flip_arms = List.filter arm_starts ~f:(fun (_, f) -> Option.is_some (field f "flip")) in
  p "the flip surface's lowering is named before the refinement starts"
    (let rec before = function
       | (ev, f) :: (ev', _) :: _
         when String.equal ev "stage"
              && Option.equal String.equal (field f "stage") (Some "flip_surface")
              && String.equal ev' "flips_start" ->
           true
       | _ :: rest -> before rest
       | [] -> false
     in
     before events);
  p "the flip refinement is framed by one flips_start and one flips_done"
    (List.length (named "flips_start") = 1 && List.length (named "flips_done") = 1);
  p "flips_done's measured count equals the flip searches' arm_start lines"
    (match named "flips_done" with
    | [ (_, f) ] -> Option.equal Int.equal (int_field f "measured") (Some (List.length flip_arms))
    | _ -> false);
  (* gh-ocannl-1424: the memory fields. Working bytes are the census's, deterministic on every
     backend; the driver's free bytes vary with everything else on the device, so only their
     presence is claimed, against the query the lines are read through. *)
  let placements_starts = named "placements_start" in
  let reads_driver = Option.is_some (Context.get_free_and_total_memory ctx) in
  let driver_field f key =
    match field f key with
    | Some "none" -> not reads_driver
    | Some v -> reads_driver && Option.is_some (Int.of_string_opt v)
    | None -> false
  in
  p "one placements_start line, ahead of every arm"
    (List.length placements_starts = 1
    &&
    match
      List.find events ~f:(fun (ev, _) ->
          String.equal ev "placements_start" || String.equal ev "arm_start")
    with
    | Some (ev, _) -> String.equal ev "placements_start"
    | None -> false);
  p_all
    "placements_start reports working bytes and the driver's free and total bytes where it has them"
    placements_starts ~f:(fun (_, f) ->
      Option.is_some (int_field f "working_bytes")
      && driver_field f "device_free" && driver_field f "device_total");
  p_all "every arm line reports working bytes and the driver's free bytes where it has them"
    (arm_starts @ arm_dones) ~f:(fun (_, f) ->
      Option.is_some (int_field f "working_bytes") && driver_field f "device_free");
  let base_working =
    Option.bind (List.hd placements_starts) ~f:(fun (_, f) -> int_field f "working_bytes")
  in
  p_all "every arm starts with the working bytes the call started with" arm_starts ~f:(fun (_, f) ->
      Option.is_some base_working
      && Option.equal Int.equal (int_field f "working_bytes") base_working);
  List.iter (placements_starts @ arm_starts) ~f:(fun (ev, f) ->
      Stdio.eprintf "%s %s: working_bytes=%s device_free=%s (not part of the golden)\n%!" ev
        (Option.value (field f "arm") ~default:"-")
        (Option.value (field f "working_bytes") ~default:"?")
        (Option.value (field f "device_free") ~default:"?"))
