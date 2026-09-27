(* gh-ocannl-1061: the tuner's progress lines, the cost record a killed search leaves behind.

   Runs [Train.tune_placements] (both arms, one beam round each, a one-flip refinement budget) with
   [--ocannl_autotune_progress=0] -- a candidate line as every candidate attempt starts -- while
   stderr is routed into a file, then reads the [autotune-progress:] lines back and checks them
   against what the search reported through its callbacks: the format every line keeps, one
   [search_start]/[search_done] pair per report and in the same order, the outcome and timed count
   of each [search_done] equal to its report's, one candidate line per attempt, each phase's last
   candidate line at [tried=N/N], and the arm and flip framing around the searches. Timings vary run
   to run, so only relationships between them are claimed. Everything captured is echoed back to
   stderr afterwards, so nothing the run wrote is hidden and the lines can be read there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* One progress line: its event and its [key=value] fields, values unquoted. [None] when the line
   does not have the documented shape. *)
let parse_line line =
  match String.chop_prefix line ~prefix:"autotune-progress: " with
  | None -> None
  | Some rest -> (
      let len = String.length rest in
      let fields = ref [] and ok = ref true and i = ref 0 in
      while !ok && !i < len do
        match String.index_from rest !i '=' with
        | None -> ok := false
        | Some eq ->
            let key = String.sub rest ~pos:!i ~len:(eq - !i) in
            if String.is_empty key || String.exists key ~f:(fun c -> Char.equal c ' ') then
              ok := false
            else if eq + 1 < len && Char.equal rest.[eq + 1] '"' then (
              (* An OCaml [%S] string: ends at the first unescaped quote. *)
              let j = ref (eq + 2)
              and buf = Buffer.create 16 in
              while !j < len && not (Char.equal rest.[!j] '"') do
                if Char.equal rest.[!j] '\\' && !j + 1 < len then (
                  Buffer.add_char buf rest.[!j + 1];
                  j := !j + 2)
                else (
                  Buffer.add_char buf rest.[!j];
                  Int.incr j)
              done;
              if !j >= len then ok := false
              else (
                fields := (key, Buffer.contents buf) :: !fields;
                i := !j + 2))
            else
              let stop = Option.value (String.index_from rest eq ' ') ~default:len in
              let value = String.sub rest ~pos:(eq + 1) ~len:(stop - eq - 1) in
              if String.is_empty value then ok := false
              else (
                fields := (key, value) :: !fields;
                i := stop + 1)
      done;
      match List.rev !fields with
      | ("wall_s", w) :: ("event", e) :: rest when !ok && Option.is_some (Float.of_string_opt w) ->
          Some (e, ("wall_s", w) :: rest)
      | _ -> None)

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
  let file = Stdlib.Filename.temp_file "autotune_progress" ".stderr" in
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
  (match
     Train.tune_placements ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir:"" ~report:capture
       ~flip_report:capture ~inline_flips:1 (Context.auto ()) c comp Ir.Indexing.Empty
   with
  | _ -> restore ()
  | exception exn ->
      restore ();
      raise exn);
  let text = Stdio.In_channel.read_all file in
  Stdlib.Sys.remove file;
  let lines = String.split_lines text in
  List.iter lines ~f:(fun l -> Stdio.eprintf "%s\n" l);
  let progress = List.filter lines ~f:(String.is_prefix ~prefix:"autotune-progress:") in
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
  let candidates s = List.filter s ~f:(fun (ev, _) -> String.equal ev "candidate") in
  (* With a 0 s interval every attempt prints. *)
  p_all "each search printed one candidate line per attempt it reports" searches ~f:(fun s ->
      let _, done_fields = List.last_exn s in
      Option.equal Int.equal (int_field done_fields "attempts") (Some (List.length (candidates s))));
  p_all "every candidate line names the attempt it starts" (List.concat_map searches ~f:candidates)
    ~f:(fun (_, f) -> Option.exists (field f "attempt") ~f:(Fn.non String.is_empty));
  p_all "each search's candidate compile and timing seconds fit within its elapsed time" dones
    ~f:(fun (_, f) ->
      match (float_field f "compile_s", float_field f "timing_s", float_field f "elapsed_s") with
      | Some c, Some t, Some e -> Float.(c + t <= e + 0.2)
      | _ -> false);
  (* Each phase line announces a total; the phase's candidate lines count up to it. *)
  let phases =
    List.concat_map searches ~f:(fun s ->
        List.filter_mapi s ~f:(fun i (ev, f) ->
            if String.equal ev "phase" then
              let name = field f "phase" in
              let rest = List.drop s (i + 1) in
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
      List.mem [ "ok"; "failed" ] (Option.value (field f "result") ~default:"") ~equal:String.equal);
  let flip_arms = List.filter arm_starts ~f:(fun (_, f) -> Option.is_some (field f "flip")) in
  p "the flip refinement is framed by one flips_start and one flips_done"
    (List.length (named "flips_start") = 1 && List.length (named "flips_done") = 1);
  p "flips_done's measured count equals the flip searches' arm_start lines"
    (match named "flips_done" with
    | [ (_, f) ] -> Option.equal Int.equal (int_field f "measured") (Some (List.length flip_arms))
    | _ -> false)
