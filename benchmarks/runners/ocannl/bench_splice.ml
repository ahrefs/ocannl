(* The splice-segments A/B (gh-ocannl-1277): the measurement for a schedule effect smaller than the
   variance between two independent searches.

   Comparing two separately searched step schedules attributes to the treatment everything the two
   searches crowned differently, and on gpt2_mini that is several percent of a step in either
   direction (gh-ocannl-1165: a q/k/v saving of ~1% of the step read +1.3% under one profile and
   -1.8% under the other). Splicing removes that variance by construction: both arms replay the
   CONTROL's schedule for every segment except the target ones, which come from each arm's own
   crowned winner. The control arm is therefore the control's cache as it is; the treatment arm is
   the control's cache with the target segments' schedules taken from the treatment's.

   Entries pair by cache key, which carries the routine's source digest: the method applies to a
   treatment that changes what the search finds, not the program it searches over. Within a pair,
   segments align by position (as a replay applies them, {!Ir.Schedule_cache.saved_segment}), so the
   splice is refused -- never repaired -- when the two winners' segmentations differ anywhere in an
   entry that holds a target: splicing by position into a different cut would replay one arm's
   schedule against another arm's segment.

   [bench_gpt] exposes this as its splice mode ({!from_env}); a runner adopts it by calling
   {!from_env} before compiling and running its compile inside {!replay_guarded}. *)

open Base
module SC = Ir.Schedule_cache

(** A target: every segment whose pre-schedule structural digest starts with a hex prefix, in every
    entry present in both caches; or one segment by position, in the one entry present in both
    caches that is fissioned in both, or in the one whose source digest starts with [entry]. *)
type target = Seg_digest of string | Seg_index of { entry : string option; index : int }

type arm = Control | Treatment

let min_digest_prefix = 6

let target_to_string = function
  | Seg_digest p -> p
  | Seg_index { entry; index } -> Printf.sprintf "%s#%d" (Option.value entry ~default:"") index

let is_hex s = (not (String.is_empty s)) && String.for_all s ~f:Char.is_hex_digit

let parse_target item =
  match String.lsplit2 item ~on:'#' with
  | Some (entry, idx) -> (
      match Int.of_string_opt idx with
      | Some index when index >= 0 ->
          if String.is_empty entry then Ok (Seg_index { entry = None; index })
          else if is_hex entry then Ok (Seg_index { entry = Some (String.lowercase entry); index })
          else
            Error (Printf.sprintf "target %S: the entry before '#' is not a hex digest prefix" item)
      | _ ->
          Error (Printf.sprintf "target %S: the index after '#' is not a non-negative integer" item)
      )
  | None ->
      if is_hex item && String.length item >= min_digest_prefix then
        Ok (Seg_digest (String.lowercase item))
      else
        Error
          (Printf.sprintf
             "target %S is neither a segment digest prefix (at least %d hex digits) nor \
              [<entry>]#<index>"
             item min_digest_prefix)

let parse_targets s =
  let items =
    String.split s ~on:',' |> List.map ~f:String.strip |> List.filter ~f:(Fn.non String.is_empty)
  in
  if List.is_empty items then Error "the target list is empty"
  else Result.all (List.map items ~f:parse_target)

(** {1 Reading the two caches} *)

let placements_prefix = "placements-"
let abandonment_prefix = "abandonment-"

type contents = {
  schedules : (string * SC.entry) list;  (** Key and entry, in filename order. *)
  placements : (string * SC.placement_entry) list;
  undecodable : string list;
      (** Keys whose entry did not decode at an accepted version: a stale entry, or an older version
          the caller did not adopt. *)
  abandonments : int;
      (** Abandoned-search records: evidence for a search, which a replay runs none of. *)
}

(* Read through {!SC.read_entry_file}, never {!SC.lookup}: the inputs are another binary's caches,
   and a lookup reads a directory whose regime stamp is older than this build's as all misses
   (gh-ocannl-1439) -- the very schedules the A/B is about to replay. An adopted control entry is
   re-stamped at the current version, which is what [write_arm] stores. *)
let read_dir ?adopt_version dir =
  match Stdlib.Sys.readdir dir with
  | exception Stdlib.Sys_error msg -> Error msg
  | names ->
      let files =
        Array.to_list names
        |> List.filter ~f:(fun name ->
            String.is_suffix name ~suffix:".sexp" && not (Utils.Atomic_file.is_staging_file name))
        |> List.sort ~compare:String.compare
      in
      let decode of_sexp payload = Option.try_with (fun () -> of_sexp payload) in
      let init = { schedules = []; placements = []; undecodable = []; abandonments = 0 } in
      let c =
        List.fold files ~init ~f:(fun c name ->
            match SC.read_entry_file (Stdlib.Filename.concat dir name) with
            | None -> { c with undecodable = name :: c.undecodable }
            | Some (key, payload) -> (
                if String.is_prefix key ~prefix:abandonment_prefix then
                  { c with abandonments = c.abandonments + 1 }
                else if String.is_prefix key ~prefix:placements_prefix then
                  match decode SC.placement_entry_of_sexp payload with
                  | Some p when p.SC.version = SC.placement_entry_version ->
                      { c with placements = (key, p) :: c.placements }
                  | _ -> { c with undecodable = key :: c.undecodable }
                else
                  match decode SC.entry_of_sexp payload with
                  | Some e when e.SC.version = SC.entry_version ->
                      { c with schedules = (key, e) :: c.schedules }
                  | Some e when Option.equal Int.equal (Some e.SC.version) adopt_version ->
                      {
                        c with
                        schedules = (key, { e with SC.version = SC.entry_version }) :: c.schedules;
                      }
                  | _ -> { c with undecodable = key :: c.undecodable }))
      in
      Ok
        {
          c with
          schedules = List.rev c.schedules;
          placements = List.rev c.placements;
          undecodable = List.rev c.undecodable;
        }

(** {1 The plan} *)

type row = {
  index : int;
  seg : SC.saved_segment;  (** The control's segment. *)
  treatment_saved : SC.saved_schedule option;
      (** The treatment's schedule for the segment at the same position, when the treatment has one.
      *)
  targeted : bool;  (** Selected by a target, in an entry that is spliced. *)
  group : int;  (** How many segments of this entry (control side) share this segment's digest. *)
}

type pair = {
  key : string;
  control : SC.entry;
  treatment : SC.entry;
  rows : row list option;  (** [None] unless the control's entry is fissioned. *)
  ships : bool;
      (** Whether the control's placement decisions ship this routine: its source digest is one of
          their [outcome_digest]s, the lowering the decision replays to. With no decision in the
          control cache every entry counts as shipping. *)
  splice : [ `Untargeted | `Spliced | `Left of string ];
      (** [`Left reason]: a target selects segments of it that cannot be spliced (the reason), and
          since the routine does not ship it is left as the control's rather than refused. *)
}

type plan = {
  control_dir : string;
  treatment_dir : string;
  adopt_version : int option;
  targets : (target * int) list;  (** Each target with the number of segments it splices. *)
  pairs : pair list;
  control_only : (string * SC.entry) list;
      (** Entries only the control has: copied into both arms unchanged. *)
  treatment_only : string list;  (** Keys only the treatment has: in neither arm. *)
  placements : (string * SC.placement_entry) list;  (** The control's, copied into both arms. *)
  control_undecodable : string list;
  treatment_undecodable : string list;
  abandonments : int;
}

let entry_id (e : SC.entry) = String.prefix e.SC.source_digest 12

let kind_name = function
  | `Normal -> "normal"
  | `Zeros -> "zeros"
  | `Solo -> "solo"
  | `Unfolded -> "unfolded"

let fissioned (e : SC.entry) = Option.is_some e.SC.segments
let digest_matches prefix (s : SC.saved_segment) = String.is_prefix s.SC.seg_digest ~prefix

(* The positions [target] selects in [pair], or an error when it names this entry and cannot. *)
let positions ~index_entry target (c : SC.entry) (t : SC.entry) =
  let segs (e : SC.entry) = Option.value e.SC.segments ~default:[] in
  match target with
  | Seg_digest prefix ->
      let hits e =
        List.filter_mapi (segs e) ~f:(fun i s -> Option.some_if (digest_matches prefix s) i)
      in
      List.dedup_and_sort ~compare:Int.compare (hits c @ hits t)
  | Seg_index { index; _ } ->
      if Option.equal String.equal index_entry (Some c.SC.source_digest) then [ index ] else []

let make_plan ?adopt_version ~control ~treatment ~targets () =
  let ( let* ) = Result.( >>= ) in
  let* cc = read_dir ?adopt_version control in
  let* tc = read_dir treatment in
  let treatment_tbl = Map.of_alist_reduce (module String) tc.schedules ~f:(fun a _ -> a) in
  let raw_pairs, control_only =
    List.partition_map cc.schedules ~f:(fun (key, c) ->
        match Map.find treatment_tbl key with Some t -> First (key, c, t) | None -> Second (key, c))
  in
  let control_keys = Set.of_list (module String) (List.map cc.schedules ~f:fst) in
  let treatment_only =
    List.filter_map tc.schedules ~f:(fun (k, _) -> Option.some_if (not (Set.mem control_keys k)) k)
  in
  let* () =
    match
      List.find raw_pairs ~f:(fun (_, (c : SC.entry), (t : SC.entry)) ->
          not (String.equal c.SC.source_digest t.SC.source_digest))
    with
    | Some (key, _, _) ->
        Error (Printf.sprintf "entry %s: the two caches hold it for different source digests" key)
    | None -> Ok ()
  in
  (* Index targets name one entry: the one fissioned in both caches, or the one their prefix
     picks. *)
  let both_fissioned = List.filter raw_pairs ~f:(fun (_, c, t) -> fissioned c && fissioned t) in
  let resolve_entry = function
    | Seg_digest _ -> Ok None
    | Seg_index { entry; index } -> (
        let candidates =
          List.filter both_fissioned ~f:(fun (_, (c : SC.entry), _) ->
              Option.value_map entry ~default:true ~f:(fun prefix ->
                  String.is_prefix c.SC.source_digest ~prefix))
        in
        let ids () =
          String.concat ~sep:", " (List.map both_fissioned ~f:(fun (_, c, _) -> entry_id c))
        in
        match candidates with
        | [ (_, c, _) ] ->
            let n = List.length (Option.value c.SC.segments ~default:[]) in
            if index < n then Ok (Some c.SC.source_digest)
            else
              Error
                (Printf.sprintf "target %s: entry %s has %d segments"
                   (target_to_string (Seg_index { entry; index }))
                   (entry_id c) n)
        | [] ->
            Error
              (Printf.sprintf "target %s names no entry fissioned in both caches (those are: %s)"
                 (target_to_string (Seg_index { entry; index }))
                 (if List.is_empty both_fissioned then "none" else ids ()))
        | _ ->
            Error
              (Printf.sprintf
                 "target %s is ambiguous: %d entries are fissioned in both caches (%s); name one \
                  as <entry>#<index>"
                 (target_to_string (Seg_index { entry; index }))
                 (List.length candidates) (ids ())))
  in
  let* resolved =
    Result.all (List.map targets ~f:(fun tg -> Result.map (resolve_entry tg) ~f:(fun e -> (tg, e))))
  in
  let pair_positions c t =
    List.concat_map resolved ~f:(fun (tg, index_entry) -> positions ~index_entry tg c t)
    |> List.dedup_and_sort ~compare:Int.compare
  in
  (* Refusals for an entry holding a target: the segmentation, the codes cut, and any whole-routine
     prelude must agree, since everything but the targets' schedules comes from the control. *)
  let check_aligned key (c : SC.entry) (t : SC.entry) =
    let id = entry_id c in
    match (c.SC.segments, t.SC.segments) with
    | None, _ | _, None ->
        Error
          (Printf.sprintf
             "entry %s (%s): a target selects segments of it, but the %s winner is whole-routine, \
              so it has no segments to splice"
             id key
             (if fissioned c then "treatment's" else "control's"))
    | Some cs, Some ts ->
        if not (SC.equal_saved_schedule c.SC.saved t.SC.saved) then
          Error
            (Printf.sprintf
               "entry %s: the two winners' whole-routine preludes differ (a split-reduce winner's \
                [saved]); the prelude schedules the whole routine, not a segment"
               id)
        else if List.length cs <> List.length ts then
          Error
            (Printf.sprintf
               "entry %s: the segmentations differ: %d control segments, %d treatment segments" id
               (List.length cs) (List.length ts))
        else
          List.fold_until (List.zip_exn cs ts) ~init:0
            ~f:(fun i ((cs : SC.saved_segment), (ts : SC.saved_segment)) ->
              if
                Poly.equal cs.SC.seg_kind ts.SC.seg_kind
                && cs.SC.seg_units = ts.SC.seg_units
                && String.equal cs.SC.seg_digest ts.SC.seg_digest
              then Continue (i + 1)
              else
                Stop
                  (Error
                     (Printf.sprintf
                        "entry %s: the segmentations differ at segment %d: control %s/%d units/%s, \
                         treatment %s/%d units/%s"
                        id i (kind_name cs.SC.seg_kind) cs.SC.seg_units cs.SC.seg_digest
                        (kind_name ts.SC.seg_kind) ts.SC.seg_units ts.SC.seg_digest)))
            ~finish:(fun _ -> Ok ())
  in
  (* The routines the control's placement decisions replay to. An entry outside them is replayed
     only when the placement store is bypassed, so a target there that cannot be spliced -- the
     other placement arm's winner cut differently, say -- is left as the control's and reported
     rather than refused; in a shipping routine it is refused. *)
  let shipped =
    List.map cc.placements ~f:(fun (_, (p : SC.placement_entry)) -> p.SC.outcome_digest)
  in
  let ships (c : SC.entry) =
    List.is_empty shipped || List.mem shipped c.SC.source_digest ~equal:String.equal
  in
  let* pairs =
    Result.all
      (List.map raw_pairs ~f:(fun (key, c, t) ->
           let hits = pair_positions c t in
           let ships = ships c in
           let* splice =
             if List.is_empty hits then Ok `Untargeted
             else
               match check_aligned key c t with
               | Ok () -> Ok `Spliced
               | Error e when not ships -> Ok (`Left e)
               | Error e -> Error e
           in
           let spliced = Poly.equal splice `Spliced in
           let rows =
             Option.map c.SC.segments ~f:(fun cs ->
                 let tsegs = Option.value t.SC.segments ~default:[] in
                 List.mapi cs ~f:(fun i (s : SC.saved_segment) ->
                     {
                       index = i;
                       seg = s;
                       treatment_saved =
                         Option.map (List.nth tsegs i) ~f:(fun (ts : SC.saved_segment) ->
                             ts.SC.seg_saved);
                       targeted = spliced && List.mem hits i ~equal:Int.equal;
                       group =
                         List.count cs ~f:(fun (s' : SC.saved_segment) ->
                             String.equal s'.SC.seg_digest s.SC.seg_digest);
                     }))
           in
           Ok { key; control = c; treatment = t; rows; ships; splice }))
  in
  let counts =
    List.map resolved ~f:(fun (tg, index_entry) ->
        ( tg,
          List.sum
            (module Int)
            pairs
            ~f:(fun p ->
              if Poly.equal p.splice `Spliced then
                List.length (positions ~index_entry tg p.control p.treatment)
              else 0) ))
  in
  let* () =
    match List.find counts ~f:(fun (_, n) -> n = 0) with
    | Some (tg, _) ->
        Error
          (Printf.sprintf
             "target %s selects no segment that can be spliced, of any entry present in both \
              caches: a splice of it would replay the control in both arms%s"
             (target_to_string tg)
             (match (cc.undecodable, tc.undecodable) with
             | [], [] -> ""
             | c, t ->
                 Printf.sprintf
                   " (%d control and %d treatment entries did not decode at this build's entry \
                    version: stale, or an older version to adopt with BENCH_SPLICE_ADOPT_VERSION)"
                   (List.length c) (List.length t)))
    | None -> Ok ()
  in
  let* () =
    if
      List.is_empty targets
      || List.exists pairs ~f:(fun p -> p.ships && Poly.equal p.splice `Spliced)
    then Ok ()
    else
      Error
        "the targets splice no routine the control's placement decisions ship: the treatment arm \
         would replay the control wherever it is timed"
  in
  Ok
    {
      control_dir = control;
      treatment_dir = treatment;
      adopt_version;
      targets = counts;
      pairs;
      control_only;
      treatment_only;
      placements = cc.placements;
      control_undecodable = cc.undecodable;
      treatment_undecodable = tc.undecodable;
      abandonments = cc.abandonments;
    }

(** {1 The arms} *)

(** What [arm] replays for one of the control's entries. *)
let arm_entry plan arm key (c : SC.entry) =
  match arm with
  | Control -> c
  | Treatment -> (
      match
        List.find plan.pairs ~f:(fun p -> String.equal p.key key && Poly.equal p.splice `Spliced)
      with
      | None -> c
      | Some p ->
          let ts = Option.value_exn p.treatment.SC.segments in
          let rows = Option.value_exn p.rows in
          (* The hybrid was never timed: the control winner's measurements describe another
             schedule, so they are dropped -- [best_ms] to the never-measured [infinity] a result
             line emits as null. The unscheduled baseline's time is the routine's, and stays. *)
          {
            c with
            SC.best_ms = Float.infinity;
            default_ms = None;
            default_fingerprint = None;
            mma_best_ms = None;
            best_steps = None;
            budget_cut = false;
            segments =
              Some
                (List.map2_exn rows ts ~f:(fun r (t : SC.saved_segment) ->
                     if r.targeted then t else r.seg));
          })

let spliced_segments plan =
  List.sum
    (module Int)
    plan.pairs
    ~f:(fun p -> List.count (Option.value p.rows ~default:[]) ~f:(fun r -> r.targeted))

let differing_targets plan =
  List.sum
    (module Int)
    plan.pairs
    ~f:(fun p ->
      List.count (Option.value p.rows ~default:[]) ~f:(fun r ->
          r.targeted
          && not
               (Option.value_map r.treatment_saved ~default:false
                  ~f:(SC.equal_saved_schedule r.seg.SC.seg_saved))))

(** Writes [arm]'s cache into [cache_dir] -- by default the configured [autotune_cache_dir], the
    directory the runner's tuning call then replays ({!Autotune.resolve_cache_dir}) -- which must be
    absent or an empty directory: every schedule entry the control holds (the targeted ones spliced
    for the treatment arm) and the control's placement decisions, so both arms ship the control's
    placement. Refused when nothing is spliced, and when the cache refused a store (read back from
    {!SC.recording_cache_io}, since {!SC.store} absorbs refusals). Returns the number of entries
    written. *)
let write_arm ?cache_dir plan ~arm () =
  let cache_dir = Autotune.resolve_cache_dir ?cache_dir ~search:true () in
  if not (List.exists plan.pairs ~f:(fun p -> Poly.equal p.splice `Spliced)) then
    Error "nothing is targeted: both arms would replay the control"
  else if
    Stdlib.Sys.file_exists cache_dir
    && ((not (Stdlib.Sys.is_directory cache_dir))
       || not (Array.is_empty (Stdlib.Sys.readdir cache_dir)))
  then Error (cache_dir ^ ": the arm's cache directory must be absent or empty")
  else
    let n, io =
      SC.recording_cache_io (fun () ->
          List.iter plan.control_only ~f:(fun (key, c) -> SC.store ~dir:cache_dir ~key:(Some key) c);
          List.iter plan.pairs ~f:(fun p ->
              SC.store ~dir:cache_dir ~key:(Some p.key) (arm_entry plan arm p.key p.control));
          List.iter plan.placements ~f:(fun (key, pl) ->
              SC.store_placements ~dir:cache_dir ~key:(Some key) pl);
          List.length plan.control_only + List.length plan.pairs + List.length plan.placements)
    in
    match List.find io ~f:(fun (r : SC.cache_io) -> Option.is_some r.SC.refusal) with
    | Some r ->
        Error
          (Printf.sprintf "storing %s into %s was refused: %s" r.SC.key cache_dir
             (Option.value_exn r.SC.refusal))
    | None -> Ok n

(** {1 The listing} *)

(* A schedule's operations by constructor, with repeats counted: enough to tell two winners' choices
   apart at a glance (a [Coalesce] in one arm only, a [Stage] the other lacks). *)
let ops_summary (saved : SC.saved_schedule) =
  if List.is_empty saved then "-"
  else
    let names =
      List.map saved ~f:(fun op ->
          match SC.sexp_of_saved_optop op with
          | Sexp.Atom a | Sexp.List (Sexp.Atom a :: _) -> a
          | Sexp.List _ -> "?")
    in
    let rec runs acc = function
      | [] -> List.rev acc
      | n :: rest ->
          let same, rest = List.split_while rest ~f:(String.equal n) in
          let k = 1 + List.length same in
          runs ((if k = 1 then n else Printf.sprintf "%s*%d" n k) :: acc) rest
    in
    (* The constructors alone cannot tell two factors apart; the short hash of the whole schedule
       can. *)
    let hash =
      String.prefix
        (Stdlib.Digest.to_hex
           (Stdlib.Digest.string (Sexp.to_string (SC.sexp_of_saved_schedule saved))))
        6
    in
    String.concat ~sep:" " (runs [] names) ^ " #" ^ hash

(** The plan as lines for stderr: every entry present in both caches, with a row per segment of a
    fissioned one -- position, kind, units, digest, how many segments share it, whether the two
    winners' schedules agree, whether a target selects it, and both schedules' operations. *)
let describe plan =
  let line = Printf.sprintf in
  let header =
    [
      line "splice: control %s, treatment %s%s" plan.control_dir plan.treatment_dir
        (Option.value_map plan.adopt_version ~default:"" ~f:(fun v ->
             line " (control entries of version %d adopted as %d)" v SC.entry_version));
    ]
  in
  let ship_note p =
    if List.is_empty plan.placements then ""
    else if p.ships then " [ships]"
    else " [does not ship under the control's placement decisions]"
  in
  let left p =
    match p.splice with
    | `Left reason -> [ line "splice: NOT spliced: %s" reason ]
    | `Untargeted | `Spliced -> []
  in
  let pair_lines p =
    let id = entry_id p.control ^ ship_note p in
    left p
    @
    match p.rows with
    | None ->
        [
          line "splice: entry %s: whole-routine in the control%s, %s" id
            (if fissioned p.treatment then " (fissioned in the treatment)" else "")
            (if
               SC.equal_saved_schedule p.control.SC.saved p.treatment.SC.saved
               && Option.is_none p.treatment.SC.segments
             then "same schedule in both"
             else "schedules differ");
        ]
    | Some rows ->
        let differ =
          List.count rows ~f:(fun r ->
              not
                (Option.value_map r.treatment_saved ~default:false
                   ~f:(SC.equal_saved_schedule r.seg.SC.seg_saved)))
        in
        line "splice: entry %s: %d segments (treatment: %s), %d schedules differ, %d targeted" id
          (List.length rows)
          (match p.treatment.SC.segments with
          | None -> "whole-routine"
          | Some ts -> line "%d segments" (List.length ts))
          differ
          (List.count rows ~f:(fun r -> r.targeted))
        :: List.map rows ~f:(fun r ->
            let same =
              Option.value_map r.treatment_saved ~default:false
                ~f:(SC.equal_saved_schedule r.seg.SC.seg_saved)
            in
            line "  %3d %-6s %3du %s x%-2d %s %s  %s%s" r.index (kind_name r.seg.SC.seg_kind)
              r.seg.SC.seg_units
              (String.prefix r.seg.SC.seg_digest 12)
              r.group
              (if same then "same" else "DIFF")
              (if r.targeted then "T" else ".")
              (ops_summary r.seg.SC.seg_saved)
              (if same then ""
               else
                 " | " ^ Option.value_map r.treatment_saved ~default:"(no segment)" ~f:ops_summary))
  in
  let footer =
    [
      line
        "splice: %d entries in both caches, %d only in the control (copied into both arms), %d \
         only in the treatment (in neither arm), %d placement decisions (the control's, in both \
         arms)"
        (List.length plan.pairs) (List.length plan.control_only) (List.length plan.treatment_only)
        (List.length plan.placements);
    ]
    @ (if List.is_empty plan.control_undecodable && List.is_empty plan.treatment_undecodable then []
       else
         [
           line
             "splice: undecodable entries (stale, or an unadopted version): %d control, %d \
              treatment"
             (List.length plan.control_undecodable)
             (List.length plan.treatment_undecodable);
         ])
    @ List.map plan.targets ~f:(fun (tg, n) ->
        line "splice: target %s selects %d segments" (target_to_string tg) n)
    @
    if List.is_empty plan.targets then []
    else
      let n = spliced_segments plan and d = differing_targets plan in
      [
        line "splice: %d targeted segments, %d with schedules that differ between the arms%s" n d
          (if d = 0 then ": the arms are identical, a null A/B" else "");
      ]
  in
  header @ List.concat_map plan.pairs ~f:pair_lines @ footer

(** {1 The runner mode} *)

let env name =
  Option.bind (Stdlib.Sys.getenv_opt name) ~f:(fun v -> Option.some_if (not (String.is_empty v)) v)

let fail fmt =
  Printf.ksprintf
    (fun s ->
      Stdio.eprintf "splice: %s\n%!" s;
      Stdlib.exit 2)
    fmt

(** The splice mode of a runner, from the environment: [None] when [BENCH_SPLICE_CONTROL] is unset.
    Otherwise it needs [BENCH_SPLICE_TREATMENT] (both schedule-cache directories, read and never
    written), and then either [BENCH_SPLICE_LIST=1] -- print {!describe} on stderr and exit 0,
    before any fixture is read -- or [BENCH_SPLICE_TARGET] (a comma-separated list of digest
    prefixes and [[<entry>]#<index>]) and [BENCH_SPLICE_ARM=control|treatment], in which case it
    writes that arm's cache into the configured [autotune_cache_dir], which must be set explicitly
    and absent or empty, and returns the plan: the runner's compile, inside {!replay_guarded}, then
    replays it from there. [BENCH_SPLICE_ADOPT_VERSION=<n>] adopts the control's entries stored at
    entry version [n]: a treatment that adds a search branch bumps [Schedule_cache.entry_version]
    without changing what a saved schedule means (across a bump that changed the saved format the
    entries fail to decode; one that changed the canonical rendering fails the replay's
    segment-digest check). A replay arm must not search, so it is refused unless [tune] is on and
    [autotune_search] is off: a key the arm's cache lacks then ships the untuned default, which
    {!replay_guarded} turns into a failed run. Every refusal exits 2 with its reason on stderr. *)
let from_env ~tune =
  match env "BENCH_SPLICE_CONTROL" with
  | None -> None
  | Some control ->
      let treatment =
        match env "BENCH_SPLICE_TREATMENT" with
        | Some t -> t
        | None -> fail "BENCH_SPLICE_CONTROL is set but BENCH_SPLICE_TREATMENT is not"
      in
      let adopt_version =
        Option.map (env "BENCH_SPLICE_ADOPT_VERSION") ~f:(fun v ->
            match Int.of_string_opt v with
            | Some n -> n
            | None -> fail "BENCH_SPLICE_ADOPT_VERSION=%s is not an integer" v)
      in
      let listing = match env "BENCH_SPLICE_LIST" with Some "1" -> true | _ -> false in
      let targets =
        match env "BENCH_SPLICE_TARGET" with
        | None when listing -> []
        | None ->
            fail "BENCH_SPLICE_TARGET is required (or BENCH_SPLICE_LIST=1 to list the segments)"
        | Some s -> ( match parse_targets s with Ok t -> t | Error e -> fail "%s" e)
      in
      let plan =
        match make_plan ?adopt_version ~control ~treatment ~targets () with
        | Ok plan -> plan
        | Error e -> fail "refused: %s" e
      in
      if listing then (
        List.iter (describe plan) ~f:(Stdio.eprintf "%s\n");
        Stdio.eprintf "%!";
        Stdlib.exit 0);
      let arm =
        match env "BENCH_SPLICE_ARM" with
        | Some "control" -> Control
        | Some "treatment" -> Treatment
        | Some a -> fail "BENCH_SPLICE_ARM=%s: expected control or treatment" a
        | None -> fail "BENCH_SPLICE_ARM is required: control or treatment"
      in
      if not tune then fail "the splice mode replays tuned schedules: set BENCH_TUNE=1";
      if Utils.registered_flag ~arg_name:"autotune_search" then
        fail
          "a splice arm must replay, never search: pass --ocannl_autotune_search=false (a key its \
           cache lacks then ships the untuned default, and the run fails)";
      (* Which routine is timed must be the control's placement decision, replayed: without it
         [Train.tune_placements] replays both placement candidates and ships the faster, and a
         target confined to the discarded one is a null measurement however the lookups read. *)
      if List.is_empty plan.placements then
        fail
          "the control cache records no placement decision, so the arm cannot ship the routine the \
           control shipped (both placement candidates would replay and the faster ship)";
      if not (Utils.get_global_flag ~arg_name:"tune_placement_store" ~default:true) then
        fail
          "tune_placement_store=false compares the placement candidates afresh instead of \
           replaying the control's decision: keep the placement store on";
      (match
         String.lowercase
           (String.strip (Utils.get_global_arg ~arg_name:"tune_ship_arm" ~default:"auto"))
       with
      | "auto" | "measured" -> ()
      | forced ->
          fail
            "tune_ship_arm=%s forces a placement candidate instead of replaying the control's \
             decision: leave it at auto"
            forced);
      (match Utils.get_global_arg_with_source ~arg_name:"autotune_cache_dir" ~default:"" with
      | _, Utils.From_default ->
          fail
            "the arm's cache goes into the configured schedule cache: pass \
             --ocannl_autotune_cache_dir=<a fresh directory> (absent or empty), one per run"
      | _ -> ());
      (match write_arm plan ~arm () with Ok _ -> () | Error e -> fail "refused: %s" e);
      List.iter (describe plan) ~f:(Stdio.eprintf "%s\n");
      Stdio.eprintf "splice: arm %s replays %s\n%!"
        (match arm with Control -> "control" | Treatment -> "treatment")
        (Autotune.resolve_cache_dir ~search:true ());
      Some plan

(** Whether a compile that looked up the cache keys [looked_up] and reported [arm_reports] placement
    arms measured the splice: it must have replayed the control's placement decision, which decides
    which candidate ships (one arm reported per decision lookup), replayed at least one routine the
    plan splices -- inputs holding several workloads or profiles can splice one routine while the
    run replays another, untargeted one, whose two arms are then the same control schedule -- and no
    routine a target was left unspliced in, which replays only off the control's placement decision
    (a forced [tune_ship_arm], say). Other untargeted routines replay the control in both arms,
    which is the method. *)
let replay_verdict plan ~looked_up ~arm_reports =
  let looked p = List.mem looked_up p.key ~equal:String.equal in
  (* A lookup is recorded whether or not its entry is then accepted, so a looked-up decision is no
     evidence that it replayed: one the tuning call rejects (its outcome digest no longer reproduces
     the lowering) falls back to timing both placement candidates. What tells the two apart is the
     arm count: each [Train.tune_placements] call looks its decision up once, and reports exactly
     one arm when it replays it, two when it compares the candidates afresh. *)
  let decision_lookups =
    List.count looked_up ~f:(fun key ->
        List.exists plan.placements ~f:(fun (pkey, _) -> String.equal key pkey))
  in
  let decided = decision_lookups > 0 && arm_reports = decision_lookups in
  let left p = match p.splice with `Left _ -> true | `Untargeted | `Spliced -> false in
  match List.find plan.pairs ~f:(fun p -> looked p && left p) with
  | Some p ->
      Error
        (Printf.sprintf
           "the compile replayed entry %s, where a target was left unspliced: its step is the \
            control's in both arms"
           (entry_id p.control))
  | None when not decided ->
      Error
        (Printf.sprintf
           "the compile did not replay the control's placement decision (%d decision lookups, %d \
            placement arms reported; a replayed decision reports one arm per lookup): which \
            candidate shipped is not the control's"
           decision_lookups arm_reports)
  | None ->
      if List.exists plan.pairs ~f:(fun p -> looked p && Poly.equal p.splice `Spliced) then Ok ()
      else
        Error
          "the compile replayed no spliced routine: the two arms time the same control schedules \
           (are the caches of another workload, profile or placement than this run?)"

(** [replay_guarded session ~counts compile] runs a runner's [compile] -- directly without a splice
    [session] -- and otherwise records the schedule-cache lookups it makes, then exits 1 when the
    step did not come entirely from the arm's cache ([counts] reads the runner's searches, replays,
    untuned fallbacks and placement-arm reports after the compile), or did not replay the control's
    placement decision and a spliced routine ({!replay_verdict}). *)
let replay_guarded session ~counts compile =
  match session with
  | None -> compile ()
  | Some plan ->
      let result, io = SC.recording_cache_io compile in
      let searches, replays, no_searches, arm_reports = counts () in
      if searches > 0 || no_searches > 0 || replays = 0 then (
        Stdio.eprintf
          "splice: the arm did not replay its cache throughout (searches=%d replays=%d \
           no_searches=%d): its step is not the spliced schedule\n\
           %!"
          searches replays no_searches;
        Stdlib.exit 1);
      let looked_up =
        List.filter_map io ~f:(fun (r : SC.cache_io) ->
            match r.SC.op with SC.Lookup -> Some r.SC.key | SC.Store -> None)
      in
      (match replay_verdict plan ~looked_up ~arm_reports with
      | Ok () -> ()
      | Error e ->
          Stdio.eprintf "splice: %s\n%!" e;
          Stdlib.exit 1);
      result
