(** The mechanical half of keeping [Refusal_control_manifest] in step with the scanners it
    catalogues (gh-ocannl-1429): after a diagnostic is reworded, its marker digest moves, and the
    row listing it, the [raw_direct_evidence] or [raw_catalogue_only] key naming it, and a
    [raw_direct_evidence] label naming a reworded control claim all go stale together. This module
    reads those three tables from the manifest's TEXT, works out what a rewording replaced, and
    writes the replacement text; [tools/refusal-manifest-sync.sh] runs it, reformats the file and
    re-runs the goldens.

    What it decides mechanically is only what a rewording determines: a row is the extraction
    ({!Refusal_control_manifest.row}'s contract), and a key or label follows the diagnostic it named
    when the registered row and the extraction pair that diagnostic one for one ({!align}). Anything
    else -- a key no rewording pairs, a control claim reworded into a format with substitutions, a
    direct failure that became a claim -- is reported as unresolved and left for a hand edit, never
    guessed: which control answers for a refusal is a judgment the tables record. *)

open Base
open Ppxlib.Parsetree
module Scan = Refusal_control_scan
module Read = Config_key_scan

type literal = { value : string; start : int; stop : int }
(** A string literal of the manifest's text: its value and its byte span, quotes included. *)

type row = { source : string; markers : string list; list_start : int; list_stop : int }
(** A [raw_entries] row: its source and the span of its marker list. *)

type entry = { key : literal; value : literal }
(** A [raw_direct_evidence] or [raw_catalogue_only] entry. *)

type tables = { rows : row list; direct_evidence : entry list; catalogue_only : entry list }

let span expression = (expression.pexp_loc.loc_start.pos_cnum, expression.pexp_loc.loc_end.pos_cnum)

let literal expression =
  Option.map (Read.string_literal expression) ~f:(fun value ->
      let start, stop = span expression in
      { value; start; stop })

let rec elements expression =
  match expression.pexp_desc with
  | Pexp_construct ({ txt = Lident "[]"; _ }, None) -> Some []
  | Pexp_construct ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ head; tail ]; _ }) ->
      Option.map (elements tail) ~f:(fun rest -> head :: rest)
  | _ -> None

let pair expression =
  match expression.pexp_desc with Pexp_tuple [ first; second ] -> Some (first, second) | _ -> None

(** The list a binding named [name] holds, through a type constraint. *)
let bound_list structure name =
  let rec unwrap e = match e.pexp_desc with Pexp_constraint (inner, _) -> unwrap inner | _ -> e in
  List.find_map structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.find_map bindings ~f:(fun binding ->
              match binding.pvb_pat.ppat_desc with
              | (Ppat_var { txt; _ } | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _))
                when String.equal txt name ->
                  Some (unwrap binding.pvb_expr)
              | _ -> None)
      | _ -> None)

(** Reads the three tables from the manifest's [text]. Raises, naming the table, on a shape it does
    not read -- an element that is not a literal pair -- so a rewrite never runs over a table it
    read only in part. *)
let read_tables text =
  let structure = Read.structure_of text in
  let refuse table = failwith ("Refusal_manifest_sync: cannot read the `" ^ table ^ "` list") in
  let list table =
    match Option.bind (bound_list structure table) ~f:elements with
    | Some items -> items
    | None -> refuse table
  in
  let rows =
    List.map (list "raw_entries") ~f:(fun item ->
        match pair item with
        | Some (source, markers) -> (
            match (Read.string_literal source, elements markers) with
            | Some source, Some marker_items -> (
                match Option.all (List.map marker_items ~f:Read.string_literal) with
                | Some markers_values ->
                    let list_start, list_stop = span markers in
                    { source; markers = markers_values; list_start; list_stop }
                | None -> refuse "raw_entries")
            | _ -> refuse "raw_entries")
        | None -> refuse "raw_entries")
  in
  let entries table =
    List.map (list table) ~f:(fun item ->
        match Option.bind (pair item) ~f:(fun (k, v) -> Option.both (literal k) (literal v)) with
        | Some (key, value) -> { key; value }
        | None -> refuse table)
  in
  {
    rows;
    direct_evidence = entries "raw_direct_evidence";
    catalogue_only = entries "raw_catalogue_only";
  }

(** An OCaml string literal of [value], escaping only what the lexer needs, so that non-ASCII text
    stays readable; ocamlformat rewraps a long one. *)
let quote value =
  let buffer = Buffer.create (String.length value + 2) in
  Buffer.add_char buffer '"';
  String.iter value ~f:(function
    | '"' -> Buffer.add_string buffer "\\\""
    | '\\' -> Buffer.add_string buffer "\\\\"
    | '\n' -> Buffer.add_string buffer "\\n"
    | '\t' -> Buffer.add_string buffer "\\t"
    | '\r' -> Buffer.add_string buffer "\\r"
    | c -> Buffer.add_char buffer c);
  Buffer.add_char buffer '"';
  Buffer.contents buffer

(** The identity a marker carries: the digest after its [scanner-refusal:] prefix, up to the closing
    bracket. *)
let marker_identity marker =
  Option.bind (String.chop_prefix marker ~prefix:"[scanner-refusal:") ~f:(fun rest ->
      Option.map (String.lsplit2 rest ~on:']') ~f:fst)

(** [align ~before ~after]: which elements of [before] the elements of [after] replaced. The two
    lists are compared through their longest common subsequence under [equal]; between two kept
    elements, a run of [before] elements dropped and a run of [after] elements inserted of the SAME
    length are paired one for one, in order -- a rewording keeps its place in the source -- while a
    run of unequal lengths pairs nothing, since which of them replaced which is no longer
    determined. Answers the pairs, then the unpaired [before] and [after] elements. *)
let align ~equal ~before ~after =
  let before = Array.of_list before and after = Array.of_list after in
  let n = Array.length before and m = Array.length after in
  let lcs = Array.make_matrix ~dimx:(n + 1) ~dimy:(m + 1) 0 in
  for i = n - 1 downto 0 do
    for j = m - 1 downto 0 do
      lcs.(i).(j) <-
        (if equal before.(i) after.(j) then 1 + lcs.(i + 1).(j + 1)
         else max lcs.(i + 1).(j) lcs.(i).(j + 1))
    done
  done;
  let pairs = ref [] and dropped = ref [] and inserted = ref [] in
  let flush hunk_before hunk_after =
    let hunk_before = List.rev hunk_before and hunk_after = List.rev hunk_after in
    if List.length hunk_before = List.length hunk_after then
      pairs := List.rev_append (List.zip_exn hunk_before hunk_after) !pairs
    else (
      dropped := List.rev_append hunk_before !dropped;
      inserted := List.rev_append hunk_after !inserted)
  in
  let rec walk i j hunk_before hunk_after =
    if i < n && j < m && equal before.(i) after.(j) then (
      flush hunk_before hunk_after;
      walk (i + 1) (j + 1) [] [])
    else if i < n && (j >= m || lcs.(i + 1).(j) >= lcs.(i).(j + 1)) then
      walk (i + 1) j (before.(i) :: hunk_before) hunk_after
    else if j < m then walk i (j + 1) hunk_before (after.(j) :: hunk_after)
    else flush hunk_before hunk_after
  in
  walk 0 0 [] [];
  (List.rev !pairs, List.rev !dropped, List.rev !inserted)

(** The control claims reworded between two versions of one source: the formats of its [Claim]
    diagnostics that {!align} pairs, old to new, where the text changed, then the old formats it
    pairs with nothing -- a claim removed, or reworded beside an insertion or a removal, which no
    one-for-one pairing settles. Feeds {!plan} the claims of files outside the manifest's rows -- a
    [*_cases.ml] control whose claim a [raw_direct_evidence] entry names. *)
let claim_renames ~before ~after =
  let claims source =
    List.filter (Scan.diagnostics source) ~f:(fun d ->
        match d.Scan.kind with Scan.Claim -> true | Scan.Fail -> false)
  in
  let after = claims after in
  let pairs, dropped, _ =
    align
      ~equal:(fun a b -> String.equal a.Scan.format b.Scan.format)
      ~before:(claims before) ~after
  in
  (* A label the source still claims at another site names that site still: no rename. *)
  let survives format = List.exists after ~f:(fun d -> String.equal d.Scan.format format) in
  ( List.filter_map pairs ~f:(fun (old, renamed) ->
        Option.some_if
          ((not (String.equal old.Scan.format renamed.Scan.format))
          && not (survives old.Scan.format))
          (old.Scan.format, renamed.Scan.format)),
    (* A dropped claim whose format the source still has -- moved, not removed -- is no lost one. *)
    List.filter_map dropped ~f:(fun d ->
        Option.some_if (not (survives d.Scan.format)) d.Scan.format) )

(** What a sync changed or could not. *)
type change =
  | Row of { source : string; added : string list; removed : string list }
  | Rekeyed of { table : string; old_key : string; new_key : string }
  | Relabeled of { key : string; old_label : string; new_label : string }

type plan = {
  edits : (int * int * string) list;  (** Byte spans of the manifest text and their replacement. *)
  changes : change list;
  unresolved : string list;  (** What a hand edit must settle, one sentence each. *)
  markers : (string * string) list;
      (** Each marker a rewording replaced, old to new: the text a golden holding it now prints. *)
  claims : (string * (string * string)) list;
      (** Each claim a rewording replaced, by its old identity, to the scanner source it is in and
          its new format: a golden row [<old label>: true] of that source now prints the new one,
          where the format has no substitutions. *)
}

let evidence_suffix = ": true"
let identity_of text = Stdlib.Digest.string (Scan.normalize text) |> Stdlib.Digest.to_hex

(** The plan for the manifest [text]. [diagnostics_of] gives a row source's current diagnostics
    ([None] for a source that is gone), [renamed_claims] the control claims reworded outside the
    rows, old format to new, and [unpaired_claims] the old formats there that nothing replaced one
    for one ({!claim_renames}).

    One old diagnostic is followed only to ONE successor: an identity occurring twice in a row (two
    sites of one format) whose occurrences were reworded differently has no single successor for its
    key, label or golden marker, and is unresolved -- never the last pairing that happened to win.
*)
let plan ?(unpaired_claims = []) ~text ~diagnostics_of ~renamed_claims () =
  let tables = read_tables text in
  let edits = ref [] and changes = ref [] and unresolved = ref [] in
  let unresolved_f fmt = Printf.ksprintf (fun s -> unresolved := s :: !unresolved) fmt in
  (* Per row source: its current diagnostics, and the diagnostic each replaced key's refusal
     became. *)
  let current = Hashtbl.create (module String) and replaced = Hashtbl.create (module String) in
  let markers = ref [] and claims = ref [] and unpaired = ref [] and pending = ref [] in
  List.iter tables.rows ~f:(fun row ->
      match diagnostics_of row.source with
      | None ->
          unresolved_f "%s has a `raw_entries` row but no source to extract it from" row.source
      | Some diagnostics ->
          Hashtbl.set current ~key:row.source ~data:diagnostics;
          let extracted = List.map diagnostics ~f:Scan.marker in
          if not (List.equal String.equal extracted row.markers) then (
            edits :=
              ( row.list_start,
                row.list_stop,
                "[ " ^ String.concat ~sep:"; " (List.map extracted ~f:quote) ^ " ]" )
              :: !edits;
            changes :=
              Row
                {
                  source = row.source;
                  added = Scan.minus extracted row.markers;
                  removed = Scan.minus row.markers extracted;
                }
              :: !changes;
            let pairs, dropped, _ =
              align
                ~equal:(fun marker d -> String.equal marker (Scan.marker d))
                ~before:row.markers ~after:diagnostics
            in
            (* A dropped marker whose format the source still has elsewhere is no lost claim. *)
            unpaired :=
              List.filter_map dropped ~f:marker_identity
              |> List.filter ~f:(fun identity ->
                  not (List.exists diagnostics ~f:(fun d -> String.equal d.Scan.identity identity)))
              |> Fn.flip ( @ ) !unpaired;
            (* An old identity the source still has at another site was reworded at one site only:
               its key, label and golden rows now name two things, so none of them moves. *)
            let survives identity =
              List.exists diagnostics ~f:(fun d -> String.equal d.Scan.identity identity)
            in
            List.iter pairs ~f:(fun (marker, d) ->
                Option.iter (marker_identity marker) ~f:(fun identity ->
                    if String.equal marker (Scan.marker d) then ()
                    else if survives identity then
                      unresolved_f
                        "%s was reworded at one of its sites in %s and still occurs at another: \
                         the goldens holding it, and any entry naming it, are a hand edit"
                        marker row.source
                    else pending := (row.source, marker, identity, d) :: !pending))));
  (* Every row read: a replaced marker another scanner's row still lists, or a reworded claim
     another scanner still makes, is text two sources share -- a golden swap or a relabel by text
     would move the other's too, so neither moves. *)
  let elsewhere ~source predicate =
    List.exists tables.rows ~f:(fun r -> (not (String.equal r.source source)) && predicate r)
  in
  List.iter (List.rev !pending) ~f:(fun (source, marker, identity, d) ->
      let shared_marker = elsewhere ~source (fun r -> List.mem r.markers marker ~equal:String.equal)
      and shared_claim =
        elsewhere ~source (fun r ->
            Option.value_map (Hashtbl.find current r.source) ~default:false ~f:(fun ds ->
                List.exists ds ~f:(fun other ->
                    String.equal other.Scan.identity identity
                    && match other.Scan.kind with Scan.Claim -> true | Scan.Fail -> false)))
      in
      if shared_marker || shared_claim then
        unresolved_f
          "%s, reworded in %s, is also another scanner's: the goldens holding it, and any entry \
           naming it, are a hand edit"
          marker source
      else (
        markers := (marker, Scan.marker d) :: !markers;
        Hashtbl.add_multi replaced ~key:(source ^ ":" ^ identity) ~data:d;
        match d.Scan.kind with
        | Scan.Claim -> claims := (identity, (source, d.Scan.format)) :: !claims
        | Scan.Fail -> ()));
  (* A row's own alignment outranks the claims-only pairing of the same source: a claim the row
     leaves unpaired (reworded beside an inserted failure, say) is not renamed by a pass that never
     saw the failure. *)
  let row_unpaired = !unpaired in
  (* A claim reworded outside the scanner sources (a [*_cases.ml] control) is never followed by
     text: the same label may back another scanner's evidence from another file, which a relabel
     would hand to a claim its control never runs. Its labels are a hand edit, with the new text as
     the hint. *)
  let external_renames =
    List.filter_map renamed_claims ~f:(fun (old_format, format) ->
        let identity = identity_of old_format in
        Option.some_if (not (List.mem row_unpaired identity ~equal:String.equal)) (identity, format))
  in
  unpaired := List.map unpaired_claims ~f:identity_of @ !unpaired;
  (* One old text to more than one new one: no single successor, so none is chosen. *)
  let ambiguous pairs =
    Hashtbl.filter_map
      (Hashtbl.of_alist_multi (module String) pairs)
      ~f:(fun successors ->
        match List.dedup_and_sort successors ~compare:String.compare with
        | [ _ ] -> None
        | successors -> Some successors)
  in
  let ambiguous_markers = ambiguous !markers
  and ambiguous_claims =
    ambiguous (List.map !claims ~f:(fun (identity, (_, format)) -> (identity, format)))
  in
  Hashtbl.iteri ambiguous_markers ~f:(fun ~key ~data ->
      unresolved_f
        "%s occurs more than once and its occurrences were reworded into %d different successors: \
         the goldens holding it, and any entry naming it, are a hand edit"
        key (List.length data));
  markers := List.filter !markers ~f:(fun (old, _) -> not (Hashtbl.mem ambiguous_markers old));
  claims := List.filter !claims ~f:(fun (old, _) -> not (Hashtbl.mem ambiguous_claims old));
  let answers_now key =
    match String.rsplit2 key ~on:':' with
    | None -> false
    | Some (source, identity) ->
        Option.value_map (Hashtbl.find current source) ~default:false ~f:(fun diagnostics ->
            List.exists diagnostics ~f:(fun d ->
                String.equal d.Scan.identity identity
                && match d.Scan.kind with Scan.Fail -> true | Scan.Claim -> false))
  in
  let rekeys = ref [] in
  let rekey table entries =
    List.iter entries ~f:(fun entry ->
        let key = entry.key.value in
        if not (answers_now key) then
          (* A key in [replaced] is a row source's, so it splits. *)
          let successors =
            Hashtbl.find_multi replaced key
            |> List.dedup_and_sort ~compare:(fun a b ->
                String.compare a.Scan.identity b.Scan.identity)
          in
          match (successors, String.lsplit2 key ~on:':') with
          | _ :: _ :: _, _ ->
              unresolved_f
                "%s in `%s` names a format occurring more than once, whose occurrences were \
                 reworded differently: which successor it answers for is a hand edit"
                key table
          | [ { Scan.kind = Scan.Fail; identity; _ } ], Some (source, _) ->
              rekeys := (entry, table, key, source ^ ":" ^ identity) :: !rekeys
          | [ { Scan.kind = Scan.Fail; _ } ], None -> ()
          | [ { Scan.kind = Scan.Claim; format; _ } ], _ ->
              unresolved_f
                "%s in `%s` names a direct failure now reworded into a claim (%S): move or drop \
                 the entry by hand"
                key table format
          | [], _ ->
              unresolved_f
                "%s in `%s` answers to no current direct failure, and no rewording pairs it with \
                 one: re-key or drop it by hand"
                key table)
  in
  rekey "raw_direct_evidence" tables.direct_evidence;
  rekey "raw_catalogue_only" tables.catalogue_only;
  (* Two keys converging on one successor -- a failure reworded into a sibling's words -- or a
     successor some other entry already holds would leave one key in two places (both tables at
     once, or a duplicate a lookup silently picks one of): such a re-key is a hand edit. *)
  let final_keys =
    List.map (tables.direct_evidence @ tables.catalogue_only) ~f:(fun entry ->
        match List.find !rekeys ~f:(fun (rekeyed, _, _, _) -> phys_equal rekeyed entry) with
        | Some (_, _, _, new_key) -> new_key
        | None -> entry.key.value)
  in
  List.iter (List.rev !rekeys) ~f:(fun (entry, table, old_key, new_key) ->
      if List.count final_keys ~f:(String.equal new_key) > 1 then
        unresolved_f
          "%s in `%s` would be re-keyed to %s, which another entry holds too: merge or drop the \
           entries by hand"
          old_key table new_key
      else (
        edits := (entry.key.start, entry.key.stop, quote new_key) :: !edits;
        changes := Rekeyed { table; old_key; new_key } :: !changes));
  (* A control claim reworded: the label of a [raw_direct_evidence] entry names it verbatim. *)
  List.iter tables.direct_evidence ~f:(fun entry ->
      let value = entry.value.value in
      let label, suffix =
        match String.chop_suffix value ~suffix:evidence_suffix with
        | Some label -> (label, evidence_suffix)
        | None -> (value, "")
      in
      let identity = identity_of label in
      (* A label a FORMATTED claim rendered is no format's identity: matched against each old format
         with substitutions that a rewording or removal retired, it is a hand edit. *)
      let formatted_retired =
        List.exists
          (List.map renamed_claims ~f:fst @ unpaired_claims)
          ~f:(fun old -> String.mem old '%' && Scan.format_matches ~format:old label)
      in
      let key_source =
        Option.value_map (String.lsplit2 entry.key.value ~on:':') ~default:"" ~f:fst
      in
      match List.Assoc.find !claims identity ~equal:String.equal with
      | Some (source, format) when not (String.equal source key_source) ->
          unresolved_f
            "the claim %s's evidence names was reworded in another scanner's source, %s (it now \
             reads %S): relabel the entry by hand"
            entry.key.value source format
      | None when List.Assoc.mem external_renames identity ~equal:String.equal ->
          unresolved_f
            "the claim %s's evidence names was reworded outside the scanner sources (it now reads \
             %S): relabel the entry by hand"
            entry.key.value
            (List.Assoc.find_exn external_renames identity ~equal:String.equal)
      | None when formatted_retired ->
          unresolved_f
            "the claim %s's evidence names was rendered by a formatted claim since reworded or \
             removed: write the label its successor renders by hand"
            entry.key.value
      | None when Hashtbl.mem ambiguous_claims identity ->
          unresolved_f
            "the claim %s's evidence names occurs more than once and was reworded differently: \
             write its label by hand"
            entry.key.value
      | None when List.mem !unpaired identity ~equal:String.equal ->
          unresolved_f
            "the claim %s's evidence names was removed, or reworded beside an insertion or a \
             removal, so nothing pairs it with a successor: relabel or re-map the entry by hand"
            entry.key.value
      | None -> ()
      | Some (_, format) when String.mem format '%' ->
          unresolved_f
            "the claim %s's evidence names was reworded into a format with substitutions (%S): \
             write the label it prints by hand"
            entry.key.value format
      | Some (_, format) ->
          edits := (entry.value.start, entry.value.stop, quote (format ^ suffix)) :: !edits;
          changes :=
            Relabeled { key = entry.key.value; old_label = label; new_label = format } :: !changes);
  {
    edits = !edits;
    changes = List.rev !changes;
    unresolved = List.rev !unresolved;
    markers = List.rev !markers;
    claims = List.rev !claims;
  }

(** [text] with [edits] applied; they must not overlap. *)
let apply text edits =
  let sorted = List.sort edits ~compare:(fun (a, _, _) (b, _, _) -> Int.compare b a) in
  let _, result =
    List.fold sorted ~init:(Int.max_value, text) ~f:(fun (limit, text) (start, stop, replacement) ->
        if stop > limit || start > stop then
          failwith "Refusal_manifest_sync.apply: overlapping or inverted edits";
        ( start,
          String.concat [ String.prefix text start; replacement; String.drop_prefix text stop ] ))
  in
  result

(** A golden's [text] as the rewording prints it: each replaced marker swapped for its successor,
    and each row [<label>: true] or [<label>: PASS] of a reworded claim -- the label's identity the
    old claim's, the new format free of substitutions -- swapped for the new label. Nothing else
    moves: a golden row the rewording changes in any other way is a diff the confirming run shows.
    Without this the audit could never pass on the first run: its coverage claims read the goldens
    for the NEW markers, which only a run that passes would print. *)
let rewrite_golden ?sources plan text =
  (* A claim row is the claim's own source's to rewrite: the same label in another source's golden
     is another claim. [sources], when given, are the sources whose goldens [text] is. *)
  let claims =
    match sources with
    | None -> plan.claims
    | Some sources ->
        List.filter plan.claims ~f:(fun (_, (source, _)) ->
            List.mem sources source ~equal:String.equal)
  in
  if List.is_empty plan.markers && List.is_empty claims then text
  else
    let swap_markers line =
      if not (String.is_substring line ~substring:"[scanner-refusal:") then line
      else
        List.fold plan.markers ~init:line ~f:(fun line (old, renamed) ->
            String.substr_replace_all line ~pattern:old ~with_:renamed)
    in
    let swap_claim line =
      List.find_map [ ": true"; ": PASS" ] ~f:(fun suffix ->
          Option.bind (String.chop_suffix line ~suffix) ~f:(fun label ->
              match List.Assoc.find claims (identity_of label) ~equal:String.equal with
              | Some (_, format) when not (String.mem format '%') -> Some (format ^ suffix)
              | _ -> None))
      |> Option.value ~default:line
    in
    let ends_with_newline = String.is_suffix text ~suffix:"\n" in
    let lines = String.split text ~on:'\n' in
    let lines = if ends_with_newline then List.drop_last_exn lines else lines in
    String.concat ~sep:"\n" (List.map lines ~f:(fun line -> swap_claim (swap_markers line)))
    ^ if ends_with_newline then "\n" else ""

(** The sources a plan touched: rows rewritten, and the scanners of keys and labels it changed. *)
let touched_sources plan =
  List.concat_map plan.changes ~f:(function
    | Row { source; _ } -> [ source ]
    | Rekeyed { new_key = key; _ } | Relabeled { key; _ } ->
        Option.to_list (Option.map (String.lsplit2 key ~on:':') ~f:fst))
  |> List.dedup_and_sort ~compare:String.compare
