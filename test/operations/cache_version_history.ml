(** Every version constant of the schedule cache names the change that set its current value.

    [arrayjit/lib/schedule_cache.ml] stamps its stores with integer versions ([entry_version],
    [placement_entry_version], [cache_regime_version]) and spells the CUDA/HIP timing policy's
    generation into its keys ([queued_objective_version]), and a bump is how a change says "what was
    stored before me is not the answer any more". Two branches bumping the same constant in parallel
    both write the SAME new line -- [let entry_version = 13] becomes [14] on each -- so git merges
    them without a word, and the second change ships under the first one's number: entries the first
    change's search stored replay as current for the second one's menu. The wave of October 2026 hit
    it twice in one day (gh-ocannl-1175 had to move from 9 to 12, gh-ocannl-1165 from 10 to 13), and
    each time only a human noticed.

    So each constant carries a verbatim history comment directly above it, whose last line is the
    current value and the issue that set it:
    {v
      (*= entry_version history -- one line per value, oldest first; a bump appends its own line:
         12: gh-ocannl-1175 -- a reduction's zero folds into its own segment's sketches
         13: gh-ocannl-1165 -- the coalesced layout's seeds
      *)
      let entry_version = 13
    v}
    Two parallel bumps now each append a DIFFERENT line at the same place, which git reports as a
    conflict. Whoever resolves it by keeping both lines is refused for a value that does not
    increase; whoever renumbers their line but not the constant is refused for a history that does
    not end on the current value; and a bump that skips the history is refused the same way. The
    comment is verbatim (["(*="]) because ocamlformat wraps ordinary comments into paragraphs, which
    would fold the entries into one line.

    The grammar, rigid on purpose -- a malformed block is refused, never repaired:
    - the constant is a top-level [let <name>version = <integer literal>], declared once; a computed
      or annotated one is refused rather than skipped, since it would escape the rule;
    - the line directly above it is the block's closing ["*)"], alone on its line;
    - every line between that and the header ["(*= <name> history"] is an entry [<value>: <text>],
      where the text cites [gh-ocannl-<number>] or [staging#<number>];
    - values strictly increase (a gap is a value that never landed; a retired value keeps its line,
      since entries stored under it may still be on disk), and the last one is the constant's value.

    The constants are DERIVED from the file -- every top-level [let] whose name ends in [version] --
    so a new versioned store is held to the rule the day it appears. A floor keeps the derivation
    from passing an empty or misrouted file, and synthesized controls hold every refusal against a
    text built to trip it beside the nearest text that must pass, since the live file, on a good
    day, trips none of them. *)

open Base
open Stdio
open Verdict.Claims

type finding =
  | Not_a_literal of { line : int; name : string }
      (** A top-level [let <name>version] whose right-hand side is not an integer literal. *)
  | Declared_again of { line : int; name : string }
  | Missing_history of { line : int; name : string }
      (** No ["*)"] alone on the line directly above, or no header above the entries. *)
  | Stray_line of { line : int; name : string }
      (** A line inside the block that is neither an entry nor the header. *)
  | Uncited of { line : int; name : string }
  | Not_increasing of { line : int; name : string; value : int; previous : int }
  | History_ends_elsewhere of { line : int; name : string; history : int; constant : int }

let describe = function
  | Not_a_literal { line; name } ->
      Printf.sprintf
        "line %d: %s is not an integer literal, so its history cannot be checked against it" line
        name
  | Declared_again { line; name } -> Printf.sprintf "line %d: %s is declared again" line name
  | Missing_history { line; name } ->
      Printf.sprintf
        "line %d: %s has no `(*= %s history` block ending in `*)` alone on the line directly above \
         it"
        line name name
  | Stray_line { line; name } ->
      Printf.sprintf
        "line %d: inside %s's history, neither an entry `<value>: <text citing gh-ocannl-<N> or \
         staging#<N>>` nor the `(*= %s history` header"
        line name name
  | Uncited { line; name } ->
      Printf.sprintf "line %d: %s's history entry cites no gh-ocannl-<N> or staging#<N>" line name
  | Not_increasing { line; name; value; previous } ->
      Printf.sprintf
        "line %d: %s's history goes from %d to %d -- two bumps to one value, or an entry out of \
         order"
        line name previous value
  | History_ends_elsewhere { line; name; history; constant } ->
      Printf.sprintf
        "line %d: %s is %d but its history ends at %d -- append `%d: gh-ocannl-<N> -- <what \
         changed>` as its last entry"
        line name constant history constant

let is_ident_char c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '\''

(** Whether [text] contains [prefix] immediately followed by a digit. *)
let cites_with text ~prefix =
  List.exists (String.substr_index_all text ~may_overlap:false ~pattern:prefix) ~f:(fun i ->
      let j = i + String.length prefix in
      j < String.length text && Char.is_digit text.[j])

let cites text = cites_with text ~prefix:"gh-ocannl-" || cites_with text ~prefix:"staging#"

(** An entry line, stripped: [<value>: <text>]. *)
let entry s =
  match String.lsplit2 s ~on:':' with
  | Some (value, text)
    when (not (String.is_empty value))
         && String.for_all value ~f:Char.is_digit
         && String.is_prefix text ~prefix:" " ->
      Option.map (Int.of_string_opt value) ~f:(fun v -> (v, text))
  | _ -> None

(** The top-level version constants of [lines]: [(index, name, value)] with [value = None] when the
    right-hand side is not an integer literal. *)
let constants lines =
  List.filter_mapi lines ~f:(fun i line ->
      match String.chop_prefix line ~prefix:"let " with
      | None -> None
      | Some rest ->
          let name =
            String.prefix rest
              (Option.value ~default:(String.length rest)
                 (String.lfindi rest ~f:(fun _ c -> not (is_ident_char c))))
          in
          if not (String.is_suffix name ~suffix:"version") then None
          else
            let rhs =
              String.chop_prefix (String.drop_prefix rest (String.length name)) ~prefix:" = "
              |> Option.map ~f:String.rstrip
            in
            let value =
              match rhs with
              | Some digits
                when (not (String.is_empty digits)) && String.for_all digits ~f:Char.is_digit ->
                  Int.of_string_opt digits
              | _ -> None
            in
            Some (i, name, value))

(** The findings of the history rule over the source [text], and the names of the constants it
    found, in order. *)
let check text =
  let lines = Array.of_list (String.split_lines text) in
  let found = constants (Array.to_list lines) in
  let seen = Hash_set.create (module String) in
  let findings =
    List.concat_map found ~f:(fun (i, name, value) ->
        let line = i + 1 in
        if Hash_set.mem seen name then [ Declared_again { line; name } ]
        else (
          Hash_set.add seen name;
          match value with
          | None -> [ Not_a_literal { line; name } ]
          | Some constant -> (
              if i = 0 || not (String.equal (String.strip lines.(i - 1)) "*)") then
                [ Missing_history { line; name } ]
              else
                let header = "(*= " ^ name ^ " history" in
                (* Walk up from the closing line to the header, collecting entries top-down. *)
                let rec walk k entries =
                  if k < 0 then Error (Missing_history { line; name })
                  else
                    let s = String.strip lines.(k) in
                    if String.is_prefix s ~prefix:header then Ok entries
                    else
                      match entry s with
                      | Some (v, text) -> walk (k - 1) ((k + 1, v, text) :: entries)
                      | None -> Error (Stray_line { line = k + 1; name })
                in
                match walk (i - 2) [] with
                | Error finding -> [ finding ]
                | Ok entries ->
                    let uncited =
                      List.filter_map entries ~f:(fun (line, _, text) ->
                          if cites text then None else Some (Uncited { line; name }))
                    in
                    let rec order acc = function
                      | (_, previous, _) :: ((line, value, _) :: _ as rest) ->
                          order
                            (if value > previous then acc
                             else Not_increasing { line; name; value; previous } :: acc)
                            rest
                      | _ -> List.rev acc
                    in
                    let last =
                      match List.last entries with
                      | Some (_, history, _) when history = constant -> []
                      | Some (_, history, _) ->
                          [ History_ends_elsewhere { line; name; history; constant } ]
                      | None -> [ Missing_history { line; name } ]
                    in
                    uncited @ order [] entries @ last)))
  in
  (findings, List.map found ~f:(fun (_, name, _) -> name))

(** A synthesized store with two landed values and a gap; each control below changes one thing. *)
let sample ?(history = [ "1: gh-ocannl-1 -- the first payload"; "3: staging#2 -- the second" ])
    ?(above = "") ?(constant = "let sample_version = 3") () =
  String.concat ~sep:"\n"
    ([ "(* Prose about the store, which the rule does not read. *)" ]
    @ [ "(*= sample_version history -- one line per value, oldest first:" ]
    @ List.map history ~f:(fun l -> "   " ^ l)
    @ [ "*)" ^ above; constant; "" ])

let refuses text ~f =
  let findings, _ = check text in
  List.exists findings ~f

let () =
  let path =
    match Array.to_list Stdlib.Sys.argv |> List.tl_exn with
    | [ path ] -> path
    | args ->
        eprintf "FAILED: expected exactly one path, the schedule cache's source, got %d arguments\n"
          (List.length args);
        Stdlib.exit 1
  in
  let text = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all in
  let findings, names = check text in
  eprintf "%s: version constants %s (not part of the golden)\n" path (String.concat ~sep:", " names);
  List.iter findings ~f:(fun finding -> eprintf "%s: %s\n" path (describe finding));
  printf
    "Each version constant of the schedule cache ends a verbatim history naming the change that\n\
     set its current value, so two parallel bumps conflict in git instead of sharing one number.\n\n";
  p "the scan was handed the schedule cache's source"
    (String.equal (Stdlib.Filename.basename path) "schedule_cache.ml");
  p_all ~min:3 "it found the store's version constants, not an empty derivation" names
    ~f:(String.is_suffix ~suffix:"version");
  p_empty "every version constant's history is well formed and ends on its current value"
    ~over:names findings;
  (* The CUDA/HIP queued key's generation is spelled into a key rather than stamped into a store.
     Inlined back as a literal, it would drop out of [names] while the floor above still held, so
     both its membership and the absence of a literal generation are claimed. The population of the
     second is the key's spelling sites, which must exist. *)
  p "the queued timing objective's generation is one of them"
    (List.mem names "queued_objective_version" ~equal:String.equal);
  let quote = "\"queued-v" in
  p_none "no spelling of the queued key hard-codes its generation as a literal queued-v<N>"
    (String.substr_index_all text ~may_overlap:false ~pattern:quote) ~f:(fun i ->
      let j = i + String.length quote in
      j < String.length text && Char.is_digit text.[j]);
  (* The controls: each refusal on a text built to trip it, beside the nearest text that passes. *)
  let accepted label text = p_empty label ~over:(String.split_lines text) (fst (check text)) in
  accepted "a history whose last entry is the constant's value passes, a gap included" (sample ());
  accepted "a retired value cited in a parenthetical passes"
    (sample
       ~history:
         [ "1: gh-ocannl-1 -- first"; "2: retired -- staging#934 (reverted)"; "3: gh-ocannl-3" ]
       ());
  p "a bump without its history line is refused"
    (refuses (sample ~constant:"let sample_version = 4" ()) ~f:(function
      | History_ends_elsewhere { history = 3; constant = 4; _ } -> true
      | _ -> false));
  p "a merge that kept both sides' lines for one value is refused"
    (refuses
       (sample
          ~history:
            [ "1: gh-ocannl-1"; "3: gh-ocannl-2"; "4: gh-ocannl-10 -- A"; "4: gh-ocannl-11 -- B" ]
          ~constant:"let sample_version = 4" ())
       ~f:(function Not_increasing { value = 4; previous = 4; _ } -> true | _ -> false));
  p "an entry citing no issue is refused, and so is a template's gh-ocannl-<N> placeholder"
    (refuses
       (sample ~history:[ "1: gh-ocannl-1"; "3: the second" ] ())
       ~f:(function Uncited { line = 4; _ } -> true | _ -> false)
    && refuses
         (sample ~history:[ "1: gh-ocannl-1"; "3: gh-ocannl-<N> -- the second" ] ())
         ~f:(function Uncited _ -> true | _ -> false));
  p "an entry wrapped onto a continuation line is refused"
    (refuses
       (sample ~history:[ "1: gh-ocannl-1"; "3: gh-ocannl-2 -- the second,"; "and more" ] ())
       ~f:(function Stray_line { line = 5; _ } -> true | _ -> false));
  p "a blank line between the block and the constant, or no block, is refused"
    (refuses (sample ~above:"\n" ()) ~f:(function Missing_history _ -> true | _ -> false)
    && refuses "let sample_version = 3\n" ~f:(function Missing_history _ -> true | _ -> false));
  p "a computed or annotated constant is refused rather than skipped"
    (refuses (sample ~constant:"let sample_version = base + 1" ()) ~f:(function
       | Not_a_literal _ -> true
       | _ -> false)
    && refuses (sample ~constant:"let sample_version : int = 3" ()) ~f:(function
      | Not_a_literal _ -> true
      | _ -> false));
  p "a constant declared twice is refused"
    (refuses
       (sample () ^ "let sample_version = 3\n")
       ~f:(function Declared_again _ -> true | _ -> false));
  p "a header naming another constant does not open this one's block"
    (refuses
       (String.substr_replace_first (sample ()) ~pattern:"(*= sample_version"
          ~with_:"(*= other_version")
       ~f:(function Stray_line _ -> true | _ -> false))
