(** Machine-check of [docs/resource-fault-injection-inventory.md] against the live
    [Ir.Resource_fault_injection.point] type, repository test files, and cited module hooks
    (gh-ocannl-778).

    {1 Recognised forms (closed boundary)}

    - {b Table structure}: Markdown pipe-table lines (stripped line starting and ending with ['|']),
      skipping the first header row and any separator row whose cells contain only ['-'], [':'], and
      whitespace. Every data row must have exactly four pipe-delimited cells
      ([Seam | Failure mode | Status and acceptance evidence | Remaining leg]).
    - {b Inline code tokens}: single-line, single-backtick spans ([`...`]) inside table data rows.
      Unclosed backticks on a row are refused; double-backtick fences and prose outside the pipe
      table are deliberately not read.
    - {b Point coverage}: the complete constructor set is derived directly from
      [Ir.Resource_fault_injection.all_of_point] (never a copied list). Every constructor must
      appear as a backticked token ([`<Point>`] or [`Resource_fault_injection.<Point>`]) in at least
      one table data row.
    - {b Qualified fault-injector references}: every backticked token starting with
      [Resource_fault_injection.] or [FI.] in a table row must name a live constructor of
      [all_of_point] or an exported identifier of [arrayjit/lib/resource_fault_injection.mli].
    - {b Cited repository paths}: every backticked token in a table row containing ['/'] and ending
      in [".ml"], [".mli"], [".md"], or [".sh"] must exist in the repository's [Source_inventory].
    - {b Covered-row test requirement}: every data row whose [Status and acceptance evidence] cell
      starts with ["Covered"] must cite at least one existing [".ml"] test file under ["test/"] or
      ["arrayjit/test/"] in that cell.
    - {b Cited [Autotune.*] hooks}: every backticked token starting with ["Autotune."] in a table
      row must name a value, module, type, or exception declared in [arrayjit/lib/autotune.mli]
      (parsed via [Ppxlib.Parse.interface]).

    Deliberately out of scope: relative Markdown links (enforced by [markdown_links]), prose outside
    the pipe table, bare unbackticked identifiers, and runtime test execution. *)

open Base

type finding =
  | Malformed_row of { line : int; cells : int }
  | Unclosed_backtick of { line : int }
  | Missing_point of { point : string }
  | Unknown_fi_member of { line : int; token : string }
  | Missing_cited_file of { line : int; path : string }
  | Covered_without_test of { line : int; seam : string }
  | Unknown_autotune_hook of { line : int; token : string }

let describe = function
  | Malformed_row { line; cells } ->
      Printf.sprintf "line %d: expected 4 pipe-table cells, found %d" line cells
  | Unclosed_backtick { line } -> Printf.sprintf "line %d: unclosed backtick span in table row" line
  | Missing_point { point } ->
      Printf.sprintf
        "Resource_fault_injection.point constructor `%s` is not named in any inventory table row"
        point
  | Unknown_fi_member { line; token } ->
      Printf.sprintf "line %d: `%s` is not a live constructor or export of Resource_fault_injection"
        line token
  | Missing_cited_file { line; path } ->
      Printf.sprintf "line %d: cited file `%s` does not exist in the repository" line path
  | Covered_without_test { line; seam } ->
      Printf.sprintf
        "line %d: Covered row %s cites no test `.ml` path under `test/` or `arrayjit/test/`" line
        seam
  | Unknown_autotune_hook { line; token } ->
      Printf.sprintf "line %d: cited hook `%s` is not declared in arrayjit/lib/autotune.mli" line
        token

(** All constructor names of {!Ir.Resource_fault_injection.point}, derived directly from the type.
*)
let derived_points () =
  List.map Ir.Resource_fault_injection.all_of_point ~f:(fun pt ->
      Sexp.to_string (Ir.Resource_fault_injection.sexp_of_point pt))

(** Collects all value, module, type, and exception names declared in an OCaml [.mli] signature. *)
let mli_exports content =
  let lexbuf = Lexing.from_string content in
  let sig_items = Ppxlib.Parse.interface lexbuf in
  let names = Hash_set.create (module String) in
  let rec visit_sig items =
    List.iter items ~f:(fun (item : Ppxlib.signature_item) ->
        match item.psig_desc with
        | Psig_value vd -> Hash_set.add names vd.pval_name.txt
        | Psig_type (_, tds) -> List.iter tds ~f:(fun td -> Hash_set.add names td.ptype_name.txt)
        | Psig_exception te -> Hash_set.add names te.ptyexn_constructor.pext_name.txt
        | Psig_module md ->
            Option.iter md.pmd_name.txt ~f:(Hash_set.add names);
            visit_mty md.pmd_type
        | Psig_modtype mtd ->
            Hash_set.add names mtd.pmtd_name.txt;
            Option.iter mtd.pmtd_type ~f:visit_mty
        | Psig_include id -> visit_mty id.pincl_mod
        | _ -> ())
  and visit_mty (mty : Ppxlib.module_type) =
    match mty.pmty_desc with
    | Pmty_signature items -> visit_sig items
    | Pmty_functor (_, body) -> visit_mty body
    | Pmty_with (body, _) -> visit_mty body
    | _ -> ()
  in
  visit_sig sig_items;
  Set.of_list (module String) (Hash_set.to_list names)

let is_separator_cell cell =
  let s = String.strip cell in
  (not (String.is_empty s))
  && String.for_all s ~f:(fun c -> Char.equal c '-' || Char.equal c ':' || Char.is_whitespace c)

let split_table_row line =
  let stripped = String.strip line in
  if not (String.is_prefix stripped ~prefix:"|" && String.is_suffix stripped ~suffix:"|") then None
  else
    let inner = String.sub stripped ~pos:1 ~len:(String.length stripped - 2) in
    Some (String.split inner ~on:'|' |> List.map ~f:String.strip)

let backtick_tokens text =
  let len = String.length text in
  let rec loop i acc =
    if i >= len then Ok (List.rev acc)
    else if Char.equal text.[i] '`' then
      match String.index_from text (i + 1) '`' with
      | None -> Error ()
      | Some j ->
          let tok = String.sub text ~pos:(i + 1) ~len:(j - i - 1) in
          let acc = if String.is_empty tok then acc else tok :: acc in
          loop (j + 1) acc
    else loop (i + 1) acc
  in
  loop 0 []

let is_source_path tok =
  String.contains tok '/'
  && (not (String.contains tok ' '))
  && List.exists [ ".ml"; ".mli"; ".md"; ".sh" ] ~f:(fun suffix -> String.is_suffix tok ~suffix)

let is_test_source_path tok =
  String.is_suffix tok ~suffix:".ml"
  && (String.is_prefix tok ~prefix:"test/" || String.is_prefix tok ~prefix:"arrayjit/test/")

let head_ident tok =
  let len = String.length tok in
  let stop =
    String.lfindi tok ~f:(fun _ c ->
        not (Char.is_alphanum c || Char.equal c '_' || Char.equal c '\''))
    |> Option.value ~default:len
  in
  String.prefix tok stop

type result = { data_rows : int; findings : finding list }

let check ~points ~fi_exports ~autotune_exports ~file_exists doc =
  let point_set = Set.of_list (module String) points in
  let seen_points = Hash_set.create (module String) in
  let findings = ref [] in
  let add f = findings := f :: !findings in
  let lines = String.split_lines doc in
  let header_seen = ref false in
  let data_rows = ref 0 in
  List.iteri lines ~f:(fun idx line ->
      let line_no = idx + 1 in
      match split_table_row line with
      | None -> ()
      | Some cells ->
          if List.for_all cells ~f:is_separator_cell then ()
          else if not !header_seen then (
            header_seen := true;
            if List.length cells <> 4 then
              add (Malformed_row { line = line_no; cells = List.length cells }))
          else (
            Int.incr data_rows;
            match cells with
            | [ seam; _failure_mode; status; _remaining ] ->
                let row_tokens_of cell =
                  match backtick_tokens cell with
                  | Ok toks -> toks
                  | Error () ->
                      add (Unclosed_backtick { line = line_no });
                      []
                in
                let all_cell_tokens = List.map cells ~f:row_tokens_of in
                let status_tokens = List.nth_exn all_cell_tokens 2 in
                let row_tokens = List.concat all_cell_tokens in
                (if String.is_prefix status ~prefix:"Covered" then
                   let has_existing_test =
                     List.exists status_tokens ~f:(fun tok ->
                         is_test_source_path tok && file_exists tok)
                   in
                   if not has_existing_test then add (Covered_without_test { line = line_no; seam }));
                List.iter row_tokens ~f:(fun tok ->
                    if Set.mem point_set tok then Hash_set.add seen_points tok;
                    (match
                       Option.first_some
                         (String.chop_prefix tok ~prefix:"Resource_fault_injection.")
                         (String.chop_prefix tok ~prefix:"FI.")
                     with
                    | Some rest ->
                        let member = head_ident rest in
                        if Set.mem point_set member then Hash_set.add seen_points member
                        else if not (Set.mem fi_exports member) then
                          add (Unknown_fi_member { line = line_no; token = tok })
                    | None -> ());
                    (match String.chop_prefix tok ~prefix:"Autotune." with
                    | Some rest ->
                        let member = head_ident rest in
                        if (not (String.is_empty member)) && not (Set.mem autotune_exports member)
                        then add (Unknown_autotune_hook { line = line_no; token = tok })
                    | None -> ());
                    if is_source_path tok && not (file_exists tok) then
                      add (Missing_cited_file { line = line_no; path = tok }))
            | _ -> add (Malformed_row { line = line_no; cells = List.length cells })));
  List.iter points ~f:(fun point ->
      if not (Hash_set.mem seen_points point) then add (Missing_point { point }));
  { data_rows = !data_rows; findings = List.rev !findings }
