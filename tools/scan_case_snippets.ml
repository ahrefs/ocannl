(* The engine behind tools/typecheck-scan-cases.sh: type-check every OCaml snippet a scan's cases
   module feeds its scanner, the way docs/agent-notes/build-and-test.md (Repository scans) asks
   before an arm is trusted. The parser accepts programs the type checker rejects -- a duplicate
   top-level [module] parses cleanly -- and an arm over such a snippet pins the scanner's reading of
   code no compiler would build.

   WHICH SNIPPETS. A point-in-time reading of the cases module's parse tree, not of its text: a
   snippet is a string constant spelled as a quoted string with the [ocaml] delimiter
   ({ocaml|...|ocaml}), wherever it sits in the module. That delimiter is the shared case-table
   shape: the [*_scan_cases] tables spell their OCaml inputs that way and nothing else. A snippet
   spelled as an ordinary "..." literal (as the arms whose case IS an escape, a line continuation,
   must be), or with another delimiter, is not read. Text in comments is never read.

   ITS NAME. The first ordinary string constant (any delimiter but [ocaml]) among the elements of
   the tuple the snippet is an element of, or among the unlabelled arguments of the application it
   is an argument of, that precedes the snippet -- the two case-table shapes the tables use,
   [("name", {ocaml|...|ocaml}, expected)] and [case "name" {ocaml|...|ocaml} ~flag]. Otherwise the
   snippet is named by its line alone (and cannot be listed below). Every report carries the line.

   HOW IT IS CHECKED. Each snippet becomes its own compilation unit in a scratch directory, preceded
   by [open! <stub>] when a stub module is given (compiled first, never pasted in: an arm may rebind
   a stub's name, legal over an open and a duplicate definition beside a paste) and by a line
   directive, so a rejection points at the snippet's own line in the cases module. It is compiled
   with [ocamlc -stop-after typing -w -a -c]: warnings are not rejections. A library a snippet or
   the stub opens is reached with [-I] its directory of compiled interfaces ([-I "$(opam var
   lib)/base"]), which is all typing reads.

   KNOWN REJECTIONS. An arm may feed its scanner a parse shape no compiler accepts -- on purpose, to
   pin what the parser makes of a spelling the language does not produce, or as a finding not yet
   settled. The stub lists each such case by name in a floating attribute, [[@@@known_rejection
   ("case name", "text of the error")]], with its reason in a comment beside it, and every run
   prints it. Those snippets are compiled like the rest and must be rejected with output containing
   that text, which must not be empty (every rejection contains the empty text): rejected for
   another reason is a rejection like any other, and accepted, or a listed name that matches no
   checked snippet, is a stale listing; both fail the run.

   PROJECT SYNTAX IS REFUSED, BY NAME. A snippet carrying an extension node ([let%op], [[%cd ...]],
   any [%name]) or a [[@@deriving ...]] attribute needs the owning stanza's preprocessors, which
   bare ocamlc lacks; it is reported as REFUSED and not compiled. A snippet the parser rejects goes
   to the compiler anyway, whose syntax error is then the rejection.

   Exit status: 0 when every checked snippet was accepted (or rejected as the stub lists); 1 when
   any was rejected unlisted, or a listing is stale; 2 on a usage error, a stub that does not
   compile, or a cases module in which no snippet could be checked (no snippet at all, or every one
   refused) -- a run that checked nothing is not a clean verdict. *)

let usage =
  "usage: scan_case_snippets.exe [--stub STUB.ml|STUB.mli] [-I DIR]... [--verbose] CASES.ml..."

let die fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("typecheck-scan-cases: " ^ s);
      exit 2)
    fmt

type snippet = { line : int; column : int; text : string; mutable name : string option }

let read_file path =
  try In_channel.with_open_bin path In_channel.input_all
  with Sys_error msg -> die "cannot read %s" msg

let write_file path contents = Out_channel.with_open_bin path (fun oc -> output_string oc contents)

(* The snippets of one cases module, in source order. *)
let snippets_of ~path source =
  let open Ppxlib in
  let structure =
    try Parse.implementation (Lexing.from_string source)
    with exn -> die "%s does not parse: %s" path (Printexc.to_string exn)
  in
  let found = Hashtbl.create 64 in
  let order = ref [] in
  let snippet_key (e : expression) =
    match e.pexp_desc with
    | Pexp_constant (Pconst_string (_, loc, Some "ocaml")) -> Some loc.loc_start.pos_cnum
    | _ -> None
  in
  let plain_string (e : expression) =
    match e.pexp_desc with
    | Pexp_constant (Pconst_string (s, _, delimiter)) when delimiter <> Some "ocaml" -> Some s
    | _ -> None
  in
  (* Name every snippet among [children] by the first ordinary string before it. *)
  let name_siblings children =
    ignore
      (List.fold_left
         (fun name child ->
           match (name, plain_string child) with
           | None, Some s -> Some s
           | _ ->
               (match (name, snippet_key child) with
               | Some n, Some key -> Hashtbl.replace found key (`Named n)
               | _ -> ());
               name)
         None children)
  in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_constant (Pconst_string (text, loc, Some "ocaml")) ->
            let key = loc.loc_start.pos_cnum in
            let column = loc.loc_start.pos_cnum - loc.loc_start.pos_bol + String.length "{ocaml|" in
            order := (key, { line = loc.loc_start.pos_lnum; column; text; name = None }) :: !order
        | Pexp_tuple elements -> name_siblings elements
        | Pexp_apply (_, args) ->
            name_siblings (List.filter_map (function Nolabel, arg -> Some arg | _ -> None) args)
        | _ -> ());
        super#expression e
    end
  in
  collector#structure structure;
  List.rev_map
    (fun (key, snippet) ->
      (match Hashtbl.find_opt found key with
      | Some (`Named n) -> snippet.name <- Some n
      | None -> ());
      snippet)
    !order

(* The project syntax a snippet carries, if any: what bare ocamlc cannot expand. *)
let project_syntax text =
  let open Ppxlib in
  match Parse.implementation (Lexing.from_string text) with
  | exception _ -> None
  | structure ->
      let seen = ref None in
      let note s = if Option.is_none !seen then seen := Some s in
      let finder =
        object
          inherit Ast_traverse.iter as super

          method! extension ((name, _) as ext) =
            note ("%" ^ name.txt);
            super#extension ext

          method! attribute attr =
            (match attr.attr_name.txt with
            | "deriving" | "deriving_inline" -> note ("[@@" ^ attr.attr_name.txt ^ "]")
            | _ -> ());
            super#attribute attr
        end
      in
      finder#structure structure;
      !seen

(* The case names a stub lists in [[@@@known_rejection "name"]]. *)
let known_rejection path =
  let open Ppxlib in
  let source = read_file path in
  let lexbuf = Lexing.from_string source in
  let floating =
    try
      if Filename.check_suffix path ".mli" then
        List.filter_map
          (fun item -> match item.psig_desc with Psig_attribute a -> Some a | _ -> None)
          (Parse.interface lexbuf)
      else
        List.filter_map
          (fun item -> match item.pstr_desc with Pstr_attribute a -> Some a | _ -> None)
          (Parse.implementation lexbuf)
    with exn -> die "the stub %s does not parse: %s" path (Printexc.to_string exn)
  in
  List.filter_map
    (fun attr ->
      if attr.attr_name.txt <> "known_rejection" then None
      else
        let string e =
          match e.pexp_desc with Pexp_constant (Pconst_string (s, _, _)) -> Some s | _ -> None
        in
        match attr.attr_payload with
        | PStr [ { pstr_desc = Pstr_eval ({ pexp_desc = Pexp_tuple [ name; error ]; _ }, _); _ } ]
          when Option.is_some (string name) && not (List.mem (string error) [ None; Some "" ]) ->
            Some (Option.get (string name), Option.get (string error))
        | _ ->
            die
              "%s, line %d: [@@@known_rejection] takes a pair of strings: the case's name, and the \
               non-empty text the compiler's rejection must contain"
              path attr.attr_loc.loc_start.pos_lnum)
    floating

let run_quiet ~log command = Sys.command (command ^ " >" ^ Filename.quote log ^ " 2>&1") = 0

let () =
  let stub = ref None and includes = ref [] and verbose = ref false in
  let cases = ref [] in
  let absolute d = if Filename.is_relative d then Filename.concat (Sys.getcwd ()) d else d in
  let specs =
    [
      ("--stub", Arg.String (fun s -> stub := Some s), "STUB  a module compiled first and opened");
      ( "-I",
        Arg.String (fun d -> includes := absolute d :: !includes),
        "DIR  a directory of compiled interfaces (.cmi) a snippet or the stub opens" );
      ("--verbose", Arg.Set verbose, " also print every accepted snippet");
    ]
  in
  (try Arg.parse_argv Sys.argv specs (fun f -> cases := f :: !cases) usage with
  | Arg.Bad msg -> die "%s" (String.trim msg)
  | Arg.Help msg ->
      print_string msg;
      exit 0);
  let cases = List.rev !cases in
  if cases = [] then die "no cases module given\n%s" usage;
  let dir = Filename.temp_dir "typecheck-scan-cases" "" in
  at_exit (fun () -> ignore (Sys.command ("rm -rf " ^ Filename.quote dir)));
  let flags =
    String.concat " " (List.map (fun d -> "-I " ^ Filename.quote d) (dir :: List.rev !includes))
  in
  let ocamlc ~log args =
    run_quiet ~log (Printf.sprintf "cd %s && ocamlc -w -a %s %s" (Filename.quote dir) flags args)
  in
  let opener =
    match !stub with
    | None -> ""
    | Some path ->
        let ext = Filename.extension path in
        if ext <> ".ml" && ext <> ".mli" then die "--stub %s: expected a .ml or .mli file" path;
        let target = "scan_case_stub" ^ ext in
        write_file (Filename.concat dir target) (read_file path);
        let log = Filename.concat dir "stub.log" in
        if not (ocamlc ~log ("-c " ^ target)) then
          die "the stub %s does not compile:\n%s" path (read_file log);
        "open! Scan_case_stub\n"
  in
  let compile ~log source = ocamlc ~log ("-stop-after typing -c " ^ Filename.quote source) in
  let known = match !stub with Some path -> known_rejection path | None -> [] in
  let contains ~sub s =
    let n = String.length sub and m = String.length s in
    let rec at i = i + n <= m && (String.sub s i n = sub || at (i + 1)) in
    at 0
  in
  let rejected = ref 0 and refused = ref 0 and accepted = ref 0 and known_rejected = ref 0 in
  let stale = ref 0 and matched = ref [] and nothing_checked = ref [] in
  let counter = ref 0 in
  let print_text text =
    String.split_on_char '\n' (String.trim text)
    |> List.iter (fun l -> Printf.printf "          %s\n" l)
  in
  List.iter
    (fun path ->
      let snippets = snippets_of ~path (read_file path) in
      let checked_here = ref 0 in
      List.iter
        (fun s ->
          let name = match s.name with Some n -> n | None -> "(unnamed snippet)" in
          let label = Printf.sprintf "%s:%d: %s" path s.line name in
          match project_syntax s.text with
          | Some form ->
              incr refused;
              Printf.printf
                "REFUSED   %s\n\
                \          project syntax (%s): bare ocamlc cannot expand it; check it as a module \
                 of a scratch stanza copied from the owning one\n"
                label form
          | None -> (
              incr counter;
              incr checked_here;
              let unit = Printf.sprintf "arm_%d" !counter in
              let source = unit ^ ".ml" in
              (* The directive puts the snippet at its own line of the cases module, and the padding
                 at its own column, so a rejection points at the arm's text. *)
              write_file (Filename.concat dir source)
                (Printf.sprintf "%s# %d %S\n%s%s\n" opener s.line path (String.make s.column ' ')
                   s.text);
              let log = Filename.concat dir (unit ^ ".log") in
              let ok = compile ~log source in
              let output = read_file log in
              let listing = match s.name with Some n -> List.assoc_opt n known | None -> None in
              if Option.is_some listing then matched := name :: !matched;
              match (ok, listing) with
              | true, None ->
                  incr accepted;
                  if !verbose then Printf.printf "ok        %s\n" label
              | false, None ->
                  incr rejected;
                  Printf.printf "REJECTED  %s\n" label;
                  print_text output
              | false, Some error when contains ~sub:error output ->
                  incr known_rejected;
                  Printf.printf "REJECTED (known: the stub lists it)  %s\n" label;
                  if !verbose then print_text output
              | false, Some error ->
                  incr rejected;
                  Printf.printf "REJECTED  %s\n          not with the stub's listed %S:\n" label
                    error;
                  print_text output
              | true, Some _ ->
                  incr stale;
                  Printf.printf
                    "STALE     %s\n          accepted, though the stub lists it known_rejection\n"
                    label))
        snippets;
      if !checked_here = 0 then nothing_checked := path :: !nothing_checked)
    cases;
  List.iter
    (fun (name, _) ->
      if not (List.mem name !matched) then (
        incr stale;
        Printf.printf
          "STALE     the stub lists known_rejection %S, which names no checked snippet\n" name))
    known;
  Printf.printf
    "typecheck-scan-cases: %d snippet(s) checked: %d accepted, %d rejected, %d known rejection(s); \
     %d refused; %d stale listing(s)\n"
    !counter !accepted !rejected !known_rejected !refused !stale;
  if !nothing_checked <> [] then (
    List.iter
      (fun p -> Printf.eprintf "typecheck-scan-cases: %s: no snippet was checked\n" p)
      (List.rev !nothing_checked);
    exit 2);
  exit (if !rejected + !stale > 0 then 1 else 0)
