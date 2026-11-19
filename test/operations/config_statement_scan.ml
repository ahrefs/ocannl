(* gh-ocannl-2354: a test states a configuration key with [Utils.state_config] (or
   [Utils.with_stated_config]), never by writing [Utils.config_file_args].

   WHY. [Utils.config_file_args] is the config FILE's table, the lowest source but the profile
   payloads and the defaults: the commandline and every [OCANNL_<KEY>] outrank it. A test that wrote
   its setting there ran under the caller's value whenever the caller exported the variable -- the
   recipe AGENTS.md gives for a one-off configuration change -- and nothing said so (26 test files
   did, lukstafi/ocannl-staging#2144 tripped on two). [Utils.state_config] ranks above every source
   and refuses a commandline argument that contradicts it, so a stated value is the value read. This
   scan keeps the table the file's.

   WHAT IT READS -- a fail-closed allowlist. Every [.ml] under test/ and arrayjit/test/ (the two
   test trees the rule's sandbox holds), parsed with ppxlib. A USE is an identifier expression whose
   last component is [config_file_args], under any qualifier or none: [Utils.config_file_args], an
   alias's [U.config_file_args], a bare [config_file_args] under an open of [Utils] -- and a local
   binding that merely shares the name, the loud direction. A use is ACCEPTED only as the first
   positional argument (labelled ones may precede it) of [Hashtbl.r] or [Base.Hashtbl.r] for a
   reader [r] of the closed list below (applied in full or partially, [~file:(Hashtbl.find
   Utils.config_file_args)]). The bare [Hashtbl.r] is accepted only in a file that binds no module
   named [Hashtbl] anywhere, and [Base.Hashtbl.r] only in one that binds no [Base] -- [module
   Hashtbl = …], [let module], a functor parameter, a first-class module a pattern unpacks -- since
   there the name may not be Base's: resolved file-wide, not lexically, the loud direction. Every
   other use is REFUSED, naming file and line: a [Hashtbl.set]/[add]/[remove]/[clear]/… on the
   table, the table bound to a name ([let t = Utils.config_file_args in …]) or handed to any other
   function. A mention in a comment or a string literal is no identifier, so it is not read (the
   control holds both). The files in [exemptions] are inventoried rather than refused, each with its
   reason, and an exemption whose file is gone or no longer has a refused use is itself refused
   (stale).

   WHAT IT DOES NOT READ. Sources outside the two test trees (the library reads the table in [Utils]
   itself; bin/ and benchmarks/ state configuration on their commandlines). An opened module that
   defines its own [Hashtbl] ([open Foo] with a [Foo.Hashtbl]) is trusted to be Base's or Core's --
   no test source opens one, and a [module Hashtbl] binding in the file itself is resolved. Not
   followed: [include Utils], a functor or first-class module carrying [Utils], or a write reached
   through [Obj].

   Live: [config_statement_scan.exe <root> <generated>...] reads a dune sandbox holding the test
   trees. Control: [config_statement_scan.exe --control] runs the reader on synthetic sources held
   here: each refused form beside its nearest accepted one, and each of this file's refusals routed
   (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Inventory = Test_scan.Source_inventory

let in_corpus path =
  (String.is_prefix path ~prefix:"test/" || String.is_prefix path ~prefix:"arrayjit/test/")
  && String.is_suffix path ~suffix:".ml"

(* The files a raw write is sanctioned in, each with the reason it is. *)
let exemptions =
  [
    ( "test/operations/config_statement_precedence.ml",
      "the negative control: it shows a raw write to the table outranked by an exported variable, \
       the hazard the setter exists for" );
  ]

let is_exempt path = List.Assoc.mem exemptions path ~equal:String.equal

(* The reads of a Base hash table that leave it as it is. *)
let readers =
  [
    "find";
    "find_exn";
    "mem";
    "length";
    "is_empty";
    "keys";
    "data";
    "to_alist";
    "iter";
    "iteri";
    "iter_keys";
    "fold";
    "count";
    "exists";
    "existsi";
    "for_all";
    "for_alli";
  ]

let rec flatten = function
  | Ppxlib.Longident.Lident s -> Some [ s ]
  | Ldot (p, s) -> Option.map (flatten p) ~f:(fun p -> p @ [ s ])
  | Lapply _ -> None

let ident_path (e : Ppxlib.expression) =
  match e.pexp_desc with Pexp_ident { txt; _ } -> flatten txt | _ -> None

let is_table_ident e =
  Option.exists (ident_path e) ~f:(fun path -> String.equal (List.last_exn path) "config_file_args")

(* Whether [fn] is a sanctioned reader, given the module names the file binds itself: the bare
   [Hashtbl.r] is Base's only where no [Hashtbl] is bound, and [Base.Hashtbl.r] only where no [Base]
   is. *)
let is_reader ~bound fn =
  let unbound name = not (List.mem bound name ~equal:String.equal) in
  match ident_path fn with
  | Some [ "Hashtbl"; r ] -> unbound "Hashtbl" && List.mem readers r ~equal:String.equal
  | Some [ "Base"; "Hashtbl"; r ] -> unbound "Base" && List.mem readers r ~equal:String.equal
  | _ -> false

(* One use the scan refuses: its line and its spelling. *)
type use = { line : int; spelled : string }

let parse ~file source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  Ppxlib.Parse.implementation lexbuf

let refused_uses ~file source =
  let structure = parse ~file source in
  (* The trusted module names the file binds anywhere: a module binding, a [let module], a functor
     parameter, a first-class module unpacked in a pattern. *)
  let bound = ref [] in
  let bind = function
    | Some (("Hashtbl" | "Base") as name) -> bound := name :: !bound
    | Some _ | None -> ()
  in
  let collector =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! module_binding mb =
        bind mb.pmb_name.txt;
        super#module_binding mb

      method! module_expr me =
        (match me.pmod_desc with Pmod_functor (Named ({ txt; _ }, _), _) -> bind txt | _ -> ());
        super#module_expr me

      method! pattern pat =
        (match pat.ppat_desc with Ppat_unpack { txt; _ } -> bind txt | _ -> ());
        super#pattern pat

      method! expression e =
        (match e.pexp_desc with Pexp_letmodule ({ txt; _ }, _, _) -> bind txt | _ -> ());
        super#expression e
    end
  in
  collector#structure structure;
  let bound = !bound in
  let found = ref [] in
  let finder =
    object (self)
      inherit Ppxlib.Ast_traverse.iter as super

      method! expression e =
        let first_positional args =
          List.find_mapi args ~f:(fun i (lbl, arg) ->
              match lbl with Ppxlib.Nolabel -> Some (i, arg) | _ -> None)
        in
        match e.pexp_desc with
        | Pexp_apply (fn, args) when is_reader ~bound fn -> (
            match first_positional args with
            | Some (i, table) when is_table_ident table ->
                (* The sanctioned read: its table argument is accepted, the rest is read on. *)
                self#expression fn;
                List.iteri args ~f:(fun j (_, arg) -> if j <> i then self#expression arg)
            | _ -> super#expression e)
        | Pexp_ident { txt; loc } when is_table_ident e ->
            found :=
              {
                line = loc.loc_start.pos_lnum;
                spelled = String.concat ~sep:"." (Option.value ~default:[] (flatten txt));
              }
              :: !found
        | _ -> super#expression e
    end
  in
  finder#structure structure;
  List.rev !found

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   control can plant the defect. *)
let reading_or_refusal ~fail ~what read =
  match Test_scan.Scan_attempt.run read with
  | Scanned result -> Some result
  | Unparsed why ->
      fail
        (Printf.sprintf
           "%s does not parse as OCaml (%s), so its writes to Utils.config_file_args went unchecked"
           what why);
      None
  | Raised exn ->
      fail
        (Printf.sprintf
           "the config-statement reader raised %s on %s, which parses -- a defect in the scanner \
            rather than in the source"
           exn what);
      None

(* [sources] maps each relative path to its content. Returns each corpus file with its refused uses,
   the exempted files' included; every use outside an exemption is refused through [fail]. *)
let analyze ~fail sources =
  let corpus = List.filter sources ~f:(fun (path, _) -> in_corpus path) in
  if List.is_empty corpus then
    fail
      "no source under test/ or arrayjit/test/ is among the sources read -- the rule's source \
       trees match nothing";
  let reports =
    List.filter_map corpus ~f:(fun (file, source) ->
        Option.map
          (reading_or_refusal ~fail ~what:file (fun () -> refused_uses ~file source))
          ~f:(fun uses -> (file, uses)))
  in
  List.iter reports ~f:(fun (file, uses) ->
      if not (is_exempt file) then
        List.iter uses ~f:(fun use ->
            fail
              (Printf.sprintf
                 "%s:%d: `%s` is written or handed on outside a read -- a test states a key with \
                  Utils.state_config or Utils.with_stated_config, which an ambient OCANNL_<KEY> \
                  cannot outrank (gh-ocannl-2354); the config-file table is only read, with \
                  Hashtbl.find, mem or another reader"
                 file use.line use.spelled)));
  (* Only a run that read some corpus judges the exemptions: an empty one has said so already. *)
  if not (List.is_empty corpus) then
    List.iter exemptions ~f:(fun (file, _) ->
        match List.Assoc.find reports file ~equal:String.equal with
        | Some (_ :: _) -> ()
        | Some [] | None ->
            fail
              (Printf.sprintf
                 "the exemption of %s is stale: the file is not read or no longer writes \
                  Utils.config_file_args -- delete its entry"
                 file));
  reports

let control () =
  let source = "test/operations/config_statement_scan.ml" in
  let exempt_site =
    ( "test/operations/config_statement_precedence.ml",
      "let () = Hashtbl.set Utils.config_file_args ~key:\"k\" ~data:\"v\"\n" )
  in
  let run_tree sources =
    let messages = ref [] in
    let reports = analyze ~fail:(fun m -> messages := m :: !messages) sources in
    (List.rev !messages, reports)
  in
  let refused_by label ~format ~fragment run =
    let messages = ref [] in
    run (fun m -> messages := m :: !messages);
    let messages = List.rev !messages in
    let routed =
      match messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if not routed then
      List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    if routed then Test_scan.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  let refused label ~format ~fragment sources =
    refused_by label ~format ~fragment (fun fail -> ignore (analyze ~fail sources : _ list))
  in
  (* No refusal, and the exempted site's write read -- so a reader gone blind cannot pass. *)
  let accepted label sources =
    let messages, reports = run_tree sources in
    List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    let exempt_uses =
      List.Assoc.find reports (fst exempt_site) ~equal:String.equal
      |> Option.value ~default:[]
      |> List.map ~f:(fun use -> use.spelled)
    in
    p label (Poly.equal (messages, exempt_uses) ([], [ "Utils.config_file_args" ]))
  in
  let write_format =
    "%s:%d: `%s` is written or handed on outside a read -- a test states a key with \
     Utils.state_config or Utils.with_stated_config, which an ambient OCANNL_<KEY> cannot outrank \
     (gh-ocannl-2354); the config-file table is only read, with Hashtbl.find, mem or another \
     reader"
  in
  let stray text = [ exempt_site; ("test/operations/stray.ml", text) ] in
  accepted "the exempted negative control's raw write is inventoried, not refused" [ exempt_site ];
  accepted "reads of the table, full and partial, are accepted"
    (stray
       "let a = Hashtbl.find Utils.config_file_args \"k\"\n\
        let b = Hashtbl.mem Utils.config_file_args \"k\"\n\
        let c = Base.Hashtbl.to_alist Utils.config_file_args\n\
        let () = Hashtbl.iter_keys ~f:print_endline Utils.config_file_args\n\
        let lookup = Hashtbl.find Utils.config_file_args\n\
        let r = Utils.resolve_config_value ~file:(Hashtbl.find Utils.config_file_args)\n");
  accepted "a Base.Hashtbl reader is accepted beside a Hashtbl module the file binds"
    (stray "module Hashtbl = struct end\nlet a = Base.Hashtbl.find Utils.config_file_args \"k\"\n");
  accepted "a statement through the setter is accepted"
    (stray
       "let () = Utils.state_config ~key:\"k\" ~data:\"v\"\n\
        let x = Utils.with_stated_config [ (\"k\", \"v\") ] (fun () -> 1)\n");
  accepted "a raw write in a comment and in string literals is not refused"
    (stray
       "(* Hashtbl.set Utils.config_file_args ~key ~data *)\n\
        let s = \"Hashtbl.set Utils.config_file_args\"\n\
        let t = {|Hashtbl.remove Utils.config_file_args k|}\n");
  accepted "a source outside the two test trees is outside the corpus"
    [ exempt_site; ("bin/other.ml", "let () = Hashtbl.set Utils.config_file_args ~key ~data\n") ];
  refused "a raw Hashtbl.set on the table is refused, naming its file and line" ~format:write_format
    ~fragment:"test/operations/stray.ml:2: `Utils.config_file_args` is written or handed on"
    (stray "let x = 1\nlet () = Hashtbl.set Utils.config_file_args ~key:\"k\" ~data:\"v\"\n");
  refused "a raw Hashtbl.remove on the table is refused" ~format:write_format
    ~fragment:"stray.ml:1: `Utils.config_file_args` is written"
    (stray "let () = Hashtbl.remove Utils.config_file_args \"k\"\n");
  refused "a raw write under arrayjit/test/ is refused" ~format:write_format
    ~fragment:"arrayjit/test/other.ml:1: `Utils.config_file_args`"
    [
      exempt_site;
      ("arrayjit/test/other.ml", "let () = Hashtbl.set Utils.config_file_args ~key ~data\n");
    ];
  refused "the table bound to a name is refused at the binding" ~format:write_format
    ~fragment:"`Utils.config_file_args` is written or handed on"
    (stray "let () =\n  let t = Utils.config_file_args in\n  Hashtbl.set t ~key:\"k\" ~data:\"v\"\n");
  refused "a write through a module alias is refused" ~format:write_format
    ~fragment:"`U.config_file_args` is written"
    (stray "module U = Utils\nlet () = Hashtbl.add_exn U.config_file_args ~key ~data\n");
  refused "a write to the bare table under a local open is refused" ~format:write_format
    ~fragment:"`config_file_args` is written"
    (stray "let () = Utils.(Hashtbl.set config_file_args ~key:\"k\" ~data:\"v\")\n");
  refused "a reader of a Hashtbl the file binds itself is refused" ~format:write_format
    ~fragment:"stray.ml:2: `Utils.config_file_args` is written"
    (stray
       "module Hashtbl = struct let find t _ = Base.Hashtbl.clear t end\n\
        let () = Hashtbl.find Utils.config_file_args \"k\"\n");
  refused "a reader of a Hashtbl a functor parameter binds is refused" ~format:write_format
    ~fragment:"stray.ml:3: `Utils.config_file_args` is written"
    (stray
       "module F (Hashtbl : sig val find : 'a -> string -> unit end) = struct\n\
       \  let () =\n\
       \    Hashtbl.find Utils.config_file_args \"k\"\n\
        end\n");
  refused "a Base.Hashtbl reader in a file binding its own Base is refused" ~format:write_format
    ~fragment:"stray.ml:2: `Utils.config_file_args` is written"
    (stray
       "module Base = struct module Hashtbl = struct let find t _ = ignore t end end\n\
        let () = Base.Hashtbl.find Utils.config_file_args \"k\"\n");
  refused "an exemption whose file no longer writes the table reaches the stale-exemption refusal"
    ~format:
      "the exemption of %s is stale: the file is not read or no longer writes \
       Utils.config_file_args -- delete its entry"
    ~fragment:"the exemption of test/operations/config_statement_precedence.ml is stale"
    [ (fst exempt_site, "let () = Utils.state_config ~key:\"k\" ~data:\"v\"\n") ];
  refused "an invalid OCaml source reaches the parse refusal"
    ~format:
      "%s does not parse as OCaml (%s), so its writes to Utils.config_file_args went unchecked"
    ~fragment:"test/operations/stray.ml does not parse as OCaml" (stray "let =");
  refused_by
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the config-statement reader raised %s on %s, which parses -- a defect in the scanner rather \
       than in the source" ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses"
    (fun fail ->
      ignore
        (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted")
          : use list option));
  refused "a run without corpus sources reaches the empty-corpus refusal"
    ~format:
      "no source under test/ or arrayjit/test/ is among the sources read -- the rule's source \
       trees match nothing"
    ~fragment:"no source under test/ or arrayjit/test/ is among the sources read"
    [ ("bin/moons_benchmark.ml", "let x = 1\n") ];
  Test_scan.Refusal_control_manifest.print "config_statement_scan.ml"

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--control" ] -> control ()
  | _ :: root :: generated ->
      let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
      let sources =
        Inventory.select inventory ~f:in_corpus
        |> List.map ~f:(fun (file : Inventory.file) ->
            (file.path, In_channel.read_all file.on_disk))
      in
      let reports = analyze ~fail sources in
      let others, exempt = List.partition_tf reports ~f:(fun (file, _) -> not (is_exempt file)) in
      eprintf "Test sources read: %d (not part of the golden).\n" (List.length reports);
      printf "Exempted sources, each with its reason:\n";
      List.iter exemptions ~f:(fun (file, why) ->
          let count =
            Option.value_map
              (List.Assoc.find exempt file ~equal:String.equal)
              ~default:0 ~f:List.length
          in
          printf "  %s (%d use%s) -- %s\n" file count (if count = 1 then "" else "s") why);
      (* The floor sits well below the day's count (some 750 sources), so a moved tree or a blind
         glob fails it and an ordinary addition moves nothing. *)
      p_all ~min:400
        "every test source outside the exemptions only reads Utils.config_file_args, and states \
         its configuration with Utils.state_config"
        others ~f:(fun (_, uses) -> List.is_empty uses);
      p_exists ~min:1 "the test sources read include arrayjit/test/" others ~f:(fun (file, _) ->
          String.is_prefix file ~prefix:"arrayjit/test/");
      Test_scan.Refusal_control_manifest.print "config_statement_scan.ml"
  | _ -> Stdlib.exit 2
