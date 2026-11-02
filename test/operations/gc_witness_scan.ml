(* Post-wave task post-1-task-12 follow-up (lukstafi/ocannl-staging#1581, gh-ocannl-1797): a test
   reads the GC through Gc_witness, and nowhere else.

   WHY. [Gc_witness] (test/support/gc_witness.mli) owns what a test reads off the GC: a [Weak]
   witness taken where no register of the caller holds the value, claims read over several
   collection rounds rather than one, and an allocation window that leaves out the finalisers of
   earlier garbage. A test that forces a collection or makes a weak observer of its own hand-rolls
   that recipe, and the copies it replaced (#1558's, then lacks_grad_reset_hint's) each rested on
   one collection or lacked the positive control. This scan refuses the hand-rolled forms, so a new
   one goes through the helper.

   WHAT IT READS -- the closed list. Every [.ml] under test/ (the source tree a dune sandbox holds),
   parsed with ppxlib, except test/support/gc_witness.ml itself, the one sanctioned site, whose uses
   are listed as the inventory. An identifier expression is REFUSED when it is: - [Gc.full_major],
   [Gc.major] or [Gc.compact] (each forces a major collection), or [Weak.create] (a weak observer),
   qualified by [Gc]/[Weak] or [Stdlib.Gc]/[Stdlib.Weak]; - the same through a module alias bound
   ANYWHERE in the file, [module G = Gc] or [let module G = Stdlib.Gc in …], alias chains included
   ([module H = G]); - a bare [full_major]/[major]/[compact] (resp. [create]) in a file that opens
   [Gc] (resp. [Weak]) or an alias of it ANYWHERE: [open Gc], [let open Gc in …], [Gc.( … )].
   Aliases and opens are resolved file-wide, not lexically: a file that shadows the name refuses
   more than it must, never less -- the loud direction. A mention in a comment or a string literal
   is no identifier, so it is not refused (the control holds both).

   WHAT IT DOES NOT READ. arrayjit/test/ is another package with no Gc_witness dependency, so it is
   outside the corpus. Not followed: [include Gc], a functor or first-class module carrying [Gc],
   [Gc.full_major] reached through [Obj] or a record field, and every other GC entry point --
   [Gc.minor] (uint4x32_stub_bounds resets the minor heap on purpose), [Gc.stat] and the counters,
   [Gc.finalise], [Ephemeron] -- none of which is the witness recipe.

   Live: [gc_witness_scan.exe <root> <generated>...] reads a dune sandbox holding test/. Control:
   [gc_witness_scan.exe --control] runs the reader on synthetic sources held here: each refused form
   beside its nearest accepted one, and each of this file's refusals routed (gh-ocannl-800). *)
open Base
open Stdio
open Verdict.Claims
module Inventory = Test_utils.Source_inventory

let sanctioned = "test/support/gc_witness.ml"
let in_corpus path = String.is_prefix path ~prefix:"test/" && String.is_suffix path ~suffix:".ml"

type target = Gc | Weak

let target_name = function Gc -> "Gc" | Weak -> "Weak"
let refused_values = function Gc -> [ "full_major"; "major"; "compact" ] | Weak -> [ "create" ]
let is_refused target value = List.mem (refused_values target) value ~equal:String.equal

let rec flatten = function
  | Ppxlib.Longident.Lident s -> Some [ s ]
  | Ldot (p, s) -> Option.map (flatten p) ~f:(fun p -> p @ [ s ])
  | Lapply _ -> None

(* A module path, by what it denotes: [Gc]/[Weak] themselves, or an alias of one. *)
let resolve aliases = function
  | [ "Gc" ] | [ "Stdlib"; "Gc" ] -> Some Gc
  | [ "Weak" ] | [ "Stdlib"; "Weak" ] -> Some Weak
  | [ name ] -> Map.find aliases name
  | _ -> None

(* One use: its line, its spelling, and the canonical [Gc.x]/[Weak.x] it denotes. *)
type use = { line : int; spelled : string; target : target; canonical : string }

let parse ~file source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf file;
  Ppxlib.Parse.implementation lexbuf

let uses ~file source =
  let structure = parse ~file source in
  (* Every [module N = P] and [let module N = P in], and every opened module path. *)
  let bindings = ref [] and opens = ref [] in
  let module_ident (me : Ppxlib.module_expr) =
    match me.pmod_desc with Pmod_ident { txt; _ } -> flatten txt | _ -> None
  in
  let collector =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! module_binding mb =
        (match (mb.pmb_name.txt, module_ident mb.pmb_expr) with
        | Some name, Some path -> bindings := (name, path) :: !bindings
        | _ -> ());
        super#module_binding mb

      method! open_declaration od =
        Option.iter (module_ident od.popen_expr) ~f:(fun path -> opens := path :: !opens);
        super#open_declaration od

      method! expression e =
        (match e.pexp_desc with
        | Pexp_letmodule ({ txt = Some name; _ }, me, _) ->
            Option.iter (module_ident me) ~f:(fun path -> bindings := (name, path) :: !bindings)
        | _ -> ());
        super#expression e
    end
  in
  collector#structure structure;
  let rec close aliases =
    let grown =
      List.fold !bindings ~init:aliases ~f:(fun aliases (name, path) ->
          match (Map.mem aliases name, resolve aliases path) with
          | false, Some target -> Map.set aliases ~key:name ~data:target
          | _ -> aliases)
    in
    if Map.length grown = Map.length aliases then aliases else close grown
  in
  let aliases = close (Map.empty (module String)) in
  let opened = List.filter_map !opens ~f:(resolve aliases) in
  let found = ref [] in
  let finder =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_ident { txt; loc } -> (
            let use target value =
              found :=
                {
                  line = loc.loc_start.pos_lnum;
                  spelled = String.concat ~sep:"." (Option.value_exn (flatten txt));
                  target;
                  canonical = target_name target ^ "." ^ value;
                }
                :: !found
            in
            match Option.map (flatten txt) ~f:List.rev with
            | Some (value :: (_ :: _ as rev_path)) -> (
                match resolve aliases (List.rev rev_path) with
                | Some target when is_refused target value -> use target value
                | _ -> ())
            | Some [ value ] ->
                Option.iter
                  (List.find opened ~f:(fun target -> is_refused target value))
                  ~f:(fun target -> use target value)
            | _ -> ())
        | _ -> ());
        super#expression e
    end
  in
  finder#structure structure;
  List.rev !found

(* One reading, its failures told apart (gh-ocannl-1292): a source that does not parse is the
   source's to fix, and any other exception is the reader's own defect. [read] is a thunk so the
   control can plant the defect. *)
let reading_or_refusal ~fail ~what read =
  match Test_utils.Scan_attempt.run read with
  | Scanned result -> Some result
  | Unparsed why ->
      fail
        (Printf.sprintf "%s does not parse as OCaml (%s), so its GC reads went unchecked" what why);
      None
  | Raised exn ->
      fail
        (Printf.sprintf
           "the GC-read reader raised %s on %s, which parses -- a defect in the scanner rather \
            than in the source"
           exn what);
      None

(* [sources] maps each relative path to its content. Returns each corpus file with its uses, the
   sanctioned site's included; every use elsewhere is refused through [fail]. *)
let analyze ~fail sources =
  let corpus = List.filter sources ~f:(fun (path, _) -> in_corpus path) in
  if List.is_empty corpus then
    fail "no source under test/ is among the sources read -- the rule's source tree matches nothing"
  else if not (List.exists corpus ~f:(fun (path, _) -> String.equal path sanctioned)) then
    fail
      "test/support/gc_witness.ml is not among the sources read -- the one sanctioned site moved, \
       so the exemption names nothing; follow it here";
  let reports =
    List.filter_map corpus ~f:(fun (file, source) ->
        Option.map
          (reading_or_refusal ~fail ~what:file (fun () -> uses ~file source))
          ~f:(fun uses -> (file, uses)))
  in
  List.iter reports ~f:(fun (file, uses) ->
      if not (String.equal file sanctioned) then
        List.iter uses ~f:(fun use ->
            match use.target with
            | Gc ->
                fail
                  (Printf.sprintf
                     "%s:%d: `%s` forces a collection outside test/support/gc_witness.ml -- a test \
                      reads the GC through Gc_witness (settle, reclaimed, kept, \
                      drained_finalisers), so that no claim rests on one collection"
                     file use.line use.spelled)
            | Weak ->
                fail
                  (Printf.sprintf
                     "%s:%d: `%s` makes a weak observer outside test/support/gc_witness.ml -- take \
                      it with Gc_witness.witness and read it with Gc_witness.reclaimed or \
                      Gc_witness.kept"
                     file use.line use.spelled)));
  reports

let control () =
  let source = "test/operations/gc_witness_scan.ml" in
  let witness_site = (sanctioned, "let settle () = Gc.full_major ()\nlet w = Weak.create 1\n") in
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
    if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
    p label routed
  in
  let refused label ~format ~fragment sources =
    refused_by label ~format ~fragment (fun fail -> ignore (analyze ~fail sources : _ list))
  in
  (* No refusal, and the sanctioned site's uses read -- so a reader gone blind cannot pass. *)
  let accepted label sources =
    let messages, reports = run_tree sources in
    List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    let sanctioned_uses =
      List.Assoc.find reports sanctioned ~equal:String.equal
      |> Option.value ~default:[]
      |> List.map ~f:(fun use -> use.canonical)
    in
    p label (Poly.equal (messages, sanctioned_uses) ([], [ "Gc.full_major"; "Weak.create" ]))
  in
  let gc_format =
    "%s:%d: `%s` forces a collection outside test/support/gc_witness.ml -- a test reads the GC \
     through Gc_witness (settle, reclaimed, kept, drained_finalisers), so that no claim rests on \
     one collection"
  in
  let weak_format =
    "%s:%d: `%s` makes a weak observer outside test/support/gc_witness.ml -- take it with \
     Gc_witness.witness and read it with Gc_witness.reclaimed or Gc_witness.kept"
  in
  let stray text = [ witness_site; ("test/operations/stray.ml", text) ] in
  accepted "the sanctioned site's own collections and weak observer are inventoried, not refused"
    [ witness_site ];
  accepted "a stray collection in a comment is not refused"
    (stray "(* Stdlib.Gc.full_major () *)\nlet x = 1\n");
  accepted "a stray collection and weak observer in string literals are not refused"
    (stray "let s = \"Stdlib.Gc.full_major ()\"\nlet t = {|Weak.create 1|}\n");
  accepted "the GC entry points outside the closed list are not refused"
    (stray
       "let () = Gc.minor ()\n\
        let s = Gc.stat ()\n\
        let w = Gc_witness.witness s\n\
        let alive w = Weak.check w 0\n\
        let () = Gc_witness.settle ()\n");
  accepted "a local full_major with no open of Gc is not refused"
    (stray "let full_major () = ()\nlet () = full_major ()\n");
  accepted "a source outside test/ is outside the corpus"
    [ witness_site; ("arrayjit/test/other_package.ml", "let () = Stdlib.Gc.full_major ()\n") ];
  refused "a stray Stdlib.Gc.full_major is refused, naming its file and line" ~format:gc_format
    ~fragment:"test/operations/stray.ml:2: `Stdlib.Gc.full_major` forces a collection"
    (stray "let x = 1\nlet () = Stdlib.Gc.full_major ()\n");
  refused "a stray Gc.compact is refused" ~format:gc_format
    ~fragment:"`Gc.compact` forces a collection"
    (stray "let () = Gc.compact ()\n");
  refused "a stray Weak.create is refused" ~format:weak_format
    ~fragment:"test/operations/stray.ml:1: `Stdlib.Weak.create` makes a weak observer"
    (stray "let w = Stdlib.Weak.create 1\n");
  refused "a collection through a module alias chain is refused" ~format:gc_format
    ~fragment:"`H.major` forces a collection"
    (stray "module G = Stdlib.Gc\nmodule H = G\nlet () = H.major ()\n");
  refused "a weak observer through a let-module alias is refused" ~format:weak_format
    ~fragment:"`W.create` makes a weak observer"
    (stray "let w = let module W = Weak in W.create 1\n");
  refused "a bare full_major under a local open of Gc is refused" ~format:gc_format
    ~fragment:"`full_major` forces a collection"
    (stray "let () = Stdlib.Gc.(full_major ())\n");
  refused "a bare create under an open of an alias of Weak is refused" ~format:weak_format
    ~fragment:"`create` makes a weak observer"
    (stray "module W = Weak\nopen W\nlet w = create 1\n");
  refused "an invalid OCaml source reaches the parse refusal"
    ~format:"%s does not parse as OCaml (%s), so its GC reads went unchecked"
    ~fragment:"test/operations/stray.ml does not parse as OCaml" (stray "let =");
  refused_by
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    ~format:
      "the GC-read reader raised %s on %s, which parses -- a defect in the scanner rather than in \
       the source" ~fragment:"raised Invalid_argument(\"planted\") on good.ml, which parses"
    (fun fail ->
      ignore
        (reading_or_refusal ~fail ~what:"good.ml" (fun () -> invalid_arg "planted") : unit option));
  refused "a run without corpus sources reaches the empty-corpus refusal"
    ~format:
      "no source under test/ is among the sources read -- the rule's source tree matches nothing"
    ~fragment:"no source under test/ is among the sources read"
    [ ("bin/moons_benchmark.ml", "let x = 1\n") ];
  refused "a corpus without the sanctioned site reaches the moved-exemption refusal"
    ~format:
      "test/support/gc_witness.ml is not among the sources read -- the one sanctioned site moved, \
       so the exemption names nothing; follow it here"
    ~fragment:"the one sanctioned site moved"
    [ ("test/operations/plain.ml", "let x = 1\n") ];
  Test_utils.Refusal_control_manifest.print "gc_witness_scan.ml"

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
      let others, site =
        List.partition_tf reports ~f:(fun (file, _) -> not (String.equal file sanctioned))
      in
      let site_uses =
        List.concat_map site ~f:(fun (_, uses) -> List.map uses ~f:(fun use -> use.canonical))
        |> List.dedup_and_sort ~compare:String.compare
      in
      eprintf "Test sources read: %d (not part of the golden).\n" (List.length reports);
      printf "The sanctioned site, %s, uses:\n" sanctioned;
      List.iter site_uses ~f:(printf "  %s\n");
      (* The floor sits well below the day's count (some 700 sources), so a moved tree or a blind
         glob fails it and an ordinary addition moves nothing. *)
      p_all ~min:400
        "every test source outside test/support/gc_witness.ml neither forces a collection nor \
         makes a weak observer"
        others ~f:(fun (_, uses) -> List.is_empty uses);
      (* The exemption's own canary: the site it names still holds what it is exempted for. *)
      p "the sanctioned site still forces collections and makes the weak observer"
        (List.mem site_uses "Gc.full_major" ~equal:String.equal
        && List.mem site_uses "Weak.create" ~equal:String.equal);
      Test_utils.Refusal_control_manifest.print "gc_witness_scan.ml"
  | _ -> Stdlib.exit 2
