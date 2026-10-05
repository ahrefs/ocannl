(* gh-ocannl-803: raw rename publication belongs in [Utils.Atomic_file].

   The fourth hand-rolled temp-then-rename in the repository used a fixed [<tar>.part] staging path,
   so concurrent dune runs streamed into one file. The first three copies had already moved behind
   [Atomic_file], but nothing kept a fifth from appearing. This source scan is that ratchet: every
   reference whose terminal identifier is [rename] must live in a named exemption carrying the
   reason it cannot use the helper. Matching the terminal name rather than four qualified paths is
   deliberate: [module FS = Sys; FS.rename] and [let open Sys in rename] are ordinary OCaml
   spellings of the same primitive. An unrelated API with that terminal name earns a reasoned
   exemption instead of becoming a silent escape hatch.

   This is deliberately an OCaml scan. [tools/test-run.sh] has the shell twin (the publication of
   its [last] pointer, near line 710 when this check was written); an OCaml parse cannot reach it,
   and the shell-side discipline remains tracked through gh-ocannl-607 rather than being silently
   claimed here.

   The reader uses ppxlib's parse tree, not text: comments and strings describing [Sys.rename] are
   not calls, while a source spelling the identifier over ordinary whitespace still is. *)

open Base
open Stdio

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

module Read = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse

type reference = { source : string; line : int; identifier : string list }
type exemption = { source : string; identifier : string list; reason : string }
type corpus = { generators : string list; generated : string list; sources : string list }

(* Each row exempts one occurrence with one exact identifier, not its whole source. The staleness
   claim below requires every row to be consumed exactly once, so another [rename] in the same file
   is an offender even when it uses the commit primitive's spelling (Codex P2, round 3). *)
let exempt_references =
  [
    {
      source = "arrayjit/lib/atomic_file.ml";
      identifier = [ "Stdlib"; "Sys"; "rename" ];
      reason =
        "implements Atomic_file's single commit primitive; every other OCaml publisher routes \
         through this module";
    };
  ]

let rename_references ~source content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        (match Read.longident_of expression with
        | Some identifier
          when Option.value_map (List.last identifier) ~default:false ~f:(String.equal "rename") ->
            found := { source; line = expression.pexp_loc.loc_start.pos_lnum; identifier } :: !found
        | _ -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  List.rev !found

(* Permanent controls for every spelling the terminal-name policy deliberately catches. The live
   repository currently exercises only the exempt [Stdlib.Sys.rename]; without these snippets the
   alias/open arms could disappear and shrink the census to a still-green singleton. The unrelated
   module is intentional too: terminal [rename] is conservative, whatever module owns it. *)
let matcher_cases =
  [
    ("Sys", "let publish a b = Sys.rename a b", [ "Sys"; "rename" ]);
    ("Stdlib.Sys", "let publish a b = Stdlib.Sys.rename a b", [ "Stdlib"; "Sys"; "rename" ]);
    ("Unix", "let publish a b = Unix.rename a b", [ "Unix"; "rename" ]);
    ("Stdlib.Unix", "let publish a b = Stdlib.Unix.rename a b", [ "Stdlib"; "Unix"; "rename" ]);
    ("module alias", "module FS = Sys\nlet publish a b = FS.rename a b", [ "FS"; "rename" ]);
    ("opened module", "open Sys\nlet publish a b = rename a b", [ "rename" ]);
    ("local open", "let publish a b = let open Sys in rename a b", [ "rename" ]);
    ("unrelated API", "let publish a b = Other.rename a b", [ "Other"; "rename" ]);
  ]

let generated_output source =
  match String.chop_suffix source ~suffix:".mll" with
  | Some stem -> stem ^ ".ml"
  | None -> (
      match String.chop_suffix source ~suffix:".mly" with
      | Some stem -> stem ^ ".ml"
      | None ->
          failwith (Printf.sprintf "source-generation input has no .mll or .mly suffix: %s" source))

let generated_control_source = "test/operations/atomic_file_rename_scan_fixture/raw_rename.ml"
let generated_control_identifier = [ "Sys"; "rename" ]

let parse_corpus arguments =
  let rec take_until marker acc = function
    | [] -> failwith (Printf.sprintf "missing corpus section %s" marker)
    | head :: tail when String.equal head marker -> (List.rev acc, tail)
    | head :: tail -> take_until marker (head :: acc) tail
  in
  match arguments with
  | "--generators" :: arguments ->
      let generators, arguments = take_until "--generated" [] arguments in
      let generated, sources = take_until "--sources" [] arguments in
      if List.exists [ generators; generated; sources ] ~f:List.is_empty then
        failwith "each corpus section must be present and nonempty";
      if List.exists sources ~f:(String.is_prefix ~prefix:"--") then
        failwith "unexpected corpus section after --sources";
      { generators; generated; sources }
  | _ -> failwith "corpus arguments must begin with --generators"

let emit_generated_dependencies prefix arguments =
  List.map arguments ~f:(fun source -> Stdlib.Filename.concat prefix (generated_output source))
  |> List.sort ~compare:String.compare |> List.iter ~f:print_endline

let require_sources ~fail sources =
  if List.is_empty sources then (
    fail "no OCaml sources among the arguments -- the rule's glob matches nothing";
    false)
  else true

let references_or_refusal ~fail ~source contents =
  match rename_references ~source contents with
  | references -> references
  | exception exn ->
      fail
        (Printf.sprintf "%s does not parse as OCaml, so this scan cannot vouch for it: %s" source
           (Exn.to_string exn));
      []

let refusal_control () =
  let source = "test/operations/atomic_file_rename_scan.ml" in
  let case label ~format run =
    let refused =
      match run () with
      | _ -> false
      | exception Failure _ ->
          Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
          true
    in
    Verdict.p label refused
  in
  case "a source-generation input with the wrong suffix is refused"
    ~format:"source-generation input has no .mll or .mly suffix: %s" (fun () ->
      generated_output "fixture.ml");
  case "a missing generated corpus section is refused" ~format:"missing corpus section %s"
    (fun () -> parse_corpus [ "--generators"; "fixture.mll" ]);
  case "an empty corpus section is refused"
    ~format:"each corpus section must be present and nonempty" (fun () ->
      parse_corpus [ "--generators"; "--generated"; "fixture.ml"; "--sources"; "fixture.ml" ]);
  case "a corpus section after sources is refused"
    ~format:"unexpected corpus section after --sources" (fun () ->
      parse_corpus
        [
          "--generators";
          "fixture.mll";
          "--generated";
          "fixture.ml";
          "--sources";
          "fixture.ml";
          "--extra";
        ]);
  case "a corpus without the generators header is refused"
    ~format:"corpus arguments must begin with --generators" (fun () -> parse_corpus []);
  let direct_case label ~format run =
    let refused = ref false in
    let fail _message =
      refused := true;
      Test_utils.Refusal_control_manifest.observe_failure ~source ~format
    in
    run fail;
    Verdict.p label !refused
  in
  direct_case "an empty OCaml source corpus reaches its refusal"
    ~format:"no OCaml sources among the arguments -- the rule's glob matches nothing" (fun fail ->
      ignore (require_sources ~fail [] : bool));
  direct_case "an invalid OCaml source reaches the parse refusal"
    ~format:"%s does not parse as OCaml, so this scan cannot vouch for it: %s" (fun fail ->
      ignore (references_or_refusal ~fail ~source:"bad.ml" "let =" : reference list));
  Test_utils.Refusal_control_manifest.print source

let () =
  if Array.length argv = 2 && String.equal argv.(1) "--refusal-control" then (
    refusal_control ();
    Stdlib.exit 0);
  if Array.length argv >= 3 && String.equal argv.(1) "--generated-deps" then (
    Array.to_list (Array.subo argv ~pos:3) |> emit_generated_dependencies argv.(2);
    Stdlib.exit 0);
  if Array.length argv < 8 then (
    eprintf
      "Usage: %s <workspace_root> --generators <input...> --generated <output...> --sources \
       <source...>\n"
      argv.(0);
    Stdlib.exit 1);
  let base = Test_utils.Dune_stanza_scan.base_dir argv.(1) in
  let corpus = Array.to_list (Array.subo argv ~pos:2) |> parse_corpus in
  let repo_relative path = Test_utils.Dune_stanza_scan.repo_relative base path in
  let generators = List.map corpus.generators ~f:repo_relative in
  let generated = List.map corpus.generated ~f:repo_relative in
  let expected_generated =
    List.map generators ~f:generated_output |> List.sort ~compare:String.compare
  in
  let generated = List.sort generated ~compare:String.compare in
  let arguments =
    corpus.generated @ corpus.sources |> List.map ~f:(fun path -> (repo_relative path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let sources = Read.sources_among (List.map arguments ~f:fst) in
  if not (require_sources ~fail:Verdict.fail sources) then Stdlib.exit 1;
  let references =
    List.concat_map sources ~f:(fun source ->
        let path = Map.find_exn on_disk source in
        references_or_refusal ~fail:Verdict.fail ~source (In_channel.read_all path))
  in
  let generated_control_references, references =
    List.partition_tf references ~f:(fun reference ->
        String.equal reference.source generated_control_source)
  in
  let exemption_key ({ source; identifier; _ } : exemption) = (source, identifier) in
  let reference_key ({ source; identifier; _ } : reference) = (source, identifier) in
  let remaining =
    ref
      (List.fold exempt_references ~init:Map.Poly.empty ~f:(fun counts exemption ->
           Map.Poly.update counts (exemption_key exemption) ~f:(function
             | None -> 1
             | Some count -> count + 1)))
  in
  let offenders =
    List.filter references ~f:(fun reference ->
        let key = reference_key reference in
        match Map.Poly.find !remaining key with
        | Some count when count > 0 ->
            remaining := Map.Poly.set !remaining ~key ~data:(count - 1);
            false
        | _ -> true)
  in
  List.iter offenders ~f:(fun { source; line; _ } ->
      eprintf
        "%s:%d: raw rename reference bypasses Utils.Atomic_file -- route publication through \
         Atomic_file, or add a named exemption with the reason this rename is unrelated\n"
        source line);
  eprintf "Discovered %d generator input(s) and derived %d generated OCaml output(s).\n"
    (List.length generators) (List.length generated);
  eprintf "Scanned %d OCaml sources; found %d raw rename reference(s).\n" (List.length sources)
    (List.length references + List.length generated_control_references);
  printf "Named raw rename exemptions:\n";
  List.iter exempt_references ~f:(fun { source; identifier; reason } ->
      printf "  %s: %s -- %s\n" source (String.concat ~sep:"." identifier) reason);
  printf "\n";
  Verdict.p_all ~min:8 "every qualified, aliased, opened, and unrelated rename spelling is detected"
    matcher_cases ~f:(fun (label, content, expected) ->
      match rename_references ~source:("synthetic " ^ label) content with
      | [ { identifier; _ } ] -> List.equal String.equal identifier expected
      | _ -> false);
  Verdict.p "every .mll and .mly input contributes its derived .ml output"
    (List.equal String.equal expected_generated generated);
  Verdict.p "the generated-source negative control is detected through the derived corpus"
    (List.mem generated generated_control_source ~equal:String.equal
    &&
    match generated_control_references with
    | [ { identifier; _ } ] -> List.equal String.equal identifier generated_control_identifier
    | _ -> false);
  Verdict.p_empty "no raw rename reference exists outside Atomic_file" ~over:references offenders;
  Verdict.p_all "every named raw rename exemption has a reason" exempt_references
    ~f:(fun { reason; _ } -> not (String.is_empty (String.strip reason)));
  Verdict.p_all "every named raw rename exemption matches exactly one live reference"
    exempt_references ~f:(fun exemption ->
      Map.Poly.find !remaining (exemption_key exemption) |> Option.value ~default:0 |> Int.equal 0);
  Test_utils.Refusal_control_manifest.print "atomic_file_rename_scan.ml"
