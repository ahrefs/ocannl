(** The inventory of every placement-provenance tag and of every file citing one (gh-ocannl-1015,
    gh-ocannl-1081): the checklist a change to a tag -- retiring it, renumbering it, or moving where
    it can fire -- reads.

    Whether a virtualizer rejection code can fire is decided by which passes run before the one that
    raises it, not by the raise site: gh-ocannl-483 put the algebraic rewrite tier ahead of
    [Low_level.optimize], one of its rewrites emits a hoisted local, and a refusal every description
    called defensive became reachable on the ordinary path. Its "cannot fire" claim then had six
    live copies across the library, the lowering doc, an agent note and a test header; the PR that
    moved the pass updated none of them, and the sweep that found them later missed one twice. The
    other families share the rest of that story: their numbers are allocated across nine modules by
    folklore, and a retired [N:reason] in prose read exactly like a live one.

    The tags are DERIVED, never listed here: {!Test_utils.Provenance_tag_scan} reads the families
    off [Tnode.provenance] and the tags off the sources -- its header states what a family, a tag
    and a citation are, and what is not read. The golden holds the families, the tags by family and
    minting function, the pinned number collisions, the tests' own tags, and every file citing a
    library tag with the numbers it cites. A new tag, a new file citing one, or a file that stops
    citing one each moves a line.

    Everything the reader refuses goes to stderr with a nonzero exit: a citation of a tag no source
    mints (a retired tag, or a typo), a number minted under two tags that is not pinned below, a
    rejection code minted in two functions, a test constructing a tag no library source mints under
    a name not reserved for tests. And the boundary test's phase table -- the one place the phase of
    a rejection code is written down -- must place each tag in the phase whose function mints it,
    which is what makes that table a restatement under test rather than beside the source. [phases]
    below is the bridge between the test's vocabulary and the functions. *)

open Base
open Stdio
module Scan = Test_utils.Provenance_tag_scan
module Inventory = Test_utils.Source_inventory

(* The owner of the families: the provenance type and the function rendering its constructors. *)
let type_source = "arrayjit/lib/tnode.ml"
let type_name = "provenance"
let renderer = "provenance_to_string"
let table_source = "test/operations/virtual_rejection_boundary.ml"

(* The relayed family whose phases the boundary test's table names. *)
let phase_family = "Non_virtual"

(* This scan's own golden cites every tag by construction, so listing it would say nothing; its
   cases file and that file's golden spell invented tags on purpose, each one a stale citation by
   design. *)
let own_files =
  [
    "test/operations/provenance_tag_inventory.expected";
    "test/operations/provenance_tag_scan_cases.ml";
    "test/operations/provenance_tag_scan_cases.expected";
  ]

(* The boundary test's phase constructors, and the function whose codes each one names. The cap
   phase is absent on purpose: the test derives it from [Low_level.is_cap_provenance], and caps are
   provenance constructors rather than codes. *)
(* Consumption is [inline_computation]'s instantiation core, which mints its codes (gh-ocannl-1011). *)
let phases = [ ("Store", "check_and_store_virtual"); ("Consumption", "instantiate_computations") ]

(* The number collisions that predate the inventory (gh-ocannl-1081), pinned rather than renumbered:
   a renumbering would churn every golden and page citing the tags. A new collision, a third tag on
   one of these, or a pin that stops colliding is refused. *)
let pinned =
  [
    ( 176,
      [ "176:stage-packed-tile"; "176:tensorize-acc-tile" ],
      "two schedule tile placements that were one integer before gh-ocannl-609" );
    ( 178,
      [ "178:fission-live-range-crossing"; "178:mma-fragment" ],
      "a fission promotion and a tensor-core fragment placement that were one integer before \
       gh-ocannl-609" );
  ]

(* Read, but not part of the checklist: text that describes the code as it stood on a date, and that
   a later change does not revise. A prefix no file lives under is stale and fails. *)
let records =
  [
    ("docs/proposals/", "design proposals, each describing the code as it stood when written");
    ("docs/in-progress/", "links into docs/proposals/, the proposals currently being worked on");
  ]

let scan ~records ~pinned root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let read (file : Inventory.file) = In_channel.read_all file.on_disk in
  let own path = List.mem own_files path ~equal:String.equal in
  let type_text =
    List.find (Inventory.files inventory) ~f:(fun file -> String.equal file.path type_source)
    |> Option.map ~f:read
  in
  let shape = Option.bind type_text ~f:(Scan.type_shape ~type_name) in
  let carriers = Option.value_map shape ~default:[] ~f:(fun s -> s.Scan.carriers) in
  let rendered =
    Option.value_map type_text ~default:[] ~f:(Scan.renderings ~renderer ~source:type_source)
  in
  let unparsed = ref [] in
  let candidates =
    Inventory.select inventory ~f:(fun path ->
        String.is_suffix path ~suffix:".ml" && not (own path))
    |> List.filter_map ~f:(fun file ->
        let content = read file in
        (* A source reaches a carrier by spelling it, or relays one through a local exception whose
           payload a caller elsewhere hands on. *)
        Option.some_if
          (List.exists (Scan.exception_keyword :: carriers) ~f:(fun substring ->
               String.is_substring content ~substring))
          (file.path, content))
  in
  let parses f (path, content) =
    match f content with
    | x -> Some x
    | exception _ ->
        unparsed := path :: !unparsed;
        None
  in
  (* The modules declaring a constructor of a carrier's name that is not the owner's. *)
  let foreign =
    List.filter_map candidates ~f:(fun ((path, _) as candidate) ->
        if String.equal path type_source then None
        else
          match parses (Scan.declares_own_carrier ~carriers) candidate with
          | Some true -> Some (Scan.module_of_path path)
          | _ -> None)
  in
  let unparsed_once = !unparsed in
  let reads =
    List.filter_map candidates ~f:(fun ((path, _) as candidate) ->
        if List.mem unparsed_once path ~equal:String.equal then None
        else parses (Scan.read_source ~carriers ~foreign ~source:path) candidate)
  in
  let resolved, malformed = Scan.resolve reads in
  let mints = Scan.merge (rendered @ resolved) in
  let library, tests = List.partition_tf mints ~f:(fun m -> not (Scan.is_test_source m.source)) in
  let test_tags =
    List.filter_map tests ~f:(fun m ->
        Option.some_if
          (Scan.is_test_tag m.tag
          && not (List.exists library ~f:(fun l -> String.equal l.tag m.tag)))
          m.tag)
    |> List.dedup_and_sort ~compare:(fun a b ->
        match Option.compare Int.compare (Scan.tag_number a) (Scan.tag_number b) with
        | 0 -> String.compare a b
        | c -> c)
  in
  let in_record path = List.exists records ~f:(fun (prefix, _) -> String.is_prefix path ~prefix) in
  let files =
    List.filter_map (Inventory.files inventory) ~f:(fun file ->
        let content = read file in
        if in_record file.path || own file.path || String.contains content '\000' then None
        else
          let mention = Scan.mentions ~mints ~test_tags content in
          if List.is_empty mention.named && List.is_empty mention.stale then None
          else Some (file.path, mention))
  in
  let table =
    Option.bind
      (List.find (Inventory.files inventory) ~f:(fun file -> String.equal file.path table_source))
      ~f:(fun file -> Scan.phase_table (read file))
  in
  let paths = List.map (Inventory.files inventory) ~f:(fun (file : Inventory.file) -> file.path) in
  List.iter
    (List.map (List.rev !unparsed) ~f:(fun path ->
         path ^ ": does not parse as OCaml, so the tags it mints are unread")
    @ Scan.family_violations
        ~identities:(Option.value_map type_text ~default:[] ~f:(Scan.identity_renderings ~renderer))
        ~type_source ~shape ~mints ()
    @ Scan.violations ~malformed ~mints
        ~pinned:(List.map pinned ~f:(fun (n, tags, _) -> (n, tags)))
        ~files ()
    @ Scan.table_violations ~mints ~exn:phase_family ~table_source ~table ~phases
    @ Scan.stale_records ~records paths)
    ~f:Verdict.fail;
  let exceptions =
    List.filter_map library ~f:(fun m ->
        match m.family with Relayed r -> Some (r.exn, r.via) | _ -> None)
    |> List.dedup_and_sort ~compare:Poly.compare
  in
  printf "Families, derived from the constructors of `%s` in %s:\n" type_name type_source;
  Option.iter shape ~f:(fun (s : Scan.shape) ->
      List.iter s.rendered ~f:(fun c -> printf "  %s -- one tag, its case of %s\n" c renderer);
      List.iter s.carriers ~f:(fun c ->
          printf "  %s -- every tag literal it is applied to\n" c;
          List.iter exceptions ~f:(fun (exn, via) ->
              if String.equal via c then
                printf
                  "  %s via %s -- every tag literal in the scope of a local `%s` exception whose \
                   handler hands its payload to %s, directly or through a caller\n"
                  c exn exn c));
      List.iter s.composite ~f:(fun c -> printf "  %s -- composes provenances, mints none\n" c));
  printf "Library tags, by family and by the function minting them:\n";
  let rank (m : Scan.mint) =
    match m.family with Rendered _ -> 0 | Applied _ -> 1 | Relayed _ -> 2
  in
  let group_label (m : Scan.mint) =
    match m.family with Rendered _ -> "Constructors" | f -> Scan.family_label f
  in
  let by_origin (a : Scan.mint) (b : Scan.mint) =
    Poly.compare
      (rank a, group_label a, a.source, a.minter)
      (rank b, group_label b, b.source, b.minter)
  in
  List.stable_sort library ~compare:by_origin
  |> List.group ~break:(fun a b -> by_origin a b <> 0)
  |> List.iter ~f:(fun group ->
      let first : Scan.mint = List.hd_exn group in
      printf "%s -- %s, %s:\n" (group_label first) first.source first.minter;
      List.dedup_and_sort group ~compare:(fun (a : Scan.mint) b ->
          Poly.compare (a.number, a.tag) (b.number, b.tag))
      |> List.iter ~f:(fun (m : Scan.mint) ->
          match m.family with
          | Rendered c -> printf "  %s (%s)\n" m.tag c
          | _ -> printf "  %s\n" m.tag));
  printf "Numbers minted under more than one library tag -- pinned; a new one is refused:\n";
  List.iter pinned ~f:(fun (_, tags, reason) ->
      printf "  %s -- %s\n" (String.concat ~sep:" " tags) reason);
  printf "Test tags, spelled `N:%s<reason>` and outside the library's number space:\n"
    Scan.test_prefix;
  List.iter test_tags ~f:(printf "  %s\n");
  printf "The phase table in %s is held to those functions:\n" table_source;
  List.iter phases ~f:(fun (phase, minter) -> printf "  %s -- %s\n" phase minter);
  printf "Files citing a library tag -- as the tag, %s -- the checklist for a change to one:\n"
    (String.concat ~sep:", "
       (List.map exceptions ~f:(fun (exn, _) -> "`" ^ exn ^ " N`")
       @ [ "`" ^ Scan.provenance_word ^ " N`" ]));
  List.iter files ~f:(fun (path, (mention : Scan.mention)) ->
      if not (List.is_empty mention.named) then
        printf "%s --%s\n" path
          (String.concat (List.map mention.named ~f:(fun n -> " " ^ Int.to_string n))));
  printf "Read, not part of the checklist:\n";
  List.iter records ~f:(fun (prefix, reason) -> printf "  %s -- %s\n" prefix reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~records:[] ~pinned:[] root []
  | _ :: root :: generated -> scan ~records ~pinned root generated
  | _ -> Stdlib.exit 2
