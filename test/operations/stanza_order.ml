(** A test directory's stanzas stay in order of test name, so that independent test additions do not
    meet at the end of a file (gh-ocannl-2193).

    Every new test in [test/operations] used to append its stanzas at the end of one file,
    [test/operations/dune]. Two branches that each add a test then conflict textually though they
    share nothing, GitHub builds no merge-ref jobs for a conflicting pull request, and the branch
    that loses the race rebases after a green head and pays a whole verification cycle: 14 of the 46
    workers of wave-20261111 did, several after their reviewed head was green. The workaround they
    found -- move the block beside related tests -- holds only until the next append.

    The layout that removes the append point is an order every stanza has exactly one place in.
    Below the marker comment a file carries, top-level stanzas are in non-decreasing order of the
    {!group} of the test they belong to, so a new test goes where its name sorts and two additions
    conflict only when their names are neighbours. The directory's own machinery (the config copy,
    the ambient gate, the aggregate aliases with their explanations) sits above the marker and is
    not ordered. The same rule reaches the other append points of a test addition: the members of
    every aggregate alias ([slow], [scans], the hand-written [runtest] lists) are sorted too, above
    the marker and below it.

    A file opts in by carrying the marker, and {!required} lists the files that must carry it: a
    marker dropped in a merge would otherwise turn the whole check off silently.

    The decision is {!findings}, a pure function of a dune file's text, so the controls run the code
    the live check runs, on files built to be refused. The scan reads top-level Dune stanzas and
    explicit alias dependencies. It leaves include expansion and nested subdir bodies to Dune. *)

open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Dune_stanza_scan
module Refusals = Test_scan.Refusal_control_manifest

let source = "stanza_order.ml"
let argv = Test_scan.Scan_argv.expand Stdlib.Sys.argv

(** The comment line that opens a file's ordered part, as {!Scan.read_raw} hands a comment over: the
    text after its [;]. *)
let marker = " ==== Tests, sorted by name (gh-ocannl-2193) ===="

(** The files that must carry the marker. *)
let required = [ "test/operations/dune" ]

(** The prefixes of the per-test alias families: [runtest-<name>], [slow-<name>], [train-<name>]. *)
let alias_prefixes = [ "runtest-"; "slow-"; "train-" ]

(** The test a per-test alias belongs to. An alias of another directory is spelled with its path
    ([../../benchmarks/runtest-x]), and names its test by its last component. *)
let test_of_alias alias =
  let name = List.last_exn (String.split alias ~on:'/') in
  List.find_map alias_prefixes ~f:(fun prefix -> String.chop_prefix name ~prefix)

let first_atom = function Some (Sexp.Atom a :: _) -> Some a | _ -> None

(** The name a stanza sorts under: its test's. [None] is a stanza this rule cannot place. *)
let key stanza =
  match Scan.head stanza with
  | Some ("test" | "tests" | "executable" | "executables" | "library") ->
      List.hd (Scan.names_of stanza)
  | Some "rule" -> (
      let aliases = Scan.aliases_of stanza in
      let target =
        match first_atom (Scan.field stanza "target") with
        | Some _ as t -> t
        | None -> first_atom (Scan.field stanza "targets")
      in
      match List.find_map aliases ~f:test_of_alias with
      | Some _ as test -> test
      | None -> ( match target with Some _ -> target | None -> List.hd aliases))
  | Some "alias" -> (
      match Option.bind (Scan.alias_stanza_name stanza) ~f:test_of_alias with
      | Some _ as test -> test
      | None -> List.find_map (Scan.alias_deps stanza) ~f:test_of_alias)
  | _ -> None

(** A key cut at its first [-] or [.]: a test's variants ([foo-tf32]) and its files ([foo.actual])
    sort with the test, in any order among themselves. Test names use neither character. *)
let group key =
  match String.lfindi key ~f:(fun _ c -> Char.equal c '-' || Char.equal c '.') with
  | Some i -> String.prefix key i
  | None -> key

(** What [content], the text of the dune file [path], breaks -- nothing for a file that does not
    carry the marker, unless [must_carry]. *)
let findings ?(must_carry = false) ~path content =
  let forms, comments = Scan.read_raw content in
  let stanzas = List.zip_exn (Scan.stanzas content) forms in
  let line_of offset = Scan.line_of content offset in
  let markers =
    List.filter comments ~f:(fun (c : Scan.comment) -> String.equal c.comment_text marker)
  in
  let describe stanza =
    match (Scan.head stanza, key stanza) with
    | Some head, Some key -> Printf.sprintf "the (%s) stanza of `%s`" head key
    | Some head, None -> Printf.sprintf "a (%s) stanza" head
    | None, _ -> "a form that is no stanza"
  in
  (* Aggregate members: every alias stanza's `(alias ...)` dependencies, strictly increasing as
     written, so a new member has one place and a duplicate is refused. *)
  let unsorted_members =
    List.filter_map stanzas ~f:(fun (stanza, (raw : Scan.raw_form)) ->
        match Scan.alias_stanza_name stanza with
        | None -> None
        | Some name ->
            let members = Scan.alias_deps stanza in
            let rec first_break = function
              | a :: (b :: _ as rest) ->
                  if String.compare a b < 0 then first_break rest else Some (a, b)
              | _ -> None
            in
            Option.map (first_break members) ~f:(fun (a, b) ->
                Refusals.fail
                  "%s:%d: the members of the `%s` alias are not in order: `%s` is listed before \
                   `%s`. Keep the list sorted, each member once, so that two branches adding one \
                   do not both append to it (gh-ocannl-2193)"
                  path (line_of raw.raw_start) name a b))
  in
  match markers with
  | [] ->
      if must_carry then
        [
          Refusals.fail
            "%s does not carry the marker comment `;%s` that opens its sorted part, so nothing \
             holds its test stanzas in order (gh-ocannl-2193)"
            path marker;
        ]
      else []
  | _ :: _ :: _ ->
      [
        Refusals.fail
          "%s carries the marker comment `;%s` %d times: which stanzas are ordered is ambiguous"
          path marker (List.length markers);
      ]
  | [ m ] ->
      let body =
        List.filter stanzas ~f:(fun (_, (raw : Scan.raw_form)) -> raw.raw_start > m.comment_start)
      in
      let unplaced =
        List.filter_map body ~f:(fun (stanza, (raw : Scan.raw_form)) ->
            match key stanza with
            | Some _ -> None
            | None ->
                Some
                  (Refusals.fail
                     "%s:%d: %s sits in the sorted part of the file and names no test to sort \
                      under: move it above the marker, beside the directory's other machinery, or \
                      into the stanza it serves"
                     path (line_of raw.raw_start) (describe stanza)))
      in
      let placed =
        List.filter_map body ~f:(fun (stanza, raw) ->
            Option.map (key stanza) ~f:(fun key -> (stanza, raw, group key)))
      in
      (* Where a group belongs: before the first stanza whose group sorts after it. *)
      let place_of g =
        match List.find placed ~f:(fun (_, _, other) -> String.compare other g > 0) with
        | Some (stanza, (raw : Scan.raw_form), _) ->
            Printf.sprintf "before %s (line %d)" (describe stanza) (line_of raw.raw_start)
        | None -> "at the end of the file"
      in
      let rec out_of_place = function
        | (above, (above_raw : Scan.raw_form), g_above)
          :: ((stanza, (raw : Scan.raw_form), g) :: _ as rest) ->
            if String.compare g_above g <= 0 then out_of_place rest
            else
              Refusals.fail
                "%s:%d: %s follows %s (line %d), which sorts after it. Below the marker a stanza \
                 sits where its test's name sorts, never at the end of the file: whichever of the \
                 two is misplaced, `%s` belongs %s and `%s` belongs %s (gh-ocannl-2193)"
                path (line_of raw.raw_start) (describe stanza) (describe above)
                (line_of above_raw.raw_start) g (place_of g) g_above (place_of g_above)
              :: out_of_place rest
        | _ -> []
      in
      unsorted_members @ unplaced @ out_of_place placed

(** How many stanzas [content] holds below its marker: the live check's floor. *)
let ordered_count content =
  let forms, comments = Scan.read_raw content in
  match List.filter comments ~f:(fun (c : Scan.comment) -> String.equal c.comment_text marker) with
  | [ m ] -> List.count forms ~f:(fun (raw : Scan.raw_form) -> raw.raw_start > m.comment_start)
  | _ -> 0

(* {1 Controls} *)

let file stanzas = String.concat ~sep:"\n\n" stanzas ^ "\n"
let marker_line = ";" ^ marker
let test name = Printf.sprintf "(test\n (name %s))" name
let exe name = Printf.sprintf "(executable\n (name %s))" name

let actual name =
  Printf.sprintf
    "(rule\n (target %s.actual)\n (action\n  (with-stdout-to\n   %%{target}\n   (run ./x.exe))))"
    name

let diff name =
  Printf.sprintf "(rule\n (alias runtest-%s)\n (action\n  (diff %s.expected %s.actual)))" name name
    name

let aggregate name members =
  Printf.sprintf "(alias\n (name %s)\n (deps\n%s))" name
    (String.concat ~sep:"\n" (List.map members ~f:(fun m -> Printf.sprintf "  (alias %s)" m)))

let machinery = "(copy_files\n (files ../config/ocannl_config))"
let found ?must_carry content = findings ?must_carry ~path:"t/dune" content

let refused ?must_carry ~format ~message content =
  let findings = found ?must_carry content in
  Refusals.check_failures ~source:("test/operations/" ^ source) ~formats:[ format ] findings
  &&
  match findings with
  | [ finding ] -> String.is_substring finding.message ~substring:message
  | _ -> false

let controls () =
  let sorted = [ machinery; marker_line; test "alpha"; test "beta"; test "delta" ] in
  let in_order = file sorted in
  let appended = file (sorted @ [ test "charlie" ]) in
  let in_place =
    file [ machinery; marker_line; test "alpha"; test "beta"; test "charlie"; test "delta" ]
  in
  let one_group =
    file
      [
        marker_line;
        test "alpha";
        exe "beta";
        actual "beta-wide";
        diff "beta-wide";
        actual "beta";
        diff "beta";
        aggregate "runtest" [ "runtest-beta"; "runtest-beta-wide" ];
        test "beta_more";
      ]
  in
  let above = file [ test "zeta"; test "alpha"; machinery; marker_line; test "beta" ] in
  let unmarked = file [ test "zeta"; test "alpha" ] in
  (* Each accepting control is quantified over the stanzas of its file, so "nothing found" is a
     statement about stanzas that were read and not about an empty file. *)
  p_empty "a file whose tests are in order of name is accepted" ~over:(Scan.stanzas in_order)
    (found in_order);
  let misplaced_format =
    "%s:%d: %s follows %s (line %d), which sorts after it. Below the marker a stanza sits where \
     its test's name sorts, never at the end of the file: whichever of the two is misplaced, `%s` \
     belongs %s and `%s` belongs %s (gh-ocannl-2193)"
  in
  let members_format =
    "%s:%d: the members of the `%s` alias are not in order: `%s` is listed before `%s`. Keep the \
     list sorted, each member once, so that two branches adding one do not both append to it \
     (gh-ocannl-2193)"
  in
  p "a test appended at the end of the file is refused, with the place it belongs"
    (refused ~format:misplaced_format
       ~message:
         "t/dune:15: the (test) stanza of `charlie` follows the (test) stanza of `delta` (line \
          12), which sorts after it"
       appended
    && refused ~format:misplaced_format
         ~message:"`charlie` belongs before the (test) stanza of `delta` (line 12)" appended);
  p_empty "the same test in its place is accepted" ~over:(Scan.stanzas in_place) (found in_place);
  p_empty "a test's executable, its variant rules and its golden diffs are one group, in any order"
    ~over:(Scan.stanzas one_group) (found one_group);
  p "a rule sorts under its per-test alias, so one filed under another test's name is refused"
    (refused ~format:misplaced_format
       ~message:"the (rule) stanza of `alpha` follows the (executable) stanza of `beta`"
       (file [ marker_line; exe "beta"; diff "alpha"; actual "beta" ]));
  p_empty "stanzas above the marker are not ordered" ~over:(Scan.stanzas above) (found above);
  p "a stanza below the marker that names no test is refused"
    (refused
       ~format:
         "%s:%d: %s sits in the sorted part of the file and names no test to sort under: move it \
          above the marker, beside the directory's other machinery, or into the stanza it serves"
       ~message:"a (copy_files) stanza sits in the sorted part of the file"
       (file [ marker_line; test "alpha"; machinery; test "beta" ]));
  p "an aggregate alias whose members are out of order is refused, above the marker too"
    (refused ~format:members_format
       ~message:
         "the members of the `slow` alias are not in order: `slow-b` is listed before `slow-a`"
       (file [ aggregate "slow" [ "slow-b"; "slow-a" ]; marker_line; test "alpha" ]));
  p "an aggregate alias listing a member twice is refused"
    (refused ~format:members_format ~message:"`slow-a` is listed before `slow-a`"
       (file [ aggregate "slow" [ "slow-a"; "slow-a" ]; marker_line; test "alpha" ]));
  p_empty "a file without the marker is left alone" ~over:(Scan.stanzas unmarked) (found unmarked);
  p "a file that must carry the marker and does not is refused"
    (refused ~must_carry:true
       ~format:
         "%s does not carry the marker comment `;%s` that opens its sorted part, so nothing holds \
          its test stanzas in order (gh-ocannl-2193)"
       ~message:"does not carry the marker comment" unmarked);
  p "a file carrying the marker twice is refused"
    (refused
       ~format:
         "%s carries the marker comment `;%s` %d times: which stanzas are ordered is ambiguous"
       ~message:"carries the marker comment"
       (file [ marker_line; test "alpha"; marker_line; test "beta" ]));
  p "a key is cut at its first `-` or `.`, and at nothing else"
    (List.equal String.equal
       (List.map [ "foo"; "foo-tf32"; "foo.actual"; "foo_bar-x.actual"; "foo2" ] ~f:group)
       [ "foo"; "foo"; "foo"; "foo_bar"; "foo2" ])

let () =
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file...>\n" argv.(0);
    Stdlib.exit 1);
  let base = Scan.base_dir argv.(1) in
  let dune_files =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
    |> List.filter ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune")
    |> List.map ~f:(fun (path, on_disk) -> (path, In_channel.read_all on_disk))
  in
  printf
    "Test stanzas in order of test name (gh-ocannl-2193): below a dune file's marker comment a\n\
     stanza sits where its test's name sorts, and every aggregate alias lists its members in\n\
     order, so two branches that each add a test meet only when the names are neighbours. The\n\
     counts go to stderr, since a tally in a golden moves with every test.\n\n";
  let must_carry path = List.mem required path ~equal:String.equal in
  p_all "every file that must carry the marker was handed to the scan" required ~f:(fun path ->
      List.Assoc.mem dune_files path ~equal:String.equal);
  let ordered = List.sum (module Int) dune_files ~f:(fun (_, content) -> ordered_count content) in
  eprintf "stanza_order: %d dune files read, %d stanzas below a marker (not part of the golden)\n"
    (List.length dune_files) ordered;
  (* The floor: test/operations alone holds several hundred, so a reader that sees a handful is
     reading the wrong files or lost the marker. *)
  p "the ordered part the scan read holds hundreds of stanzas, so the order claim is not vacuous"
    (ordered >= 500);
  let found =
    List.concat_map dune_files ~f:(fun (path, content) ->
        findings ~must_carry:(must_carry path) ~path content)
  in
  List.iter found ~f:(fun (f : Refusals.refusal) -> eprintf "FINDING: %s\n" f.message);
  p_empty "no stanza is out of place and no aggregate alias lists its members out of order"
    ~over:dune_files found;
  printf "\nControls, on files built to be refused:\n";
  controls ();
  Refusals.print source
