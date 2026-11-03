(* gh-ocannl-1749: a program that calls [Generated.init] and runs under several stanzas shares a
   lock across all of them.

   [Test_utils.Generated.init] empties [build_files/<exe>/] at startup, on the premise its header
   states: the directory is derived from the executable's own name and "dune runs one process per
   executable, so nothing else writes there". That premise is about STANZAS, not executables. A
   second stanza running the same executable -- a [-tf32] rerun beside the plain [(test)], one rule
   per piece of a census -- is a second process writing that directory, and dune schedules the two
   concurrently: each empties the directory under the other's compile, and a read then finds no
   artifact or the sibling's. gh-ocannl-1720 found the pair [schedule_conv_gemm] / its [-tf32] rerun
   only because Windows refuses to delete a loaded kernel library; elsewhere the sweep succeeds and
   the race is silent. A shared [(locks …)] serializes the runs, which restores the premise.

   WHAT IT READS. Every dune file in the repository, and every OCaml source. The STANZA side is
   dune's own semantics as [Test_utils.Dune_stanza_scan] models them for the other per-program
   checks, reused unchanged through [Scan.stanza_programs], which [env_var_deps] reads programs with
   too: a program is a name of a [(test)], [(tests)], [(executable)] or [(executables)] stanza, in
   the directory the stanza applies to ([(subdir …)] resolved); its modules are [Scan.modules_of]
   and [Scan.program_modules] (dune's default module set, ordered-set subtraction, and the
   main-module rule that keeps [(executables (names a b))] apart), plus each module's [(select …)]
   arms; its runners are [Scan.program_runners] over EVERY stanza of EVERY dune file, so a rule in
   another directory naming [../x/probe.exe], a [%{bin:…}] public name and a [chdir] are all
   resolved as the env-var and config checks resolve them -- plus, for a [(test)]/[(tests)], the
   stanza itself, which dune runs. The SOURCE side is
   [Config_key_scan.generated_init_calls_in_source], also reused unchanged: the parsed call, written
   out, through a module alias of any name or a chain of them, or bare under an [open]/[include] of
   the module, never a mention in a comment or a string. It reads a program's OWN modules only; that
   is complete because [env_var_deps] refuses the initializer in any library module (gh-ocannl-723),
   so no linked library can be the caller.

   THE RULE. A program whose own modules call the initializer, and which two or more stanzas run, is
   refused unless one lock is named by every one of those stanzas. A lock is read from the stanza's
   [(locks …)] field and resolved as dune resolves it: a name starting with [/] is global, any other
   is a path relative to the directory the stanza applies to -- so the same relative name in two
   directories is two locks. A lock built from a pform, or written as anything but an atom, is
   refused as unread rather than guessed. The exemption is an in-place marker on the PROGRAM's
   stanza, [; generated-lock: none -- <reason>], under the shared contained-marker contract
   (malformed, misplaced and out-of-comment spellings refused): for a program whose runs provably
   write different directories, which no lock should serialize. A marker that exempts nothing -- the
   program does not call the initializer, has one runner, or its runners already share a lock -- is
   refused as stale.

   WHAT IT DOES NOT READ, each erring toward a demand rather than a pass unless said otherwise: -
   the working directory of a run. [build_files/] is relative to it, so two runners under different
   [chdir]s do not share a directory; they are still required to share a lock (or to carry the
   marker). Over-demanding is the safe direction. - the configuration a run is handed. A run given
   its own [build_files_prefix] writes elsewhere -- and [Generated.init] aborts under one -- but
   that is a value-level fact this reader does not model; such a pair takes the lock or the marker.
   - whether a source's call is reached on every run (a branch on argv, a child process that
   [chdir]s first): a call anywhere in the program's own modules counts. - two DIFFERENT executables
   with one basename run from one directory, which would share [build_files/<exe>/] too. None
   exists; it is outside this rule. - command sites [Dune_stanza_scan] cannot place (a shell line, a
   PATH-rewritten bare name): those are [config_dep_completeness]'s to refuse, and a runner hidden
   in one is not counted here. - two runs of one program inside ONE stanza's [(concurrent …)]
   action, which no lock can order. No dune file uses [concurrent]. *)

open Base
open Stdio

(* Every file this scan reads arrives as a `@<path>` response file, because the list is longer than
   a Windows command line may be; see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

module Scan = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan

let marker_sentinel = "generated-lock:"
let marker_declaration = "none"
let program_heads = Scan.program_heads

(* One stanza of one dune file, in the directory dune applies it to. *)
type stanza = {
  dune_file : string;
  dir : string;  (** repository-relative; [""] at the root *)
  line : int;
  index : int;
      (** its position among every stanza read: the identity two stanzas on one line keep *)
  head : string;
  sexp : Sexp.t;
  markers : int list;  (** the lines of the valid generated-lock markers it encloses *)
}

(* How a program that needs one is satisfied: the census the golden lists. *)
type resolution = Shares_lock of string list | Exempt_by_marker

let describe_stanza s = Printf.sprintf "%s:%d (%s)" s.dune_file s.line s.head

(* The locks one stanza names, resolved to the path dune keys them by. *)
let locks_of ~fail s =
  match Scan.field s.sexp "locks" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.Atom lock when String.is_substring lock ~substring:"%{" ->
            fail
              (Printf.sprintf
                 "%s names the lock `%s`, built from a pform this scan does not expand -- spell \
                  the lock literally, so whether it is shared can be read"
                 (describe_stanza s) lock);
            None
        | Sexp.Atom lock when String.is_prefix lock ~prefix:"/" -> Some lock
        | Sexp.Atom lock -> Some (Scan.normalize_path (Scan.in_subdir s.dir lock))
        | Sexp.List _ as form ->
            fail
              (Printf.sprintf
                 "%s writes `%s` in its `(locks …)` field, which is not a lock name this scan can \
                  read -- list the locks as plain atoms"
                 (describe_stanza s) (Sexp.to_string form));
            None)
      |> List.dedup_and_sort ~compare:String.compare

(* The calls of the initializer in one source, or a refusal. *)
let init_calls_or_refusal ?(reader = Sources.generated_init_calls_in_source) ~fail ~source content =
  if not (Sources.could_call_generated_init content) then []
  else
    match Test_utils.Scan_attempt.run (fun () -> reader content) with
    | Scanned calls -> calls
    | Unparsed where ->
        fail
          (Printf.sprintf
             "%s names `Generated` and does not parse as OCaml (%s) -- whether its program calls \
              the artifact initializer cannot be established"
             source where);
        []
    | Raised exn ->
        fail
          (Printf.sprintf
             "the Generated.init reader raised %s on %s, which parses -- a defect in the scanner \
              rather than in the source, so its program went unchecked"
             exn source);
        []

(* The stanzas of one dune file, with the markers the shared contract attributes to each; every
   contract refusal is reported here. *)
let stanzas_of_dune_file ~fail (dune_file, content) =
  let dir = match Stdlib.Filename.dirname dune_file with "." -> "" | dir -> dir in
  let contract =
    Scan.contained_marker_contract content ~sentinel:marker_sentinel
      ~parse_declaration:(fun _ ~declaration ~reason:_ ->
        if String.equal declaration marker_declaration then Ok ()
        else
          Error
            [
              Printf.sprintf
                "`%s` is not `%s` -- a program whose runs share an artifact directory takes a \
                 shared `(locks …)` instead"
                declaration marker_declaration;
            ])
      ~belongs:(fun stanza () ->
        if List.mem program_heads stanza.Scan.marked_head ~equal:String.equal then Ok ()
        else
          Error
            [
              Printf.sprintf
                "a `%s` stanza declares no program -- the marker goes on the `(test)` or \
                 `(executable)` that builds the program it exempts"
                stanza.Scan.marked_head;
            ])
  in
  let where line = Printf.sprintf "%s:%d" dune_file line in
  List.iter contract.Scan.contract_issues ~f:(function
    | Scan.Malformed_marker { issue_line; issue_text; issue_malformed } ->
        fail
          (Printf.sprintf
             "%s has a `%s` comment that does not parse as a marker: %s. The line reads `;%s`"
             (where issue_line) marker_sentinel
             (Scan.marker_malformed_reason ~sentinel:marker_sentinel
                ~separator_subject:"declaration"
                ~grammar:(Printf.sprintf "%s %s -- <reason>" marker_sentinel marker_declaration)
                issue_malformed)
             issue_text)
    | Scan.Marker_in_wrong_stanza { issue_line; issue_why; _ } ->
        fail
          (Printf.sprintf "%s carries a generated-lock marker on the wrong stanza: %s"
             (where issue_line) issue_why)
    | Scan.Marker_outside_stanza { issue_line; _ } ->
        fail
          (Printf.sprintf
             "%s has a generated-lock marker that sits inside no stanza -- move it inside the \
              parentheses of the program it exempts"
             (where issue_line))
    | Scan.Marker_outside_comment
        { issue_text_occurrences = in_text; issue_comment_occurrences = in_comments } ->
        fail
          (Printf.sprintf
             "%s spells `%s` %d times and only %d of them are in a comment -- a generated-lock \
              marker outside a comment declares nothing"
             dune_file marker_sentinel in_text in_comments));
  List.map contract.Scan.contract_stanzas ~f:(fun { Scan.marker_stanza = m; stanza_markers } ->
      {
        dune_file;
        dir = Scan.in_subdir dir m.Scan.marked_subdir;
        line = m.Scan.marked_line;
        index = 0;
        head = m.Scan.marked_head;
        sexp = m.Scan.marked_sexp;
        markers = List.map stanza_markers ~f:(fun marker -> marker.Scan.contained_line);
      })

(* The census over a whole tree: each program that calls the initializer and has several runners,
   with how it is satisfied. [sources] are repository-relative paths with their contents. *)
let analyze ~fail ~dune_files ~sources =
  let source_table =
    List.map sources ~f:(fun (path, content) -> (String.lowercase path, content))
    |> Map.of_alist_reduce (module String) ~f:(fun first _ -> first)
  in
  let all =
    List.concat_map dune_files ~f:(stanzas_of_dune_file ~fail)
    |> List.mapi ~f:(fun index s -> { s with index })
  in
  let runner_stanzas = List.map all ~f:(fun s -> (s.dir, s.sexp)) in
  let stanza_of_sexp sexp = List.find all ~f:(fun s -> phys_equal s.sexp sexp) in
  let census = ref [] in
  let callers = ref 0 in
  List.iter all ~f:(fun s ->
      if List.mem program_heads s.head ~equal:String.equal then (
        (match s.markers with
        | _ :: second :: _ ->
            fail
              (Printf.sprintf
                 "%s:%d carries a second generated-lock marker for one stanza -- one says it"
                 s.dune_file second)
        | _ -> ());
        let group =
          List.filter_map all ~f:(fun o ->
              if String.equal o.dune_file s.dune_file && String.equal o.dir s.dir then Some o.sexp
              else None)
        in
        let directory_modules = Scan.directory_modules ~dir:s.dir (List.map sources ~f:fst) in
        (* [Some] under the [program_heads] guard above, which is the list it answers for. *)
        let programs =
          Option.value_exn
            (Scan.stanza_programs ~subdir:s.dir ~runner_stanzas ~directory_modules group s.sexp)
        in
        let arms = Scan.select_arms s.sexp in
        let runners_of =
          List.map programs.Scan.programs ~f:(fun p ->
              (p.Scan.program, p.Scan.own_modules, List.map p.Scan.rule_runners ~f:fst))
        in
        (* One marker per stanza, so on a stanza declaring several programs it would exempt every
           one of them, whichever it was written for. *)
        (match (s.markers, runners_of) with
        | _ :: _, _ :: _ :: _ ->
            fail
              (Printf.sprintf
                 "%s carries a generated-lock marker and declares %d programs -- the marker would \
                  exempt every one of them; give the program it is about a stanza of its own"
                 (describe_stanza s) (List.length runners_of))
        | _ -> ());
        let marker_used = ref false in
        List.iter runners_of ~f:(fun (name, own_modules, runner_sexps) ->
            let program = Scan.in_subdir s.dir (name ^ ".exe") in
            let calls =
              List.concat_map own_modules ~f:(fun m ->
                  let target = m ^ ".ml" in
                  let files =
                    target
                    :: List.filter_map arms ~f:(fun (selected, arm) ->
                        Option.some_if (String.equal selected target) arm)
                  in
                  let found =
                    List.filter_map files ~f:(fun file ->
                        let path = Scan.in_subdir s.dir file in
                        Option.map
                          (Map.find source_table (String.lowercase path))
                          ~f:(fun content -> (path, content)))
                  in
                  if List.is_empty found && Scan.needs_source programs m then
                    fail
                      (Printf.sprintf
                         "%s names the module `%s` of %s, and this scan was handed no source for \
                          it -- whether the program calls the artifact initializer cannot be \
                          established; add its directory to the rule's source globs"
                         (describe_stanza s) m program);
                  List.concat_map found ~f:(fun (path, content) ->
                      init_calls_or_refusal ~fail ~source:path content))
            in
            if not (List.is_empty calls) then (
              Int.incr callers;
              let runners =
                (List.filter_map runner_sexps ~f:stanza_of_sexp
                @ if programs.Scan.runs_itself then [ s ] else [])
                |> List.dedup_and_sort ~compare:(fun a b -> Int.compare a.index b.index)
              in
              if List.length runners >= 2 then
                let shared =
                  match List.map runners ~f:(locks_of ~fail) with
                  | [] -> []
                  | first :: rest ->
                      List.fold rest ~init:first ~f:(fun acc locks ->
                          List.filter acc ~f:(List.mem locks ~equal:String.equal))
                in
                match (shared, s.markers) with
                | _ :: _, _ :: _ ->
                    marker_used := true;
                    fail
                      (Printf.sprintf
                         "%s carries a generated-lock marker, and every stanza running %s already \
                          shares the lock %s -- drop the marker"
                         (describe_stanza s) program (String.concat ~sep:", " shared))
                | _ :: _, [] -> census := (program, Shares_lock shared) :: !census
                | [], _ :: _ ->
                    marker_used := true;
                    census := (program, Exempt_by_marker) :: !census
                | [], [] ->
                    fail
                      (Printf.sprintf
                         "%s calls Generated.init (%s) and runs under %d stanzas that share no \
                          lock: %s -- each run empties and rewrites its build_files/ directory, so \
                          two of them at once sweep each other's artifacts (gh-ocannl-1720). Give \
                          every one of them the same `(locks …)`, or, where each run's artifacts \
                          provably go elsewhere, mark %s with `; %s %s -- <why>`"
                         program
                         (String.concat ~sep:", "
                            (List.dedup_and_sort calls ~compare:String.compare))
                         (List.length runners)
                         (String.concat ~sep:"; " (List.map runners ~f:describe_stanza))
                         (describe_stanza s) marker_sentinel marker_declaration)));
        if (not (List.is_empty s.markers)) && not !marker_used then
          fail
            (Printf.sprintf
               "%s carries a generated-lock marker, and no program of it both calls Generated.init \
                and runs under several stanzas -- the marker exempts nothing; drop it"
               (describe_stanza s))));
  (!callers, List.sort !census ~compare:(fun (a, _) (b, _) -> String.compare a b))

(* An empty corpus on either side is a broken rule, not a clean tree. *)
let require_inputs ~fail ~dune_files ~sources =
  if List.is_empty dune_files then
    fail "no dune files among the arguments -- the rule's dune-file glob matches nothing";
  if List.is_empty sources then
    fail "no OCaml sources among the arguments -- the rule's source globs match nothing"

let print_census census =
  List.iter census ~f:(fun (program, resolution) ->
      match resolution with
      | Shares_lock locks ->
          printf "  %s: every runner holds %s\n" program (String.concat ~sep:", " locks)
      | Exempt_by_marker -> printf "  %s: exempt by its `%s` marker\n" program marker_sentinel)

(* Synthetic trees: the repository satisfies the rule, so its own corpus cannot tell a rule from a
   tautology. Each case builds the smallest tree holding one violation, or the nearest legitimate
   one, and claims on what [analyze] says of it. *)

let probe_calls =
  {ocaml|module Generated = Test_utils.Generated
let () = Generated.init ~backend_name:"cc"|ocaml}

let probe_mentions =
  {ocaml|(** [Test_utils.Generated.init] is not called here. *)
let message = "Generated.init ~backend_name"|ocaml}

(* A rule running [exe] into [target], under [locks] when given. *)
let rule ?(locks = "") ?(target = "probe") exe =
  Printf.sprintf
    "(rule\n (target %s.actual)%s\n (action (with-stdout-to %%{target} (run %%{dep:%s}))))\n" target
    (if String.is_empty locks then "" else "\n (locks " ^ locks ^ ")")
    exe

let control () =
  let source = "test/operations/generated_lock_scan.ml" in
  let run_tree ~dune_files ~sources =
    let messages = ref [] in
    let fail message = messages := message :: !messages in
    let callers, census = analyze ~fail ~dune_files ~sources in
    (List.rev !messages, callers, census)
  in
  let probe ?(content = probe_calls) dir = [ (Scan.in_subdir dir "probe.ml", content) ] in
  (* A case that must be refused by exactly one message carrying [fragment]; the observation is
     recorded only for the refusal the format names. *)
  let refused label ~format ~fragment ~dune_files ~sources =
    let messages, _, _ = run_tree ~dune_files ~sources in
    let routed =
      match messages with [ m ] -> String.is_substring m ~substring:fragment | _ -> false
    in
    if not routed then
      List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
    Verdict.p label routed
  in
  (* The whole answer against the expected one: no refusal, the callers the reader found, and the
     census -- so a reader gone blind (no caller found) cannot pass an accepted case. *)
  let accepted label ~callers ~expect ~dune_files ~sources =
    let answer = run_tree ~dune_files ~sources in
    let messages, _, _ = answer in
    List.iter messages ~f:(fun m -> eprintf "  %s said: %s\n" label (Verdict.one_line m));
    Verdict.p label (Poly.equal answer ([], callers, expect))
  in
  let unlocked_format =
    "%s calls Generated.init (%s) and runs under %d stanzas that share no lock: %s -- each run \
     empties and rewrites its build_files/ directory, so two of them at once sweep each other's \
     artifacts (gh-ocannl-1720). Give every one of them the same `(locks …)`, or, where each run's \
     artifacts provably go elsewhere, mark %s with `; %s %s -- <why>`"
  in
  let exe = "(executable (name probe))\n" in
  refused "two rules running a Generated.init program with no lock are refused"
    ~format:unlocked_format ~fragment:"t/probe.exe calls Generated.init"
    ~dune_files:[ ("t/dune", exe ^ rule "probe.exe" ^ rule ~target:"probe-rerun" "probe.exe") ]
    ~sources:(probe "t");
  refused "a test and its rerun with the lock on only one of them are refused"
    ~format:unlocked_format ~fragment:"runs under 2 stanzas that share no lock"
    ~dune_files:
      [ ("t/dune", "(test (name probe) (locks probe))\n" ^ rule ~target:"probe-rerun" "probe.exe") ]
    ~sources:(probe "t");
  refused "one relative lock name in two directories is two locks, and refused"
    ~format:unlocked_format ~fragment:"t/probe.exe calls Generated.init"
    ~dune_files:
      [
        ("t/dune", "(test (name probe) (locks probe))\n");
        ("u/dune", rule ~locks:"probe" "../t/probe.exe");
      ]
    ~sources:(probe "t");
  accepted "a test and its rerun sharing one lock are accepted" ~callers:1
    ~expect:[ ("t/probe.exe", Shares_lock [ "t/probe" ]) ]
    ~dune_files:
      [
        ( "t/dune",
          "(test (name probe) (locks probe))\n"
          ^ rule ~locks:"probe" ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  accepted "a global lock named from two directories is one lock" ~callers:1
    ~expect:[ ("t/probe.exe", Shares_lock [ "/probe" ]) ]
    ~dune_files:
      [
        ("t/dune", "(test (name probe) (locks /probe))\n");
        ("u/dune", rule ~locks:"/probe" "../t/probe.exe");
      ]
    ~sources:(probe "t");
  accepted "a program that only mentions Generated.init needs no lock" ~callers:0 ~expect:[]
    ~dune_files:[ ("t/dune", exe ^ rule "probe.exe" ^ rule ~target:"probe-rerun" "probe.exe") ]
    ~sources:(probe ~content:probe_mentions "t");
  accepted "a Generated.init program with one runner needs no lock" ~callers:1 ~expect:[]
    ~dune_files:[ ("t/dune", "(test (name probe))\n") ]
    ~sources:(probe "t");
  accepted "a marked program is exempt" ~callers:1
    ~expect:[ ("t/probe.exe", Exempt_by_marker) ]
    ~dune_files:
      [
        ( "t/dune",
          "(executable\n\
          \ ; generated-lock: none -- each run chdirs into a directory of its own\n\
          \ (name probe))\n" ^ rule "probe.exe"
          ^ rule ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  refused "a marker on a program with one runner is refused as stale"
    ~format:
      "%s carries a generated-lock marker, and no program of it both calls Generated.init and runs \
       under several stanzas -- the marker exempts nothing; drop it"
    ~fragment:"the marker exempts nothing"
    ~dune_files:
      [ ("t/dune", "(test\n ; generated-lock: none -- each run is the only one\n (name probe))\n") ]
    ~sources:(probe "t");
  refused "a marker beside a shared lock is refused as redundant"
    ~format:
      "%s carries a generated-lock marker, and every stanza running %s already shares the lock %s \
       -- drop the marker"
    ~fragment:"already shares the lock t/probe"
    ~dune_files:
      [
        ( "t/dune",
          "(test\n\
          \ ; generated-lock: none -- each run chdirs into a directory of its own\n\
          \ (name probe)\n\
          \ (locks probe))\n"
          ^ rule ~locks:"probe" ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  refused "two markers on one stanza are refused"
    ~format:"%s:%d carries a second generated-lock marker for one stanza -- one says it"
    ~fragment:"second generated-lock marker"
    ~dune_files:
      [
        ( "t/dune",
          "(executable\n\
          \ ; generated-lock: none -- each run chdirs into a directory of its own\n\
          \ ; generated-lock: none -- and says so twice over here\n\
          \ (name probe))\n" ^ rule "probe.exe"
          ^ rule ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  refused "a marker on a stanza declaring two programs is refused"
    ~format:
      "%s carries a generated-lock marker and declares %d programs -- the marker would exempt \
       every one of them; give the program it is about a stanza of its own"
    ~fragment:"declares 2 programs"
    ~dune_files:
      [
        ( "t/dune",
          "(executables\n\
          \ ; generated-lock: none -- each run chdirs into a directory of its own\n\
          \ (names probe other))\n" ^ rule "probe.exe"
          ^ rule ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t" @ [ ("t/other.ml", "let () = ()") ]);
  (* Two runners written on one line are two stanzas, not one. *)
  refused "two runners on one line are two runners" ~format:unlocked_format
    ~fragment:"runs under 2 stanzas that share no lock"
    ~dune_files:
      [
        ( "t/dune",
          "(executable (name probe))\n\
           (rule (alias a) (action (run %{dep:probe.exe}))) (rule (alias b) (action (run \
           %{dep:probe.exe})))\n" );
      ]
    ~sources:(probe "t");
  refused "a malformed marker is refused"
    ~format:"%s has a `%s` comment that does not parse as a marker: %s. The line reads `;%s`"
    ~fragment:"does not parse as a marker"
    ~dune_files:[ ("t/dune", "(test\n ; generated-lock: none\n (name probe))\n") ]
    ~sources:(probe "t");
  refused "a marker on a rule is refused as misplaced"
    ~format:"%s carries a generated-lock marker on the wrong stanza: %s"
    ~fragment:"on the wrong stanza"
    ~dune_files:
      [
        ( "t/dune",
          "(test (name probe))\n\
           (rule\n\
          \ ; generated-lock: none -- placed on the runner instead\n\
          \ (alias other)\n\
          \ (action (echo hi)))\n" );
      ]
    ~sources:(probe "t");
  refused "a marker between stanzas is refused"
    ~format:
      "%s has a generated-lock marker that sits inside no stanza -- move it inside the parentheses \
       of the program it exempts"
    ~fragment:"inside no stanza"
    ~dune_files:
      [ ("t/dune", "; generated-lock: none -- above the stanza, not in it\n(test (name probe))\n") ]
    ~sources:(probe "t");
  refused "a marker outside a comment is refused"
    ~format:
      "%s spells `%s` %d times and only %d of them are in a comment -- a generated-lock marker \
       outside a comment declares nothing"
    ~fragment:"outside a comment declares nothing"
    ~dune_files:
      [ ("t/dune", "(rule\n (alias other)\n (action (echo \"generated-lock: none -- x y\")))\n") ]
    ~sources:[];
  (* The program is exempted by its marker, so the unread lock is the one refusal the tree holds. *)
  let marked_exe =
    "(executable\n\
    \ ; generated-lock: none -- each run chdirs into a directory of its own\n\
    \ (name probe))\n"
  in
  refused "a lock built from a pform is refused as unread"
    ~format:
      "%s names the lock `%s`, built from a pform this scan does not expand -- spell the lock \
       literally, so whether it is shared can be read"
    ~fragment:"built from a pform"
    ~dune_files:
      [
        ( "t/dune",
          marked_exe ^ rule "probe.exe"
          ^ rule ~locks:"%{env:L=probe}" ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  refused "a lock that is not an atom is refused as unread"
    ~format:
      "%s writes `%s` in its `(locks …)` field, which is not a lock name this scan can read -- \
       list the locks as plain atoms"
    ~fragment:"not a lock name this scan can read"
    ~dune_files:
      [
        ( "t/dune",
          marked_exe ^ rule "probe.exe" ^ rule ~locks:"(probe)" ~target:"probe-rerun" "probe.exe" );
      ]
    ~sources:(probe "t");
  refused "a program module with no source is refused"
    ~format:
      "%s names the module `%s` of %s, and this scan was handed no source for it -- whether the \
       program calls the artifact initializer cannot be established; add its directory to the \
       rule's source globs"
    ~fragment:"handed no source for it"
    ~dune_files:[ ("t/dune", "(test (name probe) (modules probe helper))\n") ]
    ~sources:(probe "t");
  refused "a program source that does not parse is refused"
    ~format:
      "%s names `Generated` and does not parse as OCaml (%s) -- whether its program calls the \
       artifact initializer cannot be established"
    ~fragment:"t/probe.ml names `Generated` and does not parse"
    ~dune_files:[ ("t/dune", "(test (name probe))\n") ]
    ~sources:(probe ~content:"let () = Generated.init ~backend_name:" "t");
  let input_case label ~format ~dune_files ~sources =
    let messages = ref [] in
    require_inputs ~fail:(fun m -> messages := m :: !messages) ~dune_files ~sources;
    let routed = match !messages with [ m ] -> String.equal m format | _ -> false in
    if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
    Verdict.p label routed
  in
  input_case "an empty dune-file corpus is refused"
    ~format:"no dune files among the arguments -- the rule's dune-file glob matches nothing"
    ~dune_files:[] ~sources:(probe "t");
  input_case "an empty OCaml source corpus is refused"
    ~format:"no OCaml sources among the arguments -- the rule's source globs match nothing"
    ~dune_files:[ ("t/dune", "") ]
    ~sources:[];
  (* The scanner-defect refusal, routed apart from the parse one (gh-ocannl-1292). *)
  let messages = ref [] in
  ignore
    (init_calls_or_refusal
       ~reader:(fun _ -> invalid_arg "planted")
       ~fail:(fun m -> messages := m :: !messages)
       ~source:"good.ml" "let _ = Generated.init"
      : string list);
  let routed =
    match !messages with
    | [ m ] -> String.is_substring m ~substring:"raised Invalid_argument(\"planted\") on good.ml"
    | _ -> false
  in
  if routed then
    Test_utils.Refusal_control_manifest.observe_failure ~source
      ~format:
        "the Generated.init reader raised %s on %s, which parses -- a defect in the scanner rather \
         than in the source, so its program went unchecked";
  Verdict.p
    "a reader exception on a source that parses reaches the scanner-defect refusal, not the parse \
     one"
    routed;
  Test_utils.Refusal_control_manifest.print "generated_lock_scan.ml"

let () =
  if Array.length argv = 2 && String.equal argv.(1) "--control" then (
    control ();
    Stdlib.exit 0);
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune files and OCaml sources...>|--control\n" argv.(0);
    Stdlib.exit 1);
  let base = Scan.base_dir argv.(1) in
  let paths =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  (* The globs reach into the build tree, which holds dune's `.pp.ml` expansions beside the sources
     and test/ppx's captured `_actual.ml` outputs: neither is a program's own module source. *)
  let derived path =
    String.is_suffix path ~suffix:".pp.ml"
    || (String.is_prefix path ~prefix:"test/ppx/" && String.is_suffix path ~suffix:"_actual.ml")
  in
  let read = List.map ~f:(fun (path, on_disk) -> (path, In_channel.read_all on_disk)) in
  let dune_files =
    read
      (List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune"))
  in
  let sources =
    read
      (List.filter paths ~f:(fun (path, _) ->
           String.is_suffix path ~suffix:".ml" && not (derived path)))
  in
  require_inputs ~fail:Verdict.fail ~dune_files ~sources;
  let callers, census = analyze ~fail:Verdict.fail ~dune_files ~sources in
  (* The tally moves with every new test that reads generated code, so it stays off the golden; the
     floor is what says the reader is not blind. *)
  eprintf "Programs calling Generated.init: %d (not part of the golden).\n" callers;
  Verdict.p "the census finds programs that call Generated.init" (callers > 0);
  printf "\nPrograms calling Generated.init that several stanzas run:\n";
  print_census census;
  Verdict.success_trailer ~blank_line:true
    [ "every such program's runners share a lock, or its stanza says why none is needed." ];
  Test_utils.Refusal_control_manifest.print "generated_lock_scan.ml"
