(** How {!Test_utils.Dune_stanza_scan} reads a dune file, exercised directly.

    The scan decides which stanzas are subject to the dependency rule, so every one of its mistakes
    is silent in the same way [Config_key_scan]'s are: a stanza it fails to recognise looks exactly
    like a stanza that is not there. The cases below are therefore written against the shapes the
    repository's dune files actually take — a dep buried under a named group, an executable reached
    through a variable, a stanza quoted in a comment — rather than against the one shape a checker
    is easiest to write for. *)

open Base
module Scan = Test_utils.Dune_stanza_scan

let printf = Stdio.printf

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

let render (site : Scan.site) =
  let where = Scan.in_subdir site.Scan.subdir site.Scan.cwd in
  Printf.sprintf "%s%s %s%s"
    (if String.is_empty where then "" else "in " ^ where ^ ": ")
    (Scan.kind_name site.Scan.kind) site.Scan.name
    (if site.Scan.declares_config then " [declares]" else "")
  ^ if site.Scan.isolating_sandbox then " [sandboxed]" else ""

(* Each case pairs a dune-file body with the sites the scan should report, in order. *)
let cases =
  [
    ( "a test declaring the dep",
      {dune|(test (name t) (deps ocannl_config) (libraries base))|dune},
      [ "test t [declares]" ] );
    ("a test omitting the dep", {dune|(test (name t) (libraries base))|dune}, [ "test t" ]);
    ( "a test with deps but not this one",
      {dune|(test (name t) (deps (env_var OCANNL_BACKEND) t.expected))|dune},
      [ "test t" ] );
    (* The dep may sit anywhere in the field, including inside a group, and a directory reaching for
       a config elsewhere still declares one. *)
    ( "the dep is found wherever in deps it sits",
      {dune|(test (name t) (deps (glob_files *.data) (env_var OCANNL_BACKEND) ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "an explicit file dependency declares it",
      {dune|(test (name t) (deps (file ocannl_config)))|dune},
      [ "test t [declares]" ] );
    (* A config elsewhere is still a dependency on a file -- WHICH file it has to be is settled by
       the check, which knows where configs exist; see declared_paths_cases below. *)
    ( "a config elsewhere is a dependency too",
      {dune|(test (name t) (deps ../config/ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "so does one bound to a name",
      {dune|(test (name t) (deps (:cfg ocannl_config)))|dune},
      [ "test t [declares]" ] );
    (* Dune's dependency language names other things than files, and depending on an ALIAS or an
       environment variable of that name is not depending on the file (Codex P2, round 3). *)
    ( "an alias of the same name is not the file",
      {dune|(test (name t) (deps (alias ocannl_config)))|dune},
      [ "test t" ] );
    ( "nor is an environment variable of that name",
      {dune|(test (name t) (deps (env_var ocannl_config)))|dune},
      [ "test t" ] );
    (* A glob matches the source tree, and in a directory whose config arrives through `(copy_files
       …)` the file is a generated target -- so the glob matches nothing and builds nothing, while
       looking like a declaration (Codex P2, round 7). *)
    ( "a glob does not depend on the copied config",
      {dune|(test (name t) (deps (glob_files ocannl_config)))|dune},
      [ "test t" ] );
    ( "a multi-test stanza is one site naming both",
      {dune|(tests (names a b) (deps ocannl_config))|dune},
      [ "test a, b [declares]" ] );
    (* An inline-test library runs tests too, and its deps live one level in. *)
    ( "inline tests declare the dep inside their own field",
      {dune|(library (name l) (inline_tests (deps ocannl_config (env_var OCANNL_BACKEND))))|dune},
      [ "inline tests l [declares]" ] );
    ( "an inline-test field with no deps at all",
      {dune|(library (name l) (inline_tests))|dune},
      [ "inline tests l" ] );
    ( "a library's own deps are not the inline tests' deps",
      {dune|(library (name l) (deps ocannl_config) (inline_tests (deps (env_var X))))|dune},
      [ "inline tests l" ] );
    ("a plain library is not a site", {dune|(library (name l) (libraries base))|dune}, []);
    (* An executable is not a site: dune runs it only where a rule says so, which is what keeps
       diagnostic and tutorial executables off the list without an exemption. *)
    ("an executable is not a site", {dune|(executable (name bench) (libraries base))|dune}, []);
    ( "a rule running an executable is a site",
      {dune|(rule (target x.actual) (deps ocannl_config)
 (action (with-stdout-to %{target} (run %{dep:bench.exe}))))|dune},
      [ "rule running bench.exe [declares]" ] );
    ( "a rule running an executable through a named dep is one too",
      {dune|(rule (targets out.ml) (deps (:pp pp.exe) (:input in.ml))
 (action (run ./%{pp} --impl %{input} -o %{targets})))|dune},
      [ "rule running pp.exe" ] );
    (* A dependency form wrapping the path binds the executable just as surely; keeping only bare
       atoms lost it, after which the binding looked empty and the command external (Codex P2, round
       6). *)
    ( "a named dep may wrap its path in a dependency form",
      {dune|(rule (deps ocannl_config (:runner (file probe.exe))) (action (run %{runner})))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "a named dep resolving to no executable is not evidence of an external tool",
      {dune|(rule (deps (:script run.sh)) (action (run %{script})))|dune},
      [ "rule whose command this scan cannot read: %{script}" ] );
    (* dune runs a chdir'd action elsewhere, and OCANNL searches upward from THERE, so that
       directory's config is the one that has to be built (Codex P2, round 6). *)
    ( "a chdir moves the directory whose config matters",
      {dune|(rule (deps ocannl_config) (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    ( "and the dependency that satisfies it is the one reaching that directory",
      {dune|(rule (deps ../sibling/ocannl_config)
 (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    (* A `(test)` may carry a custom action, and a chdir in one moves the test's own process (Codex
       P2, round 8). *)
    ( "a test's own action can move where it runs",
      {dune|(test (name t) (deps ocannl_config) (action (chdir ../sibling (run %{test}))))|dune},
      [ "in ../sibling: test t [declares]" ] );
    ( "a test with an ordinary action still runs where it is",
      {dune|(test (name t) (deps ocannl_config) (action (run %{test} --flag)))|dune},
      [ "test t [declares]" ] );
    (* The test's own command is what says where the TEST runs; a helper sent elsewhere in the same
       action is a site of its own, not a relocation of the test (Codex P2, round 10). *)
    ( "a helper chdir'd elsewhere does not move the test",
      {dune|(test (name t) (deps ocannl_config)
 (action (progn (chdir ../scratch (run diff a b)) (run %{test}))))|dune},
      [ "test t [declares]" ] );
    ( "but a helper running an executable elsewhere is its own site",
      {dune|(test (name t) (deps ocannl_config)
 (action (progn (chdir ../scratch (run %{dep:probe.exe})) (run %{test}))))|dune},
      [ "test t [declares]"; "in ../scratch: rule running probe.exe [declares]" ] );
    ( "a test's shell action leaves its directory unestablished too",
      {dune|(test (name t) (deps ocannl_config) (action (bash "cd ../sibling && ./%{test}")))|dune},
      [
        "test t [declares]";
        "rule whose working directory this scan cannot establish: shell: cd ../sibling && \
         ./%{test} [declares]";
      ] );
    (* OCANNL searches UPWARD, so an action that chdirs into a child finds the stanza's own config;
       demanding the child have one rejects a correctly configured rule (Codex P2, round 9). *)
    ( "a chdir into a child is answered by the stanza's own config",
      {dune|(rule (deps ocannl_config) (action (chdir child (run %{dep:probe.exe}))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "the child's own config answers it as well",
      {dune|(rule (deps child/ocannl_config) (action (chdir child (run %{dep:probe.exe}))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "a sibling chdir is reported at the sibling",
      {dune|(rule (deps ocannl_config) (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    (* An explicit path names something this repository produced, whatever its extension; stripping
       the `./` before asking loses what distinguishes it from `python3` (Codex P2, round 8). *)
    ( "an explicit relative program is a site without an extension",
      {dune|(rule (deps ocannl_config) (action (run ./probe)))|dune},
      [ "rule running probe [declares]" ] );
    ("a bare word is still a tool on PATH", {dune|(rule (action (run diff a b)))|dune}, []);
    (* An absolute path names something the system provides -- no more ours than a bare word is
       (Codex P2, round 18). *)
    ( "an absolute path is a system tool",
      {dune|(rule (action (run /usr/bin/python3 --version)))|dune},
      [] );
    (* A PATH tool handed something this repository builds may be launching it (`env probe.exe`) or
       reading it (`diff old.exe new.exe`), and dune's grammar does not say which -- so neither is
       guessed: the pair is reported, and the check settles it the way it settles every other
       unreadable command (Codex P2, rounds 12 and 13). *)
    ( "a launcher's target is reported without guessing that it is one",
      {dune|(rule (deps ocannl_config) (action (run env %{dep:probe.exe} --flag)))|dune},
      [
        "rule whose working directory this scan cannot establish: env, handed %{dep:probe.exe} \
         [declares]";
      ] );
    ( "and so is a tool that is only reading them",
      {dune|(rule (action (run diff old.exe new.exe)))|dune},
      [ "rule whose working directory this scan cannot establish: diff, handed old.exe, new.exe" ]
    );
    (* Fail closed on an argument the scan cannot resolve either: `./%{runner}` names whatever the
       binding does, and dropping it would hide a launched executable (Codex P2, round 14). *)
    ( "an unresolvable argument counts as much as a resolved one",
      {dune|(rule (deps (:runner probe.exe)) (action (run env ./%{runner})))|dune},
      [ "rule whose working directory this scan cannot establish: env, handed ./%{runner}" ] );
    ( "ordinary arguments are not executables",
      {dune|(rule (action (run %{dep:probe.exe} --input data.csv --out %{targets})))|dune},
      [ "rule running probe.exe" ] );
    (* Where a chdir destination is built out of a pform, the directory cannot be resolved, and
       taking it literally would search a directory that does not exist (Codex P2, round 12). *)
    ( "a chdir to a pform leaves the directory unestablished",
      {dune|(rule (deps ocannl_config) (action (chdir %{workspace_root} (run %{dep:probe.exe}))))|dune},
      [
        "rule whose working directory this scan cannot establish: %{dep:probe.exe}, under `(chdir \
         %{workspace_root} ...)` [declares]";
      ] );
    (* But only for something that could read a configuration there: a PATH tool reads none wherever
       it runs, so an unresolvable destination over one is not a finding (round 13). *)
    ( "a chdir to a pform over a PATH tool is not",
      {dune|(rule (action (chdir %{workspace_root} (run diff a b))))|dune},
      [] );
    (* A rewritten PATH is the other way a command name stops saying what it names: there even a
       bare word may be a local executable, so the External verdict is the one that cannot be
       trusted under it (Codex P2, round 16). *)
    (* An action head the scan cannot place keeps the directory it sits in, since it may run an
       OCANNL program there (Codex P2, round 18). *)
    ( "an unclassified action under a chdir keeps that directory",
      {dune|(rule (deps child/ocannl_config) (action (chdir child (invent-an-action probe.exe))))|dune},
      [ "in child: rule with an action this scan cannot place: invent-an-action [declares]" ] );
    ( "and an unclassified action under a pform chdir is unplaceable, like a command",
      {dune|(rule (deps ocannl_config)
 (action (chdir %{workspace_root} (invent-an-action probe.exe))))|dune},
      [
        "rule whose working directory this scan cannot establish: invent-an-action, under `(chdir \
         %{workspace_root} ...)` [declares]";
      ] );
    ( "a rewritten PATH makes a bare command unplaceable",
      {dune|(rule (deps ocannl_config) (action (setenv PATH . (run env probe))))|dune},
      [
        "rule whose working directory this scan cannot establish: env, under `(setenv PATH . ...)` \
         [declares]";
      ] );
    ( "a Windows-case PATH spelling makes a bare executable unplaceable",
      {dune|(rule (deps ocannl_config) (action (setenv Path . (run probe.exe))))|dune},
      [
        "rule whose working directory this scan cannot establish: probe.exe, under `(setenv PATH . \
         ...)` [declares]";
      ] );
    ( "but a path-qualified command still names what it names",
      {dune|(rule (deps ocannl_config) (action (setenv PATH . (run %{dep:probe.exe}))))|dune},
      [ "rule running probe.exe [declares]" ] );
    (* And the directory a nested chdir chose is still where the process runs: PATH says nothing
       about that (Codex P2, round 17). *)
    ( "a chdir under a PATH rewrite keeps its directory",
      {dune|(rule (deps child/ocannl_config)
 (action (setenv PATH . (chdir child (run ./probe.exe)))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "a chdir to the stanza's own directory changes nothing",
      {dune|(rule (deps ocannl_config) (action (chdir . (run %{dep:probe.exe}))))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "a nested action does not hide the executable",
      {dune|(rule (alias slow) (deps ocannl_config)
 (action (no-infer (progn (with-stdout-to x.actual (run %{dep:mnist_conv.exe}))
  (diff x.expected x.actual)))))|dune},
      [ "rule running mnist_conv.exe [declares]" ] );
    ( "a rule running no executable is not a site",
      {dune|(rule (alias runtest) (action (diff a.expected a.actual)))|dune},
      [] );
    ( "a PATH tool with no workspace argument is not a site",
      {dune|(rule (alias runtest) (action (run python3 --version)))|dune},
      [] );
    (* But one handed something from the workspace is: `env probe.exe` launches it, `diff old.exe
       new.exe` reads it, and `env -C ../s probe.exe` launches it elsewhere -- three readings dune's
       grammar does not tell apart (Codex P2, rounds 12 to 14). The strongest is assumed, so the
       check asks for an exemption with a reason rather than guessing. *)
    ( "a PATH tool handed something from the workspace is unplaceable",
      {dune|(rule (alias runtest) (deps helper.py) (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed %{dep:test_it.py}";
      ] );
    (* gh-ocannl-1359: such a site must run in a sandbox that isolates it, and only both settings
       together do -- `always` alone may pick the symlink sandbox, which Python resolves back into
       _build, and `preserve_file_kind` alone sandboxes nothing unless the invocation asks. *)
    ( "a copying sandbox is read off the deps, in either order",
      {dune|(rule (alias runtest) (deps (sandbox preserve_file_kind always))
 (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed \
         %{dep:test_it.py} [sandboxed]";
      ] );
    ( "the symlink-permitting sandbox does not isolate",
      {dune|(rule (alias runtest) (deps (sandbox always)) (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed %{dep:test_it.py}";
      ] );
    ( "a file-kind constraint without `always` sandboxes nothing",
      {dune|(rule (alias runtest) (deps (sandbox preserve_file_kind))
 (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed %{dep:test_it.py}";
      ] );
    ( "only a top-level sandbox form is read",
      {dune|(rule (alias runtest) (deps (:s (sandbox always preserve_file_kind)))
 (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed %{dep:test_it.py}";
      ] );
    (* An `(alias ...)` stanza took an action of its own before dune 2.0 and can still depend on an
       executable, so it is read like a rule rather than passed over (Codex P2, round 1). *)
    ( "an alias stanza is read like a rule",
      {dune|(alias (name runtest) (deps ocannl_config) (action (run %{dep:probe.exe})))|dune},
      [ "rule running probe.exe [declares]" ] );
    (* What the command names, in the spellings that do not say `.exe` (Codex P2, round 2). A
       `%{bin:…}` resolves a public executable of this workspace before it looks at PATH, so it
       counts; an external tool that reads no configuration is what the exemption list is for. *)
    ( "a public executable through %{bin:...}",
      {dune|(rule (deps ocannl_config) (action (run %{bin:probe} --read=backend)))|dune},
      [ "rule running probe [declares]" ] );
    ( "%{exe:...} names one too",
      {dune|(rule (action (run %{exe:probe.exe})))|dune},
      [ "rule running probe.exe" ] );
    (* This scan does not parse shell, and splitting a command line on whitespace only looked like
       reading it: `if ready; then ./probe.exe; fi` yields `./probe.exe;`, which passes for an
       external tool (Codex P2, round 5). A shell line is reported as unreadable, which the check
       settles by requiring the dependency. *)
    ( "a shell action is not parsed, it is reported",
      {dune|(rule (deps ocannl_config) (action (bash "./probe.exe --flag > out")))|dune},
      [
        "rule whose working directory this scan cannot establish: shell: ./probe.exe --flag > out \
         [declares]";
      ] );
    ( "including one where a word split would have lost the executable",
      {dune|(rule (action (system "if ready; then ./probe.exe; fi")))|dune},
      [
        "rule whose working directory this scan cannot establish: shell: if ready; then \
         ./probe.exe; fi";
      ] );
    (* Command position, not "an .exe somewhere in the stanza": a rule that only moves an executable
       around runs nothing. *)
    ( "a rule that copies an executable does not run it",
      {dune|(rule (target probe.copy) (action (copy %{dep:probe.exe} %{target})))|dune},
      [] );
    ( "a command this scan cannot place is reported, not ignored",
      {dune|(rule (deps ocannl_config) (action (run %{dep:helper.sh})))|dune},
      [ "rule whose command this scan cannot read: %{dep:helper.sh} [declares]" ] );
    (* dynamic-run executes a program too (Codex P2, round 3), and an action head on neither list is
       reported rather than passed over -- which is what makes a fourth such action harmless. *)
    ( "dynamic-run executes a program as much as run does",
      {dune|(rule (deps ocannl_config) (action (dynamic-run %{dep:probe.exe} --flag)))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "an action head on neither list is reported",
      {dune|(rule (deps ocannl_config) (action (invent-an-action probe.exe)))|dune},
      [ "rule with an action this scan cannot place: invent-an-action [declares]" ] );
    ( "a program action's arguments are not actions",
      {dune|(rule (action (run %{dep:probe.exe} --diff --copy)))|dune},
      [ "rule running probe.exe" ] );
    (* Identity is the path as written: a basename would let one directory's exemption cover a
       different executable of the same name (Codex P2, round 3). *)
    ( "an executable elsewhere keeps its path",
      {dune|(rule (action (run %{dep:../../tools/pp.exe} --impl x.ml)))|dune},
      [ "rule running ../../tools/pp.exe" ] );
    (* And the whole filename, punctuation included: splitting on "characters a path may contain"
       cut `helper+pp.exe` down to `pp.exe`, which is a different executable and a different
       exemption (Codex P2, round 4). Dune's own `%{` and `}` are the only boundaries. *)
    ( "a filename's punctuation is part of its identity",
      {dune|(rule (action (run %{dep:helper+pp.exe} --impl x.ml)))|dune},
      [ "rule running helper+pp.exe" ] );
    ( "a toolchain pform runs a compiler, not a test",
      {dune|(rule (action (run %{ocaml} script.ml)))|dune},
      [] );
    ( "a path built out of a pform is not guessed at",
      {dune|(rule (action (run %{dep:tools}/probe.exe)))|dune},
      [ "rule whose command this scan cannot read: %{dep:tools}/probe.exe" ] );
    (* What the file says, not what its prose says. *)
    ( "a stanza inside a comment is not a stanza",
      {dune|; (test (name phantom))
(test (name t) (deps ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "a config named in a comment does not declare it",
      {dune|; this one reads ocannl_config at startup
(test (name t))|dune},
      [ "test t" ] );
    ( "a config named in a string is not a dep",
      {dune|(test (name t) (action (run %{test} "ocannl_config")))|dune},
      [ "test t" ] );
    ( "an executable named in a comment does not make a rule a site",
      {dune|(rule (alias runtest)
 ; unlike bench.exe, this one only diffs
 (action (diff a.expected a.actual)))|dune},
      [] );
    (* A `(subdir …)` wrapper configures another directory from this dune file, and
       test/operations/dune runs two rules in test/operations/config that way. The stanzas inside
       are subject to the same rules, and the directory they name is the one whose config they
       need. *)
    ( "a subdir's stanzas are the same kinds of site",
      {dune|(subdir config
 (rule (deps ocannl_config) (action (run %{dep:reader.exe})))
 (test (name inner)))|dune},
      [ "in config: rule running reader.exe [declares]"; "in config: test inner" ] );
    ( "a nested subdir names the path it applies to",
      {dune|(subdir a (subdir b (test (name t) (deps ocannl_config))))|dune},
      [ "in a/b: test t [declares]" ] );
    ( "stanzas outside the subdir keep the dune file's own directory",
      {dune|(subdir config (test (name inner) (deps ocannl_config)))
(test (name outer) (deps ocannl_config))|dune},
      [ "in config: test inner [declares]"; "test outer [declares]" ] );
    ( "stanzas are reported in file order",
      {dune|(test (name a) (deps ocannl_config))
(executable (name e))
(rule (deps ocannl_config) (action (run %{dep:e.exe})))
(test (name b))|dune},
      [ "test a [declares]"; "rule running e.exe [declares]"; "test b" ] );
  ]

(* Stanzas that rewrite the executable search path for a whole directory, which the check refuses
   rather than modelling. Only the NAME position of an `env-vars` binding counts: setting some other
   variable to the literal value `PATH` rewrites nothing (Codex P2, round 18). *)
let path_rewriting_cases =
  [
    ("an env stanza setting PATH", {dune|(env (_ (env-vars (PATH .))))|dune}, [ "env" ]);
    ("an env stanza setting Windows-case Path", {dune|(env (_ (env-vars (Path .))))|dune}, [ "env" ]);
    ("an env stanza mapping binaries", {dune|(env (_ (binaries helper.exe)))|dune}, [ "env" ]);
    ("one setting something else to PATH", {dune|(env (_ (env-vars (OTHER PATH))))|dune}, []);
    ("an env stanza that touches neither", {dune|(env (_ (flags (:standard))))|dune}, []);
  ]

let path_rewriting_scope_cases =
  [
    ("a root env stanza", {dune|(env (_ (env-vars (PATH .))))|dune}, [ "" ]);
    ( "an env stanza nested under subdir",
      {dune|(subdir tools (env (_ (env-vars (PATH .)))))|dune},
      [ "tools" ] );
    ( "a binaries mapping nested under subdir",
      {dune|(subdir tools (env (_ (binaries helper.exe))))|dune},
      [ "tools" ] );
    ( "nested subdirs compose",
      {dune|(subdir tools (subdir private (env (_ (env-vars (Path .))))))|dune},
      [ "tools/private" ] );
    ( "an unrelated nested env stanza",
      {dune|(subdir tools (env (_ (env-vars (OTHER PATH)))))|dune},
      [] );
  ]

(* WHICH config file a stanza depends on, as written. The scan reports the paths; which of them is
   the one the process will actually read is the check's decision, since only it knows where configs
   exist -- OCANNL walks up from the process directory and reads the first it finds, so an
   ancestor's answers only while no nearer directory has its own (Codex P2, rounds 9 to 11). *)
let declared_paths_cases =
  [
    ("the local one", {dune|(test (name t) (deps ocannl_config))|dune}, [ "ocannl_config" ]);
    ( "one in a sibling",
      {dune|(rule (deps ../sibling/ocannl_config) (action (run %{dep:probe.exe})))|dune},
      [ "../sibling/ocannl_config" ] );
    ( "one in a common parent, which a sibling chdir may legitimately need",
      {dune|(rule (deps ../ocannl_config) (action (chdir ../b (run %{dep:probe.exe}))))|dune},
      [ "../ocannl_config" ] );
    ( "several, reported in full",
      {dune|(test (name t) (deps ocannl_config child/ocannl_config))|dune},
      [ "child/ocannl_config"; "ocannl_config" ] );
    ( "a config the dependency language does not depend on",
      {dune|(test (name t) (deps (alias ocannl_config) (glob_files ocannl_config)))|dune},
      [] );
  ]

(* Which directories the file materializes a config into: "." stands for its own. *)
let copy_cases =
  [
    ("a copy_files stanza", {dune|(copy_files (files ../config/ocannl_config))|dune}, [ "." ]);
    ("the short spelling", {dune|(copy_files ../config/ocannl_config)|dune}, [ "." ]);
    ( "a copy_files inside a subdir names that directory",
      {dune|(subdir config (copy_files (files ../../config/ocannl_config)))|dune},
      [ "config" ] );
    ("copying something else", {dune|(copy_files (files ../data/*.csv))|dune}, []);
    (* A wildcard that could cover the config counts as materializing it: this decides where a
       config EXISTS, so guessing wide risks accepting a directory that has one anyway, while
       guessing narrow rejects a correctly configured one (Codex P2, round 12). *)
    ("a wildcard that covers it", {dune|(copy_files (files ../config/*))|dune}, [ "." ]);
    ("a wildcard that cannot", {dune|(copy_files (files ../config/*.expected))|dune}, []);
    ( "dune's set syntax is taken as possibly matching",
      {dune|(copy_files (files ../config/{ocannl_config,other}))|dune},
      [ "." ] );
    ("no copy_files at all", {dune|(test (name t) (deps ocannl_config))|dune}, []);
  ]

(* Which stanza kinds the scan has no classification for. One it cannot place might carry an action
   that runs a test executable, so it is reported rather than counted as nothing. *)
let unclassified_cases =
  [
    ("a classified stanza", {dune|(test (name t) (deps ocannl_config))|dune}, []);
    ("a stanza that runs nothing", {dune|(ocamllex lexer)|dune}, []);
    (* Dune runs cram tests; this repository has none, and the day one appears the scan says so. *)
    ("a cram stanza", {dune|(cram (applies_to my_test) (deps helper.exe))|dune}, [ "cram" ]);
    ( "an include, whose contents the scan never sees",
      {dune|(include stanzas.inc)|dune},
      [ "include" ] );
    ("inside a subdir, too", {dune|(subdir gen (cram (applies_to x)))|dune}, [ "cram" ]);
    ("something that is not a stanza at all", {dune|bare_atom|dune}, [ "<not a stanza>" ]);
  ]

(* A stanza dune reads whole and sexplib reads with a hole in it: the markers are atoms to the one
   and comments to the other, and counting only the file's top-level forms would have found both
   readers reporting one stanza (Codex P2, round 17). *)
let nested_marker_case =
  ( "a marker nested inside a stanza",
    {dune|(rule (deps ocannl_config)
 (action (progn (echo #|) (run %{dep:probe.exe}) (echo |#))))|dune}
  )

(* The two sequences sexplib reads as comments and dune does not. Reading such a file would drop
   whatever they enclose, so the scan refuses it -- but only when something was actually dropped,
   which a second count of the top-level forms is what establishes. Inside a string or after a `;`,
   sexplib does not treat them as comments either, and refusing there would take the whole suite
   down over an unrelated argument (Codex P2, round 12). *)
let refused_cases =
  [
    ("a block comment", {dune|#| (test (name hidden)) |#
(test (name t))|dune});
    ("a datum comment", {dune|#;(test (name hidden))
(test (name t))|dune});
  ]

let accepted_marker_cases =
  [
    ( "the marker inside a string is not a comment to either reader",
      {dune|(rule (deps ocannl_config) (action (echo "#| not a comment |#")))|dune},
      [] );
    ( "nor is one in a line comment",
      {dune|; dune would read #| as an atom; sexplib would not
(test (name t) (deps ocannl_config))|dune},
      [ "test t [declares]" ] );
  ]

(* [raw_stanzas] is the independent floor `config_dep_completeness` holds the sexp walk to: a
   paren-depth reading of the raw text, which a walk going blind cannot take down with it. It has to
   agree with `sites` about SCOPE, or it fails a correct scan rather than a blind one, and each case
   below pins one of the places where a scope-blind reader disagrees (Codex P2, rounds 1 to 3).

   Rendered as "<head>[+inline] {cwd:exe …}", one entry per stanza. *)
let render_raw (r : Scan.raw_stanza) =
  let runs =
    List.map r.Scan.raw_runs ~f:(fun (cwd, exe) ->
        if String.is_empty cwd then exe else cwd ^ ":" ^ exe)
  in
  let tests =
    List.map r.Scan.raw_test_cwds ~f:(fun cwd ->
        if String.is_empty cwd then "%{test}" else cwd ^ ":%{test}")
  in
  let unnameable =
    List.map r.Scan.raw_unnameable ~f:(fun cwd -> if String.is_empty cwd then "!" else "!" ^ cwd)
  in
  (* `?` for what the reader records without naming: it says THAT something runs and nothing more,
     which is a weaker entry than the `!` of a bare command under `(setenv PATH …)` -- that one
     still names a directory (gh-ocannl-690). *)
  let opaque = List.map r.Scan.raw_opaque ~f:(fun what -> "?" ^ what) in
  Printf.sprintf "%s%s%s{%s}" r.Scan.raw_head
    (if String.is_empty r.Scan.raw_subdir then "" else "@" ^ r.Scan.raw_subdir)
    (if r.Scan.raw_inline_tests then "+inline" else "")
    (String.concat ~sep:" " (tests @ runs @ unnameable @ opaque))

let raw_stanza_cases =
  [
    (* Stanza position: only the top level, and inside `subdir`, which is the one form that contains
       stanzas. A `test` nested anywhere else is not a stanza -- `(env (test …))` names a build
       PROFILE, and `sites` rightly makes no test site of it. *)
    ("a test stanza", {dune|(test (name t))|dune}, [ "test{%{test}}" ]);
    ( "two of them",
      {dune|(test (name a))
(test (name b))|dune},
      [ "test{%{test}}"; "test{%{test}}" ] );
    ("a plural stanza is one stanza", {dune|(tests (names a b c))|dune}, [ "tests{%{test}}" ]);
    ( "a build profile named test is not a test stanza",
      {dune|(env (test (flags (:standard))))|dune},
      [ "env{}" ] );
    (* A `(subdir …)` moves where its stanzas run, and that directory is where the config is
       resolved -- so it composes into the recorded one, ready to compare against `in_subdir
       site.subdir site.cwd`. *)
    ( "a stanza inside a subdir runs there",
      {dune|(subdir gen (test (name t)))|dune},
      [ "subdir{}"; "test@gen{gen:%{test}}" ] );
    ( "and a rule inside one too",
      {dune|(subdir gen (rule (action (run %{dep:a.exe}))))|dune},
      [ "subdir{}"; "rule@gen{gen:a.exe}" ] );
    ( "nested subdirs compose",
      {dune|(subdir a (subdir b (rule (action (run %{dep:x.exe})))))|dune},
      [ "subdir{}"; "subdir@a{}"; "rule@a/b{a/b:x.exe}" ] );
    ( "a subdir composes with a chdir",
      {dune|(subdir gen (rule (action (chdir sub (run %{dep:a.exe})))))|dune},
      [ "subdir{}"; "rule@gen{gen/sub:a.exe}" ] );
    (* An inline_tests field belongs to the stanza it sits directly inside. *)
    ( "a library with inline tests",
      {dune|(library (name l) (inline_tests))|dune},
      [ "library+inline{}" ] );
    ("a library without", {dune|(library (name l) (libraries base))|dune}, [ "library{}" ]);
    ( "an inline_tests nested deeper is not the library's field",
      {dune|(library (name l) (env (inline_tests)))|dune},
      [ "library{}" ] );
    (* A library in a subdir runs its inline tests there, so that directory travels with it. *)
    ( "a library with inline tests inside a subdir",
      {dune|(subdir child (library (name l) (inline_tests)))|dune},
      [ "subdir{}"; "library@child+inline{}" ] );
    (* What runs things, and where. *)
    ( "a rule running a dep pform",
      {dune|(rule (action (run %{dep:probe.exe})))|dune},
      [ "rule{probe.exe}" ] );
    ("a bare path", {dune|(rule (action (run ./probe.exe)))|dune}, [ "rule{probe.exe}" ]);
    (* An explicit relative path names something this repository built whatever its extension --
       `classify_command` treats `./probe` as a site, so the floor must too. *)
    ("an extensionless explicit path", {dune|(rule (action (run ./probe)))|dune}, [ "rule{probe}" ]);
    (* A bare word is a tool on PATH, and while it is handed nothing this workspace provides it is
       the end of the story for BOTH readers -- `sites` places no site for it either. This is the
       negative control the rule below is written against: what makes the difference is the
       ARGUMENT, not the command. *)
    ( "but a bare word is a tool on PATH",
      {dune|(rule (action (run python3 x.py)))|dune},
      [ "rule{}" ] );
    (* Handed a file this workspace builds, the same tool is a stanza both readers see. The walk
       reads it as a program that may run in a directory it cannot establish -- dune's grammar does
       not say whether `python3 %{dep:orchestrate.py}` runs our file or merely reads it -- and until
       gh-ocannl-708 the floor named nothing here, leaving this one stanza in `benchmarks/dune`
       standing on the walk alone. *)
    ( "but handed a file this workspace builds, it is one both readers see",
      {dune|(rule (action (run python3 %{dep:orchestrate.py})))|dune},
      [ "rule{?python3, handed %{dep:orchestrate.py}}" ] );
    (* The counter-example that rules out narrowing the walk instead: `env -C ../sibling probe.exe`
       is the same shape -- an external command handed workspace paths -- and it does launch
       something of ours, somewhere else. Both paths are named, since either could be the
       program. *)
    ( "and a launcher pointed elsewhere counts for the same reason",
      {dune|(rule (action (run env -C ../sibling ./probe.exe)))|dune},
      [ "rule{?env, handed ../sibling, ./probe.exe}" ] );
    (* A tool merely READING our executables is the same text, and neither reader guesses which it
       is: the walk reports the pair and the check settles it, so the floor must name the stanza
       too. *)
    ( "a tool only reading them is the same text",
      {dune|(rule (action (run diff old.exe new.exe)))|dune},
      [ "rule{?diff, handed old.exe, new.exe}" ] );
    (* But an argument out of the TOOLCHAIN is not out of this workspace, which is the whole use of
       sharing `Scan.toolchain_pforms` as data: without the exclusion this line would claim a stanza
       the walk places nothing for, turning the floor from a lower bound into a false alarm. *)
    ( "a toolchain pform handed to one is not a workspace file",
      {dune|(rule (action (run ocamlfind %{ocaml} -version)))|dune},
      [ "rule{}" ] );
    (* Nor is an absolute path, or a plain word: they name what the system provides. *)
    ( "nor is an absolute path",
      {dune|(rule (action (run cp /usr/bin/probe.copy dest)))|dune},
      [ "rule{}" ] );
    (* The lost directory tags what it encloses here as everywhere else. *)
    ( "one under an unresolvable chdir keeps both readings",
      {dune|(rule (action (chdir %{root} (run python3 %{dep:orchestrate.py}))))|dune},
      [ "rule{?python3, handed %{dep:orchestrate.py}, under `(chdir %{root} ...)`}" ] );
    ( "a path elsewhere keeps its path",
      {dune|(rule (action (run %{dep:../../tools/minised.exe})))|dune},
      [ "rule{../../tools/minised.exe}" ] );
    ( "a comma in the path is part of it",
      {dune|(rule (action (run %{dep:helper,pp.exe})))|dune},
      [ "rule{helper,pp.exe}" ] );
    ( "several in one stanza",
      {dune|(rule (action (progn (run %{dep:a.exe} x) (run %{dep:b.exe}))))|dune},
      [ "rule{a.exe b.exe}" ] );
    ( "the same one twice in a stanza is one entry",
      {dune|(rule (action (progn (run %{dep:a.exe} x) (run %{dep:a.exe} y))))|dune},
      [ "rule{a.exe}" ] );
    ( "the same one in two stanzas is two stanzas",
      {dune|(rule (action (run %{dep:a.exe} x)))
(rule (action (run %{dep:a.exe} y)))|dune},
      [ "rule{a.exe}"; "rule{a.exe}" ] );
    (* A chdir moves the process, and `sites` emits one site per working directory because each
       resolves a different config -- so the same executable under two chdirs is two entries. *)
    ( "a chdir names where it runs",
      {dune|(rule (action (chdir ../sibling (run %{dep:a.exe}))))|dune},
      [ "rule{../sibling:a.exe}" ] );
    ( "one executable under two chdirs is two entries",
      {dune|(rule (action (progn (chdir d1 (run %{dep:a.exe})) (chdir d2 (run %{dep:a.exe})))))|dune},
      [ "rule{d1:a.exe d2:a.exe}" ] );
    (* `./%{test}` is the test binary too -- the `./` says only "here" -- so the directory its
       branch names is where the Test site goes. *)
    ( "the test pform behind a ./",
      {dune|(test (name t) (action (chdir child (run ./%{test}))))|dune},
      [ "test{child:%{test}}" ] );
    (* Quoted atoms are atoms wherever they appear, including a form's HEAD and a binding's value.
       Nothing here handles them specially: the text is parsed, so they arrive already decoded. *)
    ( "a quoted head is the same head",
      {dune|(rule (action ("run" %{dep:probe.exe})))|dune},
      [ "rule{probe.exe}" ] );
    ( "a quoted path in a binding",
      {dune|(rule (deps (:pp "pp.exe")) (action (run %{pp})))|dune},
      [ "rule{pp.exe}" ] );
    ( "a quoted stanza head",
      {dune|("rule" (action (run %{dep:probe.exe})))|dune},
      [ "rule{probe.exe}" ] );
    (* A quoted atom is an atom: `(chdir "scratch dir" …)` names a directory with a space in it. *)
    ( "a quoted chdir destination",
      {dune|(rule (action (chdir "scratch dir" (run %{dep:probe.exe}))))|dune},
      [ "rule{scratch dir:probe.exe}" ] );
    ("a quoted command", {dune|(rule (action (run "./probe.exe")))|dune}, [ "rule{probe.exe}" ]);
    ( "an escape inside a quoted atom",
      {dune|(rule (action (chdir "with\"quote" (run %{dep:probe.exe}))))|dune},
      [ "rule{with\"quote:probe.exe}" ] );
    ( "nested chdirs compose",
      {dune|(rule (action (chdir a (chdir b (run %{dep:x.exe})))))|dune},
      [ "rule{a/b:x.exe}" ] );
    ("an alias runs things too", {dune|(alias (action (run %{dep:a.exe})))|dune}, [ "alias{a.exe}" ]);
    ( "a test's custom action",
      {dune|(test (name t) (action (run %{dep:helper.exe})))|dune},
      [ "test{%{test} helper.exe}" ] );
    (* A test's own binary runs where its action puts it, and `sites` emits one Test site per
       directory because each resolves a different config -- so two chdir branches are two. *)
    ( "a test's own binary under a chdir",
      {dune|(test (name t) (action (chdir ../sibling (run %{test}))))|dune},
      [ "test{../sibling:%{test}}" ] );
    ( "and in two branches, two directories",
      {dune|(test (name t) (action (progn (chdir d1 (run %{test})) (chdir d2 (run %{test})))))|dune},
      [ "test{d1:%{test} d2:%{test}}" ] );
    (* Under a chdir the text cannot resolve, neither reader can say WHERE the process runs -- the
       walk emits a site carrying no executables, and this one records the command without its
       directory. Tagged rather than dropped: that something runs there is the whole of what the
       per-stanza floor needs, and dropping it left such a stanza with no floor under it at all
       (gh-ocannl-690). *)
    ( "a run under an unresolvable chdir is tagged, not dropped",
      {dune|(rule (action (chdir %{workspace_root} (run %{dep:probe.exe}))))|dune},
      [ "rule{?probe.exe, under `(chdir %{workspace_root} ...)`}" ] );
    (* Only a command it could otherwise NAME. A PATH tool is external wherever it runs, so the walk
       places no site for it and claiming one here would turn the floor into a false alarm. *)
    ( "but a PATH tool under one is external wherever it runs",
      {dune|(rule (action (chdir %{workspace_root} (run python3 x.py))))|dune},
      [ "rule{}" ] );
    ( "a name the stanza binds resolves under one too",
      {dune|(rule (deps (:pp pp.exe)) (action (chdir %{root} (run ./%{pp}))))|dune},
      [ "rule{?pp.exe, under `(chdir %{root} ...)`}" ] );
    (* Evidence from two enclosing forms must not cancel out. A bare name is external under a
       `chdir` alone, but NOT under a rewritten PATH -- there the walk places a site because PATH
       may point it at a workspace executable -- so an unresolvable `chdir` around that must not
       erase what the `setenv` established. Either nesting order, since dune admits both. *)
    ( "a bare command under both a rewritten PATH and an unresolvable chdir",
      {dune|(rule (action (setenv PATH . (chdir %{root} (run probe)))))|dune},
      [ "rule{?probe, under `(setenv PATH ...)` and `(chdir %{root} ...)`}" ] );
    ( "and in the other nesting order",
      {dune|(rule (action (chdir %{root} (setenv PATH . (run probe)))))|dune},
      [ "rule{?probe, under `(setenv PATH ...)` and `(chdir %{root} ...)`}" ] );
    (* Still the raw_unnameable floor's own entry when the chdir CAN be resolved: that one carries a
       directory, and the walk's site for it carries `path_rewritten`. *)
    ( "a resolvable chdir around one keeps it a directoried entry",
      {dune|(rule (action (setenv PATH . (chdir sub (run probe)))))|dune},
      [ "rule{!sub}" ] );
    (* The test's own binary is the exception that proves the rule: the walk's `%{test}` filter
       drops the wrapped command too, so its Test site falls back to the stanza's own directory --
       and this reader falls back the same way, which is what keeps them equal. *)
    ( "a test's own binary under one falls back to its own directory",
      {dune|(test (name t) (action (chdir %{workspace_root} (run %{test}))))|dune},
      [ "test{%{test}}" ] );
    ( "and the whole subtree beneath it, however deep",
      {dune|(rule (action (chdir %{root} (chdir sub (run %{dep:probe.exe})))))|dune},
      [ "rule{?probe.exe, under `(chdir %{root} ...)`}" ] );
    (* Dune allows whitespace and comments after an opening paren; a head read as empty would make
       the stanza invisible to a floor whose whole job is seeing it. *)
    ("whitespace before the head", "(\n test (name probe)\n)", [ "test{%{test}}" ]);
    ("a comment before the head", "(; which test\n test (name probe))", [ "test{%{test}}" ]);
    ( "and before a chdir destination",
      "(rule (action (chdir ; here\n ../sibling (run %{dep:a.exe}))))",
      [ "rule{../sibling:a.exe}" ] );
    (* What `sites` makes no site of, this makes no entry of. *)
    ( "a library's preprocessor is not a test-running rule",
      {dune|(library (name l) (preprocess (action (run %{dep:pp.exe} x))))|dune},
      [ "library{}" ] );
    ( "nor is an executable's",
      {dune|(executable (name e) (preprocess (action (run %{dep:pp.exe}))))|dune},
      [ "executable{}" ] );
    (* Declined because the text alone does not say what they resolve to: all under-report, which is
       the safe direction for a floor. *)
    ( "the test pform runs where the action puts it",
      {dune|(test (name t) (action (run %{test} --flag)))|dune},
      [ "test{%{test}}" ] );
    (* A shell line is not parsed -- splitting it on whitespace would be reading it, and reading it
       wrong -- but it is RECORDED, because a rule that runs its test through a shell is subject to
       the same rules as one that runs it directly. *)
    ( "a shell line runs something unnamed",
      {dune|(rule (action (bash "./probe.exe")))|dune},
      [ "rule{?(bash ./probe.exe)}" ] );
    ( "and so does a system one",
      {dune|(rule (action (system "probe")))|dune},
      [ "rule{?(system probe)}" ] );
    ( "one whose shell the text could not read either way",
      {dune|(rule (action (bash "if ready; then ./probe.exe; fi")))|dune},
      [ "rule{?(bash if ready; then ./probe.exe; fi)}" ] );
    ( "a test stanza running its own binary through a shell",
      {dune|(test (name t) (action (bash "./t.exe --flag")))|dune},
      [ "test{%{test} ?(bash ./t.exe --flag)}" ] );
    ( "two shell lines are two, and repeats collapse",
      {dune|(rule (action (progn (bash "a") (bash "b") (bash "a"))))|dune},
      [ "rule{?(bash a) ?(bash b)}" ] );
    ( "a shell line under an unresolvable chdir is still one thing that runs",
      {dune|(rule (action (chdir %{root} (bash "./probe.exe"))))|dune},
      [ "rule{?(bash ./probe.exe)}" ] );
    (* Where no stanza runs anything, a shell line is no more a run than a `(run …)` there is. *)
    ( "a shell line in a library's preprocessor is not one",
      {dune|(library (name l) (preprocess (action (bash "./pp.exe"))))|dune},
      [ "library{}" ] );
    (* A `(:name …)` dependency binds an executable, and the binding sits in the same stanza -- so
       the text CAN resolve it, and the three rules of test/ppx/dune are all of this shape. *)
    ( "a name bound by a dep",
      {dune|(rule (deps (:pp pp.exe)) (action (run %{pp})))|dune},
      [ "rule{pp.exe}" ] );
    ( "and one behind a ./, which says only \"here\"",
      {dune|(rule (deps (:pp pp.exe)) (action (run ./%{pp} --impl x)))|dune},
      [ "rule{pp.exe}" ] );
    ( "a binding may wrap its path in a dependency form",
      {dune|(rule (deps (:runner (file probe.exe))) (action (run %{runner})))|dune},
      [ "rule{probe.exe}" ] );
    (* The binding may be written after the action that uses it, so resolution waits for the whole
       stanza. *)
    ( "a binding written after the action",
      {dune|(rule (action (run ./%{pp})) (deps (:pp pp.exe)))|dune},
      [ "rule{pp.exe}" ] );
    (* Each stanza has its own bindings: one rule's does not resolve another's pform. The second
       rule's command is still SOMETHING out of this workspace -- the walk reports it as a command
       it cannot read and places a site -- so the floor names the stanza while declining to name the
       program (gh-ocannl-708). Which list the entry lands in is the difference: `pp.exe` is an
       identity `config_dep_completeness` matches against the walk's, and `?…` says only that the
       stanza runs something. *)
    ( "a binding does not leak to the next stanza",
      {dune|(rule (deps (:pp pp.exe)) (action (run %{pp})))
(rule (action (run %{pp})))|dune},
      [ "rule{pp.exe}"; "rule{?%{pp}, itself named out of this workspace}" ] );
    (* A binding's paths are the atoms and the forms that CARRY paths -- an `(alias …)` names
       something dune does not run, so an `.exe`-looking atom inside one is not the binding's
       value. *)
    ( "a non-path form in a binding is not its value",
      {dune|(rule (deps (:runner (alias fake.exe) (file real.exe))) (action (run %{runner})))|dune},
      [ "rule{real.exe}" ] );
    ( "and a binding of only a non-path form resolves to nothing",
      {dune|(rule (deps (:runner (alias fake.exe))) (action (run %{runner})))|dune},
      [ "rule{?%{runner}, itself named out of this workspace}" ] );
    ( "a binding outside the deps field is not a binding",
      {dune|(rule (action (progn (:pp pp.exe) (run %{pp}))))|dune},
      [ "rule{?%{pp}, itself named out of this workspace}" ] );
    (* `(setenv PATH …)` changes what a bare name resolves to, so the walk stops vouching for the
       program -- including an executable-looking literal -- and the floor records that it must have
       said so. An explicit path or pform stays an ordinary run because PATH cannot redirect it. *)
    ( "a bare command under setenv PATH is unnameable",
      {dune|(rule (action (setenv PATH . (run probe))))|dune},
      [ "rule{!}" ] );
    ( "a bare executable-looking command under setenv PATH is unnameable",
      {dune|(rule (action (setenv PATH . (run probe.exe))))|dune},
      [ "rule{!}" ] );
    ( "a Windows-case PATH spelling is unnameable too",
      {dune|(rule (action (setenv Path . (run probe.exe))))|dune},
      [ "rule{!}" ] );
    ( "but a named executable under one is still a run",
      {dune|(rule (action (setenv PATH . (run %{dep:probe.exe}))))|dune},
      [ "rule{probe.exe}" ] );
    ( "two bare ones are two, and repeats collapse",
      {dune|(rule (action (setenv PATH . (progn (run a) (run b) (run a)))))|dune},
      [ "rule{! !}" ] );
    (* The identity matters, not unnameability in general: a `(bash …)` in the same file is
       unnameable for an unrelated reason and must not answer for this one. *)
    ( "a shell line alongside one is a different thing",
      {dune|(rule (action (setenv PATH . (run probe))))
(rule (action (bash "./thing.exe")))|dune},
      [ "rule{!}"; "rule{?(bash ./thing.exe)}" ] );
    ( "a chdir under one still names where it runs",
      {dune|(rule (action (setenv PATH . (chdir sub (run %{dep:a.exe})))))|dune},
      [ "rule{sub:a.exe}" ] );
    ( "a binding resolving to no executable stays unnamed, and the stanza still floored",
      {dune|(rule (deps (:script run.sh)) (action (run %{script})))|dune},
      [ "rule{?%{script}, itself named out of this workspace}" ] );
    (* And the exclusion that keeps the two readers apart from a compiler: `%{ocamlc}` runs the
       toolchain, `-c` and `x.ml` name nothing dune resolves out of this workspace, so neither
       reader sees a stanza here. The list saying so is `Scan.toolchain_pforms`, read by both
       (gh-ocannl-708). *)
    ( "a toolchain pform is not ours",
      {dune|(rule (action (run %{ocamlc} -c x.ml)))|dune},
      [ "rule{}" ] );
    ("an executable only depended on", {dune|(rule (deps %{dep:probe.exe}))|dune}, [ "rule{}" ]);
    ( "one named in a comment",
      {dune|(rule (action (progn)))
; (run %{dep:probe.exe})|dune},
      [ "rule{}" ] );
    ( "one inside a string",
      {dune|(rule (action (echo "(run %{dep:probe.exe})")))|dune},
      [ "rule{}" ] );
  ]

(* gh-ocannl-659's marker: a comment inside a stanza's parentheses saying which backend the stanza
   is pinned to -- or `none` -- and why, for the stanzas that do not declare `(env_var
   OCANNL_BACKEND)`. The grammar is rigid on purpose, so the cases that matter are the near misses:
   a marker the reader silently declines to parse leaves its stanza declaring nothing, reported as
   if the author had written none at all.

   Rendered as "<what>|<reason>" for a marker, "!<why>" for one the grammar rejects, and "-" for a
   comment that is not a marker. *)
let render_marker text =
  match Scan.parse_marker text with
  | None -> "-"
  | Some (Scan.Marker m) -> m.Scan.backend ^ "|" ^ m.Scan.reason
  | Some (Scan.Malformed why) -> "!" ^ why

let marker_grammar_cases =
  [
    ("ordinary prose is not a marker", " a note about the backend", "-");
    ("the plain shape", " ocannl-backend: none -- links no backend", "none|links no backend");
    (* The em dash is what this repository's prose uses and `--` is what a keyboard produces;
       refusing either would be a grammar that fails for a reason nobody can see in a diff. *)
    ( "an em dash separates too",
      " ocannl-backend: cc \xe2\x80\x94 names its backend",
      "cc|names its backend" );
    ( "spacing around the colon is free",
      ";ocannl-backend:metal -- pins MSL emission",
      "metal|pins MSL emission" );
    (* A stanza may honestly name two backends; `none` makes no such pair. *)
    ( "two backends, for a stanza that names both",
      " ocannl-backend: cc,multidev_cc -- names both by argument",
      "cc,multidev_cc|names both by argument" );
    ( "none and a backend at once is a contradiction",
      " ocannl-backend: none,cc -- both at once",
      "!`none,cc` says both that the run depends on a backend and that it depends on none" );
    (* The near misses. Each of these WOULD read as "no marker at all" under a looser grammar, and
       its stanza would then be reported as undeclared -- or, worse, as declared. *)
    ( "a backend the repository does not have",
      " ocannl-backend: metl -- a typo for metal",
      "!`metl` is not one of none, cc, multidev_cc, cuda, hip, metal" );
    ( "no separator at all",
      " ocannl-backend: none links no backend",
      "!no `--` separating the backend from the reason -- the grammar is `; ocannl-backend: "
      ^ "<none|cc|multidev_cc|cuda|hip|metal> -- <reason>`" );
    ( "a one-word reason is a label, not a reason",
      " ocannl-backend: none -- pure",
      "!the reason `pure` is one word -- say why, not what" );
    ("an empty reason", " ocannl-backend: cc --", "!the reason `` is one word -- say why, not what");
    (* The separator is the EARLIEST one, not the first spelling that occurs anywhere: taking the
       `--` here would put "cc \xe2\x80\x94 pinned" in the backend position and swallow half the
       sentence. *)
    ( "an em dash before a double dash in the reason",
      " ocannl-backend: cc \xe2\x80\x94 pinned -- really pinned",
      "cc|pinned -- really pinned" );
    (* Announced anywhere in the comment: a marker someone annotated is still a marker, and reading
       it from the sentinel rather than from the start of the line is what keeps it one. *)
    ( "annotated prose before it",
      " NOTE ocannl-backend: hip -- names its backend",
      "hip|names its backend" );
    (* Refused rather than normalised. Each of these is a TYPO in the one comment whose whole job is
       to be checkable, and a grammar that quietly repaired it would hand back a clean answer for a
       marker its author got wrong -- which is the failure mode the malformed/absent distinction
       exists to prevent (Codex P2, round 1). *)
    ( "a trailing comma is an empty entry, not a clean single backend",
      " ocannl-backend: cc, -- pins the backend",
      "!`cc,` has an empty entry between commas -- name each backend, or drop the comma" );
    ( "and so is a doubled comma between two real ones",
      " ocannl-backend: cc,,metal -- names both",
      "!`cc,,metal` has an empty entry between commas -- name each backend, or drop the comma" );
    ( "the same backend twice is a typo, not a list",
      " ocannl-backend: cc,cc -- names its backend",
      "!`cc,cc` names the same backend twice" );
    (* Reading from the earliest sentinel and letting the rest fall into the reason would absorb a
       whole second declaration into prose -- and the accounting check cannot see it, since both
       occurrences ARE in a comment the scan places. *)
    ( "two declarations sharing one comment",
      " ocannl-backend: none -- links no backend ocannl-backend: cc -- names its backend",
      "!two `ocannl-backend:` declarations in one comment -- the second would be read as part of \
       the first's reason; put each on its own line" );
  ]

(* Which stanza a marker belongs to. The attribution rule is containment -- between the stanza's
   parentheses -- and these cases are the ones an adjacency rule gets wrong: this repository's dune
   files habitually leave a blank line between a comment block and the stanza below it, so "the
   comment above" would have to guess how far above.

   Rendered as "<head> <name>[*] {<markers>}", where `*` marks a stanza that runs something. *)
let render_marked (m : Scan.marked_stanza) =
  Printf.sprintf "%s %s%s {%s}" m.Scan.marked_head
    (if String.is_empty m.Scan.marked_name then "-" else m.Scan.marked_name)
    (if List.is_empty m.Scan.marked_sites then "" else "*")
    (String.concat ~sep:","
       (List.filter_map m.Scan.marked_comments ~f:(fun (_, text) ->
            match Scan.parse_marker text with
            | Some (Scan.Marker m) -> Some m.Scan.backend
            | Some (Scan.Malformed _) -> Some "!"
            | None -> None)))

let marker_placement_cases =
  [
    ( "inside the stanza it is about",
      {dune|(test (name t)
 ; ocannl-backend: none -- links no backend
 (deps ocannl_config))|dune},
      [ "test t* {none}" ] );
    (* Between two stanzas it belongs to neither, which is what makes a misplaced one reportable
       rather than silently absent. *)
    ( "between two stanzas it belongs to neither",
      {dune|(test (name a) (deps ocannl_config))
; ocannl-backend: none -- links no backend
(test (name b) (deps ocannl_config))|dune},
      [ "test a* {}"; "test b* {}" ] );
    (* The blank line an adjacency rule would have to see past. *)
    ( "a comment above the stanza, blank line and all",
      {dune|; ocannl-backend: none -- links no backend

(test (name t) (deps ocannl_config))|dune},
      [ "test t* {}" ] );
    (* An `(executable)` has no `deps` field at all, so its companion rule is where both the config
       dep and this marker go; a marker on the executable declares nothing. *)
    ( "on an executable, which runs nothing",
      {dune|(executable (name e)
 ; ocannl-backend: none -- links no backend
 (modules e))|dune},
      [ "executable e {none}" ] );
    ( "a subdir's stanzas keep their own markers",
      {dune|(subdir sub (rule
 ; ocannl-backend: cc -- pins the backend on the command line
 (deps ocannl_config) (action (run %{dep:probe.exe}))))|dune},
      [ "rule -* {cc}" ] );
    (* Sitting in the subdir but in none of its stanzas: attributed to nothing, exactly as the
       between-stanzas case above. *)
    ( "one in the subdir but in no stanza of it",
      {dune|(subdir sub
 ; ocannl-backend: cc -- pins the backend on the command line
 (rule (deps ocannl_config) (action (run %{dep:probe.exe}))))|dune},
      [ "rule -* {}" ] );
    ( "the declaration and the marker are read independently",
      {dune|(test (name t)
 ; ocannl-backend: none -- links no backend
 (deps ocannl_config (env_var OCANNL_BACKEND)))|dune},
      [ "test t* {none}" ] );
  ]

(* gh-ocannl-659's rule itself, put to stanzas this repository does not contain.
   [Scan.backend_rule_of] is the decision `env_var_deps` acts on, and until gh-ocannl-690 the shapes
   below were where the two readers disagreed: a rule that runs its test through a shell, and one
   under a `chdir` no reader can resolve. The rule always applied to them -- the walk places their
   sites -- but nothing independent vouched for that, so a walk that stopped seeing them would have
   looked exactly like a file with nothing to check.

   Rendered as "<head> <verdict>", with "+floor" appended when the SECOND reader also sees the
   stanza running something. The pairing is the point: "reported, +floor" is a hole named by both
   readers, and "reported, no floor" is one named by the walk alone -- which is the state a blind
   walk can quietly leave. *)
let render_rule (contract : Scan.marker_body Scan.marker_contract)
    (marked : Scan.marker_body Scan.marker_stanza) =
  let m = marked.Scan.marker_stanza in
  let verdict =
    match
      List.find contract.Scan.contract_issues ~f:(function
        | Scan.Marker_in_wrong_stanza { issue_stanza; _ } -> phys_equal issue_stanza m
        | Scan.Malformed_marker _ | Scan.Marker_outside_stanza _ | Scan.Marker_outside_comment _ ->
            false)
    with
    | Some (Scan.Marker_in_wrong_stanza _) -> "a marker on a stanza that runs nothing"
    | Some _ -> assert false
    | None -> (
        match Scan.backend_rule_of marked with
        | Scan.Runs_nothing -> "runs nothing"
        | Scan.Declares_variable -> "declares the variable"
        | Scan.Names_backend (_, b) -> "names " ^ b.Scan.backend
        | Scan.Declares_and_names (_, b) -> "declares AND names " ^ b.Scan.backend
        | Scan.Names_twice _ -> "names a backend twice"
        | Scan.Names_neither -> "REPORTED: declares neither")
  in
  Printf.sprintf "%s %s%s" m.Scan.marked_head verdict
    (if m.Scan.marked_raw_subject then " +floor" else "")

let backend_rule_cases =
  [
    (* The shape the rule is built for, as a baseline: a rule that runs an executable and says
       nothing about the backend. *)
    ( "a plain run declaring neither is reported",
      {dune|(rule (deps ocannl_config) (action (run %{dep:probe.exe})))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    (* And the same thing said through a shell. Before gh-ocannl-690 this line read "REPORTED:
       declares neither" with no floor under it. *)
    ( "a shell action declaring neither is reported, and the floor says so too",
      {dune|(rule (deps ocannl_config) (action (bash "./probe.exe")))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    ( "the same rule carrying a marker passes",
      {dune|(rule
 ; ocannl-backend: cc -- runs the cc probe through a shell
 (deps ocannl_config)
 (action (bash "./probe.exe")))|dune},
      [ "rule names cc +floor" ] );
    ( "and one declaring the variable instead",
      {dune|(rule (deps ocannl_config (env_var OCANNL_BACKEND)) (action (bash "./probe.exe")))|dune},
      [ "rule declares the variable +floor" ] );
    ( "both at once stays contradictory through a shell",
      {dune|(rule
 ; ocannl-backend: cc -- runs the cc probe through a shell
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (bash "./probe.exe")))|dune},
      [ "rule declares AND names cc +floor" ] );
    ( "a system action is the same action",
      {dune|(rule (deps ocannl_config) (action (system "./probe.exe")))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    ( "a test running its own binary through a shell",
      {dune|(test (name t) (deps ocannl_config) (action (bash "./t.exe")))|dune},
      [ "test REPORTED: declares neither +floor" ] );
    (* A `chdir` the text cannot resolve moves where the process runs and nothing else: the rule
       still applies, and now the floor still holds. *)
    ( "a run under an unresolvable chdir is reported, with a floor under it",
      {dune|(rule (deps ocannl_config) (action (chdir %{workspace_root} (run %{dep:probe.exe}))))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    ( "and passes with a marker",
      {dune|(rule
 ; ocannl-backend: none -- copies a file, links no backend
 (deps ocannl_config)
 (action (chdir %{workspace_root} (run %{dep:probe.exe}))))|dune},
      [ "rule names none +floor" ] );
    (* The negative control on the other side: over-claiming is as bad as under-claiming, because a
       floor that sees a stanza the walk does not is a floor that fails a correct scan. A PATH tool
       is external wherever a `chdir` sends it, so neither reader counts it. *)
    ( "a PATH tool under an unresolvable chdir is counted by neither reader",
      {dune|(rule (action (chdir %{workspace_root} (run python3 x.py))))|dune},
      [ "rule runs nothing" ] );
    (* But the same bare name under a rewritten PATH is a site the walk DOES place, and an
       unresolvable `chdir` around it does not take that back. *)
    ( "a bare command under a rewritten PATH and an unresolvable chdir keeps its floor",
      {dune|(rule (deps ocannl_config) (action (setenv PATH . (chdir %{root} (run probe)))))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    ( "nor is a shell line where no stanza runs anything",
      {dune|(library (name l) (preprocess (action (bash "./pp.exe"))))|dune},
      [ "library runs nothing" ] );
    (* A marker on a stanza that runs nothing declares nothing -- including one whose only action is
       a shell line the walk does place. Kept here so the arm cannot be reached by accident. *)
    ( "a marker on a library that runs nothing is still misplaced",
      {dune|(library (name l)
 ; ocannl-backend: none -- links no backend
 (preprocess (action (bash "./pp.exe"))))|dune},
      [ "library a marker on a stanza that runs nothing" ] );
    (* gh-ocannl-708, as the rule sees it. `benchmarks/dune` runs its orchestrator this way, and it
       was the last stanza in the repository the walk placed a site for and the floor named nothing
       for: the rule applied to it, and nothing independent vouched for that. *)
    ( "an external command handed a workspace file is reported, with a floor under it",
      {dune|(rule (deps ocannl_config) (action (run python3 %{dep:orchestrate.py})))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
    ( "and passes with a marker, still floored",
      {dune|(rule
 ; ocannl-backend: none -- hands a script to python3, which links no backend
 (deps ocannl_config)
 (action (run python3 %{dep:orchestrate.py})))|dune},
      [ "rule names none +floor" ] );
    (* The negative control for it: the same tool handed nothing this workspace provides is a stanza
       NEITHER reader sees, so the rule does not apply and no floor claims it does. Over-claiming
       here would fail a correct scan. *)
    ( "the same tool handed nothing of ours is counted by neither reader",
      {dune|(rule (action (run python3 x.py)))|dune},
      [ "rule runs nothing" ] );
    (* And the counter-example that kept the walk as it is: `env -C ../sibling ./probe.exe` wears
       the same clothes and does launch something of ours. *)
    ( "a launcher pointed at a sibling directory keeps its floor",
      {dune|(rule (deps ocannl_config) (action (run env -C ../sibling ./probe.exe)))|dune},
      [ "rule REPORTED: declares neither +floor" ] );
  ]

(* gh-ocannl-1518: a backend the action fixes is the marker's to name. [Scan.backend_pins] reads the
   forms its header lists and [Scan.backend_pin_issues] decides; each stanza renders as "<head>:"
   and its issues, "clean" when it has none. Every violating arm sits beside the nearest legitimate
   one, and the forms that merely MENTION the backend -- a declaration, an `enabled_if`, a path read
   out of the configuration -- are pinned as no pin at all. *)
let render_pin_issue = function
  | Scan.Pin_unread form -> "unread " ^ form
  | Scan.Pin_off_vocabulary { off_backend; _ } -> "off the vocabulary: " ^ off_backend
  | Scan.Pin_unnamed { unnamed_backend; named; _ } ->
      Printf.sprintf "unnamed %s (%s)" unnamed_backend
        (match named with
        | None -> "no marker"
        | Some words -> "marker: " ^ String.concat ~sep:"," words)

let render_pins source =
  let contract = Scan.backend_marker_contract source in
  List.map contract.Scan.contract_stanzas ~f:(fun marked ->
      let stanza = marked.Scan.marker_stanza in
      let issues =
        Scan.backend_pin_issues (Scan.backend_rule_of marked)
          (Scan.backend_pins stanza.Scan.marked_sexp)
      in
      stanza.Scan.marked_head ^ ": "
      ^
      match issues with
      | [] -> "clean"
      | issues -> String.concat ~sep:"; " (List.map issues ~f:render_pin_issue))

let backend_pin_cases =
  [
    ( "a setenv pin its marker names",
      {dune|(rule
 ; ocannl-backend: hip -- the action pins hip
 (action (setenv OCANNL_BACKEND hip (run %{dep:probe.exe}))))|dune},
      [ "rule: clean" ] );
    (* The shape staging#1318's round 1 found: `Slot_kind` would read cc and run hip. *)
    ( "a setenv pin its marker leaves out",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (setenv OCANNL_BACKEND hip (run %{dep:probe.exe}))))|dune},
      [ "rule: unnamed hip (marker: cc)" ] );
    ( "a setenv pin under a declaration instead of a marker",
      {dune|(rule (deps (env_var OCANNL_BACKEND))
 (action (setenv OCANNL_BACKEND hip (run %{dep:probe.exe}))))|dune},
      [ "rule: unnamed hip (no marker)" ] );
    (* Windows folds the name, so the lowercase spelling pins there. *)
    ( "a lowercase setenv name pins too",
      {dune|(rule
 ; ocannl-backend: cuda -- the action pins cuda
 (action (setenv ocannl_backend cuda (run %{dep:probe.exe}))))|dune},
      [ "rule: clean" ] );
    ( "a command-line pin its marker names",
      {dune|(rule
 ; ocannl-backend: cc -- the command line pins cc
 (action (run %{dep:probe.exe} --ocannl_backend=cc)))|dune},
      [ "rule: clean" ] );
    ( "a command-line pin under a `none` marker",
      {dune|(rule
 ; ocannl-backend: none -- claims no backend
 (action (run %{dep:probe.exe} --ocannl_backend=cc)))|dune},
      [ "rule: unnamed cc (marker: none)" ] );
    (* The runtime's own reader decides what is a pin, so every spelling it accepts is one. *)
    ( "the prefix-free and dashed spellings are pins as well",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (progn (run %{dep:probe.exe} --backend=cuda) (run %{dep:probe.exe} -OCANNL-BACKEND=metal))))|dune},
      [ "rule: unnamed cuda (marker: cc); unnamed metal (marker: cc)" ] );
    ( "a marker naming several backends answers for each pinned one",
      {dune|(rule
 ; ocannl-backend: cc,multidev_cc -- runs the model on both
 (action
  (progn
   (run %{dep:probe.exe} --ocannl_backend=cc)
   (setenv OCANNL_BACKEND multidev_cc (run %{dep:probe.exe})))))|dune},
      [ "rule: clean" ] );
    (* Per stanza, not per run: a pin around a helper fixes the backend the stanza holds too. *)
    ( "a pin around one run of a progn is the stanza's",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (progn (run %{dep:probe.exe}) (setenv OCANNL_BACKEND hip (run %{dep:helper.exe})))))|dune},
      [ "rule: unnamed hip (marker: cc)" ] );
    ( "a deprecated alias is no name a marker can carry",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (run %{dep:probe.exe} --ocannl_backend=sync_cc)))|dune},
      [ "rule: off the vocabulary: sync_cc" ] );
    ( "nor is `none`",
      {dune|(rule
 ; ocannl-backend: none -- claims no backend
 (action (setenv OCANNL_BACKEND none (run %{dep:probe.exe}))))|dune},
      [ "rule: off the vocabulary: none" ] );
    (* The unread forms: each is a position where a backend could be fixed, in a spelling this
       reader does not resolve. *)
    ( "a backend flag handed to a PATH tool",
      {dune|(rule
 ; ocannl-backend: none -- hands a script to python3
 (action (run python3 %{dep:driver.py} --ocannl_backend=cuda)))|dune},
      [ "rule: unread --ocannl_backend=cuda" ] );
    ( "an env launcher's assignment",
      {dune|(rule
 ; ocannl-backend: cuda -- launched under cuda
 (action (run env OCANNL_BACKEND=cuda %{dep:probe.exe})))|dune},
      [ "rule: unread OCANNL_BACKEND=cuda" ] );
    ( "a setenv value read from a pform",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (setenv OCANNL_BACKEND %{env:B=cc} (run %{dep:probe.exe}))))|dune},
      [ "rule: unread (setenv OCANNL_BACKEND %{env:B=cc} …)" ] );
    ( "an emptied variable, which hands the run back to the configuration",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (setenv OCANNL_BACKEND "" (run %{dep:probe.exe}))))|dune},
      [ "rule: unread (setenv OCANNL_BACKEND \"\" …)" ] );
    ( "a command-line value read from a pform",
      {dune|(rule
 ; ocannl-backend: cc -- says cc
 (action (run %{dep:probe.exe} --ocannl_backend=%{read:b.txt})))|dune},
      [ "rule: unread --ocannl_backend=%{read:b.txt}" ] );
    ( "a shell line naming the key",
      {dune|(rule
 ; ocannl-backend: cuda -- launched under cuda
 (action (bash "OCANNL_BACKEND=cuda ./probe.exe")))|dune},
      [ "rule: unread OCANNL_BACKEND=cuda ./probe.exe" ] );
    ( "a shell line carrying a backend flag",
      {dune|(rule
 ; ocannl-backend: cuda -- launched under cuda
 (action (system "./probe.exe --backend=cuda")))|dune},
      [ "rule: unread ./probe.exe --backend=cuda" ] );
    ( "an inline-test runner's flag",
      {dune|(library (name l)
 ; ocannl-backend: cc -- says cc
 (inline_tests (deps ocannl_config) (flags --ocannl_backend=cc)))|dune},
      [ "library: unread --ocannl_backend=cc" ] );
    (* A stanza the XOR already refuses is not also told its pin is unnamed... *)
    ( "a pin on a stanza declaring neither is the XOR's to report",
      {dune|(rule (action (setenv OCANNL_BACKEND hip (run %{dep:probe.exe}))))|dune},
      [ "rule: clean" ] );
    (* ...but an unread form is reported whatever the XOR says. *)
    ( "an unread form on such a stanza is still reported",
      {dune|(rule (action (run env OCANNL_BACKEND=hip %{dep:probe.exe})))|dune},
      [ "rule: unread OCANNL_BACKEND=hip" ] );
    (* The nearest honest neighbours: mentions of the backend that fix nothing. *)
    ( "a declaration, an enabled_if and a path read from the configuration pin nothing",
      {dune|(rule (deps ocannl_config (env_var OCANNL_BACKEND))
 (enabled_if (<> %{env:OCANNL_BACKEND=cc} metal))
 (action
  (progn
   (run %{dep:probe.exe})
   (run %{dep:minised.exe} "log/%{read:../config/ocannl_backend.txt}-0-0.log" %{target}))))|dune},
      [ "rule: clean" ] );
    ( "another variable set around the run pins nothing",
      {dune|(rule (deps (env_var OCANNL_BACKEND))
 (action (setenv OCANNL_TOOL_PROBE 1 (run %{dep:probe.exe} --ocannl_log_level=1))))|dune},
      [ "rule: clean" ] );
  ]

(* The directory-wide binding, refused outright: the [(subdir …)] scope of each `env` stanza binding
   the backend in `env-vars`, "." for the dune file's own. *)
let backend_env_vars_cases =
  [
    ( "an env stanza binding the backend",
      {dune|(env (_ (env-vars (OCANNL_BACKEND cuda))))|dune},
      [ "." ] );
    ( "and one inside a subdir, in any case",
      {dune|(subdir gen (env (dev (env-vars (FOO 1) (ocannl_backend hip)))))|dune},
      [ "gen" ] );
    ( "an env stanza binding something else, or setting the backend's NAME as a value",
      {dune|(env (_ (env-vars (FOO OCANNL_BACKEND)) (flags (:standard -w +50))))|dune},
      [] );
  ]

(* gh-ocannl-1501: [Scan.program_runners] alone, put to dune files this repository does not contain.
   It is the identity rule the ambient gate, the cc search's width pin and the refused-action scan
   all match runners with, and its cases went with the artifact rule that once carried them
   (gh-ocannl-1425); the family and search controls in `env_var_deps --control` still reach it, but
   only through verdicts that a different identity rule could also produce.

   Each case names the subdirectory the executables sit in and renders every program the stanzas
   there declare as "<name> <- <runners>", each runner by the alias it sits on and the variables
   pinned around every run of THAT program, or "<name> unrun". Runners are taken from the whole file
   with their own subdirectories, as the callers pass them. *)
let render_runners ~program_subdir source =
  let placed = Scan.walk "" (Scan.stanzas source) ~f:(fun subdir stanza -> [ (subdir, stanza) ]) in
  let group =
    List.filter_map placed ~f:(fun (subdir, stanza) ->
        Option.some_if (String.equal subdir program_subdir) stanza)
  in
  List.concat_map group ~f:(fun stanza ->
      match Scan.head stanza with
      | Some ("executable" | "executables") ->
          Scan.program_runners ~subdir:program_subdir ~runner_stanzas:placed group stanza
          |> List.map ~f:(fun (name, runners) ->
              match runners with
              | [] -> name ^ " unrun"
              | runners ->
                  name ^ " <- "
                  ^ String.concat ~sep:", "
                      (List.map runners ~f:(fun (runner, pinned) ->
                           String.concat ~sep:"+" (Scan.aliases_of runner)
                           ^
                           if Set.is_empty pinned then ""
                           else " [" ^ String.concat ~sep:" " (Set.to_list pinned) ^ "]")))
      | _ -> [])

let program_runners_cases =
  [
    (* The baseline: a dependency expansion names the stanza's own executable. *)
    ( "a dependency expansion runs the local executable",
      "",
      {dune|(executable (name probe) (modules probe))
(rule (alias a) (action (run %{dep:probe.exe})))|dune},
      [ "probe <- a" ] );
    (* The path AS WRITTEN is the identity: a basename rule made `../support/probe.exe` the runner
       of a local `probe` (Codex P2, round 2 of PR #457). *)
    ( "a same-named executable elsewhere is not the local one",
      "",
      {dune|(executable (name probe) (modules probe))
(rule (alias a) (action (run ../support/probe.exe)))|dune},
      [ "probe unrun" ] );
    (* A public name is a name, resolved from anywhere (gh-ocannl-783)... *)
    ( "a runner naming the public name is a runner",
      "",
      {dune|(executable (name probe) (public_name pkg.probe) (modules probe))
(rule (alias a) (action (run %{bin:pkg.probe})))|dune},
      [ "probe <- a" ] );
    (* ...and a path spelled like it is a path (Codex P2, round 10 of PR #457). *)
    ( "a local path spelled like the public name is not",
      "",
      {dune|(executable (name probe) (public_name pkg.probe) (modules probe))
(rule (alias a) (action (run ./pkg.probe)))|dune},
      [ "probe unrun" ] );
    (* gh-ocannl-747: one program per name, and `(public_names …)` pairs positionally, `-` being the
       placeholder for a name that is not installed. *)
    ( "each name of an executables stanza has its own runners",
      "",
      {dune|(executables (names a b) (public_names - pkg.b) (modules a b))
(rule (alias ra) (action (run %{dep:a.exe})))
(rule (alias rb) (action (run %{bin:pkg.b})))|dune},
      [ "a <- ra"; "b <- rb" ] );
    ( "one rule running both names is a runner of each",
      "",
      {dune|(executables (names a b) (modules a b))
(rule (alias both) (action (progn (run %{dep:a.exe}) (run %{dep:b.exe}))))|dune},
      [ "a <- both"; "b <- both" ] );
    (* gh-ocannl-982: a dependency expansion resolves from the stanza, whatever `chdir` says, in
       each of its spellings. *)
    ( "a dependency expansion under chdir keeps the stanza's identity",
      "",
      {dune|(executable (name probe) (modules probe))
(rule (alias dep) (action (chdir nested (run %{dep:probe.exe}))))
(rule (deps (:runner probe.exe)) (alias named) (action (chdir nested (run %{runner}))))
(rule (alias exe) (action (chdir nested (run %{exe:./probe.exe}))))|dune},
      [ "probe <- dep, named, exe" ] );
    (* A literal path follows `chdir`: this rule runs `a`'s program, not `b`'s (Codex P2, round 4 of
       PR #484). *)
    ( "a literal path under chdir names the program it moves to",
      "b",
      {dune|(subdir b (executable (name probe) (modules probe))
 (rule (alias moved) (action (chdir ../a (run probe.exe)))))|dune},
      [ "probe unrun" ] );
    (* Across `(subdir …)`: a rule outside the wrapper runs the nested executable under the
       qualified path, and a rule in a sibling subdirectory runs a different program (Codex P2,
       rounds 3-4 of PR #484). *)
    ( "a top-level rule runs a nested executable by its qualified path",
      "gen",
      {dune|(subdir gen (executable (name probe) (modules probe)))
(rule (alias literal) (action (run gen/probe.exe)))
(rule (alias dep) (action (chdir elsewhere (run %{dep:gen/../gen/probe.exe}))))|dune},
      [ "probe <- literal, dep" ] );
    ( "a nested dependency expansion under chdir keeps its subdirectory",
      "gen",
      {dune|(subdir gen (executable (name probe) (modules probe))
 (rule (alias inner) (action (chdir ../other (run %{dep:probe.exe})))))|dune},
      [ "probe <- inner" ] );
    ( "a rule in a sibling subdirectory is not this program's runner",
      "a",
      {dune|(subdir a (executable (name probe) (modules probe)))
(subdir b (rule (alias sibling) (action (run probe.exe))))|dune},
      [ "probe unrun" ] );
    (* A shell line names no program the identity rule can match: it is a site of its own kind
       (`Unknown_directory`), and no runner of anything here. *)
    ( "a shell line is no named runner",
      "",
      {dune|(executable (name probe) (modules probe))
(rule (alias sh) (action (bash "./probe.exe")))|dune},
      [ "probe unrun" ] );
    (* The pins: a variable counts where it is in force at EVERY run of the program, so an unpinned
       second run takes it back (gh-ocannl-749, Codex P1 round 1), a pin around a helper says
       nothing of the subject (Codex P2, round 4), and a value reading the variable back pins
       nothing (gh-ocannl-1320). *)
    ( "the variables pinned around every run of the program",
      "",
      {dune|(executables (names probe helper) (modules probe helper))
(rule (alias pinned) (action (setenv FOO 1 (setenv BAR 2 (run %{dep:probe.exe})))))
(rule (alias half) (action (progn (setenv FOO 1 (run %{dep:probe.exe})) (run %{dep:probe.exe}))))
(rule (alias helper) (action (progn (setenv FOO 1 (run %{dep:helper.exe})) (run %{dep:probe.exe}))))
(rule (alias echo) (action (setenv FOO %{env:FOO=1} (run %{dep:probe.exe}))))|dune},
      [ "probe <- pinned [BAR FOO], half, helper, echo"; "helper <- helper [FOO]" ] );
  ]

(* The dumb reading against the placed one, in the contract's own three counts: the whole text, the
   comments the lexer finds, and the comments a stanza's parentheses contain. A marker written where
   the comment lexer does not look -- into a quoted argument, into a field -- shows up as the first
   gap, and one written in a comment no stanza encloses as the second. Both are the same shape, "the
   author declared something no stanza carries", which is why the contract answers the placed count
   itself instead of leaving each consumer to subtract its way to it (gh-ocannl-863). *)
let sentinel_counting_cases =
  [
    ( "in a comment",
      {dune|(test (name t)
 ; ocannl-backend: none -- links no backend
 (deps ocannl_config))|dune},
      (1, 1, 1) );
    ( "inside a quoted argument, where it declares nothing",
      {dune|(rule (deps ocannl_config) (action (echo "ocannl-backend: none -- links no backend")))|dune},
      (1, 0, 0) );
    (* The count that separates the placed answer from the comment answer: a lexed comment that no
       stanza's parentheses contain. Subtracting one from the other would give 1 here. *)
    ( "in a comment above every stanza, which encloses it in none",
      {dune|; ocannl-backend: none -- sits between stanzas and declares nothing
(test (name t) (deps ocannl_config))|dune},
      (1, 1, 0) );
    (* A malformed marker was still PLACED: what is wrong with it is its own issue, not its
       position, so the placed count keeps it. *)
    ( "a malformed marker inside a stanza is still placed",
      {dune|(test (name t)
 ; ocannl-backend: none
 (deps ocannl_config))|dune},
      (1, 1, 1) );
    ("none at all", {dune|(test (name t) (deps ocannl_config))|dune}, (0, 0, 0));
  ]

(* gh-ocannl-863: every Dune comment convention inherits this outer grammar and placement contract.
   The synthetic sentinel keeps these controls independent of either current consumer; the callback
   below owns only the declaration and which stanza may carry it. *)
let contained_marker_sentinel = "ocannl-test-marker:"
let contained_marker_grammar = contained_marker_sentinel ^ " <value> -- <reason>"

let contained_marker_contract source =
  Scan.contained_marker_contract source ~sentinel:contained_marker_sentinel
    ~parse_declaration:(fun _stanza ~declaration ~reason ->
      if String.equal declaration "valid" then Ok (declaration ^ "|" ^ reason)
      else Error [ Printf.sprintf "the value `%s` is not valid" declaration ])
    ~belongs:(fun stanza _marker ->
      if String.equal stanza.Scan.marked_head "rule" then Ok ()
      else Error [ "the marker belongs on a rule" ])

let render_contained_issue = function
  | Scan.Malformed_marker { issue_line; issue_malformed; _ } ->
      Printf.sprintf "malformed at line %d: %s" issue_line
        (Scan.marker_malformed_reason ~sentinel:contained_marker_sentinel ~separator_subject:"value"
           ~grammar:contained_marker_grammar issue_malformed)
  | Scan.Marker_outside_stanza { issue_line; _ } ->
      Printf.sprintf "outside every stanza at line %d" issue_line
  | Scan.Marker_in_wrong_stanza { issue_line; issue_stanza; issue_why; _ } ->
      Printf.sprintf "wrong %s stanza at line %d: %s" issue_stanza.Scan.marked_head issue_line
        issue_why
  | Scan.Marker_outside_comment { issue_text_occurrences; issue_comment_occurrences } ->
      Printf.sprintf "%d sentinel(s) in text but %d in comments" issue_text_occurrences
        issue_comment_occurrences

let render_contained_marker_contract source =
  let contract = contained_marker_contract source in
  let markers =
    List.concat_map contract.Scan.contract_stanzas ~f:(fun marked ->
        List.map marked.Scan.stanza_markers ~f:(fun marker ->
            Printf.sprintf "marker at line %d in %s: %s" marker.Scan.contained_line
              marked.Scan.marker_stanza.Scan.marked_head marker.Scan.contained_value))
  in
  markers @ List.map contract.Scan.contract_issues ~f:render_contained_issue

let contained_marker_cases =
  [
    ( "a valid contained marker",
      {dune|(rule
 ; ocannl-test-marker: valid -- explains the fixture
 (action (echo ok)))|dune},
      [ "marker at line 2 in rule: valid|explains the fixture" ] );
    ( "annotated prose before the sentinel remains a marker",
      {dune|(rule
 ; NOTE ocannl-test-marker: valid -- explains the fixture
 (action (echo ok)))|dune},
      [ "marker at line 2 in rule: valid|explains the fixture" ] );
    ( "the earliest admitted separator wins",
      {dune|(rule
 ; ocannl-test-marker: valid — explains -- the fixture
 (action (echo ok)))|dune},
      [ "marker at line 2 in rule: valid|explains -- the fixture" ] );
    ( "the shared separator grammar refuses a malformed marker",
      {dune|(rule
 ; ocannl-test-marker: valid explains the fixture
 (action (echo ok)))|dune},
      [
        "malformed at line 2: no `--` separating the value from the reason -- the grammar is `; \
         ocannl-test-marker: <value> -- <reason>`";
      ] );
    ( "the shared reason grammar refuses a one-word reason",
      {dune|(rule
 ; ocannl-test-marker: valid -- fixture
 (action (echo ok)))|dune},
      [ "malformed at line 2: the reason `fixture` is one word -- say why, not what" ] );
    ( "two declarations in one comment are refused before either is parsed",
      {dune|(rule
 ; ocannl-test-marker: valid -- explains the fixture ocannl-test-marker: valid -- repeats it here
 (action (echo ok)))|dune},
      [
        "malformed at line 2: two `ocannl-test-marker:` declarations in one comment -- the second \
         would be read as part of the first's reason; put each on its own line";
      ] );
    ( "a marker outside every stanza",
      {dune|; ocannl-test-marker: valid -- explains the fixture
(rule (action (echo ok)))|dune},
      [ "outside every stanza at line 1" ] );
    ( "a marker inside a convention-specific wrong stanza",
      {dune|(library
 ; ocannl-test-marker: valid -- explains the fixture
 (name helper))|dune},
      [ "wrong library stanza at line 2: the marker belongs on a rule" ] );
    ( "a sentinel outside a comment",
      {dune|(rule (action (echo "ocannl-test-marker: valid -- explains the fixture")))|dune},
      [ "1 sentinel(s) in text but 0 in comments" ] );
  ]

(* gh-ocannl-603: the resolution half of config_dep_completeness -- which config each site's process
   reads, whether its dependencies name that one, whether the ambient environment reaches the run,
   and which exemptions were spent -- driven on synthetic trees with known verdicts. The cases above
   pin how a stanza is READ; these pin what the check CONCLUDES from what was read, which used to be
   tested only by a golden over the live repository: a golden that can say "today's tree is
   consistent" and never "a tree where X is wrong gets rejected". Every case states the shape one
   review round fixed by hand, so the fix stays fixed.

   A tree is a list of repository-relative files: a [dune] file's text is read, an [ocannl_config]
   only exists. Each rendered line is a site outside [test/config] (the ambient producer every tree
   but one carries), as "<exemption key> (reads <the config dir its process finds>): <verdict>",
   then the exemptions no site spent. *)
let ambient_producer =
  [
    ("test/config/ocannl_config", "");
    ( "test/config/dune",
      {dune|(rule (target ocannl_host_capabilities) (deps (universe)) (action (run ./host.exe)))
(rule (target ocannl_ambient) (deps ocannl_config (universe))
 (action (run %{exe:ocannl_ambient_env.exe} --host-record=%{dep:ocannl_host_capabilities})))
(subdir ambient (rule (target ocannl_config) (deps ../ocannl_config ../ocannl_ambient)
 (action (with-stdout-to %{target} (progn (cat ../ocannl_config) (cat ../ocannl_ambient))))))|dune}
    );
  ]

(* A directory whose config is the copy that carries the ambient record, as the repository's test
   directories have it. *)
let configured dir extra =
  let up = String.concat (List.map (List.tl_exn (String.split dir ~on:'/')) ~f:(fun _ -> "../")) in
  (dir ^ "/dune", "(copy_files " ^ up ^ "config/ambient/ocannl_config)\n" ^ extra)

let render_resolution_verdict = function
  | Test_utils.Config_dep_resolution.Counted -> "counted"
  | Exempt -> "exempt"
  | No_config_in_reach -> "no config in reach"
  | Unestablished_directory -> "directory cannot be established"
  | Ambient_unreached dir -> "ambient environment unreached through " ^ dir
  | Unreadable_undeclared -> "unreadable command, undeclared"
  | Unclassified_undeclared -> "unclassified action, undeclared"
  | Undeclared -> "undeclared"

let render_resolution ~exemptions tree =
  let module R = Test_utils.Config_dep_resolution in
  let is name (path, _) = String.equal (Stdlib.Filename.basename path) name in
  let report =
    R.analyze ~exemption_keys:exemptions
      ~config_files:(List.filter tree ~f:(is Scan.config_file) |> List.map ~f:fst)
      (List.filter tree ~f:(is "dune"))
  in
  (match report.R.ambient_producer with
    | R.Producer_ok -> []
    | R.Environment_unrecorded -> [ "the ambient producer is missing" ]
    | R.Host_unrecorded -> [ "the host record is missing" ])
  @ List.concat_map report.R.files ~f:(fun file ->
      if String.equal file.R.dune_file (R.ambient_home ^ "/dune") then []
      else
        List.map file.R.sites ~f:(fun r ->
            Printf.sprintf "%s (reads %s): %s%s" r.R.key
              (match r.R.nearest with None -> "nothing" | Some "" -> "." | Some dir -> dir)
              (render_resolution_verdict r.R.verdict)
              (if r.R.needs_isolating_sandbox then " [needs an isolating sandbox]" else "")))
  @ List.map report.R.stale_exemptions ~f:(fun key -> "stale exemption: " ^ key)

let resolution_cases =
  [
    ( "a site reading its own directory's config",
      [ configured "test/a" "(rule (alias runtest-x) (deps ocannl_config) (action (run ./x.exe)))" ],
      [],
      [ "test/a:x.exe (reads test/a): counted" ] );
    (* Codex P2, rounds 9 to 11 of PR #343: the config a process reads is the one where it RUNS. *)
    ( "a sibling's config, reached by a chdir, is the one to name",
      [
        configured "test/a"
          {dune|(rule (alias runtest-x) (deps ../b/ocannl_config) (action (chdir ../b (run ./x.exe))))
(rule (alias runtest-y) (deps ocannl_config) (action (chdir ../b (run ./y.exe))))|dune};
        configured "test/b" "";
      ],
      [],
      [ "test/b:x.exe (reads test/b): counted"; "test/b:y.exe (reads test/b): undeclared" ] );
    ( "a descendant's config, nearer than the stanza's own, is the one to name",
      [
        configured "test/a"
          {dune|(rule (alias runtest-x) (deps sub/ocannl_config) (action (chdir sub (run ./x.exe))))
(rule (alias runtest-y) (deps ocannl_config) (action (chdir sub (run ./y.exe))))|dune};
        configured "test/a/sub" "";
      ],
      [],
      [
        "test/a/sub:x.exe (reads test/a/sub): counted";
        "test/a/sub:y.exe (reads test/a/sub): undeclared";
      ] );
    (* An ancestor's config is the right one exactly while no nearer directory has its own. *)
    ( "an ancestor's config serves where no nearer one exists",
      [
        configured "test/a"
          {dune|(rule (alias runtest-x) (deps ocannl_config) (action (chdir deep (run ./x.exe))))|dune};
      ],
      [],
      [ "test/a/deep:x.exe (reads test/a): counted" ] );
    (* Codex P2, round 10 of PR #343: declaring SOME ancestor's config is not declaring the
       nearest. *)
    ( "an ancestor's config is not the one read once a nearer one exists",
      [
        ("test/ocannl_config", "");
        configured "test/a"
          "(rule (alias runtest-x) (deps ../ocannl_config) (action (run ./x.exe)))";
      ],
      [],
      [ "test/a:x.exe (reads test/a): undeclared" ] );
    (* Codex P2, round 14 of PR #343: the config may be materialized by a THIRD dune file. *)
    ( "a config another dune file copies in counts",
      [
        ("test/dune", "(subdir q (copy_files ../config/ambient/ocannl_config))");
        ( "test/p/dune",
          {dune|(rule (alias runtest-x) (deps ../q/ocannl_config) (action (chdir ../q (run ./x.exe))))|dune}
        );
      ],
      [],
      [ "test/q:x.exe (reads test/q): counted" ] );
    (* gh-ocannl-982's fixture guidance: the program a `%{dep:...}` names stays relative to its
       stanza while a literal one resolves under the action's cwd, and neither moves where the
       PROCESS runs -- both read the chdir's config. *)
    ( "a dependency-expanded program and a literal runner under one chdir read one config",
      [
        configured "test/a"
          {dune|(rule (alias runtest-x) (deps ../b/ocannl_config)
 (action (chdir ../b (progn (run %{dep:x.exe}) (run ./y.exe)))))
(rule (alias runtest-z) (deps ocannl_config)
 (action (chdir ../b (progn (run %{dep:x.exe}) (run ./y.exe)))))|dune};
        configured "test/b" "";
      ],
      [],
      [
        "test/b:y.exe, x.exe (reads test/b): counted";
        "test/b:y.exe, x.exe (reads test/b): undeclared";
      ] );
    (* lukstafi/ocannl-staging#346: a root `ocannl_config` is gitignored personal settings, so a
       dependency resolving to it satisfies nothing -- the developer's machine would pass what CI
       fails. Restoring the root as a stop on the walk turns these lines "counted". *)
    ( "the repository root's config is never in reach",
      [
        ("ocannl_config", "");
        ( "test/r/dune",
          {dune|(rule (alias runtest-x) (deps ../../ocannl_config) (action (run ./x.exe)))
(test (name t) (deps ../../ocannl_config))|dune}
        );
        ("dune", "(test (name root_level) (deps ocannl_config))");
      ],
      [],
      [
        ":root_level (reads nothing): no config in reach";
        "test/r:x.exe (reads nothing): no config in reach";
        "test/r:t (reads nothing): no config in reach";
      ] );
    (* gh-ocannl-1149: a configuration checked in beside its tests carries no record of the ambient
       environment, so the record itself, or `(universe)`, has to be depended on. *)
    ( "a checked-in config needs the ambient record beside it",
      [
        ("test/k/ocannl_config", "");
        ( "test/k/dune",
          {dune|(copy_files ../config/ocannl_ambient)
(rule (alias runtest-x) (deps ocannl_config) (action (run ./x.exe)))
(rule (alias runtest-y) (deps ocannl_config ocannl_ambient) (action (run ./y.exe)))
(rule (alias runtest-z) (deps ocannl_config (universe)) (action (run ./z.exe)))
(rule (alias runtest-w) (deps ocannl_config ../elsewhere/ocannl_ambient) (action (run ./w.exe)))|dune}
        );
      ],
      [],
      [
        "test/k:x.exe (reads test/k): ambient environment unreached through test/k";
        "test/k:y.exe (reads test/k): counted";
        "test/k:z.exe (reads test/k): counted";
        "test/k:w.exe (reads test/k): ambient environment unreached through test/k";
      ] );
    ( "the shared configuration's own producer is checked",
      [ configured "test/a" "(test (name t) (deps ocannl_config))" ],
      [],
      [ "test/a:t (reads test/a): counted" ] );
    (* An exemption is spent only where the dep is absent, and only by the site it names: the
       spending is what keeps the list from growing quietly (round 11 of PR #343 put availability
       after it, so an exempt site in a directory with no config at all is still exempt). *)
    ( "an exemption spent, one gone stale by declaring the dep, one naming nothing",
      [
        configured "test/e"
          {dune|(rule (alias runtest-x) (action (run ./x.exe)))
(rule (alias runtest-y) (deps ocannl_config) (action (run ./y.exe)))|dune};
        ("test/bare/dune", "(rule (alias runtest-v) (action (run ./v.exe)))");
      ],
      [ "test/e:x.exe"; "test/e:y.exe"; "test/gone:z.exe"; "test/bare:v.exe" ],
      [
        "test/bare:v.exe (reads nothing): exempt";
        "test/e:x.exe (reads test/e): exempt";
        "test/e:y.exe (reads test/e): counted";
        "stale exemption: test/e:y.exe";
        "stale exemption: test/gone:z.exe";
      ] );
    (* lukstafi/ocannl-staging#981's canary controls, as this check keys them: by the directory a
       process runs in and what it runs, never by alias -- so a configuration reader on the canary's
       alias inherits nothing, and a canary that moves leaves its entry stale. *)
    ( "a reader on an exempt canary's alias is not exempt",
      [
        configured "test/e"
          {dune|(rule (alias bin-smoke) (action (run ./probe.exe)))
(rule (alias bin-smoke) (action (run ./reader.exe)))|dune};
      ],
      [ "test/e:probe.exe" ],
      [ "test/e:probe.exe (reads test/e): exempt"; "test/e:reader.exe (reads test/e): undeclared" ]
    );
    ( "a canary that moves directory leaves its exemption stale",
      [ configured "test/moved" "(rule (alias bin-smoke) (action (run ./probe.exe)))" ],
      [ "test/e:probe.exe" ],
      [ "test/moved:probe.exe (reads test/moved): undeclared"; "stale exemption: test/e:probe.exe" ]
    );
    (* Codex P2, round 8 of PR #343, and gh-ocannl-1359: a shell line's directory is unknown, so a
       declared dep settles nothing, and an exemption is the only way through -- which still owes an
       isolating sandbox. *)
    ( "a shell action needs an exemption and an isolating sandbox",
      [
        configured "test/s"
          {dune|(rule (alias runtest-x) (deps ocannl_config) (action (system "./x.exe")))
(rule (alias runtest-y) (action (system "./y.exe")))
(rule (alias runtest-z) (deps (sandbox always preserve_file_kind)) (action (system "./z.exe")))|dune};
      ],
      [ "test/s:shell: ./y.exe"; "test/s:shell: ./z.exe" ],
      [
        "test/s:shell: ./x.exe (reads test/s): directory cannot be established [needs an isolating \
         sandbox]";
        "test/s:shell: ./y.exe (reads test/s): exempt [needs an isolating sandbox]";
        "test/s:shell: ./z.exe (reads test/s): exempt";
      ] );
    ( "what the scan cannot read is accepted only with the dep",
      [
        configured "test/u"
          {dune|(rule (alias runtest-x) (action (run %{env:TOOL=x})))
(rule (alias runtest-y) (deps ocannl_config) (action (run %{env:TOOL=y})))
(rule (alias runtest-z) (action (frobnicate ./z.exe)))
(rule (alias runtest-w) (deps ocannl_config) (action (frobnicate ./w.exe)))|dune};
      ],
      [],
      [
        "test/u:%{env:TOOL=x} (reads test/u): unreadable command, undeclared";
        "test/u:%{env:TOOL=y} (reads test/u): counted";
        "test/u:frobnicate (reads test/u): unclassified action, undeclared";
        "test/u:frobnicate (reads test/u): counted";
      ] );
    ( "inline tests and tests are held to the same dep",
      [
        configured "test/i"
          {dune|(library (name l) (inline_tests (deps ocannl_config)))
(library (name m) (inline_tests))
(test (name t) (deps ocannl_config))
(test (name u))|dune};
      ],
      [],
      [
        "test/i:l (reads test/i): counted";
        "test/i:m (reads test/i): undeclared";
        "test/i:t (reads test/i): counted";
        "test/i:u (reads test/i): undeclared";
      ] );
  ]

(* gh-ocannl-1038: a producer recording the environment but not the host, run without
   [ambient_producer] like the case below. *)
let hostless_resolution_case =
  ( "a producer that records the environment but not the host is reported",
    [
      ("test/config/ocannl_config", "");
      ( "test/config/dune",
        {dune|(rule (target ocannl_ambient) (deps ocannl_config (universe)) (action (run ./env.exe)))
(subdir ambient (rule (target ocannl_config) (deps ../ocannl_config ../ocannl_ambient)
 (action (with-stdout-to %{target} (progn (cat ../ocannl_config) (cat ../ocannl_ambient))))))|dune}
      );
    ],
    [],
    [ "the host record is missing" ] )

(* The one tree without [ambient_producer], which every case above is run with. *)
let no_producer_case =
  ( "a tree with no ambient producer is reported",
    [ ("test/a/dune", "(copy_files ../config/ambient/ocannl_config)") ],
    [],
    [ "the ambient producer is missing" ] )

(* gh-ocannl-1207: config_dep_completeness's own refusals, executed rather than vouched for by the
   cases above, which exercise the reading they rest on in process. The scanner runs as a child over
   a tree built to trip them -- a stanza head and an action head it cannot classify, a binaries
   mapping, a shell action, a command it cannot read, a test without the config dep, a test in a
   directory no config reaches, a rule reading a checked-in config with no record of the ambient
   environment (gh-ocannl-1149), and a Python script run under the symlink-permitting sandbox
   (gh-ocannl-1359) -- and once over no dune file at all. Every exemption is stale in such a tree
   too, and the tree has no test/config/dune producing that record. What each child refused is what
   `raw_direct_evidence` attributes to the two claims. *)
let refusal_tree =
  [
    ("t/ocannl_config", "");
    ( "t/dune",
      {dune|(frobnicate (name x))
(env (_ (binaries (tool.exe as tool))))
(rule (alias runtest-b) (deps ocannl_config) (action (system "./b.exe")))
(rule (alias runtest-c) (action (run %{env:TOOL=x})))
(rule (alias runtest-d) (action (frobnicate ./d.exe)))
(test (name e) (modules e))
(rule (alias runtest-f) (deps ocannl_config) (action (run ./f.exe)))
(rule (alias runtest-g) (deps ocannl_config (sandbox always)) (action (run python3 %{dep:g.py})))
|dune}
    );
    ("unconfigured/dune", "(test (name a) (modules a))\n");
  ]

(* gh-ocannl-1038: a test/config/dune that records the ambient environment -- both producers the
   check above asks for, intact -- and not the host. The refusal it must draw is the host record's
   own, which the environment's producers passing leaves as the only producer refusal possible. *)
let hostless_tree =
  [
    ("test/config/ocannl_config", "");
    ( "test/config/dune",
      {dune|(rule (target ocannl_ambient) (deps ocannl_config (universe)) (action (run ./env.exe)))
(subdir ambient
 (rule (target ocannl_config) (deps ../ocannl_config ../ocannl_ambient)
  (action (with-stdout-to %{target} (progn (cat ../ocannl_config) (cat ../ocannl_ambient))))))
|dune}
    );
  ]

(* The same producers with a host rule beside them that the record's producer DEPENDS on and never
   reads: the dependency reruns the action, which writes the same bytes, so dune's early cutoff
   still serves every test's cached verdict. *)
let host_unread_tree =
  [
    ("test/config/ocannl_config", "");
    ( "test/config/dune",
      {dune|(rule (target ocannl_host_capabilities) (deps (universe)) (action (run ./host.exe)))
(rule (target ocannl_ambient) (deps ocannl_config (universe) ocannl_host_capabilities)
 (action (run ./env.exe)))
(subdir ambient
 (rule (target ocannl_config) (deps ../ocannl_config ../ocannl_ambient)
  (action (with-stdout-to %{target} (progn (cat ../ocannl_config) (cat ../ocannl_ambient))))))
|dune}
    );
  ]

let scanner_refusal_controls exe =
  let source = "test/operations/config_dep_completeness.ml" in
  let refuses = Test_utils.Refusal_control_manifest.refuses ~source ~exe in
  Test_utils.Refusal_control_manifest.with_tree refusal_tree (fun root ->
      Verdict.p "config_dep_completeness refuses a tree built to trip its refusals, exiting 1"
        (refuses
           ~claim:"config_dep_completeness refuses a tree built to trip its refusals, exiting 1"
           (root :: List.map refusal_tree ~f:(fun (path, _) -> Stdlib.Filename.concat root path)));
      Verdict.p "config_dep_completeness refuses a run handed no dune file, exiting 1"
        (refuses ~claim:"config_dep_completeness refuses a run handed no dune file, exiting 1"
           [ root ]));
  (* Run from inside the tree, handed `.` and relative paths: the scanner places a dune file
     relative to the workspace root it is given, and the producer check asks for `test/config/dune`
     by that path -- an absolute temporary path is never it, which would leave the environment's
     refusal firing in place of the one this control is for. The scanner's own source goes into the
     tree too, where its closing manifest looks for it from a workspace root
     (`Refusal_control_manifest.print`); this directory has it as a dependency. *)
  let scanner_source = "config_dep_completeness.ml" in
  let here = Stdlib.Sys.getcwd () in
  let exe = if Stdlib.Filename.is_relative exe then Stdlib.Filename.concat here exe else exe in
  List.iter
    [
      ("an ambient record that leaves out the host", hostless_tree);
      ( "an ambient record whose producer depends on the host record and never reads it",
        host_unread_tree );
    ]
    ~f:(fun (what, files) ->
      let tree =
        ("test/operations/" ^ scanner_source, Stdio.In_channel.read_all scanner_source) :: files
      in
      Test_utils.Refusal_control_manifest.with_tree tree (fun root ->
          let claim = "config_dep_completeness refuses " ^ what ^ ", exiting 1" in
          let refused =
            Stdlib.Sys.chdir root;
            Exn.protect
              ~f:(fun () ->
                Test_utils.Refusal_control_manifest.refuses ~source ~claim ~exe
                  ("." :: List.map files ~f:fst))
              ~finally:(fun () -> Stdlib.Sys.chdir here)
          in
          Verdict.p claim refused))

let () =
  let check name expected found =
    if List.equal String.equal found expected then printf "ok: %s\n" name
    else
      fail "%s -- expected [%s], found [%s]" name
        (String.concat ~sep:"; " expected)
        (String.concat ~sep:"; " found)
  in
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try List.map (Scan.sites source) ~f:render
        with exn ->
          fail "%s -- the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check name expected found);
  List.iter raw_stanza_cases ~f:(fun (name, source, expected) ->
      check ("raw stanzas -- " ^ name) expected (List.map (Scan.raw_stanzas source) ~f:render_raw));
  List.iter path_rewriting_cases ~f:(fun (name, source, expected) ->
      check ("path-rewriting stanzas -- " ^ name) expected (Scan.path_rewriting_stanzas source));
  List.iter path_rewriting_scope_cases ~f:(fun (name, source, expected) ->
      check
        ("path-rewriting stanza scope -- " ^ name)
        expected
        (Scan.path_rewriting_stanza_scopes source));
  List.iter declared_paths_cases ~f:(fun (name, source, expected) ->
      let found =
        List.concat_map (Scan.sites source) ~f:(fun site -> site.Scan.declared_config_paths)
        |> List.dedup_and_sort ~compare:String.compare
      in
      check ("declared config paths -- " ^ name) expected found);
  List.iter copy_cases ~f:(fun (name, source, expected) ->
      let found =
        List.map (Scan.config_copy_dirs source) ~f:(fun dir ->
            if String.is_empty dir then "." else dir)
      in
      check ("copies the config -- " ^ name) expected found);
  List.iter unclassified_cases ~f:(fun (name, source, expected) ->
      check ("unclassified heads -- " ^ name) expected (Scan.unclassified_heads source));
  List.iter accepted_marker_cases ~f:(fun (name, source, expected) ->
      let found =
        try List.map (Scan.sites source) ~f:render
        with exn ->
          fail "accepted marker -- %s: the scan refused it: %s" name (Exn.to_string exn);
          []
      in
      check ("accepted marker -- " ^ name) expected found);
  List.iter marker_grammar_cases ~f:(fun (name, text, expected) ->
      check ("backend marker grammar -- " ^ name) [ expected ] [ render_marker text ]);
  List.iter marker_placement_cases ~f:(fun (name, source, expected) ->
      let found =
        try List.map (Scan.marked_stanzas source) ~f:render_marked
        with exn ->
          fail "backend marker placement -- %s: the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check ("backend marker placement -- " ^ name) expected found);
  List.iter backend_rule_cases ~f:(fun (name, source, expected) ->
      let found =
        try
          let contract = Scan.backend_marker_contract source in
          List.map contract.Scan.contract_stanzas ~f:(render_rule contract)
        with exn ->
          fail "backend rule -- %s: the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check ("backend rule -- " ^ name) expected found);
  List.iter backend_pin_cases ~f:(fun (name, source, expected) ->
      let found =
        try render_pins source
        with exn ->
          fail "backend pins -- %s: the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check ("backend pins -- " ^ name) expected found);
  List.iter backend_env_vars_cases ~f:(fun (name, source, expected) ->
      check ("backend env-vars -- " ^ name) expected
        (List.map (Scan.backend_env_vars_scopes source) ~f:(fun dir ->
             if String.is_empty dir then "." else dir)));
  List.iter program_runners_cases ~f:(fun (name, program_subdir, source, expected) ->
      let found =
        try render_runners ~program_subdir source
        with exn ->
          fail "program runners -- %s: the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check ("program runners -- " ^ name) expected found);
  List.iter sentinel_counting_cases ~f:(fun (name, source, (in_text, in_comments, in_stanzas)) ->
      let contract = Scan.backend_marker_contract source in
      let render text comments stanzas =
        Printf.sprintf "%d in the text, %d in comments, %d in stanzas" text comments stanzas
      in
      check
        ("backend marker sentinel -- " ^ name)
        [ render in_text in_comments in_stanzas ]
        [
          render contract.Scan.contract_text_occurrences contract.Scan.contract_comment_occurrences
            contract.Scan.contract_stanza_occurrences;
        ]);
  List.iter contained_marker_cases ~f:(fun (name, source, expected) ->
      let found =
        try render_contained_marker_contract source
        with exn ->
          fail "contained marker contract -- %s: the scan raised: %s" name (Exn.to_string exn);
          []
      in
      check ("contained marker contract -- " ^ name) expected found);
  printf "\nContained-marker refusal diagnostics exercised by the cases above:\n";
  List.iter contained_marker_cases ~f:(fun (name, source, _expected) ->
      let contract = contained_marker_contract source in
      List.map contract.Scan.contract_issues ~f:render_contained_issue
      |> List.iter ~f:(fun diagnostic -> printf "  %s -- %s\n" name diagnostic));
  printf "\n";
  List.iter (nested_marker_case :: refused_cases) ~f:(fun (name, source) ->
      match Scan.sites source with
      | exception _ -> printf "ok: refused -- %s\n" name
      | sites ->
          fail "refused -- %s: read the file as %d sites instead of refusing it" name
            (List.length sites));
  List.iter resolution_cases ~f:(fun (name, tree, exemptions, expected) ->
      check ("resolution -- " ^ name) expected
        (render_resolution ~exemptions (ambient_producer @ tree)));
  List.iter [ no_producer_case; hostless_resolution_case ]
    ~f:(fun (name, tree, exemptions, expected) ->
      check ("resolution -- " ^ name) expected (render_resolution ~exemptions tree));
  (match Stdlib.Sys.argv with
  | [| _; exe |] -> scanner_refusal_controls exe
  | _ -> Verdict.fail "usage: dune_scan_cases <config_dep_completeness.exe>");
  Test_utils.Refusal_control_manifest.print "config_dep_completeness.ml"
