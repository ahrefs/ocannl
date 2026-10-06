(** What {!Test_utils.Slot_kind} says a dune argv reaches, on a fixture tree built to separate the
    cases (gh-ocannl-1004, gh-ocannl-1066, gh-ocannl-1095): a stanza that NAMES a backend holds it
    whatever the configuration says, so tools/batch-backends.sh counts every backend a reachable
    marker names among the batch's -- for the fleet slot ([--cpu] only when none is a GPU) and for
    the width (the tightest any of them meets) -- and counts the configurations' backends only when
    a reachable stanza reads them. Every verdict below is phrased so that [true] is the passing
    reading, and the golden records the backends each argv was judged to reach, so a change of
    answer shows as a diff as well as a failed claim. *)

open Base
open Stdio
open Verdict.Claims
module Slot_kind = Test_utils.Slot_kind

(* A tree with a GPU stanza, a configuration-reading stanza and a CPU-named one side by side in [a];
   a hip stanza one level down; a metal rule in [b]; [c] with nothing but a configuration-reading
   test; and [n] with nothing but a stanza that links no backend, under its own [scans]. The
   aggregates in [a] reach one member each.

   Then one directory per way dune builds a stanza that no alias the argv names carries
   (gh-ocannl-1095; Codex GPT-6.1 Sol review on PR #1027) -- each a way a configuration reader or a
   named backend could run in a batch judged to hold neither: [f] reaches producing rules through
   files (bandwidth_calibration's executable -> generated [.actual] -> diff alias, an explicit file
   dependency, a [%{read:…}], a glob), [x] reaches [c]'s reader through an alias in another
   directory (plain and recursive), and [g] reaches a reader through the dependencies of a test's
   and an inline-test library's GENERATED per-stanza aliases. *)
let tree =
  [
    ( "a",
      {dune|
(test
 ; ocannl-backend: cuda -- names the CUDA backend by name in this fixture.
 (name t_cuda)
 (deps ocannl_config))

(test
 (name t_cfg)
 (deps ocannl_config (env_var OCANNL_BACKEND)))

(test
 ; ocannl-backend: cc -- names the cc backend by name in this fixture.
 (name t_cc)
 (deps ocannl_config))

(alias
 (name scans)
 (deps (alias runtest-t_cfg) (alias runtest-t_cc)))

(alias
 (name gpuagg)
 (deps (alias scans) (alias runtest-t_cuda)))
|dune}
    );
    ( "a/sub",
      {dune|
(test
 ; ocannl-backend: hip -- names the HIP backend by name in this fixture.
 (name t_hip)
 (deps ocannl_config))
|dune}
    );
    ( "b",
      {dune|
(executable (name m) (modules m))

(rule
 ; ocannl-backend: cc,metal -- runs the same probe on cc and on Metal by name.
 (alias runtest-m)
 (deps ocannl_config)
 (action (run %{exe:m.exe})))

(alias (name runtest) (deps (alias runtest-m)))
|dune}
    );
    ("c", {dune|
(test
 (name t_only)
 (deps ocannl_config (env_var OCANNL_BACKEND)))
|dune});
    ( "n",
      {dune|
(test
 ; ocannl-backend: none -- links no backend in this fixture.
 (name t_none)
 (deps ocannl_config))

(alias
 (name scans)
 (deps (alias runtest-t_none)))
|dune}
    );
    ( "f",
      {dune|
(executable (name probe) (modules probe))

(rule
 (target probe.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:probe.exe}))))

(rule
 ; ocannl-backend: cuda -- names the CUDA backend by name in this fixture.
 (target named.actual)
 (deps ocannl_config)
 (action (with-stdout-to %{target} (run %{dep:probe.exe}))))

(rule
 (alias runtest-probe)
 (action (diff probe.expected probe.actual)))

(rule
 (alias runtest-named)
 (deps named.actual)
 (action (progn)))

(rule
 (alias runtest-read)
 (action (echo "%{read:probe.actual}")))

(rule
 (alias runtest-glob)
 (deps (glob_files *.actual))
 (action (progn)))
|dune}
    );
    ( "x",
      {dune|
(alias
 (name cross)
 (deps (alias ../c/runtest-t_only)))

(alias
 (name cross_rec)
 (deps (alias_rec ../c/runtest)))
|dune}
    );
    ( "g",
      {dune|
(test
 ; ocannl-backend: none -- links no backend in this fixture.
 (name t_gate)
 (deps ocannl_config (alias runtest-local_reader)))

(test
 (name local_reader)
 (deps ocannl_config (env_var OCANNL_BACKEND)))

(test
 ; ocannl-backend: hip -- names the HIP backend by name in this fixture.
 (name t_hip_dep)
 (deps ocannl_config))

(library
 (name l)
 (modules l)
 (inline_tests
  ; ocannl-backend: none -- links no backend in this fixture.
  (deps ocannl_config (alias runtest-t_hip_dep))))
|dune}
    );
  ]

let judge ?(dune_files = tree) argv =
  let answer = Slot_kind.answer ~dune_files (String.split argv ~on:' ') in
  let shown =
    match answer with
    | Reaches { named; reads_config } ->
        (match named with
          | [] -> "names nothing"
          | _ -> "names " ^ String.concat ~sep:"," (List.map named ~f:fst))
        ^ if Option.is_some reads_config then " + reads config" else ""
    | Unknown why -> "unknown: " ^ why
  in
  (answer, shown)

let cases =
  [
    (* What a run reaches, alias by alias. *)
    ("build @a/runtest-t_cfg", `Cpu);
    ("build @a/runtest-t_cc", `Cpu);
    ("build @a/runtest-t_cuda", `Gpu);
    ("build @a/scans", `Cpu);
    ("build @a/gpuagg", `Gpu);
    (* A recursive per-test alias reaches only its own name below; a recursive suite reaches all. *)
    ("build @@c/runtest", `Cpu);
    ("build @c/runtest", `Cpu);
    ("build @@a/runtest", `Gpu);
    ("runtest a/sub", `Gpu);
    ("runtest a", `Gpu);
    ("runtest c", `Cpu);
    ("runtest n", `Cpu);
    ("build @n/scans", `Cpu);
    ("build @b/runtest", `Gpu);
    ("build @b/runtest-m", `Gpu);
    ("runtest", `Gpu);
    ("build @runtest", `Gpu);
    ("build @@runtest", `Cpu);
    (* Only a closed set of argv shapes is read; anything else is unmodelled and answered as a GPU
       (Codex review rounds 3-4 on PR #803): other spellings of a directory, words past `--`,
       alias-naming options, path targets, options that change the tree, and unknown ones. *)
    ("runtest .", `Gpu);
    ("runtest ./c", `Gpu);
    ("runtest c/", `Gpu);
    ("runtest c/../a", `Gpu);
    ("runtest /c", `Gpu);
    ("build @./c/runtest", `Gpu);
    ("build @c/runtest -- @a/runtest-t_cuda", `Gpu);
    ("build --default-target=@runtest", `Gpu);
    ("build --alias-rec runtest-t_cfg", `Gpu);
    ("build --alias c/runtest", `Gpu);
    ("build --root /elsewhere @c/runtest", `Gpu);
    ("build --workspace=other @c/runtest", `Gpu);
    ("build --frobnicate @c/runtest", `Gpu);
    ("runtest --root /elsewhere c", `Gpu);
    ("build ./a/t_cfg.exe", `Gpu);
    (* A build-context root is no source directory: dune runs that context's whole suite. *)
    ("build @_build/default/runtest", `Gpu);
    ("build @@_build/default/c/runtest", `Gpu);
    ("runtest _build/default", `Gpu);
    ("build @.hidden/runtest", `Gpu);
    ("build _build/default/c/t_only.exe", `Gpu);
    (* ...while the listed harmless options, in each spelling, change nothing. *)
    ("build -j 8 @a/runtest-t_cfg", `Cpu);
    ("build -j8 --force --profile=release @c/runtest", `Cpu);
    ("build --display short @c/runtest", `Cpu);
    ("runtest -j 4 c", `Cpu);
    (* No target is the default alias everywhere. *)
    ("build", `Gpu);
    (* A subcommand that runs no test reaches nothing. *)
    ("promote", `Cpu);
    ("clean", `Cpu);
    (* Any other subcommand may build workspace code: [install] builds what it installs. *)
    ("install", `Gpu);
    (* An alias no stanza carries is one of dune's own, modelled only where listed. *)
    ("build @c/revdep-runtest", `Gpu);
    (* Options that change what is built or run something: a moved build directory (whose
       context-rooted aliases read as source directories), an instrumentation ppx, a diff
       program. *)
    ("build --build-dir out @out/default/runtest", `Gpu);
    ("build --instrument-with bisect_ppx @c/runtest", `Gpu);
    ("build --diff-command=cmp @c/runtest", `Gpu);
    ("exec ./x.exe", `Gpu);
  ]

(* The whole set, where a GPU answer's backends matter to the width: a suite reaching stanzas that
   name different GPUs holds each of them, and a marker naming several contributes each. And whether
   the configurations count: only where a reached stanza reads them -- so a run reaching only
   [none]-marked stanzas holds nothing at all, whatever a test configuration names (gh-ocannl-1095),
   and neither does a subcommand that runs no test. *)
let sets =
  [
    ("runtest a", "names cuda,cc,hip + reads config");
    ("runtest", "names cuda,cc,hip,metal + reads config");
    ("build @b/runtest-m", "names cc,metal");
    ("build @a/gpuagg", "names cuda,cc + reads config");
    ("build @a/scans", "names cc + reads config");
    ("build @a/runtest-t_cc", "names cc");
    ("runtest c", "names nothing + reads config");
    ("runtest n", "names nothing");
    ("build @n/scans", "names nothing");
    ("promote", "names nothing");
    (* What no alias the argv names carries, reached all the same. *)
    ("build @f/runtest-probe", "names nothing + reads config");
    ("build @f/runtest-named", "names cuda");
    ("build @f/runtest-read", "names nothing + reads config");
    ("build @f/runtest-glob", "names cuda + reads config");
    ("build @@f/default", "names cuda + reads config");
    ("build @x/cross", "names nothing + reads config");
    ("build @x/cross_rec", "names nothing + reads config");
    ("build @g/runtest-t_gate", "names nothing + reads config");
    ("build @g/runtest-l", "names hip");
  ]

let () =
  List.iter cases ~f:(fun (argv, want) ->
      let answer, shown = judge argv in
      printf "%-40s %s\n" argv shown;
      let holds_gpu = Slot_kind.holds_gpu answer in
      match want with
      | `Cpu -> p (Printf.sprintf "%s reaches no GPU stanza" argv) (not holds_gpu)
      | `Gpu -> p (Printf.sprintf "%s can hold a GPU" argv) holds_gpu);
  List.iter sets ~f:(fun (argv, want) ->
      let _, shown = judge argv in
      p (Printf.sprintf "%s reaches exactly %s" argv want) (String.equal shown want));
  (* A marker the env_var_deps contract refuses is not read as no marker: the file is unreadable,
     which the caller takes as a GPU. *)
  let _, malformed =
    judge
      ~dune_files:
        [ ("d", {dune|(test ; ocannl-backend: cuda
 (name t) (deps ocannl_config))|dune}) ]
      "runtest d"
  in
  printf "%-40s %s\n" "runtest d (malformed marker)" malformed;
  p "a malformed marker makes the tree unreadable, never CPU"
    (String.is_prefix malformed ~prefix:"unknown: the dune file in d is unreadable");
  (* The two declarations env_var_deps refuses that the contract does not are read the widening way:
     a stanza declaring neither reads the configuration, and a second marker is not read as none --
     it makes the file unreadable, where reading it as naming nothing would now drop the stanza from
     the batch altogether. *)
  let _, neither =
    judge ~dune_files:[ ("d", {dune|(test (name t) (deps ocannl_config))|dune}) ] "runtest d"
  in
  printf "%-40s %s\n" "runtest d (no declaration)" neither;
  p "a stanza declaring neither reads the configuration"
    (String.equal neither "names nothing + reads config");
  let _, twice =
    judge
      ~dune_files:
        [
          ( "d",
            {dune|(test
 ; ocannl-backend: none -- one marker.
 ; ocannl-backend: cuda -- and another.
 (name t)
 (deps ocannl_config))|dune}
          );
        ]
      "runtest d"
  in
  printf "%-40s %s\n" "runtest d (two markers)" twice;
  p "a stanza carrying two markers makes the tree unreadable, never names nothing"
    (String.is_prefix twice ~prefix:"unknown: the dune file in d is unreadable");
  (* The configuration is taken off a batch only on a proof (gh-ocannl-1095, review round 2 on PR
     #1027): a construct the closure does not model exactly on a stanza it reaches makes the batch
     every backend -- what it builds is unread, its named backends included (round 5) -- and costs
     nothing to a batch not reaching it. *)
  let reason ~dune_files argv =
    match fst (judge ~dune_files argv) with
    | Slot_kind.Reaches { reads_config = Some why; _ } -> why
    | Slot_kind.Reaches { reads_config = None; _ } -> "<the configuration does not count>"
    | Slot_kind.Unknown why -> "unknown: " ^ why
  in
  let n = ("n", List.Assoc.find_exn tree "n" ~equal:String.equal) in
  List.iter
    [
      ("package", {dune|(alias (name pkg) (deps (package neural_nets_lib)))|dune}, "build @d/pkg");
      ("include", {dune|(alias (name inc) (deps (include deps.sexp)))|dune}, "build @d/inc");
      ("escape", {dune|(alias (name up) (deps (alias ../../elsewhere/runtest)))|dune}, "build @d/up");
      ("pform alias", {dune|(alias (name pf) (deps (alias %{env:A=x}/runtest)))|dune}, "build @d/pf");
      ( "pform",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias odd) (action (run %{dep:d.exe} %{frobnicate})))|dune},
        "build @d/odd" );
      ( "dynamic-run",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias dyn) (action (dynamic-run ./d.exe)))|dune},
        "build @d/dyn" );
      ("env dependency", {dune|(alias (name ev) (deps %{env:INPUT=plain}))|dune}, "build @d/ev");
      ( "dependency field mixing a modelled alias with an unmodelled form",
        {dune|(alias (name mx) (deps (alias gpu-input) (package helper)))|dune},
        "build @d/mx" );
      ( "sandbox wrapper",
        {dune|(alias (name sb) (deps (sandbox (alias runtest-gpu))))|dune},
        "build @d/sb" );
      ( "alias form outside a dependency field",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias odd2) (action (run %{dep:d.exe})) (enabled_if (alias x)))|dune},
        "build @d/odd2" );
    ]
    ~f:(fun (what, dune, argv) ->
      let dune_files = [ ("d", dune); n ] in
      let _, shown = judge ~dune_files argv in
      let why = reason ~dune_files argv in
      printf "%-40s %s\n    %s\n" (argv ^ " (" ^ what ^ ")") shown why;
      p
        (Printf.sprintf "an unmodelled %s is every backend" what)
        (String.is_prefix shown ~prefix:"unknown: "
        && String.is_substring why ~substring:"does not model exactly");
      let _, beside = judge ~dune_files "build @n/scans" in
      p
        (Printf.sprintf "an unmodelled %s is nothing to a batch not reaching it" what)
        (String.equal beside "names nothing"));
  (* Holes in what EVERY batch builds are every backend everywhere: a top-level include (the stanzas
     it brings in are never read), a head the inventory does not know, and a preprocessing action --
     compilation is followed for every batch, which program links which library is not. *)
  List.iter
    [
      ("a top-level include", {dune|(include rules.inc)|dune});
      ("a cram test", {dune|(cram (deps ocannl_config))|dune});
      ("an install stanza", {dune|(install (section share) (files out.dat))|dune});
      ( "a backend flag in a preprocessing action",
        {dune|(library (name pp_user) (modules pp_user)
 (preprocess (action (run %{exe:pp.exe} --ocannl_backend=hip %{input-file}))))|dune}
      );
      ( "a copy_files attached to an alias",
        {dune|(copy_files (alias probe) (files ../g/*.dat))|dune} );
      ("an env setting variables", {dune|(env (_ (env-vars (FOO bar))))|dune});
      ("an env_vars backend", {dune|(env (_ (env_vars ((OCANNL_BACKEND hip)))))|dune});
      ("an env's other settings", {dune|(env (_ (binaries tool.exe)))|dune});
      ( "an env's preprocessor flag",
        {dune|(env (_ (flags (:standard -pp "gpu-preprocessor --ocannl_backend=hip"))))|dune} );
      ( "an included flags file",
        {dune|(library (name l3) (modules l3) (flags (:standard (:include flags.sexp))))|dune} );
      ( "a library's ppx flag",
        {dune|(library (name l2) (modules l2) (ocamlopt_flags (:standard -ppx ./gpu.exe)))|dune} );
      ( "a ppx that can reach a backend",
        {dune|(library (name evil_ppx) (kind ppx_rewriter) (modules evil_ppx) (libraries ppxlib helper))
(library (name helper) (modules helper) (libraries arrayjit.backends))
(library (name user) (modules user) (preprocess (pps evil_ppx)))|dune}
      );
      ( "a generated Reason module",
        {dune|(rule
 (target gen.re)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:gen.exe}))))|dune}
      );
      ( "a ctypes field",
        {dune|(library (name stubs) (ctypes (external_library_name m) (generated_entry_point C)))|dune}
      );
      ( "a preprocessing action",
        {dune|(library (name helper) (modules helper) (preprocess (action (run cat %{input-file}))))|dune}
      );
    ]
    ~f:(fun (what, dune) ->
      let dune_files = [ ("d", dune); n ] in
      let _, shown = judge ~dune_files "build @n/scans" in
      let why = reason ~dune_files "build @n/scans" in
      printf "%-40s %s\n    %s\n" ("build @n/scans (" ^ what ^ ")") shown why;
      (* The property is soundness -- no batch proven to read nothing -- whichever of the two safe
         answers the shape draws; the golden records which. *)
      p
        (Printf.sprintf "%s beside the tree leaves no batch proven to read nothing" what)
        (String.is_prefix shown ~prefix:"unknown: "
        || String.equal shown "names nothing + reads config"));
  (* The two shapes review round 2 ran end to end, each a configuration reader dune builds first: a
     character-class glob (matched as a glob, everything), and a host-only test whose linked
     library's preprocessing reads a configuration reader's output. *)
  let reader =
    {dune|(executable (name reader) (modules reader))
(rule
 (target a.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:reader.exe}))))|dune}
  in
  List.iter
    [
      ( "an inferred target on an alias-attached rule",
        {dune|(executable (name reader) (modules reader))
(rule
 (alias generate)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to table.dat (run %{dep:reader.exe}))))
(alias (name probe) (deps table.dat))|dune},
        "build @@d/probe" );
      ( "a target written by an action this does not read",
        {dune|(rule
 (alias generate)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (format-dune-file input.sexp table.dat)))
(alias (name probe) (deps table.dat))|dune},
        "build @@d/probe" );
      ( "a bare-star glob",
        reader ^ {dune|
(alias (name probe) (deps (glob_files *)))|dune},
        "build @@d/probe" );
      ( "an alias in a test's link_deps",
        reader
        ^ {dune|
(rule (alias gpu-input) (deps a.actual) (action (progn)))
(test
 ; ocannl-backend: none -- host-only runner in this fixture.
 (name none) (modules none) (link_deps (alias gpu-input)) (deps ocannl_config))|dune},
        "build @@d/runtest-none" );
      ( "a character-class glob",
        reader ^ {dune|
(alias (name probe) (deps (glob_files "[ab].actual")))|dune},
        "build @@d/probe" );
      ( "a linked library's preprocessing",
        reader
        ^ {dune|
(library (name helper) (modules helper)
 (preprocess (action (progn (echo %{read:a.actual}) (run cat %{input-file})))))
(test
 ; ocannl-backend: none -- host-only runner in this fixture.
 (name none) (modules none) (libraries helper) (deps ocannl_config))|dune},
        "build @@d/runtest-none" );
    ]
    ~f:(fun (what, dune, argv) ->
      let _, shown = judge ~dune_files:[ ("d", dune) ] argv in
      printf "%-40s %s\n    %s\n"
        (argv ^ " (" ^ what ^ ")")
        shown
        (reason ~dune_files:[ ("d", dune) ] argv);
      p
        (Printf.sprintf "%s reaches the reader, or is every backend" what)
        (String.is_prefix shown ~prefix:"unknown: "
        || String.equal shown "names nothing + reads config"));
  (* A rule producing a source-like file is taken as always built: anything compiling may need it,
     so a configuration-reading generator makes even the none-only batch read the configuration. *)
  let _, generated =
    judge
      ~dune_files:
        [
          ( "s",
            {dune|(rule
 (target gen.ml)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:gen.exe}))))|dune}
          );
          ("n", List.Assoc.find_exn tree "n" ~equal:String.equal);
        ]
      "build @n/scans"
  in
  printf "%-40s %s\n" "build @n/scans (beside a generated .ml)" generated;
  p "a configuration-reading generator of a source file is always reached"
    (String.equal generated "names nothing + reads config");
  (* A backend set past the configuration -- an env stanza's or a setenv's OCANNL_BACKEND, or a
     command-line flag on a stanza that reads the configuration -- is what the resolved
     configuration cannot answer for: every backend, not merely the configuration's. *)
  List.iter
    [
      ("an env's backend", {dune|(env (_ (env-vars (OCANNL_BACKEND hip))))|dune}, "build @n/scans");
      ( "a setenv'd backend",
        {dune|(rule
 (alias se) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (setenv OCANNL_BACKEND hip (run %{dep:d.exe}))))|dune},
        "build @d/se" );
      (* A directory target produces any path below it -- an ocannl_config among them. *)
      ( "a directory target",
        {dune|(executable (name reader) (modules reader))
(rule
 (targets (dir output))
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run %{dep:reader.exe})))
(alias (name probe) (deps output/result))|dune},
        "build @@d/probe" );
      ( "a generated configuration",
        {dune|(rule (target ocannl_config) (action (write-file %{target} "backend=hip")))
(alias (name gc) (deps ocannl_config))|dune},
        "build @d/gc" );
      ( "an environment-clearing env on a reader",
        {dune|(rule
 (alias ei) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run env -i ./d.exe)))|dune},
        "build @d/ei" );
      ( "an env-command assignment on a reader",
        {dune|(rule
 (alias ev2) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run env OCANNL_BACKEND=hip %{exe:d.exe})))|dune},
        "build @d/ev2" );
      ( "a shell assignment on a reader",
        {dune|(rule
 (alias sh) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (system "OCANNL_BACKEND=hip ./d.exe")))|dune},
        "build @d/sh" );
      ( "a dropped configuration on a reader",
        {dune|(rule
 (alias nc) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (setenv OCANNL_NO_CONFIG_FILE true (run %{dep:d.exe}))))|dune},
        "build @d/nc" );
      ( "a prefix-free backend flag on a reader",
        {dune|(rule
 (alias fl2) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run %{dep:d.exe} --backend=hip)))|dune},
        "build @d/fl2" );
      ( "a single-dash uppercase backend flag on a reader",
        {dune|(rule
 (alias fl3) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run %{dep:d.exe} -OCANNL-BACKEND hip)))|dune},
        "build @d/fl3" );
      ( "a backend flag on a reader",
        {dune|(rule
 (alias fl) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run %{dep:d.exe} --ocannl_backend=hip)))|dune},
        "build @d/fl" );
    ]
    ~f:(fun (what, dune, argv) ->
      let _, shown = judge ~dune_files:[ ("d", dune); n ] argv in
      printf "%-40s %s\n" (argv ^ " (" ^ what ^ ")") shown;
      p (Printf.sprintf "%s is every backend" what) (String.is_prefix shown ~prefix:"unknown: "));
  (* Private libraries are scoped by project: a ppx name two projects each define is read as
     both. *)
  let _, scoped =
    judge
      ~dune_files:
        [
          ( "p1",
            {dune|(library (name my_ppx) (kind ppx_rewriter) (modules my_ppx) (libraries ocannl))
(library (name u1) (modules u1) (preprocess (pps my_ppx)))|dune}
          );
          ( "p2",
            {dune|(library (name my_ppx) (kind ppx_rewriter) (modules my_ppx) (libraries ppxlib))|dune}
          );
          n;
        ]
      "build @n/scans"
  in
  printf "%-40s %s\n" "build @n/scans (a ppx name in two projects)" scoped;
  p "a ppx name defined twice is read as every definition"
    (String.is_prefix scoped ~prefix:"unknown: ");
  (* [default] is [(alias_rec all)] even under [@@]; and a stanza the contract found running nothing
     can run a workspace program by its public name, which then declares nothing. *)
  List.iter
    [
      ( "an implicit default recursing",
        [
          ( "r/sub",
            {dune|(executable (name p) (modules p))
(rule
 (target x.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:p.exe}))))|dune}
          );
        ],
        "build @@r/default",
        "names nothing + reads config" );
      ( "a reader beside a configuration copied from a directory the runner does not read",
        [
          ( "q",
            {dune|(copy_files ../hipcfg/ocannl_config)
(test (name rq) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
          );
        ],
        "build @q/runtest",
        "unknown: " );
      ( "a reader beside a configuration copied from one it does",
        [
          ( "q2",
            {dune|(copy_files ../test/config/ocannl_config)
(test (name rq2) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
          );
        ],
        "build @q2/runtest",
        "names nothing + reads config" );
      ( "a workspace program run by name",
        [
          ( "w",
            {dune|(executable (name tool) (public_name gpu_tool) (modules tool))
(rule (alias wp) (action (run gpu_tool)))|dune}
          );
        ],
        "build @w/wp",
        "unknown: " );
    ]
    ~f:(fun (what, dune_files, argv, want) ->
      let _, shown = judge ~dune_files argv in
      printf "%-40s %s\n" (argv ^ " (" ^ what ^ ")") shown;
      p
        (Printf.sprintf "%s answers %s" what
           (String.rstrip want ~drop:(fun c -> Char.equal c ' ' || Char.equal c ':')))
        (String.is_prefix shown ~prefix:want));
  (* A glob matches where it points, so a glob over copies reaches the reader through the copy: a
     [copy_files] produces its copies in its own directory, and needs the glob it copies from. *)
  let _, copied =
    judge
      ~dune_files:
        [
          ( "r",
            {dune|(executable (name reader) (modules reader))
(rule
 (target a.dat)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:reader.exe}))))|dune}
          );
          ("c", {dune|(copy_files ../r/*.dat)
(alias (name probe) (deps (glob_files *.dat)))|dune});
        ]
      "build @@c/probe"
  in
  printf "%-40s %s\n" "build @@c/probe (a glob over copies)" copied;
  p "a glob over a copy_files' copies reaches the reader they copy"
    (String.equal copied "names nothing + reads config");
  (* The repository's own tree (its dune files, copied beside this test by the stanza's deps): the
     batch the issue is about holds no backend, and the one review round 1 found reading the
     configuration through the [.actual] its diff consumes still does. A new stanza that breaks the
     proof for [scans] -- a construct not modelled exactly, anywhere compilation reaches -- shows
     here, rather than as scans quietly taking a GPU token again. *)
  (* The inventory goes where dune goes: a [(dirs …)] stanza admits a hidden directory, whose own
     [(dirs …)] restricts it in turn, and an underscore directory stays out. *)
  let root = Stdlib.Filename.temp_file "slot_kind_dirs" "" in
  Stdlib.Sys.remove root;
  let made = ref [] in
  let file rel content =
    let path = Stdlib.Filename.concat root rel in
    let rec mkdirs d =
      if not (Stdlib.Sys.file_exists d) then (
        mkdirs (Stdlib.Filename.dirname d);
        Stdlib.Sys.mkdir d 0o755;
        made := d :: !made)
    in
    mkdirs (Stdlib.Filename.dirname path);
    Out_channel.write_all path ~data:content;
    made := path :: !made
  in
  file ".x/dune" "(dirs keep)";
  file ".x/keep/dune" "";
  file ".x/drop/dune" "";
  file "_skip/dune" "";
  file "plain/dune" "";
  file "dune"
    "(dirs :standard .x) (data_only_dirs fixtures) (subdir tools (dirs :standard .hidden))";
  file "fixtures/dune" "(not a dune file)";
  file "tools/.hidden/dune" "";
  file "tools/sub/dune" "";
  let read = List.map (Slot_kind.dune_files ~workspace_root:false ~root ()) ~f:fst in
  (* An ordered-set operator changes what the rest of a directory set means; it is not read. *)
  Out_channel.write_all
    (Stdlib.Filename.concat root "dune")
    ~data:"(data_only_dirs :standard \\ plain)";
  let refused_for substring =
    match Slot_kind.dune_files ~workspace_root:false ~root () with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring
  in
  let refused = refused_for "directory-set" in
  (* What dune reads beside the dune files: a workspace naming more than its language, and the
     alternative dune-file name. *)
  Out_channel.write_all (Stdlib.Filename.concat root "dune") ~data:"(dirs :standard .x)";
  file "dune-workspace"
    "(lang dune 3.20)\n(context (default (env (_ (env_vars (OCANNL_BACKEND hip))))))";
  let refused_workspace = refused_for "dune-workspace" in
  Stdlib.Sys.remove (Stdlib.Filename.concat root "dune-workspace");
  made := List.filter !made ~f:(fun p -> not (String.is_suffix p ~suffix:"dune-workspace"));
  file "plain/dune-file" "";
  let refused_dune_file = refused_for "dune-file" in
  Stdlib.Sys.remove (Stdlib.Filename.concat root "plain/dune-file");
  made := List.filter !made ~f:(fun p -> not (String.is_suffix p ~suffix:"dune-file"));
  file "dune-project" "(lang dune 3.20)\n(dialect (name d) (implementation (extension dat)))";
  let refused_dialect = refused_for "dialect" in
  Stdlib.Sys.remove (Stdlib.Filename.concat root "dune-project");
  made := List.filter !made ~f:(fun p -> not (String.is_suffix p ~suffix:"/dune-project"));
  file "plain/dune-project" "(lang dune 3.20)\n(dialect (name d) (implementation (extension dat)))";
  let refused_nested_dialect = refused_for "dialect" in
  Stdlib.Sys.remove (Stdlib.Filename.concat root "plain/dune-project");
  made := List.filter !made ~f:(fun p -> not (String.is_suffix p ~suffix:"/dune-project"));
  file "plain/gpu.t" "  $ ./gpu.exe";
  let refused_cram = refused_for "cram" in
  Stdlib.Sys.remove (Stdlib.Filename.concat root "plain/gpu.t");
  made := List.filter !made ~f:(fun p -> not (String.is_suffix p ~suffix:"gpu.t"));
  Out_channel.write_all (Stdlib.Filename.concat root "dune") ~data:"(data_only_dirs plain[12])";
  let refused_class = refused_for "data_only_dirs" in
  (* An ancestor holding a dune-project takes dune's root from a directory without its own
     dune-workspace, so dune builds a tree this did not read. *)
  Out_channel.write_all (Stdlib.Filename.concat root "dune") ~data:"";
  file "dune-project" "(lang dune 3.20)";
  file "plain/tree/dune" "";
  let ancestor_root =
    match Slot_kind.dune_files ~root:(Stdlib.Filename.concat root "plain/tree") () with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"takes dune's root"
  in
  (* Rules running beside the walk delete their outputs as it reads (gh-ocannl-1227): an entry gone
     between the listing and its stat, and a directory gone before its own listing, are passed over
     rather than ending the walk. *)
  let vanished_entry =
    List.equal String.equal
      (Slot_kind.subdirectories root [ "plain"; "gone.actual"; "dune" ])
      [ "plain" ]
  in
  let gone = Stdlib.Filename.concat root "gone" in
  let vanished_dir = Slot_kind.listing gone in
  (* ... but an entry a readable, unsearchable directory lists is not gone: its stat fails with
     EACCES, which must stay a failure. Where permissions are not enforced (root), the stat succeeds
     and there is nothing to refuse. *)
  file "locked/inner/dune" "";
  let locked = Stdlib.Filename.concat root "locked" in
  Unix.chmod locked 0o644;
  let unsearchable_refused =
    match Unix.stat (Stdlib.Filename.concat locked "inner") with
    | _ -> true
    | exception Unix.Unix_error _ -> (
        match Slot_kind.subdirectories locked [ "inner" ] with
        | _ -> false
        | exception Unix.Unix_error (Unix.EACCES, _, _) -> true)
  in
  Unix.chmod locked 0o755;
  List.iter !made ~f:(fun p ->
      if Stdlib.Sys.is_directory p then Stdlib.Sys.rmdir p else Stdlib.Sys.remove p);
  printf "dirs: %s\n" (String.concat ~sep:" " (List.map read ~f:(fun d -> "[" ^ d ^ "]")));
  p "the inventory reads where dirs stanzas send dune, and nowhere else"
    (List.equal String.equal read [ ""; ".x"; ".x/keep"; "plain"; "tools/.hidden"; "tools/sub" ]);
  p "a directory set with an ordered-set operator makes the tree unreadable" refused;
  p "a workspace setting a context environment makes the tree unreadable" refused_workspace;
  p "an alternative dune-file makes the tree unreadable" refused_dune_file;
  p "a dialect declared in dune-project makes the tree unreadable" refused_dialect;
  p "a dialect declared in a nested dune-project makes the tree unreadable" refused_nested_dialect;
  p "an implicitly discovered cram test makes the tree unreadable" refused_cram;
  p "a data_only_dirs character class makes the tree unreadable" refused_class;
  p "an ancestor dune-project taking dune's root makes the tree unreadable" ancestor_root;
  p "an entry that vanished after the listing is no directory" vanished_entry;
  p_empty "a directory that vanished before its listing holds nothing" ~over:[ gone ] vanished_dir;
  p "an entry an unsearchable directory lists is not taken for vanished" unsearchable_refused;
  (* DUNE_BUILD_DIR moves the build directory the same way --build-dir does. *)
  let moved =
    match
      Slot_kind.answer
        ~getenv:(function "DUNE_BUILD_DIR" -> Some "./out" | _ -> None)
        ~dune_files:tree
        [ "build"; "@out/default/runtest" ]
    with
    | Slot_kind.Unknown _ -> true
    | Slot_kind.Reaches _ -> false
  in
  p "an alias rooted in a moved build directory is every backend" moved;
  (* ... and the environment twins of the options that run a program. *)
  List.iter [ "DUNE_DIFF_COMMAND"; "DUNE_INSTRUMENT_WITH"; "DUNE_ROOT"; "OCAMLPARAM" ]
    ~f:(fun var ->
      let unknown =
        match
          Slot_kind.answer
            ~getenv:(fun v -> Option.some_if (String.equal v var) "x")
            ~dune_files:tree [ "build"; "@c/runtest" ]
        with
        | Slot_kind.Unknown _ -> true
        | Slot_kind.Reaches _ -> false
      in
      p (Printf.sprintf "%s set makes a batch every backend" var) unknown);
  let live = Slot_kind.dune_files ~workspace_root:false ~root:"../.." () in
  List.iter
    [
      ("build @test/operations/scans", "names nothing");
      ("build @test/operations/runtest-bandwidth_calibration", "names nothing + reads config");
    ]
    ~f:(fun (argv, want) ->
      let _, shown = judge ~dune_files:live argv in
      printf "live: %-55s %s\n    %s\n" argv shown (reason ~dune_files:live argv);
      p (Printf.sprintf "live: %s answers %s" argv want) (String.equal shown want))
