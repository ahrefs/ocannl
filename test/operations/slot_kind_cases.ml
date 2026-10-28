(** What {!Test_utils.Slot_kind} says a dune argv reaches, on a fixture tree built to separate the
    cases (gh-ocannl-1004, gh-ocannl-1066, gh-ocannl-1227): a stanza that NAMES a backend holds it
    whatever the configuration says, so tools/batch-backends.sh counts every backend a reachable
    marker names among the batch's -- for the fleet slot ([--cpu] only when none is a GPU) and for
    the width (the tightest any of them meets) -- and counts the configurations' backends unless
    every target is on the configuration-free alias list or a plain compilation product. Then the
    checks that keep that list honest ({!Test_utils.Slot_kind.refusals}), each on a tree that passes
    and one that it refuses. Every verdict below is phrased so that [true] is the passing reading,
    and the golden records the backends each argv was judged to reach, so a change of answer shows
    as a diff as well as a failed claim. Every tree here is a fixture; what the repository's own
    tree answers is [slot_kind_live]'s. *)

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
   and an inline-test library's GENERATED per-stanza aliases.

   And [test/operations], whose recursive [scans] is on the configuration-free list: a stanza naming
   nothing and one naming Metal, which the list's batch still holds. *)
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
    ( "test/operations",
      {dune|
(test
 ; ocannl-backend: none -- links no backend in this fixture.
 (name s_none)
 (deps ocannl_config))

(test
 ; ocannl-backend: metal -- names the Metal backend by name in this fixture.
 (name s_metal)
 (deps ocannl_config))

(alias
 (name scans)
 (deps (alias runtest-s_none) (alias runtest-s_metal)))
|dune}
    );
  ]

let judge ?(dune_files = tree) argv =
  let answer = Slot_kind.answer ~dune_files (String.split argv ~on:' ') in
  (answer, Slot_kind.summary answer)

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
    (* A plain compilation product compiles and runs nothing -- the CUDA test's program included --
       in either spelling, several at once, a library's archive too; a path no stanza declares (a
       rule's output, a test's output, a product of no stanza there) is unmodelled. *)
    ("build ./a/t_cfg.exe", `Cpu);
    ("build a/t_cuda.exe", `Cpu);
    ("build a/t_cfg.exe b/m.exe g/l.cmxa", `Cpu);
    ("build a/t_cuda.bc", `Cpu);
    ("build f/probe.actual", `Gpu);
    ("build a/t_cfg.exe.output", `Gpu);
    ("build a/nothere.exe", `Gpu);
    ("build g/t_gate.cmxa", `Gpu);
    ("build ../a/t_cfg.exe", `Gpu);
    ("build ./../a/t_cfg.exe", `Gpu);
    ("exec a/t_cfg.exe", `Gpu);
  ]

(* The whole set, where a GPU answer's backends matter to the width: a suite reaching stanzas that
   name different GPUs holds each of them, and a marker naming several contributes each. And whether
   the configurations count: unless every target is on the configuration-free list -- the recursive
   [@test/operations/scans] here, so not a [scans] in another directory, nor the same alias without
   its recursion -- or a plain compilation product (gh-ocannl-1227). A run reaching only
   [none]-marked stanzas counts them all the same, the state before gh-ocannl-1095. *)
let sets =
  [
    ("runtest a", "names cuda,cc,hip + reads config");
    ("runtest", "names cuda,cc,hip,metal + reads config");
    ("build @b/runtest-m", "names cc,metal + reads config");
    ("build @a/gpuagg", "names cuda,cc + reads config");
    ("build @a/scans", "names cc + reads config");
    ("build @a/runtest-t_cc", "names cc + reads config");
    ("runtest c", "names nothing + reads config");
    ("runtest n", "names nothing + reads config");
    ("build @n/scans", "names nothing + reads config");
    ("promote", "names nothing");
    (* What no alias the argv names carries, reached all the same. *)
    ("build @f/runtest-probe", "names nothing + reads config");
    ("build @f/runtest-named", "names cuda + reads config");
    ("build @f/runtest-read", "names nothing + reads config");
    ("build @f/runtest-glob", "names cuda + reads config");
    ("build @@f/default", "names cuda + reads config");
    ("build @x/cross", "names nothing + reads config");
    ("build @x/cross_rec", "names nothing + reads config");
    ("build @g/runtest-t_gate", "names nothing + reads config");
    ("build @g/runtest-l", "names hip + reads config");
    (* The configuration-free list: what it names still counts. *)
    ("build @test/operations/scans", "names metal");
    ("build -j 8 @test/operations/scans", "names metal");
    ("build @@test/operations/scans", "names metal + reads config");
    ("build @test/operations/runtest-s_none", "names nothing + reads config");
    ("build @test/operations/scans @c/runtest", "names metal + reads config");
    ("build @test/operations/scans a/t_cuda.exe", "names metal");
    (* Plain compilation products. *)
    ("build ./a/t_cfg.exe", "names nothing");
    ("build a/t_cfg.exe b/m.exe g/l.cmxa", "names nothing");
    ("build a/t_cfg.exe @c/runtest", "names nothing + reads config");
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
  (* A marker the env_var_deps contract refuses is read neither as no marker nor as every backend:
     the answer refuses the batch, naming the marker's file and line -- each refused marker of the
     tree, in one answer, as tools/test-run.sh reports them (MARKER REFUSED). *)
  let _, malformed =
    judge
      ~dune_files:
        [
          ("d", {dune|(test ; ocannl-backend: cuda
 (name t) (deps ocannl_config))|dune});
          ( "e",
            {dune|(test
 ; ocannl-backend: vulkan -- a backend this repository has no word for.
 (name u) (deps ocannl_config))|dune}
          );
        ]
      "runtest d"
  in
  printf "%-40s %s\n" "runtest d (malformed markers)" malformed;
  p "a malformed marker refuses the batch, naming its dune file and line"
    (String.is_prefix malformed ~prefix:"refused: d/dune:1: a backend marker that does not parse");
  p "every refused marker of the tree is named, not only the first"
    (String.is_substring malformed ~substring:"; e/dune:2: a backend marker that does not parse");
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
  p "a stanza carrying two markers refuses the batch at the second, never names nothing"
    (String.equal twice "refused: d/dune:3: a second backend marker on one stanza");
  (* A backend the action fixes reaches the batch through the marker, which `env_var_deps` requires
     to name it (gh-ocannl-1518): this resolver never reads the action. *)
  let _, pinned =
    judge
      ~dune_files:
        [
          ( "d",
            {dune|(rule
 ; ocannl-backend: hip -- the action pins hip in this fixture.
 (alias se) (deps ocannl_config)
 (action (setenv OCANNL_BACKEND hip (run %{dep:d.exe}))))|dune}
          );
        ]
      "build @d/se"
  in
  printf "%-40s %s\n" "build @d/se (a pin its marker names)" pinned;
  p "a backend the action pins holds the batch through its marker"
    (String.is_prefix pinned ~prefix:"names hip");
  (* gh-ocannl-1227's residual shapes, and its neighbours, as policy cases: no argv outside the list
     is ever proven unread, so each counts the configuration (or is every backend) however the
     stanza is written -- none of them is read, and none needs to be. *)
  let n = ("n", List.Assoc.find_exn tree "n" ~equal:String.equal) in
  List.iter
    [
      ( "a compiler response file",
        {dune|(library (name l) (modules l) (flags (:standard -args flags.txt)))
(test
 ; ocannl-backend: none -- links no backend in this fixture.
 (name t) (modules t) (libraries l) (deps ocannl_config))|dune},
        "build @d/runtest-t" );
      ( "a read pform in a command position",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias rd) (action (system "%{read:cmd.txt}")))|dune},
        "build @d/rd" );
      ( "env reached through a bin pform",
        {dune|(rule
 (alias be) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (run %{bin:env} -i %{exe:t.exe})))|dune},
        "build @d/be" );
      ( "a setenv'd backend",
        {dune|(rule
 (alias se) (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (setenv OCANNL_BACKEND hip (run %{dep:d.exe}))))|dune},
        "build @d/se" );
    ]
    ~f:(fun (what, dune, argv) ->
      let _, shown = judge ~dune_files:[ ("d", dune); n ] argv in
      printf "%-40s %s\n" (argv ^ " (" ^ what ^ ")") shown;
      p
        (Printf.sprintf "%s off the list counts the configuration" what)
        (String.is_prefix shown ~prefix:"unknown: "
        || String.is_suffix shown ~suffix:" + reads config"));
  (* A dependency the closure cannot follow -- an included dependency file, an alias spelled through
     a pform, either wrapped in a sandbox -- leaves what it builds unread, its named backends
     included: every backend, not merely the configuration (Codex review round 1 on
     staging#1318). *)
  List.iter
    [
      ( "an included dependency",
        {dune|(alias (name inc) (deps (include deps.sexp)))|dune},
        "build @d/inc" );
      ( "an alias spelled through a pform",
        {dune|(alias (name pf) (deps (alias %{env:A=x}/runtest)))|dune},
        "build @d/pf" );
      ( "a sandboxed included dependency",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias sbi) (deps (sandbox always) (:in (include deps.sexp))) (action (run %{dep:d.exe})))|dune},
        "build @d/sbi" );
      ( "a dynamic-run",
        {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias dyn) (action (dynamic-run ./d.exe)))|dune},
        "build @d/dyn" );
    ]
    ~f:(fun (what, dune, argv) ->
      let _, shown = judge ~dune_files:[ ("d", dune); n ] argv in
      printf "%-40s %s\n" (argv ^ " (" ^ what ^ ")") shown;
      p
        (Printf.sprintf "%s is every backend" what)
        (String.is_prefix shown ~prefix:"unknown: "
        && String.is_substring shown ~substring:"this cannot follow");
      let _, beside = judge ~dune_files:[ ("d", dune); n ] "build @n/scans" in
      p
        (Printf.sprintf "%s is nothing to a batch not reaching it" what)
        (String.equal beside "names nothing + reads config"));
  (* A hole in the inventory is every batch's, the listed alias's and a plain build's included: what
     an [(include …)] or a [cram] test brings in may sit anywhere (Codex review round 3 on
     staging#1318). *)
  List.iter
    [
      ("an included stanza", {dune|(include rules.inc)|dune});
      ("a cram test", {dune|(cram (deps x))|dune});
    ]
    ~f:(fun (what, dune) ->
      let dune_files = ("w", dune) :: tree in
      List.iter [ "build @test/operations/scans"; "build a/t_cfg.exe"; "runtest c" ] ~f:(fun argv ->
          let _, shown = judge ~dune_files argv in
          printf "%-40s %s\n" (argv ^ " (beside " ^ what ^ ")") shown;
          p
            (Printf.sprintf "%s beside %s is every backend" argv what)
            (String.is_prefix shown ~prefix:"unknown: ")));
  (* A long-form copy_files attaches its copies to an alias: the alias builds what they copy (Codex
     review round 4 on staging#1318). *)
  let _, copied_on_alias =
    judge
      ~dune_files:
        [
          ( "g2",
            {dune|(rule
 ; ocannl-backend: cuda -- names the CUDA backend by name in this fixture.
 (target probe.actual)
 (action (with-stdout-to %{target} (run ./gpu_tool))))|dune}
          );
          ("test/operations", {dune|(copy_files (alias scans) (files ../../g2/*.actual))|dune});
        ]
      "build @test/operations/scans"
  in
  printf "%-40s %s\n" "build @test/operations/scans (a copy_files on it)" copied_on_alias;
  p "a copy_files attached to a listed alias reaches what it copies"
    (String.equal copied_on_alias "names cuda");
  (* A rule producing what looks like a compilation product is no plain build. *)
  let _, lookalike =
    judge
      ~dune_files:
        [
          ( "d",
            {dune|(rule
 ; ocannl-backend: cuda -- names the CUDA backend by name in this fixture.
 (target r.exe) (action (run ./gpu_tool --out %{target})))|dune}
          );
        ]
      "build d/r.exe"
  in
  printf "%-40s %s\n" "build d/r.exe (a rule's .exe)" lookalike;
  p "a rule-produced .exe is every backend, never a plain build"
    (String.is_prefix lookalike ~prefix:"unknown: ");
  (* Failing closed: on a tree the checks below would refuse -- a reader inside a listed alias, a
     generated source a reader writes -- the answer counts the configuration anyway, since the scan
     may not have run on the checkout yet (Codex review round 2 on staging#1318). *)
  List.iter
    [
      ( "a reader inside the listed alias",
        [
          ( "test/operations",
            {dune|(test (name s_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))
(alias (name scans) (deps (alias runtest-s_reader)))|dune}
          );
        ],
        "build @test/operations/scans" );
      ( "a source a reader generates, beside a plain build",
        [
          ( "d",
            {dune|(executable (name app) (modules app gen))
(rule
 (target gen.ml)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:gen_tool.exe}))))|dune}
          );
        ],
        "build d/app.exe" );
    ]
    ~f:(fun (what, dune_files, argv) ->
      let answer, shown = judge ~dune_files argv in
      let why =
        match answer with
        | Slot_kind.Reaches { reads_config = Some why; _ } -> why
        | _ -> "<the configuration does not count>"
      in
      printf "%-40s %s\n    %s\n" (argv ^ " (" ^ what ^ ")") shown why;
      p
        (Printf.sprintf "%s counts the configuration, failing closed" what)
        (String.equal shown "names nothing + reads config"));
  (* {1 The checks that keep the list honest} *)
  let scans_only =
    [ Slot_kind.Alias { dir = "test/operations"; alias = "scans"; recursive = true } ]
  in
  (* The shared configuration and its ambient copy, as the repository has them; [t] reads the
     copy. *)
  let ambient_rule =
    {dune|(subdir ambient
 (rule
  (target ocannl_config)
  (deps ../ocannl_config ../ocannl_ambient)
  (action
   (with-stdout-to %{target} (progn (cat ../ocannl_config) (cat ../ocannl_ambient))))))|dune}
  in
  let base =
    [
      ("test/config", ambient_rule);
      ( "t",
        {dune|(copy_files (files ../test/config/ambient/ocannl_config))
(test (name t_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
      );
      ( "test/operations",
        {dune|(copy_files (files ../config/ambient/ocannl_config))
(test
 ; ocannl-backend: none -- links no backend in this fixture.
 (name s_none)
 (deps ocannl_config))
(test
 ; ocannl-backend: metal -- names the Metal backend by name in this fixture.
 (name s_metal)
 (deps ocannl_config))
(alias (name scans) (deps (alias runtest-s_none) (alias runtest-s_metal)))|dune}
      );
    ]
  in
  let config_files = [ "test/config/ocannl_config"; "arrayjit/test/ocannl_config" ] in
  let refused ?(config_files = config_files) ?(config_free = scans_only) dune_files =
    Slot_kind.refusals ~config_free ~config_files dune_files
  in
  let replace dir dune files = (dir, dune) :: List.Assoc.remove files dir ~equal:String.equal in
  let clean = refused base in
  List.iter clean ~f:(printf "unexpected refusal: %s\n");
  p_empty "the fixture tree passes every check" ~over:base clean;
  let in_test_config_itself =
    refused
      (replace "test/config"
         (ambient_rule
        ^ {dune|
(test (name cfg_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune})
         base)
  in
  p_empty "a reader beside a configuration the runner resolves passes" ~over:base
    in_test_config_itself;
  (* A path-position read leaves a listed alias exact: it is a dependency like any file. *)
  let path_read =
    refused
      (replace "test/operations"
         {dune|(rule (target name.txt) (action (write-file %{target} x)))
(rule (alias runtest-logs) (deps "%{read:name.txt}.dat") (action (progn)))
(alias (name scans) (deps (alias runtest-logs)))|dune}
         base)
  in
  p_empty "a path-position read on a listed alias is refused nothing" ~over:base path_read;
  List.iter
    [
      ( "a listed alias's member reading the configuration",
        replace "test/operations"
          {dune|(copy_files (files ../config/ambient/ocannl_config))
(test (name s_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))
(alias (name scans) (deps (alias runtest-s_reader)))|dune}
          base,
        config_files,
        "the listed @test/operations/scans" );
      ( "a listed alias reaching a reader through a file",
        replace "test/operations"
          {dune|(copy_files (files ../config/ambient/ocannl_config))
(executable (name probe) (modules probe))
(rule
 (target probe.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:probe.exe}))))
(rule (alias runtest-probe) (action (diff probe.expected probe.actual)))
(alias (name scans) (deps (alias runtest-probe)))|dune}
          base,
        config_files,
        "the listed @test/operations/scans" );
      ( "a listed alias no stanza carries",
        List.Assoc.remove base "test/operations" ~equal:String.equal,
        config_files,
        "no stanza carries it" );
      ( "a listed alias with an included dependency",
        replace "test/operations" {dune|(alias (name scans) (deps (include deps.sexp)))|dune} base,
        config_files,
        "(include deps.sexp) this cannot follow" );
      ( "a listed alias with a member it cannot follow",
        replace "test/operations"
          {dune|(alias (name scans) (deps (alias %{env:WHERE=x}/runtest)))|dune} base,
        config_files,
        "cannot follow" );
      ( "a source file generated by a configuration reader",
        replace "t"
          {dune|(copy_files (files ../test/config/ambient/ocannl_config))
(rule
 (target gen.ml)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:gen.exe}))))|dune}
          base,
        config_files,
        "a plain compilation" );
      ( "a source file generated on a named GPU",
        replace "t"
          {dune|(rule
 ; ocannl-backend: cuda -- names the CUDA backend by name in this fixture.
 (target gen.ml)
 (deps ocannl_config)
 (action (with-stdout-to %{target} (run %{dep:gen.exe}))))|dune}
          base,
        config_files,
        "which names cuda" );
      ( "a reader with no configuration in reach",
        replace "u" {dune|(test (name u_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
          base,
        config_files,
        "no checked-in configuration in reach" );
      ( "a reader beside a copy of a configuration the runner does not resolve",
        replace "q"
          {dune|(copy_files ../hipcfg/ocannl_config)
(test (name q_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
          base,
        "hipcfg/ocannl_config" :: config_files,
        "reads the configuration in hipcfg" );
      ( "an undeclared stanza beside a copy of a configuration the runner does not resolve",
        replace "q"
          {dune|(copy_files ../hipcfg/ocannl_config)
(test (name q_undeclared) (deps ocannl_config))|dune}
          base,
        "hipcfg/ocannl_config" :: config_files,
        "q_undeclared running in q, which reads the configuration in hipcfg" );
      ( "a reader of an ambient configuration copied from elsewhere",
        replace "test/config"
          {dune|(subdir ambient (copy_files ../../../hipcfg/ocannl_config))|dune} base,
        "hipcfg/ocannl_config" :: config_files,
        "t_reader running in t, which reads the configuration in" );
      ( "a reader of an ambient configuration with no producer",
        List.Assoc.remove base "test/config" ~equal:String.equal,
        config_files,
        "t_reader running in t, which reads the configuration in test/config/ambient" );
      ( "a listed alias carrying a copy of a reader's output",
        replace "test/operations" {dune|(copy_files (alias scans) (files ../r/*.actual))|dune}
          (( "r",
             {dune|(copy_files (files ../test/config/ambient/ocannl_config))
(executable (name probe) (modules probe))
(rule
 (target probe.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:probe.exe}))))|dune}
           )
          :: base),
        config_files,
        "the listed @test/operations/scans" );
      ( "a reader beside a checked-in configuration the runner does not resolve",
        replace "prof"
          {dune|(test (name p_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune} base,
        "prof/ocannl_config" :: config_files,
        "reads the configuration in prof" );
      ( "a reader run under a chdir out of its configuration's reach",
        replace "t"
          {dune|(copy_files (files ../test/config/ambient/ocannl_config))
(rule
 (alias ch)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (chdir ../v (run %{dep:r.exe}))))|dune}
          base,
        config_files,
        "running in v" );
      ( "an ambient configuration appending more than the record",
        replace "test/config"
          {dune|(subdir ambient
 (rule
  (target ocannl_config)
  (deps ../ocannl_config ../ocannl_ambient)
  (action
   (with-stdout-to
    %{target}
    (progn (cat ../ocannl_config) (cat ../ocannl_ambient) (echo "backend=hip\n"))))))|dune}
          base,
        config_files,
        "can write an ocannl_config" );
      ( "a configuration generated elsewhere",
        replace "w"
          {dune|(rule (target ocannl_config) (action (write-file %{target} "backend=hip")))|dune}
          base,
        config_files,
        "can write an ocannl_config" );
      ( "an included stanza",
        replace "w" {dune|(include rules.inc)|dune} base,
        config_files,
        "(include …)" );
      ( "a cram test",
        replace "w" {dune|(cram (deps ocannl_config))|dune} base,
        config_files,
        "(cram …)" );
      ( "a dynamic-run",
        replace "w"
          {dune|(rule
 ; ocannl-backend: none -- links no backend in this fixture.
 (alias dyn) (action (dynamic-run ./d.exe)))|dune}
          base,
        config_files,
        "dynamic-run" );
    ]
    ~f:(fun (what, dune_files, config_files, substring) ->
      let found = refused ~config_files dune_files in
      List.iter found ~f:(fun r -> printf "%s:\n    %s\n" what r);
      p_exists (Printf.sprintf "%s is refused" what) found ~f:(String.is_substring ~substring));
  (* The run-time answer fails closed on the same reader check, whatever the argv (Codex review
     round 5 on staging#1318): a reader of a foreign copy anywhere makes every batch every backend,
     the listed alias's included, while the clean tree answers as ever. *)
  let foreign =
    replace "q"
      {dune|(copy_files ../hipcfg/ocannl_config)
(test (name q_reader) (deps ocannl_config (env_var OCANNL_BACKEND)))|dune}
      base
  in
  List.iter [ "build @test/operations/scans"; "runtest t" ] ~f:(fun argv ->
      let shown dune_files config_files =
        Slot_kind.summary (Slot_kind.answer ~config_files ~dune_files (String.split argv ~on:' '))
      in
      let refused = shown foreign ("hipcfg/ocannl_config" :: config_files) in
      let clean = shown base config_files in
      printf "%-40s %s\n%-40s %s\n" (argv ^ " (a foreign reader)") refused (argv ^ " (clean)") clean;
      p
        (Printf.sprintf "%s beside a reader of a foreign configuration is every backend" argv)
        (String.is_prefix refused ~prefix:"unknown: ");
      p
        (Printf.sprintf "%s on the clean tree is no unknown" argv)
        (not (String.is_prefix clean ~prefix:"unknown: ")));
  (* A data file a reader produces is no plain compilation's business: nothing compiling needs
     it. *)
  let data =
    refused
      (replace "t"
         {dune|(copy_files (files ../test/config/ambient/ocannl_config))
(executable (name probe) (modules probe))
(rule
 (target probe.actual)
 (deps ocannl_config (env_var OCANNL_BACKEND))
 (action (with-stdout-to %{target} (run %{dep:probe.exe}))))|dune}
         base)
  in
  p_empty "a reader producing a data file off the list is refused nothing" ~over:base data;
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
  (* By the exact path [file] recorded: a suffix match on "/dune-project" missed the root's own,
     which Filename.concat spells with a backslash on Windows, and cleanup removed it twice. *)
  let unmake rel =
    let path = Stdlib.Filename.concat root rel in
    Stdlib.Sys.remove path;
    made := List.filter !made ~f:(fun p -> not (String.equal p path))
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
  unmake "dune-workspace";
  file "plain/dune-file" "";
  let refused_dune_file = refused_for "dune-file" in
  unmake "plain/dune-file";
  file "dune-project" "(lang dune 3.20)\n(dialect (name d) (implementation (extension dat)))";
  let refused_dialect = refused_for "dialect" in
  unmake "dune-project";
  file "plain/dune-project" "(lang dune 3.20)\n(dialect (name d) (implementation (extension dat)))";
  let refused_nested_dialect = refused_for "dialect" in
  unmake "plain/dune-project";
  file "plain/gpu.t" "  $ ./gpu.exe";
  let refused_cram = refused_for "cram" in
  unmake "plain/gpu.t";
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
    | Slot_kind.Reaches _ | Slot_kind.Refused _ -> false
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
        | Slot_kind.Reaches _ | Slot_kind.Refused _ -> false
      in
      p (Printf.sprintf "%s set makes a batch every backend" var) unknown)
