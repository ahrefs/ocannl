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
   aggregates in [a] reach one member each. *)
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
    (String.is_prefix twice ~prefix:"unknown: the dune file in d is unreadable")
