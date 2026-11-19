(* gh-ocannl-926: the outside-sweep owner reader (test/support/sweep_owner_scan.ml) on synthetic
   trees. Each refusal is reached by the deletion or drift it exists to catch -- a skip with no
   owner, a CI step that no longer builds the alias, an alias no stanza defines any more -- and each
   owner kind resolves on a tree that still holds its matrix, so a reader refusing everything fails
   here too. *)
open Base
open Verdict.Claims
module Scan = Test_scan.Sweep_owner_scan

let skip_source owner =
  Printf.sprintf
    "let () =\n  Verdict.skipped ~aggregation:(`Outside_sweep %s) ~backend:\"gate\" \"the claim\"\n"
    owner

let ci_owner ?(workflow = "ci.yml") ?(step = "Trace leg") ?(alias = "t/runtest-probe") () =
  Printf.sprintf "(Verdict.Ci_step { workflow = %S; step = %S; alias = %S })" workflow step alias

let plain_owner ?(alias = "t/runtest-probe") ?(forcing = "--forced") () =
  Printf.sprintf "(Verdict.Plain_run { alias = %S; forcing = %S })" alias forcing

let probe_dune =
  {|(test
 (name probe)
 (deps
  (env_var LEVEL))
 (modules probe))

(rule
 (alias runtest-probe-forced)
 (action
  (run %{dep:probe.exe} --forced)))
|}

let workflow ?(run = "dune build \"@check\" \"@t/runtest-probe\"") () =
  Printf.sprintf
    {|jobs:
  build:
    steps:
      - uses: actions/checkout@v4
      - name: Trace leg
        if: ${{ matrix.os == 'ubuntu-latest' }}
        env:
          LEVEL: 3
        run: %s
      - name: Later step
        run: dune build "@t/runtest-other"
|}
    run

(* The report on a tree of [files], whose [source] is the skipping [t/probe.ml]. *)
let report ?(exemptions = []) ?(files = []) ?(dune = probe_dune) ?(workflow = workflow ()) source =
  let files = files @ [ ("t/dune", dune); (".github/workflows/ci.yml", workflow) ] in
  let read path = List.Assoc.find files path ~equal:String.equal in
  Scan.check ~exemptions ~read (Scan.uses ~file:"t/probe.ml" source)

let refuses label ?exemptions ?files ?dune ?workflow ~fragment source =
  let report = report ?exemptions ?files ?dune ?workflow source in
  p label
    (match (report.findings, report.resolved) with
    | [ finding ], [] -> String.is_substring finding ~substring:fragment
    | _ -> false)

let resolves label ?files ?dune ?workflow ~fragment source =
  let report = report ?files ?dune ?workflow source in
  p label
    (match (report.resolved, report.findings) with
    | [ line ], [] -> String.is_substring line ~substring:fragment
    | _ -> false)

let () =
  (* Resolving. *)
  resolves "a CI step that builds the skipping test's alias owns it"
    ~fragment:
      "t/probe.ml: ci-step:ci.yml:t/runtest-probe:Trace leg -- step \"Trace leg\" of \
       .github/workflows/ci.yml builds @t/runtest-probe with LEVEL set"
    (skip_source (ci_owner ()));
  resolves "a plain run of the same test, passing no forcing argument, owns the forced skip"
    ~fragment:"plain-run:t/runtest-probe:--forced -- @t/runtest-probe runs the source"
    (skip_source (plain_owner ()));
  resolves
    "a select arm the owning test compiles is the same source, and so is the copy the forced rule \
     runs"
    ~dune:
      {|(rule
 (action
  (copy# probe.ml probe_copy.ml)))

(test
 (name probe)
 (modules probe_main)
 (libraries
  (select probe_main.ml from (real_lib -> probe_main.real.ml) (-> probe.ml))))

(rule
 (alias runtest-probe-forced)
 (action
  (progn
   (with-stdout-to
    probe_copy.actual
    (run %{exe:probe_copy.exe} --forced)))))

(executable
 (name probe_copy)
 (modules probe_copy))
|}
    ~fragment:"plain-run:t/runtest-probe:--forced"
    (skip_source (plain_owner ()));
  resolves "a rule running an executable that compiles the source owns it through its alias"
    ~dune:
      {|(executable
 (name probe)
 (modules probe))

(rule
 (alias runtest-probe)
 (action
  (run %{dep:probe.exe})))

(rule
 (alias runtest-probe-forced)
 (action
  (run %{dep:probe.exe} --forced)))
|}
    ~fragment:"plain-run:t/runtest-probe:--forced"
    (skip_source (plain_owner ()));
  (* The issue's controls: a missing owner, a step that no longer builds the alias, an alias that no
     longer exists. *)
  refuses "a skip with no owner is refused"
    ~fragment:"t/probe.ml:1: an `Outside_sweep skip carries no owner"
    "let () = Verdict.skipped ~aggregation:`Outside_sweep ~backend:\"gate\" \"the claim\"\n";
  refuses "a CI step that no longer builds the alias is refused"
    ~workflow:(workflow ~run:"dune build \"@check\"" ())
    ~fragment:
      "step \"Trace leg\" of .github/workflows/ci.yml (line 5) no longer builds @t/runtest-probe"
    (skip_source (ci_owner ()));
  refuses "a plain-run alias no stanza defines any more is refused"
    ~fragment:"no stanza of t/dune defines @t/runtest-gone"
    (skip_source (plain_owner ~alias:"t/runtest-gone" ()));
  (* The other refusals, each by the drift it names. *)
  refuses "a CI step that only mentions the alias in a comment no longer builds it"
    ~workflow:(workflow ~run:"dune build \"@check\" # \"@t/runtest-probe\"" ())
    ~fragment:"no longer builds @t/runtest-probe"
    (skip_source (ci_owner ()));
  refuses "a CI step building only a longer alias that starts with this one does not build it"
    ~workflow:(workflow ~run:"dune build \"@t/runtest-probe-forced\"" ())
    ~fragment:"no longer builds @t/runtest-probe"
    (skip_source (ci_owner ()));
  refuses "a renamed CI step is refused"
    ~workflow:(workflow () |> String.substr_replace_all ~pattern:"Trace leg" ~with_:"Trace leg 2")
    ~fragment:".github/workflows/ci.yml has no step named \"Trace leg\""
    (skip_source (ci_owner ()));
  refuses "a deleted workflow is refused"
    ~fragment:".github/workflows/nightly.yml is not among the files read"
    (skip_source (ci_owner ~workflow:"nightly.yml" ()));
  refuses "an owner alias outside the skipping source's directory is refused"
    ~fragment:"owner alias @u/runtest-probe lies outside t"
    (skip_source (plain_owner ~alias:"u/runtest-probe" ()));
  refuses "an owning stanza that does not compile the source is refused"
    ~dune:
      {|(test
 (name probe)
 (modules other))

(rule
 (alias runtest-probe-forced)
 (action
  (run %{dep:probe.exe} --forced)))
|}
    ~fragment:
      "the stanza of t/dune defining @t/runtest-probe does not run an executable compiling probe.ml"
    (skip_source (plain_owner ()));
  refuses "an owner that itself passes the forcing argument is refused"
    ~dune:{|(test
 (name probe)
 (modules probe)
 (action
  (run %{test} --forced)))
|}
    ~fragment:"@t/runtest-probe itself passes the forcing argument `--forced`"
    (skip_source (plain_owner ()));
  refuses "a forcing argument no stanza passes any more is refused"
    ~fragment:"no stanza beside @t/runtest-probe runs the source with `outranked`"
    (skip_source (plain_owner ~forcing:"outranked" ()));
  refuses "a CI step that names the alias only outside a dune build no longer builds it"
    ~workflow:
      (workflow ()
      |> String.substr_replace_all ~pattern:"LEVEL: 3" ~with_:"ALIAS: \"@t/runtest-probe\""
      |> String.substr_replace_all ~pattern:"dune build \"@check\" \"@t/runtest-probe\""
           ~with_:"dune build \"@check\"")
    ~fragment:"no longer builds @t/runtest-probe"
    (skip_source (ci_owner ()));
  refuses "an owning rule that only depends on the executable does not run the source"
    ~dune:
      {|(executable
 (name probe)
 (modules probe))

(rule
 (alias runtest-probe)
 (deps probe.exe)
 (action
  (copy %{dep:probe.exe} probe.copied)))

(rule
 (alias runtest-probe-forced)
 (action
  (run %{dep:probe.exe} --forced)))
|}
    ~fragment:"the stanza of t/dune defining @t/runtest-probe does not run an executable compiling"
    (skip_source (plain_owner ()));
  refuses "a forcing argument only an unrelated test passes does not run the forced skip"
    ~dune:
      {|(test
 (name probe)
 (modules probe))

(rule
 (alias runtest-other-forced)
 (action
  (run %{dep:other.exe} --forced)))

(executable
 (name other)
 (modules other))
|}
    ~fragment:"no stanza beside @t/runtest-probe runs the source with `--forced`"
    (skip_source (plain_owner ()));
  refuses
    "a CI step that builds the alias but no longer sets the stanza's gate runs what the sweep runs"
    ~workflow:(workflow () |> String.substr_replace_all ~pattern:"LEVEL: 3" ~with_:"OTHER: 3")
    ~fragment:
      "step \"Trace leg\" of .github/workflows/ci.yml (line 5) builds @t/runtest-probe but its \
       env: sets none of LEVEL"
    (skip_source (ci_owner ()));
  refuses "a CI owner whose stanza declares no gate is refused"
    ~dune:(String.substr_replace_all probe_dune ~pattern:" (deps\n  (env_var LEVEL))\n" ~with_:"")
    ~fragment:"the stanza of @t/runtest-probe declares no (env_var ...) dependency"
    (skip_source (ci_owner ()));
  refuses "an owner the record grammar cannot carry is refused"
    ~fragment:"the record grammar cannot carry it: its workflow \"c:i.yml\" holds a colon"
    (skip_source (ci_owner ~workflow:"c:i.yml" ()));
  refuses "a computed owner is refused" ~fragment:"carries an owner that is not a literal"
    (skip_source "owner");
  refuses "an owner with a computed field is refused"
    ~fragment:"its Plain_run field forcing is not a string literal"
    (skip_source "(Verdict.Plain_run { alias = \"t/runtest-probe\"; forcing = flag })");
  (* The exemption, and its staleness. *)
  p "an exempted file's computed owner is read as a use and is not a finding"
    (let uses = Scan.uses ~file:"t/probe.ml" (skip_source "owner") in
     let r =
       Scan.check ~exemptions:[ ("t/probe.ml", "replays fixtures") ] ~read:(fun _ -> None) uses
     in
     match (uses, r.findings, r.resolved) with [ _ ], [], [] -> true | _ -> false);
  refuses "an exemption whose file constructs no owner is stale"
    ~exemptions:[ ("t/replay.ml", "replays fixtures") ]
    ~fragment:
      "t/replay.ml: exempted as spelling fixture owners, but it spells no outside-sweep owner"
    "let () = ()\n";
  p "a pattern or a type spelling the tag is not a use; the one expression beside them is"
    (match
       Scan.uses ~file:"t/probe.ml"
         ("type t = [ `Outside_sweep of int ]\nlet f = function `Outside_sweep _ -> () | _ -> ()\n"
        ^ skip_source "owner")
     with
    | [ { line = 4; _ } ] -> true
    | _ -> false)
