open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Refused_action_scan

let declaration ?(libraries = "arrayjit.verdict") ?(name = "probe") () =
  Printf.sprintf "(executable (name %s) (libraries %s))" name libraries

let rule ?(deps = "") action = Printf.sprintf "(rule (deps %s) (action %s))" deps action
let run = "(run %{dep:probe.exe})"
let refusal action = "(with-accepted-exit-codes 1 " ^ action ^ ")"
let capture action = "(ignore-stdout (with-stderr-to refusal.log " ^ action ^ "))"
let source action = declaration () ^ rule action
let scan source = Scan.scan [ ("fixture/dune", source) ]

(* Empty findings are deliberately the passing result; every caller supplies a concrete action
   fixture, and neighboring violating controls demonstrate that the scanner sees its obligation. *)
let clean source = List.is_empty (scan source).problems

let has source fragment =
  List.exists (scan source).problems ~f:(fun s -> String.is_substring s ~substring:fragment)

let () =
  p "the original stdout-only wrapper refuses its leaking stderr"
    (has (source ("(ignore-stdout " ^ refusal run ^ ")")) "accepted failure inherits stderr");
  p "a stderr-only wrapper refuses its leaking stdout"
    (has (source ("(ignore-stderr " ^ refusal run ^ ")")) "accepted failure inherits stdout");
  p "an unwrapped refusal reports both inherited streams"
    (has (source (refusal run)) "accepted failure inherits stdout and stderr");
  p "dropping stdout and capturing stderr is accepted" (clean (source (capture (refusal run))));
  p "capturing stdout and dropping stderr is accepted"
    (clean (source ("(with-stdout-to out (ignore-stderr " ^ refusal run ^ "))")));
  p "both-output capture and both-output ignore are accepted"
    (clean (source ("(with-outputs-to out " ^ refusal run ^ ")"))
    && clean (source ("(ignore-outputs " ^ refusal run ^ ")")));
  p "wrappers inside the accepted-status action count too" (clean (source (refusal (capture run))));
  p "a captured sibling cannot answer for a leaking run"
    (has
       (source (refusal ("(progn " ^ capture run ^ " " ^ run ^ ")")))
       "inherits stdout and stderr");
  p "stream capture does not escape its branch"
    (has
       (source ("(progn " ^ capture (refusal run) ^ " " ^ refusal run ^ ")"))
       "inherits stdout and stderr");
  p "a zero-only accepted-status action has no refusal-stream obligation"
    (clean (source ("(with-accepted-exit-codes 0 " ^ run ^ ")")));
  p "allowing success as well as failure still requires stream protection"
    (has (source ("(with-accepted-exit-codes (or 0 1) " ^ run ^ ")")) "inherits stdout and stderr");
  p "an inner zero-only status wrapper replaces the outer predicate"
    (clean (source (refusal ("(with-accepted-exit-codes 0 " ^ run ^ ")"))));
  p "a normal Verdict run may inherit both streams" (clean (source run));
  p "an executable without a direct Verdict dependency is outside the contract"
    (clean (declaration ~libraries:"base" () ^ rule (refusal run)));
  p "ownership uses the declared program rather than its name"
    (has
       (declaration ~name:"renamed" () ^ rule (refusal "(run %{dep:renamed.exe})"))
       "fixture/renamed.exe");
  p "named dependencies retain their owner under chdir"
    (has
       (declaration () ^ rule ~deps:"(:runner probe.exe)" (refusal "(chdir nested (run %{runner}))"))
       "inherits stdout and stderr");
  p "public names retain ownership across Dune files"
    (let result =
       Scan.scan
         [
           ( "a/dune",
             "(executable (name probe) (public_name pkg.probe) (libraries arrayjit.verdict))" );
           ("b/dune", rule (refusal "(run %{bin:pkg.probe})"));
         ]
     in
     List.exists result.problems ~f:(fun s -> String.is_substring s ~substring:"a/probe.exe"));
  p "dependency expansions preserve stanza origin under chdir"
    (has (source (refusal ("(chdir nested " ^ run ^ ")"))) "fixture/probe.exe");
  p "a literal command follows chdir to its distinct owner"
    (clean
       (declaration () ^ "(subdir nested " ^ declaration ~libraries:"base" () ^ ")"
       ^ rule (refusal "(chdir nested (run ./probe.exe))")));
  p "parent subdir declarations are attributed to their own directory"
    (has ("(subdir nested " ^ source (refusal run) ^ ")") "fixture/nested/probe.exe");
  p "multi-program declarations assign the direct dependency to each main program"
    (has
       ("(executables (names alpha beta) (libraries arrayjit.verdict))"
       ^ rule (refusal "(run %{dep:beta.exe})"))
       "fixture/beta.exe");
  p "a custom self-running test action resolves its declared program"
    (has
       "(test (name probe) (libraries arrayjit.verdict) (action (with-accepted-exit-codes 1 (run \
        %{test}))))"
       "fixture/probe.exe");
  p "a bare external tool with no workspace arguments stays outside the contract"
    (clean (rule (refusal "(run false)")));
  p "opaque shell and external launchers are refused explicitly"
    (has (source (refusal "(bash ./probe.exe)")) "unsupported refused action bash"
    && has (source (refusal "(run env %{dep:probe.exe})")) "unsupported refused command env");
  p "unknown runner ownership and computed commands are refused explicitly"
    (has (rule (refusal run)) "no executable declaration"
    && has (source (refusal "(run %{read:command})")) "unsupported refused command");
  p "computed predicates and library membership are refused explicitly"
    (has
       (source ("(with-accepted-exit-codes %{read:codes} " ^ run ^ ")"))
       "unsupported accepted-exit predicate"
    && has
         (declaration ~libraries:"%{read:libs}" () ^ rule (capture (refusal run)))
         "unsupported computed libraries membership");
  p "environment command-resolution overrides cannot hide a refused owner"
    (has "(env (_ (binaries (other as probe))))"
       "unsupported environment command-resolution override");
  p "action-local PATH rewrites cannot hide a refused owner"
    (has (source (refusal "(setenv PATH elsewhere (run probe.exe))")) "unsupported refused command");
  p "accepted-exit actions in preprocessing or nested stanza fields cannot be skipped"
    (has
       (declaration () ^ "(executable (name pp) (preprocess (action " ^ refusal run ^ ")))")
       "unsupported accepted-exit action outside the direct action field");
  p "include directives cannot silently hide ownership"
    (has "(include other.dune)" "unsupported include directive");
  p "directory-spanning ownership is refused while include_subdirs no is harmless"
    (has "(include_subdirs unqualified)" "unsupported include_subdirs mode"
    && clean "(include_subdirs no)");
  p "absolute capture destinations cannot masquerade as stream protection"
    (has
       (source ("(with-outputs-to /dev/stdout " ^ refusal run ^ ")"))
       "unsupported absolute capture destination");
  p "unknown action forms in refusal branches are explicit refusals"
    (has
       (source (refusal ("(future-wrapper " ^ run ^ ")")))
       "unsupported refused action future-wrapper");
  p "a pipeline or unknown wrapper around acceptance is refused too"
    (has
       (source ("(pipe-stdout " ^ refusal run ^ " (run cat))"))
       "unsupported refused action pipe-stdout"
    && has
         (source ("(future-wrapper " ^ refusal run ^ ")"))
         "unsupported refused action future-wrapper");
  (* Permanent shipping negative controls use the shared separate-stream runner, so their Verdict
     refusal markers cannot contaminate this passing test's log. *)
  let exe = Stdlib.Sys.argv.(1) in
  let fixture = Stdlib.Filename.temp_file "refused action " ".dune" in
  Exn.protect
    ~finally:(fun () -> Unix.unlink fixture)
    ~f:(fun () ->
      let check label contents ~exit ~fragment =
        Out_channel.write_all fixture ~data:contents;
        let child = Fresh_process.run ~exe [ "--fixture"; fixture ] in
        let ok = Fresh_process.matches ~exit ~contains:[ fragment ] child in
        if not ok then Fresh_process.report ~label child;
        p label ok
      in
      check "shipping scanner refuses the original stdout-only leak"
        (source ("(ignore-stdout " ^ refusal run ^ ")"))
        ~exit:1 ~fragment:"accepted failure inherits stderr";
      check "shipping scanner refuses a stdout leak too"
        (source ("(ignore-stderr " ^ refusal run ^ ")"))
        ~exit:1 ~fragment:"accepted failure inherits stdout";
      check "shipping scanner accepts the nearest both-stream correction"
        (source (capture (refusal run)))
        ~exit:0 ~fragment:"inherit neither stream";
      check "shipping scanner accepts a non-Verdict owner"
        (declaration ~libraries:"base" () ^ rule (refusal run))
        ~exit:0 ~fragment:"inherit neither stream";
      check "shipping scanner refuses unresolved includes" "(include other.dune)" ~exit:1
        ~fragment:"unsupported include directive";
      check "shipping scanner refuses unreadable Dune input" "(rule" ~exit:1
        ~fragment:"cannot read Dune input");
  let empty_root = Stdlib.Filename.temp_dir "refused action empty root " "" in
  Exn.protect
    ~finally:(fun () -> Unix.rmdir empty_root)
    ~f:(fun () ->
      let child = Fresh_process.run ~exe [ empty_root ] in
      let ok =
        Fresh_process.matches ~exit:1 ~contains:[ "Dune source inventory below floor" ] child
      in
      if not ok then Fresh_process.report ~label:"empty source inventory" child;
      p "shipping scanner refuses an empty Dune source inventory" ok)
