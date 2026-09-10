(* gh-ocannl-652, Codex P1 on PR #389 (rounds 1 and 2): the ambient check a cached suite cannot
   hide. Copied into every test directory that runs OCANNL executables -- see the `(copy_files)` and
   `(test)` pair in each of their dune files, and `env_var_deps`, which fails if a directory with
   runtest actions has no gate.

   Setting a spelling OCANNL does not read aborts the run -- but only a run that HAPPENS. A dune
   rule reruns for an environment variable solely where the stanza declares it, and the rejected
   spellings are declared nowhere any more (that was the point: 227 declarations, growing with every
   test). So `ocannl_backend=cuda dune runtest` would rerun nothing, serve the previous run's
   outputs, and report a green suite -- the fatal check never reached, the user still believing the
   variable decided something.

   The `(universe)` dependency says "the state of the world decides this action", so dune reruns the
   gate on every invocation, and running it is the whole check: the check lives in `Utils`' own
   initialization, so a rejected spelling kills this process before `main` and the suite goes red
   naming the variable.

   ONE gate per directory rather than one for the repository, because dune aliases are per
   directory: `@test/einsum/runtest` does not build anything under `test/operations`, and scoping a
   run to a directory alias is what AGENTS.md tells people to do. A gate in the wrong directory
   protects nothing.

   `(universe)` rather than redeclaring the spellings: the honest dependency is the environment
   itself, and enumerating it would be 112 keys per directory, a consistency check to keep each list
   whole, and an exemption in `env_var_deps` for declaring names nothing reads -- to buy a narrower
   guarantee than the one line gives. It also covers what the old per-stanza declarations never did:
   those named `backend` (218 stanzas) and eight other keys once each, so a lowercase
   `ocannl_virtualize_max_visits` was served stale everywhere even before gh-ocannl-652.

   Silent when it passes: it runs in every test directory on every invocation, and a line per
   directory per run is noise in exchange for nothing. `Verdict.claim` reports on stderr and exits
   nonzero, so there is no golden here to promote a failure into. *)

open Base

let () =
  (* Unreachable with a rejected spelling set -- `Utils`' initializer exits first -- which is the
     claim: the two must not be able to disagree. It is stated rather than assumed so that making
     the abort conditional some day fails here instead of silently reopening the hole. Each rejected
     spelling is a failure of its own, naming the variable: a single claim that the list is empty
     would be the vacuous-quantifier shape verdict_ratchet refuses, and an ambient environment
     legitimately has nothing in it to be rejected. *)
  let fatal = List.filter (Utils.unread_env_vars ()) ~f:(fun (_, is_fatal, _) -> is_fatal) in
  List.iter fatal ~f:(fun (name, _, reason) ->
      Verdict.fail
        (Printf.sprintf "ambient environment variable %s is a rejected spelling of a known key: %s"
           name reason))
