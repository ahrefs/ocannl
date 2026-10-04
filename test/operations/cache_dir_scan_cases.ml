(** How the directory reader behind [cache_dir_ignores] resolves a name, exercised on input built to
    break it rather than on whatever the repository happens to spell today.

    [Cache_dir_ignores] scans every source in the tree, so what it exercises is the set of spellings
    that are currently in use — and the reader deliberately covers more than that, because the point
    of the prefix rule is that a tuning test written tomorrow is ignored the day it is written. An
    arm covering a spelling no source uses yet is invisible to that census: it can stop working, or
    stop compiling, without a single check going red.

    That is not hypothetical for the one arm below that resolves a module alias bound in EXPRESSION
    position. The two ways the reader can be wrong are both quiet: a use it fails to recognise
    shrinks the census rather than failing it, and an alias it fails to resolve turns a cache write
    into an ordinary [~dir] it never looks at.

    So each spelling is pinned here on a snippet of its own, and the resolutions the census cannot
    reach — the empty string, a forwarded parameter, an expression — are pinned beside the ones it
    can. *)

open Base
module Scan = Test_utils.Cache_dir_scan

let printf = Test_utils.Refusal_control_manifest.printf

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

(* Each case is a source and the uses the reader should find in it, rendered as the argument was
   spelled followed by what it resolves to. *)
let cases =
  [
    ( "a literal at the call site",
      {ocaml|let () = Autotune.tune ~cache_dir:"autotune_cache_x" f|ocaml},
      [ "~cache_dir names autotune_cache_x" ] );
    ( "a literal reached through a binding",
      {ocaml|let go () =
  let cache_dir = "autotune_cache_y" in
  Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names autotune_cache_y" ] );
    ( "the empty string turns the cache off",
      {ocaml|let () = Autotune.tune ~cache_dir:"" f|ocaml},
      [ "~cache_dir disables the cache" ] );
    ( "a parameter is forwarded, and named at its own call sites",
      {ocaml|let run ~cache_dir () = Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir forwards the parameter cache_dir" ] );
    ( "anything else is reported rather than assumed harmless",
      {ocaml|let () = Autotune.tune ~cache_dir:(prefix ^ suffix) f|ocaml},
      [ "~cache_dir names an expression" ] );
    ( "an unresolved name is reported by name",
      {ocaml|let () = Autotune.tune ~cache_dir:elsewhere f|ocaml},
      [ "~cache_dir names `elsewhere`" ] );
    ( "a parameter shadows an earlier literal",
      {ocaml|let cache_dir = "autotune_cache_safe"
let run ~cache_dir = Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir forwards the parameter cache_dir" ] );
    ( "same-named literals remain in separate scopes",
      {ocaml|let a () = let cache_dir = "autotune_cache_a" in Autotune.tune ~cache_dir f
let b () = let cache_dir = "autotune_cache_b" in Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names autotune_cache_a"; "~cache_dir names autotune_cache_b" ] );
    ( "a nonliteral let shadows an earlier literal",
      {ocaml|let cache_dir = "autotune_cache_safe"
let a () = let cache_dir = compute () in Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "qualified values never borrow a local literal",
      {ocaml|let cache_dir = "autotune_cache_safe"
let () = Autotune.tune ~cache_dir:Other.cache_dir f|ocaml},
      [ "~cache_dir names `Other.cache_dir`" ] );
    ( "an alias shadowed by another module loses its cache identity",
      {ocaml|module Cache = Ir.Schedule_cache
module Cache = Other
let () = Cache.store ~dir:"scratch" k v|ocaml},
      [] );
    ( "a nested alias dies with its scope",
      {ocaml|let () = let module Cache = Ir.Schedule_cache in Cache.store ~dir:"autotune_cache_x" k v
let () = Cache.store ~dir:"scratch" k v|ocaml},
      [ "~dir names autotune_cache_x" ] );
    ( "a functor parameter shadows the cache module",
      {ocaml|module F (Schedule_cache : S) = struct let () = Schedule_cache.store ~dir:"scratch" k v end|ocaml},
      [] );
    ( "an unpack shadows the cache module",
      {ocaml|let f (module Schedule_cache : S) = Schedule_cache.store ~dir:"scratch" k v|ocaml},
      [] );
    ( "the cache resolver forwards a lexical parameter",
      {ocaml|let f ?cache_dir () =
let cache_dir = Autotune.resolve_cache_dir ?cache_dir ~search:true () in
Ir.Schedule_cache.store ~dir:cache_dir k v|ocaml},
      [ "~cache_dir forwards the parameter cache_dir"; "~dir forwards the parameter cache_dir" ] );
    ( "a same-named resolver in a shadowing module proves nothing",
      {ocaml|module Autotune = Other
let f ?cache_dir () =
let cache_dir = Autotune.resolve_cache_dir ?cache_dir ~search:true () in
Ir.Schedule_cache.store ~dir:cache_dir k v|ocaml},
      [ "~cache_dir forwards the parameter cache_dir"; "~dir names `cache_dir`" ] );
    ( "an alias of an unchecked positional parameter remains unresolved",
      {ocaml|let run dir = let cache_dir = dir in Autotune.tune ~cache_dir f
let () = run "scratch"|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "a positional parameter named cache_dir remains unresolved",
      {ocaml|let run cache_dir = Autotune.tune ~cache_dir f
let () = run "scratch"|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "a renamed censused labelled parameter is forwarded",
      {ocaml|let run ~cache_dir:dir = Autotune.tune ~cache_dir:dir f|ocaml},
      [ "~cache_dir forwards the parameter dir" ] );
    ( "an alias of a censused labelled parameter is forwarded",
      {ocaml|let run ~cache_dir:dir = let cache_dir = dir in Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir forwards the parameter cache_dir" ] );
    ( "a case binder is unresolved rather than a forwarded function parameter",
      {ocaml|let cache_dir = "autotune_cache_safe"
let f x = match x with cache_dir -> Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "a for index is unresolved rather than a forwarded function parameter",
      {ocaml|let cache_dir = "autotune_cache_safe"
let () = for cache_dir = 0 to 1 do Autotune.tune ~cache_dir f done|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "a recursive alias never borrows the outer literal",
      {ocaml|let cache_dir = "autotune_cache_safe"
let f () = let rec cache_dir = cache_dir in Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "class constructor parameters are unresolved because instantiations are not censused",
      {ocaml|class runner cache_dir = object method run = Autotune.tune ~cache_dir f end
let () = new runner "scratch"|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    ( "a literal inside a class remains resolved",
      {ocaml|class runner = let cache_dir = "autotune_cache_class" in object method run = Autotune.tune ~cache_dir f end|ocaml},
      [ "~cache_dir names autotune_cache_class" ] );
    (* The direct-store spelling, whose `~dir` is told from every other `~dir` in the repository
       only by the module it is called through. *)
    ( "a direct store through a structure-level alias",
      {ocaml|module SC = Ir.Schedule_cache
let () = SC.store ~dir:"autotune_cache_z" key value|ocaml},
      [ "~dir names autotune_cache_z" ] );
    ( "a direct store through the qualified path",
      {ocaml|let () = Ir.Schedule_cache.store ~dir:"autotune_cache_q" key value|ocaml},
      [ "~dir names autotune_cache_q" ] );
    (* The arm the census cannot reach: no source in the tree binds the module this way today. It is
       also the arm whose spelling the compiler moved under it -- 5.5 represents `let module M = …
       in …` as a structure item inside the expression rather than as `Pexp_letmodule` -- which is
       why the reader works on ppxlib's parse tree, where the construct has one spelling on every
       compiler the opam files admit. *)
    ( "a direct store through an alias bound in expression position",
      {ocaml|let go () =
  let module Cache = Ir.Schedule_cache in
  Cache.store ~dir:"autotune_cache_e" key value|ocaml},
      [ "~dir names autotune_cache_e" ] );
    ( "an alias of an alias is an alias",
      {ocaml|module SC = Ir.Schedule_cache
module Cache = SC
let () = Cache.store ~dir:"autotune_cache_c" key value|ocaml},
      [ "~dir names autotune_cache_c" ] );
    (* Not every `~dir` is a cache write: inside `schedule_cache.ml` itself the directory is a
       parameter, named by whoever called in. *)
    ( "a bare store is not a call into the cache module",
      {ocaml|let () = store ~dir:"scratch" key value|ocaml},
      [] );
    ( "nor is a same-named operation on another module",
      {ocaml|let () = Other.store ~dir:"scratch" key value|ocaml},
      [] );
    (* `Schedule_cache.ensure_dir` walks the path it is handed, so a name is not a directory name
       merely by starting with the prefix. The reader reports it; the check applies the rule. *)
    ( "a traversal keeps its separators, for the prefix rule to reject",
      {ocaml|let () = Autotune.tune ~cache_dir:"autotune_cache/../leaked_cache" f|ocaml},
      [ "~cache_dir names autotune_cache/../leaked_cache" ] );
  ]

(* The other half of what the check reads: the built-in default, which no source names and which is
   therefore read out of the library that defines it. Its second read asks merely whether the key
   was set and defaults to the empty string, so an empty default is not a directory. *)
let default_cases =
  [
    ( "the default a search falls back to",
      {ocaml|let d = get ~arg_name:"autotune_cache_dir" ~default:"autotune_cache"|ocaml},
      [ "autotune_cache" ] );
    ( "an empty default names no directory",
      {ocaml|let set = get ~arg_name:"autotune_cache_dir" ~default:""|ocaml},
      [] );
    ( "another key's default is not this one's",
      {ocaml|let d = get ~arg_name:"backend" ~default:"autotune_cache"|ocaml},
      [] );
  ]

(* The ignore matcher's own reading of a gitignore pattern, on the forms the repository's rules use
   and on the ones next to them. Every expectation here is what `git check-ignore` answers, run
   against a scratch repository carrying exactly these patterns -- not what the matcher happens to
   do, which is the whole point: a matcher that has drifted from git reports a directory ignored
   that is not, or unreadable that is.

   The character-class arms arrived with gh-ocannl-780, whose staging-file rule is spelled to the
   generated shape (a nonce of exactly sixteen hex digits) so that it cannot also hide a file
   someone named `report.ocannl-stage.backup`. The last arm is the reading that measurement
   CORRECTED: an unterminated `[` is not a literal bracket, and a pattern carrying one matches
   nothing at all. *)
let glob_cases =
  [
    ("a range class", "a.[0-9a-f]", "a.a", true);
    ("a range class rejects outside its range", "a.[0-9a-f]", "a.g", false);
    ("a negated class", "b.[!0-9]", "b.a", true);
    ("a negated class rejects its members", "b.[!0-9]", "b.5", false);
    ("a bracket first in a class is a member", "c.[]a]", "c.]", true);
    ("the rest of that class still applies", "c.[]a]", "c.a", true);
    ("an escape inside a class", "e.[\\-]", "e.-", true);
    ("an escaped member excludes others", "e.[\\-]", "e.x", false);
    ("an unterminated class matches nothing, not a literal bracket", "d.[abc", "d.[abc", false);
    ( "the staging-file rule matches a generated name",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      "model.bin.ocannl-stage.00001092.00000000.00c0ffee00c0ffee",
      true );
    ( "and spares a file merely carrying the infix",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      "report.ocannl-stage.backup",
      false );
    (* `[0-9]*` for a variable-width field would have matched this one: git reads it as one digit
       followed by anything at all, which is why the staging name's fields are fixed-width. *)
    ( "and spares fields that merely start numeric",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      "x.ocannl-stage.1abc.2bar.00c0ffee00c0ffee",
      false );
    ( "and spares fields of the wrong width",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      "x.ocannl-stage.1092.0.00c0ffee00c0ffee",
      false );
    (* The leading `?*` is one-or-more: a bare `*` matches the empty stem too, and a name with no
       stem is one the helper cannot generate and so somebody else's file. *)
    ( "a one-character stem is still the helper's",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      "x.ocannl-stage.00001092.00000000.00c0ffee00c0ffee",
      true );
    ( "and a name with no stem at all is not",
      "?*.ocannl-stage.[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]",
      ".ocannl-stage.00000001.00000000.0123456789abcdef",
      false );
  ]

let render uses = String.concat ~sep:"; " uses

let () =
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try
          List.map (Scan.read source).Scan.uses ~f:(fun use ->
              use.Scan.spelling ^ " " ^ Scan.describe use.Scan.resolution)
        with _ ->
          fail "use -- %s: the snippet does not parse" name;
          []
      in
      if List.equal String.equal found expected then printf "ok: use -- %s\n" name
      else fail "use -- %s: expected [%s], found [%s]" name (render expected) (render found));
  List.iter glob_cases ~f:(fun (name, pattern, candidate, expected) ->
      let found = Scan.glob_matches pattern candidate in
      if Bool.equal found expected then printf "ok: glob -- %s\n" name
      else
        fail "glob -- %s: `%s` against %s expected %b, found %b" name pattern candidate expected
          found);
  List.iter default_cases ~f:(fun (name, source, expected) ->
      let found = (Scan.read source).Scan.builtin_defaults in
      if List.equal String.equal found expected then printf "ok: built-in default -- %s\n" name
      else
        fail "built-in default -- %s: expected [%s], found [%s]" name (render expected)
          (render found));
  Test_utils.Refusal_control_manifest.print "cache_dir_ignores.ml"
