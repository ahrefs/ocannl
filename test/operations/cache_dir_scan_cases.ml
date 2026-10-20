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
module Attempt = Test_utils.Scan_attempt

let printf = Stdio.printf

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
    (* A later top-level [module Cache] beside an earlier one is a duplicate definition no source
       can contain (gh-ocannl-1248), so the shadowing is spelled with [let module]. *)
    ( "an alias shadowed by another module loses its cache identity",
      {ocaml|module Cache = Ir.Schedule_cache
let () = let module Cache = Other in Cache.store ~dir:"scratch" k v|ocaml},
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
let tune_each ~cache_dir:(_ : int) = ()
let () = for cache_dir = 0 to 1 do tune_each ~cache_dir done|ocaml},
      [ "~cache_dir names `cache_dir`" ] );
    (* [let rec cache_dir = cache_dir] would be the direct spelling, and no compiler accepts it: an
       identifier is never a legal [let rec] right-hand side. A sibling function reaching the
       recursive binding is. *)
    ( "a recursive binding never borrows the outer literal",
      {ocaml|let cache_dir = "autotune_cache_safe"
let f () =
  let rec go () = Autotune.tune ~cache_dir f and cache_dir = "scratch" in
  go ()|ocaml},
      [ "~cache_dir names scratch" ] );
    ( "class constructor parameters are unresolved because instantiations are not censused",
      {ocaml|class runner cache_dir = object method run = Autotune.tune ~cache_dir f end
let _ = new runner "scratch"|ocaml},
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
    (* A structure carries its named module bindings out with it, so an alias exported from one is
       reached through its qualified path (gh-ocannl-1206). Each positive arm stands beside the
       same-named binding that must NOT be taken for it: what the export path reaches is the binding
       of that name in that structure, last one winning, never another structure's or the file's. *)
    ( "a direct store through an alias exported from a structure",
      {ocaml|module M = struct module Cache = Ir.Schedule_cache end
let () = M.Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "a same-named export of an unrelated module is not the cache",
      {ocaml|module M = struct module Cache = Other end
let () = M.Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "an export path reaches its own structure, not a same-named alias elsewhere",
      {ocaml|module Cache = Ir.Schedule_cache
module N = struct module Cache = Ir.Schedule_cache end
module M = struct module Cache = Other end
let () = M.Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    (* OCaml lets a later binding in a structure take a name over only from an [include], so the
       shadowing arms are spelled that way: a duplicate definition is a program no source can
       contain. *)
    ( "a later binding in the structure shadows an included alias",
      {ocaml|module N = struct module Cache = Ir.Schedule_cache end
module M = struct include N module Cache = Other end
let () = M.Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "and a later alias shadows an included unrelated module",
      {ocaml|module N = struct module Cache = Other end
module M = struct include N module Cache = Ir.Schedule_cache end
let () = M.Cache.store ~dir:"autotune_cache_s" key value|ocaml},
      [ "~dir names autotune_cache_s" ] );
    ( "a rebinding of the structure's own name takes its exports away",
      {ocaml|module M = struct module Cache = Ir.Schedule_cache end
let () = let module M = Other in M.Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "a recursive module shadows an included alias",
      {ocaml|module N = struct module Cache = Ir.Schedule_cache end
module M = struct include N module rec Cache : S = Cache end
let () = M.Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "a binding in a structure sees the earlier ones beside it",
      {ocaml|module M = struct
  module C = Ir.Schedule_cache
  module Cache = C
end
let () = M.Cache.store ~dir:"autotune_cache_b" key value|ocaml},
      [ "~dir names autotune_cache_b" ] );
    ( "and the names an earlier open brought into it",
      {ocaml|module N = struct module C = Ir.Schedule_cache end
module M = struct open N module Cache = C end
let () = M.Cache.store ~dir:"autotune_cache_c2" key value|ocaml},
      [ "~dir names autotune_cache_c2" ] );
    ( "an export resolves in the structure's scope, through outer aliases",
      {ocaml|module SC = Ir.Schedule_cache
module M = struct module Cache = SC end
let () = M.Cache.store ~dir:"autotune_cache_m" key value|ocaml},
      [ "~dir names autotune_cache_m" ] );
    ( "exports carry through nested structures and signature constraints",
      {ocaml|module type S = sig module N : sig module Cache : module type of Ir.Schedule_cache end end
module M : S = struct module N = struct module Cache = Ir.Schedule_cache end end
let () = M.N.Cache.store ~dir:"autotune_cache_n" key value|ocaml},
      [ "~dir names autotune_cache_n" ] );
    ( "a structure bound in expression position exports too",
      {ocaml|let go () =
  let module M = struct module Cache = Ir.Schedule_cache end in
  M.Cache.store ~dir:"autotune_cache_l" key value|ocaml},
      [ "~dir names autotune_cache_l" ] );
    ( "an included structure's exports are the includer's",
      {ocaml|module N = struct module Cache = Ir.Schedule_cache end
module M = struct include N end
let () = M.Cache.store ~dir:"autotune_cache_i" key value|ocaml},
      [ "~dir names autotune_cache_i" ] );
    (* A structure whose body INCLUDES a module reaches that module's own contents through its
       qualified path, so including the cache module makes the structure a cache module
       (gh-ocannl-1248): its stores are cache writes, whatever else the structure defines. *)
    ( "a structure including the cache module is the cache module",
      {ocaml|module Cache = struct include Ir.Schedule_cache end
let () = Cache.store ~dir:"autotune_cache_inc" key value|ocaml},
      [ "~dir names autotune_cache_inc" ] );
    ( "and stays one beside its own definitions",
      {ocaml|module Cache = struct
  include Ir.Schedule_cache
  module Extra = Other
  let extra = 1
end
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "through an alias of the cache module, a nested include and a signature constraint",
      {ocaml|module SC = Ir.Schedule_cache
module Cache : S = struct include struct include SC end end
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "including Ir makes its Schedule_cache reachable through the structure",
      {ocaml|module My_ir = struct include Ir end
let () = My_ir.Schedule_cache.store ~dir:"autotune_cache_ir" key value|ocaml},
      [ "~dir names autotune_cache_ir" ] );
    ( "an included module's exports still win over the structure's base",
      {ocaml|module M = struct include Ir module Schedule_cache = Other end
let () = M.Schedule_cache.store ~dir:"scratch" key value|ocaml},
      [] );
    (* A later include shadows an earlier one's values only where it defines the same name, so every
       recognized include stays a base (review round 2). *)
    ( "a later include of another named module keeps the cache's values",
      {ocaml|module M = struct include Ir.Schedule_cache include Ir end
let () = M.store ~dir:"scratch" key value
let () = M.Schedule_cache.store ~dir:"autotune_cache_both" key value|ocaml},
      [ "~dir names scratch"; "~dir names autotune_cache_both" ] );
    (* A value the structure binds itself wins over its includes, as a module name does (review
       round 3): a local [store] is not a cache write, unless a later include of the cache takes the
       name back. *)
    ( "a structure's own value shadows the included cache's",
      {ocaml|module M = struct
  include Ir.Schedule_cache
  let store ~dir _ _ = ignore (dir : string)
end
let () = M.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "an alias of the included cache's value is still the cache's",
      {ocaml|module M = struct
  include Ir.Schedule_cache
  let save = store
end
let () = M.save ~dir:"autotune_cache_alias" key value|ocaml},
      [ "~dir names autotune_cache_alias" ] );
    ( "but an alias of another module's value shadows",
      {ocaml|module M = struct
  include Ir.Schedule_cache
  let save = Other.store
end
let () = M.save ~dir:"scratch" key value|ocaml},
      [] );
    ( "and so does an alias of the structure's own value",
      {ocaml|module M = struct
  include Ir.Schedule_cache
  let store ~dir _ _ = ignore (dir : string)
  let save = store
end
let () = M.save ~dir:"scratch" key value|ocaml},
      [] );
    ( "and a later include of the cache takes the name back",
      {ocaml|module M = struct
  let store ~dir _ _ = ignore (dir : string)
  include Ir.Schedule_cache
end
let () = M.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "an include of a namespace of modules leaves the structure's own values",
      {ocaml|module M = struct
  include Ir.Schedule_cache
  let store ~dir _ _ = ignore (dir : string)
  include Ir
end
let () = M.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "opening a structure brings in the modules it inherits from an include",
      {ocaml|module Schedule_cache = Other
module M = struct include Ir end
open M
let () = Schedule_cache.store ~dir:"autotune_cache_inherited" key value|ocaml},
      [ "~dir names autotune_cache_inherited" ] );
    ( "a literal signature hides the modules an include brought in",
      {ocaml|module Schedule_cache = Other
module M : sig end = struct include Ir end
open M
let () = Schedule_cache.store ~dir:"scratch" key value|ocaml},
      [] );
    (* The forwarding resolver is certified only through the library's own module: a structure that
       includes it may define a [resolve_cache_dir] of its own (review round 3). *)
    ( "a resolver reached through a structure including Autotune certifies nothing",
      {ocaml|module A = struct
  include Autotune
  let resolve_cache_dir ?cache_dir ~search () = ignore (cache_dir, search); "scratch"
end
let f ?cache_dir () =
let cache_dir = A.resolve_cache_dir ?cache_dir ~search:true () in
Ir.Schedule_cache.store ~dir:cache_dir k v|ocaml},
      [ "~cache_dir forwards the parameter cache_dir"; "~dir names `cache_dir`" ] );
    ( "a structure including an unrelated module is not the cache",
      {ocaml|module Cache = struct include Other end
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "an open inside a structure is not an export",
      {ocaml|module N = struct module Cache = Other end
module M = struct module Cache = Ir.Schedule_cache open N end
let () = M.Cache.store ~dir:"autotune_cache_p" key value|ocaml},
      [ "~dir names autotune_cache_p" ] );
    ( "opening a structure brings its exported alias into scope",
      {ocaml|module M = struct module Cache = Ir.Schedule_cache end
open M
let () = Cache.store ~dir:"autotune_cache_o" key value|ocaml},
      [ "~dir names autotune_cache_o" ] );
    ( "and its unrelated export shadows an outer alias, locally opened too",
      {ocaml|module Cache = Ir.Schedule_cache
module M = struct module Cache = Other end
let () = M.(Cache.store ~dir:"scratch" key value)|ocaml},
      [] );
    (* A signature constraint decides what an [open] of the structure brings in: a name it hides
       does not shadow the outer alias. A literal signature is read exactly; one the scan cannot
       read may hide anything, so an unrelated export then shadows nothing, while a cache export
       stays and is reported. *)
    ( "a name a literal signature hides does not shadow on open",
      {ocaml|module Cache = Ir.Schedule_cache
module M : sig end = struct module Cache = Other end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "nor does the hidden cache alias in the mirror",
      {ocaml|module Cache = Other
module M : sig end = struct module Cache = Ir.Schedule_cache end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "a name a literal signature declares still shadows on open",
      {ocaml|module Cache = Ir.Schedule_cache
module M : sig module Cache : module type of Other end = struct module Cache = Other end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "an unreadable signature may hide an unrelated export, so it shadows nothing",
      {ocaml|module type S = sig end
module Cache = Ir.Schedule_cache
module M : S = struct module Cache = Other end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "a nested signature narrows the nested structure's exports",
      {ocaml|module Cache = Ir.Schedule_cache
module M : sig module N : sig end end = struct module N = struct module Cache = Other end end
open M.N
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "and an unreadable one may hide names at any depth",
      {ocaml|module type S = sig module N : sig end end
module Cache = Ir.Schedule_cache
module M : S = struct module N = struct module Cache = Other end end
open M.N
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    (* A [module X := P] in a signature is a LOCAL substitution: it names [P] for the items after it
       and declares nothing, nor does it take an earlier declaration of [X] away (checked against
       [ocamlc]; the arm this replaces read it as a removal, over a snippet no compiler accepts,
       gh-ocannl-1248). *)
    ( "a name a literal signature only substitutes is not declared",
      {ocaml|module Cache = Ir.Schedule_cache
module M : sig module Cache := Other end = struct module Cache = Other end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "and a substitution does not take an earlier declaration away",
      {ocaml|module Cache = Ir.Schedule_cache
module M : sig module Cache : module type of Other module Cache := Other end = struct module Cache = Other end
open M
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [] );
    ( "a structure an unreadable signature may hide does not shadow a later open",
      {ocaml|module type S = sig end
module N = struct module Cache = Ir.Schedule_cache end
module M : S = struct module N = struct module Cache = Other end end
open M
open N
let () = Cache.store ~dir:"scratch" key value|ocaml},
      [ "~dir names scratch" ] );
    ( "a resolver alias an unreadable signature may hide certifies nothing",
      {ocaml|module type S = sig end
module A = Autotune
module Autotune = Other
module M : S = struct module Autotune = A end
open M
let f ?cache_dir () =
let cache_dir = Autotune.resolve_cache_dir ?cache_dir ~search:true () in
Ir.Schedule_cache.store ~dir:cache_dir k v|ocaml},
      [ "~cache_dir forwards the parameter cache_dir"; "~dir names `cache_dir`" ] );
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
        Attempt.attempted ~what:"use" ~name ~default:[] (fun () ->
            List.map (Scan.read source).Scan.uses ~f:(fun use ->
                use.Scan.spelling ^ " " ^ Scan.describe use.Scan.resolution))
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
