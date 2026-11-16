(* A pinned GAP, not a positive control: compiler-libs.toplevel shares +compiler-libs with
   compiler-libs.common, so this use of Location credits both and the unused toplevel declaration is
   not reported (see unused_compiler_deps.ml's header). *)
let loc = Location.none
