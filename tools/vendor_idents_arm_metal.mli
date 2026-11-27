(** Interface of the Metal arm of [tools/vendor_idents.ml] (gh-ocannl-1338, gh-ocannl-2338). Both
    the real arm and its [.missing] fallback satisfy this signature, so the stand-in library in
    [arrayjit/vendor_stubs/arms/dune] checks the real arm on its own against what [vendor_idents.ml]
    expects. *)

val built : bool
val probe : unit -> Ir.C_syntax.routine_name_probe
val include_dirs : unit -> string list
val capabilities : unit -> (string * string) list
