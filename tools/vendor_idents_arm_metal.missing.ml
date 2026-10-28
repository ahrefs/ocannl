(* The stand-in for the Metal arm of tools/vendor_idents.ml on a box without the metal bindings: the
   harvest still builds, and asking for this arm says so rather than failing to link. Nothing below
   [built] is reached: an unbuilt arm is refused before it is asked anything. *)

let built = false
let unavailable () = failwith "vendor_idents: the Metal arm is not built on this box"
let probe () : Ir.C_syntax.routine_name_probe = unavailable ()
let include_dirs () : string list = unavailable ()
let capabilities () : (string * string) list = unavailable ()
