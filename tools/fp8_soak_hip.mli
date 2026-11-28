(** Interface of the HIP arm of [tools/fp8_soak.ml] (gh-ocannl-657, gh-ocannl-2338). Matches
    [Fp8_soak.ARM], which [fp8_soak.ml] enforces when packing [(module Fp8_soak_hip : ARM)], so the
    stand-in library in [arrayjit/vendor_stubs/arms/dune] checks the real arm on its own. *)

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

val last_compiled : string
val built : bool
val device_count : unit -> (int, string) Stdlib.result
val set_arch_policy : [ `Device | `Backend ] -> unit
val device_report : unit -> (string * string) list
val compile_options : unit -> string list
val kernel_macros : unit -> (string * int) list
val narrow_f32 : spelling:[ `Raw | `Guarded ] -> base:int -> count:int -> bytes_buf -> unit

val narrow_f64 :
  spelling:[ `Raw | `Guarded ] -> base:int -> count:int -> lows:int array -> bytes_buf -> unit
