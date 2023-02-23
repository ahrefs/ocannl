open Base
(* Currently, Ocannl contains Debug_runtime. *)
open Ocannl
type nonrec int = int [@@deriving sexp]
let%minidebug foo (x: int): int = let y: int = x + 1 in 2 * y
let () = ignore foo