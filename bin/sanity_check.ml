open! Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

let () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  Utils.set_log_level 2;
  Stdio.print_endline "Hello, world!"
