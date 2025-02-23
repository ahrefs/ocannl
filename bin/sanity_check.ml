open! Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

module type Backend = Arrayjit.Backend_intf.Backend

let () =
  Rand.init 0;
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  Utils.set_log_level 2;
  Stdio.print_endline "Hello, world!"
