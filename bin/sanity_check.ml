open! Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module PDSL = Operation.PDSL
module CDSL = Train.CDSL

module type Backend = Ir.Backend_intf.Backend

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  Stdio.printf "Hello, world %d!\n%!" (Backend.get_device ~ordinal:0).device_id
