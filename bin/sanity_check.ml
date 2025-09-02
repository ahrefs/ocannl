open! Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
open! Ocannl.Operation.DSL_modules
module CDSL = Train.CDSL

module type Backend = Ir.Backend_intf.Backend

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  Stdio.printf "Hello, world %d!\n%!" (Backend.get_device ~ordinal:0).device_id
