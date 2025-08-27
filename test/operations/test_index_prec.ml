open Base
module Ops = Ir.Ops
module Utils = Utils

let () =
  (* Test with default big_models=false, should use uint32 *)
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With big_models=false, index precision: %s\n" 
    (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint32");
  
  (* Test with big_models=true, should use uint64 *)
  let old_setting = Utils.settings.big_models in
  Utils.settings.big_models <- true;
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With big_models=true, index precision: %s\n" 
    (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint64");
  
  (* Restore the old setting *)
  Utils.settings.big_models <- old_setting;
  
  Stdio.printf "Index precision tests passed!\n"