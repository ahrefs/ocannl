open Base
module Ops = Ir.Ops
module Utils = Utils

let () =
  (* Test with default large_models=false, should use uint32 *)
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With large_models=false, index precision: %s\n" (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint32");

  (* Test with large_models=true, should use uint64 *)
  let old_setting = Utils.settings.large_models in
  Utils.settings.large_models <- true;
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With large_models=true, index precision: %s\n" (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint64");

  (* Restore the old setting *)
  Utils.settings.large_models <- old_setting;

  Stdio.printf "Index precision tests passed!\n"
