open Base
module Ops = Ir.Ops
module Utils = Utils

let () =
  (* Test with default large_models=false, should use uint32 *)
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With large_models=false, index precision: %s\n" (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint32");
  (* gh-ocannl-344: the Metal pooled slot table widens with the SAME setting -- 32-bit (uint) when
     large_models=false (offsets capped under 4 GB), 64-bit (ulong) when true (cap lifted, so a byte
     offset can exceed UINT32_MAX). This is the type the shader declares and the backend fills; if it
     stayed uint under large_models=true, large-model pool offsets would silently truncate. *)
  Stdio.printf "With large_models=false, pool slot type: %s\n" (Ir.C_syntax.pool_slot_msl_typ ());
  assert (String.equal (Ir.C_syntax.pool_slot_msl_typ ()) "uint");

  (* Test with large_models=true, should use uint64 *)
  let old_setting = Utils.settings.large_models in
  Utils.settings.large_models <- true;
  let idx_prec = Ops.index_prec () in
  Stdio.printf "With large_models=true, index precision: %s\n" (Ops.prec_string idx_prec);
  assert (String.equal (Ops.prec_string idx_prec) "uint64");
  Stdio.printf "With large_models=true, pool slot type: %s\n" (Ir.C_syntax.pool_slot_msl_typ ());
  assert (String.equal (Ir.C_syntax.pool_slot_msl_typ ()) "ulong");

  (* Restore the old setting *)
  Utils.settings.large_models <- old_setting;

  Stdio.printf "Index precision tests passed!\n"
