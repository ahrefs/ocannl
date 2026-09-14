(* gh-ocannl-917: canonical precision witnesses and their arithmetic families are owned by Ops. The
   golden records the policy while the claims pin the links to serialization, GADT witnesses, and
   the actual C-family renderer sweep. *)
open Base
module Ops = Ir.Ops
open Verdict.Claims

let () =
  p_pairwise_distinct "precision enumeration has no duplicates" Ops.all_precs ~equal:Ops.equal_prec
    ~to_string:Ops.prec_string;
  p_all ~min:13 "canonical precisions round-trip through serialization" Ops.all_precs
    ~f:(fun prec -> Ops.equal_prec prec (Ops.prec_of_sexp (Ops.sexp_of_prec prec)));
  p_all ~min:12 "storage precisions carry their canonical GADT witnesses" Ops.storage_precs
    ~f:(fun prec -> Ops.equal_prec prec (Ops.apply_prec { f = Ops.pack_prec } prec));
  p_all2 "C-family renderer sweep covers every storage precision"
    (Array.of_list Ir.C_syntax.all_precs)
    (Array.of_list Ops.storage_precs) ~f:Ops.equal_prec;
  p "void has no arithmetic or storage value"
    (match Ops.prec_family Ops.Void_prec with
    | No_value ->
        (not (Ops.is_integer Ops.Void_prec || Ops.is_float Ops.Void_prec))
        && not (List.mem Ops.storage_precs Ops.Void_prec ~equal:Ops.equal_prec)
    | _ -> false);
  p "uint4x32 is packed integer storage outside scalar arithmetic"
    (match Ops.prec_family Ops.uint4x32 with
    | Packed_integer ->
        (not (Ops.is_integer Ops.uint4x32 || Ops.is_float Ops.uint4x32))
        && List.mem Ops.storage_precs Ops.uint4x32 ~equal:Ops.equal_prec
        && not (List.mem Ops.scalar_precs Ops.uint4x32 ~equal:Ops.equal_prec)
    | _ -> false);
  let show label precs =
    Stdio.printf "%s: %s\n" label (String.concat ~sep:", " (List.map precs ~f:Ops.prec_string))
  in
  show "all" Ops.all_precs;
  show "scalar" Ops.scalar_precs;
  show "integer" Ops.integer_precs;
  show "float" Ops.float_precs
