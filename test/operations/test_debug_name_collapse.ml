open! Base
open! Stdio

(* Direct unit tests for Arrayjit.Tnode.get_debug_name consecutive-collapse behaviour. No tensor
   creation needed — we call get_debug_name directly. *)
let gdn label = Ir.Tnode.get_debug_name ~id:0 ~label ()

let check desc expected label =
  let got = gdn label in
  if String.equal got expected then printf "%s: PASS (%s)\n" desc got
  else printf "%s: FAIL\n  expected %s\n  got      %s\n" desc expected got

let () =
  check "single component, no suffix" "attention" [ "attention" ];
  check "two consecutive, suffix 2" "attention2" [ "attention"; "attention" ];
  check "three consecutive, suffix 3" "attention3" [ "attention"; "attention"; "attention" ];
  check "mixed: encoder + attention x2 + output" "encoder_attention2_output"
    [ "encoder"; "attention"; "attention"; "output" ];
  check "non-consecutive duplicates unchanged" "a_b_a" [ "a"; "b"; "a" ];
  check "empty label, falls back to n0" "n0" [];
  check "all-same long run" "foo5" [ "foo"; "foo"; "foo"; "foo"; "foo" ];
  check "singleton after run: a x2 then b" "a2_b" [ "a"; "a"; "b" ];
  check "run then singleton: a then b x2" "a_b2" [ "a"; "b"; "b" ];
  (* AC5: filter-before-collapse — non-alphanum_ component "-" is removed first, turning originally
     non-consecutive foo's into a consecutive run of 3. *)
  check "filter-before-collapse: foo,-,foo,foo -> foo3" "foo3" [ "foo"; "-"; "foo"; "foo" ];
  (* AC5: grad-strip-before-collapse — leading "grad" is peeled off (sets is_grad), leaving
     ["grad";"grad"] which collapses to "grad2"; then .grad is appended. *)
  check "grad-strip-before-collapse: grad,grad,grad -> grad2.grad" "grad2.grad"
    [ "grad"; "grad"; "grad" ]
