open! Base
open! Stdio

(* Direct unit tests for Arrayjit.Tnode.get_debug_name consecutive-collapse behaviour.
   No tensor creation needed — we call get_debug_name directly. *)
let gdn label = Ir.Tnode.get_debug_name ~id:0 ~label ()

let check desc expected label =
  let got = gdn label in
  if String.equal got expected then printf "%s: PASS (%s)\n" desc got
  else printf "%s: FAIL\n  expected %s\n  got      %s\n" desc expected got

let () =
  check "single component, no suffix" "attention" [ "attention" ];
  check "two consecutive, suffix 2" "attention2" [ "attention"; "attention" ];
  check "three consecutive, suffix 3" "attention3" [ "attention"; "attention"; "attention" ];
  check "mixed: encoder + attention×2 + output" "encoder_attention2_output"
    [ "encoder"; "attention"; "attention"; "output" ];
  check "non-consecutive duplicates unchanged" "a_b_a" [ "a"; "b"; "a" ];
  check "empty label, falls back to n0" "n0" [];
  check "all-same long run" "foo5" [ "foo"; "foo"; "foo"; "foo"; "foo" ];
  check "singleton after run: a×2 then b" "a2_b" [ "a"; "a"; "b" ];
  check "run then singleton: a then b×2" "a_b2" [ "a"; "b"; "b" ]
