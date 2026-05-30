open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Tensor-literal axis labelling (task-4eb929b2 AC#6): a type-annotation on an axis container inside
   a tensor literal sets that axis's dimension basis. [([…] : rgb)] labels an output (list) axis,
   [((…) : feat)] an input (tuple) axis, [([|…|] : bt)] a batch (array) axis. Multi-character tags
   work (unlike the removed char-literal form).

   We assert the label lands on the CORRECT row (batch / input / output) with the exact expected
   basis sequence — not merely that the tag appears somewhere in the flattened list. The basis is
   set on the dimension at construction (via [make_axes]/[get_dim ~basis]), so [Shape.to_bases_bio]
   reflects it directly; no forward pass needed (a bare constant literal is not a forward root). *)

let str a = String.concat_array ~sep:"," a
let eq a b = Array.equal String.equal a b

let check name t ~batch ~input ~output =
  try
    let b, i, o = Shape.to_bases_bio t.Tensor.shape in
    if eq b batch && eq i input && eq o output then
      Stdio.printf "%s: PASS (batch=[%s] input=[%s] output=[%s])\n" name (str b) (str i) (str o)
    else
      Stdio.printf
        "%s: FAIL\n  got      batch=[%s] input=[%s] output=[%s]\n  expected batch=[%s] input=[%s] \
         output=[%s]\n"
        name (str b) (str i) (str o) (str batch) (str input) (str output)
  with Row.Shape_error (msg, _) -> Stdio.printf "%s: FAIL Shape_error: %s\n" name msg

let () =
  (* Output (list) axis labelled with a multi-character tag; only the output row carries it. *)
  let%op rgb_vec = ([ 1.; 2.; 3. ] : rgb) in
  check "output-list label" rgb_vec ~batch:[||] ~input:[||] ~output:[| "rgb" |];

  (* Input (tuple) axis labelled, nested inside an output list; the label lands on the INPUT row,
     and the enclosing size-1 output list is the unannotated [default]. *)
  let%op feat_row = [ ((1., 2., 3.) : features) ] in
  check "input-tuple label" feat_row ~batch:[||] ~input:[| "features" |] ~output:[| "default" |];

  (* Batch (array) axis labelled; the inner output list is unannotated. The label lands on BATCH. *)
  let%op bt = ([| [ 1.; 2. ]; [ 3.; 4. ] |] : examples) in
  check "batch-array label" bt ~batch:[| "examples" |] ~input:[||] ~output:[| "default" |];

  (* A user-written reserved tag is honored as a basis, on the output row. *)
  let%op stretchy = ([ 1. ] : bcast_if_1) in
  check "explicit bcast_if_1 label" stretchy ~batch:[||] ~input:[||] ~output:[| "bcast_if_1" |];

  (* Two output axes (nested lists), only the OUTER labelled: outer→rows, inner→default. *)
  let%op mat = ([ [ 1.; 2. ]; [ 3.; 4. ]; [ 5.; 6. ] ] : rows) in
  check "outer-of-two-output labelled" mat ~batch:[||] ~input:[||] ~output:[| "rows"; "default" |]
