open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Tensor-literal axis labelling (task-4eb929b2 AC#6): a type-annotation on an axis container inside
   a tensor literal sets that axis's dimension basis. [([…] : rgb)] labels an output (list) axis,
   [((…) : feat)] an input (tuple) axis, [([|…|] : bt)] a batch (array) axis. Multi-character tags
   work (unlike the removed char-literal form). We assert the label survives shape inference. *)

(* The basis label is set on the dimension at construction (via [make_axes]/[get_dim ~basis]), so
   [Shape.to_bases] reflects it directly — no forward pass needed (a bare constant literal is not a
   forward root anyway). *)
let check name t ~expect =
  try
    let bases = Shape.to_bases t.Tensor.shape in
    let joined = String.concat_array ~sep:"," bases in
    if Array.mem bases expect ~equal:String.equal then
      Stdio.printf "%s: PASS (bases=[%s] includes %S)\n" name joined expect
    else Stdio.printf "%s: FAIL (bases=[%s], expected %S)\n" name joined expect
  with Row.Shape_error (msg, _) -> Stdio.printf "%s: FAIL Shape_error: %s\n" name msg

let () =
  (* Output (list) axis labelled with a multi-character tag. *)
  let%op rgb_vec = ([ 1.; 2.; 3. ] : rgb) in
  check "output-list label" rgb_vec ~expect:"rgb";

  (* Input (tuple) axis labelled, nested inside an output list. *)
  let%op feat_row = [ ((1., 2., 3.) : features) ] in
  check "input-tuple label" feat_row ~expect:"features";

  (* Batch (array) axis labelled; the inner output list is unannotated (default). *)
  let%op bt = ([| [ 1.; 2. ]; [ 3.; 4. ] |] : examples) in
  check "batch-array label" bt ~expect:"examples";

  (* A user-written reserved tag is honored as a basis. *)
  let%op stretchy = ([ 1. ] : bcast_if_1) in
  check "explicit bcast_if_1 label" stretchy ~expect:"bcast_if_1"
