(* Regression test for the realized statistics of parameter initializers, measured via
   [Context.get_values] after [Train.forward_once].

   Fan-scaled initializers ([Nn_blocks.kaiming] / [xavier]) capture the fan from the raw init tensor
   with einsum row variables ([w_raw ++ "...|..i.. -> ... => |->0" ["i"]]). Before the
   compute_row_product open-tail fix, the captured row product latched at 1 while the param's rows
   were still open, so the multiplier was sqrt(scale_sq / 1) regardless of the actual fan — e.g. std
   1.41 instead of 0.141 for a fan-in-100 weight under [TDSL.default_param_init := NTDSL.kaiming
   ~scale_sq:2.0 TDSL.O.normal1]. The std checks below guard against that: they fail hard when the
   fan resolves to 1.

   The "unique values" lines guard against a second bug: for params with inferred shapes, the PRNG
   counter tensor ([range_over_offsets]) used to connect to the result only through pointwise
   broadcasts, so its shape closed smaller than the param and the random values repeated along the
   fan-in axis (50 unique values for a [100 -> 50] weight). The counter's shape is now pinned to the
   result's by equality specs on [threefry4x32], [uint4x32_to_prec_uniform1] and
   [Nn_blocks.box_muller], so these lines must report full uniqueness.

   The packed [uniform] (the default since gh-ocannl-509) ties its counter to the result by a
   round-up total rather than an equality spec, and had its own way to repeat values
   (gh-ocannl-1868): see [bigram_weight_values] below. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Stdio

let mean arr = Array.fold arr ~init:0. ~f:( +. ) /. Float.of_int (Array.length arr)

let std arr =
  let m = mean arr in
  let var =
    Array.fold arr ~init:0. ~f:(fun acc v -> acc +. ((v -. m) *. (v -. m)))
    /. Float.of_int (Array.length arr)
  in
  Float.sqrt var

let unique arr = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list arr))

let check ~name ~expected ~tol measured =
  Verdict.pass_fail
    (Printf.sprintf "%s: expected %.4f (tol %.4f)" name expected tol)
    Float.(abs (measured -. expected) <= tol)
    ~detail:(fun () -> Printf.sprintf "got %.5f" measured)

let weight_values ~ctx t =
  let w =
    Set.find_exn t.Tensor.params ~f:(fun p ->
        String.is_prefix ~prefix:"w" (Ir.Tnode.label_name p.Tensor.value))
  in
  Context.get_values ctx w.Tensor.value

(* A weight created through Nn_blocks.mlp_layer with no init expression: the { w } param goes
   through default_param_init, and its shape [100 -> 50] (fan-in 100, fan-out 50) is only inferred
   later, through the matmul with the input. *)
let default_init_weight_values init =
  Tensor.unsafe_reinitialize ();
  TDSL.default_param_init := init;
  let layer = Nn_blocks.mlp_layer ~label:[ "probe" ] ~hid_dim:50 () in
  let x = TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 4 ] ~output_dims:[ 100 ] () in
  let y = layer x in
  let ctx = Train.forward_once (Context.auto ()) y in
  let values = weight_values ~ctx y in
  TDSL.default_param_init := Operation.default_uniform1_param_init;
  values

let () =
  Verdict.case "default uniform1" (fun () ->
      (* The standard default: centered uniform over [-0.25, 0.25), std 0.5/sqrt(12) ~ 0.1443. *)
      let values = default_init_weight_values Operation.default_uniform1_param_init in
      check ~name:"default uniform1 [-0.25,0.25): std"
        ~expected:(0.5 /. Float.sqrt 12.)
        ~tol:0.03 (std values);
      check ~name:"default uniform1 [-0.25,0.25): mean" ~expected:0. ~tol:0.04 (mean values);
      printf "default uniform1 unique values: %d of %d\n" (unique values) (Array.length values))

let () =
  Verdict.case "default kaiming" (fun () ->
      (* normal1 has std ~1, so kaiming ~scale_sq:2.0 should realize std sqrt(2/100) ~ 0.1414.
         Before the row-product fix this was sqrt(2/1) ~ 1.41. *)
      let values = default_init_weight_values (NTDSL.kaiming ~scale_sq:2.0 TDSL.O.normal1) in
      check ~name:"default kaiming normal1 scale_sq 2 fan_in 100: std"
        ~expected:(Float.sqrt (2. /. 100.))
        ~tol:0.03 (std values);
      check ~name:"default kaiming normal1 scale_sq 2 fan_in 100: mean" ~expected:0. ~tol:0.06
        (mean values);
      printf "default kaiming unique values: %d of %d\n" (unique values) (Array.length values))

let () =
  Verdict.case "default xavier" (fun () ->
      (* uniform1 is uniform over [0, 1): mean 0.5, std 1/sqrt(12) ~ 0.2887. xavier ~scale_sq:6.0
         with fan_in 100 and fan_out 50 scales by sqrt(6/150) = 0.2. *)
      let values = default_init_weight_values (NTDSL.xavier ~scale_sq:6.0 TDSL.O.uniform1) in
      check ~name:"default xavier uniform1 scale_sq 6 fans 100+50: std"
        ~expected:(0.2 /. Float.sqrt 12.)
        ~tol:0.012 (std values);
      check ~name:"default xavier uniform1 scale_sq 6 fans 100+50: mean" ~expected:0.1 ~tol:0.02
        (mean values);
      printf "default xavier unique values: %d of %d\n" (unique values) (Array.length values))

let () =
  Verdict.case "inline kaiming" (fun () ->
      (* Inline-record init path, as in test/training/mlp_bn_names.ml. The default scale_sq is 6, so
         the realized std is sqrt(6/100) ~ 0.245 — NOT PyTorch's kaiming_normal_ (std sqrt(2/fan_in)
         = 0.1414 for relu), nor Karpathy's tanh-gain (5/3)/sqrt(fan_in) ~ 0.1667: sqrt(6) is the
         gain of the kaiming UNIFORM bound. *)
      Tensor.unsafe_reinitialize ();
      let%op mk_f () x = ({ w1 = kaiming normal1 () } * x) + { b1 = 0.; o = [ 50 ] } in
      let f = mk_f () in
      let x = TDSL.range_of_shape ~label:[ "x2" ] ~batch_dims:[ 4 ] ~output_dims:[ 100 ] () in
      let y = f x in
      let ctx = Train.forward_once (Context.auto ()) y in
      let values = weight_values ~ctx y in
      check ~name:"inline kaiming normal1 default scale_sq 6 fan_in 100: std"
        ~expected:(Float.sqrt (6. /. 100.))
        ~tol:0.05 (std values);
      printf "inline kaiming unique values: %d of %d\n" (unique values) (Array.length values))

let () =
  Verdict.case "default packed uniform" (fun () ->
      (* What [TDSL.default_param_init] holds unless a test replaces it: the packed [uniform], whose
         result shape is tied to its counter's by a round-up total instead of an equality spec. *)
      let values = default_init_weight_values Operation.default_uniform_param_init in
      check ~name:"default packed uniform [-0.25,0.25): std"
        ~expected:(0.5 /. Float.sqrt 12.)
        ~tol:0.03 (std values);
      check ~name:"default packed uniform [-0.25,0.25): mean" ~expected:0. ~tol:0.04 (mean values);
      printf "default packed uniform unique values: %d of %d\n" (unique values)
        (Array.length values))

(* gh-ocannl-1868: the model of test/training/bigram.ml. Its [{ w }] reaches the matmul through a
   pointwise [+ 1], so the bound on its output row arrives late (stage 6), after its input row has
   closed to [28] -- a product the packed sampler's block size divides. The sampler's total used to
   be settled from that product alone: 28 draws for a [28 -> 28] weight, broadcast along the other
   axis. [uniform1] is the control: its equality specs never had the defect. *)
let bigram_weight_values init =
  Tensor.unsafe_reinitialize ();
  TDSL.default_param_init := init;
  let%op mk_probs () input =
    let counts = exp (({ w } + 1) * input) in
    counts /. (counts ++ "... => 0")
  in
  let probs = mk_probs () in
  let input = TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 4 ] ~output_dims:[ 28 ] () in
  let target = TDSL.range_of_shape ~label:[ "target" ] ~batch_dims:[ 4 ] ~output_dims:[ 28 ] () in
  let%op y = probs input *. target in
  let ctx = Train.forward_once (Context.auto ()) y in
  let values = weight_values ~ctx y in
  TDSL.default_param_init := Operation.default_uniform1_param_init;
  values

let all_distinct ~name values =
  let n = Array.length values in
  Verdict.pass_fail
    (Printf.sprintf "%s: all %d values distinct" name n)
    (n > 0 && unique values = n)
    ~detail:(fun () -> Printf.sprintf "%d unique of %d" (unique values) n)

let () =
  Verdict.case "bigram packed uniform" (fun () ->
      all_distinct ~name:"bigram w, default packed uniform"
        (bigram_weight_values Operation.default_uniform_param_init));
  Verdict.case "bigram uniform1" (fun () ->
      all_distinct ~name:"bigram w, default uniform1"
        (bigram_weight_values Operation.default_uniform1_param_init))
