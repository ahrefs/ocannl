open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* gh-509 task 4: virtual packed [uniform] results are inlined per cell via the lane-extract form
   [vec_convert(counter[flat / lanes]).v[flat mod lanes]], and must match materialized runs bitwise.
   For each precision and shape we run the same program twice -- once with the uniform tensor forced
   materialized, once left to virtualize -- and compare the consumer's values bit-for-bit (NaN-safe:
   fp8 random bit patterns can be NaN). The generated source is also checked structurally: the
   materialized run stores via the vectorized builtin, the virtual run reads via the lane builtin
   and emits no vectorized store. (Test disabled on Metal because of the double section — Metal has
   no f64; the lane path itself is backend-generic, fp8 included.) *)

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The consumer multiplies by 1: exact in every precision (including the fp8 float bridge), and
   reads each cell of [u] exactly once so [u] stays a virtualization candidate. Both runs create
   tensors in the same order after [unsafe_reinitialize], so the threefry keys (self ids) line
   up. *)
let run ~virtual_ ~prec ?input_dims output_dims =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let u = TDSL.uniform () ?input_dims ~output_dims () in
  Ir.Tnode.update_prec u.value prec;
  if not virtual_ then Train.set_materialized u.value;
  let%op uvl = u *. 1. in
  Ir.Tnode.update_prec uvl.value prec;
  (* Every leg of every precision compiles under the one routine name [uvl_fwd], so each compile
     overwrites the previous one's artifact. Arming deletes it first: what is read back below is
     this run's kernel or nothing, never the previous leg's. *)
  Generated.arm "uvl_fwd";
  let ctx = Train.forward_once ctx uvl in
  let values = Context.get_values ctx uvl.value in
  (values, Generated.read "uvl_fwd")

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Structural checks scan only the routine body: the prepended builtin definitions mention both the
   vectorized and the lane builtins (the lane builtins are implemented via the _vec ones). *)
let has sub s =
  let body =
    match String.substr_index s ~pattern:"Main logic" with
    | Some i -> String.subo s ~pos:i
    | None -> s
  in
  String.is_substring body ~substring:sub

let check ~name ~prec ?input_dims output_dims =
  let ref_vals, ref_src = run ~virtual_:false ~prec ?input_dims output_dims in
  let vir_vals, vir_src = run ~virtual_:true ~prec ?input_dims output_dims in
  let parity =
    Array.length ref_vals = Array.length vir_vals
    && (not (Array.is_empty vir_vals))
    && Array.for_alli vir_vals ~f:(fun i v -> bits_equal v ref_vals.(i))
  in
  let ref_vec = has "_uniform_vec(" ref_src and ref_lane = has "_uniform_lane(" ref_src in
  let vir_lane = has "_uniform_lane(" vir_src and vir_vec = has "_uniform_vec(" vir_src in
  (* A four-property census row whose shape is the point, so the claims sit beside it on the same
     [let]-bound booleans. The two the row prints as [false] are claimed in the form that holds --
     in a golden a blessed regression and a designed negative are the same line. *)
  Stdio.printf
    "%s: %d values, bitwise parity %b | materialized: vec store %b, lane %b | virtual: lane %b, \
     vec store %b\n"
    name (Array.length ref_vals) parity ref_vec ref_lane vir_lane vir_vec;
  Verdict.claimf "%s: virtual values bitwise-equal to the materialized ones" name parity;
  Verdict.claimf "%s: materialized run stores via the vectorized builtin" name ref_vec;
  Verdict.claimf "%s: materialized run emits no lane extract" name (not ref_lane);
  Verdict.claimf "%s: virtual run reads via the lane builtin" name vir_lane;
  Verdict.claimf "%s: virtual run emits no vectorized store" name (not vir_vec)

let () =
  check ~name:"single n=1 (lone partial block)" ~prec:Ir.Ops.single [ 1 ];
  check ~name:"single n=5 (tail peel)" ~prec:Ir.Ops.single [ 5 ];
  check ~name:"single n=8 (divisible)" ~prec:Ir.Ops.single [ 8 ];
  check ~name:"single 5->3 (multi-axis, 15 elements)" ~prec:Ir.Ops.single ~input_dims:[ 5 ] [ 3 ];
  (* Trailing dim-1 axis (e.g. a conv kernel's single input channel): the strided store projection
     must pair with the innermost non-unit axis -- pairing with the dim-1 axis collapsed the stride
     and left all cells beyond the first block uninitialized. *)
  check ~name:"single 9->1 (trailing dim-1 axis)" ~prec:Ir.Ops.single ~input_dims:[ 1 ] [ 9 ];
  check ~name:"half n=9" ~prec:Ir.Ops.half [ 9 ];
  (* bfloat16 pins that the packed path exists at this precision on every backend -- a missing
     vector block type shows up here as a compile-time refusal. It cannot police the element type
     itself: both runs go through the same builtin, so a builtin returning raw bits (which the
     assignment to a bfloat16 cell would convert by value) still shows parity. The value-level check
     for that lives in bf16_ops.ml. *)
  check ~name:"bfloat16 n=9" ~prec:Ir.Ops.bfloat16 [ 9 ];
  (* uint32 exercises the unsigned vec/lane builtins (full-range bit patterns). *)
  check ~name:"uint32 n=5" ~prec:Ir.Ops.uint32 [ 5 ];
  (* uint64 is the 2-lanes-per-block end of the range, and pins the widest element type against a
     missing vector block entry -- the same compile-time refusal the bfloat16 case guards above. *)
  check ~name:"uint64 n=3" ~prec:Ir.Ops.uint64 [ 3 ];
  check ~name:"double n=3" ~prec:Ir.Ops.double [ 3 ];
  (* fp8 is the stress case: 16 lanes per block, and random bit patterns include NaNs (compared by
     bits, not value). *)
  check ~name:"fp8 n=17" ~prec:Ir.Ops.fp8 [ 17 ]
