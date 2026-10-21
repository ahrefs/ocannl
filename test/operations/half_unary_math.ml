(* Half unary math builtins on Metal are the f32 builtin narrowed once (gh-ocannl-1288).

   Metal renders half [Exp], [Log], [Exp2], [Log2], [Sin], [Cos] and [Sqrt] as
   [(half)op((float)(x))], as it renders half [ToPowOf] (gh-ocannl-1198), so a half kernel moves
   together with the same f32 kernel under whichever compile-option math policy is in force: the
   modern split on macOS 15+, the legacy [fastMathEnabled=false] on macOS 14. The half overloads did
   not: measured over all 65536 half inputs on M4 Max
   ([benchmarks/runners/ocannl/metal_half_math_probe.ml]), [sin(-0)] lost its sign under both
   policies, and [sin]/[cos] of inf and NaN gave NaN under the modern one, where the fast f32
   functions give 0. The inputs below include those cases, so the executed claim fails with the
   overloads on either policy, and the source claim pins the spelling that makes it hold by
   construction. Since gh-ocannl-1435 the f32 [sin]/[cos] are [Builtins_metal]'s [ocannl_sin] and
   [ocannl_cos], which return NaN there too ([unary_math_special_values] pins the values); the
   relationship pinned here is unchanged.

   CUDA and HIP keep their native half intrinsics ([hexp], [hsin], ...), which promise no such
   relationship, so both claims are Metal-only; every backend still compiles and runs the
   kernels. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "half-unary-math backend: %s\n" backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let is_metal = String.equal backend_name "metal"
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f16 x = Ops.half_to_single (Ops.single_to_half (f32 x))

(* Each op with the f32 function its half form widens into. *)
let ops : (Ops.unop * string) list =
  [
    (Exp, "exp");
    (Log, "log");
    (Exp2, "exp2");
    (Log2, "log2");
    (Sin, "ocannl_sin");
    (Cos, "ocannl_cos");
    (Sqrt, "sqrt");
  ]

(* Signed zeros, infinities and NaN (where the overloads parted from the widened form), a subnormal,
   and finite values across the domains, negative ones included (NaN under [log] and [sqrt]); all
   exactly representable at half. *)
let inputs =
  List.map ~f:f16
    [
      -0.;
      0.;
      Float.infinity;
      Float.neg_infinity;
      Float.nan;
      0x1p-20;
      -0.75;
      0.1;
      0.5;
      1.;
      2.7;
      -3.25;
      10.;
      100.;
      1000.;
      30000.;
    ]

(* One kernel at [prec] computing every op over every input: [out.(k * n + i) = op_k inputs.(i)]. *)
let run ~prec ~name ~first_id =
  let n = List.length inputs and m = List.length ops in
  let node = Ll_test.node_factory ~prec ~first_id ~dims:[| n |] () in
  let input = node (name ^ "_in") and out = node ~dims:[| n * m |] (name ^ "_out") in
  List.iter [ input; out ] ~f:Ll_test.materialize;
  let body =
    List.foldi ops ~init:LL.Noop ~f:(fun k acc (op, _) ->
        List.foldi inputs ~init:acc ~f:(fun i acc _ ->
            Ll_test.seq acc
              (Ll_test.set out
                 [| Ll_test.fixed ((k * n) + i) |]
                 (LL.Unop (op, (Ll_test.get input [| Ll_test.fixed i |], prec))))))
  in
  let o = Ll_test.optimize ~name body in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:[ (input, Array.of_list inputs); (out, Ll_test.blank (n * m)) ]
       ~read:[ out ])

let same_half a b =
  (Float.is_nan a && Float.is_nan b) || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let () =
  Verdict.case "half unary math" (fun () ->
      let half = run ~prec:Ops.half ~name:"hum" ~first_id:19700 in
      let single = run ~prec:Ops.single ~name:"hum_f32" ~first_id:19800 in
      let n = List.length inputs in
      gated_alli ~when_:is_metal ~on:backend_name
        "Metal half exp, log, exp2, log2, sin, cos and sqrt are the same kernel's f32 results \
         rounded to half once, signed zeros, infinities and NaN included"
        (Array.to_list half) ~f:(fun j v ->
          let want = f16 single.(j) in
          let ok = same_half v want in
          if not ok then
            Stdio.eprintf "(not part of the golden) %s(%h): half %h, f32 narrowed %h\n"
              (snd (List.nth_exn ops (j / n)))
              (List.nth_exn inputs (j % n))
              v want;
          ok);
      let src = Generated.read "hum" in
      gated_all ~when_:is_metal ~on:backend_name
        "Metal renders each half unary builtin as the widened f32 call narrowed once" ops
        ~f:(fun (_, fn) -> String.is_substring src ~substring:("(half)" ^ fn ^ "((float)(")))
