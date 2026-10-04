(* Known integer exponents use a multiplication/reciprocal algorithm, independently of the optional
   algebra simplifier. Rounding is that algorithm's rounding, not libm pow's rounding. NaN^0 = 1;
   other NaN powers are NaN. Odd powers preserve the sign of zero and infinity; negative powers
   reciprocate the positive power. Fractional powers retain the vendor policy. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "integer-power backend: %s\n" backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let () = LL.optimize_integer_pow := false
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let bitwise a b =
  if Float.is_nan b then Float.is_nan a
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let bases = [ -2.; -0.5; -1.; 0.; -0.; Float.infinity; Float.neg_infinity; Float.nan ]
let exponents = [ 0; 1; 2; 3; -1; -2; -3; 8; -8; 31; -31 ]

(* Independent, linear materialized-multiplication oracle: dyadic inputs keep the finite answers
   exact, so this check asks about domain and exceptional values without assuming pow rounding. *)
let oracle ~narrow x n =
  let positive =
    List.fold (List.init (Int.abs n) ~f:Fn.id) ~init:1. ~f:(fun acc _ -> narrow (acc *. x))
  in
  if n < 0 then narrow (1. /. positive) else positive

let run ~prec ~name ~narrow =
  Verdict.case name (fun () ->
      let cases = List.concat_map bases ~f:(fun x -> List.map exponents ~f:(fun n -> (x, n))) in
      let len = List.length cases in
      let node = Ll_test.node_factory ~prec ~first_id:18100 ~dims:[| len |] () in
      let input = node "ipow_input" and output = node "ipow_output" in
      List.iter [ input; output ] ~f:Ll_test.materialize;
      let body =
        List.foldi cases ~init:LL.Noop ~f:(fun i acc (_, n) ->
            let value =
              LL.Binop
                ( ToPowOf,
                  (Ll_test.get input [| Ll_test.fixed i |], prec),
                  (Ll_test.c (Float.of_int n), prec) )
            in
            Ll_test.seq acc (Ll_test.set output [| Ll_test.fixed i |] value))
      in
      let o = Ll_test.optimize ~name body in
      let input_values = Array.of_list_map cases ~f:(fun (x, _) -> narrow x) in
      let got =
        List.hd_exn
          (Ll_test.execute ~name o
             ~seed:[ (input, input_values); (output, Ll_test.blank len) ]
             ~read:[ output ])
      in
      p_alli
        (name ^ ": integer powers match independent multiplication including exceptional values")
        cases ~f:(fun i (x, n) ->
          let want = oracle ~narrow (narrow x) n in
          let ok = bitwise got.(i) want in
          if not ok then
            Stdio.eprintf "(not part of the golden) %h^%d: got %h want %h\n" x n got.(i) want;
          ok);
      let src = Generated.read name in
      p
        (name ^ ": known integers bypass floating pow")
        (String.is_substring src ~substring:"ocannl_powi_"
        && not (String.is_substring src ~substring:"powf(")))

let () = run ~prec:Ops.single ~name:"ipow_f32" ~narrow:f32

let () =
  run ~prec:Ops.half ~name:"ipow_f16" ~narrow:(fun x ->
      Ops.half_to_single (Ops.single_to_half (f32 x)))

let () =
  if (Context.codegen_capabilities (Context.auto ())).supports_f64 then
    run ~prec:Ops.double ~name:"ipow_f64" ~narrow:Fn.id
  else begin
    Verdict.skipped ~backend:backend_name
      "ipow_f64: integer powers match independent multiplication including exceptional values";
    Verdict.skipped ~backend:backend_name "ipow_f64: known integers bypass floating pow"
  end

(* Exponents outside every machine int range still retain parity and sign, without narrowing a
   host-double exponent to float. All powers of +/-1 are exact even at these magnitudes. *)
let () =
  Verdict.case "large exponents" (fun () ->
      let cases =
        [
          (-1., 0x1p63, 1.);
          (-1., -0x1p63, 1.);
          (-1., 0x1.fffffffffffffp52, -1.);
          (-1., 0x1.fffffffffffffp1023, 1.);
          (-1., -0x1.fffffffffffffp1023, 1.);
        ]
      in
      let len = List.length cases in
      let node = Ll_test.node_factory ~first_id:18200 ~dims:[| len |] () in
      let input = node "ipow_large_input" and output = node "ipow_large_output" in
      List.iter [ input; output ] ~f:Ll_test.materialize;
      let body =
        List.foldi cases ~init:LL.Noop ~f:(fun i acc (_, n, _) ->
            Ll_test.seq acc
              (Ll_test.set output
                 [| Ll_test.fixed i |]
                 (LL.Binop
                    ( ToPowOf,
                      (Ll_test.get input [| Ll_test.fixed i |], Ops.single),
                      (Ll_test.c n, Ops.single) ))))
      in
      let o = Ll_test.optimize ~name:"ipow_large" body in
      let got =
        List.hd_exn
          (Ll_test.execute ~name:"ipow_large" o
             ~seed:
               [
                 (input, Array.of_list_map cases ~f:(fun (x, _, _) -> x));
                 (output, Ll_test.blank len);
               ]
             ~read:[ output ])
      in
      p_all "constant folding accepts integral float exponents outside the machine-int range" cases
        ~f:(fun (x, n, want) -> bitwise (Ops.interpret_binop ToPowOf x n) want);
      p_alli "edge exponents preserve exact parity without integer overflow or float narrowing"
        cases ~f:(fun i (_, _, want) -> bitwise got.(i) want))

let () =
  Verdict.case "scoped base" (fun () ->
      LL.optimize_integer_pow := true;
      let node = Ll_test.node_factory ~first_id:18300 ~dims:[| 1 |] () in
      let input = node "ipow_once_input"
      and output = node "ipow_once_output"
      and local = node "ipow_local" in
      List.iter [ input; output ] ~f:Ll_test.materialize;
      Ll_test.virtualize local;
      let id = LL.get_scope local in
      let index = [| Ll_test.fixed 0 |] in
      let base =
        LL.Local_scope
          {
            id;
            orig_indices = index;
            mint = LL.Inlined_computation;
            body =
              Ll_test.seq
                (LL.Set_local (id, Ll_test.get input index))
                (Ll_test.seq
                   (LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.c 1.)))
                   (LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.c 1.))));
          }
      in
      let body =
        Ll_test.set output index
          (LL.Binop (ToPowOf, (base, Ops.single), (Ll_test.c 2., Ops.single)))
      in
      p "integer-power simplification retains a scoped base without duplicating its body"
        (match LL.simplify_llc ~fp_algebra:"pow" [] body with
        | LL.Set { llsc = LL.Binop (ToPowOf, (LL.Local_scope _, _), _); _ } -> true
        | _ -> false);
      let raw = Ll_test.set output index (Ll_test.add (Ll_test.get input index) (Ll_test.c 1.)) in
      let o = Ll_test.optimize_scoped ~name:"ipow_once" ~raw body in
      let got =
        List.hd_exn
          (Ll_test.execute ~name:"ipow_once" o
             ~seed:[ (input, [| 1. |]); (output, [| 99. |]) ]
             ~read:[ output ])
      in
      p "scoped integer-power base produces the correct executed value with unrolling enabled"
        (Float.equal got.(0) 9.);
      let src = Generated.read "ipow_once" in
      p "scoped integer-power base remains a single helper argument with unrolling enabled"
        (String.is_substring src ~substring:"ocannl_powi_f32("
        && List.length
             (String.substr_index_all
                (String.substr_replace_all src ~pattern:")[" ~with_:"[")
                ~pattern:"ipow_once_input[" ~may_overlap:false)
           = 1))

let () =
  Verdict.case "fractional exponent" (fun () ->
      LL.optimize_integer_pow := false;
      let node = Ll_test.node_factory ~first_id:18400 ~dims:[| 1 |] () in
      let input = node "ipow_fraction_input" and output = node "ipow_fraction_output" in
      List.iter [ input; output ] ~f:Ll_test.materialize;
      let index = [| Ll_test.fixed 0 |] in
      let body =
        Ll_test.set output index
          (LL.Binop (ToPowOf, (Ll_test.get input index, Ops.single), (Ll_test.c 0.5, Ops.single)))
      in
      let o = Ll_test.optimize ~name:"ipow_fraction" body in
      let got =
        List.hd_exn
          (Ll_test.execute ~name:"ipow_fraction" o
             ~seed:[ (input, [| 4. |]); (output, [| 99. |]) ]
             ~read:[ output ])
      in
      let src = Generated.read "ipow_fraction" in
      p "fractional exponents retain vendor floating pow and its positive-base result"
        (Float.(abs (got.(0) - 2.) < 1e-6)
        && (String.is_substring src ~substring:"pow(" || String.is_substring src ~substring:"powf(")
        && not (String.is_substring src ~substring:"ocannl_powi_")))
