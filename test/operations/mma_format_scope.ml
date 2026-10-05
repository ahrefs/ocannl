(* The policy/scope descriptor judgment is pure: synthetic capabilities cover scope omissions that
   one concrete GPU cannot vary, notably CUDA sm_70's empty wide-scope lists. *)
open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module N = Ir.Numerics

let policy : N.t =
  {
    tf32_matmuls = false;
    narrow_compute_f32 = true;
    fp16_arithmetic = N.Fp16_auto;
    bf16_arithmetic = N.Bf16_auto;
  }

let scopes = [ BI.Mma_per_statement; BI.Mma_fragment_scope ]
let scope_sets = [ []; [ BI.Mma_per_statement ]; [ BI.Mma_fragment_scope ]; scopes ]
let formats = [ BI.Mma_f32; BI.Mma_f16; BI.Mma_bf16; BI.Mma_fp8_e5m2; BI.Mma_tf32 ]

let triples =
  List.concat_map formats ~f:(fun a ->
      List.concat_map formats ~f:(fun b -> List.map formats ~f:(fun d -> (a, b, d))))

let () =
  let advertised =
    [
      (BI.Mma_f32, BI.Mma_f32, BI.Mma_f32);
      (BI.Mma_f16, BI.Mma_f16, BI.Mma_f16);
      (BI.Mma_f16, BI.Mma_f16, BI.Mma_f32);
      (BI.Mma_bf16, BI.Mma_bf16, BI.Mma_bf16);
      (BI.Mma_bf16, BI.Mma_bf16, BI.Mma_f32);
      (BI.Mma_tf32, BI.Mma_tf32, BI.Mma_f32);
    ]
  in
  let check limits policy scope (a, b, d) =
    BI.advertises_mma_format_in_scope limits ~policy ~scope ~a ~b ~d
  in
  p_none "no MMA capability admits any format triple" triples
    ~f:(check BI.no_hardware_limits policy BI.Mma_per_statement);
  let rows =
    List.concat_map scope_sets ~f:(fun half_scopes ->
        List.concat_map scope_sets ~f:(fun bf_scopes ->
            List.concat_map [ N.Fp16_auto; N.Fp16_narrow; N.Fp16_wide ] ~f:(fun fp16 ->
                List.concat_map [ N.Bf16_auto; N.Bf16_narrow; N.Bf16_wide ] ~f:(fun bf16 ->
                    List.concat_map [ false; true ] ~f:(fun tf32 ->
                        List.concat_map scopes ~f:(fun scope ->
                            let policy =
                              {
                                policy with
                                fp16_arithmetic = fp16;
                                bf16_arithmetic = bf16;
                                tf32_matmuls = tf32;
                              }
                            in
                            let limits =
                              {
                                BI.no_hardware_limits with
                                mma =
                                  Some
                                    {
                                      BI.minimal_mma_capability with
                                      mma_format_tiles =
                                        List.map advertised ~f:(fun t -> (t, (16, 16, 16)));
                                      mma_f16_wide_acc_scopes = half_scopes;
                                      mma_bf16_wide_acc_scopes = bf_scopes;
                                    };
                              }
                            in
                            List.map triples ~f:(fun ((a, b, d) as triple) ->
                                let expected =
                                  List.mem advertised triple ~equal:BI.equal_mma_format_triple
                                  && (tf32
                                     || not
                                          (BI.equal_mma_input_format a BI.Mma_tf32
                                          || BI.equal_mma_input_format b BI.Mma_tf32))
                                  &&
                                  match triple with
                                  | BI.Mma_f16, BI.Mma_f16, BI.Mma_f16
                                    when N.equal_fp16_mode fp16 N.Fp16_wide ->
                                      List.mem half_scopes scope ~equal:BI.equal_mma_emission_scope
                                  | BI.Mma_bf16, BI.Mma_bf16, BI.Mma_bf16
                                    when N.equal_bf16_mode bf16 N.Bf16_wide ->
                                      List.mem bf_scopes scope ~equal:BI.equal_mma_emission_scope
                                  | _ -> true
                                in
                                (expected, check limits policy scope (a, b, d)))))))))
  in
  p_all ~min:72000
    "format scope judgment respects both policy gates, both scope lists and the full triple" rows
    ~f:(fun (expected, actual) -> Bool.equal expected actual)
