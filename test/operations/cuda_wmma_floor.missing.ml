(* The select arm with no NVRTC bindings. The shared driver owns every claim label; the forced
   missing-toolkit alias checks its stdout against the real witness's golden on every host. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  let aggregation =
    if Array.exists (fun x -> x = "--forced") Sys.argv then `Outside_sweep else `Environment
  in
  let leg = Verdict.leg ~aggregation ~when_:false ~on:"NVRTC bindings unavailable" () in
  Cuda_wmma_floor_claims.run ~leg ~compute_70_leg:leg ~k8_leg:leg ~jit_70_leg:leg ~k16_leg:leg
    (fun _ -> failwith "cuda_wmma_floor: a closed missing-NVRTC leg evaluated its probe")
