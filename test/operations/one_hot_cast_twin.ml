(* gh-ocannl-2374: one-hot embedding gather and backward scatter recognized through precision casts
   and uncollapsed selector scopes.

   Under mixed-precision training ([BENCH_PRECISION=bf16|f16] on [gpt2_mini_train]), two barriers
   prevented [rewrite_one_hot_reductions] from lowering the embedding lookup and its gradient: 1. A
   virtual cast twin sits between the master table and the one-hot product as [Unop (Round_prec,
   (Get (wte, ...), bf16))] (or a float cast [Local_scope] for non-narrow cast twins like f64 ->
   f32). 2. A [uint32] one-hot selector stays inside its [Local_scope] whenever the vocabulary
   reaches the product precision's exact integer limit ([vocab >= 256] for bf16, [>= 2048] for f16;
   gh-ocannl-1884), because collapsing it into the narrow float slot would conflate adjacent token
   IDs.

   Pinned invariants: - Structural + executed parity on bf16 [gpt2_mini_train]-shaped inputs
   (batch=8, seq=128, d_model=256, vocab=1024) for both virtual and materialized cast twins against
   the f32 route, with discriminating IDs across [0, 1023] including adjacent pairs above 256
   (256/257, 511/512, 512/513, 1022/1023), repeated IDs and untouched rows. - The precision
   conversion is kept on the gathered values: gathering a bf16 cast twin into an f32 output rounds
   [1 + under_half_ulp] to [1.0] (not unrounded f32), and OOB IDs produce zero in forward and touch
   no row in backward. - Half (f16) cast twins at both [vocab = 1024 < 2048] (collapsed selector)
   and [vocab = 2050 > 2048] (scoped selector, adjacent IDs 2048/2049). - Non-narrow cast twin (f64
   -> f32 into f64 output, gated on [supports_f64]). - Negative controls (hand-built IR): a
   narrowing [Local_scope<byte>] over [int32] IDs (257 wraps to 1) and a bare [bfloat16] comparison
   over [vocab = 300 >= 256] must NOT rewrite to [Get_dynamic]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module IDX = Train.IDX
module MP = Mixed_prec
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* Build a uint32 IDs tensor with explicit [batch; seq] batch dims, mirroring [bench_gpt_model]. *)
let ids_tensor_2d ~label (rows : int array array) =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  let open Bigarray in
  let g = Genarray.create Int32 c_layout [| batch; seq |] in
  for b = 0 to batch - 1 do
    for s = 0 to seq - 1 do
      Genarray.set g [| b; s |] (Int32.of_int_trunc rows.(b).(s))
    done
  done;
  TDSL.wrap ~l:label ~b:[ batch; seq ] ~o:[] (Ir.Ndarray.as_array Ops.Uint32 g) ()

type ir_counts = {
  gathers : int;
  rounded_gathers : int;
  scatters : int;
  vocab_loops : int;
  truncs : int;
}

let inspect_ir ~vocab (opt : LL.optimized) : ir_counts =
  let gathers = ref 0
  and rounded_gathers = ref 0
  and scatters = ref 0
  and vocab_loops = ref 0
  and truncs = ref 0 in
  (* Count vocabulary loops containing a [Cmpeq] (one-hot reduction loops), excluding plain
     pointwise copies such as a materialized cast twin [twin[d, v] := wte[d, v]] or [wte.grad[d, v]
     += twin.grad[d, v]]. *)
  Ll_test.walk opt.LL.llc
    ~on_stmt:(function
      | LL.Set_dynamic _ -> Int.incr scatters
      | LL.For_loop { from_ = 0; to_; body; _ } when to_ = vocab - 1 ->
          let is_one_hot_like = ref false in
          Ll_test.walk body ~on_scalar:(function
            | LL.Binop (Ops.Cmpeq, _, _) -> is_one_hot_like := true
            | _ -> ());
          if !is_one_hot_like then Int.incr vocab_loops
      | _ -> ())
    ~on_scalar:(function
      | LL.Get_dynamic _ -> Int.incr gathers
      | LL.Unop (Ops.Round_prec, (LL.Get_dynamic _, _)) -> Int.incr rounded_gathers
      | LL.Unop (Ops.Trunc, _) -> Int.incr truncs
      | _ -> ());
  {
    gathers = !gathers;
    rounded_gathers = !rounded_gathers;
    scatters = !scatters;
    vocab_loops = !vocab_loops;
    truncs = !truncs;
  }

(* Compile [comp] while capturing the optimized IR passed into the schedule transform (before
   fission/scheduling) so structural assertions inspect the exact routine compiled and executed. *)
let compile_capturing ctx comp =
  let captured = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx comp IDX.empty
  in
  (ctx, routine, Option.value_exn ~here:[%here] !captured)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let param_by_label root name =
  List.find_exn (Set.to_list root.Tensor.params) ~f:(fun t ->
      String.equal (Tn.label_name t.Tensor.value) name)

let under_half_ulp prec = (2. **. Float.of_int (-Ops.significand_bits prec)) -. (2. **. -20.)

(* Independent host-side reference for the embedding forward lookup and backward scatter-add, free
   of [rewrite_one_hot_reductions] (catches any bug shared by both the f32 and cast-twin routes). *)
let host_embedding_ref ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals =
  let batch = Array.length ids_rows and seq = Array.length ids_rows.(0) in
  let host_emb = Array.create ~len:(batch * seq * d_model) 0. in
  let host_grad = Array.create ~len:(d_model * vocab) 0. in
  for b = 0 to batch - 1 do
    for s = 0 to seq - 1 do
      let pos = (b * seq) + s in
      let id = ids_rows.(b).(s) in
      if id >= 0 && id < vocab then
        for d = 0 to d_model - 1 do
          host_emb.((pos * d_model) + d) <- wte_vals.((d * vocab) + id);
          let g_idx = (d * vocab) + id in
          host_grad.(g_idx) <- host_grad.(g_idx) +. coeff_vals.((pos * d_model) + d)
        done
    done
  done;
  (host_emb, host_grad)

(* Build and run one embedding forward + backward graph at [gpt2_mini_train]-style shapes, either
   plain f32 ([prec = None]) or mixed-precision with a cast twin ([prec = Some (p, placement)]). *)
let run_embedding_step ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals ~prec =
  Tensor.unsafe_reinitialize ();
  let batch = Array.length ids_rows and seq = Array.length ids_rows.(0) in
  let ids = ids_tensor_2d ~label:"ids" ids_rows in
  let onehot = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let make_wte () =
    TDSL.param ~values:wte_vals "wte" ~input_dims:[ vocab ] ~output_dims:[ d_model ] ()
  in
  let wte_used =
    match prec with
    | None -> make_wte ()
    | Some (p, placement) -> MP.with_master_weights ~placement ~prec:p make_wte
  in
  let%op emb = wte_used * onehot in
  Train.set_materialized emb.Tensor.value;
  Tn.set_observable emb.Tensor.value;
  let coeff =
    TDSL.ndarray coeff_vals ~label:[ "coeff" ] ~batch_dims:[ batch; seq ] ~output_dims:[ d_model ]
      ()
  in
  let%op loss = (emb *. coeff) ++ "...|... => |->0" in
  Option.iter prec ~f:(fun (p, _) ->
      Precision_policy.apply
        ~except:(fun tn -> Tn.equal tn loss.Tensor.value || Tn.equal tn coeff.Tensor.value)
        { Precision_policy.param_prec = None; activation_prec = Some p; grad_prec = Some p }
        loss);
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine, opt = compile_capturing ctx update in
  let ctx = Context.run ctx routine in
  let wte_master = param_by_label loss "wte" in
  let emb_vals = Context.get_values ctx emb.Tensor.value in
  let grad_vals = Context.get_values ctx (grad_of wte_master) in
  let counts = inspect_ir ~vocab opt in
  (emb_vals, grad_vals, counts)

let () =
  (* === Case 1: bf16 gpt2_mini_train-shaped cast-twin embedding (8 x 128 x 256, vocab = 1024)
     === *)
  Verdict.case "bf16 gpt2_mini_train-shaped cast-twin embedding" (fun () ->
      let batch = 8 and seq = 128 and d_model = 256 and vocab = 1024 in
      (* Discriminating IDs across [0, 1023]: explicitly hits adjacent IDs above bf16's 256 exact
         limit (256/257, 511/512, 512/513, 1022/1023), repeats IDs to exercise scatter accumulation,
         and leaves rows [1008..1021] untouched as a zero-row control. *)
      let pinned_ids = [| 0; 1; 255; 256; 257; 258; 511; 512; 513; 767; 768; 1022; 1023 |] in
      let touched = Array.create ~len:vocab false in
      let ids_rows =
        Array.init batch ~f:(fun b ->
            Array.init seq ~f:(fun s ->
                let pos = (b * seq) + s in
                let id =
                  if pos < Array.length pinned_ids then pinned_ids.(pos)
                  else ((pos * 73) + 19) % 1008
                in
                touched.(id) <- true;
                id))
      in
      (* Non-dyadic, mixed-sign discriminating table values varying with both channel [d] and token
         [v]. Adjacent rows wrap signs across the modulus, so conflating IDs flips signs. *)
      let wte_dims = [| d_model; vocab |] in
      let wte_vals =
        Array.init (d_model * vocab) ~f:(fun idx ->
            Ll_test.cycle_flat ~radix:5 ~dims:wte_dims ~modulus:29 ~offset:(-14.5) ~stride:0.037 idx)
      in
      (* Non-uniform upstream gradients varying with [b], [s], [d]. *)
      let coeff_dims = [| batch; seq; d_model |] in
      let coeff_vals =
        Array.init
          (batch * seq * d_model)
          ~f:(fun idx ->
            Ll_test.cycle_flat ~radix:7 ~dims:coeff_dims ~modulus:29 ~offset:1. ~stride:0.01 idx)
      in
      let host_emb, host_grad =
        host_embedding_ref ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
      in
      let f32_emb, f32_grad, f32_counts =
        run_embedding_step ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals ~prec:None
      in
      p "f32 baseline lowers embedding forward as Get_dynamic and backward as Set_dynamic"
        (f32_counts.gathers >= 1 && f32_counts.scatters >= 1 && f32_counts.vocab_loops = 0);
      let f32_tol w = 1e-5 +. (1e-5 *. Float.abs w) in
      p_close "f32 baseline forward embedding matches independent host reference" f32_emb host_emb
        ~tol:f32_tol;
      p_close "f32 baseline backward wte.grad matches independent host reference" f32_grad host_grad
        ~tol:f32_tol;
      let bf16_emb_tol w = 2e-2 +. (1e-2 *. Float.abs w) in
      let bf16_grad_tol w = 5e-2 +. (2e-2 *. Float.abs w) in
      let rows_have_opposite_sign a b = Array.exists2_exn a b ~f:(fun x y -> Float.(x *. y < 0.)) in
      List.iter
        [
          ("virtual twin", MP.Twin_virtual, true); ("materialized twin", MP.Twin_materialized, false);
        ]
        ~f:(fun (label, placement, expect_rounded_gather) ->
          let bf_emb, bf_grad, bf_counts =
            run_embedding_step ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:(Some (Ops.bfloat16, placement))
          in
          Stdio.eprintf
            "bf16 %s (not part of the golden): gathers=%d rounded_gathers=%d scatters=%d \
             vocab_loops=%d truncs=%d\n"
            label bf_counts.gathers bf_counts.rounded_gathers bf_counts.scatters
            bf_counts.vocab_loops bf_counts.truncs;
          pf "bf16 %s: forward lowers as Get_dynamic gather" label (bf_counts.gathers >= 1);
          if expect_rounded_gather then
            pf "bf16 %s: gather keeps Round_prec on the dynamic read" label
              (bf_counts.rounded_gathers >= 1);
          pf "bf16 %s: backward lowers as Set_dynamic scatter" label (bf_counts.scatters >= 1);
          pf "bf16 %s: no vocabulary loop survives in forward or backward" label
            (bf_counts.vocab_loops = 0);
          pf "bf16 %s: uint32 IDs emit no integrality Trunc" label (bf_counts.truncs = 0);
          pf_close "bf16 %s: forward embedding matches f32 within bf16 tolerance" label bf_emb
            f32_emb ~tol:bf16_emb_tol;
          pf_close "bf16 %s: forward embedding matches independent host reference" label bf_emb
            host_emb ~tol:bf16_emb_tol;
          (* Adjacent IDs past 256 (positions 3 vs 4 -> IDs 256 vs 257; positions 7 vs 8 -> IDs 512
             vs 513) must stay distinct and match their own table rows. *)
          let row_at pos = Array.sub bf_emb ~pos:(pos * d_model) ~len:d_model in
          let f32_row_at pos = Array.sub f32_emb ~pos:(pos * d_model) ~len:d_model in
          let r256, r257 = (row_at 3, row_at 4) in
          let r512, r513 = (row_at 7, row_at 8) in
          pf "bf16 %s: adjacent IDs 256/257 and 512/513 stay distinct" label
            (rows_have_opposite_sign r256 r257 && rows_have_opposite_sign r512 r513
            && Verdict.within "row 257" r257 (f32_row_at 4) ~tol:bf16_emb_tol
            && Verdict.within "row 513" r513 (f32_row_at 8) ~tol:bf16_emb_tol);
          (* Scatter accumulation into bf16 rounds per hit (at most ~3 hits/row here), so bf16
             relative tolerance 2e-2 covers accumulated rounding while catching any
             dropped/misrouted scatter. *)
          pf_close "bf16 %s: backward wte.grad matches f32 within bf16 tolerance" label bf_grad
            f32_grad ~tol:bf16_grad_tol;
          pf_close "bf16 %s: backward wte.grad matches independent host reference" label bf_grad
            host_grad ~tol:bf16_grad_tol;
          let untouched_cells =
            Array.foldi bf_grad ~init:[] ~f:(fun idx acc g ->
                if touched.(idx % vocab) then acc else g :: acc)
          in
          p_all (Printf.sprintf "bf16 %s: untouched vocabulary rows keep zero gradient" label)
            untouched_cells ~f:(fun g -> Float.(g = 0.))));

  (* === Case 2: conversion stays on gathered values + OOB IDs === *)
  Verdict.case "conversion kept on gathered values and OOB handling" (fun () ->
      Tensor.unsafe_reinitialize ();
      let vocab = 304 and d_model = 4 in
      let master_val = 1. +. under_half_ulp Ops.bfloat16 in
      let wte_dims = [| d_model; vocab |] in
      let wte_vals =
        Array.init (d_model * vocab) ~f:(fun idx ->
            if idx % vocab = 300 then master_val
            else Ll_test.cycle_flat ~radix:3 ~dims:wte_dims ~modulus:7 ~offset:2. ~stride:1. idx)
      in
      (* Positions: 300 (exercises rounding to 1.0), 301 (adjacent), vocab (OOB -> 0.0). *)
      let id_list = [ 300; 301; vocab ] in
      let ids = Nn_blocks.class_ids_of_int_list ~label:"ids_round" id_list in
      let wte_twin =
        MP.with_master_weights ~placement:MP.Twin_virtual ~prec:Ops.bfloat16 (fun () ->
            TDSL.param ~values:wte_vals "wte_round" ~input_dims:[ vocab ] ~output_dims:[ d_model ]
              ())
      in
      let%op emb = wte_twin * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
      (* Pin output to single (f32): without Round_prec on the gathered value, reading [wte_master]
         directly into f32 would return [master_val] (> 1.0) instead of bf16-rounded [1.0]. *)
      Tn.update_prec emb.Tensor.value Ops.single;
      Train.set_materialized emb.Tensor.value;
      Tn.set_observable emb.Tensor.value;
      let ctx = Train.init_params (Context.auto ()) IDX.empty emb in
      let ctx, routine, opt = compile_capturing ctx (Train.forward emb) in
      (* Prefill output with a non-zero sentinel so an unwritten OOB row fails loudly. *)
      let ctx =
        Context.set_values ctx emb.Tensor.value
          (Array.create ~len:(List.length id_list * d_model) 99.)
      in
      let ctx = Context.run ctx routine in
      let got = Context.get_values ctx emb.Tensor.value in
      let counts = inspect_ir ~vocab opt in
      Stdio.eprintf "bf16-into-f32 (not part of the golden): row300[0]=%h oob[0]=%h\n" got.(0)
        got.(2 * d_model);
      p "bf16-into-f32: lowers as Round_prec over Get_dynamic with no vocab loop"
        (counts.gathers >= 1 && counts.rounded_gathers >= 1 && counts.vocab_loops = 0);
      p_all "bf16-into-f32: gathered value rounds to bf16 (1.0, not unrounded f32)"
        (Array.to_list (Array.sub got ~pos:0 ~len:d_model))
        ~f:(fun x -> Float.(x = 1.));
      p_all "bf16-into-f32: OOB ID writes exact zero row"
        (Array.to_list (Array.sub got ~pos:(2 * d_model) ~len:d_model))
        ~f:(fun x -> Float.(x = 0.)));

  (* === Case 3: half (f16) cast twin below and above f16's 2048 exact-int limit === *)
  Verdict.case "f16 cast twin below and above 2048" (fun () ->
      let f16_emb_tol w = 2e-3 +. (2e-3 *. Float.abs w) in
      let f16_grad_tol w = 5e-3 +. (5e-3 *. Float.abs w) in
      List.iter
        [
          ("vocab=1024", 1024, [| 0; 1; 511; 512; 1022; 1023 |]);
          ("vocab=2050", 2050, [| 0; 1; 2047; 2048; 2049; 2048 |]);
        ]
        ~f:(fun (label, vocab, pinned) ->
          let batch = 2 and seq = 16 and d_model = 16 in
          let ids_rows =
            Array.init batch ~f:(fun b ->
                Array.init seq ~f:(fun s ->
                    let pos = (b * seq) + s in
                    if pos < Array.length pinned then pinned.(pos) else ((pos * 61) + 5) % vocab))
          in
          let wte_dims = [| d_model; vocab |] in
          let wte_vals =
            Array.init (d_model * vocab) ~f:(fun idx ->
                Ll_test.cycle_flat ~radix:5 ~dims:wte_dims ~modulus:23 ~offset:(-11.5) ~stride:0.041
                  idx)
          in
          let coeff_dims = [| batch; seq; d_model |] in
          let coeff_vals =
            Array.init
              (batch * seq * d_model)
              ~f:(fun idx ->
                Ll_test.cycle_flat ~radix:3 ~dims:coeff_dims ~modulus:17 ~offset:1. ~stride:0.02 idx)
          in
          let f32_emb, f32_grad, _ =
            run_embedding_step ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals ~prec:None
          in
          let f16_emb, f16_grad, c =
            run_embedding_step ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:(Some (Ops.half, MP.Twin_virtual))
          in
          Stdio.eprintf
            "f16 %s (not part of the golden): gathers=%d rounded=%d scatters=%d vocab_loops=%d\n"
            label c.gathers c.rounded_gathers c.scatters c.vocab_loops;
          pf "f16 %s: forward lowers as rounded Get_dynamic and backward as Set_dynamic" label
            (c.gathers >= 1 && c.rounded_gathers >= 1 && c.scatters >= 1 && c.vocab_loops = 0);
          pf "f16 %s: forward and backward match f32 within f16 tolerance" label
            (Verdict.within "f16 emb" f16_emb f32_emb ~tol:f16_emb_tol
            && Verdict.within "f16 grad" f16_grad f32_grad ~tol:f16_grad_tol)));

  (* === Case 4: non-narrow cast twin (f64 -> f32 into f64 output, gated on supports_f64) === *)
  Verdict.case "non-narrow f64 -> f32 cast twin" (fun () ->
      let supports_f64 =
        (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.supports_f64
      in
      let leg = Verdict.leg ~when_:supports_f64 ~on:backend_name () in
      let outcome =
        if not supports_f64 then None
        else (
          Tensor.unsafe_reinitialize ();
          let vocab = 8 and d_model = 4 in
          let master_val = 1. +. (2. **. -25.) in
          let wte_dims = [| d_model; vocab |] in
          let wte_vals =
            Array.init (d_model * vocab) ~f:(fun idx ->
                if idx % vocab = 3 then master_val
                else Ll_test.cycle_flat ~radix:3 ~dims:wte_dims ~modulus:7 ~offset:2. ~stride:1. idx)
          in
          let ids = Nn_blocks.class_ids_of_int_list ~label:"ids_f64" [ 3; 5 ] in
          let saved_prec = !Tensor.default_value_prec in
          Tensor.default_value_prec := Ops.double;
          let wte_twin =
            Exn.protect
              ~f:(fun () ->
                MP.with_master_weights ~placement:MP.Twin_virtual ~master_prec:Ops.double
                  ~prec:Ops.single (fun () ->
                    TDSL.param ~values:wte_vals "wte_f64" ~input_dims:[ vocab ]
                      ~output_dims:[ d_model ] ()))
              ~finally:(fun () -> Tensor.default_value_prec := saved_prec)
          in
          let%op emb = wte_twin * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
          Tn.update_prec emb.Tensor.value Ops.double;
          Train.set_materialized emb.Tensor.value;
          Tn.set_observable emb.Tensor.value;
          let ctx = Train.init_params (Context.auto ()) IDX.empty emb in
          let ctx, routine, opt = compile_capturing ctx (Train.forward emb) in
          let ctx = Context.run ctx routine in
          let got = Context.get_values ctx emb.Tensor.value in
          let counts = inspect_ir ~vocab opt in
          Stdio.eprintf "f64->f32 (not part of the golden): row3[0]=%h rounded=%d\n" got.(0)
            counts.rounded_gathers;
          Some (counts, got, d_model))
      in
      in_leg ~leg "f64->f32 cast twin: lowers as Round_prec over Get_dynamic with no vocab loop"
        (fun claim ->
          let counts, _, _ = Option.value_exn outcome in
          p claim (counts.gathers >= 1 && counts.rounded_gathers >= 1 && counts.vocab_loops = 0));
      in_leg ~leg "f64->f32 cast twin: gathered value rounds to f32 (1.0, not f64 1 + 2^-25)"
        (fun claim ->
          let _, got, d_model = Option.value_exn outcome in
          p_all claim
            (Array.to_list (Array.sub got ~pos:0 ~len:d_model))
            ~f:(fun x -> Float.(x = 1.))));

  (* === Case 5: negative controls (hand-built IR) === *)
  Verdict.case "negative controls: unsafe selector scopes are not rewritten" (fun () ->
      (* 5a: A narrowing byte scope over an int32 id truncates 257 to 1 mod 256, so the dense loop
         matches k = 1 and returns row 1 (= 20.), whereas rewriting to Get_dynamic over int32 would
         see 257 >= 4 and return 0. *)
      let vocab = 4 in
      let tbl =
        Ll_test.node_factory ~prec:Ops.single ~first_id:29500 ~dims:[| vocab |] () "nc_tbl"
      in
      let ids = Ll_test.node_factory ~prec:Ops.int32 ~first_id:29501 ~dims:[| 1 |] () "nc_ids" in
      let sel = Ll_test.node_factory ~prec:Ops.byte ~first_id:29502 ~dims:[| vocab |] () "nc_sel" in
      let out = Ll_test.node_factory ~prec:Ops.single ~first_id:29503 ~dims:[| 1 |] () "nc_out" in
      List.iter [ tbl; ids; out ] ~f:Ll_test.materialize;
      Ll_test.virtualize sel;
      let k1 = Ll_test.sym () and k2 = Ll_test.sym () in
      let prog_narrow =
        LL.unflat_lines
          [
            Ll_test.loop_n k1 vocab
              (Ll_test.set sel
                 [| Ll_test.iter k1 |]
                 (LL.Binop
                    ( Ops.Cmpeq,
                      (LL.Embed_index (Ll_test.iter k1), Ops.byte),
                      (Ll_test.get ids [| Ll_test.fixed 0 |], Ops.byte) )));
            LL.Zero_out out;
            Ll_test.loop_n k2 vocab
              (Ll_test.set out
                 [| Ll_test.fixed 0 |]
                 (LL.Binop
                    ( Ops.Add,
                      (Ll_test.get out [| Ll_test.fixed 0 |], Ops.single),
                      ( LL.Binop
                          ( Ops.Mul,
                            (Ll_test.get tbl [| Ll_test.iter k2 |], Ops.single),
                            (Ll_test.get sel [| Ll_test.iter k2 |], Ops.byte) ),
                        Ops.single ) )));
          ]
      in
      let opt_narrow =
        Ll_test.optimize ~materialized:[ tbl; ids; out ] ~name:"nc_narrow_sel" prog_narrow
      in
      let c_narrow = inspect_ir ~vocab opt_narrow in
      p "narrowing byte selector over int32 IDs is NOT rewritten to Get_dynamic"
        (c_narrow.gathers = 0);
      let got_narrow =
        Ll_test.execute ~name:"nc_narrow_sel" opt_narrow
          ~seed:
            [ (tbl, [| 10.; 20.; 30.; 40. |]); (ids, [| 257. |]); (out, [| Ll_test.sentinel |]) ]
          ~read:[ out ]
      in
      p "narrowing byte selector preserves modular wrap semantics (257 mod 256 = 1 -> 20.)"
        (Ll_test.same got_narrow [ [| 20. |] ]);

      (* 5b: A bare Cmpeq evaluated in bfloat16 with vocab = 300 >= 256 would conflate adjacent
         indices above 256 on native bfloat16, so rewrite_one_hot_reductions must decline it. *)
      let vocab_bf = 300 in
      let tbl_bf =
        Ll_test.node_factory ~prec:Ops.bfloat16 ~first_id:29510 ~dims:[| vocab_bf |] () "nc_bf_tbl"
      in
      let ids_bf =
        Ll_test.node_factory ~prec:Ops.bfloat16 ~first_id:29511 ~dims:[| 1 |] () "nc_bf_ids"
      in
      let out_bf =
        Ll_test.node_factory ~prec:Ops.bfloat16 ~first_id:29512 ~dims:[| 1 |] () "nc_bf_out"
      in
      List.iter [ tbl_bf; ids_bf; out_bf ] ~f:Ll_test.materialize;
      let kb = Ll_test.sym () in
      let prog_bf =
        LL.unflat_lines
          [
            LL.Zero_out out_bf;
            Ll_test.loop_n kb vocab_bf
              (Ll_test.set out_bf
                 [| Ll_test.fixed 0 |]
                 (LL.Binop
                    ( Ops.Add,
                      (Ll_test.get out_bf [| Ll_test.fixed 0 |], Ops.bfloat16),
                      ( LL.Binop
                          ( Ops.Mul,
                            (Ll_test.get tbl_bf [| Ll_test.iter kb |], Ops.bfloat16),
                            ( LL.Binop
                                ( Ops.Cmpeq,
                                  (LL.Embed_index (Ll_test.iter kb), Ops.bfloat16),
                                  (Ll_test.get ids_bf [| Ll_test.fixed 0 |], Ops.bfloat16) ),
                              Ops.bfloat16 ) ),
                        Ops.bfloat16 ) )));
          ]
      in
      let opt_bf =
        Ll_test.optimize ~materialized:[ tbl_bf; ids_bf; out_bf ] ~name:"nc_bf_bare" prog_bf
      in
      let c_bf = inspect_ir ~vocab:vocab_bf opt_bf in
      p "bare bfloat16 Cmpeq with vocab >= 256 is NOT rewritten to Get_dynamic" (c_bf.gathers = 0);

      (* 5c: A Where selector whose condition evaluates at bfloat16 inside an f32 accumulator over
         vocab = 300 >= 256 must be checked at the condition precision (bfloat16), not the
         accumulator precision (single), for both forward gather and backward scatter. *)
      let tbl_w =
        Ll_test.node_factory ~prec:Ops.single ~first_id:29520 ~dims:[| vocab_bf |] () "nc_w_tbl"
      in
      let ids_w =
        Ll_test.node_factory ~prec:Ops.bfloat16 ~first_id:29521 ~dims:[| 1 |] () "nc_w_ids"
      in
      let out_w =
        Ll_test.node_factory ~prec:Ops.single ~first_id:29522 ~dims:[| 1 |] () "nc_w_out"
      in
      List.iter [ tbl_w; ids_w; out_w ] ~f:Ll_test.materialize;
      let kw = Ll_test.sym () in
      let prog_where_fwd =
        LL.unflat_lines
          [
            LL.Zero_out out_w;
            Ll_test.loop_n kw vocab_bf
              (Ll_test.set out_w
                 [| Ll_test.fixed 0 |]
                 (LL.Binop
                    ( Ops.Add,
                      (Ll_test.get out_w [| Ll_test.fixed 0 |], Ops.single),
                      ( LL.Ternop
                          ( Ops.Where,
                            ( LL.Binop
                                ( Ops.Cmpeq,
                                  (LL.Embed_index (Ll_test.iter kw), Ops.bfloat16),
                                  (Ll_test.get ids_w [| Ll_test.fixed 0 |], Ops.bfloat16) ),
                              Ops.bfloat16 ),
                            (Ll_test.get tbl_w [| Ll_test.iter kw |], Ops.single),
                            (LL.Constant 0., Ops.single) ),
                        Ops.single ) )));
          ]
      in
      let opt_where_fwd =
        Ll_test.optimize ~materialized:[ tbl_w; ids_w; out_w ] ~name:"nc_where_fwd" prog_where_fwd
      in
      let c_where_fwd = inspect_ir ~vocab:vocab_bf opt_where_fwd in
      p "Where with bfloat16 condition in f32 accumulator (vocab >= 256) is NOT rewritten to gather"
        (c_where_fwd.gathers = 0);

      let dtbl_w =
        Ll_test.node_factory ~prec:Ops.single ~first_id:29530 ~dims:[| vocab_bf |] () "nc_w_dtbl"
      in
      let grad_w =
        Ll_test.node_factory ~prec:Ops.single ~first_id:29531 ~dims:[| 1 |] () "nc_w_grad"
      in
      List.iter [ dtbl_w; grad_w ] ~f:Ll_test.materialize;
      let kwb = Ll_test.sym () in
      let prog_where_bwd =
        LL.unflat_lines
          [
            LL.Zero_out dtbl_w;
            Ll_test.loop_n kwb vocab_bf
              (Ll_test.set dtbl_w
                 [| Ll_test.iter kwb |]
                 (LL.Binop
                    ( Ops.Add,
                      (Ll_test.get dtbl_w [| Ll_test.iter kwb |], Ops.single),
                      ( LL.Ternop
                          ( Ops.Where,
                            ( LL.Binop
                                ( Ops.Cmpeq,
                                  (LL.Embed_index (Ll_test.iter kwb), Ops.bfloat16),
                                  (Ll_test.get ids_w [| Ll_test.fixed 0 |], Ops.bfloat16) ),
                              Ops.bfloat16 ),
                            (Ll_test.get grad_w [| Ll_test.fixed 0 |], Ops.single),
                            (LL.Constant 0., Ops.single) ),
                        Ops.single ) )));
          ]
      in
      let opt_where_bwd =
        Ll_test.optimize ~materialized:[ dtbl_w; ids_w; grad_w ] ~name:"nc_where_bwd" prog_where_bwd
      in
      let c_where_bwd = inspect_ir ~vocab:vocab_bf opt_where_bwd in
      p
        "Where with bfloat16 condition in f32 accumulator (vocab >= 256) is NOT rewritten to \
         scatter"
        (c_where_bwd.scatters = 0))
