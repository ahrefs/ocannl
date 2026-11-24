(* gh-ocannl-2444: a narrow-float [Set_dynamic] embedding scatter accumulates repeated token hits at
   the backend's accumulator residency ([accum_prec]) and rounds once into the narrow store.

   In bf16 mixed-precision training (cast twin [wte_twin] over f32 master [wte]), backprop lowers
   the embedding lookup gradient to a [Set_dynamic] scatter into narrow [wte_twin.grad] before
   adding into f32 [wte.grad]. Because a scatter's row is data-dependent, neither
   [sink_narrow_reduction_loops] nor codegen's [peel_accum_nest] can hold it in a scalar local;
   [Low_level.widen_narrow_scatters] brackets each allowlisted narrow add-scatter in an [accum_prec]
   buffer [W] ([W = T; scatter(W); T = W]).

   Pinned invariants: - Repeated-token bf16 cast-twin parity ([Twin_virtual] and
   [Twin_materialized]) at [vocab = 264] (above bf16's 256 exact-int limit, exercising scoped uint32
   selectors too): bitwise equal to the materialized f32 reference narrowed once, with non-vacuity
   ([per_hit <> once_rounded]) across big-first, small-first, negative, single-hit, untouched, and
   OOB rows, plus structural verification that [Set_dynamic] targets an f32 node in
   [scatter_accum_namespace]. - Tied two-consumer bf16 cast twin ([lm_head]-style dense projection +
   embedding scatter sharing [wte_twin], mirroring [gpt2_mini_train]): nonzero gradient from the
   first consumer is loaded into [W] losslessly and combined in f32 with repeated scatter hits
   before one bf16 rounding. - Direct narrow table gradient with nonzero incoming values
   ([Train.grad_update ~accum_steps:1]). - [Schedule.Split_reduce] composition on the widened
   scatter: minted [partials] is f32 and executed output matches the default bitwise. - FP16 cast
   twin under [Fp16_wide] (scaled to 2048 where fp16 ULP = 2). - [Bf16_narrow] policy respect:
   widens where the backend's [accum_prec] widens (cc, CUDA) and keeps per-hit bf16 with no
   [scatter_acc] buffer where [accum_prec] stays narrow (HIP, Metal). - Hand-built IR allowlist and
   decline negative controls for [Low_level.widen_narrow_scatters]. *)

(* cc-fp-contract: exact-operands -- every input is a small dyadic float whose products and f32 sums
   fit the f32 significand exactly. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Sched = Ir.Schedule
module MP = Mixed_prec

let bf16 = Ops.bfloat16
let f16 = Ops.half
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let param_by_label root name =
  List.find_exn (Set.to_list root.Tensor.params) ~f:(fun t ->
      String.equal (Tn.label_name t.Tensor.value) name)

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

let compile_routine ~name ?lowered_transform ctx comp =
  Context.compile ~name ?lowered_transform ctx comp Idx.Empty

let scatter_targets_of_ll (llc : LL.t) : Tn.t list =
  let acc = ref [] in
  Ll_test.walk llc ~on_stmt:(function LL.Set_dynamic { tn; _ } -> acc := tn :: !acc | _ -> ());
  List.rev !acc

let scatter_targets (routine : Context.routine) : Tn.t list =
  List.concat_map routine.Context.segments ~f:(fun seg -> scatter_targets_of_ll seg.LL.llc)

let digits label values =
  Stdio.eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (List.map (List.take (Array.to_list values) 8) ~f:(fun v -> Printf.sprintf "%h" v)))

(* Construct a discriminating repeated-token fixture for a narrow format whose ULP at [base] is 2
   ([base = 256.] for bf16, [base = 2048.] for fp16). For each channel [d]: - [scale = 2^(d mod 4)],
   [big = base * scale] (ULP = 2*scale, half-ULP = scale), [tiny = 0.5 * scale] (strictly below
   half-ULP, so [big + tiny] rounds back to [big] per hit). - Four [tiny] hits sum to [2 * scale] =
   one full ULP, so [big + 4 * tiny = (base + 2) * scale] is exact in the narrow format (no tie!)
   and differs from per-hit [big]. *)
let make_fixture ~vocab ~d_model ~base =
  let batch = 2 and seq = 12 in
  (* 24 positions exercising big-first, small-first+big+small, negative big-first, single-hit,
     above-256 repeated token (257), untouched rows (e.g. 261), and OOB id ([vocab]). *)
  let pos_ids =
    [|
      (* big-first on token 2 *)
      2;
      2;
      2;
      2;
      2;
      (* small-first, then big, then small on token 5 *)
      5;
      5;
      5;
      5;
      5;
      5;
      5;
      5;
      5;
      (* negative big-first on token 7 *)
      7;
      7;
      7;
      7;
      7;
      (* repeated above-256 token 257 (falls back to 11 when vocab <= 257) *)
      (if vocab > 257 then 257 else 11);
      (if vocab > 257 then 257 else 11);
      (if vocab > 257 then 257 else 11);
      (* single-hit token 9 and out-of-range id [vocab] *)
      9;
      vocab;
    |]
  in
  let ids_rows = Array.init batch ~f:(fun b -> Array.sub pos_ids ~pos:(b * seq) ~len:seq) in
  let scale d = 2. **. Float.of_int (d % 4) in
  let coeff_at pos d =
    let s = scale d in
    let big = base *. s and tiny = 0.5 *. s in
    match pos with
    | 0 -> big
    | 1 | 2 | 3 | 4 -> tiny
    | 5 | 6 | 7 | 8 -> tiny
    | 9 -> big
    | 10 | 11 | 12 | 13 -> tiny
    | 14 -> -.big
    | 15 | 16 | 17 | 18 -> -.tiny
    | 19 -> big
    | 20 | 21 -> tiny +. (0.25 *. s)
    | 22 -> 3.5 *. s
    | 23 -> 100. *. s (* OOB position: must contribute nothing *)
    | _ -> 0.
  in
  let coeff_vals =
    Array.init
      (batch * seq * d_model)
      ~f:(fun idx ->
        let pos = idx / d_model and d = idx % d_model in
        coeff_at pos d)
  in
  let wte_vals =
    Array.init (d_model * vocab) ~f:(fun idx ->
        let d = idx / vocab and v = idx % vocab in
        Ll_test.weighted ~weights:[| 3; 5 |] ~modulus:9 ~offset:(-4.) ~stride:0.25 [| d; v |])
  in
  (batch, seq, ids_rows, coeff_vals, wte_vals)

let host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init ~step =
  let batch = Array.length ids_rows and seq = Array.length ids_rows.(0) in
  let grad = Array.copy init in
  for b = 0 to batch - 1 do
    for s = 0 to seq - 1 do
      let pos = (b * seq) + s in
      let id = ids_rows.(b).(s) in
      if id >= 0 && id < vocab then
        for d = 0 to d_model - 1 do
          let g_idx = (d * vocab) + id in
          grad.(g_idx) <- step (grad.(g_idx) +. coeff_vals.((pos * d_model) + d))
        done
    done
  done;
  grad

let run_cast_twin_step ~name ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals ~prec
    ?lowered_transform () =
  Tensor.unsafe_reinitialize ();
  let batch = Array.length ids_rows and seq = Array.length ids_rows.(0) in
  let ids = ids_tensor_2d ~label:"nsr_ids" ids_rows in
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
  let coeff =
    TDSL.ndarray coeff_vals ~label:[ "nsr_coeff" ] ~batch_dims:[ batch; seq ]
      ~output_dims:[ d_model ] ()
  in
  let%op loss = (emb *. coeff) ++ "...|... => |->0" in
  Option.iter prec ~f:(fun (p, _) ->
      Precision_policy.apply
        ~except:(fun tn -> Tn.equal tn loss.Tensor.value || Tn.equal tn coeff.Tensor.value)
        { Precision_policy.param_prec = None; activation_prec = Some p; grad_prec = Some p }
        loss);
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Idx.Empty loss in
  let ctx, routine = compile_routine ~name ?lowered_transform ctx update in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let wte_master = param_by_label loss "wte" in
  (Context.get_values ctx (grad_of wte_master), routine)

(* === Case 1: bf16 cast-twin embedding (Twin_virtual & Twin_materialized, vocab = 264) === *)
let () =
  Test_utils.with_policy
    (fun policy ->
      { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_auto; narrow_compute_f32 = true })
    (fun () ->
      case "bf16 cast-twin repeated-token scatter parity" (fun () ->
          let vocab = 264 and d_model = 8 in
          let _batch, _seq, ids_rows, coeff_vals, wte_vals =
            make_fixture ~vocab ~d_model ~base:256.
          in
          let zeros = Array.create ~len:(d_model * vocab) 0. in
          let exact = host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros ~step:Fn.id in
          let once = Array.map exact ~f:(Ops.round_to_prec bf16) in
          let per_hit =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros
              ~step:(Ops.round_to_prec bf16)
          in
          let f32_grad, _ =
            run_cast_twin_step ~name:"nsr_c1_f32" ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:None ()
          in
          digits "bf16 case1 f32" f32_grad;
          p_all2 "f32 baseline wte.grad equals exact host sums bitwise" f32_grad exact ~f:bitwise;
          p_exists "per-hit bf16 rounding differs from one final rounding (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit) (Array.to_list once))
            ~f:(fun (a, b) -> not (bitwise a b));
          List.iter
            [
              ("virtual twin", "nsr_c1_virt", MP.Twin_virtual);
              ("materialized twin", "nsr_c1_mat", MP.Twin_materialized);
            ]
            ~f:(fun (label, rname, placement) ->
              let bf_grad, routine =
                run_cast_twin_step ~name:rname ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
                  ~prec:(Some (bf16, placement))
                  ()
              in
              digits ("bf16 case1 " ^ label) bf_grad;
              let expected = Array.map f32_grad ~f:(Ops.round_to_prec bf16) in
              let targets = scatter_targets routine in
              p_all (Printf.sprintf "%s: every Set_dynamic targets an f32 scatter_acc buffer" label)
                targets ~f:(fun tn ->
                  String.equal tn.Tn.namespace LL.scatter_accum_namespace
                  && Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single);
              pf_all2 "%s: bf16 wte.grad equals f32 reference narrowed once, cell for cell" label
                bf_grad expected ~f:bitwise)))

(* === Case 2: tied two-consumer bf16 cast twin (lm_head + embedding scatter, gpt2_mini_train shape)
   === *)
let () =
  Test_utils.with_policy
    (fun policy ->
      { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_auto; narrow_compute_f32 = true })
    (fun () ->
      case "tied two-consumer bf16 cast twin (lm_head + embedding scatter)" (fun () ->
          let vocab = 16 and d_model = 8 and batch = 1 and seq = 8 in
          (* Embedding scatter hits token 3 four times with tiny = 0.5*scale, while the tied lm_head
             projection contributes big = 256*scale to token 3 (and distinct values to other rows).
             Per-hit bf16 scatter loses all four embedding hits on top of lm_head's 256*scale. *)
          let ids_rows = [| [| 3; 3; 3; 3; 6; 6; 6; vocab |] |] in
          let scale d = 2. **. Float.of_int (d % 4) in
          let emb_coeff_vals =
            Array.init
              (batch * seq * d_model)
              ~f:(fun idx ->
                let pos = idx / d_model and d = idx % d_model in
                let s = scale d in
                match pos with
                | 0 | 1 | 2 | 3 -> 0.5 *. s
                | 4 | 5 | 6 -> 0.5 *. s
                | 7 -> 50. *. s (* OOB *)
                | _ -> 0.)
          in
          (* hfinal[0,0,d] = scale d and lm_coeff[0,0,v] = 256. on v in {3, 6}, 4. on v = 11
             (untouched by scatter), and 0. elsewhere. *)
          let hfinal_vals = Array.init d_model ~f:scale in
          let lm_coeff_vals =
            Array.init vocab ~f:(fun v -> match v with 3 | 6 -> 256. | 11 -> 4. | _ -> 0.)
          in
          let wte_vals =
            Array.init (d_model * vocab) ~f:(fun idx ->
                Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:(-3.) ~stride:0.25
                  [| idx / vocab; idx % vocab |])
          in
          let run ~prec =
            Tensor.unsafe_reinitialize ();
            let ids = ids_tensor_2d ~label:"tied_ids" ids_rows in
            let onehot = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
            let make_wte () =
              TDSL.param ~values:wte_vals "wte" ~input_dims:[ vocab ] ~output_dims:[ d_model ] ()
            in
            let wte_used =
              match prec with
              | None -> make_wte ()
              | Some p -> MP.with_master_weights ~placement:MP.Twin_virtual ~prec:p make_wte
            in
            let%op emb = wte_used * onehot in
            let emb_coeff =
              TDSL.ndarray emb_coeff_vals ~label:[ "tied_emb_c" ] ~batch_dims:[ batch; seq ]
                ~output_dims:[ d_model ] ()
            in
            let hfinal =
              TDSL.ndarray hfinal_vals ~label:[ "tied_h" ] ~batch_dims:[ 1; 1 ]
                ~output_dims:[ d_model ] ()
            in
            let lm_coeff =
              TDSL.ndarray lm_coeff_vals ~label:[ "tied_lm_c" ] ~batch_dims:[ 1; 1 ]
                ~output_dims:[ vocab ] ()
            in
            let%op logits = wte_used +* "|v -> d; ... | d => ... | v" hfinal in
            let%op loss =
              (emb *. emb_coeff) ++ "...|... => |->0" + ((logits *. lm_coeff) ++ "...|... => |->0")
            in
            Option.iter prec ~f:(fun p ->
                Precision_policy.apply
                  ~except:(fun tn ->
                    Tn.equal tn loss.Tensor.value
                    || Tn.equal tn emb_coeff.Tensor.value
                    || Tn.equal tn hfinal.Tensor.value || Tn.equal tn lm_coeff.Tensor.value)
                  {
                    Precision_policy.param_prec = None;
                    activation_prec = Some p;
                    grad_prec = Some p;
                  }
                  loss);
            let update = Train.grad_update loss in
            let ctx = Context.auto () in
            let ctx = Train.init_params ctx Idx.Empty loss in
            let ctx, routine =
              compile_routine
                ~name:(match prec with None -> "nsr_c2_f32" | Some _ -> "nsr_c2_bf16")
                ctx update
            in
            let ctx = Context.run ctx routine in
            Context.sync ctx;
            let wte_master = param_by_label loss "wte" in
            Context.get_values ctx (grad_of wte_master)
          in
          let lm_init =
            Array.init (d_model * vocab) ~f:(fun idx ->
                let d = idx / vocab and v = idx % vocab in
                hfinal_vals.(d) *. lm_coeff_vals.(v))
          in
          let exact =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals:emb_coeff_vals ~init:lm_init
              ~step:Fn.id
          in
          let once = Array.map exact ~f:(Ops.round_to_prec bf16) in
          let per_hit =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals:emb_coeff_vals ~init:lm_init
              ~step:(Ops.round_to_prec bf16)
          in
          let f32_grad = run ~prec:None in
          let bf_grad = run ~prec:(Some bf16) in
          digits "tied f32" f32_grad;
          digits "tied bf16" bf_grad;
          p_all2 "tied two-consumer: f32 wte.grad equals exact host sum" f32_grad exact ~f:bitwise;
          p_exists "tied two-consumer: per-hit bf16 differs from once-rounded f32 (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit) (Array.to_list once))
            ~f:(fun (a, b) -> not (bitwise a b));
          p_all2 "tied two-consumer: bf16 wte.grad equals f32 narrowed once, cell for cell" bf_grad
            once ~f:bitwise))

(* === Case 3: direct narrow embedding gradient with nonzero incoming values (accum_steps:1) === *)
let () =
  Test_utils.with_policy
    (fun policy ->
      { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_auto; narrow_compute_f32 = true })
    (fun () ->
      case "nonzero incoming gradient (accum_steps:1, direct narrow table)" (fun () ->
          let vocab = 12 and d_model = 8 and batch = 1 and seq = 8 in
          let ids_rows = [| [| 2; 2; 2; 2; 5; 5; 7; vocab |] |] in
          let scale d = 2. **. Float.of_int (d % 4) in
          let incoming =
            Array.init (d_model * vocab) ~f:(fun idx ->
                let d = idx / vocab and v = idx % vocab in
                if v = 2 then 256. *. scale d
                else if v = 5 then -256. *. scale d
                else
                  Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:(-3.) ~stride:0.5
                    [| d; v |])
          in
          let coeff_vals =
            Array.init
              (batch * seq * d_model)
              ~f:(fun idx ->
                let pos = idx / d_model and d = idx % d_model in
                let s = scale d in
                match pos with
                | 0 | 1 | 2 | 3 -> 0.5 *. s
                | 4 | 5 -> -0.5 *. s
                | 6 -> 1.5 *. s
                | 7 -> 99. *. s (* OOB *)
                | _ -> 0.)
          in
          let run prec =
            Tensor.unsafe_reinitialize ();
            let ids = ids_tensor_2d ~label:"inc_ids" ids_rows in
            let onehot = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
            let wte =
              Operation.init ~l:"inc_wte" ~prec:Ops.single ~i:[ vocab ] ~o:[ d_model ]
                ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:0. ~stride:0.25)
                ~grad_spec:Tensor.Require_grad ()
            in
            let%op emb = wte * onehot in
            let coeff =
              TDSL.ndarray coeff_vals ~label:[ "inc_coeff" ] ~batch_dims:[ batch; seq ]
                ~output_dims:[ d_model ] ()
            in
            let%op loss = (emb *. coeff) ++ "...|... => |->0" in
            let wte_g = grad_of wte in
            Tn.update_prec wte_g prec;
            Train.set_materialized wte_g;
            let update = Train.grad_update ~accum_steps:1 loss in
            let ctx = Context.auto () in
            let ctx, routine =
              compile_routine ~name:("nsr_c3_" ^ Ops.prec_string prec) ctx update
            in
            let ctx = Context.set_values ctx wte_g incoming in
            let ctx = Context.run ctx routine in
            Context.sync ctx;
            Context.get_values ctx wte_g
          in
          let exact =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:incoming ~step:Fn.id
          in
          let once = Array.map exact ~f:(Ops.round_to_prec bf16) in
          let per_hit =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:incoming
              ~step:(Ops.round_to_prec bf16)
          in
          let f32_grad = run Ops.single in
          let bf_grad = run bf16 in
          p_all2 "accum_steps:1 f32 gradient equals exact host sum onto incoming" f32_grad exact
            ~f:bitwise;
          p_exists "accum_steps:1 per-hit bf16 differs from once-rounded f32 (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit) (Array.to_list once))
            ~f:(fun (a, b) -> not (bitwise a b));
          p_all2 "accum_steps:1 bf16 gradient equals f32 narrowed once, untouched rows preserved"
            bf_grad once ~f:bitwise))

(* === Case 4: Schedule.Split_reduce composition on the widened scatter === *)
let () =
  Test_utils.with_policy
    (fun policy ->
      { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_auto; narrow_compute_f32 = true })
    (fun () ->
      case "Schedule.Split_reduce composes with widened bf16 scatter" (fun () ->
          let vocab = 16 and d_model = 8 in
          let _batch, _seq, ids_rows, coeff_vals, wte_vals =
            make_fixture ~vocab ~d_model ~base:256.
          in
          let default_grad, _ =
            run_cast_twin_step ~name:"nsr_c4_default" ~vocab ~d_model ~ids_rows ~wte_vals
              ~coeff_vals
              ~prec:(Some (bf16, MP.Twin_virtual))
              ()
          in
          let widened_target = ref None in
          let partials_target = ref None in
          let facts = Context.backend_facts (Context.auto ()) in
          let preset opt =
            if Ir.Backend_intf.binds_hardware_grid facts then Sched.default_gpu ~min_parallel:1 opt
            else if Ir.Backend_intf.renders_on_host_pool facts then
              Sched.default_cpu ~min_work:0 opt
            else []
          in
          let split_grad, _ =
            run_cast_twin_step ~name:"nsr_c4_split" ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:(Some (bf16, MP.Twin_virtual))
              ~lowered_transform:(fun opt ->
                (* Locate the widened scatter statement and its outermost serial position loop. *)
                let scatter_stmt =
                  List.find_exn (LL.flat_lines [ opt.LL.llc ]) ~f:(fun st ->
                      not (List.is_empty (scatter_targets_of_ll st)))
                in
                let target = List.hd_exn (scatter_targets_of_ll scatter_stmt) in
                widened_target := Some target;
                let outer_sym =
                  match Ll_test.nest_paths scatter_stmt with
                  | (s :: _) :: _ -> s
                  | _ -> failwith "expected enclosing loop around widened scatter"
                in
                let sr, _, _, _ = Sched.split_reduce ~axis:outer_sym ~target ~num_blocks:2 in
                let applied = Sched.apply [ sr ] opt in
                Ll_test.walk applied.LL.llc ~on_stmt:(function
                  | LL.Set_dynamic { tn; _ } -> partials_target := Some tn
                  | _ -> ());
                Sched.fission_scheduled ~preset ~zero_sched:(fun _ -> []) ~static_indices:[] applied
                |> List.map ~f:(fun (_, _, _, scheduled) -> scheduled))
              ()
          in
          p "Split_reduce over widened scatter redirects Set_dynamic to an f32 partials buffer"
            (match (!widened_target, !partials_target) with
            | Some orig, Some part ->
                let dims = Lazy.force part.Tn.dims in
                (not (Tn.equal orig part))
                && List.mem part.Tn.label "partials" ~equal:String.equal
                && Array.length dims > 0
                && dims.(0) = 2
                && Ops.equal_prec (Lazy.force part.Tn.storage_prec) Ops.single
            | _ -> false);
          p_all2 "Split_reduce over widened scatter matches default bf16 output bitwise" split_grad
            default_grad ~f:bitwise))

(* === Case 5: FP16 cast twin under Fp16_wide === *)
let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_wide })
    (fun () ->
      case "fp16 cast-twin repeated-token scatter parity under Fp16_wide" (fun () ->
          let vocab = 16 and d_model = 8 in
          let _batch, _seq, ids_rows, coeff_vals, wte_vals =
            make_fixture ~vocab ~d_model ~base:2048.
          in
          let zeros = Array.create ~len:(d_model * vocab) 0. in
          let exact = host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros ~step:Fn.id in
          let once = Array.map exact ~f:(Ops.round_to_prec f16) in
          let per_hit =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros
              ~step:(Ops.round_to_prec f16)
          in
          let f32_grad, _ =
            run_cast_twin_step ~name:"nsr_c5_f32" ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:None ()
          in
          let f16_grad, routine =
            run_cast_twin_step ~name:"nsr_c5_f16" ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:(Some (f16, MP.Twin_virtual))
              ()
          in
          digits "fp16 case5" f16_grad;
          p_all2 "fp16 Fp16_wide: f32 baseline equals exact host sums" f32_grad exact ~f:bitwise;
          p_exists "fp16 Fp16_wide: per-hit fp16 differs from one final rounding (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit) (Array.to_list once))
            ~f:(fun (a, b) -> not (bitwise a b));
          p_all "fp16 Fp16_wide: every Set_dynamic targets an f32 scatter_acc buffer"
            (scatter_targets routine) ~f:(fun tn ->
              String.equal tn.Tn.namespace LL.scatter_accum_namespace
              && Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single);
          p_all2 "fp16 Fp16_wide: wte.grad equals f32 reference narrowed once, cell for cell"
            f16_grad once ~f:bitwise))

(* === Case 6: Bf16_narrow policy respect === *)
let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_narrow })
    (fun () ->
      case "Bf16_narrow respects backend accum_prec" (fun () ->
          let vocab = 16 and d_model = 8 in
          let _batch, _seq, ids_rows, coeff_vals, wte_vals =
            make_fixture ~vocab ~d_model ~base:256.
          in
          let zeros = Array.create ~len:(d_model * vocab) 0. in
          let exact = host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros ~step:Fn.id in
          let once = Array.map exact ~f:(Ops.round_to_prec bf16) in
          let per_hit =
            host_scatter ~vocab ~d_model ~ids_rows ~coeff_vals ~init:zeros
              ~step:(Ops.round_to_prec bf16)
          in
          let backend_name =
            String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
          in
          let expected_narrows =
            match backend_name with
            | "hip" | "metal" -> true
            | "cc" | "multidev_cc" | "cuda" -> false
            | other -> failwith ("unexpected backend: " ^ other)
          in
          let acc_bf16 =
            (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.accum_prec bf16
          in
          let narrows = Ops.equal_prec acc_bf16 bf16 in
          let bf_grad, routine =
            run_cast_twin_step ~name:"nsr_c6_narrow" ~vocab ~d_model ~ids_rows ~wte_vals ~coeff_vals
              ~prec:(Some (bf16, MP.Twin_virtual))
              ()
          in
          let targets = scatter_targets routine in
          Stdio.eprintf "Bf16_narrow (not part of the golden): backend=%s narrows=%b targets=%d\n%!"
            backend_name narrows (List.length targets);
          p "Bf16_narrow: backend accum_prec matches the backend policy table"
            (Bool.equal narrows expected_narrows);
          let narrow_leg =
            Verdict.leg ~when_:expected_narrows ~on:"backends whose Bf16_narrow widens" ()
          in
          let wide_leg =
            Verdict.leg ~when_:(not expected_narrows) ~on:"backends whose Bf16_narrow narrows" ()
          in
          in_leg ~leg:narrow_leg
            "Bf16_narrow (narrow backend): every Set_dynamic stays bf16 with no scatter_acc buffer"
            (fun label ->
              p_all label targets ~f:(fun tn ->
                  Ops.equal_prec (Lazy.force tn.Tn.storage_prec) bf16
                  && not (String.equal tn.Tn.namespace LL.scatter_accum_namespace)));
          in_leg ~leg:narrow_leg
            "Bf16_narrow (narrow backend): executed bf16 gradient matches per-hit reference"
            (fun label -> p_all2 label bf_grad per_hit ~f:bitwise);
          in_leg ~leg:wide_leg
            "Bf16_narrow (wide backend): every Set_dynamic targets an f32 scatter_acc buffer"
            (fun label ->
              p_all label targets ~f:(fun tn ->
                  Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single
                  && String.equal tn.Tn.namespace LL.scatter_accum_namespace));
          in_leg ~leg:wide_leg
            "Bf16_narrow (wide backend): executed bf16 gradient matches once-narrowed f32 reference"
            (fun label -> p_all2 label bf_grad once ~f:bitwise)))

(* === Case 7: hand-built IR allowlist and decline negative controls === *)
let () =
  case "widen_narrow_scatters allowlist and decline negative controls" (fun () ->
      let open Ll_test in
      let widen_all p = if Ops.is_narrow_float p then Ops.single else p in
      let mk_store () = (Tn.Placements.create (), Hashtbl.create (module Tn)) in
      let tbl_bf = node_factory ~prec:bf16 ~first_id:24440 ~dims:[| 4; 8 |] () "nsr_tbl_bf" in
      let tbl_f32 =
        node_factory ~prec:Ops.single ~first_id:24441 ~dims:[| 4; 8 |] () "nsr_tbl_f32"
      in
      let ids = node_factory ~prec:Ops.uint32 ~first_id:24442 ~dims:[| 6 |] () "nsr_ids_hb" in
      let grad = node_factory ~prec:bf16 ~first_id:24443 ~dims:[| 6; 4 |] () "nsr_grad_hb" in
      let pos = sym () and d = sym () in
      let mk_scatter ?(tn = tbl_bf) ?(to_pos = 5) ?(extra_body = fun s -> s)
          ?(llsc_of =
            fun tn ->
              let p = Lazy.force tn.Tn.storage_prec in
              let self =
                LL.Get_dynamic
                  {
                    tn;
                    idcs = [| iter d; fixed 0 |];
                    dyn_axis = 1;
                    dyn_value = (get ids [| iter pos |], Ops.uint32);
                  }
              in
              LL.Binop (Ops.Add, (self, p), (get grad [| iter pos; iter d |], p))) () =
        let sd =
          LL.Set_dynamic
            {
              tn;
              idcs = [| iter d; fixed 0 |];
              dyn_axis = 1;
              dyn_value = (get ids [| iter pos |], Ops.uint32);
              llsc = llsc_of tn;
              debug = "";
            }
        in
        LL.For_loop
          {
            index = pos;
            from_ = 0;
            to_ = to_pos;
            axis = LL.Serial;
            body = loop_n d 4 (extra_body sd);
          }
      in
      (* 7a: Identity accum_prec leaves IR physically unchanged. *)
      let valid = mk_scatter () in
      let plc, store = mk_store () in
      let id_res = LL.widen_narrow_scatters ~accum_prec:Fn.id plc store valid in
      p "identity accum_prec leaves scatter IR physically unchanged" (phys_equal id_res valid);
      (* 7b: f32 target is left physically unchanged. *)
      let f32_sc = mk_scatter ~tn:tbl_f32 () in
      let plc, store = mk_store () in
      let f32_res = LL.widen_narrow_scatters ~accum_prec:widen_all plc store f32_sc in
      p "f32 scatter target is left physically unchanged" (phys_equal f32_res f32_sc);
      (* 7c: Routine logging declines. *)
      let saved_log = Utils.settings.debug_log_from_routines in
      let saved_lvl = Utils.settings.log_level in
      let logged_res =
        Exn.protect
          ~f:(fun () ->
            Utils.settings.debug_log_from_routines <- true;
            Utils.settings.log_level <- 2;
            let plc, store = mk_store () in
            LL.widen_narrow_scatters ~accum_prec:widen_all plc store valid)
          ~finally:(fun () ->
            Utils.settings.debug_log_from_routines <- saved_log;
            Utils.settings.log_level <- saved_lvl)
      in
      p "debug_log_from_routines declines widening" (phys_equal logged_res valid);
      (* 7d: Non-add Set_dynamic (plain overwrite) declines. *)
      let overwrite = mk_scatter ~llsc_of:(fun _ -> LL.Constant 1.0) () in
      let plc, store = mk_store () in
      p "non-add Set_dynamic overwrite declines widening"
        (phys_equal (LL.widen_narrow_scatters ~accum_prec:widen_all plc store overwrite) overwrite);
      (* 7e: Statement reading T elsewhere declines. *)
      let extra_read =
        mk_scatter
          ~llsc_of:(fun tn ->
            let p = Lazy.force tn.Tn.storage_prec in
            let self =
              LL.Get_dynamic
                {
                  tn;
                  idcs = [| iter d; fixed 0 |];
                  dyn_axis = 1;
                  dyn_value = (get ids [| iter pos |], Ops.uint32);
                }
            in
            LL.Binop (Ops.Add, (self, p), (get tn [| fixed 0; fixed 0 |], p)))
          ()
      in
      let plc, store = mk_store () in
      p "scatter reading target T elsewhere declines widening"
        (phys_equal
           (LL.widen_narrow_scatters ~accum_prec:widen_all plc store extra_read)
           extra_read);
      (* 7f: Dead enclosing loop declines. *)
      let dead = mk_scatter ~to_pos:(-1) () in
      let plc, store = mk_store () in
      p "dead enclosing loop declines widening"
        (phys_equal (LL.widen_narrow_scatters ~accum_prec:widen_all plc store dead) dead);
      (* 7g: RNG-bearing update declines. *)
      let rng_sc =
        mk_scatter
          ~llsc_of:(fun tn ->
            let p = Lazy.force tn.Tn.storage_prec in
            let self =
              LL.Get_dynamic
                {
                  tn;
                  idcs = [| iter d; fixed 0 |];
                  dyn_axis = 1;
                  dyn_value = (get ids [| iter pos |], Ops.uint32);
                }
            in
            let draw =
              LL.Unop (Ops.Uint4x32_to_prec_uniform1, (LL.Constant_bits 1L, Ops.uint4x32))
            in
            LL.Binop (Ops.Add, (self, p), (draw, p)))
          ()
      in
      let plc, store = mk_store () in
      p "RNG-bearing scatter declines widening"
        (phys_equal (LL.widen_narrow_scatters ~accum_prec:widen_all plc store rng_sc) rng_sc);
      (* 7h: Valid bf16 scatter (even with CSE-hoisted sibling local) rewrites to 3 statements and
         memoizes the same f32 scatter_acc node across specializations. *)
      let tmp = node_factory ~prec:bf16 ~first_id:24444 ~dims:[| 1 |] () "nsr_cse_tmp" in
      let tmp_id = LL.get_scope tmp in
      let with_cse =
        mk_scatter ~extra_body:(fun sd -> LL.Seq (LL.Set_local (tmp_id, LL.Constant 0.), sd)) ()
      in
      let plc1, store1 = mk_store () in
      let r1 = LL.widen_narrow_scatters ~accum_prec:widen_all plc1 store1 with_cse in
      let plc2, store2 = mk_store () in
      let r2 = LL.widen_narrow_scatters ~accum_prec:widen_all plc2 store2 valid in
      let t1 = scatter_targets_of_ll r1 and t2 = scatter_targets_of_ll r2 in
      p
        "valid bf16 scatter (including CSE-hoisted local) rewrites to 3 statements with memoized \
         f32 node"
        (List.length (LL.flat_lines [ r1 ]) = 3
        &&
        match (t1, t2) with
        | [ w1 ], [ w2 ] ->
            Tn.equal w1 w2
            && String.equal w1.Tn.namespace LL.scatter_accum_namespace
            && Ops.equal_prec (Lazy.force w1.Tn.storage_prec) Ops.single
            && Hashtbl.mem store1 w1
        | _ -> false);
      (* 7i: FP8 E5M2 target declines widening (reverse cast saturates ±inf on untouched cells). *)
      let tbl_fp8 = node_factory ~prec:Ops.fp8 ~first_id:24445 ~dims:[| 4; 8 |] () "nsr_tbl_fp8" in
      let fp8_sc = mk_scatter ~tn:tbl_fp8 () in
      let plc, store = mk_store () in
      p "fp8_e5m2 scatter target declines widening"
        (phys_equal (LL.widen_narrow_scatters ~accum_prec:widen_all plc store fp8_sc) fp8_sc);
      (* 7j: Partially-initialized routine Local table declines widening, while fully zeroed Local
         table widens cleanly. *)
      let tbl_loc_part =
        node_factory ~prec:bf16 ~first_id:24446 ~dims:[| 4; 8 |] () "nsr_tbl_loc_part"
      in
      let j_init = sym () in
      let partial_init =
        loop_n j_init 8 (set tbl_loc_part [| fixed 0; iter j_init |] (LL.Constant 0.))
      in
      let slice_sc =
        LL.For_loop
          {
            index = pos;
            from_ = 0;
            to_ = 5;
            axis = LL.Serial;
            body =
              LL.Set_dynamic
                {
                  tn = tbl_loc_part;
                  idcs = [| fixed 0; fixed 0 |];
                  dyn_axis = 1;
                  dyn_value = (get ids [| iter pos |], Ops.uint32);
                  llsc =
                    LL.Binop
                      ( Ops.Add,
                        ( LL.Get_dynamic
                            {
                              tn = tbl_loc_part;
                              idcs = [| fixed 0; fixed 0 |];
                              dyn_axis = 1;
                              dyn_value = (get ids [| iter pos |], Ops.uint32);
                            },
                          bf16 ),
                        (get grad [| iter pos; fixed 0 |], bf16) );
                  debug = "";
                };
          }
      in
      let partial_prog = LL.Seq (partial_init, slice_sc) in
      let plc_part, store_part = mk_store () in
      Tn.Placements.update plc_part tbl_loc_part Tn.Local Tn.Visit_cap;
      p "partially-initialized Local scatter target declines widening"
        (phys_equal
           (LL.widen_narrow_scatters ~accum_prec:widen_all plc_part store_part partial_prog)
           partial_prog);
      let full_prog = LL.Seq (LL.Zero_out tbl_loc_part, slice_sc) in
      let plc_full, store_full = mk_store () in
      Tn.Placements.update plc_full tbl_loc_part Tn.Local Tn.Visit_cap;
      p "fully-zeroed Local scatter target widens cleanly"
        (List.length
           (LL.flat_lines
              [ LL.widen_narrow_scatters ~accum_prec:widen_all plc_full store_full full_prog ])
        = 4);
      (* 7k: Mismatched Get_dynamic vs Set_dynamic index precision declines widening. *)
      let mismatch_prec_sc =
        mk_scatter
          ~llsc_of:(fun tn ->
            let p = Lazy.force tn.Tn.storage_prec in
            let self =
              LL.Get_dynamic
                {
                  tn;
                  idcs = [| iter d; fixed 0 |];
                  dyn_axis = 1;
                  dyn_value = (get ids [| iter pos |], Ops.byte);
                }
            in
            LL.Binop (Ops.Add, (self, p), (get grad [| iter pos; iter d |], p)))
          ()
      in
      let plc, store = mk_store () in
      p "mismatched Get_dynamic index precision declines widening"
        (phys_equal
           (LL.widen_narrow_scatters ~accum_prec:widen_all plc store mismatch_prec_sc)
           mismatch_prec_sc);
      (* 7l: Provably single-hit scatter (no multi-trip non-channel loop) declines widening. *)
      let single_hit = mk_scatter ~to_pos:0 () in
      let plc, store = mk_store () in
      p "single-hit scatter (no multi-trip reduction loop) declines widening"
        (phys_equal
           (LL.widen_narrow_scatters ~accum_prec:widen_all plc store single_hit)
           single_hit);
      (* 7m: Decision-only lowering respects caller-supplied ~accum_prec override. *)
      Tensor.unsafe_reinitialize ();
      let ids_t = ids_tensor_2d ~label:"dec_ids" [| [| 0; 0; 1; 1 |] |] in
      let onehot_t = Nn_blocks.one_hot_of_ids ~num_classes:4 ids_t in
      let wte_t =
        MP.with_master_weights ~placement:MP.Twin_virtual ~prec:bf16 (fun () ->
            TDSL.param ~values:(Array.create ~len:8 0.5) "dec_wte" ~input_dims:[ 4 ]
              ~output_dims:[ 2 ] ())
      in
      let%op emb_t = wte_t * onehot_t in
      let%op loss_t = emb_t ++ "...|... => |->0" in
      Precision_policy.apply
        ~except:(fun tn -> Tn.equal tn loss_t.Tensor.value)
        { Precision_policy.param_prec = None; activation_prec = Some bf16; grad_prec = Some bf16 }
        loss_t;
      let update_t = Train.grad_update loss_t in
      let ctx_t = Context.auto () in
      let opt_narrow =
        Context.lowered_for_decisions ~name:"nsr_dec_narrow" ~accum_prec:Fn.id ctx_t update_t
          Idx.Empty
      in
      let opt_wide =
        Context.lowered_for_decisions ~name:"nsr_dec_wide" ~accum_prec:widen_all ctx_t update_t
          Idx.Empty
      in
      p_all "lowered_for_decisions ~accum_prec:Fn.id keeps narrow scatter target"
        (scatter_targets_of_ll opt_narrow.LL.llc) ~f:(fun tn ->
          Ops.equal_prec (Lazy.force tn.Tn.storage_prec) bf16
          && not (String.equal tn.Tn.namespace LL.scatter_accum_namespace));
      p_all "lowered_for_decisions ~accum_prec:widen_all widens scatter target to f32"
        (scatter_targets_of_ll opt_wide.LL.llc) ~f:(fun tn ->
          Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single
          && String.equal tn.Tn.namespace LL.scatter_accum_namespace))

(* === Case 8: dual-table bf16 cast-twin embedding (wte + wpe + tied lm_head, gpt2_mini_train
   two-scatter shape) === *)
let () =
  Test_utils.with_policy
    (fun policy ->
      { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_auto; narrow_compute_f32 = true })
    (fun () ->
      case "dual-table bf16 cast-twin embedding (wte + wpe in one routine)" (fun () ->
          let vocab = 12 and n_pos = 8 and d_model = 8 and batch = 5 and seq = 4 in
          (* 5 batches x 4 positions: every position 0..3 receives 5 hits into wpe (big + 4*tiny),
             and token 2 receives 5 hits into wte (big + 4*tiny) across batch 0..4, pos 0. *)
          let tok_rows =
            [|
              [| 2; 3; 4; 5 |];
              [| 2; 6; 7; 8 |];
              [| 2; 3; 6; 9 |];
              [| 2; 4; 7; 10 |];
              [| 2; 5; 8; vocab |];
            |]
          in
          let pos_rows = Array.init batch ~f:(fun _ -> [| 0; 1; 2; 3 |]) in
          let scale d = 2. **. Float.of_int (d % 4) in
          let coeff_vals =
            Array.init
              (batch * seq * d_model)
              ~f:(fun idx ->
                let pos = idx / d_model and d = idx % d_model in
                let b = pos / seq in
                let s_val = scale d in
                if b = 0 then 256. *. s_val else 0.5 *. s_val)
          in
          let wte_vals =
            Array.init (d_model * vocab) ~f:(fun idx ->
                Ll_test.weighted ~weights:[| 3; 5 |] ~modulus:9 ~offset:(-4.) ~stride:0.25
                  [| idx / vocab; idx % vocab |])
          in
          let wpe_vals =
            Array.init (d_model * n_pos) ~f:(fun idx ->
                Ll_test.weighted ~weights:[| 5; 3 |] ~modulus:7 ~offset:(-3.) ~stride:0.25
                  [| idx / n_pos; idx % n_pos |])
          in
          let run ~prec =
            Tensor.unsafe_reinitialize ();
            let tok_ids = ids_tensor_2d ~label:"dual_tok" tok_rows in
            let pos_ids = ids_tensor_2d ~label:"dual_pos" pos_rows in
            let oh_tok = Nn_blocks.one_hot_of_ids ~num_classes:vocab tok_ids in
            let oh_pos = Nn_blocks.one_hot_of_ids ~num_classes:n_pos pos_ids in
            let make_tables () =
              let wte =
                TDSL.param ~values:wte_vals "wte" ~input_dims:[ vocab ] ~output_dims:[ d_model ] ()
              in
              let wpe =
                TDSL.param ~values:wpe_vals "wpe" ~input_dims:[ n_pos ] ~output_dims:[ d_model ] ()
              in
              (wte, wpe)
            in
            let wte_used, wpe_used =
              match prec with
              | None -> make_tables ()
              | Some p -> MP.with_master_weights ~placement:MP.Twin_virtual ~prec:p make_tables
            in
            let%op emb = (wte_used * oh_tok) + (wpe_used * oh_pos) in
            let coeff =
              TDSL.ndarray coeff_vals ~label:[ "dual_coeff" ] ~batch_dims:[ batch; seq ]
                ~output_dims:[ d_model ] ()
            in
            let%op loss = (emb *. coeff) ++ "...|... => |->0" in
            Option.iter prec ~f:(fun p ->
                Precision_policy.apply
                  ~except:(fun tn ->
                    Tn.equal tn loss.Tensor.value || Tn.equal tn coeff.Tensor.value)
                  {
                    Precision_policy.param_prec = None;
                    activation_prec = Some p;
                    grad_prec = Some p;
                  }
                  loss);
            let update = Train.grad_update loss in
            let ctx = Context.auto () in
            let ctx = Train.init_params ctx Idx.Empty loss in
            let ctx, routine =
              compile_routine
                ~name:(match prec with None -> "nsr_c8_f32" | Some _ -> "nsr_c8_bf16")
                ctx update
            in
            let ctx = Context.run ctx routine in
            Context.sync ctx;
            let wte_m = param_by_label loss "wte" and wpe_m = param_by_label loss "wpe" in
            (Context.get_values ctx (grad_of wte_m), Context.get_values ctx (grad_of wpe_m), routine)
          in
          let zeros_wte = Array.create ~len:(d_model * vocab) 0. in
          let zeros_wpe = Array.create ~len:(d_model * n_pos) 0. in
          let exact_wte =
            host_scatter ~vocab ~d_model ~ids_rows:tok_rows ~coeff_vals ~init:zeros_wte ~step:Fn.id
          in
          let exact_wpe =
            host_scatter ~vocab:n_pos ~d_model ~ids_rows:pos_rows ~coeff_vals ~init:zeros_wpe
              ~step:Fn.id
          in
          let once_wte = Array.map exact_wte ~f:(Ops.round_to_prec bf16) in
          let once_wpe = Array.map exact_wpe ~f:(Ops.round_to_prec bf16) in
          let per_hit_wte =
            host_scatter ~vocab ~d_model ~ids_rows:tok_rows ~coeff_vals ~init:zeros_wte
              ~step:(Ops.round_to_prec bf16)
          in
          let per_hit_wpe =
            host_scatter ~vocab:n_pos ~d_model ~ids_rows:pos_rows ~coeff_vals ~init:zeros_wpe
              ~step:(Ops.round_to_prec bf16)
          in
          let f32_wte, f32_wpe, _ = run ~prec:None in
          let bf_wte, bf_wpe, routine = run ~prec:(Some bf16) in
          let targets = scatter_targets routine in
          p "dual-table GPT-2 shape: both wte and wpe scatters arrive separately and widen"
            (match targets with [ a; b ] -> not (Tn.equal a b) | _ -> false);
          p_all "dual-table GPT-2 shape: every Set_dynamic targets an f32 scatter_acc buffer"
            targets ~f:(fun tn ->
              String.equal tn.Tn.namespace LL.scatter_accum_namespace
              && Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single);
          p_all2 "dual-table GPT-2 shape: f32 wte.grad equals exact host sum" f32_wte exact_wte
            ~f:bitwise;
          p_all2 "dual-table GPT-2 shape: f32 wpe.grad equals exact host sum" f32_wpe exact_wpe
            ~f:bitwise;
          p_exists
            "dual-table GPT-2 shape: wte per-hit bf16 differs from once-rounded (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit_wte) (Array.to_list once_wte))
            ~f:(fun (a, b) -> not (bitwise a b));
          p_exists
            "dual-table GPT-2 shape: wpe per-hit bf16 differs from once-rounded (non-vacuity)"
            (List.zip_exn (Array.to_list per_hit_wpe) (Array.to_list once_wpe))
            ~f:(fun (a, b) -> not (bitwise a b));
          p_all2 "dual-table GPT-2 shape: bf16 wte.grad equals f32 narrowed once, cell for cell"
            bf_wte once_wte ~f:bitwise;
          p_all2 "dual-table GPT-2 shape: bf16 wpe.grad equals f32 narrowed once, cell for cell"
            bf_wpe once_wpe ~f:bitwise))
