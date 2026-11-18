(* gh-ocannl-2281, gh-ocannl-2326. Replay fixed CUDA kernel segments with the two value-select
   renderings at both PTX targets, then compile and time a whole training step (a two-conv
   classifier with ReLU and 2x2 max pooling, forward, backward and SGD) under each, beside a second
   short-circuiting compile as its control. Context.compile and Autotune.untuned_compile bypass the
   schedule cache. The CUDA value-select benchmark section of benchmarks/README.md gives the
   hermetic invocation and readiness checks.

   The source check reads only this compile's returned CUDA source. It detects the renderer's first
   arm local by its declared name and compares the segments' control source byte for byte. The step
   lowers as well as renders under each choice, so its placements may differ; its check compares the
   control's source with the short arm's up to what each lowering mints afresh ([canonical] below).
   Branch instructions and select coverage within a kernel are outside that check. The dry run
   checks emitted forms, actual PTX targets, and host parity in every cell, and the step's values
   across the arms, before enabling timings. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Mma = Ir.Cuda_mma

type fixture = { seeds : (Ir.Tnode.t * float array) list; want : float array }

type workload = {
  name : string;
  segment : LL.optimized;
  output : Ir.Tnode.t;
  fixtures : (string * (unit -> fixture)) list;
}

let unwritten = -12345.
let value i = Float.of_int (1 + (i * 67 % 997)) *. 0.125
let p fmt = Bench_out.p fmt

type pointwise = Where_zero | Relu_gate | Where_eq | Where_masked | Control

let pointwise ~elements ~node kind =
  let open Ll_test in
  let name =
    match kind with
    | Where_zero -> "where_zero"
    | Relu_gate -> "relu_gate"
    | Where_eq -> "where_eq"
    | Where_masked -> "where_masked"
    | Control -> "control"
  in
  let flag = node "flag" and payload = node "payload" and equal_to = node "equal_to" in
  let output = node "output" in
  let block = 256 in
  let group = sym () and lane = sym () in
  let idx = aff [ (block, group); (1, lane) ] 0 in
  let at tn = get tn [| idx |] in
  let where test = where_own test (at payload) (c 0.) in
  let active = match kind with Where_masked -> elements * 3 / 4 | _ -> elements in
  let expr, materialized =
    match kind with
    | Control -> (binop_own Ops.Add (at flag) (at payload), [ flag; payload; output ])
    | Where_zero | Where_masked -> (where (at flag), [ flag; payload; output ])
    | Relu_gate -> (binop_own Ops.Relu_gate (at flag) (at payload), [ flag; payload; output ])
    | Where_eq ->
        let cond = binop_own Ops.Cmpeq (at flag) (at equal_to) in
        (where cond, [ flag; payload; equal_to; output ])
  in
  let write = set_at output idx expr in
  let body = if active = elements then write else if_idx (lt (embed_idx idx) (ic active)) write in
  let code =
    loop_n ~axis:LL.Grid group (elements / block) (loop_n ~axis:LL.Workgroup lane block body)
  in
  List.iter materialized ~f:materialize;
  let segment = optimize ~name code in
  let fixture ~dense () =
    let take i =
      let one_per_warp = i % 32 = ((i / 32 * 13) + 7) % 32 in
      Bool.(one_per_warp <> dense)
    in
    let flags =
      Array.init elements ~f:(fun i ->
          match kind with
          | Control | Where_eq -> value i
          | Relu_gate -> if take i then 1. else -1.
          | Where_zero | Where_masked -> if take i then 1. else 0.)
    in
    let data = Array.init elements ~f:(fun i -> value (i + 73)) in
    let seeds = [ (flag, flags); (payload, data) ] in
    let seeds =
      match kind with
      | Where_eq ->
          (equal_to, Array.mapi flags ~f:(fun i f -> if take i then f else f +. 1.)) :: seeds
      | _ -> seeds
    in
    let want =
      Array.mapi data ~f:(fun i g ->
          match kind with
          | Control -> flags.(i) +. g
          | _ -> if i >= active then unwritten else if take i then g else 0.)
    in
    { seeds; want }
  in
  let fixtures =
    match kind with
    | Control -> [ ("control", fixture ~dense:false) ]
    | _ -> [ ("sparse", fixture ~dense:false); ("dense", fixture ~dense:true) ]
  in
  { name; segment; output; fixtures }

let pool ~root ~elements =
  let batches = elements / 16 in
  let grad t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  let xval b r c = Float.of_int ((((b * 11) + (r * 3) + (c * 5)) % 7) - 3) in
  let weight b r c = 0.25 *. Float.of_int (1 + (((b * 17) + r + (2 * c)) % 251)) in
  let h b r c = Float.max (xval b r c) 0. in
  let pooled b r c =
    List.fold
      [ (0, 0); (0, 1); (1, 0); (1, 1) ]
      ~init:0.
      ~f:(fun acc (dr, dc) -> Float.max acc (h b ((2 * r) + dr) ((2 * c) + dc)))
  in
  let x =
    Operation.init ~l:"pool_x" ~prec:Ir.Ops.single ~b:[ batches ] ~o:[ 4; 4 ]
      ~f:(fun i -> xval i.(0) i.(1) i.(2))
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    Operation.init ~l:"pool_w" ~prec:Ir.Ops.single ~b:[ batches ] ~o:[ 2; 2 ]
      ~f:(fun i -> weight i.(0) i.(1) i.(2))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let%op y = Nn_blocks.max_pool2d () (relu x) in
  let%op loss = (y *. w) ++ "... | ... => 0" in
  let output = grad x and upstream = grad y in
  List.iter [ output; upstream; y.Tensor.value ] ~f:Train.set_materialized;
  let ctx, pipeline =
    Context.compile ~name:"pool_pipeline" root (Train.grad_update loss) Ir.Indexing.Empty
  in
  let segment =
    Exn.protect
      ~finally:(fun () -> Context.release ctx)
      ~f:(fun () ->
        let writes_output seg =
          match Ir.Low_level.zero_initializer_target seg.LL.llc with
          | Some _ -> false
          | None -> List.mem (Bench_harness.writes_of seg.LL.llc) output ~equal:Ir.Tnode.equal
        in
        match List.filter pipeline.Context.segments ~f:writes_output with
        | [ segment ] -> segment
        | _ -> failwith "expected one data-writing max-pool backward kernel")
  in
  let fixture () =
    let small f = Array.init (batches * 4) ~f:(fun i -> f (i / 4) (i % 4 / 2) (i % 2)) in
    let want =
      Array.init elements ~f:(fun i ->
          let b = i / 16 and r = i % 16 / 4 and c = i % 4 in
          let wr = r / 2 and wc = c / 2 in
          if Float.(xval b r c > 0. && h b r c = pooled b wr wc) then weight b wr wc else 0.)
    in
    { seeds = [ (y.Tensor.value, small pooled); (upstream, small weight) ]; want }
  in
  { name = "pool_backward"; segment; output; fixtures = [ ("pool", fixture) ] }

let has_local src = String.is_substring src ~substring:(Ir.C_syntax.blend_local_a ^ " =")

(* Two lowerings of one program, as text that compares equal where only what each lowering mints
   afresh differs: [Ir.Kernel_code_cache.canonical_kernel] renames loop symbols and scope locals by
   first occurrence, the schedule's minted nodes ([..__n<n>_..]) join its renamed [bound] names, and
   whitespace goes, since line breaks follow the identifiers' lengths. *)
let canonical src =
  let is_ident c = Char.is_alphanum c || Char.equal c '_' in
  let schedule_node w =
    List.exists (String.substr_index_all w ~may_overlap:false ~pattern:"__n") ~f:(fun k ->
        let rest = String.drop_prefix w (k + 3) in
        match String.lfindi rest ~f:(fun _ c -> not (Char.is_digit c)) with
        | None -> not (String.is_empty rest)
        | Some d -> d > 0 && Char.equal rest.[d] '_')
  in
  let bound =
    String.map src ~f:(fun c -> if is_ident c then c else ' ')
    |> String.split ~on:' ' |> List.filter ~f:schedule_node
    |> List.dedup_and_sort ~compare:String.compare
  in
  Ir.Kernel_code_cache.canonical_kernel ~bound src
  |> String.filter ~f:(fun c -> not (Char.is_whitespace c))

let compile ~root ~target ~blend work =
  let ((ctx, routine), sources), ptx =
    Mma.with_ptx_target target (fun () ->
        Cuda_backend.with_select_rendering ~blend (fun () ->
            Context.compile ~name:work.name
              ~prelowered:(Bench_harness.scratch_segment work.segment)
              ~lowered_transform:(fun segment -> [ segment ])
              root Ir.Assignments.empty_comp Ir.Indexing.Empty))
  in
  match (sources, ptx) with
  | [ source ], [ { Mma.device_cc; target = Some arch } ] ->
      ((ctx, routine), source, device_cc, arch)
  | _ ->
      Context.release ctx;
      failwith (work.name ^ ": expected one emitted CUDA source and its PTX target")

let check label (ctx, routine) work { seeds; want } =
  let ctx = Context.set_values ctx work.output (Array.create ~len:(Array.length want) unwritten) in
  let ctx =
    List.fold seeds ~init:ctx ~f:(fun ctx (node, values) -> Context.set_values ctx node values)
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx work.output in
  let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  Verdict.pass_fail_all2 (label ^ " host parity") got want ~f:same_bits;
  if Verdict.any_failed () then (
    if Array.length got = Array.length want then
      Option.iter
        (Array.findi got ~f:(fun i a -> not (same_bits a want.(i))))
        ~f:(fun (i, a) -> p "%s cell %d got %h expected %h\n" label i a want.(i));
    failwith (label ^ ": readback failed"));
  (ctx, routine)

let measure (ctx, routine) =
  Autotune.admitted_timing_ms (Autotune.time_routine ~timing:Autotune.Queued ~repeats:3 ctx routine)

(* Each round times every arm once, in an order that rotates by one arm per round and runs backward
   on every other cycle of [n] rounds, so each arm takes each position in turn: for three arms, six
   rounds are the six orders. [before k] runs ahead of arm [k]'s timing, outside its clock. [ratios]
   name numerator and denominator arms by name. *)
let compare_times ?(before = fun _ -> ()) ~calls ~label
    ~(arms : (string * (Context.t * Context.routine)) list) ~ratios () =
  let n = List.length arms in
  let arms = Array.of_list arms in
  let rows =
    Array.init calls ~f:(fun i ->
        let row = Array.create ~len:n None in
        let order =
          List.init n ~f:(fun k ->
              let k = if i / n % 2 = 0 then k else n - 1 - k in
              (k + i) % n)
        in
        List.iter order ~f:(fun k ->
            before k;
            row.(k) <- measure (snd arms.(k)));
        row)
  in
  (* Treatment and control have to use the same rounds for their spreads to be comparable. *)
  let rows =
    Array.map rows ~f:(fun row ->
        if Array.for_all row ~f:Option.is_some then row else Array.create ~len:n None)
  in
  let column name =
    match Array.findi arms ~f:(fun _ (arm, _) -> String.equal arm name) with
    | Some (k, _) -> Array.map rows ~f:(fun row -> row.(k))
    | None -> failwith (label ^ ": no arm " ^ name)
  in
  let show (name, _) =
    Bench_retime.spread_fields ~name:(name ^ "_ms") (Bench_retime.spread_of (column name))
  in
  let ratio (num, den) =
    let r = Bench_retime.paired_ratio ~num:(column num) ~den:(column den) in
    if r.pairs < Bench_retime.min_readings then
      failwith (Printf.sprintf "%s: only %d of %d timing rounds admitted" label r.pairs calls);
    Bench_retime.ratio_fields ~name:(num ^ "_over_" ^ den) r
  in
  p "%s %s %s\n" label
    (String.concat ~sep:" " (List.map (Array.to_list arms) ~f:show))
    (String.concat ~sep:" " (List.map ratios ~f:ratio))

let compare_code ~root ~target ~control work f =
  let short, src_a, cc_a, arch_a = compile ~root ~target ~blend:false work in
  Exn.protect
    ~finally:(fun () -> Context.release (fst short))
    ~f:(fun () ->
      let expected_native = Printf.sprintf "sm_%d" cc_a in
      let is_native = Mma.equal_ptx_target target Mma.Native in
      if (not is_native) && String.equal arch_a expected_native then
        failwith
          (Printf.sprintf
             "%s: floor and native PTX coincide at %s. This sweep needs distinct targets." work.name
             arch_a);
      let blend, src_b, cc_b, arch_b = compile ~root ~target ~blend:true work in
      Exn.protect
        ~finally:(fun () -> Context.release (fst blend))
        ~f:(fun () ->
          let label = Printf.sprintf "%s/%s" arch_a work.name in
          let matches =
            if control then String.equal src_a src_b && not (has_local src_b)
            else (not (has_local src_a)) && has_local src_b
          in
          Verdict.pass_fail (label ^ " emitted rendering") matches;
          Verdict.pass_fail (label ^ " actual PTX targets")
            (cc_a = cc_b && String.equal arch_a arch_b
            && ((not is_native) || String.equal arch_a expected_native));
          if Verdict.any_failed () then failwith (label ^ ": source or PTX check failed");
          f ~label short blend))

(* === The training step === *)

(* A cifar-scale two-conv classifier (benchmarks/README.md's [cifar_conv] geometry: 3-channel 44x44
   images, valid 5x5 convs into 32 then 64 channels, each followed by ReLU and a 2x2 max pool, then
   fc 128, fc 84 and 10 logits) on generated inputs, one step of cross-entropy, backward and SGD.
   Its backward holds both selects the segments above isolate: ReLU's gate and max-pool's [where (m
   = h) g 0]. *)
type step = {
  loss : Tensor.t;
  comp : Ir.Assignments.comp;
  init_ctx : Context.t;
  snapshot : (Ir.Tnode.t * float array) list;
}

let build_step ~root ~batch =
  let x =
    Operation.init ~l:"sel_step_x" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 44; 44; 3 ]
      ~f:(fun i ->
        Float.of_int ((((i.(0) * 7) + (i.(1) * 3) + (i.(2) * 5) + i.(3)) % 17) - 8) /. 8.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let targets =
    Operation.init ~l:"sel_step_y" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 10 ]
      ~f:(fun i -> if i.(1) = i.(0) * 3 % 10 then 1. else 0.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let conv label c = Nn_blocks.conv2d ~label ~kernel_size:5 ~use_padding:false ~out_channels:c () in
  let conv1 = conv [ "sel_conv1" ] 32 and conv2 = conv [ "sel_conv2" ] 64 in
  let pool1 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let pool2 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let fc1 = Nn_blocks.mlp_layer ~label:[ "sel_fc1" ] ~hid_dim:128 () in
  let fc2 = Nn_blocks.mlp_layer ~label:[ "sel_fc2" ] ~hid_dim:84 () in
  let%op logits =
    ({ w_logits } * fc2 (fc1 (pool2 (relu (conv2 (pool1 (relu (conv1 x)))))))) + { b_logits = 0. }
  in
  let%op loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch () ~logits ~targets
  in
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.01) loss in
  let comp = Ir.Assignments.sequence [ update; sgd ] in
  let init_ctx = Train.init_params root Ir.Indexing.Empty loss in
  let snapshot = Bench_harness.param_values init_ctx loss in
  { loss; comp; init_ctx; snapshot }

let compile_step ~target ~blend step =
  let ((ctx, routine), sources), ptx =
    Mma.with_ptx_target target (fun () ->
        Cuda_backend.with_select_rendering ~blend (fun () ->
            Autotune.untuned_compile ~name:"sel_step" step.init_ctx step.comp Ir.Indexing.Empty))
  in
  (* The routine that runs is the extent's last compile: an untuned compile may retry after a
     rejected attempt that already emitted, so only the last source and PTX describe it. *)
  match (List.last sources, List.last ptx) with
  | Some source, Some { Mma.device_cc; target = Some arch } ->
      if List.length sources > 1 then
        p "sel_step: %d compiles in the extent, the last one checked\n" (List.length sources);
      ((ctx, routine), source, device_cc, arch)
  | _ ->
      Context.release ctx;
      failwith "sel_step: expected emitted CUDA source and its PTX target"

(* One step from the initial parameters: the loss and every updated parameter, in order. *)
let step_outputs step (ctx, routine) =
  let ctx =
    List.fold step.snapshot ~init:ctx ~f:(fun ctx (tn, values) -> Context.set_values ctx tn values)
  in
  let ctx = Context.run ctx routine in
  Array.concat
    (Context.get_values ctx step.loss.Tensor.value
    :: List.map step.snapshot ~f:(fun (tn, _) -> Context.get_values ctx tn))

let count_sub src ~sub = List.length (String.substr_index_all src ~may_overlap:false ~pattern:sub)

let compare_step ~target ~dry ~calls step =
  (* Each arm's context is released however the rest goes, from the moment it links. *)
  let linked = ref [] in
  Exn.protect
    ~finally:(fun () -> List.iter !linked ~f:Context.release)
    ~f:(fun () ->
      let arms =
        List.map
          [ ("short", false); ("blend", true); ("control", false) ]
          ~f:(fun (name, blend) ->
            let (((ctx, _), _, _, _) as arm) = compile_step ~target ~blend step in
            linked := ctx :: !linked;
            (name, arm))
      in
      let get name = List.Assoc.find_exn arms ~equal:String.equal name in
      let _, src_a, cc_a, arch_a = get "short" in
      let _, src_b, _, _ = get "blend" in
      let _, src_c, _, _ = get "control" in
      let expected_native = Printf.sprintf "sm_%d" cc_a in
      let is_native = Mma.equal_ptx_target target Mma.Native in
      let label = Printf.sprintf "%s/train_step" arch_a in
      (* Each lowering mints its loop symbols, local names and schedule nodes afresh. *)
      let same x y = String.equal (canonical x) (canonical y) in
      let control_matches = same src_a src_c in
      if not control_matches then (
        let la = String.split_lines src_a and lc = String.split_lines src_c in
        p "%s control source differs: %d vs %d lines\n" label (List.length la) (List.length lc);
        Option.iter
          (let n = Int.min (List.length la) (List.length lc) in
           List.findi
             (List.zip_exn (List.take la n) (List.take lc n))
             ~f:(fun _ (x, y) -> not (String.equal x y)))
          ~f:(fun (k, (x, y)) ->
            let cut s = String.prefix s 240 in
            p "%s line %d short:   %s\n%s line %d control: %s\n" label (k + 1) (cut x) label (k + 1)
              (cut y)));
      p "%s control_matches=%b short_has_local=%b blend_has_local=%b\n" label control_matches
        (has_local src_a) (has_local src_b);
      Verdict.pass_fail (label ^ " emitted rendering")
        (control_matches && (not (has_local src_a)) && has_local src_b);
      Verdict.pass_fail (label ^ " actual PTX targets")
        (List.for_all arms ~f:(fun (_, (_, _, cc, arch)) -> cc = cc_a && String.equal arch arch_a)
        && Bool.equal is_native (String.equal arch_a expected_native));
      if Verdict.any_failed () then failwith (label ^ ": source or PTX check failed");
      (* [BENCH_SELECT_SOURCES=<dir>]: each arm's emitted step source, for reading what moved. *)
      Option.iter (Stdlib.Sys.getenv_opt "BENCH_SELECT_SOURCES") ~f:(fun dir ->
          List.iter arms ~f:(fun (name, (_, src, _, _)) ->
              Stdlib.Out_channel.with_open_text (Printf.sprintf "%s/%s_%s.cu" dir arch_a name)
                (fun oc -> Stdlib.Out_channel.output_string oc src)));
      List.iter arms ~f:(fun (name, (_, src, _, _)) ->
          p "%s/%s source_bytes=%d blend_locals=%d short_selects=%d\n" label name
            (String.length src)
            (count_sub src ~sub:(Ir.C_syntax.blend_local_a ^ " ="))
            (count_sub src ~sub:" ? "));
      (* The policies may place nodes differently, so values are compared within a tolerance: the
         arithmetic is the same, its contraction into FMAs need not be. *)
      let want =
        let c, _, _, _ = get "short" in
        step_outputs step c
      in
      List.iter arms ~f:(fun (name, (c, _, _, _)) ->
          let got = step_outputs step c in
          let close a b = Float.(abs (a -. b) <= 1e-4 *. max 1. (abs b)) in
          let bitwise =
            Array.length got = Array.length want
            && Array.for_all2_exn got want ~f:(fun a b ->
                Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b))
          in
          p "%s/%s step_values=%d bitwise_equal_to_short=%b loss=%.6f\n" label name
            (Array.length got) bitwise got.(0);
          Verdict.pass_fail_all2 (label ^ "/" ^ name ^ " step parity with short") got want ~f:close;
          if Verdict.any_failed () then failwith (label ^ ": step parity failed"));
      (* Every timing call runs SGD, so each arm starts its timing from the initial parameters, not
         from where the arm before it left them. *)
      let restore k =
        let _, ((ctx, _), _, _, _) = List.nth_exn arms k in
        ignore
          (List.fold step.snapshot ~init:ctx ~f:(fun ctx (tn, values) ->
               Context.set_values ctx tn values)
            : Context.t)
      in
      if not dry then
        compare_times ~before:restore ~calls ~label
          ~arms:(List.map arms ~f:(fun (name, (c, _, _, _)) -> (name, c)))
          ~ratios:[ ("blend", "short"); ("control", "short") ]
          ())

let () =
  let args = Bench_args.create "bench_cuda_select" in
  let elements = Bench_args.int args 0 ~name:"elements" ~default:(1 lsl 22) in
  let batch = Bench_args.int args 1 ~flag:"step-batch" ~name:"step-batch" ~default:64 in
  let dry = Bench_args.flag args ~name:"dry-run" in
  if elements % 256 <> 0 then invalid_arg "elements must be a multiple of 256";
  if not (String.equal (Utils.get_global_arg ~arg_name:"cuda_kernel_cache_mib" ~default:"64") "0")
  then invalid_arg "set OCANNL_CUDA_KERNEL_CACHE_MIB=0 to witness every PTX compile";
  let root = Context.auto () in
  Exn.protect
    ~finally:(fun () -> Context.release root)
    ~f:(fun () ->
      let backend = Context.backend_name root in
      if not (String.equal backend "cuda") then
        invalid_arg "bench_cuda_select requires --ocannl_backend=cuda";
      let calls = Bench_retime.calls_of_env () in
      p
        "backend=%s elements=%d pool_images=%d mask_density=1/32,31/32 step=cifar_conv \
         step_batch=%d timing=queued calls=%d dry_run=%b\n"
        backend elements (elements / 16) batch calls dry;
      let node = Ll_test.node_factory ~first_id:20000 ~dims:[| elements |] () in
      let control = pointwise ~elements ~node Control in
      let workloads =
        List.map [ Where_zero; Relu_gate; Where_eq; Where_masked ] ~f:(pointwise ~elements ~node)
        @ [ pool ~root ~elements ]
      in
      let step = build_step ~root ~batch in
      (* Released after the step arms, which each release their own context first. *)
      Exn.protect
        ~finally:(fun () -> Context.release step.init_ctx)
        ~f:(fun () ->
          List.iter [ Mma.Floor; Mma.Native ] ~f:(fun target ->
              compare_code ~root ~target ~control:true control
                (fun ~label control_short control_blend ->
                  let fixture = (snd (List.hd_exn control.fixtures)) () in
                  let control_short = check (label ^ "/short") control_short control fixture in
                  let control_blend = check (label ^ "/blend") control_blend control fixture in
                  List.iter workloads ~f:(fun work ->
                      compare_code ~root ~target ~control:false work (fun ~label short blend ->
                          List.iter work.fixtures ~f:(fun (density, fixture) ->
                              let label = label ^ "/" ^ density in
                              let fixture = fixture () in
                              if String.equal density "pool" then
                                p "%s nonzero_gradient=%d/%d\n" label
                                  (Array.count fixture.want ~f:(fun v -> Float.(v <> 0.)))
                                  (Array.length fixture.want);
                              let short = check (label ^ "/short") short work fixture in
                              let blend = check (label ^ "/blend") blend work fixture in
                              if not dry then
                                compare_times ~calls ~label
                                  ~arms:
                                    [
                                      ("short", short);
                                      ("control_short", control_short);
                                      ("blend", blend);
                                      ("control_blend", control_blend);
                                    ]
                                  ~ratios:[ ("blend", "short"); ("control_blend", "control_short") ]
                                  ()))));
              compare_step ~target ~dry ~calls step)))
