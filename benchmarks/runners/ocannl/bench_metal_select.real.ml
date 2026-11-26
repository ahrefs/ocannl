(* gh-ocannl-2400. Replay fixed Metal kernel segments with short-circuiting [?:] and local-value
   blending ([Blend_over_locals]), then compile and time a whole training step (the [cifar_conv]
   two-conv classifier with ReLU and 2x2 max pooling, forward, backward and SGD) under each, beside
   a second short-circuiting compile as its control. Context.compile and Autotune.untuned_compile
   bypass the schedule cache.

   The pointwise fixtures span both divergent within-simdgroup densities (1/32 sparse, 16/32 half,
   31/32 dense) and simdgroup-uniform densities (0/32 all-false, alternating all-true/all-false
   simdgroups), plus a causal-mask score-load kernel ([where mask (score * scale) (-inf)], matching
   gh-ocannl-2427's shape) and the realistic ReLU + 2x2 max-pool backward kernel. Comparing
   uniform-false and causal-mask ratios against divergent ratios distinguishes AGX predication from
   simdgroup branch skipping without an AGX disassembler.

   The source check reads only this compile's returned MSL source. It detects the renderer's first
   arm local by its declared name and compares the segments' control source byte for byte. The step
   lowers as well as renders under each choice, so its placements may differ; its check compares the
   control's source with the short arm's up to what each lowering mints afresh ([canonical] below).
   The dry run checks emitted forms and host parity in every cell, and the step's values across the
   arms, before enabling timings. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level

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
type density = Uniform_zero | Uniform_half | Sparse | Half_div | Dense

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
  let fixture density () =
    let take i =
      let one_per_simd = i % 32 = ((i / 32 * 13) + 7) % 32 in
      match density with
      | Uniform_zero -> false
      | Uniform_half -> i / 32 % 2 = 0
      | Sparse -> one_per_simd
      | Half_div -> i % 2 = 0
      | Dense -> not one_per_simd
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
    | Control -> [ ("control", fixture Sparse) ]
    | _ ->
        [
          ("uniform_zero", fixture Uniform_zero);
          ("uniform_half", fixture Uniform_half);
          ("sparse", fixture Sparse);
          ("half_div", fixture Half_div);
          ("dense", fixture Dense);
        ]
  in
  { name; segment; output; fixtures }

(* [out[i] = where mask[i] (score[i] * 0.125) (-inf)] over a causal lower-triangular mask, modeling
   gh-ocannl-2427's online-softmax score load on Metal: ~50% masked, mostly simdgroup-uniform
   upper-triangle blocks. *)
let causal_mask ~elements ~node =
  let open Ll_test in
  let name = "causal_mask" in
  let mask = node "cm_mask" and score = node "cm_score" and output = node "cm_output" in
  let block = 256 in
  let group = sym () and lane = sym () in
  let idx = aff [ (block, group); (1, lane) ] 0 in
  let at tn = get tn [| idx |] in
  let scaled = binop_own Ops.Mul (at score) (c 0.125) in
  let expr = where_own (at mask) scaled (c Float.neg_infinity) in
  let code =
    loop_n ~axis:LL.Grid group (elements / block)
      (loop_n ~axis:LL.Workgroup lane block (set_at output idx expr))
  in
  List.iter [ mask; score; output ] ~f:materialize;
  let segment = optimize ~name code in
  let seq = if elements >= 512 * 512 && elements % (512 * 512) = 0 then 512 else 16 in
  let fixture () =
    let masks =
      Array.init elements ~f:(fun i ->
          let within = i % (seq * seq) in
          let q = within / seq and k = within % seq in
          if k <= q then 1. else 0.)
    in
    let scores = Array.init elements ~f:(fun i -> value (i + 19)) in
    let want =
      Array.init elements ~f:(fun i ->
          if Float.(masks.(i) <> 0.) then scores.(i) *. 0.125 else Float.neg_infinity)
    in
    { seeds = [ (mask, masks); (score, scores) ]; want }
  in
  { name; segment; output; fixtures = [ ("causal", fixture) ] }

(* Register-only arm ([out[i] = where flag[i] (flag[i] * 0.5 + 1.0) 0] where the arm holds no device
   load, via a scalar local bound before the select): tests gh-ocannl-2427's load-free arm-cost rule
   on Metal. *)
let reg_arm ~elements ~node =
  let open Ll_test in
  let name = "reg_arm" in
  let flag = node "ra_flag" and output = node "ra_output" in
  let block = 256 in
  let group = sym () and lane = sym () in
  let idx = aff [ (block, group); (1, lane) ] 0 in
  let at tn = get tn [| idx |] in
  let local_v = LL.get_scope (node "ra_loc") in
  virtualize local_v.tn;
  let decl_local = declare_local local_v in
  let init_local =
    LL.Set_local (local_v, binop_own Ops.Add (binop_own Ops.Mul (at flag) (c 0.5)) (c 1.))
  in
  let arm = binop_own Ops.Mul (LL.Get_local local_v) (c 0.25) in
  let expr = where_own (at flag) arm (c 0.) in
  let body = LL.Seq (decl_local, LL.Seq (init_local, set_at output idx expr)) in
  let code =
    loop_n ~axis:LL.Grid group (elements / block) (loop_n ~axis:LL.Workgroup lane block body)
  in
  List.iter [ flag; output ] ~f:materialize;
  let segment = optimize ~name code in
  let fixture density () =
    let take i =
      let one_per_simd = i % 32 = ((i / 32 * 13) + 7) % 32 in
      match density with
      | Uniform_zero -> false
      | Uniform_half -> i / 32 % 2 = 0
      | Sparse -> one_per_simd
      | Half_div -> i % 2 = 0
      | Dense -> not one_per_simd
    in
    let flags = Array.init elements ~f:(fun i -> if take i then value i else 0.) in
    let want =
      Array.map flags ~f:(fun f -> if Float.(f <> 0.) then ((f *. 0.5) +. 1.) *. 0.25 else 0.)
    in
    { seeds = [ (flag, flags) ]; want }
  in
  {
    name;
    segment;
    output;
    fixtures =
      [
        ("uniform_zero", fixture Uniform_zero); ("sparse", fixture Sparse); ("dense", fixture Dense);
      ];
  }

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

let compile ~root ~blend work =
  let (ctx, routine), sources =
    Metal_backend.with_select_rendering ~blend (fun () ->
        Context.compile ~name:work.name
          ~prelowered:(Bench_harness.scratch_segment work.segment)
          ~lowered_transform:(fun segment -> [ segment ])
          root Ir.Assignments.empty_comp Ir.Indexing.Empty)
  in
  match sources with
  | [ source ] -> ((ctx, routine), source)
  | _ ->
      Context.release ctx;
      failwith (work.name ^ ": expected one emitted Metal source")

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

let dump_source ~name ~arm src =
  Option.iter (Stdlib.Sys.getenv_opt "BENCH_SELECT_SOURCES") ~f:(fun dir ->
      Stdlib.Out_channel.with_open_text (Printf.sprintf "%s/%s_%s.metal" dir name arm) (fun oc ->
          Stdlib.Out_channel.output_string oc src))

let compare_code ~root ~control work f =
  let short, src_a = compile ~root ~blend:false work in
  Exn.protect
    ~finally:(fun () -> Context.release (fst short))
    ~f:(fun () ->
      let blend, src_b = compile ~root ~blend:true work in
      Exn.protect
        ~finally:(fun () -> Context.release (fst blend))
        ~f:(fun () ->
          let label = "metal/" ^ work.name in
          let matches =
            if control then String.equal src_a src_b && not (has_local src_b)
            else (not (has_local src_a)) && has_local src_b
          in
          dump_source ~name:work.name ~arm:"short" src_a;
          dump_source ~name:work.name ~arm:"blend" src_b;
          Verdict.pass_fail (label ^ " emitted rendering") matches;
          if Verdict.any_failed () then failwith (label ^ ": source check failed");
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

let compile_step ~blend step =
  let (ctx, routine), sources =
    Metal_backend.with_select_rendering ~blend (fun () ->
        Autotune.untuned_compile ~name:"sel_step" step.init_ctx step.comp Ir.Indexing.Empty)
  in
  (* The routine that runs is the extent's last compile: an untuned compile may retry after a
     rejected attempt that already emitted, so only the last source describes it. *)
  match List.last sources with
  | Some source ->
      if List.length sources > 1 then
        p "sel_step: %d compiles in the extent, the last one checked\n" (List.length sources);
      ((ctx, routine), source)
  | None ->
      Context.release ctx;
      failwith "sel_step: expected emitted Metal source"

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

let compare_step ~dry ~step_segments ~calls step =
  let linked = ref [] in
  Exn.protect
    ~finally:(fun () -> List.iter !linked ~f:Context.release)
    ~f:(fun () ->
      let arms =
        List.map
          [ ("short", false); ("blend", true); ("control", false) ]
          ~f:(fun (name, blend) ->
            let (((ctx, _), _) as arm) = compile_step ~blend step in
            linked := ctx :: !linked;
            (name, arm))
      in
      let get name = List.Assoc.find_exn arms ~equal:String.equal name in
      let _, src_a = get "short" in
      let _, src_b = get "blend" in
      let _, src_c = get "control" in
      let label = "metal/train_step" in
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
      if Verdict.any_failed () then failwith (label ^ ": source check failed");
      List.iter arms ~f:(fun (name, (_, src)) ->
          dump_source ~name:"train_step" ~arm:name src;
          p "%s/%s source_bytes=%d blend_locals=%d short_selects=%d\n" label name
            (String.length src)
            (count_sub src ~sub:(Ir.C_syntax.blend_local_a ^ " ="))
            (count_sub src ~sub:" ? "));
      let want =
        let c, _ = get "short" in
        step_outputs step c
      in
      List.iter arms ~f:(fun (name, (c, _)) ->
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
      let restore k =
        let _, ((ctx, _), _) = List.nth_exn arms k in
        ignore
          (List.fold step.snapshot ~init:ctx ~f:(fun ctx (tn, values) ->
               Context.set_values ctx tn values)
            : Context.t)
      in
      if not dry then (
        compare_times ~before:restore ~calls ~label
          ~arms:(List.map arms ~f:(fun (name, (c, _)) -> (name, c)))
          ~ratios:[ ("blend", "short"); ("control", "short") ]
          ();
        if step_segments then
          let (short_ctx, short_routine), _ = get "short" in
          let (blend_ctx, blend_routine), _ = get "blend" in
          let short_segs = short_routine.Context.segments in
          let blend_segs = blend_routine.Context.segments in
          if List.length short_segs = List.length blend_segs then
            List.iteri (List.zip_exn short_segs blend_segs) ~f:(fun i (s_seg, b_seg) ->
                let seg_name = Printf.sprintf "sel_step_seg%d" i in
                let compile_one ctx ~blend seg =
                  let (c, r), srcs =
                    Metal_backend.with_select_rendering ~blend (fun () ->
                        Context.compile ~name:seg_name
                          ~prelowered:(Bench_harness.scratch_segment seg)
                          ~lowered_transform:(fun s -> [ s ])
                          ctx Ir.Assignments.empty_comp Ir.Indexing.Empty)
                  in
                  ((c, r), String.concat ~sep:"\n" srcs)
                in
                let sc, _s_src = compile_one short_ctx ~blend:false s_seg in
                Exn.protect
                  ~finally:(fun () -> Context.release (fst sc))
                  ~f:(fun () ->
                    let bc, b_src = compile_one blend_ctx ~blend:true b_seg in
                    Exn.protect
                      ~finally:(fun () -> Context.release (fst bc))
                      ~f:(fun () ->
                        if has_local b_src then
                          compare_times ~calls
                            ~label:(Printf.sprintf "%s/seg%d" label i)
                            ~arms:[ ("short", sc); ("blend", bc) ]
                            ~ratios:[ ("blend", "short") ]
                            ())))
          else
            p "%s: segment count mismatch (%d vs %d)\n" label (List.length short_segs)
              (List.length blend_segs)))

let () =
  let args = Bench_args.create "bench_metal_select" in
  let elements = Bench_args.int args 0 ~name:"elements" ~default:(1 lsl 22) in
  let batch = Bench_args.int args 1 ~flag:"step-batch" ~name:"step-batch" ~default:64 in
  let dry = Bench_args.flag args ~name:"dry-run" in
  let step_only = Bench_args.flag args ~name:"step-only" in
  let step_segments = Bench_args.flag args ~name:"step-segments" in
  if elements % 256 <> 0 then invalid_arg "elements must be a multiple of 256";
  if not (String.equal (Utils.get_global_arg ~arg_name:"metal_kernel_cache_mib" ~default:"64") "0")
  then invalid_arg "set OCANNL_METAL_KERNEL_CACHE_MIB=0 to witness every Metal compile";
  let root = Context.auto () in
  Exn.protect
    ~finally:(fun () -> Context.release root)
    ~f:(fun () ->
      let backend = Context.backend_name root in
      if not (String.equal backend "metal") then
        invalid_arg "bench_metal_select requires --ocannl_backend=metal";
      let calls = Bench_retime.calls_of_env () in
      let caps = Metal_backend.host_capabilities () in
      let cap k = Option.value ~default:"unknown" (List.Assoc.find caps ~equal:String.equal k) in
      p
        "backend=%s elements=%d pool_images=%d \
         densities=uniform_zero,uniform_half,sparse,half_div,dense step=cifar_conv step_batch=%d \
         timing=queued calls=%d dry_run=%b metal_math_api=%s metal_driver_build=%s\n"
        backend elements (elements / 16) batch calls dry (cap "metal_math_api")
        (cap "metal_driver_build");
      let node = Ll_test.node_factory ~first_id:20000 ~dims:[| elements |] () in
      let control = pointwise ~elements ~node Control in
      let workloads =
        if step_only then []
        else
          List.map [ Where_zero; Relu_gate; Where_eq; Where_masked ] ~f:(pointwise ~elements ~node)
          @ [ causal_mask ~elements ~node; reg_arm ~elements ~node; pool ~root ~elements ]
      in
      let step = build_step ~root ~batch in
      Exn.protect
        ~finally:(fun () -> Context.release step.init_ctx)
        ~f:(fun () ->
          if not step_only then
            compare_code ~root ~control:true control (fun ~label control_short control_blend ->
                let fixture = (snd (List.hd_exn control.fixtures)) () in
                let control_short = check (label ^ "/short") control_short control fixture in
                let control_blend = check (label ^ "/blend") control_blend control fixture in
                List.iter workloads ~f:(fun work ->
                    compare_code ~root ~control:false work (fun ~label short blend ->
                        List.iter work.fixtures ~f:(fun (density, fixture) ->
                            let label = label ^ "/" ^ density in
                            let fixture = fixture () in
                            if String.equal density "pool" || String.equal density "causal" then
                              p "%s nonzero_or_active=%d/%d\n" label
                                (Array.count fixture.want ~f:(fun v ->
                                     Float.(is_finite v && v <> 0.)))
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
          compare_step ~dry ~step_segments ~calls step))
