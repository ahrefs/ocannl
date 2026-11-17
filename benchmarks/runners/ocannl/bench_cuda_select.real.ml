(* gh-ocannl-2281. Replay fixed CUDA kernel segments with the two value-select renderings at both
   PTX targets. Context.compile bypasses the schedule cache. Use OCANNL_BACKEND=cuda
   OCANNL_CUDA_KERNEL_CACHE_MIB=0 dune exec \ benchmarks/runners/ocannl/bench_cuda_select.exe --
   [elements] [--dry-run] An exclusive measurement session and a ready measurement box are required
   for timings.

   The source check reads only this compile's returned CUDA source. It detects the renderer's first
   arm local by its declared name and compares the control source byte for byte. Branch instructions
   and select coverage within a kernel are outside that check. The dry run checks emitted forms,
   actual PTX targets, and host parity in every cell before enabling timings. *)
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

let compare_times ~calls ~label ~control_short ~control_blend ~short ~blend =
  let rows =
    Array.init calls ~f:(fun i ->
        if i % 2 = 0 then
          let a = measure short in
          let ca = measure control_short in
          let b = measure blend in
          let cb = measure control_blend in
          (a, b, ca, cb)
        else
          let cb = measure control_blend in
          let b = measure blend in
          let ca = measure control_short in
          let a = measure short in
          (a, b, ca, cb))
  in
  (* Treatment and control have to use the same rounds for their spreads to be comparable. *)
  let rows =
    Array.map rows ~f:(function
      | (Some _, Some _, Some _, Some _) as row -> row
      | _ -> (None, None, None, None))
  in
  let show name f = Bench_retime.spread_fields ~name (Bench_retime.spread_of (Array.map rows ~f)) in
  let ratio name num den =
    let r = Bench_retime.paired_ratio ~num:(Array.map rows ~f:num) ~den:(Array.map rows ~f:den) in
    if r.pairs < Bench_retime.min_readings then
      failwith (Printf.sprintf "%s: only %d of %d timing rounds admitted" label r.pairs calls);
    Bench_retime.ratio_fields ~name r
  in
  let a (a, _, _, _) = a and b (_, b, _, _) = b in
  let ca (_, _, ca, _) = ca and cb (_, _, _, cb) = cb in
  p "%s %s %s %s %s %s %s\n" label (show "short_ms" a) (show "blend_ms" b)
    (show "control_short_ms" ca) (show "control_blend_ms" cb) (ratio "blend_over_short" b a)
    (ratio "control_blend_over_short" cb ca)

let compare_code ~root ~target ~control work f =
  let short, src_a, cc_a, arch_a = compile ~root ~target ~blend:false work in
  Exn.protect
    ~finally:(fun () -> Context.release (fst short))
    ~f:(fun () ->
      let blend, src_b, cc_b, arch_b = compile ~root ~target ~blend:true work in
      Exn.protect
        ~finally:(fun () -> Context.release (fst blend))
        ~f:(fun () ->
          let label = Printf.sprintf "%s/%s" arch_a work.name in
          let has_local src =
            String.is_substring src ~substring:(Ir.C_syntax.blend_local_a ^ " =")
          in
          let matches =
            if control then String.equal src_a src_b && not (has_local src_b)
            else (not (has_local src_a)) && has_local src_b
          in
          Verdict.pass_fail (label ^ " emitted rendering") matches;
          let expected_native = Printf.sprintf "sm_%d" cc_a in
          let is_native = Mma.equal_ptx_target target Mma.Native in
          Verdict.pass_fail (label ^ " actual PTX targets")
            (cc_a = cc_b && String.equal arch_a arch_b
            && Bool.(String.equal arch_a expected_native = is_native));
          if Verdict.any_failed () then failwith (label ^ ": source or PTX check failed");
          f ~label short blend))

let () =
  let args = Bench_args.create "bench_cuda_select" in
  let elements = Bench_args.int args 0 ~name:"elements" ~default:(1 lsl 22) in
  let dry = Bench_args.flag args ~name:"dry-run" in
  if elements % 256 <> 0 then invalid_arg "elements must be a multiple of 256";
  if not (String.equal (Utils.get_global_arg ~arg_name:"cuda_kernel_cache_mib" ~default:"64") "0")
  then invalid_arg "set OCANNL_CUDA_KERNEL_CACHE_MIB=0 to witness every PTX compile";
  let root = Context.cuda () in
  Exn.protect
    ~finally:(fun () -> Context.release root)
    ~f:(fun () ->
      let calls = Bench_retime.calls_of_env () in
      p
        "backend=%s elements=%d pool_images=%d mask_density=1/32,31/32 timing=queued calls=%d \
         dry_run=%b\n"
        (Context.backend_name root) elements (elements / 16) calls dry;
      let node = Ll_test.node_factory ~first_id:20000 ~dims:[| elements |] () in
      let control = pointwise ~elements ~node Control in
      let workloads =
        List.map [ Where_zero; Relu_gate; Where_eq; Where_masked ] ~f:(pointwise ~elements ~node)
        @ [ pool ~root ~elements ]
      in
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
                            compare_times ~calls ~label ~control_short ~control_blend ~short ~blend))))))
