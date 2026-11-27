(* gh-ocannl-2281, gh-ocannl-2326, gh-ocannl-2385, gh-ocannl-2499. Replay fixed CUDA segments with
   the two value-select renderings at both PTX targets, then compile and time a whole training step
   (a two-conv classifier with ReLU and 2x2 max pooling, forward, backward and SGD) under each,
   beside a second short-circuiting compile as its control. Context.compile and
   Autotune.untuned_compile bypass the schedule cache. The CUDA value-select benchmark section of
   benchmarks/README.md gives the hermetic invocation and readiness checks.

   The source check reads only this compile's returned CUDA source. It detects the renderer's first
   arm local by its declared name and compares each segment rendering to its independently linked
   repeat byte for byte. The step lowers as well as renders under each choice, so its placements may
   differ; its check compares the control's source with the short arm's up to what each lowering
   mints afresh ([canonical] below). Branch instructions and select coverage within a kernel are
   outside that check. The dry run checks emitted forms, actual PTX targets, and host parity in
   every cell, and the step's values across the arms, before enabling timings.

   The output-placement replay accepts only the emitted where_masked declaration with its three
   ordered f32 pointer parameters. It reads the declaration modulo whitespace and leaves the kernel
   body to NVRTC. It checks the replay's PTX target, output addresses, and bitwise output including
   the masked tail before taking CUDA event timings. A timed run belongs in a generated-input
   measurement window after check-box.sh and a passing dry run. *)
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

type pointwise = Where_zero | Relu_gate | Where_eq | Where_masked

let pointwise ~elements ~node kind =
  let open Ll_test in
  let name =
    match kind with
    | Where_zero -> "where_zero"
    | Relu_gate -> "relu_gate"
    | Where_eq -> "where_eq"
    | Where_masked -> "where_masked"
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
          | Where_eq -> value i
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
      Array.mapi data ~f:(fun i g -> if i >= active then unwritten else if take i then g else 0.)
    in
    { seeds; want }
  in
  let fixtures = [ ("sparse", fixture ~dense:false); ("dense", fixture ~dense:true) ] in
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

let output_seed fixture = Array.create ~len:(Array.length fixture.want) unwritten
let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let check label (ctx, routine) work fixture =
  let ctx = Context.set_values ctx work.output (output_seed fixture) in
  let ctx =
    List.fold fixture.seeds ~init:ctx ~f:(fun ctx (node, values) ->
        Context.set_values ctx node values)
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx work.output in
  let want = fixture.want in
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

(* [before k] runs ahead of arm [k]'s timing, outside its clock. Each round uses the same readings
   for the treatment and its controls, so a refused reading drops the entire round. *)
let compare_times ?(before = fun _ -> ()) ?(show_rounds = false) ~measure_arm ~calls ~label ~arms ()
    =
  let n = List.length arms in
  let arms = Array.of_list arms in
  let orders = Bench_design.orders ~arms:n ~batches:calls in
  let rows =
    Array.mapi orders ~f:(fun i order ->
        let row = Array.create ~len:n None in
        Array.iter order ~f:(fun k ->
            before k;
            row.(k) <- measure_arm (snd arms.(k)));
        if show_rounds then
          p "%s round=%d %s\n" label (i + 1)
            (Array.to_list order
            |> List.map ~f:(fun k ->
                Printf.sprintf "%s_ms=%.6f" (fst arms.(k)) (Option.value row.(k) ~default:Float.nan))
            |> String.concat ~sep:" ");
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
  let admitted_orders =
    Array.filter_mapi orders ~f:(fun i order -> Option.some_if (Option.is_some rows.(i).(0)) order)
  in
  let admitted = Array.length admitted_orders in
  if admitted < Bench_retime.min_readings then
    failwith (Printf.sprintf "%s: only %d of %d timing rounds admitted" label admitted calls);
  let balance = Bench_design.balance ~arms:n admitted_orders in
  (String.concat ~sep:" " (List.map (Array.to_list arms) ~f:show), column, balance)

let with_segment_arms ~root ~target ~reverse work f =
  let order =
    [ ("short", false); ("blend", true); ("short_repeat", false); ("blend_repeat", true) ]
  in
  let link_order = if reverse then List.rev order else order in
  let linked = ref [] in
  Exn.protect
    ~finally:(fun () -> List.iter !linked ~f:(fun (_, ((ctx, _), _, _, _)) -> Context.release ctx))
    ~f:(fun () ->
      List.iter link_order ~f:(fun (name, blend) ->
          let arm = compile ~root ~target ~blend work in
          linked := (name, arm) :: !linked);
      let get name = List.Assoc.find_exn !linked ~equal:String.equal name in
      let _, src_a, cc_a, arch_a = get "short" and _, src_b, _, _ = get "blend" in
      let _, src_ar, _, _ = get "short_repeat" and _, src_br, _, _ = get "blend_repeat" in
      let label = Printf.sprintf "%s/%s" arch_a work.name in
      let expected_native = Printf.sprintf "sm_%d" cc_a in
      let is_native = Mma.equal_ptx_target target Mma.Native in
      Verdict.pass_fail (label ^ " emitted rendering") ((not (has_local src_a)) && has_local src_b);
      Verdict.pass_fail
        (label ^ " repeat source matches byte for byte")
        (String.equal src_a src_ar && String.equal src_b src_br);
      Verdict.Claims.p_all (label ^ " actual PTX targets") !linked ~f:(fun (_, (_, _, cc, arch)) ->
          cc = cc_a && String.equal arch arch_a
          && Bool.equal is_native (String.equal arch expected_native));
      if Verdict.any_failed () then failwith (label ^ ": source or PTX check failed");
      p "%s link_order=%s\n" label (String.concat ~sep:"," (List.map link_order ~f:fst));
      f ~label
        (List.map order ~f:(fun (name, _) ->
             let arm, _, _, _ = get name in
             (name, arm))))

let check_segment_buffers ~shared_output ~label arms work fixture =
  let n = List.length arms in
  let nodes =
    (work.output, if shared_output then 1 else n)
    :: List.map fixture.seeds ~f:(fun (node, _) -> (node, 1))
  in
  Verdict.Claims.p_all
    (label ^ if shared_output then " shared device buffers" else " only output buffers separate")
    nodes
    ~f:(fun (node, expected_locations) ->
      let locations =
        List.filter_map arms ~f:(fun (_, (ctx, _)) -> Context.buffer_location ctx node)
      in
      List.length locations = n
      && List.length (List.dedup_and_sort locations ~compare:Poly.compare) = expected_locations);
  if Verdict.any_failed () then failwith (label ^ ": segment buffer placement check failed")

let time_segment ~calls ~label ~show_rounds arms =
  let times, column, balance =
    compare_times ~show_rounds ~measure_arm:measure ~calls ~label ~arms ()
  in
  let r =
    Bench_retime.controlled_ratio ~base:(column "short") ~base_repeat:(column "short_repeat")
      ~variant:(column "blend") ~variant_repeat:(column "blend_repeat")
  in
  let balanced = balance.histograms_equal && balance.pairs_exchanged in
  let treatment = { r.treatment with resolved = balanced && r.treatment.resolved } in
  p
    "%s %s %s %s %s effect_lo=%.3f effect_hi=%.3f control_factor=%.3f timing_positions_balanced=%b\n"
    label times
    (Bench_retime.ratio_fields ~name:"blend_over_short" treatment)
    (Bench_retime.ratio_fields ~name:"short_repeat_over_short" r.base_control)
    (Bench_retime.ratio_fields ~name:"blend_repeat_over_blend" r.variant_control)
    r.effect_lo r.effect_hi r.control_factor balanced

let compare_segment ~root ~target ~dry ~calls ~shared_output ~reverse ~show_rounds work =
  let first = (snd (List.hd_exn work.fixtures)) () in
  let run parent =
    with_segment_arms ~root:parent ~target ~reverse work (fun ~label arms ->
        check_segment_buffers ~shared_output ~label arms work first;
        List.iteri work.fixtures ~f:(fun i (density, make_fixture) ->
            let fixture = if i = 0 then first else make_fixture () in
            let label = label ^ "/" ^ density in
            if String.equal density "pool" then
              p "%s nonzero_gradient=%d/%d\n" label
                (Array.count fixture.want ~f:(fun v -> Float.(v <> 0.)))
                (Array.length fixture.want);
            let arms =
              List.map arms ~f:(fun (name, arm) ->
                  (name, check (label ^ "/" ^ name) arm work fixture))
            in
            if not dry then time_segment ~calls ~label ~show_rounds arms))
  in
  (* Children inherit the parent's inputs in both layouts. Seeding its output as well removes the
     output placement difference without changing which stream the four links use. *)
  let parent = ref (Context.cuda ~ordinal:(Context.ordinal root) ()) in
  Exn.protect
    ~finally:(fun () -> Context.release !parent)
    ~f:(fun () ->
      List.iter first.seeds ~f:(fun (node, values) ->
          parent := Context.set_values !parent node values);
      if shared_output then parent := Context.set_values !parent work.output (output_seed first);
      run !parent)

(* A single function and stream per rendering. Within a cell the only launch argument allowed to
   change is the output pointer. The same-address visits measure repeat noise on both slabs. *)
let replay_output_placement ~root ~target ~dry ~calls ~elements ~reverse_payload_order work =
  let module Cu = Cuda in
  let module D = Cu.Deviceptr in
  let module B = Stdlib.Bigarray in
  let f32 values = B.(genarray_of_array1 (Array1.of_array float32 c_layout values)) in
  let with_buffer bytes f =
    let ptr = D.mem_alloc ~size_in_bytes:bytes in
    Exn.protect ~finally:(fun () -> D.mem_free ptr) ~f:(fun () -> f ptr)
  in
  let replay ~blend =
    let (ctx, routine), source, device_cc, arch = compile ~root ~target ~blend work in
    Context.release ctx;
    let launch =
      match routine.Context.segments with
      | [ segment ] -> LL.launch_dims segment.llc
      | _ -> failwith "expected one where_masked launch"
    in
    let label = arch ^ "/" ^ work.name ^ if blend then "/blend" else "/short" in
    let abi =
      {|extern "C" __global__ void where_masked(
          float *__restrict__ t_output, float *__restrict__ t_flag, float *__restrict__ t_payload) {|}
    in
    let compact = String.filter ~f:(fun c -> not (Char.is_whitespace c)) in
    Verdict.pass_fail (label ^ " replay declaration")
      (List.length
         (String.substr_index_all (compact source) ~may_overlap:false ~pattern:(compact abi))
       = 1
      && Bool.equal (has_local source) blend);
    if Verdict.any_failed () then failwith (label ^ ": unexpected replay declaration");
    let options =
      Ir.Compiler_options.nvrtc
        ~cuda_include_options:(Cuda_backend.cuda_include_options ())
        ~arch_options:(Mma.gpu_arch_options ~target ~device_cc source)
        ~fast_math:(Ir.Cuda_like_config.cuda_fast_math ())
        ~with_device_debug:(Utils.with_runtime_debug ())
        ~with_line_info:Utils.settings.output_debug_files_in_build_directory ()
    in
    let ptx =
      Nvrtc.compile_to_ptx ~cu_src:source ~name:(work.name ^ ".cu") ~options ~with_debug:false
    in
    Verdict.pass_fail (label ^ " replay PTX target")
      (Option.equal String.equal (Some arch) (Mma.ptx_target_directive (Nvrtc.string_from_ptx ptx))
      && Bool.equal
           (Mma.equal_ptx_target target Mma.Native)
           (String.equal arch (Printf.sprintf "sm_%d" device_cc)));
    if Verdict.any_failed () then failwith (label ^ ": unexpected replay PTX target");
    let primary = Cu.Context.get_primary (Cu.Device.get ~ordinal:(Context.ordinal root)) in
    Cu.Context.set_current primary;
    let func = Cu.Module.get_function (Cu.Module.load_data_ex ptx []) ~name:work.name in
    let stream = Cu.Stream.create ~non_blocking:true () in
    let upload ?dst_offset ~dst src =
      Cu.Stream.memcpy_H_to_D ~length:elements ?dst_offset ~dst ~src stream;
      Cu.Stream.synchronize stream;
      ignore (Stdlib.Sys.opaque_identity src)
    in
    let start = Cu.Event.create ~enable_timing:true ()
    and stop = Cu.Event.create ~enable_timing:true () in
    let batch = 128 in
    let submit args =
      Cu.Stream.launch_kernel func ~grid_dim_x:launch.grid.(0) ~grid_dim_y:launch.grid.(1)
        ~grid_dim_z:launch.grid.(2) ~block_dim_x:launch.block.(0) ~block_dim_y:launch.block.(1)
        ~block_dim_z:launch.block.(2) ~shared_mem_bytes:0 stream args
    in
    let queue args =
      for _ = 1 to batch do
        submit args
      done
    in
    let measure_arm args =
      queue args;
      Cu.Event.record start stream;
      queue args;
      Cu.Event.record stop stream;
      (* The stream wait also releases cudajit's retained launch arguments for this reading. *)
      Cu.Stream.synchronize stream;
      let ms = Cu.Event.elapsed_time ~start ~end_:stop /. Float.of_int batch in
      Option.some_if (Float.is_finite ms && Float.is_positive ms) ms
    in
    let bytes = elements * Ir.Ops.prec_in_bytes Ir.Ops.single in
    let slab_bytes = (2 * bytes) + 128 in
    with_buffer bytes @@ fun flag ->
    with_buffer slab_bytes @@ fun a ->
    with_buffer slab_bytes @@ fun b ->
    Exn.protect
      ~finally:(fun () -> Cu.Stream.synchronize stream)
      ~f:(fun () ->
        let cell ~payload_slab ~payload ~separate =
          let label = label ^ "/payload_" ^ payload_slab in
          let placements =
            [
              ("same", payload, 0);
              ("separate", separate, 0);
              ("same_4", payload, 4);
              ("separate_4", separate, 4);
              ("same_128", payload, 128);
              ("separate_128", separate, 128);
            ]
            |> List.map ~f:(fun (name, ptr, shift) ->
                (name, D.offset ptr ~bytes:(bytes + shift), shift))
          in
          let address ptr = Int64.of_string ("0x" ^ D.string_of ptr) in
          let output_addr (region : D.region) =
            Int64.(address region.base + of_int region.offset_bytes)
          in
          let addresses = List.map placements ~f:(fun (_, region, _) -> output_addr region) in
          Verdict.Claims.p_all (label ^ " output address residues") placements
            ~f:(fun (_, region, shift) ->
              Int64.(to_int_exn (output_addr region land of_int 255)) = shift);
          Verdict.Claims.p_pairwise_distinct
            (label ^ " distinct output addresses")
            addresses ~equal:Int64.equal ~to_string:(Printf.sprintf "0x%Lx");
          if Verdict.any_failed () then failwith (label ^ ": output address check failed");
          List.iter placements ~f:(fun (name, (region : D.region), _) ->
              p "%s %s base=0x%Lx offset_bytes=%d address=0x%Lx\n" label name (address region.base)
                region.offset_bytes (output_addr region));
          p "%s flag=0x%Lx payload=0x%Lx grid=%dx%dx%d block=%dx%dx%d launches_per_reading=%d\n"
            label (address flag) (address payload) launch.grid.(0) launch.grid.(1) launch.grid.(2)
            launch.block.(0) launch.block.(1) launch.block.(2) batch;
          let args region = Cu.Stream.[ Tensor_at region; Tensor flag; Tensor payload ] in
          let arms = List.map placements ~f:(fun (name, region, _) -> (name, args region)) in
          let repeat name = (name ^ "_repeat", List.Assoc.find_exn arms ~equal:String.equal name) in
          let arms = arms @ [ repeat "same"; repeat "separate" ] in
          let planned =
            Bench_design.balance ~arms:(List.length arms)
              (Bench_design.orders ~arms:(List.length arms) ~batches:calls)
          in
          if not (planned.histograms_equal && planned.pairs_exchanged) then
            invalid_arg
              "output-placement probe requires BENCH_RETIME_CALLS to be a multiple of eight";
          List.iter work.fixtures ~f:(fun (density, make_fixture) ->
              let label = label ^ "/" ^ density in
              let fixture = make_fixture () in
              let seed = f32 (output_seed fixture) in
              let readback = B.Array1.create B.float32 B.c_layout elements in
              let got = Array.create ~len:elements 0. in
              (match fixture.seeds with
              | [ (_, flags); (_, data) ] ->
                  upload ~dst:flag (f32 flags);
                  upload ~dst:payload (f32 data)
              | _ -> failwith "expected where_masked flag and payload");
              List.iter placements ~f:(fun (name, (region : D.region), _) ->
                  upload ~dst_offset:region.offset_bytes ~dst:region.base seed;
                  submit (args region);
                  Cu.Stream.synchronize stream;
                  D.memcpy_D_to_H ~length:elements ~src_offset:region.offset_bytes ~src:region.base
                    ~dst:(B.genarray_of_array1 readback) ();
                  for i = 0 to elements - 1 do
                    got.(i) <- readback.{i}
                  done;
                  Verdict.pass_fail_all2
                    (label ^ "/" ^ name ^ " host parity")
                    got fixture.want ~f:same_bits);
              if Verdict.any_failed () then failwith (label ^ ": placement readback failed");
              if not dry then (
                let _, column, balance =
                  compare_times ~show_rounds:true ~measure_arm ~calls ~label ~arms ()
                in
                if not (balance.histograms_equal && balance.pairs_exchanged) then
                  failwith (label ^ ": unbalanced output-placement readings");
                let aligned =
                  Bench_retime.controlled_ratio ~base:(column "same")
                    ~base_repeat:(column "same_repeat") ~variant:(column "separate")
                    ~variant_repeat:(column "separate_repeat")
                in
                p
                  "%s separate_over_same_all_pairs_lo=%.3f separate_over_same_all_pairs_hi=%.3f \
                   control_factor=%.3f aligned_allocation_resolved=%b\n"
                  label aligned.effect_lo aligned.effect_hi aligned.control_factor
                  aligned.treatment.resolved;
                List.iter
                  [
                    ("same_repeat", "same");
                    ("separate_repeat", "separate");
                    ("separate", "same");
                    ("separate_4", "same_4");
                    ("separate_128", "same_128");
                    ("same_4", "same");
                    ("same_128", "same");
                    ("separate_4", "separate");
                    ("separate_128", "separate");
                  ]
                  ~f:(fun (num, den) ->
                    let r = Bench_retime.paired_ratio ~num:(column num) ~den:(column den) in
                    p "%s %s\n" label (Bench_retime.ratio_fields ~name:(num ^ "_over_" ^ den) r))))
        in
        (* Each output address serves both roles, with fixed inputs in each timing cell. *)
        let roles = [ ("a", a, b); ("b", b, a) ] in
        let roles = if reverse_payload_order then List.rev roles else roles in
        List.iter roles ~f:(fun (payload_slab, payload, separate) ->
            cell ~payload_slab ~payload ~separate))
  in
  List.iter [ false; true ] ~f:(fun blend -> replay ~blend)

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
        let times, column, _ =
          compare_times ~before:restore ~measure_arm:measure ~calls ~label
            ~arms:(List.map arms ~f:(fun (name, (c, _, _, _)) -> (name, c)))
            ()
        in
        let ratio name = Bench_retime.paired_ratio ~num:(column name) ~den:(column "short") in
        p "%s %s %s %s\n" label times
          (Bench_retime.ratio_fields ~name:"blend_over_short" (ratio "blend"))
          (Bench_retime.ratio_fields ~name:"control_over_short" (ratio "control")))

let () =
  let args = Bench_args.create "bench_cuda_select" in
  let elements = Bench_args.int args 0 ~name:"elements" ~default:(1 lsl 22) in
  let batch = Bench_args.int args 1 ~flag:"step-batch" ~name:"step-batch" ~default:64 in
  let dry = Bench_args.flag args ~name:"dry-run" in
  let output_placement = Bench_args.flag args ~name:"output-placement-probe" in
  let reverse_payload_order = Bench_args.flag args ~name:"reverse-probe-payload-order" in
  let masked_only = Bench_args.flag args ~name:"where-masked-only" in
  let shared_output = not (Bench_args.flag args ~name:"separate-segment-outputs") in
  let reverse = Bench_args.flag args ~name:"reverse-segment-link-order" in
  if output_placement && ((not masked_only) || (not shared_output) || reverse) then
    invalid_arg "--output-placement-probe requires --where-masked-only and sets its own layouts";
  if reverse_payload_order && not output_placement then
    invalid_arg "--reverse-probe-payload-order requires --output-placement-probe";
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
      let calls = Bench_retime.calls_of_env ~default:8 () in
      if output_placement then
        p
          "backend=%s elements=%d mask_density=1/32,31/32 timing=cuda_event_fixed_batch calls=%d \
           dry_run=%b where_masked_output_placement=true payload_order=%s\n"
          backend elements calls dry
          (if reverse_payload_order then "b,a" else "a,b")
      else
        p
          "backend=%s elements=%d mask_density=1/32,31/32 timing=queued calls=%d dry_run=%b \
           where_masked_only=%b shared_segment_output=%b\n"
          backend elements calls dry masked_only shared_output;
      p "%s\n" (Bench_design.describe ~arms:(if output_placement then 8 else 4) ~batches:calls);
      let node = Ll_test.node_factory ~first_id:20000 ~dims:[| elements |] () in
      let segment target =
        compare_segment ~root ~target ~dry ~calls ~shared_output ~reverse ~show_rounds:masked_only
      in
      let targets = [ Mma.Floor; Mma.Native ] in
      if masked_only then
        let work = pointwise ~elements ~node Where_masked in
        List.iter targets ~f:(fun target ->
            if output_placement then
              replay_output_placement ~root ~target ~dry ~calls ~elements ~reverse_payload_order
                work
            else segment target work)
      else
        let step_calls = Bench_retime.calls_of_env () in
        p "pool_images=%d step=cifar_conv step_batch=%d step_calls=%d\n" (elements / 16) batch
          step_calls;
        let workloads =
          List.map [ Where_zero; Relu_gate; Where_eq; Where_masked ] ~f:(pointwise ~elements ~node)
          @ [ pool ~root ~elements ]
        in
        let step = build_step ~root ~batch in
        Exn.protect
          ~finally:(fun () -> Context.release step.init_ctx)
          ~f:(fun () ->
            List.iter targets ~f:(fun target ->
                List.iter workloads ~f:(segment target);
                compare_step ~target ~dry ~calls:step_calls step)))
