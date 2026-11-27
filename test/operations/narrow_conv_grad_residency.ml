(* gh-ocannl-2423, gh-ocannl-2464: an overlapping 1-D or 2-D convolution's narrow input gradient
   rounds once after all windows and channels reaching a cell. The materialized f32 twin and the
   privatized schedule both execute. The host controls distinguish rounding per step and rounding
   per window. Hand-built 1-D and 2-D affine nests check original contribution order with
   order-sensitive sums and interleaved channel loops, and pin the coupled-pair and residue-budget
   declines. Strided unit-tap axes exercise both affine paths (gh-ocannl-2479), with hand-built
   controls for shifted, negative, and guard-dependent single-term axes. *)

(* cc-fp-contract: exact-operands -- the dyadic products and all sums onto the incoming values fit
   the f32 significand. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Sched = Ir.Schedule

let bf16 = Ir.Ops.bfloat16
let positions = 253
let channels = 6
let taps = 5
let input_extent stride = ((positions - 1) * stride) + taps
let weight k d = Float.of_int (1 + (((3 * k) + d) % 5)) /. 8.

let cotangent ~stride o d =
  let scale = if stride < taps then 1 else 3 in
  Float.of_int (scale * (1 + (((5 * o) + (3 * d)) % 7))) /. 16.

let incoming ~base stride =
  Array.init (input_extent stride) ~f:(fun i -> Float.of_int (base + (2 * (i % 3))))

let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let gradient ~stride prec =
  let x =
    Operation.init ~l:"ncgr_x" ~prec:Ir.Ops.single
      ~o:[ input_extent stride ]
      ~f:(fun idx -> Float.of_int (1 + idx.(0)) /. 8.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    NTDSL.init ~l:"ncgr_w" ~prec:Ir.Ops.single ~o:[ taps; channels ]
      ~f:(fun idx -> weight idx.(0) idx.(1))
      ()
  in
  let c =
    NTDSL.init ~l:"ncgr_c" ~prec:Ir.Ops.single ~o:[ positions; channels ]
      ~f:(fun idx -> cotangent ~stride idx.(0) idx.(1))
      ()
  in
  let use_padding = false in
  let%op y = x +* "stride*o+k; k, d => o, d" w in
  let%op loss = (y *. c) ++ "... => 0" in
  let grad (t : Tensor.t) = (Option.value_exn t.diff).Tensor.grad in
  let dx = grad x in
  Tn.update_prec dx prec;
  Train.set_materialized dx;
  Train.set_materialized (grad y);
  (Train.grad_update ~accum_steps:1 loss, dx)

type schedule = Privatized | Split_privatized | Split_unrolled

let run_fixture ~stride ~initial ~tag ?schedule prec =
  let comp, dx = gradient ~stride prec in
  let ctx = Context.auto () in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let lowered_transform =
    Option.map schedule ~f:(fun schedule opt ->
        let preset =
          match schedule with
          | Privatized -> []
          | Split_privatized | Split_unrolled -> (
              let phase =
                LL.flat_lines [ opt.LL.llc ]
                |> List.filter ~f:(fun st -> not (List.is_empty (LL.accumulating_writes ~tn:dx st)))
                |> List.last_exn
              in
              let owner, reduction =
                match Ll_test.nest_paths phase with
                | [ owner :: reduction :: _ ] -> (owner, reduction)
                | _ -> failwith "expected a cell-first residue nest"
              in
              let axis = owner in
              let op, _, _ = Sched.split ~axis ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
              match schedule with
              | Split_unrolled -> [ op; Sched.Unroll { axis = reduction; materialize = true } ]
              | _ -> [ op ])
        in
        let sched =
          match schedule with
          | Split_unrolled -> preset
          | _ -> Autotune.extend_with_privatize ~accum_prec ~static_indices:[] preset opt
        in
        [ Sched.apply sched opt ])
  in
  let name = Printf.sprintf "ncgr_%s_%d" tag stride in
  let ctx, routine = Context.compile ~name ?lowered_transform ctx comp Ir.Indexing.Empty in
  let ctx = Context.set_values ctx dx initial in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let tiles = Hash_set.create (module Tn) in
  List.iter routine.Context.segments ~f:(fun segment ->
      Ll_test.walk segment.LL.llc ~on_stmt:(function
        | LL.Set { tn; _ } ->
            if List.equal String.equal tn.Tn.label ("acc" :: dx.Tn.label) then Hash_set.add tiles tn
        | _ -> ()));
  let values = Context.get_values ctx dx in
  Context.release ctx;
  (values, Hash_set.to_list tiles)

let terms ~stride cell =
  List.filter_map (List.range 0 positions) ~f:(fun o ->
      let k = cell - (stride * o) in
      if k < 0 || k >= taps then None
      else Some (List.init channels ~f:(fun d -> weight k d *. cotangent ~stride o d)))

let modeled ~stride initial ~step ~window =
  Array.mapi initial ~f:(fun cell initial ->
      List.fold (terms ~stride cell) ~init:initial ~f:(fun acc terms ->
          window (List.fold terms ~init:acc ~f:(fun acc term -> step (acc +. term)))))

let discriminates ~label ~once when_ other =
  p_exists
    (Printf.sprintf "%s: rounding %s differs from rounding once per cell" label when_)
    (List.zip_exn (Array.to_list other) (Array.to_list once))
    ~f:(fun (a, b) -> not (same a b))

let check stride =
  let label = Printf.sprintf "stride %d" stride in
  let initial = incoming ~base:256 stride in
  let modeled = modeled ~stride initial in
  let run_fixture = run_fixture ~stride ~initial in
  let round = Ir.Ops.round_to_prec bf16 in
  let exact = modeled ~step:Fn.id ~window:Fn.id in
  let twin, _ = run_fixture ~tag:"f32" Ir.Ops.single in
  let narrow, _ = run_fixture ~tag:"bf16" bf16 in
  let scheduled, tiles = run_fixture ~tag:"priv" ~schedule:Privatized bf16 in
  let once = Array.map twin ~f:round in
  pf_all2 "%s: the materialized f32 gradient equals the host sums" label twin exact ~f:same;
  discriminates ~label ~once "per step" (modeled ~step:round ~window:Fn.id);
  if stride < taps then discriminates ~label ~once "per window" (modeled ~step:Fn.id ~window:round);
  pf_all2 "%s: the bf16 gradient equals the f32 gradient rounded once per cell" label narrow once
    ~f:same;
  p_all (Printf.sprintf "%s: the input gradient's accumulator tiles are f32" label) tiles
    ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single);
  pf_all2 "%s: the privatized gradient equals the f32 gradient rounded once per cell" label
    scheduled once ~f:same;
  if stride = 2 then (
    let split, tiles = run_fixture ~tag:"split" ~schedule:Split_privatized bf16 in
    p "the split residue with ambiguous entry provenance stays a serial accumulation"
      (List.length tiles = 1);
    p_all2 "the split owner's tail guard preserves once-rounded parity across both residues" split
      once ~f:same;
    let unrolled, _ = run_fixture ~tag:"unrolled" ~schedule:Split_unrolled bf16 in
    p_all2 "materializing the unrolled reduction under the tail guard preserves parity" unrolled
      once ~f:same)

let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_wide })
    (fun () -> List.iter [ 1; 2; 6 ] ~f:check)

let check_fp16 stride =
  let label = Printf.sprintf "FP16 stride %d" stride in
  (* At 2048, adjacent FP16 values are two apart, as in BF16 at 256. The same window contributions
     can expose early rounding in either format. *)
  let initial = incoming ~base:2048 stride in
  let round = Ir.Ops.round_to_prec Ir.Ops.half in
  let exact = modeled ~stride initial ~step:Fn.id ~window:Fn.id in
  let per_window = modeled ~stride initial ~step:Fn.id ~window:round in
  let twin, _ = run_fixture ~stride ~initial ~tag:"fp16_f32" Ir.Ops.single in
  let narrow, _ = run_fixture ~stride ~initial ~tag:"fp16" Ir.Ops.half in
  let once = Array.map twin ~f:round in
  pf_all2 "%s: the materialized f32 gradient equals the host sums" label twin exact ~f:same;
  discriminates ~label ~once "per window" per_window;
  pf_all2 "%s: the gradient equals the f32 gradient rounded once per cell" label narrow once ~f:same

let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_wide })
    (fun () -> List.iter [ 1; 2 ] ~f:check_fp16)

(* gh-ocannl-2464: executed 2-D overlapping (and mixed overlapping/non-overlapping) bf16 and
   overlapping fp16 convolution input-gradient residency parity against materialized f32 sums
   rounded once. *)
type geom2d = {
  pos_h : int;
  pos_w : int;
  taps_h : int;
  taps_w : int;
  channels_2d : int;
  stride_h : int;
  stride_w : int;
}

let input_extent_2d g =
  (((g.pos_h - 1) * g.stride_h) + g.taps_h, ((g.pos_w - 1) * g.stride_w) + g.taps_w)

let weight_2d k t d = Float.of_int (1 + (((3 * k) + (2 * t) + d) % 5)) /. 8.

let cotangent_2d g o p d =
  let scale = if g.stride_h < g.taps_h || g.stride_w < g.taps_w then 1 else 3 in
  Float.of_int (scale * (1 + (((5 * o) + (3 * p) + (2 * d)) % 7))) /. 16.

let incoming_2d ~base g =
  let in_h, in_w = input_extent_2d g in
  List.init in_h ~f:(fun ih ->
      List.init in_w ~f:(fun iw -> Float.of_int (base + (2 * ((ih + iw) % 3)))))
  |> List.concat |> Array.of_list

let gradient_2d g prec =
  let in_h, in_w = input_extent_2d g in
  let stride_h = g.stride_h and stride_w = g.stride_w in
  let x =
    Operation.init ~l:"ncgr2d_x" ~prec:Ir.Ops.single ~o:[ in_h; in_w ]
      ~f:(fun idx -> Float.of_int (1 + (2 * idx.(0)) + idx.(1)) /. 8.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    NTDSL.init ~l:"ncgr2d_w" ~prec:Ir.Ops.single
      ~o:[ g.taps_h; g.taps_w; g.channels_2d ]
      ~f:(fun idx -> weight_2d idx.(0) idx.(1) idx.(2))
      ()
  in
  let c =
    NTDSL.init ~l:"ncgr2d_c" ~prec:Ir.Ops.single
      ~o:[ g.pos_h; g.pos_w; g.channels_2d ]
      ~f:(fun idx -> cotangent_2d g idx.(0) idx.(1) idx.(2))
      ()
  in
  let use_padding = false in
  let%op y = x +* "stride_h*o+k, stride_w*p+t; k, t, d => o, p, d" w in
  let%op loss = (y *. c) ++ "... => 0" in
  let grad (t : Tensor.t) = (Option.value_exn t.diff).Tensor.grad in
  let dx = grad x in
  Tn.update_prec dx prec;
  Train.set_materialized dx;
  Train.set_materialized (grad y);
  (Train.grad_update ~accum_steps:1 loss, dx)

let run_fixture_2d g ~initial ~tag ?(privatize = false) prec =
  let comp, dx = gradient_2d g prec in
  let ctx = Context.auto () in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let lowered_transform =
    if not privatize then None
    else
      Some
        (fun opt ->
          let sched = Autotune.extend_with_privatize ~accum_prec ~static_indices:[] [] opt in
          [ Sched.apply sched opt ])
  in
  let name = Printf.sprintf "ncgr2d_%s_%d_%d" tag g.stride_h g.stride_w in
  let ctx, routine = Context.compile ~name ?lowered_transform ctx comp Ir.Indexing.Empty in
  let ctx = Context.set_values ctx dx initial in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let tiles = Hash_set.create (module Tn) in
  List.iter routine.Context.segments ~f:(fun segment ->
      Ll_test.walk segment.LL.llc ~on_stmt:(function
        | LL.Set { tn; _ } ->
            if List.equal String.equal tn.Tn.label ("acc" :: dx.Tn.label) then Hash_set.add tiles tn
        | _ -> ()));
  let values = Context.get_values ctx dx in
  Context.release ctx;
  (values, Hash_set.to_list tiles)

let terms_2d g ih iw =
  List.concat_map (List.range 0 g.pos_h) ~f:(fun o ->
      let k = ih - (g.stride_h * o) in
      if k < 0 || k >= g.taps_h then []
      else
        List.filter_map (List.range 0 g.pos_w) ~f:(fun p ->
            let t = iw - (g.stride_w * p) in
            if t < 0 || t >= g.taps_w then None
            else
              Some (List.init g.channels_2d ~f:(fun d -> weight_2d k t d *. cotangent_2d g o p d))))

let modeled_2d g initial ~step ~window =
  let _in_h, in_w = input_extent_2d g in
  Array.mapi initial ~f:(fun flat initial ->
      let ih = flat / in_w and iw = flat % in_w in
      List.fold (terms_2d g ih iw) ~init:initial ~f:(fun acc terms ->
          window (List.fold terms ~init:acc ~f:(fun acc term -> step (acc +. term)))))

let check_2d_prec ~label_prefix ~prec_name ~base ~f32_tag ~narrow_tag ~priv_tag prec g =
  let label = Printf.sprintf "%s stride (%d, %d)" label_prefix g.stride_h g.stride_w in
  let initial = incoming_2d ~base g in
  let modeled = modeled_2d g initial in
  let round = Ir.Ops.round_to_prec prec in
  let exact = modeled ~step:Fn.id ~window:Fn.id in
  let twin, _ = run_fixture_2d g ~initial ~tag:f32_tag Ir.Ops.single in
  let narrow, _ = run_fixture_2d g ~initial ~tag:narrow_tag prec in
  let scheduled, tiles = run_fixture_2d g ~initial ~tag:priv_tag ~privatize:true prec in
  let once = Array.map twin ~f:round in
  pf_all2 "%s: the materialized f32 gradient equals the host sums" label twin exact ~f:same;
  discriminates ~label ~once "per step" (modeled ~step:round ~window:Fn.id);
  if g.stride_h < g.taps_h || g.stride_w < g.taps_w then
    discriminates ~label ~once "per window" (modeled ~step:Fn.id ~window:round);
  pf_all2 "%s: the %s gradient equals the f32 gradient rounded once per cell" label prec_name narrow
    once ~f:same;
  p_all (Printf.sprintf "%s: the input gradient's accumulator tiles are f32" label) tiles
    ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single);
  pf_all2 "%s: the privatized gradient equals the f32 gradient rounded once per cell" label
    scheduled once ~f:same

let check_2d =
  check_2d_prec ~label_prefix:"2-D" ~prec_name:"bf16" ~base:256 ~f32_tag:"f32" ~narrow_tag:"bf16"
    ~priv_tag:"priv" bf16

let check_fp16_2d =
  check_2d_prec ~label_prefix:"2-D FP16" ~prec_name:"fp16" ~base:2048 ~f32_tag:"fp16_f32"
    ~narrow_tag:"fp16" ~priv_tag:"fp16_priv" Ir.Ops.half

let overlapping_geoms_2d =
  [
    { pos_h = 13; pos_w = 11; taps_h = 3; taps_w = 3; channels_2d = 3; stride_h = 1; stride_w = 1 };
    { pos_h = 13; pos_w = 11; taps_h = 5; taps_w = 5; channels_2d = 3; stride_h = 2; stride_w = 2 };
  ]

let unit_tap_overlap_2d =
  { pos_h = 7; pos_w = 11; taps_h = 1; taps_w = 3; channels_2d = 6; stride_h = 3; stride_w = 2 }

let () =
  let geoms =
    overlapping_geoms_2d
    @ [
        {
          pos_h = 13;
          pos_w = 7;
          taps_h = 5;
          taps_w = 5;
          channels_2d = 3;
          stride_h = 2;
          stride_w = 6;
        };
        {
          pos_h = 5;
          pos_w = 4;
          taps_h = 3;
          taps_w = 2;
          channels_2d = 3;
          stride_h = 4;
          stride_w = 3;
        };
        unit_tap_overlap_2d;
        {
          pos_h = 7;
          pos_w = 11;
          taps_h = 3;
          taps_w = 1;
          channels_2d = 3;
          stride_h = 4;
          stride_w = 2;
        };
      ]
  in
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_wide })
    (fun () -> List.iter geoms ~f:check_2d)

let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_wide })
    (fun () -> List.iter (overlapping_geoms_2d @ [ unit_tap_overlap_2d ]) ~f:check_fp16_2d)

let () =
  let open Ll_test in
  let i = sym () and d = sym () in
  let out = node_factory ~prec:bf16 ~first_id:24790 ~dims:[| 6 |] () "ncgr_lone_axis" in
  let source ?(guard = Fn.id) cell =
    let update = set_at out cell (add_own (get out [| cell |]) (embed d)) in
    loop_n d 3 (loop_n i 3 (guard update))
  in
  let path_is path code =
    List.equal (List.equal Ir.Indexing.equal_symbol) (nest_paths code) [ path ]
  in
  let width_is wide code =
    match LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:out code with
    | [ site ] -> Bool.equal site.LL.sa_widens wide
    | _ -> false
  in
  let shifted = source (aff [ (2, i) ] 1) |> LL.sink_narrow_reduction_loops in
  p "the shifted positive affine axis sinks its reduction and widens"
    (path_is [ i; d ] shifted && width_is true shifted);
  let reverse = source (aff [ (-2, i) ] 5) |> LL.sink_narrow_reduction_loops in
  p "the in-bounds negative affine axis keeps its reduction outside"
    (path_is [ d; i ] reverse && width_is false reverse);
  let outside = aff [ (2, i) ] (-1) in
  let guard = if_ (cmp Ir.Ops.Cmple (embed_idx (fixed 0)) (embed_idx outside)) in
  let guarded = source ~guard outside |> LL.sink_narrow_reduction_loops in
  p "the positive affine axis needing a bounds guard keeps its reduction outside"
    (path_is [ d; i ] guarded && width_is false guarded)

let () =
  let open Ll_test in
  let module Nd = Ir.Ndarray in
  let same_index = List.equal Ir.Indexing.equal_symbol in
  let check_guarded_fp8 ~label ~first_id ~dims ~initial ~untouched ~updated ~expected_sunk_paths
      ~build () =
    let out = node_factory ~prec:Ir.Ops.fp8 ~first_id ~dims () ("ncgr_guarded_fp8_" ^ label) in
    materialize out;
    let source = build out in
    let sunk = LL.sink_narrow_reduction_loops source in
    let seed =
      Bigarray.Array1.of_array Bigarray.Char Bigarray.C_layout
        (Array.map initial ~f:Char.of_int_exn)
      |> Bigarray.genarray_of_array1
    in
    let run ~name code =
      let opt = optimize_scoped ~name ~raw:source code in
      let ctx, routine = link ~name opt in
      let ctx = Context.from_host ctx out (Nd.Fp8_nd seed) in
      let ctx = Context.run ctx routine in
      let result = Context.to_host ctx out in
      Context.release ctx;
      match result with
      | Nd.Fp8_nd bytes ->
          let bytes = Bigarray.array1_of_genarray bytes in
          Array.init dims.(0) ~f:(fun j -> Char.to_int bytes.{j})
      | _ -> failwith "expected FP8 storage"
    in
    let reference = run ~name:("ncgr_guarded_fp8_" ^ label ^ "_reference") source in
    let actual = run ~name:("ncgr_guarded_fp8_" ^ label ^ "_sunk") sunk in
    let fmt_bytes bs =
      Array.to_list bs |> List.map ~f:(Printf.sprintf "0x%02x") |> String.concat ~sep:" "
    in
    Stdlib.Printf.eprintf "guarded_fp8 %s: ref=[%s] actual=[%s] (not part of the golden)\n%!" label
      (fmt_bytes reference) (fmt_bytes actual);
    let widths = LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:out sunk in
    p
      (label ^ ": guarded FP8 reduction sinks to expected nest order")
      (expected_sunk_paths (nest_paths sunk));
    p_all (label ^ ": every sunk guarded FP8 reduction widens") widths ~f:(fun s -> s.LL.sa_widens);
    p_all (label ^ ": the guarded FP8 reference preserves untouched bytes") untouched ~f:(fun idx ->
        reference.(idx) = initial.(idx));
    p_all (label ^ ": the guarded FP8 reference updates its finite cells") updated ~f:(fun idx ->
        reference.(idx) <> initial.(idx));
    p_all2
      (label ^ ": guarded FP8 preserves the materialized result byte for byte")
      actual reference ~f:Int.equal
  in
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.narrow_compute_f32 = true })
    (fun () ->
      (* Plain iterator path: [for d, for i, if (d + 1 < i) out[i] += 1]. *)
      let d = sym () and i = sym () in
      check_guarded_fp8 ~label:"plain" ~first_id:24791 ~dims:[| 4 |]
        ~initial:[| 0x7c; 0x7d; 0x3c; 0x3c |] ~untouched:[ 0; 1 ] ~updated:[ 2; 3 ]
        ~expected_sunk_paths:(fun paths -> List.equal same_index paths [ [ i; d ] ])
        ~build:(fun out ->
          let cell = iter i in
          let guard = cmp Ir.Ops.Cmplt (embed_idx (aff [ (1, d) ] 1)) (embed i) in
          let update = set_at out cell (add_own (get out [| cell |]) (c 1.)) in
          loop_n d 3 (loop_n i 4 (if_ guard update)))
        ();
      (* Single-term positive affine path: [for d, for i, if (d + 1 < i) out[2*i] += 1]. *)
      let d = sym () and i = sym () in
      check_guarded_fp8 ~label:"single_term" ~first_id:24792 ~dims:[| 8 |]
        ~initial:[| 0x7c; 0x40; 0x7d; 0x44; 0x3c; 0x48; 0x3c; 0x4c |]
        ~untouched:[ 0; 1; 2; 3; 5; 7 ] ~updated:[ 4; 6 ]
        ~expected_sunk_paths:(fun paths -> List.equal same_index paths [ [ i; d ] ])
        ~build:(fun out ->
          let cell = aff [ (2, i) ] 0 in
          let guard = cmp Ir.Ops.Cmplt (embed_idx (aff [ (1, d) ] 1)) (embed i) in
          let update = set_at out cell (add_own (get out [| cell |]) (c 1.)) in
          loop_n d 3 (loop_n i 4 (if_ guard update)))
        ();
      (* Injective-pair path: [for d, for i, for j, if (d + 1 < 2*i + j) out[2*i + j] += 1]. *)
      let d = sym () and i = sym () and j = sym () in
      check_guarded_fp8 ~label:"injective_pair" ~first_id:24793 ~dims:[| 4 |]
        ~initial:[| 0x7c; 0x7d; 0x3c; 0x3c |] ~untouched:[ 0; 1 ] ~updated:[ 2; 3 ]
        ~expected_sunk_paths:(fun paths -> List.equal same_index paths [ [ i; j; d ] ])
        ~build:(fun out ->
          let cell = aff [ (2, i); (1, j) ] 0 in
          let guard = cmp Ir.Ops.Cmplt (embed_idx (aff [ (1, d) ] 1)) (embed_idx cell) in
          let update = set_at out cell (add_own (get out [| cell |]) (c 1.)) in
          loop_n d 3 (loop_n i 2 (loop_n j 2 (if_ guard update))))
        ();
      (* Overlapping-pair path: [for d, for i, for j, if (d + 1 < i + j) out[i + j] += 1]. *)
      let d = sym () and i = sym () and j = sym () in
      check_guarded_fp8 ~label:"overlapping_pair" ~first_id:24794 ~dims:[| 5 |]
        ~initial:[| 0x7c; 0x7d; 0x3c; 0x3c; 0x3c |]
        ~untouched:[ 0; 1 ] ~updated:[ 2; 3; 4 ]
        ~expected_sunk_paths:(fun paths ->
          match paths with [ [ _owner; d'; _step ] ] -> Ir.Indexing.equal_symbol d d' | _ -> false)
        ~build:(fun out ->
          let cell = aff [ (1, i); (1, j) ] 0 in
          let guard = cmp Ir.Ops.Cmplt (embed_idx (aff [ (1, d) ] 1)) (embed_idx cell) in
          let update = set_at out cell (add_own (get out [| cell |]) (c 1.)) in
          loop_n d 3 (loop_n i 3 (loop_n j 3 (if_ guard update))))
        ();
      (* Schedule-minted [Unroll ~materialize:true] and [Partition] over a guarded FP8 reduction
         share the same conditional write-back, and wide residency is preserved on updated cells:
         [4.0 (0x44) + 4 * 0.5] stays [4.0 (0x44)] under per-step E5M2 rounding (4.5 ties to even
         4.0) while single-rounding produces [6.0 (0x46)], and untouched [0x7c]/[0x7d] stay
         intact. *)
      let run_fp8 ~name ~raw ~out ~initial ?(sched = []) code =
        let seed =
          Bigarray.Array1.of_array Bigarray.Char Bigarray.C_layout
            (Array.map initial ~f:Char.of_int_exn)
          |> Bigarray.genarray_of_array1
        in
        let opt = optimize_scoped ~name ~raw code in
        let opt = if List.is_empty sched then opt else Sched.apply sched opt in
        let ctx, routine = link ~name opt in
        let ctx = Context.from_host ctx out (Nd.Fp8_nd seed) in
        let ctx = Context.run ctx routine in
        let result = Context.to_host ctx out in
        Context.release ctx;
        match result with
        | Nd.Fp8_nd bytes ->
            let bytes = Bigarray.array1_of_genarray bytes in
            Array.init (Array.length initial) ~f:(fun j -> Char.to_int bytes.{j})
        | _ -> failwith "expected FP8 storage"
      in
      let d = sym () and i = sym () in
      let out = node_factory ~prec:Ir.Ops.fp8 ~first_id:24795 ~dims:[| 3 |] () "ncgr_fp8_wide" in
      materialize out;
      let cell = iter i in
      (* Guard [d < 4 * i - 4]: cells 0 and 1 receive zero updates; cell 2 receives four updates of
         0.5 onto 4.0 (0x44). *)
      let guard = cmp Ir.Ops.Cmplt (embed d) (embed_idx (aff [ (4, i) ] (-4))) in
      let update = set_at out cell (add_own (get out [| cell |]) (c 0.5)) in
      let source = loop_n d 4 (loop_n i 3 (if_ guard update)) in
      let sunk = LL.sink_narrow_reduction_loops source in
      let initial = [| 0x7c; 0x7d; 0x44 |] in
      let per_step = run_fp8 ~name:"ncgr_fp8_wide_ref" ~raw:source ~out ~initial source in
      let wide = run_fp8 ~name:"ncgr_fp8_wide_sunk" ~raw:source ~out ~initial sunk in
      let unrolled =
        run_fp8 ~name:"ncgr_fp8_wide_unroll" ~raw:source ~out ~initial
          ~sched:[ Sched.Unroll { axis = d; materialize = true } ]
          sunk
      in
      let s0 = sym () and s1 = sym () in
      let partitioned =
        run_fp8 ~name:"ncgr_fp8_wide_partition" ~raw:source ~out ~initial
          ~sched:[ Sched.Partition { axis = d; breakpoints = [ 2 ]; segment_indices = [ s0; s1 ] } ]
          sunk
      in
      let want_wide = [| 0x7c; 0x7d; 0x46 |] in
      p "guarded FP8 per-step rounding loses 4*0.5 onto 4.0 while preserving untouched bytes"
        (Array.equal Int.equal per_step [| 0x7c; 0x7d; 0x44 |]);
      p_all2 "guarded FP8 sunk accumulation rounds once to 6.0 and preserves untouched bytes" wide
        want_wide ~f:Int.equal;
      p_all2 "guarded FP8 materialized unroll rounds once to 6.0 and preserves untouched bytes"
        unrolled want_wide ~f:Int.equal;
      p_all2 "guarded FP8 partition rounds once to 6.0 and preserves untouched bytes" partitioned
        want_wide ~f:Int.equal)

let () =
  let open Ll_test in
  let i = sym () and d = sym () and j = sym () in
  let out = node_factory ~prec:bf16 ~first_id:24234 ~dims:[| 200001 |] () "ncgr_residues" in
  let cell = aff [ (100000, i); (1, j) ] 0 in
  let update = set_at out cell (add_own (get out [| cell |]) (embed d)) in
  let raw = loop_n i 2 (loop_n d 2 (loop_n j 100001 update)) in
  let rewritten = LL.sink_narrow_reduction_loops raw in
  p "a tied traversal chooses the frame with one residue nest"
    (List.length (nest_paths rewritten) = 1
    &&
    match LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:out rewritten with
    | [ site ] -> site.LL.sa_widens
    | _ -> false)

let () =
  let open Ll_test in
  let i = sym () and d = sym () and j = sym () in
  let out = node_factory ~prec:bf16 ~first_id:24230 ~dims:[| 24 |] () "ncgr_order" in
  let terms = node_factory ~first_id:24231 ~dims:[| 4; 2; 7 |] () "ncgr_terms" in
  let cell = aff [ (2, i); (3, j) ] (-1) in
  let body =
    set_at out cell (add_own (get out [| cell |]) (get terms [| iter i; iter d; iter j |]))
  in
  let guard = cmp Ir.Ops.Cmple (embed_idx (fixed 0)) (embed_idx cell) in
  let source = loop_n i 4 (loop_n d 2 (loop_n j 7 (if_ guard body))) in
  let rewritten = LL.sink_narrow_reduction_loops source in
  let contribution i d j =
    match (i, d, j) with
    | 0, 0, 2 -> 0x1p54
    | 0, 1, 2 -> 1.
    | 3, 0, 0 -> -0x1p54
    | 3, 1, 0 -> 3.
    | _ -> Float.of_int (1 + (14 * i) + (7 * d) + j)
  in
  let values =
    List.init 4 ~f:(fun i -> List.init 2 ~f:(fun d -> List.init 7 ~f:(contribution i d)))
    |> List.concat |> List.concat |> Array.of_list
  in
  let evaluate llc =
    let memory = interp_memory [ (out, Array.create ~len:24 0.); (terms, values) ] in
    interpret memory llc;
    Hashtbl.find_exn memory out
  in
  let before = evaluate source in
  (* At cell 5, the original [i,d,j] order loses the first small term. Interchanging [i,d] would
     make it survive, and starting with [i=3] would round the small terms differently. The negative
     offset also makes the original output-access guard live. *)
  p "the order witness loses the first small term at the overlapping cell"
    (Float.equal before.(5) 3.);
  p_all "the guarded nonunit-stride pair has a wide reduction in every residue"
    (LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:out rewritten)
    ~f:(fun s -> s.LL.sa_widens);
  p_all2 "the affine rewrite preserves the source contribution order and guarded cells"
    (evaluate rewritten) before ~f:Float.equal

(* gh-ocannl-2464: hand-built 2-D order witness with crosswise interleaving [i1, d1, i2, d2, j2,
   j1], negative offsets/guards on both axes, and asymmetric extents so axis 0 chooses the
   ascending-q frame ([descending = false], 3 residues) while axis 1 chooses the descending-q frame
   ([descending = true], 2 residues) — 6 residue nests in one rewrite. *)
let () =
  let open Ll_test in
  let i1 = sym () and d1 = sym () and i2 = sym () and d2 = sym () and j2 = sym () and j1 = sym () in
  let out = node_factory ~prec:bf16 ~first_id:24640 ~dims:[| 24; 24 |] () "ncgr_order2d" in
  let terms = node_factory ~first_id:24641 ~dims:[| 4; 2; 7; 2; 4; 7 |] () "ncgr_terms2d" in
  let cell0 = aff [ (2, i1); (3, j1) ] (-1) and cell1 = aff [ (2, i2); (3, j2) ] (-1) in
  let body =
    set out [| cell0; cell1 |]
      (add_own
         (get out [| cell0; cell1 |])
         (get terms [| iter i1; iter d1; iter i2; iter d2; iter j2; iter j1 |]))
  in
  let guard0 = cmp Ir.Ops.Cmple (embed_idx (fixed 0)) (embed_idx cell0) in
  let guard1 = cmp Ir.Ops.Cmple (embed_idx (fixed 0)) (embed_idx cell1) in
  let guarded = if_ guard0 (if_ guard1 body) in
  let source =
    loop_n i1 4 (loop_n d1 2 (loop_n i2 7 (loop_n d2 2 (loop_n j2 4 (loop_n j1 7 guarded)))))
  in
  let swapped_early =
    loop_n i2 7 (loop_n i1 4 (loop_n d1 2 (loop_n d2 2 (loop_n j2 4 (loop_n j1 7 guarded)))))
  in
  let misplaced_d1 =
    loop_n i1 4 (loop_n i2 7 (loop_n d1 2 (loop_n d2 2 (loop_n j2 4 (loop_n j1 7 guarded)))))
  in
  let rewritten = LL.sink_narrow_reduction_loops source in
  let contribution i1 d1 i2 d2 j2 j1 =
    let c0 = (2 * i1) + (3 * j1) - 1 and c1 = (2 * i2) + (3 * j2) - 1 in
    if c0 = 5 && c1 = 5 then
      match (i1, d1, i2, d2) with
      | 0, 0, 0, 0 -> 0x1p54
      | 0, 0, 0, 1 -> 1.
      | 0, 0, 3, 0 -> -0x1p54
      | 0, 0, 3, 1 -> 3.
      | 0, 1, 0, 0 -> 1.
      | 3, 0, 0, 0 -> 1.
      | _ -> 0.
    else Float.of_int (1 + (((((((((i1 * 2) + d1) * 7) + i2) * 2) + d2) * 4) + j2 + j1) % 13))
  in
  let values =
    List.init 4 ~f:(fun i1 ->
        List.init 2 ~f:(fun d1 ->
            List.init 7 ~f:(fun i2 ->
                List.init 2 ~f:(fun d2 ->
                    List.init 4 ~f:(fun j2 -> List.init 7 ~f:(contribution i1 d1 i2 d2 j2))))))
    |> List.concat |> List.concat |> List.concat |> List.concat |> List.concat |> Array.of_list
  in
  let evaluate llc =
    let memory = interp_memory [ (out, Array.create ~len:(24 * 24) 0.); (terms, values) ] in
    interpret memory llc;
    Hashtbl.find_exn memory out
  in
  let witness_idx = (5 * 24) + 5 in
  let before = evaluate source in
  let widths = LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:out rewritten in
  p "the 2-D order witness evaluates to 5 at the doubly overlapping cell"
    (Float.equal before.(witness_idx) 5.);
  p
    "swapping the two early loops or misplacing the interleaved channel loop changes the 2-D \
     witness"
    ((not (Float.equal (evaluate swapped_early).(witness_idx) before.(witness_idx)))
    && not (Float.equal (evaluate misplaced_d1).(witness_idx) before.(witness_idx)));
  p "the mixed-frame 2-D rewrite emits all 6 residue nests" (List.length widths = 6);
  p_all "every residue nest of the guarded 2-D nonunit-stride pair widens" widths ~f:(fun s ->
      s.LL.sa_widens);
  p_all2 "the 2-D affine rewrite preserves the source contribution order and guarded cells"
    (evaluate rewritten) before ~f:Float.equal

(* gh-ocannl-2464: decline controls paired with passing boundary twins for the remaining allowlist
   boundaries — coupled affine pairs sharing an iterator across axes, a residue-tuple product above
   the 64-nest cap, and a single-axis shorter-scan frame exceeding 64 residues. *)
let () =
  let open Ll_test in
  let equal_paths = List.equal (List.equal Ir.Indexing.equal_symbol) in
  let i = sym () and d = sym () and j = sym () and t = sym () in
  let coupled_out = node_factory ~prec:bf16 ~first_id:24645 ~dims:[| 12; 12 |] () "ncgr_coupled" in
  let c0 = aff [ (1, i); (1, j) ] 0 and c1 = aff [ (1, i); (1, t) ] 0 in
  let coupled_body =
    set coupled_out [| c0; c1 |] (add_own (get coupled_out [| c0; c1 |]) (embed d))
  in
  let coupled_src = loop_n i 4 (loop_n d 2 (loop_n j 4 (loop_n t 4 coupled_body))) in
  let coupled_rw = LL.sink_narrow_reduction_loops coupled_src in
  p "coupled affine pairs sharing an iterator across axes decline"
    (equal_paths (nest_paths coupled_rw) (nest_paths coupled_src)
    &&
    match LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:coupled_out coupled_rw with
    | [ s ] -> not s.LL.sa_widens
    | _ -> false);
  let ui = sym () and uk = sym () and ud = sym () and uj = sym () and ut = sym () in
  let uncoupled_out =
    node_factory ~prec:bf16 ~first_id:24648 ~dims:[| 12; 12 |] () "ncgr_uncoupled"
  in
  let u0 = aff [ (1, ui); (1, uj) ] 0 and u1 = aff [ (1, uk); (1, ut) ] 0 in
  let uncoupled_body =
    set uncoupled_out [| u0; u1 |] (add_own (get uncoupled_out [| u0; u1 |]) (embed ud))
  in
  let uncoupled_src =
    loop_n ui 4 (loop_n uk 4 (loop_n ud 2 (loop_n uj 4 (loop_n ut 4 uncoupled_body))))
  in
  let uncoupled_rw = LL.sink_narrow_reduction_loops uncoupled_src in
  p "an uncoupled 2-D twin of the same shape widens"
    (match LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:uncoupled_out uncoupled_rw with
    | [ s ] -> s.LL.sa_widens
    | _ -> false);
  let i1 = sym () and j1 = sym () and i2 = sym () and j2 = sym () and c = sym () in
  let over_out = node_factory ~prec:bf16 ~first_id:24646 ~dims:[| 200; 200 |] () "ncgr_over_cap" in
  let o0 = aff [ (9, i1); (10, j1) ] 0 and o1 = aff [ (9, i2); (10, j2) ] 0 in
  let over_body = set over_out [| o0; o1 |] (add_own (get over_out [| o0; o1 |]) (embed c)) in
  let over_src = loop_n i1 11 (loop_n i2 11 (loop_n c 2 (loop_n j1 10 (loop_n j2 10 over_body)))) in
  let over_rw = LL.sink_narrow_reduction_loops over_src in
  p "independent pairs whose residue product exceeds 64 decline"
    (equal_paths (nest_paths over_rw) (nest_paths over_src));
  let bi1 = sym () and bj1 = sym () and bi2 = sym () and bj2 = sym () and bc = sym () in
  let cap_out = node_factory ~prec:bf16 ~first_id:24649 ~dims:[| 200; 200 |] () "ncgr_at_cap" in
  let b0 = aff [ (8, bi1); (9, bj1) ] 0 and b1 = aff [ (8, bi2); (9, bj2) ] 0 in
  let cap_body = set cap_out [| b0; b1 |] (add_own (get cap_out [| b0; b1 |]) (embed bc)) in
  let cap_src =
    loop_n bi1 10 (loop_n bi2 10 (loop_n bc 2 (loop_n bj1 9 (loop_n bj2 9 cap_body))))
  in
  let cap_rw = LL.sink_narrow_reduction_loops cap_src in
  let cap_widths = LL.serial_accum_widths ~serialized:(fun _ _ -> false) ~tn:cap_out cap_rw in
  p "independent pairs at the 64-residue boundary emit 64 residue nests"
    (List.length cap_widths = 64);
  p_all "every residue nest at the 64-residue boundary widens" cap_widths ~f:(fun s ->
      s.LL.sa_widens);
  let p1 = sym () and k1 = sym () and ch = sym () in
  let amp_out = node_factory ~prec:bf16 ~first_id:24647 ~dims:[| 7000 |] () "ncgr_amp_cap" in
  let a0 = aff [ (65, p1); (1, k1) ] 0 in
  let amp_body = set_at amp_out a0 (add_own (get amp_out [| a0 |]) (embed ch)) in
  let amp_src = loop_n p1 100 (loop_n ch 2 (loop_n k1 66 amp_body)) in
  let amp_rw = LL.sink_narrow_reduction_loops amp_src in
  p "a 1-D pair whose shorter-scan frame exceeds 64 declines instead of amplifying the scan"
    (equal_paths (nest_paths amp_rw) (nest_paths amp_src))
