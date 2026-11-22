(* gh-ocannl-2423: an overlapping convolution's narrow input gradient rounds once after all windows
   and channels reaching a cell. The materialized f32 twin and the privatized schedule both execute.
   The host controls distinguish rounding per step and rounding per window. A hand-built affine pair
   checks the order with a sum sensitive to window order and an interleaved channel loop. *)

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

let incoming stride =
  Array.init (input_extent stride) ~f:(fun i -> Float.of_int (256 + (2 * (i % 3))))

let round = Ir.Ops.round_to_prec bf16
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

let run_fixture ~stride ~tag ?schedule prec =
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
  let ctx = Context.set_values ctx dx (incoming stride) in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let tiles = Hash_set.create (module Tn) in
  List.iter routine.Context.segments ~f:(fun segment ->
      Ll_test.walk segment.LL.llc ~on_stmt:(function
        | LL.Set { tn; _ } ->
            if List.equal String.equal tn.Tn.label ("acc" :: dx.Tn.label) then Hash_set.add tiles tn
        | _ -> ()));
  (Context.get_values ctx dx, Hash_set.to_list tiles)

let terms ~stride cell =
  List.filter_map (List.range 0 positions) ~f:(fun o ->
      let k = cell - (stride * o) in
      if k < 0 || k >= taps then None
      else Some (List.init channels ~f:(fun d -> weight k d *. cotangent ~stride o d)))

let check stride =
  let label = Printf.sprintf "stride %d" stride in
  let modeled ~step ~window =
    Array.mapi (incoming stride) ~f:(fun cell initial ->
        List.fold (terms ~stride cell) ~init:initial ~f:(fun acc terms ->
            window (List.fold terms ~init:acc ~f:(fun acc term -> step (acc +. term)))))
  in
  let exact = modeled ~step:Fn.id ~window:Fn.id in
  let twin, _ = run_fixture ~stride ~tag:"f32" Ir.Ops.single in
  let narrow, _ = run_fixture ~stride ~tag:"bf16" bf16 in
  let scheduled, tiles = run_fixture ~stride ~tag:"priv" ~schedule:Privatized bf16 in
  let once = Array.map twin ~f:round in
  let discriminates when_ other =
    p_exists
      (Printf.sprintf "%s: rounding %s differs from rounding once per cell" label when_)
      (List.zip_exn (Array.to_list other) (Array.to_list once))
      ~f:(fun (a, b) -> not (same a b))
  in
  pf_all2 "%s: the materialized f32 gradient equals the host sums" label twin exact ~f:same;
  discriminates "per step" (modeled ~step:round ~window:Fn.id);
  if stride < taps then discriminates "per window" (modeled ~step:Fn.id ~window:round);
  pf_all2 "%s: the bf16 gradient equals the f32 gradient rounded once per cell" label narrow once
    ~f:same;
  p_all (Printf.sprintf "%s: the input gradient's accumulator tiles are f32" label) tiles
    ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single);
  pf_all2 "%s: the privatized gradient equals the f32 gradient rounded once per cell" label
    scheduled once ~f:same;
  if stride = 2 then (
    let split, tiles = run_fixture ~stride ~tag:"split" ~schedule:Split_privatized bf16 in
    p "the split residue with ambiguous entry provenance stays a serial accumulation"
      (List.length tiles = 1);
    p_all2 "the split owner's tail guard preserves once-rounded parity across both residues" split
      once ~f:same;
    let unrolled, _ = run_fixture ~stride ~tag:"unrolled" ~schedule:Split_unrolled bf16 in
    p_all2 "materializing the unrolled reduction under the tail guard preserves parity" unrolled
      once ~f:same)

let () =
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.bf16_arithmetic = Ir.Numerics.Bf16_wide })
    (fun () -> List.iter [ 1; 2; 6 ] ~f:check)

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
