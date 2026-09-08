(* Warp-shuffle rendering of [Workgroup_reduce] accumulation loops (gh-ocannl-462, llm.c's
   warpReduce/blockReduce idiom): executed parity of shuffle-rendered reductions against
   OCaml-computed references, plus structural checks on the generated source.

   Unlike test/operations/hardware_workgroup_reduce.ml, which stages the tree reduction explicitly
   (shared tile + barriers), here the [?lowered_transform] replaces the lowered serial reduction
   with a single [Workgroup_reduce]-typed loop whose body stays the plain accumulation statement
   [acc = op(acc, contrib(i))] — the renderer owns the communication. On GPU backends (Metal
   locally, CUDA in CI) that renders as the two-phase warp-shuffle pattern ([ocannl_shfl_xor] tree
   within each warp, one shared slot per warp, barrier, first-warp combine); on the C backends the
   same body legally renders as the ordinary serial loop, so every printed boolean holds on every
   backend.

   Covered: a 4-warp sum (two-phase, with the shared per-warp staging), a single-warp FMA
   dot-product (pure shuffles, no staging), a 2-warp max-reduce (non-Add combine), and the clean
   rejection of a recognized accumulation whose extent does not cover whole warps (GPU) vs. its
   serial execution (CPU).

   Narrow accumulators (gh-ocannl-682) are covered at the end: the shuffle stages the value at the
   backend's accumulator RESIDENCY ([C_syntax_config.accum_prec], gh-ocannl-663) rather than at the
   node's storage precision, so a bf16 reduction on a backend that widens bf16 shuffles f32 and
   narrows once into the cell — the same width its serial rendering accumulates at. Where the
   residency stays narrow (bf16 on HIP and Metal; f16 under the default [fp16_arithmetic] policy)
   there is nothing wider to shuffle and the rendering keeps refusing loudly.

   f16's residency is a POLICY question (gh-ocannl-680), so it gets both legs: the refusal under the
   default policy, and — at the very end of this file, under [Numerics.Fp16_wide] — the twin of the
   bf16 legs, where every backend resolves f16 accumulators to f32 and the shuffle carries float. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Ir.Schedule.backend_is_gpu backend_name
let on_cpu = Ir.Schedule.backend_is_cpu backend_name
let codegen_capabilities = Context.codegen_capabilities (Context.auto ())
let single = Ir.Ops.single
let bf16 = Ir.Ops.bfloat16
let half = Ir.Ops.half
let skipped = Verdict.skipped ~backend:backend_name

type rival_values = { once_narrowed : float; storage_tree : float; per_step : float }
type rival_fixture = { n : int; term : int -> float; narrow : float -> float }

let values { once_narrowed; storage_tree; per_step } = [ once_narrowed; storage_tree; per_step ]
let warp_size = 32

(* Host image of the renderer's descending [shfl_xor] offsets. Only the lower half needs updating:
   those are exactly the lanes that can feed lane 0 at the next offset. [narrow] models the
   plausible-wrong spelling whose staging register lives at storage precision. *)
let reduce_storage_tree ~narrow terms =
  let lanes = Array.of_list terms in
  let offset = ref (Array.length lanes / 2) in
  while !offset > 0 do
    for lane = 0 to !offset - 1 do
      lanes.(lane) <- narrow (lanes.(lane) +. lanes.(lane + !offset))
    done;
    offset := !offset / 2
  done;
  lanes.(0)

let render_rivals { n; term; narrow } =
  if n % warp_size <> 0 || not (Int.is_pow2 (n / warp_size)) then
    invalid_arg "warp-shuffle rival fixture must contain a power-of-two number of whole warps";
  let terms = List.init n ~f:term in
  let once_narrowed = narrow (List.fold terms ~init:0.0 ~f:( +. )) in
  let per_step = List.fold terms ~init:0.0 ~f:(fun acc x -> narrow (acc +. x)) in
  let partials =
    List.chunks_of terms ~length:warp_size |> List.map ~f:(reduce_storage_tree ~narrow)
  in
  let storage_tree = reduce_storage_tree ~narrow partials in
  { once_narrowed; storage_tree; per_step }

(* Query the C-syntax accumulator policy itself (gh-ocannl-822). HIP and Metal keep bf16 residency,
   which for the shuffle means a loud refusal because no bf16 shuffle overload is advertised. *)
let widens_bf16 =
  not
    (Ir.Ops.equal_prec
       (codegen_capabilities.Ir.Backend_intf.accum_prec Ir.Ops.bfloat16)
       Ir.Ops.bfloat16)

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Replace the lowered serial reduction with a single [Workgroup_reduce] accumulation loop over a
   fresh index; the renderer owns the communication. The transform drops the lowered [Zero_out] of
   the accumulator, so un-mark [zero_initialized_by_code]: allocation then zeroes the buffer, giving
   the accumulation the same all-zeros starting point as the serial lowering. *)
let reduce_transform ~n ~body_of (s : Tn.t) (opt : LL.optimized) : LL.optimized =
  (LL.get_node opt.traced_store s).LL.zero_initialized_by_code <- false;
  let i = Idx.get_symbol () in
  {
    opt with
    llc =
      LL.For_loop { index = i; from_ = 0; to_ = n - 1; body = body_of i; axis = Workgroup_reduce };
  }

let run ~name ~transform t =
  let comp = named name (Train.forward t) in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx t.Tensor.value).(0)

let it i = Idx.Iterator i
let f0 = Idx.Fixed_idx 0

let () =
  (* --- Two-phase sum: 128 = 4 warps of 32. --- *)
  let n = 128 in
  let vv = Array.init n ~f:(fun k -> (Float.of_int (k % 21) *. 0.25) -. 2.) in
  let expected_sum = Array.fold vv ~init:0. ~f:( +. ) in
  let v = TDSL.ndarray vv ~label:[ "v" ] ~output_dims:[ n ] () in
  let%op s0 = v ++ "i=>0" in
  let got_serial = run ~name:"wshfl_sum_serial" ~transform:(fun opt -> opt) s0 in
  p "serial sum correct" (approx got_serial expected_sum);
  let%op s1 = v ++ "i=>0" in
  let got =
    run ~name:"sum_wshfl"
      ~transform:
        (reduce_transform ~n s1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = s1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Add,
                       (Get (s1.Tensor.value, [| f0 |]), single),
                       (Get (v.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      s1
  in
  p "warp-shuffle sum parity" (approx got expected_sum);
  (let src = Generated.read "sum_wshfl" in
   let has sub = String.is_substring src ~substring:sub in
   let ok =
     if on_gpu then has "ocannl_shfl_xor" && has "wred_partials_"
     else (not (has "ocannl_shfl_xor")) && not (has "wred_partials_")
   in
   p "two-phase shuffle rendering (GPU) or serial fallback (CPU)" ok);

  (* --- Single-warp FMA dot-product: 32 = 1 warp (no staging, no barrier). --- *)
  let m = 32 in
  let av = Array.init m ~f:(fun k -> (Float.of_int (k % 7) *. 0.5) -. 1.) in
  let bv = Array.init m ~f:(fun k -> Float.of_int (k % 5) -. 2.) in
  let expected_dot = Array.fold2_exn av bv ~init:0. ~f:(fun acc a b -> acc +. (a *. b)) in
  let va = TDSL.ndarray av ~label:[ "va" ] ~output_dims:[ m ] () in
  let vb = TDSL.ndarray bv ~label:[ "vb" ] ~output_dims:[ m ] () in
  let%op d1 = va +* "i;i=>0" vb in
  let got_dot =
    run ~name:"dot_wshfl"
      ~transform:
        (reduce_transform ~n:m d1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = d1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Ternop
                     ( Ir.Ops.FMA,
                       (Get (va.Tensor.value, [| it i |]), single),
                       (Get (vb.Tensor.value, [| it i |]), single),
                       (Get (d1.Tensor.value, [| f0 |]), single) );
                 debug = "";
               }))
      d1
  in
  p "warp-shuffle fma dot parity" (approx got_dot expected_dot);
  (let src = Generated.read "dot_wshfl" in
   let has sub = String.is_substring src ~substring:sub in
   let ok =
     if on_gpu then has "ocannl_shfl_xor" && not (has "wred_partials_")
     else not (has "ocannl_shfl_xor")
   in
   p "single-warp shuffle rendering (GPU) or serial fallback (CPU)" ok);

  (* --- 2-warp max-reduce: a non-Add combine. The max is positive, so the allocation-zeroed
     accumulator start does not affect the result. --- *)
  let q = 64 in
  let wv = Array.init q ~f:(fun k -> Float.of_int (k * 13 % 29) -. 5.) in
  let expected_max = Array.fold wv ~init:Float.neg_infinity ~f:Float.max in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ q ] () in
  let%op x1 = w @^^ "i=>0" in
  let got_max =
    run ~name:"max_wshfl"
      ~transform:
        (reduce_transform ~n:q x1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = x1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Max,
                       (Get (x1.Tensor.value, [| f0 |]), single),
                       (Get (w.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      x1
  in
  p "warp-shuffle max parity" (approx got_max expected_max);

  (* --- A recognized accumulation whose extent (48) does not cover whole warps: the GPU renderer
     must reject it cleanly (binding the index would race); the C backends run it serially. --- *)
  let r = 48 in
  let uv = Array.init r ~f:(fun k -> Float.of_int (k % 11) *. 0.125) in
  let expected_u = Array.fold uv ~init:0. ~f:( +. ) in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ r ] () in
  let%op y1 = u ++ "i=>0" in
  let transform =
    reduce_transform ~n:r y1.Tensor.value ~body_of:(fun i ->
        LL.Set
          {
            tn = y1.Tensor.value;
            idcs = [| f0 |];
            llsc =
              Binop
                ( Ir.Ops.Add,
                  (Get (y1.Tensor.value, [| f0 |]), single),
                  (Get (u.Tensor.value, [| it i |]), single) );
            debug = "";
          })
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"odd_extent_wshfl" ~transform y1 : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)"
          (String.is_substring msg ~substring:"multiple of the warp size")
    | None -> p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)" false
  else
    p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)"
      (approx (run ~name:"odd_extent_wshfl" ~transform y1) expected_u);

  (* --- A recognized accumulation sharing workgroup slot 0 with a LARGER sibling extent: on GPU
     [guard_annotated_extents] wraps the reduce body in the synthetic [If (i < 64)] launch guard,
     and the renderer must still see through it and reject (a plain binding would race the
     accumulator; PR #119 review). The sibling nest is a benign self-copy of the input, so on the C
     backends the whole kernel runs serially with a partial (first-64) sum. --- *)
  let t = 128 in
  let tv = Array.init t ~f:(fun k -> (Float.of_int (k % 17) *. 0.5) -. 3.) in
  let expected_partial = Array.fold (Array.sub tv ~pos:0 ~len:64) ~init:0. ~f:( +. ) in
  let tt = TDSL.ndarray tv ~label:[ "tt" ] ~output_dims:[ t ] () in
  let%op z1 = tt ++ "i=>0" in
  let sibling_transform (opt : LL.optimized) : LL.optimized =
    (LL.get_node opt.traced_store z1.Tensor.value).LL.zero_initialized_by_code <- false;
    let j = Idx.get_symbol () in
    let i = Idx.get_symbol () in
    let copy_nest =
      LL.For_loop
        {
          index = j;
          from_ = 0;
          to_ = t - 1;
          axis = Workgroup;
          body =
            LL.Set
              {
                tn = tt.Tensor.value;
                idcs = [| it j |];
                llsc = Get (tt.Tensor.value, [| it j |]);
                debug = "";
              };
        }
    in
    let reduce_nest =
      LL.For_loop
        {
          index = i;
          from_ = 0;
          to_ = 63;
          axis = Workgroup_reduce;
          body =
            LL.Set
              {
                tn = z1.Tensor.value;
                idcs = [| f0 |];
                llsc =
                  Binop
                    ( Ir.Ops.Add,
                      (Get (z1.Tensor.value, [| f0 |]), single),
                      (Get (tt.Tensor.value, [| it i |]), single) );
                debug = "";
              };
        }
    in
    { opt with llc = LL.Seq (copy_nest, reduce_nest) }
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"guarded_extent_wshfl" ~transform:sibling_transform z1 : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)"
          (String.is_substring msg ~substring:"cover the whole workgroup")
    | None -> p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)" false
  else
    p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)"
      (approx (run ~name:"guarded_extent_wshfl" ~transform:sibling_transform z1) expected_partial)

(* --- gh-ocannl-754: an accumulation the shared width decision recognizes but the shuffle cannot
   render — a Serial level NESTED under the [Workgroup_reduce] one, accumulating into a cell every
   lane shares. Before the decision was shared, the shuffle's own recognizer saw no single statement
   and fell through to the plain hardware binding, under which every lane read-modify-wrote the one
   cell: a race, and a silent one. Now a backend that binds the lane index refuses the level loudly,
   while on the C backends it is a serial nest the localizing peel takes whole. --- *)
let () =
  let n = 32 and inner = 4 in
  let mv = Array.init (n * inner) ~f:(fun k -> (Float.of_int (k % 13) *. 0.25) -. 1.5) in
  let expected = Array.fold mv ~init:0. ~f:( +. ) in
  let mx = TDSL.ndarray mv ~label:[ "mx" ] ~output_dims:[ n; inner ] () in
  let%op ms = mx ++ "ij=>0" in
  let transform =
    reduce_transform ~n ms.Tensor.value ~body_of:(fun i ->
        let k = Idx.get_symbol () in
        LL.For_loop
          {
            index = k;
            from_ = 0;
            to_ = inner - 1;
            axis = LL.Serial;
            body =
              LL.Set
                {
                  tn = ms.Tensor.value;
                  idcs = [| f0 |];
                  llsc =
                    Binop
                      ( Ir.Ops.Add,
                        (Get (ms.Tensor.value, [| f0 |]), single),
                        (Get (mx.Tensor.value, [| it i; it k |]), single) );
                  debug = "";
                };
          })
  in
  let claim =
    "an unguarded accumulation nest under the Workgroup_reduce level (a Serial level inside it) is \
     refused loudly where a lane index is bound (GPU) or localized whole as a serial nest (CPU)"
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"nested_wshfl" ~transform ms : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg -> p claim (String.is_substring msg ~substring:"single accumulation statement")
    | None -> p claim false
  else p claim (approx (run ~name:"nested_wshfl" ~transform ms) expected)

(* --- gh-ocannl-682: narrow accumulators. The shuffle stages the value at the backend's accumulator
   RESIDENCY rather than at the node's storage precision, so a bf16 reduction on a widening backend
   computes the same number its serial rendering does, and a residency that stays narrow is refused
   rather than shuffled at a width no builtin overload covers.

   The terms [1 + (k mod 11)/128] are each exact in bf16 and discriminate all three renderings: over
   32 lanes the exact f32 total 33.2109375 narrows once to 33.25, while a tree staged at bf16 gives
   33 and a per-step read-modify-write of the bf16 cell gives 32.75. Every f32 partial sum here is a
   multiple of 1/128 below 2^15, so the tree's reassociation costs nothing and the claim is bitwise
   rather than approximate. The 128-lane version repeats the pattern — exact total 132.890625,
   narrowed once 133.0, against 132.0 for a bf16 tree and 129.0 for per-step narrowing — and it is
   the one that also stages per-warp partials, so it pins the shared slots' element type too. *)

let bf16_term k = 1.0 +. (Float.of_int (k % 11) /. 128.0)
let narrow_bf16 x = Ir.Ops.bfloat16_to_single (Ir.Ops.single_to_bfloat16 x)
let bf16_1w_fixture = { n = 32; term = bf16_term; narrow = narrow_bf16 }
let bf16_4w_fixture = { n = 128; term = bf16_term; narrow = narrow_bf16 }
let bf16_1w_values = render_rivals bf16_1w_fixture
let bf16_4w_values = render_rivals bf16_4w_fixture

let bf16_sum ~name ({ n; term; _ } : rival_fixture) =
  let x = NTDSL.init ~l:(name ^ "_x") ~prec:bf16 ~o:[ n ] ~f:(fun idcs -> term idcs.(0)) () in
  let%op s = x ++ "i=>0" in
  Tn.update_prec s.Tensor.value bf16;
  run ~name
    ~transform:
      (reduce_transform ~n s.Tensor.value ~body_of:(fun i ->
           LL.Set
             {
               tn = s.Tensor.value;
               idcs = [| f0 |];
               llsc =
                 Binop
                   ( Ir.Ops.Add,
                     (Get (s.Tensor.value, [| f0 |]), bf16),
                     (Get (x.Tensor.value, [| it i |]), bf16) );
               debug = "";
             }))
    s

let claim_bf16_1w =
  "a bf16 single-warp Workgroup_reduce accumulates at the widened residency (32 terms narrow once \
   to 33.25, not the 33 a bf16-staged tree or the 32.75 a per-step narrowing gives)"

let claim_bf16_4w =
  "a bf16 four-warp Workgroup_reduce stages its per-warp partials at the widened residency (128 \
   terms narrow once to 133, not the 132 a bf16-staged tree or the 129 a per-step narrowing gives)"

let claim_bf16_types =
  "the emitted shuffle declares its staging register and its per-warp slots at the residency type, \
   never at bf16 storage"

let claim_narrow_refused =
  "an f16 accumulator whose residency stays narrow, as the default fp16_arithmetic policy resolves \
   it on every GPU backend, is refused by the warp-shuffle rendering (GPU) or runs serially (CPU)"

let () =
  p_pairwise_distinct "the bf16 single-warp rival-rendering values are pairwise distinct"
    (values bf16_1w_values) ~equal:Float.equal ~to_string:Float.to_string;
  p_pairwise_distinct "the bf16 four-warp rival-rendering values are pairwise distinct"
    (values bf16_4w_values) ~equal:Float.equal ~to_string:Float.to_string;
  if widens_bf16 then begin
    p claim_bf16_1w
      (Float.equal (bf16_sum ~name:"bf16_1warp_wshfl" bf16_1w_fixture) bf16_1w_values.once_narrowed);
    p claim_bf16_4w
      (Float.equal (bf16_sum ~name:"bf16_4warp_wshfl" bf16_4w_fixture) bf16_4w_values.once_narrowed)
  end
  else begin
    skipped claim_bf16_1w;
    skipped claim_bf16_4w
  end;
  if on_gpu && widens_bf16 then begin
    let src = Generated.read "bf16_4warp_wshfl" in
    let has sub = String.is_substring src ~substring:sub in
    p claim_bf16_types
      (has "float wred_v_" && has "float wred_partials_"
      && (not (has "__nv_bfloat16 wred_v_"))
      && not (has "__nv_bfloat16 wred_partials_"))
  end
  else skipped claim_bf16_types;
  (* The other half of the gate: under the default [fp16_arithmetic] policy f16 resolves to itself
     on every GPU backend (CUDA's seeded wmma triple accumulates f16 natively, RDNA has genuine f16
     accumulator variants, MSL's [half] is a native scalar), so there is no wider value to shuffle
     and the rendering must keep refusing — loudly, since binding the index like a plain [Workgroup]
     axis would race the accumulator. On the C backends [warp_size = 0] and the loop is simply
     serial, which is its correct meaning. This pins what [Fp16_auto] resolves to TODAY, not a
     contract that it always will (gh-ocannl-680 keeps latitude to resolve wide on hardware where
     wide f16 accumulate is free); the wide policy's twin legs are at the end of this file. *)
  let n = 32 in
  let hv = Array.init n ~f:(fun k -> Float.of_int (k % 5) *. 0.5) in
  let expected = Array.fold hv ~init:0. ~f:( +. ) in
  let hx = NTDSL.init ~l:"wshfl_hx" ~prec:half ~o:[ n ] ~f:(fun idcs -> hv.(idcs.(0))) () in
  let%op hs = hx ++ "i=>0" in
  Tn.update_prec hs.Tensor.value half;
  let transform =
    reduce_transform ~n hs.Tensor.value ~body_of:(fun i ->
        LL.Set
          {
            tn = hs.Tensor.value;
            idcs = [| f0 |];
            llsc =
              Binop
                ( Ir.Ops.Add,
                  (Get (hs.Tensor.value, [| f0 |]), half),
                  (Get (hx.Tensor.value, [| it i |]), half) );
            debug = "";
          })
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"f16_wshfl" ~transform hs : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p claim_narrow_refused (String.is_substring msg ~substring:"accumulator residency")
    | None -> p claim_narrow_refused false
  else p claim_narrow_refused (approx (run ~name:"f16_wshfl" ~transform hs) expected)

(* gh-ocannl-682 (Codex review, P1): the widening is sound only where the SERIAL rendering widens
   too, and one class of body it never widens is an RNG-bearing accumulation. An RNG conversion
   picks both its result type and which random bits it consumes from the precision it renders at
   (gh-ocannl-517), so [try_localize_serial_reduce] declines to localize such an update and its
   serial form accumulates in the narrow cell, narrowing on every iteration. Shuffling the same body
   at the residency would accumulate the whole tree wide and narrow once — a change in the
   accumulation WIDTH, not merely its association, which is the one property gh-ocannl-682 exists to
   preserve. So the rendering refuses it wherever the residency is wider than storage.

   At f32/f64 storage the two coincide and nothing is refused, which is why this leg is bf16 and
   runs only where bf16 actually widens. *)
let claim_rng_refused =
  "a bf16 Workgroup_reduce whose contribution mentions an RNG conversion is refused where the \
   residency is wider than storage (GPU) or runs serially (CPU)"

let () =
  let n = 32 in
  (* [Constant_bits] rather than a uint4x32 tensor node: the refusal fires from the contribution's
     SHAPE at codegen, so the leg needs a well-typed RNG conversion, not a live bit source. *)
  let rx = NTDSL.init ~l:"wshfl_rng_x" ~prec:bf16 ~o:[ n ] ~f:(fun idcs -> bf16_term idcs.(0)) () in
  let%op rs = rx ++ "i=>0" in
  Tn.update_prec rs.Tensor.value bf16;
  let transform =
    reduce_transform ~n rs.Tensor.value ~body_of:(fun i ->
        LL.Set
          {
            tn = rs.Tensor.value;
            idcs = [| f0 |];
            llsc =
              Binop
                ( Ir.Ops.Add,
                  (Get (rs.Tensor.value, [| f0 |]), bf16),
                  ( Binop
                      ( Ir.Ops.Mul,
                        (Get (rx.Tensor.value, [| it i |]), bf16),
                        ( Unop
                            ( Ir.Ops.Uint4x32_to_prec_uniform1,
                              (Constant_bits (Int64.of_int 0x9E3779B9), Ir.Ops.uint4x32) ),
                          bf16 ) ),
                    bf16 ) );
            debug = "";
          })
  in
  if widens_bf16 && on_gpu then
    match
      try
        ignore (run ~name:"bf16_rng_wshfl" ~transform rs : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg -> p claim_rng_refused (String.is_substring msg ~substring:"free of RNG conversions")
    | None -> p claim_rng_refused false
  else if on_cpu then
    (* [warp_size = 0] on the C backends: the loop is serial, which is its correct meaning, and the
       refusal has nothing to fire on. Finiteness is all that is claimed — the draw itself is
       gh-ocannl-517's business, not this test's. *)
    p claim_rng_refused (Float.is_finite (run ~name:"bf16_rng_wshfl" ~transform rs))
  else skipped claim_rng_refused

(* --- gh-ocannl-680: the f16 twin of the bf16 legs, under [Numerics.Fp16_wide]. That policy gives
   f16 reduction accumulators f32 residency on EVERY backend, so an f16 [Workgroup_reduce] passes
   the residency gate the leg above pins the refusal of, and shuffles float exactly as bf16 does on
   CUDA. The policy is what changes; the rendering is unchanged, which is the point — the same
   residency staging, the same once-narrowed value, on backends where f16 is the storage precision a
   model actually trains in.

   The terms [1 + (k mod 11)/1024] are the f16 analogue of the bf16 legs' [1 + (k mod 11)/128]:
   1/1024 is ulp(1) at f16's 10 stored mantissa bits as 1/128 is at bf16's 7, so each term is exact
   in f16, while the partial sums are not. The 11-cycle is load-bearing at both widths: at f16's
   finer grid a 7-cycle leaves the four-warp staging indistinguishable from the once-narrowed value;
   at bf16, deriving the actual XOR-tree association exposes the same collision for its old 7-cycle.
   Over 32 lanes the exact f32 total 32.1513671875 narrows once to 32.15625, against 32.125 for a
   tree staged at f16 and 32.09375 for a per-step read-modify-write of the f16 cell; over 128 lanes
   the totals are 128.625 / 128.5 / 128.125, and that case also stages per-warp partials, pinning
   the shared slots' element type. Every f32 partial sum is a multiple of 1/1024 below 2^8, so the
   tree's reassociation is exact and the claims are bitwise rather than approximate. *)

let f16_term k = 1.0 +. (Float.of_int (k % 11) /. 1024.0)
let narrow_f16 x = Ir.Ops.half_to_single (Ir.Ops.single_to_half x)
let f16_1w_fixture = { n = 32; term = f16_term; narrow = narrow_f16 }
let f16_4w_fixture = { n = 128; term = f16_term; narrow = narrow_f16 }
let f16_1w_values = render_rivals f16_1w_fixture
let f16_4w_values = render_rivals f16_4w_fixture

let f16_sum ~name ({ n; term; _ } : rival_fixture) =
  let x = NTDSL.init ~l:(name ^ "_x") ~prec:half ~o:[ n ] ~f:(fun idcs -> term idcs.(0)) () in
  let%op s = x ++ "i=>0" in
  Tn.update_prec s.Tensor.value half;
  run ~name
    ~transform:
      (reduce_transform ~n s.Tensor.value ~body_of:(fun i ->
           LL.Set
             {
               tn = s.Tensor.value;
               idcs = [| f0 |];
               llsc =
                 Binop
                   ( Ir.Ops.Add,
                     (Get (s.Tensor.value, [| f0 |]), half),
                     (Get (x.Tensor.value, [| it i |]), half) );
               debug = "";
             }))
    s

let claim_f16_wide_1w =
  "under Fp16_wide an f16 single-warp Workgroup_reduce accumulates at the widened residency (32 \
   terms narrow once to 32.15625, not the 32.125 an f16-staged tree or the 32.09375 a per-step \
   narrowing gives)"

let claim_f16_wide_4w =
  "under Fp16_wide an f16 four-warp Workgroup_reduce stages its per-warp partials at the widened \
   residency (128 terms narrow once to 128.625, not the 128.5 an f16-staged tree or the 128.125 a \
   per-step narrowing gives)"

let claim_f16_wide_types =
  "under Fp16_wide the emitted f16 shuffle declares its staging register and its per-warp slots at \
   the residency type, never at half storage"

let () =
  let saved = Numerics.get () in
  Exn.protect
    ~finally:(fun () -> Numerics.set_policy saved)
    ~f:(fun () ->
      p_pairwise_distinct "the f16 single-warp rival-rendering values are pairwise distinct"
        (values f16_1w_values) ~equal:Float.equal ~to_string:Float.to_string;
      p_pairwise_distinct "the f16 four-warp rival-rendering values are pairwise distinct"
        (values f16_4w_values) ~equal:Float.equal ~to_string:Float.to_string;
      Numerics.set_policy { saved with fp16_arithmetic = Numerics.Fp16_wide };
      p claim_f16_wide_1w
        (Float.equal
           (f16_sum ~name:"f16_wide_1warp_wshfl" f16_1w_fixture)
           f16_1w_values.once_narrowed);
      p claim_f16_wide_4w
        (Float.equal
           (f16_sum ~name:"f16_wide_4warp_wshfl" f16_4w_fixture)
           f16_4w_values.once_narrowed);
      if on_gpu then begin
        let src = Generated.read "f16_wide_4warp_wshfl" in
        let has sub = String.is_substring src ~substring:sub in
        (* "half wred_v_" also catches CUDA's "__half wred_v_" as a substring. *)
        p claim_f16_wide_types
          (has "float wred_v_" && has "float wred_partials_"
          && (not (has "half wred_v_"))
          && not (has "half wred_partials_"))
      end
      else skipped claim_f16_wide_types)

(* --- gh-ocannl-950: the RACE criterion for a bound lane index. gh-ocannl-754's arm refuses an
   unguarded nest the shuffle cannot render; every other body the peel declined — a sibling
   statement beside the update, a data-dependent guard over it — fell through to the hardware
   binding, which is the correct rendering of an explicitly staged tree
   ([hardware_workgroup_reduce]) and a silent race for these: every lane read-modify-writes the one
   cell [s[0]]. The criterion that tells the two apart is on the CELL, not the nest: a [Set] to
   storage the lanes share whose cell does not mention the lane and whose value reads that cell.
   There is no exemption for a guard pinning the lane — one lane per block, per coordinate of any
   other bound axis, per barrier phase, is not one thread, and the sound single-lane form is a
   per-lane cell with a plain final store (which the staged tree uses and which stays out of the
   criterion). On the C backends [warp_size = 0], the loop is serial and each body has its serial
   value. *)

let claim_sibling_refused =
  "a Workgroup_reduce level holding a sibling statement beside a lane-invariant self-updating Set \
   is refused where a lane index is bound (GPU: the hardware binding would race the cell) or runs \
   serially (CPU)"

let claim_data_guard_refused =
  "a Workgroup_reduce level holding a data-dependent guard over a lane-invariant self-updating Set \
   is refused where a lane index is bound (GPU: the guard leaves several lanes on the cell) or \
   runs serially (CPU)"

let claim_lane_pin_no_exemption =
  "a lane-invariant self-updating Set under a guard pinning the lane index to one value is still \
   refused where a lane index is bound (GPU: every block and every row has that lane), or the \
   serial loop admits the one iteration (CPU)"

let claim_local_scratch_renders =
  "a self-update of a per-thread local array beside a sibling statement is not a race under a \
   bound Workgroup_reduce (one array per lane): it renders, and lane 0's fold reads its own \
   scratch"

let claim_extent_one_refused =
  "a bound Workgroup_reduce of extent one holding the sibling body is refused too (GPU: lane 0 \
   exists per block and per coordinate of any other bound axis), or renders with the one term \
   (CPU)"

let claim_projection_operand_no_read =
  "a store whose value is a projection discarding the cell's old value (Arg2) is not a \
   read-modify-write: every lane stores the same bytes and the level renders"

let claim_cond_read_refused =
  "a scope that tests the cell it writes back in an If condition reads that cell: refused where a \
   lane index is bound (GPU) or the serial alternation ending at 0 (CPU)"

let claim_guard_read_refused =
  "a store under a guard that reads the cell it writes (If (s[0] == 0) s[0] = x[i]) is a \
   test-then-set every lane performs: refused where a lane index is bound (GPU) or the first \
   iteration's value (CPU)"

let claim_dead_loop_renders =
  "a self-update inside a dead inner loop (to_ < from_) beside a sibling performs no accesses: the \
   level renders and the cell keeps its zero"

let claim_staged_form_renders =
  "the sound single-lane form — per-lane cells combined under lane-selecting guards and a plain \
   final store — is not a race: it renders (GPU) or the barrier it needs is rejected (CPU)"

let () =
  let n = 32 in
  let gv = Array.init n ~f:(fun k -> (Float.of_int (k % 9) *. 0.5) -. 2.) in
  let iprec = Ir.Ops.index_prec () in
  let refused ~name ~transform t =
    try
      ignore (run ~name ~transform t : float);
      None
    with Invalid_argument msg -> Some msg
  in
  let update s x i =
    LL.Set
      {
        tn = s;
        idcs = [| f0 |];
        llsc = Binop (Ir.Ops.Add, (Get (s, [| f0 |]), single), (Get (x, [| it i |]), single));
        debug = "";
      }
  in
  let refused_leg claim ~name ~transform t ~cpu_value =
    if on_gpu then
      match refused ~name ~transform t with
      | Some msg -> p claim (String.is_substring msg ~substring:"race the read-modify-write")
      | None -> p claim false
    else if on_cpu then p claim (approx (run ~name ~transform t) cpu_value)
    else skipped claim
  in
  (* Sibling: a per-lane store into a fresh kernel-local array, registered in the traced store the
     way [hardware_workgroup_reduce] registers its tile. Per-lane, so the sibling itself is no race;
     it is what keeps the level from being a single accumulation statement. *)
  let side =
    Tn.create (Tn.Specified single) ~id:999005 ~label:[ "race_side" ]
      ~unpadded_dims:(lazy [| n |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode side Tn.Local 992;
  let with_side opt =
    ignore (LL.get_node opt.LL.traced_store side : LL.traced_array);
    opt
  in
  let sx = TDSL.ndarray gv ~label:[ "race_sx" ] ~output_dims:[ n ] () in
  let%op ss = sx ++ "i=>0" in
  let sibling_transform opt =
    reduce_transform ~n ss.Tensor.value (with_side opt) ~body_of:(fun i ->
        LL.Seq
          ( update ss.Tensor.value sx.Tensor.value i,
            LL.Set
              { tn = side; idcs = [| it i |]; llsc = Get (sx.Tensor.value, [| it i |]); debug = "" }
          ))
  in
  let expected_sum = Array.fold gv ~init:0. ~f:( +. ) in
  refused_leg claim_sibling_refused ~name:"race_sibling_wshfl" ~transform:sibling_transform ss
    ~cpu_value:expected_sum;
  (* Data guard: the update admitted only for positive terms. *)
  let dx = TDSL.ndarray gv ~label:[ "race_dx" ] ~output_dims:[ n ] () in
  let%op ds = dx ++ "i=>0" in
  let data_guard_transform =
    reduce_transform ~n ds.Tensor.value ~body_of:(fun i ->
        LL.If
          {
            cond =
              ( Binop
                  (Ir.Ops.Cmplt, (Constant 0., single), (Get (dx.Tensor.value, [| it i |]), single)),
                single );
            body = update ds.Tensor.value dx.Tensor.value i;
          })
  in
  let expected_positive =
    Array.fold gv ~init:0. ~f:(fun acc x -> if Float.(x > 0.) then acc +. x else acc)
  in
  refused_leg claim_data_guard_refused ~name:"race_data_guard_wshfl" ~transform:data_guard_transform
    ds ~cpu_value:expected_positive;
  (* A pin is no exemption: the same update under [If (i == 0)] is one lane per block and per row,
     not one thread. *)
  let px = TDSL.ndarray gv ~label:[ "race_px" ] ~output_dims:[ n ] () in
  let%op ps = px ++ "i=>0" in
  let pinned_transform =
    reduce_transform ~n ps.Tensor.value ~body_of:(fun i ->
        LL.If
          {
            cond = (Binop (Ir.Ops.Cmpeq, (Embed_index (it i), iprec), (Constant 0., iprec)), iprec);
            body = update ps.Tensor.value px.Tensor.value i;
          })
  in
  refused_leg claim_lane_pin_no_exemption ~name:"race_pinned_wshfl" ~transform:pinned_transform ps
    ~cpu_value:gv.(0);
  (* Per-thread local scratch: a self-update of a kernel-local array beside a sibling is one array
     per lane, so it is not a race. The scratch is written, then self-updated, then folded into [s]
     by lane 0 alone with a plain store: [2 * x[0]] on every backend. *)
  let scratch =
    Tn.create (Tn.Specified single) ~id:999006 ~label:[ "race_scratch" ]
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode scratch Tn.Local 993;
  let lx = TDSL.ndarray gv ~label:[ "race_lx" ] ~output_dims:[ n ] () in
  let%op ls = lx ++ "i=>0" in
  let local_transform opt =
    ignore (LL.get_node opt.LL.traced_store scratch : LL.traced_array);
    reduce_transform ~n ls.Tensor.value opt ~body_of:(fun i ->
        LL.Seq
          ( LL.Set
              {
                tn = scratch;
                idcs = [| f0 |];
                llsc = Get (lx.Tensor.value, [| it i |]);
                debug = "";
              },
            LL.Seq
              ( update scratch lx.Tensor.value i,
                LL.If
                  {
                    cond =
                      ( Binop (Ir.Ops.Cmpeq, (Embed_index (it i), iprec), (Constant 0., iprec)),
                        iprec );
                    body =
                      LL.Set
                        {
                          tn = ls.Tensor.value;
                          idcs = [| f0 |];
                          llsc = Get (scratch, [| f0 |]);
                          debug = "";
                        };
                  } ) ))
  in
  if on_gpu || on_cpu then
    p claim_local_scratch_renders
      (approx (run ~name:"race_local_wshfl" ~transform:local_transform ls) (2. *. gv.(0)))
  else skipped claim_local_scratch_renders;
  (* Extent one: still refused where a lane is bound, since lane 0 exists per block. The full
     operand, not a one-element one: a one-element constant is inlined and would not be a kernel
     parameter for the transformed body to read. *)
  let ox = TDSL.ndarray gv ~label:[ "race_ox" ] ~output_dims:[ n ] () in
  let%op os = ox ++ "i=>0" in
  let one_transform opt =
    reduce_transform ~n:1 os.Tensor.value (with_side opt) ~body_of:(fun i ->
        LL.Seq
          ( update os.Tensor.value ox.Tensor.value i,
            LL.Set
              { tn = side; idcs = [| it i |]; llsc = Get (ox.Tensor.value, [| it i |]); debug = "" }
          ))
  in
  refused_leg claim_extent_one_refused ~name:"race_extent1_wshfl" ~transform:one_transform os
    ~cpu_value:gv.(0);
  (* A projection's discarded operand is no read: [s[0] = Arg2 (s[0], 3)] is a store every lane
     makes with the same bytes, not a read-modify-write. *)
  let px2 = TDSL.ndarray gv ~label:[ "race_px2" ] ~output_dims:[ n ] () in
  let%op ps2 = px2 ++ "i=>0" in
  let projection_transform opt =
    reduce_transform ~n ps2.Tensor.value (with_side opt) ~body_of:(fun i ->
        LL.Seq
          ( LL.Set
              {
                tn = ps2.Tensor.value;
                idcs = [| f0 |];
                llsc =
                  Binop
                    (Ir.Ops.Arg2, (Get (ps2.Tensor.value, [| f0 |]), single), (Constant 3., single));
                debug = "";
              },
            LL.Set
              {
                tn = side;
                idcs = [| it i |];
                llsc = Get (px2.Tensor.value, [| it i |]);
                debug = "";
              } ))
  in
  if on_gpu || on_cpu then
    p claim_projection_operand_no_read
      (approx (run ~name:"race_projection_wshfl" ~transform:projection_transform ps2) 3.)
  else skipped claim_projection_operand_no_read;
  (* A guard's condition reads: the scope writing [s[0]] back tests [s[0]] in its [If], which is the
     read the lanes race on. Serially the cell alternates 0 → 1 → 0 over the [n] iterations and ends
     at 0 for even [n]. *)
  let cx = TDSL.ndarray gv ~label:[ "race_cx" ] ~output_dims:[ n ] () in
  let%op cs = cx ++ "i=>0" in
  let cond_read_transform opt =
    reduce_transform ~n cs.Tensor.value (with_side opt) ~body_of:(fun i ->
        let id = LL.get_scope cs.Tensor.value in
        LL.Seq
          ( LL.Set
              {
                tn = cs.Tensor.value;
                idcs = [| f0 |];
                llsc =
                  Binop
                    ( Ir.Ops.Add,
                      ( Local_scope
                          {
                            id;
                            body =
                              LL.Seq
                                ( LL.Set_local (id, Constant 0.),
                                  LL.If
                                    {
                                      cond =
                                        ( Binop
                                            ( Ir.Ops.Cmpeq,
                                              (Get (cs.Tensor.value, [| f0 |]), single),
                                              (Constant 0., single) ),
                                          single );
                                      body = LL.Set_local (id, Constant 1.);
                                    } );
                            orig_indices = [||];
                            mint = LL.Schedule_minted;
                          },
                        single ),
                      (Constant 0., single) );
                debug = "";
              },
            LL.Set
              { tn = side; idcs = [| it i |]; llsc = Get (cx.Tensor.value, [| it i |]); debug = "" }
          ))
  in
  if on_gpu then
    match refused ~name:"race_condread_wshfl" ~transform:cond_read_transform cs with
    | Some msg ->
        p claim_cond_read_refused (String.is_substring msg ~substring:"race the read-modify-write")
    | None -> p claim_cond_read_refused false
  else if on_cpu then
    p claim_cond_read_refused
      (Float.equal (run ~name:"race_condread_wshfl" ~transform:cond_read_transform cs) 0.)
  else skipped claim_cond_read_refused;
  (* The sound single-lane form, in miniature: a workgroup-shared tile of per-lane cells, each lane
     writing its own cell, a barrier, and lane 0 combining the first two cells with a PLAIN store
     into the device cell. Nothing in it is a lane-invariant self-update, so the criterion is silent
     and the hardware binding renders it; the C backends reject the barrier. *)
  let tile =
    Tn.create (Tn.Specified single) ~id:999007 ~label:[ "race_tile" ]
      ~unpadded_dims:(lazy [| n |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode tile Tn.Local 994;
  let hx = TDSL.ndarray gv ~label:[ "race_hx" ] ~output_dims:[ n ] () in
  let%op hs = hx ++ "i=>0" in
  let staged_transform (opt : LL.optimized) =
    ignore (LL.get_node opt.LL.traced_store tile : LL.traced_array);
    let opt =
      reduce_transform ~n hs.Tensor.value opt ~body_of:(fun i ->
          LL.Seq
            ( LL.Set
                {
                  tn = tile;
                  idcs = [| it i |];
                  llsc = Get (hx.Tensor.value, [| it i |]);
                  debug = "";
                },
              LL.Seq
                ( LL.Workgroup_barrier,
                  LL.If
                    {
                      cond =
                        ( Binop (Ir.Ops.Cmpeq, (Embed_index (it i), iprec), (Constant 0., iprec)),
                          iprec );
                      body =
                        LL.Set
                          {
                            tn = hs.Tensor.value;
                            idcs = [| f0 |];
                            llsc =
                              Binop
                                ( Ir.Ops.Add,
                                  (Get (tile, [| Idx.Fixed_idx 0 |]), single),
                                  (Get (tile, [| Idx.Fixed_idx 1 |]), single) );
                            debug = "";
                          };
                    } ) ))
    in
    { opt with workgroup_shared = Set.add opt.workgroup_shared tile }
  in
  if on_gpu then
    p claim_staged_form_renders
      (approx (run ~name:"race_staged_wshfl" ~transform:staged_transform hs) (gv.(0) +. gv.(1)))
  else if on_cpu then
    p claim_staged_form_renders
      (Option.is_some (refused ~name:"race_staged_wshfl" ~transform:staged_transform hs))
  else skipped claim_staged_form_renders;
  (* A guard's read of the cell a store under it writes: [If (s[0] == 0) s[0] = x[i]] is a
     test-then-set every lane performs (Codex P1 on staging#674). Serially the first iteration sets
     [x[0]] (nonzero here) and no later one fires. *)
  let gx2 = TDSL.ndarray gv ~label:[ "race_gx2" ] ~output_dims:[ n ] () in
  let%op gs2 = gx2 ++ "i=>0" in
  let guard_read_transform =
    reduce_transform ~n gs2.Tensor.value ~body_of:(fun i ->
        LL.If
          {
            cond =
              ( Binop
                  (Ir.Ops.Cmpeq, (Get (gs2.Tensor.value, [| f0 |]), single), (Constant 0., single)),
                single );
            body =
              LL.Set
                {
                  tn = gs2.Tensor.value;
                  idcs = [| f0 |];
                  llsc = Get (gx2.Tensor.value, [| it i |]);
                  debug = "";
                };
          })
  in
  refused_leg claim_guard_read_refused ~name:"race_guardread_wshfl" ~transform:guard_read_transform
    gs2 ~cpu_value:gv.(0);
  (* A dead inner loop performs no accesses: the sibling body with its self-update inside a [to_ <
     from_] loop is not a race, and renders as the sibling store alone (Codex P2 on staging#674). *)
  let zx = TDSL.ndarray gv ~label:[ "race_zx" ] ~output_dims:[ n ] () in
  let%op zs = zx ++ "i=>0" in
  let dead_loop_transform opt =
    reduce_transform ~n zs.Tensor.value (with_side opt) ~body_of:(fun i ->
        let k = Idx.get_symbol () in
        LL.Seq
          ( LL.For_loop
              {
                index = k;
                from_ = 1;
                to_ = 0;
                axis = LL.Serial;
                body = update zs.Tensor.value zx.Tensor.value i;
              },
            LL.Set
              { tn = side; idcs = [| it i |]; llsc = Get (zx.Tensor.value, [| it i |]); debug = "" }
          ))
  in
  if on_gpu || on_cpu then
    p claim_dead_loop_renders
      (approx (run ~name:"race_deadloop_wshfl" ~transform:dead_loop_transform zs) 0.)
  else skipped claim_dead_loop_renders
