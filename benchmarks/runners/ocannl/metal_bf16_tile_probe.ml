(* Can a Metal simdgroup tile accumulate bf16 operands in f32 at the bfloat8x8 accumulator's speed?
   gh-ocannl-2322.

   Since gh-ocannl-2029, Metal's [Bf16_auto] tile runs bfloat fragments into a [simdgroup_float8x8]
   accumulator (the wide arm), and on the M4 Max it ran at the f32 tile's speed, about 3.5x slower
   at n = 2048 than the [simdgroup_bfloat8x8] accumulator (gh-ocannl-2150). This probe times other
   stagings of the same GEMM beside the two, and checks how each one accumulates.

   Scope. What it reads: one hand-written kernel per staging, all sharing the skeleton [mm_mma_pd1]
   renders (one 32-thread simdgroup per threadgroup owning a 32x32 output tile as a 4x4 grid of 8x8
   accumulator fragments, A and B staged through 32x32 threadgroup tiles per 32-deep k step, the
   [ki]/[mi]/[ni] fragment loops and the barriers in their order), compiled with the production
   option state ([Compiler_options.metal], so the math policy is the backend's). The destination
   boundary is a per-staging choice ([boundary]); the [Table] rows reproduce the codegen's converted
   boundary in shape, so [wide-table] is today's [Bf16_auto] tile and [half-table] the f16 wide arm.
   What it deliberately does not vary: the schedule (tile sizes, simdgroups per threadgroup,
   pipelining) and the pooled-buffer prologue; every row is the same schedule with a different
   staging.

   Stagings (the [stagings] list). [wide-table]: today's wide tile. [table-dup]: the same source
   under another name, the session's drift control. [table-pragma]/[table-const]: the table boundary
   with its per-lane element loop unrolled, by pragma or with literal indices (the [Table]'s AIR
   indexes the accumulator's [<64 x float>] storage with the loop's runtime index; these say whether
   that indexing is the cost). [wide-scratch]: the same accumulation with the boundary through a
   threadgroup scratch. [wide]: the same with no boundary (zero fragments, an f32 store). [convert]:
   bfloat fragments converted elementwise into float fragments per k block, then a float x float
   multiply-accumulate. [f32-stage]: bf16 device operands widened to float while staging into
   threadgroup memory, then float fragments throughout. [mul-add]: [simdgroup_multiply] of the
   bfloat fragments into a float temporary, added elementwise into the accumulator. [hybrid]:
   [simdgroup_multiply] into a bfloat temporary (rounded once per 8-deep block), added into a float
   accumulator; it rounds by construction, so it is a speed reference, not a wide candidate.
   [narrow]: the bfloat8x8 accumulator, the speed target. [f32]: the f32 tile, the f32 control.
   [half-wide]/[half-table]/[half-scratch]/[half-narrow]: the f16 rows, to say whether a cost is
   bf16's or the float accumulator's, and what the f16 wide arm pays.

   Accumulation verdicts. Each kernel runs on exact inputs (operands and products exact in the
   narrow type, every f32 partial exact: gh-ocannl-2029's premise), and its output is matched
   against five host models computed exactly: [exact] (equal to the serial f32 accumulation, since
   every partial is exact), [once-narrow] ([exact] rounded once to the narrow type, what a wide
   accumulator stored to a narrow destination holds), [serial-narrow] (per-step rounding to the
   narrow type), [block-narrow] (an 8-deep block summed exactly, then added to the accumulator and
   rounded once per block, gh-ocannl-2029's bfloat8x8 finding) and [block-into-f32] (each block
   rounded to the narrow type, then summed in f32). The probe checks the models are pairwise
   distinct on its data, so a match names one. A staging is WIDE when it matches [exact], or
   [once-narrow] where its destination is narrower than its accumulator: it keeps the wide arm's
   property. Beside it, a third run feeds inexact data (bf16 operands spanning eight binades, k =
   256) and counts outputs bitwise equal to a serial f32 FMA chain in k order and to the [wide]
   kernel (both rounded to a narrow destination where the staging has one): a staging equal to
   [wide] there changes no value at all. Exit status 1 means the probe lost its own discriminating
   power: a [Baseline_narrow] kernel did not match its known model (block-narrow for bfloat8x8,
   serial-narrow for half8x8; the models are distinct, so it then also is not [exact]), or a
   [Baseline_wide] one was not WIDE. The inexact run also compares every bf16 staging with the
   [wide-table] kernel, today's tile, directly.

   Timings: the command buffer's GPU clock, one dispatch per command buffer, every pipeline warmed
   first. Each round visits every staging in a rotated order and takes the median of [--reps]
   dispatches; ratios are paired within a round. Reported per staging: the median time and its range
   over rounds, and the paired ratio to [wide-table] (today) and to [narrow] (median and range).

   Run on a Metal Mac:

   dune exec benchmarks/runners/ocannl/metal_bf16_tile_probe.exe -- [--sizes=1024,2048]
   [--rounds=16] [--reps=3] [--check-only] [--print-source=N]

   [--check-only] runs the accumulation verdicts and no timing. [--print-source=N] prints the MSL
   for an N x N x N GEMM and exits, for [xcrun metal -std=metal3.1 -S -emit-llvm] (the AIR). *)

module Me = Metal

type elt = Bf16 | F32 | F16

let msl = function Bf16 -> "bfloat" | F32 -> "float" | F16 -> "half"
let bytes_of = function Bf16 | F16 -> 2 | F32 -> 4

(* The significand bits a storage type rounds to, for the host models. *)
let sig_bits = function Bf16 -> 8 | F16 -> 11 | F32 -> 24

(* [Baseline_narrow m]: a narrow accumulator whose M4 Max model is known (gh-ocannl-2029); it must
   match [m] exactly, so a kernel that writes garbage or matches no model fails rather than passing
   as "not wide". *)
type role = Baseline_wide | Baseline_narrow of string | Candidate | Reference

(* How the codegen's per-lane element loop is spelled: [Loop] as rendered today, [Unroll_pragma] the
   same loop under [#pragma unroll], [Constant_index] the two lanes' statements written out with
   literal indices. *)
type table_form = Loop | Unroll_pragma | Constant_index

(* How the accumulator meets the destination. [Direct]: zero fragments, [simdgroup_store] into an
   output of the accumulator's own type. [Table]: the codegen's converted boundary, verbatim in
   shape ([Metal_backend.mma_d_boundary_lines], gh-ocannl-837/923): the kernel zeroes its bf16
   destination tile, loads it into the float fragments elementwise through [thread_elements()] and
   the [ocannl_mma_rc8] coordinate table, and stores back the same way. [Scratch]: the same
   destination and conversion, staged through a 64-float threadgroup scratch with
   [simdgroup_load]/[simdgroup_store], so no [thread_elements()] touches the accumulators. *)
type boundary = Direct | Table of table_form | Scratch

type staging = {
  name : string;
  input : elt;  (** device A and B *)
  tile : elt;  (** threadgroup tiles *)
  frag : elt;  (** operand fragments *)
  acc : elt;  (** accumulator fragments *)
  out : elt;  (** the destination buffer *)
  boundary : boundary;
  convert : bool;  (** load [frag] fragments, convert to float fragments *)
  mac : string;  (** the update of [acc[mi][ni]] from [af] and [bf[ni]] *)
  narrow_elt : elt;  (** the storage type the host models round to *)
  role : role;
  note : string;
}

let mac_plain = "simdgroup_multiply_accumulate(acc[mi][ni], af, bf[ni], acc[mi][ni]);"

let mac_mul_add tmp =
  Printf.sprintf
    "{ simdgroup_%s8x8 p; simdgroup_multiply(p, af, bf[ni]); for (int e = 0; e < ELTS; ++e) \
     acc[mi][ni].thread_elements()[e] += (float)p.thread_elements()[e]; }"
    tmp

let stagings =
  let s ?(convert = false) ?(mac = mac_plain) ?(narrow_elt = Bf16) ?(boundary = Direct) name input
      tile frag acc role note =
    let out = if boundary = Direct then acc else narrow_elt in
    { name; input; tile; frag; acc; out; boundary; convert; mac; narrow_elt; role; note }
  in
  [
    s ~boundary:(Table Loop) "wide-table" Bf16 Bf16 Bf16 F32 Baseline_wide
      "today's Bf16_auto tile, with the codegen's table boundary";
    s ~boundary:(Table Loop) "table-dup" Bf16 Bf16 Bf16 F32 Reference
      "the same source again: drift control";
    s ~boundary:(Table Unroll_pragma) "table-pragma" Bf16 Bf16 Bf16 F32 Candidate
      "the table boundary's lane loop under #pragma unroll";
    s ~boundary:(Table Constant_index) "table-const" Bf16 Bf16 Bf16 F32 Candidate
      "the table boundary with literal lane indices";
    s ~boundary:Scratch "wide-scratch" Bf16 Bf16 Bf16 F32 Candidate
      "the same, the boundary through threadgroup scratch";
    s "wide" Bf16 Bf16 Bf16 F32 Baseline_wide "the same, no boundary: zero init, f32 store";
    s ~convert:true "convert" Bf16 Bf16 F32 F32 Candidate
      "bfloat fragments converted to float per k block";
    s "f32-stage" Bf16 F32 F32 F32 Candidate "bf16 widened while staging to threadgroup";
    s ~mac:(mac_mul_add "float") "mul-add" Bf16 Bf16 Bf16 F32 Candidate
      "multiply into a float temporary, then add";
    s ~mac:(mac_mul_add "bfloat") "hybrid" Bf16 Bf16 Bf16 F32 Reference
      "multiply into a bfloat temporary, then add (rounds)";
    s "narrow" Bf16 Bf16 Bf16 Bf16 (Baseline_narrow "block-narrow")
      "bfloat8x8 accumulator: the speed target";
    s "f32" F32 F32 F32 F32 Baseline_wide "the f32 tile: the f32 control";
    s ~narrow_elt:F16 "half-wide" F16 F16 F16 F32 Reference "half fragments, float acc";
    s ~narrow_elt:F16 ~boundary:(Table Loop) "half-table" F16 F16 F16 F32 Reference
      "the f16 wide arm as rendered: table boundary";
    s ~narrow_elt:F16 ~boundary:Scratch "half-scratch" F16 F16 F16 F32 Reference
      "the f16 wide arm, scratch boundary";
    s ~narrow_elt:F16 "half-narrow" F16 F16 F16 F16 (Baseline_narrow "serial-narrow")
      "half8x8 accumulator";
  ]

let kernel_name st = "k_" ^ String.map (fun c -> if c = '-' then '_' else c) st.name

let kernel ~n ~k st =
  let load dst src =
    if st.convert then
      Printf.sprintf
        "{ simdgroup_%s8x8 raw; simdgroup_load(raw, %s, (ulong)32); for (int e = 0; e < ELTS; ++e) \
         %s.thread_elements()[e] = (float)raw.thread_elements()[e]; }"
        (msl st.tile) src dst
    else Printf.sprintf "simdgroup_load(%s, %s, (ulong)32);" dst src
  in
  let frag_at = Printf.sprintf "(dp + mi * 8 * %d + ni * 8)" n in
  let table_lines form dir =
    let elt ei =
      Printf.sprintf "{ const int __rc = (int)__mma_rc.thread_elements()[%s]; %s; }" ei
        (match dir with
        | `Load ->
            Printf.sprintf
              "acc[mi][ni].thread_elements()[%s] = (float)%s[(__rc >> 3) * %d + (__rc & 7)]" ei
              frag_at n
        | `Store ->
            Printf.sprintf
              "%s[(__rc >> 3) * %d + (__rc & 7)] = (%s)acc[mi][ni].thread_elements()[%s]" frag_at n
              (msl st.out) ei)
    in
    let loop pragma =
      Printf.sprintf "%s        for (int __ei = 0; __ei < ELTS; ++__ei) %s" pragma (elt "__ei")
    in
    Printf.sprintf
      {|      { simdgroup_float8x8 __mma_rc;
        simdgroup_load(__mma_rc, ocannl_mma_rc8, (ulong)8);
%s }|}
      (match form with
      | Loop -> loop ""
      | Unroll_pragma -> loop "        #pragma unroll\n"
      | Constant_index ->
          "        static_assert(ELTS == 2, \"two elements per lane\");\n        " ^ elt "0"
          ^ "\n        " ^ elt "1")
  in
  let scratch_lines dir =
    match dir with
    | `Load ->
        Printf.sprintf
          {|      { for (int e = t; e < 64; e += 32) dscr[e] = (float)%s[(e >> 3) * %d + (e & 7)];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(acc[mi][ni], dscr, (ulong)8);
        simdgroup_barrier(mem_flags::mem_threadgroup); }|}
          frag_at n
    | `Store ->
        Printf.sprintf
          {|      { simdgroup_store(acc[mi][ni], dscr, (ulong)8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = t; e < 64; e += 32) %s[(e >> 3) * %d + (e & 7)] = (%s)dscr[e];
        simdgroup_barrier(mem_flags::mem_threadgroup); }|}
          frag_at n (msl st.out)
  in
  let decls =
    match st.boundary with
    | Direct -> ""
    | Table _ ->
        {|  threadgroup float ocannl_mma_rc8[64];
  if (all(lid == uint3(0))) {
    for (int __rc = 0; __rc < 64; ++__rc) ocannl_mma_rc8[__rc] = (float)__rc;
  }
|}
    | Scratch -> "  threadgroup float dscr[64];\n"
  in
  let fragments lines =
    Printf.sprintf "  for (int mi = 0; mi < 4; ++mi)\n    for (int ni = 0; ni < 4; ++ni)\n%s\n"
      lines
  in
  let init =
    match st.boundary with
    | Direct ->
        Printf.sprintf
          "  for (int mi = 0; mi < 4; ++mi)\n\
          \    for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = make_filled_simdgroup_matrix<%s, 8, \
           8>(0);\n\
          \  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n"
          (msl st.acc)
    | Table _ | Scratch ->
        Printf.sprintf
          "  device %s *dp = C + row0 * %d + col0;\n\
          \  for (int r = 0; r < 32; ++r) dp[r * %d + t] = (%s)(0.0);\n\
          \  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n\
           %s"
          (msl st.out) n n (msl st.out)
          (fragments
             ((match st.boundary with Table form -> table_lines form | _ -> scratch_lines) `Load))
  in
  let store =
    match st.boundary with
    | Direct ->
        Printf.sprintf
          "  for (int mi = 0; mi < 4; ++mi)\n\
          \    for (int ni = 0; ni < 4; ++ni)\n\
          \      simdgroup_store(acc[mi][ni], C + (row0 + mi * 8) * %d + col0 + ni * 8, (ulong)%d);\n"
          n n
    | Table _ | Scratch ->
        fragments
          ((match st.boundary with Table form -> table_lines form | _ -> scratch_lines) `Store)
  in
  Printf.sprintf
    {|kernel void %s(
    device const %s* A [[buffer(0)]], device const %s* B [[buffer(1)]], device %s* C [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup %s ta[1024];
  threadgroup %s tb[1024];
%s  const int row0 = 32 * (int)gid.y;
  const int col0 = 32 * (int)gid.x;
  const int t = (int)lid.x;
  simdgroup_%s8x8 acc[4][4];
%s  for (int kb = 0; kb < %d; ++kb) {
    for (int r = 0; r < 32; ++r) tb[r * 32 + t] = (%s)B[(32 * kb + r) * %d + col0 + t];
    for (int r = 0; r < 32; ++r) ta[r * 32 + t] = (%s)A[(row0 + r) * %d + 32 * kb + t];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (int ki = 0; ki < 4; ++ki) {
      simdgroup_%s8x8 bf[4];
      for (int ni = 0; ni < 4; ++ni) %s
      for (int mi = 0; mi < 4; ++mi) {
        simdgroup_%s8x8 af;
        %s
        for (int ni = 0; ni < 4; ++ni) %s
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
%s}
|}
    (kernel_name st) (msl st.input) (msl st.input) (msl st.out) (msl st.tile) (msl st.tile) decls
    (msl st.acc) init (k / 32) (msl st.tile) n (msl st.tile) k (msl st.frag)
    (load "bf[ni]" "tb + ki * 8 * 32 + ni * 8")
    (msl st.frag)
    (load "af" "ta + mi * 8 * 32 + ki * 8")
    st.mac store

(* [ELTS] is each thread's share of a fragment, spelled as the codegen's converted boundary spells
   it; the elementwise stagings rely on the float and narrow fragments sharing one thread layout,
   which the exact-premise verdicts check. *)
let source ~n ~k =
  String.concat "\n"
    ("#include <metal_stdlib>\n\
      #include <metal_simdgroup_matrix>\n\
      using namespace metal;\n\
      #define ELTS ((int)(sizeof(simdgroup_float8x8::storage_type) / sizeof(float) / 32))\n"
    :: List.map (kernel ~n ~k) stagings)

(* -- host numerics -------------------------------------------------------------------------- *)

(* Round to nearest-even at [bits] significant bits; exact for the normal-range values here. *)
let round_sig bits x =
  if x = 0.0 then x
  else
    let fr, e = Float.frexp x in
    let y = Float.ldexp fr bits in
    let lo = Float.of_int (truncate y) in
    let diff = Float.abs (y -. lo) in
    let r =
      if diff = 0.5 then if Float.rem lo 2.0 = 0.0 then lo else lo +. Float.copy_sign 1.0 y
      else Float.round y
    in
    Float.ldexp r (e - bits)

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let bits_of elt x =
  match elt with
  | F32 -> Int32.to_int (Int32.bits_of_float x) land 0xffffffff
  | Bf16 ->
      let b = Int32.to_int (Int32.bits_of_float x) land 0xffffffff in
      if b land 0xffff <> 0 then failwith "bits_of: value not exact in bf16";
      b lsr 16
  | F16 ->
      if x = 0.0 then 0
      else
        let fr, e = Float.frexp (Float.abs x) in
        (* x = fr * 2^e, fr in [0.5, 1): half exponent field (e - 1) + 15, mantissa 10 bits *)
        let ex = e - 1 + 15 in
        if ex < 1 || ex > 30 then failwith "bits_of: value outside half's normal range";
        let mant = Float.ldexp fr 11 -. 1024.0 in
        if not (Float.is_integer mant) then failwith "bits_of: value not exact in half";
        (if x < 0.0 then 0x8000 else 0) lor (ex lsl 10) lor truncate mant

let of_bits elt b =
  match elt with
  | F32 -> Int32.float_of_bits (Int32.of_int b)
  | Bf16 -> Int32.float_of_bits (Int32.of_int (b lsl 16))
  | F16 ->
      let s = if b land 0x8000 <> 0 then -1.0 else 1.0 in
      let ex = (b lsr 10) land 0x1f and mant = b land 0x3ff in
      if ex = 0 then s *. Float.ldexp (Float.of_int mant) (-24)
      else if ex = 31 then if mant = 0 then s *. Float.infinity else Float.nan
      else s *. Float.ldexp (Float.of_int (mant + 1024)) (ex - 25)

let models = [ "exact"; "once-narrow"; "serial-narrow"; "block-narrow"; "block-into-f32" ]

(* Each model's [m x n] output for row-major [a] (m x k) and [b] (k x n). Products of these operands
   are exact in double, and so is every block sum and every exact-premise partial. *)
let model ~narrow ~m ~n ~k a b name =
  let rn = round_sig (sig_bits narrow) in
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let prod kk = a.((i * k) + kk) *. b.((kk * n) + j) in
      let block kb =
        let s = ref 0.0 in
        for kk = 8 * kb to (8 * kb) + 7 do
          s := !s +. prod kk
        done;
        !s
      in
      match name with
      | "exact" | "once-narrow" ->
          let s = ref 0.0 in
          for kk = 0 to k - 1 do
            s := to_f32 (!s +. prod kk)
          done;
          if name = "exact" then !s else rn !s
      | "serial-narrow" ->
          let s = ref 0.0 in
          for kk = 0 to k - 1 do
            s := rn (!s +. prod kk)
          done;
          !s
      | "block-narrow" ->
          let s = ref 0.0 in
          for kb = 0 to (k / 8) - 1 do
            s := rn (!s +. block kb)
          done;
          !s
      | "block-into-f32" ->
          let s = ref 0.0 in
          for kb = 0 to (k / 8) - 1 do
            s := to_f32 (!s +. rn (block kb))
          done;
          !s
      | _ -> assert false)

(* The serial f32 FMA chain in k order, for inexact data: each product of two bf16 values is exact
   in double and every partial sum here fits double's significand, so one rounding per step is a
   correctly rounded f32 FMA. *)
let serial_f32 ~m ~n ~k a b =
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let s = ref 0.0 in
      for kk = 0 to k - 1 do
        s := to_f32 (!s +. (a.((i * k) + kk) *. b.((kk * n) + j)))
      done;
      !s)

(* -- device plumbing ------------------------------------------------------------------------- *)

let ropts = Me.ResourceOptions.(storage_mode_shared + hazard_tracking_mode_untracked)

let compile_options () =
  let options = Me.CompileOptions.init () in
  List.iter
    (function
      | Ir.Compiler_options.Language_version_3_1 ->
          Me.CompileOptions.set_language_version options
            Me.CompileOptions.LanguageVersion.version_3_1
      | Language_version_3_2 ->
          Me.CompileOptions.set_language_version options
            Me.CompileOptions.LanguageVersion.version_3_2
      | Fast_math_enabled enabled -> Me.CompileOptions.set_fast_math_enabled options enabled
      | Math_mode_safe -> Me.CompileOptions.set_math_mode options Me.CompileOptions.MathMode.Safe
      | Math_functions_fast ->
          Me.CompileOptions.set_math_floating_point_functions options
            Me.CompileOptions.MathFloatingPointFunctions.Fast
      | Enable_logging -> Me.CompileOptions.set_enable_logging options true)
    (Ir.Compiler_options.metal ~routine_logging:false ~math_api:(Metal_math_api_runtime.get ()));
  options

type gemm = {
  m : int;
  n : int;
  states : (string, Me.ComputePipelineState.t) Hashtbl.t;
  inputs : (elt, Me.Buffer.t * Me.Buffer.t) Hashtbl.t;
  out : Me.Buffer.t;
}

let fill buf elt values =
  let open Ctypes in
  match elt with
  | F32 ->
      let p = coerce (ptr void) (ptr float) (Me.Buffer.contents buf) in
      Array.iteri (fun i v -> p +@ i <-@ v) values
  | Bf16 | F16 ->
      let p = coerce (ptr void) (ptr uint16_t) (Me.Buffer.contents buf) in
      Array.iteri (fun i v -> p +@ i <-@ Unsigned.UInt16.of_int (bits_of elt v)) values

let read_out g elt =
  let open Ctypes in
  match elt with
  | F32 ->
      let p = coerce (ptr void) (ptr float) (Me.Buffer.contents g.out) in
      Array.init (g.m * g.n) (fun i -> to_f32 !@(p +@ i))
  | Bf16 | F16 ->
      let p = coerce (ptr void) (ptr uint16_t) (Me.Buffer.contents g.out) in
      Array.init (g.m * g.n) (fun i -> of_bits elt (Unsigned.UInt16.to_int !@(p +@ i)))

(* All-ones bytes: a NaN in every element type, so a kernel that leaves an output unwritten matches
   no model. *)
let poison g =
  let open Ctypes in
  let p = coerce (ptr void) (ptr uint8_t) (Me.Buffer.contents g.out) in
  for i = 0 to (4 * g.m * g.n) - 1 do
    p +@ i <-@ Unsigned.UInt8.of_int 0xff
  done

let make_gemm device ~m ~n ~k =
  let library = Me.Library.on_device device ~source:(source ~n ~k) (compile_options ()) in
  let states = Hashtbl.create 16 in
  List.iter
    (fun st ->
      let fn = Me.Library.new_function_with_name library (kernel_name st) in
      Hashtbl.replace states st.name
        (fst (Me.ComputePipelineState.on_device_with_function device fn)))
    stagings;
  let inputs = Hashtbl.create 3 in
  List.iter
    (fun elt ->
      Hashtbl.replace inputs elt
        ( Me.Buffer.on_device device ~length:(bytes_of elt * m * k) ropts,
          Me.Buffer.on_device device ~length:(bytes_of elt * k * n) ropts ))
    [ Bf16; F32; F16 ];
  { m; n; states; inputs; out = Me.Buffer.on_device device ~length:(4 * m * n) ropts }

let set_inputs g ?(elts = [ Bf16; F32; F16 ]) a b =
  List.iter
    (fun elt ->
      let ba, bb = Hashtbl.find g.inputs elt in
      fill ba elt a;
      fill bb elt b)
    elts

let dispatch queue g st =
  let cb = Me.CommandBuffer.on_queue queue in
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find g.states st.name);
  let ba, bb = Hashtbl.find g.inputs st.input in
  Me.ComputeCommandEncoder.set_buffer enc ~index:0 ba;
  Me.ComputeCommandEncoder.set_buffer enc ~index:1 bb;
  Me.ComputeCommandEncoder.set_buffer enc ~index:2 g.out;
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = g.n / 32; height = g.m / 32; depth = 1 }
    ~threads_per_threadgroup:{ width = 32; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  Me.CommandBuffer.commit cb;
  Me.CommandBuffer.wait_until_completed cb;
  (match Me.CommandBuffer.get_error cb with
  | Some err ->
      Printf.eprintf "metal_bf16_tile_probe: %s failed: %s\n" st.name err;
      exit 2
  | None -> ());
  Me.CommandBuffer.get_gpu_end_time cb -. Me.CommandBuffer.get_gpu_start_time cb

(* -- accumulation verdicts ------------------------------------------------------------------- *)

(* Exact-premise operands: integers 1..[top] over 4 and 8, varying with both indices; [signed] mixes
   signs from a third pattern. Per narrow type, [top] keeps every product exact in it (at most 8
   significant bits for bf16 with [top] = 15, at most 10 for half with [top] = 31) while an 8-deep
   block sum still needs more bits than the type has, which keeps the four models distinct. *)
let exact_top = function Bf16 | F32 -> 15 | F16 -> 31

let exact_operands ~top ~signed ~m ~n ~k =
  let sgn x = if signed && x mod 3 = 0 then -1.0 else 1.0 in
  let a =
    Array.init (m * k) (fun idx ->
        let i = idx / k and kk = idx mod k in
        sgn (i + (2 * kk)) *. Float.of_int ((((7 * i) + (3 * kk)) mod top) + 1) /. 4.0)
  in
  let b =
    Array.init (k * n) (fun idx ->
        let kk = idx / n and j = idx mod n in
        sgn ((5 * kk) + j + 1) *. Float.of_int ((((5 * kk) + (11 * j)) mod top) + 1) /. 8.0)
  in
  (a, b)

(* Inexact operands: bf16 values with 8-bit significands over eight binades. *)
let inexact_operands ~m ~n ~k =
  let st = ref 0x2322 in
  let next () =
    st := ((!st * 1103515245) + 12345) land 0x7fffffff;
    !st lsr 8
  in
  let v () =
    let r = next () in
    let mant = 128 + (r land 127) and e = (r lsr 7) land 7 and s = (r lsr 10) land 1 in
    (if s = 1 then -1.0 else 1.0) *. Float.ldexp (Float.of_int mant) (-7 - e)
  in
  (Array.init (m * k) (fun _ -> v ()), Array.init (k * n) (fun _ -> v ()))

let equal_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)

let count_equal xs ys =
  Array.fold_left ( + ) 0 (Array.map2 (fun x y -> if equal_bits x y then 1 else 0) xs ys)

let checks device queue =
  let failures = ref 0 in
  let m = 64 and n = 64 in
  Printf.printf "Accumulation verdicts (m = n = %d; outputs matched bitwise to host models):\n" m;
  List.iter
    (fun (k, signed) ->
      let g = make_gemm device ~m ~n ~k in
      let operands narrow = exact_operands ~top:(exact_top narrow) ~signed ~m ~n ~k in
      let a, b = operands Bf16 in
      set_inputs g ~elts:[ Bf16; F32 ] a b;
      let ah, bh = operands F16 in
      set_inputs g ~elts:[ F16 ] ah bh;
      let preds =
        List.map
          (fun narrow ->
            let a, b = operands narrow in
            (narrow, List.map (fun name -> (name, model ~narrow ~m ~n ~k a b name)) models))
          [ Bf16; F16 ]
      in
      List.iter
        (fun (narrow, ps) ->
          List.iteri
            (fun i (na, pa) ->
              List.iteri
                (fun j (nb, pb) ->
                  if i < j && count_equal pa pb = Array.length pa then (
                    Printf.printf "  models %s and %s coincide (%s): premise lost\n" na nb
                      (msl narrow);
                    incr failures))
                ps)
            ps)
        preds;
      Printf.printf "  exact premise, k = %d, %s operands (models pairwise distinct per type):\n" k
        (if signed then "mixed-sign" else "positive");
      List.iter
        (fun st ->
          poison g;
          ignore (dispatch queue g st : float);
          let got = read_out g st.out in
          let ps = List.assoc st.narrow_elt preds in
          let matched =
            List.filter_map
              (fun (name, p) -> if count_equal got p = Array.length got then Some name else None)
              ps
          in
          (* A wide staging keeps f32 partials: equal to [exact], rounded once more where its
             destination is narrower than its accumulator. *)
          let exact_ok = List.mem (if st.out = st.acc then "exact" else "once-narrow") matched in
          let verdict =
            match matched with
            | [] ->
                let best =
                  List.fold_left
                    (fun (bn, bc) (name, p) ->
                      let c = count_equal got p in
                      if c > bc then (name, c) else (bn, bc))
                    ("-", -1) ps
                in
                Printf.sprintf "no model (closest %s: %d/%d)" (fst best) (snd best)
                  (Array.length got)
            | l -> String.concat "," l
          in
          let bad =
            match st.role with
            | Baseline_wide -> not exact_ok
            | Baseline_narrow expected -> not (List.mem expected matched)
            | Candidate | Reference -> false
          in
          if bad then incr failures;
          Printf.printf "    %-12s %-15s %s%s\n" st.name
            (if exact_ok then "WIDE" else "not wide")
            verdict
            (if bad then "  <- contradicts its role" else ""))
        stagings)
    [ (32, false); (256, true) ];
  (* Inexact data: equality to the serial f32 chain, to the [wide] kernel, and to the [wide-table]
     kernel (today's tile) directly. A staging with a narrow destination is compared with the f32
     references rounded to that destination; the table kernel's output already has one. *)
  let k = 256 in
  let g = make_gemm device ~m ~n ~k in
  let a, b = inexact_operands ~m ~n ~k in
  set_inputs g ~elts:[ Bf16; F32 ] a b;
  let serial = serial_f32 ~m ~n ~k a b in
  let run name =
    let st = List.find (fun st -> st.name = name) stagings in
    poison g;
    ignore (dispatch queue g st : float);
    read_out g st.out
  in
  let wide = run "wide" and table = run "wide-table" in
  Printf.printf "  inexact bf16 operands over 8 binades, k = %d: outputs (of %d) bitwise equal to\n"
    k (m * n);
  Printf.printf "    %-12s %14s %14s %14s\n" "staging" "serial f32" "wide kernel" "today (table)";
  List.iter
    (fun st ->
      if st.input <> F16 then
        let got = run st.name in
        let narrowed xs = if st.out = F32 then xs else Array.map (round_sig (sig_bits st.out)) xs in
        let table_cmp = if st.out = Bf16 then string_of_int (count_equal got table) else "-" in
        Printf.printf "    %-12s %14d %14d %14s\n" st.name
          (count_equal got (narrowed serial))
          (count_equal got (narrowed wide))
          table_cmp)
    stagings;
  !failures

(* -- timings --------------------------------------------------------------------------------- *)

let median a =
  let a = Array.copy a in
  Array.sort Float.compare a;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let range a = (Array.fold_left Float.min Float.infinity a, Array.fold_left Float.max 0.0 a)

let rotate r xs =
  let len = List.length xs in
  let r = r mod len in
  List.filteri (fun i _ -> i >= r) xs @ List.filteri (fun i _ -> i < r) xs

let timings device queue ~sizes ~rounds ~reps =
  List.iter
    (fun size ->
      let g = make_gemm device ~m:size ~n:size ~k:size in
      let a, b = inexact_operands ~m:size ~n:size ~k:size in
      set_inputs g ~elts:[ Bf16; F32 ] a b;
      (* Halved, the same values stay exact in half (operands down to 2^-8, normal; their products
         reach 2^-16, below half's normal range). The half8x8 timing does not depend on this: with
         unhalved operands, every product normal, it was unchanged (gh-ocannl-2322, 6.61 against
         6.56 ms at n = 2048, a non-exclusive run). *)
      set_inputs g ~elts:[ F16 ] (Array.map (fun x -> x /. 2.0) a) (Array.map (fun x -> x /. 2.0) b);
      List.iter
        (fun st ->
          for _ = 1 to 3 do
            ignore (dispatch queue g st : float)
          done)
        stagings;
      let samples = Hashtbl.create 16 in
      List.iter (fun st -> Hashtbl.replace samples st.name (Array.make rounds 0.0)) stagings;
      for r = 0 to rounds - 1 do
        List.iter
          (fun st ->
            let ts = Array.init reps (fun _ -> dispatch queue g st) in
            (Hashtbl.find samples st.name).(r) <- median ts *. 1000.0)
          (rotate r stagings)
      done;
      let get name = Hashtbl.find samples name in
      let ratio name base = Array.map2 ( /. ) (get name) (get base) in
      let gflop = 2.0 *. (Float.of_int size ** 3.0) /. 1e9 in
      Printf.printf
        "\n\
         n = %d: GPU-clock ms, %d rotated rounds x median of %d dispatches; ratios paired per round\n"
        size rounds reps;
      Printf.printf "  %-12s %8s %17s %8s %22s %22s\n" "staging" "median" "range" "GFLOP/s"
        "/ today median (range)" "/ narrow median (range)";
      List.iter
        (fun st ->
          let t = get st.name in
          let lo, hi = range t in
          let rw = ratio st.name "wide-table" and rn = ratio st.name "narrow" in
          let rwl, rwh = range rw and rnl, rnh = range rn in
          Printf.printf "  %-12s %8.3f %8.3f-%-8.3f %8.0f %7.3f (%.3f-%.3f) %7.3f (%.3f-%.3f)  %s\n"
            st.name (median t) lo hi
            (gflop /. (median t /. 1000.0))
            (median rw) rwl rwh (median rn) rnl rnh st.note)
        stagings)
    sizes

(* -- main ------------------------------------------------------------------------------------ *)

let () =
  let sizes = ref [ 1024; 2048 ] and rounds = ref 16 and reps = ref 3 in
  let check_only = ref false and print_source = ref None in
  let usage () =
    prerr_endline
      "usage: metal_bf16_tile_probe [--sizes=N,...] [--rounds=R] [--reps=P] [--check-only] \
       [--print-source=N]";
    exit 2
  in
  let pos s = match int_of_string_opt s with Some v when v > 0 -> v | _ -> usage () in
  List.iter
    (fun arg ->
      let value name =
        let p = name ^ "=" in
        if String.starts_with ~prefix:p arg then
          Some (String.sub arg (String.length p) (String.length arg - String.length p))
        else None
      in
      match (value "--sizes", value "--rounds", value "--reps", value "--print-source") with
      | Some v, _, _, _ ->
          sizes :=
            List.map
              (fun s ->
                let s = pos s in
                if s mod 32 <> 0 then usage ();
                s)
              (String.split_on_char ',' v)
      | _, Some v, _, _ -> rounds := pos v
      | _, _, Some v, _ -> reps := pos v
      | _, _, _, Some v ->
          let v = pos v in
          if v mod 32 <> 0 then usage ();
          print_source := Some v
      | None, None, None, None -> if arg = "--check-only" then check_only := true else usage ())
    (List.tl (Array.to_list Sys.argv));
  (match !print_source with
  | Some s ->
      print_string (source ~n:s ~k:s);
      exit 0
  | None -> ());
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  Printf.printf "device: %s; options: %s\n" (Me.Device.get_attributes device).name
    (Ir.Compiler_options.render_metal
       (Ir.Compiler_options.metal ~routine_logging:false ~math_api:(Metal_math_api_runtime.get ())));
  let failures = checks device queue in
  if not !check_only then timings device queue ~sizes:!sizes ~rounds:!rounds ~reps:!reps;
  if failures = 0 then
    Printf.printf "\nVERDICT: the models stayed distinct and every baseline accumulated as known\n"
  else (
    Printf.printf "\nVERDICT: %d contradiction(s): the probe cannot discriminate as claimed\n"
      failures;
    exit 1)
