(* What do Metal's f32 [sin], [cos] and [tanh] return on special inputs, near zero and at large
   magnitudes, and what does each candidate spelling cost? gh-ocannl-1435. Since gh-ocannl-1515 also
   [exp], [log], [exp2] and [log2], in their [plain] and [precise::] spellings only.

   Standalone like [metal_half_math_probe]: it links Metal, ctypes and Arrayjit's pure helpers, not
   OCANNL. Each op is compiled in several spellings -- [plain] (whatever the policy's functions
   are), [precise::op], and the guarded spellings considered for the backend: [isfinite] (NaN for
   non-finite inputs only), [range] (the fast function below 2^22, [precise::] elsewhere: what
   [Builtins_metal.ocannl_sin]/[ocannl_cos] emit), and for [tanh] [small-x] (pass x through below
   2^-12) -- under the two production policies ([Compiler_options.metal]'s legacy
   [fastMathEnabled=false], which macOS 14 takes, and the modern split, Safe arithmetic with fast
   functions). The backend renders f32 [tanh] as [precise::tanh].

   Three tables. Special inputs (signed zeros, infinities, NaN, f32 subnormals): the result bits.
   Accuracy: the worst error in f32 ulps against an f64 reference per |x| band, over the inputs
   whose reference rounds to a finite f32 (a NaN or infinite result there counts as [inf]), with the
   worst input where it exceeds 4 ulps, over a sample of 2048 mantissas from every binade of both
   signs (1048576 inputs); how many results differ bitwise from [precise::]; and, kept out of the
   bands, how many subnormal inputs or results are more than 4 ulps off (a flush). Cost: GPU-clock
   medians, rotated interleaved repeats, relative to [plain] under the same policy, for 1024 calls
   per thread on advancing inputs summed into an accumulator (ALU-bound throughput; the calls do not
   depend on each other) and for one call per element over 16M elements (streaming).

   Run on a Metal Mac (macOS 15+) with:

   dune exec benchmarks/runners/ocannl/metal_f32_math_probe.exe [OP...]

   naming the ops to probe (all of them by default). *)

module Me = Metal

type math_policy = Legacy_fast_false | Production_safe_fast

let policies = [ ("legacy-fast=false", Legacy_fast_false); ("production", Production_safe_fast) ]

let apply_policy options = function
  | Legacy_fast_false -> Me.CompileOptions.set_fast_math_enabled options false
  | Production_safe_fast ->
      Me.CompileOptions.set_math_mode options Safe;
      Me.CompileOptions.set_math_floating_point_functions options Fast

(* Spelling name and MSL expression over the float [x]. *)
let forms op =
  let plain = Printf.sprintf "%s(x)" op in
  [ ("plain", plain); ("precise", Printf.sprintf "precise::%s(x)" op) ]
  @
  match op with
  | "tanh" -> [ ("small-x", Printf.sprintf "(fabs(x) < 0x1p-12f ? x : %s)" plain) ]
  | "exp" | "log" | "exp2" | "log2" -> []
  | _ ->
      [
        ("isfinite", Printf.sprintf "(isfinite(x) ? %s : x - x)" plain);
        ("range", Printf.sprintf "(fabs(x) < 0x1p22f ? %s : precise::%s(x))" plain op);
      ]

let all_ops =
  [
    ("sin", Float.sin);
    ("cos", Float.cos);
    ("tanh", Float.tanh);
    ("exp", Float.exp);
    ("log", Float.log);
    ("exp2", Float.exp2);
    ("log2", Float.log2);
  ]

(* The ops named on the command line, or all of them. *)
let ops =
  match List.tl (Array.to_list Sys.argv) with
  | [] -> all_ops
  | names -> List.filter (fun (op, _) -> List.mem op names) all_ops

let kname op i = Printf.sprintf "%s_%d" op i

let source =
  "#include <metal_stdlib>\nusing namespace metal;\n"
  ^ String.concat ""
      (List.concat_map
         (fun (op, _) ->
           List.mapi
             (fun i (_, expr) ->
               Printf.sprintf
                 {|kernel void %s(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { float x = in[gid]; out[gid] = %s; }
kernel void t%s(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    device const uint* n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  float x = in[gid]; float acc = 0.0f; uint m = n[0];
  for (uint i = 0; i < m; i++) { acc += %s; x += 0x1p-10f; }
  out[gid] = acc; }
|}
                 (kname op i) expr (kname op i) expr)
             (forms op))
         ops)

let specials =
  [
    -0.0;
    0.0;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    0x1p-140;
    -0x1p-140;
    0x1p-20;
    1.0;
    -1.0;
    2.0;
    88.75;
    -103.5;
    128.0;
    -150.0;
  ]

(* 2048 evenly spread mantissas from each of the 256 exponent fields, both signs. *)
let sample =
  Array.init (1 lsl 20) (fun i ->
      let sign = i lsr 19 and e = (i lsr 11) land 0xff and m = (i land 0x7ff) lsl 12 in
      Int32.float_of_bits (Int32.of_int ((sign lsl 31) lor (e lsl 23) lor m)))

let f32_ulp v =
  let a = Float.abs v in
  if a < 0x1p-126 then 0x1p-149 else Float.ldexp 1.0 (snd (Float.frexp a) - 24)

let bands =
  [
    ("|x|<2^-12", fun a -> a < 0x1p-12);
    ("2^-12..2^-6", fun a -> a >= 0x1p-12 && a < 0x1p-6);
    ("2^-6..1", fun a -> a >= 0x1p-6 && a < 1.0);
    ("1..2^5", fun a -> a >= 1.0 && a < 0x1p5);
    ("2^5..2^22", fun a -> a >= 0x1p5 && a < 0x1p22);
    (">=2^22", fun a -> a >= 0x1p22);
  ]

let ropts =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

let median xs =
  let a = Array.of_list xs in
  Array.sort Float.compare a;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let rotate n xs =
  let len = List.length xs in
  let n = n mod len in
  List.filteri (fun i _ -> i >= n) xs @ List.filteri (fun i _ -> i < n) xs

let () =
  (match Metal_math_api_runtime.get () with
  | Ir.Compiler_options.Modern_split -> ()
  | Legacy ->
      Printf.eprintf "metal_f32_math_probe needs the macOS 15+ split math selectors\n";
      exit 2);
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let n = Array.length sample in
  let input = Me.Buffer.on_device device ~length:(4 lsl 24) ropts in
  let output = Me.Buffer.on_device device ~length:(4 lsl 24) ropts in
  let nbuf = Me.Buffer.on_device device ~length:4 ropts in
  let open Ctypes in
  let input_f = coerce (ptr void) (ptr float) (Me.Buffer.contents input) in
  let output_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output) in
  let n_u32 = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents nbuf) in
  let load xs = Array.iteri (fun i x -> input_f +@ i <-@ x) xs in
  let states = Hashtbl.create 64 in
  List.iter
    (fun (label, policy) ->
      let options = Me.CompileOptions.init () in
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
      apply_policy options policy;
      let library = Me.Library.on_device device ~source options in
      List.iter
        (fun (op, _) ->
          List.iteri
            (fun i _ ->
              List.iter
                (fun k ->
                  let fn = Me.Library.new_function_with_name library k in
                  Hashtbl.add states (label, k)
                    (fst (Me.ComputePipelineState.on_device_with_function device fn)))
                [ kname op i; "t" ^ kname op i ])
            (forms op))
        ops)
    policies;
  let dispatch label k ~threads =
    let cb = Me.CommandBuffer.on_queue queue in
    let enc = Me.ComputeCommandEncoder.on_buffer cb in
    Me.ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find states (label, k));
    Me.ComputeCommandEncoder.set_buffer enc ~index:0 input;
    Me.ComputeCommandEncoder.set_buffer enc ~index:1 output;
    Me.ComputeCommandEncoder.set_buffer enc ~index:2 nbuf;
    let width = min 256 threads in
    Me.ComputeCommandEncoder.dispatch_threadgroups enc
      ~threadgroups_per_grid:{ width = (threads + width - 1) / width; height = 1; depth = 1 }
      ~threads_per_threadgroup:{ width; height = 1; depth = 1 };
    Me.ComputeCommandEncoder.end_encoding enc;
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    Me.CommandBuffer.get_gpu_end_time cb -. Me.CommandBuffer.get_gpu_start_time cb
  in
  let run label k xs =
    load xs;
    ignore (dispatch label k ~threads:(Array.length xs) : float);
    Array.init (Array.length xs) (fun i -> !@(output_f +@ i))
  in
  Printf.printf "device: %s; MSL 3.1\n" (Me.Device.get_attributes device).name;
  Printf.printf "\nspecial inputs (result bits): %s\n"
    (String.concat ", " (List.map (Printf.sprintf "%h") specials));
  List.iter
    (fun (label, _) ->
      List.iter
        (fun (op, _) ->
          List.iteri
            (fun i (form, _) ->
              let r = run label (kname op i) (Array.of_list specials) in
              Printf.printf "  %-18s %-5s %-9s %s\n" label op form
                (String.concat " "
                   (Array.to_list
                      (Array.map (fun v -> Printf.sprintf "%08lx" (Int32.bits_of_float v)) r))))
            (forms op))
        ops)
    policies;
  Printf.printf
    "\n\
     worst f32 ulps vs f64 per |x| band over %d sampled inputs (finite results, subnormal inputs \
     and results aside), the count of results bitwise different from precise:: under the same \
     policy (NaNs equal), and how many of the subnormal inputs or results are more than 4 ulps off \
     (a flush):\n"
    n;
  Printf.printf "  %-18s %-5s %-9s %s %8s %s\n" "policy" "op" "form"
    (String.concat " " (List.map (fun (b, _) -> Printf.sprintf "%11s" b) bands))
    "!=precise" "subnormal-off";
  let same a b =
    Int32.equal (Int32.bits_of_float a) (Int32.bits_of_float b) || (Float.is_nan a && Float.is_nan b)
  in
  List.iter
    (fun (label, _) ->
      List.iter
        (fun (op, reference) ->
          let precise = run label (kname op 1) sample in
          List.iteri
            (fun i (form, _) ->
              let r = run label (kname op i) sample in
              let worst = Array.make (List.length bands) (0.0, 0) in
              let differ = ref 0 and subnormal = ref 0 and flushed = ref 0 in
              Array.iteri
                (fun j x ->
                  if not (same r.(j) precise.(j)) then incr differ;
                  let want = reference x in
                  (* Rounded once to f32: past the f32 range the reference is infinite, so not an
                     error of the function. *)
                  let want32 = Int32.float_of_bits (Int32.bits_of_float want) in
                  let is_subnormal v = v <> 0.0 && Float.abs v < 0x1p-126 in
                  if Float.is_finite want32 && (is_subnormal x || is_subnormal want32) then (
                    (* Subnormal inputs or results are counted apart from the bands, so a flush to
                       zero does not hide the accuracy of the band they fall in. *)
                    incr subnormal;
                    if (not (same r.(j) want32)) && Float.abs (r.(j) -. want) > 4.0 *. f32_ulp want
                    then incr flushed)
                  else if Float.is_finite want32 then
                    (* A non-finite result for a finite reference is an unbounded error. *)
                    let err =
                      if Float.is_finite r.(j) then Float.abs (r.(j) -. want) /. f32_ulp want
                      else Float.infinity
                    in
                    List.iteri
                      (fun b (_, inside) ->
                        if inside (Float.abs x) && err > fst worst.(b) then worst.(b) <- (err, j))
                      bands)
                sample;
              Printf.printf "  %-18s %-5s %-9s %s %8d %d/%d\n" label op form
                (String.concat " "
                   (Array.to_list (Array.map (fun (w, _) -> Printf.sprintf "%11.3f" w) worst)))
                !differ !flushed !subnormal;
              Array.iter
                (fun (w, j) ->
                  if w > 4.0 then
                    Printf.printf "      e.g. %s(%h): got %h, f64 %h\n" op sample.(j) r.(j)
                      (reference sample.(j)))
                worst)
            (forms op))
        ops)
    policies;
  let repeats = 15 and chain_threads = 1 lsl 16 and stream_threads = 1 lsl 24 in
  load (Array.init stream_threads (fun i -> Float.of_int ((i land 0xffff) - 0x8000) *. 0x1p-12));
  n_u32 <-@ Unsigned.UInt32.of_int 1024;
  Printf.printf "\nGPU-clock medians, %d rotated interleaved repeats per arm:\n" repeats;
  List.iter
    (fun (shape, prefix, threads) ->
      Printf.printf "  %s:\n" shape;
      List.iter
        (fun (label, _) ->
          List.iter
            (fun (op, _) ->
              let arms = List.mapi (fun i (form, _) -> (form, prefix ^ kname op i)) (forms op) in
              List.iter (fun (_, k) -> ignore (dispatch label k ~threads : float)) arms;
              let samples = Hashtbl.create 8 in
              for r = 0 to repeats - 1 do
                List.iter
                  (fun (form, k) ->
                    let t = dispatch label k ~threads in
                    Hashtbl.replace samples form
                      (t :: Option.value ~default:[] (Hashtbl.find_opt samples form)))
                  (rotate r arms)
              done;
              let base = median (Hashtbl.find samples "plain") in
              List.iter
                (fun (form, _) ->
                  let t = median (Hashtbl.find samples form) in
                  Printf.printf "    %-18s %-5s %-9s %9.4f ms %7.3fx plain\n" label op form
                    (t *. 1000.0) (t /. base))
                arms)
            ops)
        policies)
    [
      ("ALU-bound: 65536 threads x 1024 calls on advancing inputs, summed", "t", chain_threads);
      ("streaming: 16777216 threads x 1 call (load, call, store)", "", stream_threads);
    ]
