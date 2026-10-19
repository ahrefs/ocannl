(* Do Metal's half math overloads return f32 results narrowed once, under both compile-option math
   policies? gh-ocannl-1288.

   Standalone like [metal_reassoc_probe]: it links Metal, ctypes and Arrayjit's pure helpers, not
   OCANNL, so it measures the compiler directly. Each unary builtin OCANNL renders on half operands
   is compiled twice -- the half overload [op(h)] and the widened [(half)op((float)(h))] -- and run
   over every one of the 65536 half bit patterns, under the two production policies
   ([Compiler_options.metal]'s legacy [fastMathEnabled=false], which macOS 14 takes, and the modern
   split, Safe arithmetic with fast functions) plus the compiler default. The legacy property is
   still accepted by newer runtimes, so one macOS 15+ host can exercise both policies; it cannot
   exercise macOS 14's compiler.

   Columns: [differ] counts patterns whose half-overload result is not bitwise the widened result
   (all NaNs compare equal); [h-ulp]/[w-ulp] are each form's worst error in half ulps against an f64
   reference over the finite in-range results; the first differing patterns follow each row. Run on
   a Metal Mac with:

   dune exec benchmarks/runners/ocannl/metal_half_math_probe.exe *)

module Me = Metal

type math_policy = Default | Legacy_fast_false | Production_safe_fast

let policies =
  [
    ("legacy-fast=false", Legacy_fast_false);
    ("production-safe-fast", Production_safe_fast);
    ("default", Default);
  ]

let apply_policy options = function
  | Default -> ()
  | Legacy_fast_false -> Me.CompileOptions.set_fast_math_enabled options false
  | Production_safe_fast ->
      Me.CompileOptions.set_math_mode options Safe;
      Me.CompileOptions.set_math_floating_point_functions options Fast

(* MSL name and an f64 reference. [Tanh_approx] renders as [tanh] on Metal. *)
let ops =
  [
    ("exp", Float.exp);
    ("log", Float.log);
    ("exp2", fun x -> Float.pow 2.0 x);
    ("log2", fun x -> Float.log x /. Float.log 2.0);
    ("sin", Float.sin);
    ("cos", Float.cos);
    ("sqrt", Float.sqrt);
    ("rsqrt", fun x -> 1.0 /. Float.sqrt x);
    ("tanh", Float.tanh);
  ]

let source =
  "#include <metal_stdlib>\nusing namespace metal;\n"
  ^ String.concat ""
      (List.map
         (fun (op, _) ->
           Printf.sprintf
             {|kernel void h_%s(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { out[gid] = %s(in[gid]); }
kernel void w_%s(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { out[gid] = (half)%s((float)(in[gid])); }
|}
             op op op op)
         ops)

let n = 65536

let ropts =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

let half_value bits = Ir.Ops.half_to_single bits
let is_nan_bits bits = bits land 0x7c00 = 0x7c00 && bits land 0x03ff <> 0

(* One half ulp at [v]'s binade (normal halves), or the subnormal spacing. *)
let half_ulp v =
  let a = Float.abs v in
  if a < 0x1p-14 then 0x1p-24 else Float.ldexp 1.0 (snd (Float.frexp a) - 11)

let () =
  (match Metal_math_api_runtime.get () with
  | Ir.Compiler_options.Modern_split -> ()
  | Legacy ->
      Printf.eprintf
        "metal_half_math_probe needs the macOS 15+ split math selectors for the modern policy\n";
      exit 2);
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let input = Me.Buffer.on_device device ~length:(2 * n) ropts in
  let output = Me.Buffer.on_device device ~length:(2 * n) ropts in
  let open Ctypes in
  let input_u = coerce (ptr void) (ptr uint16_t) (Me.Buffer.contents input) in
  let output_u = coerce (ptr void) (ptr uint16_t) (Me.Buffer.contents output) in
  for i = 0 to n - 1 do
    input_u +@ i <-@ Unsigned.UInt16.of_int i
  done;
  let run library name =
    let fn = Me.Library.new_function_with_name library name in
    let state, _ = Me.ComputePipelineState.on_device_with_function device fn in
    let cb = Me.CommandBuffer.on_queue queue in
    let enc = Me.ComputeCommandEncoder.on_buffer cb in
    Me.ComputeCommandEncoder.set_compute_pipeline_state enc state;
    Me.ComputeCommandEncoder.set_buffer enc ~index:0 input;
    Me.ComputeCommandEncoder.set_buffer enc ~index:1 output;
    Me.ComputeCommandEncoder.dispatch_threadgroups enc
      ~threadgroups_per_grid:{ width = n / 256; height = 1; depth = 1 }
      ~threads_per_threadgroup:{ width = 256; height = 1; depth = 1 };
    Me.ComputeCommandEncoder.end_encoding enc;
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    Array.init n (fun i -> Unsigned.UInt16.to_int !@(output_u +@ i))
  in
  let worst_ulp reference got =
    let worst = ref 0.0 in
    Array.iteri
      (fun i bits ->
        let want = reference (half_value i) in
        if Float.is_finite want && Float.abs want <= 65504.0 && not (is_nan_bits bits) then
          worst := Float.max !worst (Float.abs (half_value bits -. want) /. half_ulp want))
      got;
    !worst
  in
  let same a b = a = b || (is_nan_bits a && is_nan_bits b) in
  Printf.printf "device: %s; MSL 3.1; all %d half bit patterns per op\n"
    (Me.Device.get_attributes device).name n;
  let results = Hashtbl.create 64 in
  let production_differ = ref 0 in
  List.iter
    (fun (label, policy) ->
      let options = Me.CompileOptions.init () in
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
      apply_policy options policy;
      Printf.printf "\n%s (effective fastMathEnabled=%b mathMode=%s functions=%s)\n" label
        (Me.CompileOptions.get_fast_math_enabled options)
        (match Me.CompileOptions.get_math_mode options with
        | Safe -> "safe"
        | Relaxed -> "relaxed"
        | Fast -> "fast")
        (match Me.CompileOptions.get_math_floating_point_functions options with
        | Fast -> "fast"
        | Precise -> "precise");
      Printf.printf "  %-6s %8s %8s %8s\n" "op" "differ" "h-ulp" "w-ulp";
      let library = Me.Library.on_device device ~source options in
      List.iter
        (fun (op, reference) ->
          let h = run library ("h_" ^ op) and w = run library ("w_" ^ op) in
          Hashtbl.replace results (label, op) w;
          let differ = ref 0 and examples = ref [] in
          Array.iteri
            (fun i hb ->
              if not (same hb w.(i)) then (
                if !differ < 3 then
                  examples :=
                    Printf.sprintf "%s(%h): half %h, widened %h" op (half_value i) (half_value hb)
                      (half_value w.(i))
                    :: !examples;
                incr differ))
            h;
          if policy <> Default then production_differ := !production_differ + !differ;
          Printf.printf "  %-6s %8d %8.3f %8.3f\n" op !differ (worst_ulp reference h)
            (worst_ulp reference w);
          List.iter (Printf.printf "           e.g. %s\n") (List.rev !examples))
        ops)
    policies;
  Printf.printf
    "\nwidened results, legacy-fast=false vs production-safe-fast (patterns differing):\n";
  List.iter
    (fun (op, _) ->
      let a = Hashtbl.find results ("legacy-fast=false", op)
      and b = Hashtbl.find results ("production-safe-fast", op) in
      let differ = ref 0 in
      Array.iteri (fun i x -> if not (same x b.(i)) then incr differ) a;
      Printf.printf "  %-6s %8d\n" op !differ)
    ops;
  Printf.printf
    "\n\
     summary: %d half-overload result(s) differ from the widened form under the two production \
     policies\n"
    !production_differ
