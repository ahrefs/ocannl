(* Metal's definition table (gh-ocannl-1580), compiled by the host C++ compiler beside the C table
   the cc backend prepends, and compared with it bitwise -- what [cuda_like_builtins_emulation] does
   for the CUDA and HIP tables. [Builtins_metal] is a stdlib-only library, so every PR compiles it
   as OCaml; this checks the MSL text inside it too, on machines with no Metal device: the uniform
   draws (scalar, packed [_vec] and per-[_lane]), the [<prec>_to_uint4x32] counters, the threefry
   functions, the f32 integer power, and the e5m2 codec.

   MSL is C++14 with its own vector types and casts, so the table compiles as C++ under shims for
   the few it uses: [as_type] (a bit copy between equal sizes), [uint4]/[int4]/[float4]/[float2]
   (members, indexing, the constructors and [+=] the table calls), [half] and [bfloat] (two bytes,
   converted from float by the C table's own round-to-nearest codecs -- the table only ever converts
   values the narrow format represents, so the rounding is never exercised on a correct table), and
   [rotate]. Each copy sits in a namespace of its own, beside the C table compiled as C in the same
   shared library, and is assembled by [Ir.C_syntax.builtin_definitions] over the probed names as a
   kernel's prelude is, so a dependency list missing an entry fails the compile here rather than on
   a device.

   The helpers compared are DERIVED from the tables: every function the Metal table shares by name
   with the C table is probed or excluded with its reason, and every Metal function the C table
   lacks is named Metal-only with its reason; a function in neither list fails the run. Negative
   control: a copy whose [uint4] indexing reverses the lanes must differ from the C table somewhere,
   or the comparison does not reach the packed forms.

   The compiler is the cc backend's, driven in C++ mode ([-x c++]). A host without a C++ front end
   fails the run, except under Windows' mingw toolchain, whose packaging can omit one: there the
   comparison is skipped and says so. *)
open Verdict.Claims
module B = Context.Builtins_cc
module Cc = Context.Cc_backend

let metal = Builtins_metal.builtins

(* How a probe calls a helper, read off its name: the emitted names follow [Ops.prec_string]. *)
type shape =
  | From_u4  (** [uint4x32_to_<prec>_uniform] and [_vec]: one [uint4] draw. *)
  | Lane  (** [uint4x32_to_<prec>_uniform_lane]: a draw and a lane of its [_vec] pack. *)
  | From_u32  (** [uint32_to_<prec>_uniform]: one 32-bit word. *)
  | To_u4 of string  (** [<prec>_to_uint4x32]: a value of [<prec>]. *)
  | Key_counter  (** [arrayjit_threefry4x32*]: a key and a counter. *)
  | Powi  (** [ocannl_powi_f32]: a base and [C_syntax.integer_power_operands]. *)
  | Fp8_decode  (** [fp8_to_single]: an e5m2 byte. *)
  | Fp8_encode  (** [single_to_fp8]: a float. *)
  | Excluded of string
  | Unknown

let no_double =
  "Metal has no double: its double forms are f32 fallbacks, which the C table's f64 forms are not"

let excluded =
  [
    ("uint4x32_to_double_uniform", no_double);
    ("uint4x32_to_double_uniform_vec", no_double);
    ("uint4x32_to_double_uniform_lane", no_double);
    ("double_to_uint4x32", no_double);
  ]

(* The Metal functions with no C counterpart. *)
let metal_only =
  let threefry = "internal to the threefry functions, which are compared" in
  [
    ("rotl32", threefry);
    ("threefry_round", threefry);
    ( "ocannl_sin",
      "a domain guard over Metal's fast sin, executed on Metal by unary_math_special_values" );
    ( "ocannl_cos",
      "a domain guard over Metal's fast cos, executed on Metal by unary_math_special_values" );
    ("ocannl_shfl_xor", "a simdgroup shuffle, with no host meaning");
  ]

(* The parameter type of the C table's [<prec>_to_uint4x32]; the Metal one's is read off its
   definition by the probe ([ocannl_arg_of]). *)
let c_param_type = function
  | "single" -> Some "float"
  | "int32" -> Some "int"
  | "int64" -> Some "long long"
  | "uint32" -> Some "unsigned int"
  | "uint64" -> Some "unsigned long long"
  | "byte" -> Some "unsigned char"
  | "uint16" | "bfloat16" | "half" -> Some "uint16_t"
  | "fp8" -> Some "uint8_t"
  | _ -> None

let chop_suffix s ~suffix =
  if String.ends_with ~suffix s then Some (String.sub s 0 (String.length s - String.length suffix))
  else None

let keys table = List.map (fun (k, _, _) -> k) table
let c_keys = keys B.builtins

let shape key =
  match List.assoc_opt key excluded with
  | Some reason -> Excluded reason
  | None -> (
      match key with
      | "arrayjit_threefry4x32" | "arrayjit_threefry4x32_crypto" | "arrayjit_threefry4x32_light" ->
          Key_counter
      | "ocannl_powi_f32" -> Powi
      | "fp8_to_single" -> Fp8_decode
      | "single_to_fp8" -> Fp8_encode
      | _ -> (
          match chop_suffix key ~suffix:"_to_uint4x32" with
          | Some prec -> if Option.is_some (c_param_type prec) then To_u4 prec else Unknown
          | None -> (
              match chop_suffix key ~suffix:"_uniform_lane" with
              | Some stem when List.mem (stem ^ "_uniform_vec") c_keys -> Lane
              | Some _ -> Unknown
              | None ->
                  if String.starts_with ~prefix:"uint4x32_to_" key then From_u4
                  else if
                    String.starts_with ~prefix:"uint32_to_" key
                    && String.ends_with ~suffix:"_uniform" key
                  then From_u32
                  else Unknown)))

(* MSL declarations that are not functions: the header, the threefry constants, the vector
   structs. *)
let is_function (_, definition, _) =
  let d = String.trim definition in
  not
    (List.exists
       (fun prefix -> String.starts_with ~prefix d)
       [ "#include"; "constant "; "struct "; "typedef" ])

let functions =
  List.filter_map (fun ((k, _, _) as e) -> if is_function e then Some k else None) metal

let shared = List.filter (fun key -> List.mem key c_keys) functions
let unshared = List.filter (fun key -> not (List.mem key c_keys)) functions

let probed =
  List.filter (fun key -> match shape key with Excluded _ | Unknown -> false | _ -> true) shared

(* The shims, emitted inside each copy's namespace ahead of the table. [broken] reverses [uint4]'s
   indexing: the negative control's defect. *)
let shims ~broken =
  Printf.sprintf
    {|
typedef unsigned int uint;
typedef unsigned short ushort;
template <typename T, typename F> inline T as_type(F v) {
  static_assert(sizeof(T) == sizeof(F), "as_type between different sizes");
  T r; memcpy(&r, &v, sizeof r); return r;
}
struct uint4 {
  uint32_t x, y, z, w;
  uint4() = default;
  uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) : x(a), y(b), z(c), w(d) {}
  uint32_t operator[](int i) const {
    switch (%s) { case 0: return x; case 1: return y; case 2: return z; default: return w; }
  }
  uint4 &operator+=(const uint4 &o) { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
};
struct int4 {
  int32_t x, y, z, w;
  int4() = default;
  explicit int4(uint4 u) : x(int32_t(u.x)), y(int32_t(u.y)), z(int32_t(u.z)), w(int32_t(u.w)) {}
  int32_t operator[](int i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
};
struct float4 {
  float x, y, z, w;
  float operator[](int i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
};
struct float2 {
  float x, y;
  float operator[](int i) const { return i == 0 ? x : y; }
};
struct half {
  uint16_t bits;
  half() = default;
  explicit half(float f) : bits(ocannl_shim_single_to_half(f)) {}
};
struct bfloat {
  uint16_t bits;
  bfloat() = default;
  explicit bfloat(float f) : bits(ocannl_shim_single_to_bfloat16(f)) {}
};
inline uint32_t rotate(uint32_t x, uint n) { return (x << n) | (x >> ((32u - n) & 31u)); }
|}
    (if broken then "3 - i" else "i")

let copies = [ ("ocannl_metal", "metal_", false); ("ocannl_metalbroken", "metalbroken_", true) ]
let probe_name ~prefix key = Printf.sprintf "probe_%s%s" prefix key

(* Every probe has one C signature: eight words of input, a lane, and the result's bytes out; it
   returns the result's size. *)
let probe ~extern ~name ~args ~call =
  Printf.sprintf
    "%sint %s(const unsigned int *in, int lane, unsigned char *out) {\n\
     %s  __typeof__(%s) r = %s;\n\
    \  (void)in; (void)lane;\n\
    \  memcpy(out, &r, sizeof r);\n\
    \  return (int)sizeof r;\n\
     }\n"
    (if extern then "extern \"C\" " else "")
    name args call call

(* A shape's argument declarations and call: [draw] declares the [uint4] draw [x] and counter [c]
   from the eight words, [param] spells a counter helper's parameter type, [fn] the helper. *)
let args_and_call ~draw ~param ~fn key =
  match shape key with
  | From_u4 -> (draw, fn ^ "(x)")
  | Lane -> (draw, fn ^ "(x, lane)")
  | Key_counter -> (draw, fn ^ "(x, c)")
  | From_u32 -> ("", fn ^ "(in[0])")
  | To_u4 prec ->
      ( Printf.sprintf "  %s a; memset(&a, 0, sizeof a); memcpy(&a, in, sizeof a);\n" (param prec),
        fn ^ "(a)" )
  | Powi ->
      ("  float f; memcpy(&f, in, sizeof f);\n", fn ^ "(f, in[1], in[2], (int)in[3], (int)in[4])")
  | Fp8_encode -> ("  float f; memcpy(&f, in, sizeof f);\n", fn ^ "(f)")
  | Fp8_decode -> ("", fn ^ "((uint8_t)in[0])")
  | Excluded _ | Unknown -> assert false

let c_probe key =
  let args, call =
    args_and_call key ~fn:key
      ~param:(fun prec -> Option.get (c_param_type prec))
      ~draw:
        "  uint4x32_t x, c; memcpy(x.v, in, sizeof x.v); memcpy(c.v, in + 4, sizeof c.v);\n\
        \  (void)x; (void)c;\n"
  in
  probe ~extern:false ~name:(probe_name ~prefix:"c_" key) ~args ~call

let metal_probe ~ns ~prefix key =
  let fn = ns ^ "::" ^ key in
  let args, call =
    args_and_call key ~fn
      ~param:(fun _ -> Printf.sprintf "__typeof__(ocannl_arg_of(&%s))" fn)
      ~draw:
        (Printf.sprintf
           "  %s::uint4 x(in[0], in[1], in[2], in[3]), c(in[4], in[5], in[6], in[7]);\n\
           \  (void)x; (void)c;\n"
           ns)
  in
  probe ~extern:true ~name:(probe_name ~prefix key) ~args ~call

(* The C table, as C, with the two narrow codecs the [half] and [bfloat] shims convert through. *)
let c_source =
  String.concat "\n"
    ([ B.includes ]
    @ Ir.C_syntax.builtin_definitions B.builtins
        ~keys:(probed @ [ "HALF_T"; "FLOAT_TO_HALF"; "HALF_TO_UINT16"; "single_to_bfloat16" ])
    @ [
        "uint16_t ocannl_shim_single_to_half(float f) { return HALF_TO_UINT16(FLOAT_TO_HALF(f)); }";
        "uint16_t ocannl_shim_single_to_bfloat16(float f) { return single_to_bfloat16(f); }";
      ]
    @ List.map c_probe probed)

(* The Metal copies, as C++. *)
let cxx_source =
  String.concat "\n"
    ([
       {|#include <stdint.h>
#include <string.h>
extern "C" uint16_t ocannl_shim_single_to_half(float);
extern "C" uint16_t ocannl_shim_single_to_bfloat16(float);
/* The parameter type of a one-parameter function, for a counter helper's probe. */
template <typename A, typename R> A ocannl_arg_of(R (*)(A));
#define constant const
#define thread|};
     ]
    @ List.concat_map
        (fun (ns, _, broken) ->
          [ Printf.sprintf "namespace %s {" ns; shims ~broken ]
          @ Ir.C_syntax.builtin_definitions metal ~keys:probed
          @ [ "}" ])
        copies
    @ List.concat_map (fun (ns, prefix, _) -> List.map (metal_probe ~ns ~prefix) probed) copies)

let no_cxx = "mingw with no C++ front end"

let write path contents =
  let out = open_out_bin path in
  output_string out contents;
  close_out out

let temp ext = Filename.temp_file "ocannl_metal_builtins_" ext
let remove paths = List.iter (fun p -> try Sys.remove p with Sys_error _ -> ()) paths

(* Whether the host compiler has a C++ front end: asked only under Windows, where a missing one is a
   skip rather than a failure. *)
let host_cxx =
  lazy
    ((not Sys.win32)
    ||
    let src = temp ".cpp" and obj = temp ".o" in
    write src "extern \"C\" int ocannl_cxx_probe(void) { return 0; }\n";
    let ok =
      Sys.command
        (Printf.sprintf "%s -x c++ -c %s -o %s >%s 2>&1" (Cc.compiler_command ())
           (Filename.quote src) (Filename.quote obj) Filename.null)
      = 0
    in
    remove [ src; obj ];
    ok)

(* The two sources compiled into one shared library, once. An implicit declaration is an error, as
   clang already makes it: it is what a definition missing from a dependency list looks like. *)
let library =
  lazy
    (let c_src = temp ".c" and cxx_src = temp ".cpp" and log = temp ".log" in
     let dll = temp (if Sys.win32 then ".dll" else ".so") in
     write c_src c_source;
     write cxx_src cxx_source;
     let command =
       Printf.sprintf
         "%s -O0 -Werror=implicit-function-declaration %s -o %s %s -x c++ -fno-exceptions \
          -fno-rtti %s -lm >%s 2>&1"
         (Cc.compiler_command ()) (Lazy.force Cc.kernel_link_flags) (Filename.quote dll)
         (Filename.quote c_src) (Filename.quote cxx_src) (Filename.quote log)
     in
     if Sys.command command <> 0 then (
       let ic = open_in_bin log in
       prerr_string (really_input_string ic (in_channel_length ic));
       close_in ic;
       remove [ c_src; cxx_src; log; dll ];
       failwith ("cannot compile the Metal builtins probe: " ^ command));
     let lib = Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ] in
     at_exit (fun () ->
         Dl.dlclose ~handle:lib;
         remove [ c_src; cxx_src; log; dll ]);
     lib)

let bound = Hashtbl.create 64

let bind name =
  match Hashtbl.find_opt bound name with
  | Some fn -> fn
  | None ->
      let fn =
        let open Ctypes in
        Foreign.foreign ~from:(Lazy.force library) name
          (ptr uint32_t @-> int @-> ptr uchar @-> returning int)
      in
      Hashtbl.replace bound name fn;
      fn

(* One call of a probe: the result's size and bytes. *)
let run name words lane =
  let open Ctypes in
  let input = CArray.of_list uint32_t (List.map Unsigned.UInt32.of_int words) in
  let out = CArray.make uchar ~initial:Unsigned.UChar.zero 64 in
  let size = bind name (CArray.start input) lane (CArray.start out) in
  (size, List.init size (fun i -> Unsigned.UChar.to_int (CArray.get out i)))

(* A [_lane] helper's lanes are its [_vec] pack's: the pack's size over the lane's. *)
let lanes key =
  match shape key with
  | Lane ->
      let stem = Option.get (chop_suffix key ~suffix:"_uniform_lane") in
      let zeros = List.init 8 (fun _ -> 0) in
      let vec_size, _ = run (probe_name ~prefix:"c_" (stem ^ "_uniform_vec")) zeros 0 in
      let lane_size, _ = run (probe_name ~prefix:"c_" key) zeros 0 in
      List.init (vec_size / lane_size) Fun.id
  | _ -> [ 0 ]

(* Inputs, eight words each. Draws: boundary words in every position, and a deterministic
   pseudo-random sample. *)
let edges =
  [
    0;
    0xFFFFFFFF;
    0x80000000;
    0x7FFFFFFF;
    0x000000FF;
    0x0000FFFF;
    0x00008000;
    0x00007FFF;
    0x00FFFFFF;
    0xFF000000;
    0x00000100;
    0x7F800000;
    0x7FC00000;
    0x3F800000;
    0x000080FF;
    0x01234567;
  ]

let draws =
  let lcg = ref 0x2545F491 in
  let next () =
    lcg := ((!lcg * 1103515245) + 12345) land 0xFFFFFFFFFFFF;
    (!lcg lsr 16) land 0xFFFFFFFF
  in
  let n = List.length edges in
  let edge i = List.nth edges (i mod n) in
  List.init n (fun i -> List.init 8 (fun _ -> edge i))
  @ List.init n (fun i -> List.init 8 (fun k -> edge (i + (3 * k))))
  @ List.init 2048 (fun _ -> List.init 8 (fun _ -> next ()))

let pad words = words @ List.init (8 - List.length words) (fun _ -> 0)

(* Every f32 exponent of both signs under boundary mantissas: zeros, subnormals, infinities, NaNs,
   and the e5m2 rounding boundaries (mantissa bits 21 and below). *)
let f32_patterns =
  let mantissas =
    [ 0; 1; 2; 3; 0xFFFFF; 0x100000; 0x100001; 0x1FFFFF; 0x200000; 0x300000; 0x7FFFFF; 0x2AAAAA ]
  in
  List.concat_map
    (fun sign ->
      List.concat_map
        (fun e -> List.map (fun m -> pad [ sign lor (e lsl 23) lor m ]) mantissas)
        (List.init 256 Fun.id))
    [ 0; 0x80000000 ]

(* [integer_power_domain]'s domain, with the operands the emitter writes. *)
let powi_inputs =
  let exponents =
    List.map Float.of_int Integer_power_cases.exponents
    @ List.map (fun (_, n, _) -> n) Integer_power_cases.large_exponents
  in
  List.concat_map
    (fun x ->
      List.map
        (fun n ->
          let lo, hi, shift, negative = Ir.C_syntax.integer_power_operands n in
          pad
            [
              Int32.to_int (Int32.bits_of_float x) land 0xFFFFFFFF;
              Int64.to_int lo;
              Int64.to_int hi;
              shift land 0xFFFFFFFF;
              (if negative then 1 else 0);
            ])
        exponents)
    Integer_power_cases.bases

let inputs key =
  match shape key with
  | Fp8_decode -> List.init 256 (fun b -> pad [ b ])
  | Fp8_encode -> f32_patterns @ draws
  | Powi -> powi_inputs
  | From_u32 -> List.init 65536 (fun v -> pad [ (v lsl 16) lor 0xFFFF ]) @ draws
  | _ -> draws

let () =
  case "the Metal table's functions are accounted for" @@ fun () ->
  p_all
    "Metal: every function the table shares with the C table is probed or excluded with a reason"
    shared ~f:(fun key ->
      let ok = shape key <> Unknown in
      if not ok then Printf.eprintf "(not part of the golden) no probe shape for %s\n" key;
      ok);
  p_all "Metal: every function the C table lacks is named Metal-only with a reason" unshared
    ~f:(fun key ->
      let ok = List.mem_assoc key metal_only in
      if not ok then Printf.eprintf "(not part of the golden) %s is in neither table's list\n" key;
      ok);
  p_all "Metal: every excluded or Metal-only name is a function of the table" (excluded @ metal_only)
    ~f:(fun (key, _) -> List.mem key functions);
  p_all
    "Metal: the probed helpers take every probe shape: draws, lanes, counters, threefry, powers, \
     codec" [ From_u4; Lane; From_u32; To_u4 "single"; Key_counter; Powi; Fp8_decode; Fp8_encode ]
    ~f:(fun s ->
      List.exists
        (fun key -> match (s, shape key) with To_u4 _, To_u4 _ -> true | _, s' -> s = s')
        probed);
  Printf.eprintf "(not part of the golden) %d Metal helpers probed: %s\n" (List.length probed)
    (String.concat " " probed)

let hex (size, bytes) =
  Printf.sprintf "%d bytes [%s]" size (String.concat " " (List.map (Printf.sprintf "%02x") bytes))

(* Whether a copy's helper returns the C one's bytes at every input and lane. *)
let agrees ~log ~prefix key =
  let lanes = lanes key in
  List.for_all
    (fun words ->
      List.for_all
        (fun lane ->
          let got = run (probe_name ~prefix key) words lane
          and want = run (probe_name ~prefix:"c_" key) words lane in
          let ok = got = want in
          if log && not ok then
            Printf.eprintf "(not part of the golden) %s lane %d of [%s]: got %s want %s\n" key lane
              (String.concat " " (List.map (Printf.sprintf "%08x") words))
              (hex got) (hex want);
          ok)
        lanes)
    (inputs key)

let () =
  case "Metal helpers against the C table" @@ fun () ->
  let when_ = Lazy.force host_cxx in
  gated_all ~aggregation:`Environment ~when_ ~on:no_cxx
    "Metal: every probed helper returns the C one's bytes, for every input and lane" probed
    ~f:(agrees ~log:true ~prefix:"metal_");
  gated_exists ~aggregation:`Environment ~when_ ~on:no_cxx
    "Metal: a copy whose uint4 indexing reverses its lanes differs from the C table somewhere"
    probed ~f:(fun key -> not (agrees ~log:false ~prefix:"metalbroken_" key))
