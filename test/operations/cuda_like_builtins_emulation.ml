(* The CUDA and HIP definition tables' device-free conversion helpers (gh-ocannl-1353), compiled by
   the host C compiler beside the C table the cc backend prepends, and compared with it bitwise.
   Since the two tables became stdlib-only libraries, every PR compiles them as OCaml; this checks
   the C text inside them too, without a GPU: the helpers that turn a [uint4x32_t] draw into a value
   of each precision (scalar, packed [_vec] and per-[_lane]), the ones that turn a value of each
   precision back into a [uint4x32_t] counter, and HIP's guards around ROCm's fp8 cast.

   The helpers compared are DERIVED from the tables: every function the CUDA (or HIP) table shares
   by name with the C table is probed, or excluded below with its reason, and a shared name this
   test has no probe shape for fails the run. Each copy is assembled the way a kernel's prelude is,
   by [Ir.C_syntax.builtin_definitions] over the probed names, so a dependency list missing an entry
   fails the compile here rather than on a device. The device intrinsics are host shims faithful to
   the vendor's definition for every value these helpers pass them, so a difference is the helper's;
   the fp16 and bf16 rounding they shim is the C table's own, which this test therefore does not
   check -- [half_codec_exhaustive] and [bf16_builtins] do.

   Negative controls: a copy of each vendor table whose [__hiloint2double] swaps its words must
   differ from the C table somewhere (or the comparison does not reach the 64-bit helpers), and a
   HIP fp8 guard whose zero drops its sign must differ from the codec somewhere. *)
open Verdict.Claims
module B = Context.Builtins_cc
module Cc = Context.Cc_backend

type vendor = { name : string; label : string; table : (string * string * string list) list }

let cuda = { name = "cuda"; label = "CUDA"; table = Builtins_cuda.builtins }
let hip = { name = "hip"; label = "HIP"; table = Builtins_hip.builtins }
let vendors = [ cuda; hip ]

(* How a probe calls a helper, read off its name: the emitted names follow [Ops.prec_string]. *)
type shape =
  | From_u4  (** [uint4x32_to_<prec>_uniform] and [_vec]: one [uint4x32_t] draw. *)
  | Lane  (** [uint4x32_to_<prec>_uniform_lane]: a draw and a lane of its [_vec] pack. *)
  | From_u32  (** [uint32_to_<prec>_uniform]: one 32-bit word. *)
  | To_u4 of string  (** [<prec>_to_uint4x32]: a value of [<prec>]. *)
  | Excluded of string
  | Unknown

let excluded =
  let threefry =
    "the CUDA-like threefry round takes its state by C++ reference, which no host C compiler reads"
  and powers = "compared by runtime_power_fast_math_emulation" in
  [
    ("arrayjit_threefry4x32", threefry);
    ("arrayjit_threefry4x32_crypto", threefry);
    ("arrayjit_threefry4x32_light", threefry);
    ("ocannl_powi_f32", powers);
    ("ocannl_powi_f64", powers);
  ]

(* The parameter type of [<prec>_to_uint4x32], per table: the C table passes the 16- and 8-bit
   floating formats as their storage integers, the vendors as their own types. *)
let param_type prec ~vendor =
  match (prec, vendor) with
  | "single", _ -> Some "float"
  | "double", _ -> Some "double"
  | "int32", _ -> Some "int"
  | "int64", _ -> Some "long long"
  | "uint32", _ -> Some "unsigned int"
  | "uint64", _ -> Some "unsigned long long"
  | "byte", _ -> Some "unsigned char"
  | "uint16", _ -> Some "unsigned short"
  | ("bfloat16" | "half"), None -> Some "uint16_t"
  | "fp8", None -> Some "uint8_t"
  | "bfloat16", Some "cuda" -> Some "__nv_bfloat16"
  | "bfloat16", Some "hip" -> Some "__hip_bfloat16"
  | "half", Some _ -> Some "__half"
  | "fp8", Some "cuda" -> Some "__nv_fp8_e5m2"
  | "fp8", Some "hip" -> Some "__hip_fp8_e5m2"
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
      match chop_suffix key ~suffix:"_to_uint4x32" with
      | Some prec ->
          if
            List.for_all
              (fun vendor -> Option.is_some (param_type prec ~vendor))
              (None :: List.map (fun v -> Some v.name) vendors)
          then To_u4 prec
          else Unknown
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
              else Unknown))

let is_function (_, definition, _) =
  not (String.starts_with ~prefix:"typedef" (String.trim definition))

(* The functions a vendor table shares by name with the C table. *)
let shared vendor =
  List.filter_map
    (fun ((key, _, _) as entry) ->
      if is_function entry && List.mem key c_keys then Some key else None)
    vendor.table

let probed vendor =
  List.filter
    (fun key -> match shape key with Excluded _ | Unknown -> false | _ -> true)
    (shared vendor)

(* HIP's guards around ROCm's float-to-e5m2 cast (gh-ocannl-647/648). The cast is a C++ conversion
   to a class type, which C has no spelling for: the probe substitutes the C codec for it -- a
   correct cast, so what is compared is that the guard is the identity on one. *)
let fp8_guards =
  [ ("ocannl_single_to_fp8_uniform", "float"); ("ocannl_double_to_fp8_uniform", "double") ]

let hip_cast = "(__hip_fp8_e5m2)(y)"
let host_cast = "ocannl_shim_hip_fp8_cast(y)"

let count_occurrences ~sub s =
  let n = String.length sub in
  let rec go i acc =
    if i + n > String.length s then acc
    else if String.sub s i n = sub then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let replace_all ~sub ~by s =
  let n = String.length sub and b = Buffer.create (String.length s) in
  let rec go i =
    if i + n <= String.length s && String.sub s i n = sub then (
      Buffer.add_string b by;
      go (i + n))
    else if i < String.length s then (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

let host_table vendor =
  List.map
    (fun ((key, definition, deps) as entry) ->
      if List.mem_assoc key fp8_guards then
        (key, replace_all ~sub:hip_cast ~by:host_cast definition, deps)
      else entry)
    vendor.table

let roots vendor =
  probed vendor
  @ List.filter_map
      (fun (key, _) -> if List.mem key (keys vendor.table) then Some key else None)
      fp8_guards

(* The vendor intrinsics and types, over the C table's own definitions (emitted above them). *)
let shims =
  {|
#define __device__
#define __forceinline__ static inline
#define __constant__
#define __align__(n) __attribute__((aligned(n)))
#define __uint2float_rn(x) ((float)(unsigned int)(x))
#define __uint2double_rn(x) ((double)(unsigned int)(x))
static inline unsigned int __float_as_uint(float x) { unsigned int b; memcpy(&b, &x, sizeof b); return b; }
static inline float __uint_as_float(unsigned int b) { float x; memcpy(&x, &b, sizeof x); return x; }
static inline long long __double_as_longlong(double x) { long long b; memcpy(&b, &x, sizeof b); return b; }
static inline double __hiloint2double(int hi, int lo) {
  unsigned long long b = ((unsigned long long)(unsigned int)hi << 32) | (unsigned int)lo;
  double x; memcpy(&x, &b, sizeof x); return x;
}
static inline double ocannl_shim_swapped_hiloint2double(int hi, int lo) { return __hiloint2double(lo, hi); }
typedef HALF_T __half;
static inline __half __float2half(float f) { return FLOAT_TO_HALF(f); }
static inline unsigned short __half_as_ushort(__half h) { unsigned short b; memcpy(&b, &h, sizeof b); return b; }
typedef struct { uint16_t ocannl_bits; } __nv_bfloat16;
typedef struct { uint16_t ocannl_bits; } __hip_bfloat16;
static inline __nv_bfloat16 __float2bfloat16(float f) { __nv_bfloat16 r = { single_to_bfloat16(f) }; return r; }
static inline unsigned short __bfloat16_as_ushort(__nv_bfloat16 b) { return b.ocannl_bits; }
typedef unsigned char __nv_fp8_storage_t;
typedef struct { __nv_fp8_storage_t __x; } __nv_fp8_e5m2;
typedef unsigned char __hip_fp8_storage_t;
typedef struct { __hip_fp8_storage_t __x; } __hip_fp8_e5m2;
static inline __hip_fp8_e5m2 ocannl_shim_fp8_of_single(float x) { __hip_fp8_e5m2 r; r.__x = single_to_fp8(x); return r; }
static inline __hip_fp8_e5m2 ocannl_shim_fp8_of_double(double x) { __hip_fp8_e5m2 r; r.__x = double_to_fp8(x); return r; }
#define ocannl_shim_hip_fp8_cast(v) \
  _Generic((v), float: ocannl_shim_fp8_of_single, double: ocannl_shim_fp8_of_double)(v)
|}

(* HIP's bf16 conversion is the same function under its own type. *)
let hip_bf16_shims =
  {|
#define __float2bfloat16(f) ((__hip_bfloat16){ single_to_bfloat16(f) })
#define __bfloat16_as_ushort(b) ((b).ocannl_bits)
|}

let unhip_bf16_shims = "#undef __float2bfloat16\n#undef __bfloat16_as_ushort\n"

(* The negative controls' defects. *)
let broken_macros =
  {|
#define __hiloint2double(hi, lo) ocannl_shim_swapped_hiloint2double(hi, lo)
#define copysignf(m, s) (m)
#define copysign(m, s) (m)
|}

let unbroken_macros = "#undef __hiloint2double\n#undef copysignf\n#undef copysign\n"

(* One copy of a vendor table, every key renamed under [prefix] so that it can sit beside the C
   table and the other copies. *)
let vendor_copy vendor ~prefix ~broken =
  let table = host_table vendor in
  let renames =
    List.map (fun k -> Printf.sprintf "#define %s %s%s\n" k prefix k) (keys table)
    |> String.concat ""
  in
  let unrenames =
    List.map (fun k -> Printf.sprintf "#undef %s\n" k) (keys table) |> String.concat ""
  in
  String.concat "\n"
    ([
       renames;
       (if vendor.name = "hip" then hip_bf16_shims else "");
       (if broken then broken_macros else "");
     ]
    @ Ir.C_syntax.builtin_definitions table ~keys:(roots vendor)
    @ [
        (if broken then unbroken_macros else "");
        (if vendor.name = "hip" then unhip_bf16_shims else "");
        unrenames;
      ])

let impls =
  List.concat_map (fun v -> [ (v, v.name ^ "_", false); (v, v.name ^ "broken_", true) ]) vendors

let all_probed = List.sort_uniq compare (List.concat_map probed vendors)

(* Every probe has one C signature: four words of input, a lane, and the result's bytes out; it
   returns the result's size. *)
let probe_name ~prefix key = Printf.sprintf "probe_%s%s" prefix key

let probe ~vendor ~prefix key =
  let call =
    match shape key with
    | From_u4 -> Printf.sprintf "%s%s(x)" prefix key
    | Lane -> Printf.sprintf "%s%s(x, lane)" prefix key
    | From_u32 -> Printf.sprintf "%s%s(in[0])" prefix key
    | To_u4 _ -> Printf.sprintf "%s%s(a)" prefix key
    | Excluded _ | Unknown -> assert false
  in
  let arg =
    match shape key with
    | To_u4 prec ->
        let t = Option.get (param_type prec ~vendor) in
        Printf.sprintf "  %s a; memset(&a, 0, sizeof a); memcpy(&a, in, sizeof a);\n" t
    | _ -> ""
  in
  Printf.sprintf
    "int %s(const unsigned int *in, int lane, unsigned char *out) {\n\
    \  %suint4x32_t x; memcpy(x.v, in, sizeof x.v);\n\
     %s  __typeof__(%s) r = %s;\n\
    \  (void)x; (void)lane;\n\
    \  memcpy(out, &r, sizeof r);\n\
    \  return (int)sizeof r;\n\
     }\n"
    (probe_name ~prefix key) prefix arg call call

let fp8_probe ~prefix (key, typ) =
  let bits = if typ = "float" then "unsigned int" else "unsigned long long" in
  Printf.sprintf
    "unsigned int %s(%s bits) { %s x; memcpy(&x, &bits, sizeof x); return %s%s(x).__x; }\n"
    (probe_name ~prefix key) bits typ prefix key

let c_fp8_probes =
  {|
unsigned int probe_c_single_to_fp8(unsigned int bits) { float x; memcpy(&x, &bits, sizeof x); return single_to_fp8(x); }
unsigned int probe_c_double_to_fp8(unsigned long long bits) { double x; memcpy(&x, &bits, sizeof x); return double_to_fp8(x); }
|}

let probe_source =
  let c_roots =
    all_probed
    @ [ "HALF_T"; "FLOAT_TO_HALF"; "single_to_bfloat16"; "single_to_fp8"; "double_to_fp8" ]
  in
  String.concat "\n"
    ([ B.includes ]
    @ Ir.C_syntax.builtin_definitions B.builtins ~keys:c_roots
    @ [ shims ]
    @ List.map (fun (v, prefix, broken) -> vendor_copy v ~prefix ~broken) impls
    @ List.map (fun key -> probe ~vendor:None ~prefix:"" key) all_probed
    @ List.concat_map
        (fun (v, prefix, _) ->
          List.map (fun key -> probe ~vendor:(Some v.name) ~prefix key) (probed v)
          @ if v.name = "hip" then List.map (fp8_probe ~prefix) fp8_guards else [])
        impls
    @ [ c_fp8_probes ])

(* Compiles [probe_source] into a shared library and hands it to [f]. An implicit declaration is an
   error, as clang already makes it: it is what a definition missing from a dependency list looks
   like, and gcc before 14 would only warn and leave the symbol to fail at [dlopen]. *)
let with_probes f =
  let src = Filename.temp_file "ocannl_cuda_like_builtins_" ".c" in
  let dll = Filename.temp_file "ocannl_cuda_like_builtins_" (if Sys.win32 then ".dll" else ".so") in
  let log = Filename.temp_file "ocannl_cuda_like_builtins_" ".log" in
  let cleanup () =
    List.iter (fun p -> try Sys.remove p with Sys_error _ -> ()) [ src; dll; log ]
  in
  Fun.protect ~finally:cleanup (fun () ->
      let out = open_out_bin src in
      output_string out probe_source;
      close_out out;
      let command =
        Printf.sprintf "%s -O0 -Werror=implicit-function-declaration %s -o %s %s -lm >%s 2>&1"
          (Cc.compiler_command ()) (Lazy.force Cc.kernel_link_flags) (Filename.quote dll)
          (Filename.quote src) (Filename.quote log)
      in
      if Sys.command command <> 0 then (
        let ic = open_in_bin log in
        prerr_string (really_input_string ic (in_channel_length ic));
        close_in ic;
        failwith ("cannot compile the CUDA-like builtins probe: " ^ command));
      let lib = Dl.dlopen ~filename:dll ~flags:[ RTLD_NOW; RTLD_LOCAL ] in
      Fun.protect ~finally:(fun () -> Dl.dlclose ~handle:lib) (fun () -> f lib))

(* Draws: boundary words in every position, and a deterministic pseudo-random sample. *)
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
  List.init n (fun i -> [ edge i; edge i; edge i; edge i ])
  @ List.init n (fun i -> [ edge i; edge (i + 1); edge (i + 5); edge (i + 11) ])
  @ List.init 2048 (fun _ -> [ next (); next (); next (); next () ])

let () =
  case "shared helpers have probes" @@ fun () ->
  List.iter
    (fun v ->
      p_all
        (v.label
       ^ ": every function the table shares with the C table is probed or excluded with a reason")
        (shared v) ~f:(fun key ->
          let ok = shape key <> Unknown in
          if not ok then
            Printf.eprintf "(not part of the golden) %s: no probe shape for %s\n" v.label key;
          ok);
      Printf.eprintf "(not part of the golden) %s: %d helpers probed: %s\n" v.label
        (List.length (probed v))
        (String.concat " " (probed v));
      if v.name = "hip" then
        p_all
          "HIP: each fp8 guard is in the table and casts once, as the probe's substitution expects"
          fp8_guards ~f:(fun (key, _) ->
            match List.find_opt (fun (k, _, _) -> k = key) v.table with
            | Some (_, definition, _) -> count_occurrences ~sub:hip_cast definition = 1
            | None -> false))
    vendors

(* A probe of the compiled library, and one call of it: the result's size and bytes. *)
let bind lib name =
  let open Ctypes in
  Foreign.foreign ~from:lib name (ptr uint32_t @-> int @-> ptr uchar @-> returning int)

let run fn words lane =
  let open Ctypes in
  let input = CArray.of_list uint32_t (List.map Unsigned.UInt32.of_int words) in
  let out = CArray.make uchar ~initial:Unsigned.UChar.zero 64 in
  let size = fn (CArray.start input) lane (CArray.start out) in
  (size, List.init size (fun i -> Unsigned.UChar.to_int (CArray.get out i)))

(* A [_lane] helper's lanes are its [_vec] pack's: the pack's size over the lane's. *)
let lanes lib key =
  match shape key with
  | Lane ->
      let stem = Option.get (chop_suffix key ~suffix:"_uniform_lane") in
      let vec_size, _ =
        run (bind lib (probe_name ~prefix:"" (stem ^ "_uniform_vec"))) [ 0; 0; 0; 0 ] 0
      in
      let lane_size, _ = run (bind lib (probe_name ~prefix:"" key)) [ 0; 0; 0; 0 ] 0 in
      List.init (vec_size / lane_size) Fun.id
  | _ -> [ 0 ]

let () =
  case "conversion helpers against the C table" @@ fun () ->
  with_probes (fun lib ->
      let bind = bind lib and lanes = lanes lib in
      let c_probe = Hashtbl.create 64 in
      List.iter
        (fun key -> Hashtbl.replace c_probe key (bind (probe_name ~prefix:"" key)))
        all_probed;
      let agrees ~log ~label ~prefix key =
        let mine = bind (probe_name ~prefix key) and theirs = Hashtbl.find c_probe key in
        let lanes = lanes key in
        List.for_all
          (fun words ->
            List.for_all
              (fun lane ->
                let got = run mine words lane and want = run theirs words lane in
                let ok = got = want in
                (if log && not ok then
                   let hex (size, bytes) =
                     Printf.sprintf "%d bytes [%s]" size
                       (String.concat " " (List.map (Printf.sprintf "%02x") bytes))
                   in
                   Printf.eprintf "(not part of the golden) %s %s lane %d of [%s]: got %s want %s\n"
                     label key lane
                     (String.concat " " (List.map (Printf.sprintf "%08x") words))
                     (hex got) (hex want));
                ok)
              lanes)
          draws
      in
      List.iter
        (fun (v, prefix, broken) ->
          if broken then
            p_exists
              (v.label
             ^ ": a copy whose __hiloint2double swaps its words differs from the C table somewhere"
              ) (probed v) ~f:(fun key -> not (agrees ~log:false ~label:v.label ~prefix key))
          else
            p_all
              (v.label
             ^ ": every probed conversion helper returns the C one's bytes, for every draw and lane"
              )
              (probed v)
              ~f:(agrees ~log:true ~label:v.label ~prefix))
        impls)

(* The narrow uniform draws below 1 (gh-ocannl-1434, gh-ocannl-1486). A half or bf16 draw used to be
   the f32 draw of its bits ROUNDED to the narrow format, which is exactly 1.0 for every f32 draw
   within half a narrow ulp of 1 (1 in 4096 half draws, 1 in 512 bf16 ones); an fp8 draw was the raw
   bits as an e5m2 pattern, negatives, infinities and NaNs included. Each is now the f32 draw
   truncated toward zero. The oracle reaches that value by another route than the helpers' mantissa
   mask or e5m2 field arithmetic: the host codec's round-to-nearest, stepped one encoding down when
   it landed above the draw -- the greatest value of the format not above the f32 draw. [bytes] is
   an element's width, which also fixes how many bits a packed lane draws from. *)
type narrow = { widen : int -> float; round : float -> int; bytes : int }

let half = { widen = Ir.Ops.half_to_single; round = Ir.Ops.single_to_half; bytes = 2 }
let bf16 = { widen = Ir.Ops.bfloat16_to_single; round = Ir.Ops.single_to_bfloat16; bytes = 2 }
let fp8 = { widen = Ir.Ops.fp8_to_single; round = Ir.Ops.single_to_fp8; bytes = 1 }
let contains ~sub s = count_occurrences ~sub s > 0

(* Which narrow format a draw helper produces, read off its name as the emission spells it. *)
let narrow_of key =
  match shape key with
  | From_u4 | Lane | From_u32 ->
      if contains ~sub:"_to_half_uniform" key then Some half
      else if contains ~sub:"_to_bfloat16_uniform" key then Some bf16
      else if contains ~sub:"_to_fp8_uniform" key then Some fp8
      else None
  | _ -> None

(* The f32 draw of a word, [uint32_to_single_uniform]: its top 24 bits scaled by 2^-24. *)
let f32_draw word = Float.of_int ((word land 0xFFFFFFFF) lsr 8) *. 0x1p-24

let floor_in n f =
  let r = n.round f in
  if n.widen r > f then r - 1 else r

(* The encodings a probe's result bytes hold, little-endian elements of the format's width. *)
let elements n (size, bytes) =
  let b = Array.of_list bytes in
  List.init (size / n.bytes) (fun j ->
      List.fold_left
        (fun acc k -> acc lor (b.((n.bytes * j) + k) lsl (8 * k)))
        0 (List.init n.bytes Fun.id))

(* The word whose f32 draw element [j] of a helper's result narrows: a scalar form draws from the
   first word, a packed or lane form from the [j]-th element-wide slice of the words, lowest bits
   first -- 16 bits for half and bf16 (the low then high half of word [j / 2]), 8 for fp8 (byte [j
   mod 4] of word [j / 4]) -- moved to the top of a word: the slice's draw scaled by its width. *)
let source_word n key words j =
  match shape key with
  | From_u32 -> List.hd words
  | From_u4 when not (String.ends_with ~suffix:"_vec" key) -> List.hd words
  | _ ->
      let bits = 8 * n.bytes in
      let per_word = 32 / bits in
      let w = List.nth words (j / per_word) in
      let slice = (w lsr (bits * (j mod per_word))) land ((1 lsl bits) - 1) in
      slice lsl (32 - bits)

let narrow_keys = List.filter (fun k -> Option.is_some (narrow_of k)) all_probed
let all_ones = [ 0xFFFFFFFF; 0xFFFFFFFF; 0xFFFFFFFF; 0xFFFFFFFF ]

let () =
  case "narrow uniform draws stay below 1" @@ fun () ->
  p_all
    "every half, bf16 and fp8 draw the C table defines is probed in the C and both vendor copies"
    (List.filter (fun k -> Option.is_some (narrow_of k)) c_keys)
    ~f:(fun key -> List.for_all (fun v -> List.mem key (probed v)) vendors);
  with_probes (fun lib ->
      let bound = Hashtbl.create 64 in
      let bind name =
        match Hashtbl.find_opt bound name with
        | Some fn -> fn
        | None ->
            let fn = bind lib name in
            Hashtbl.replace bound name fn;
            fn
      in
      let lanes = lanes lib in
      (* Every top-16-bit pattern under all-ones low bits besides the shared draws, for the word
         helpers: that walks every 16-bit lane value and every scalar draw's top binades. The words
         around 2^18, under which the f32 draw is below e5m2's smallest normal, are added with zero
         low bits, the other side of that boundary. *)
      let words_for key =
        match shape key with
        | From_u32 ->
            List.init 65536 (fun v -> [ (v lsl 16) lor 0xFFFF; 0; 0; 0 ])
            @ List.map
                (fun w -> [ w; 0; 0; 0 ])
                [ 0x10000; 0x20000; 0x30000; 0x3FF00; 0x40000; 0x40100; 0x50000; 0x80000 ]
            @ draws
        | _ -> draws
      in
      let elements_of prefix key words =
        let n = Option.get (narrow_of key) in
        List.concat_map
          (fun lane ->
            let got = elements n (run (bind (probe_name ~prefix key)) words lane) in
            match shape key with
            | Lane -> [ (lane, List.hd got) ]
            | _ -> List.mapi (fun j e -> (j, e)) got)
          (lanes key)
      in
      List.iter
        (fun (label, prefix, keys) ->
          p_all
            (label
           ^ ": every half, bf16 and fp8 uniform form draws, at every probed input and lane, the \
              greatest value of its format not above the f32 draw of the same bits") keys
            ~f:(fun key ->
              let n = Option.get (narrow_of key) in
              List.for_all
                (fun words ->
                  List.for_all
                    (fun (j, e) ->
                      let want = floor_in n (f32_draw (source_word n key words j)) in
                      let ok = e = want in
                      if not ok then
                        Printf.eprintf
                          "(not part of the golden) %s %s element %d of [%s]: got 0x%04x want 0x%04x\n"
                          label key j
                          (String.concat " " (List.map (Printf.sprintf "%08x") words))
                          e want;
                      ok)
                    (elements_of prefix key words))
                (words_for key));
          p_all
            (label
           ^ ": at the all-ones input every half, bf16 and fp8 uniform form draws below 1 in every \
              lane") keys ~f:(fun key ->
              let n = Option.get (narrow_of key) in
              List.for_all (fun (_, e) -> n.widen e < 1.) (elements_of prefix key all_ones)))
        ([ ("C", "", narrow_keys) ]
        @ List.map
            (fun v ->
              (v.label, v.name ^ "_", List.filter (fun k -> List.mem k narrow_keys) (probed v)))
            vendors))

(* Float bit patterns: every exponent of both signs under boundary and sampled mantissas, and the
   neighbourhood of the guard's threshold, half the smallest e5m2 subnormal. *)
let f32_patterns =
  let mantissas =
    [ 0; 1; 2; 3; 0x1FFFFF; 0x200000; 0x200001; 0x3FFFFF; 0x400000; 0x7FFFFE; 0x7FFFFF; 0x2AAAAA ]
  in
  let threshold = Int32.to_int (Int32.bits_of_float 0x1p-17) in
  List.concat_map
    (fun sign ->
      List.concat_map
        (fun e -> List.map (fun m -> sign lor (e lsl 23) lor m) mantissas)
        (List.init 256 Fun.id)
      @ List.init 129 (fun k -> sign lor (threshold + k - 64)))
    [ 0; 0x80000000 ]

let f64_patterns =
  let mantissas =
    [
      0L;
      1L;
      2L;
      0x3FFFFFFFFFFFFL;
      0x4000000000000L;
      0x4000000000001L;
      0x7FFFFFFFFFFFFL;
      0x8000000000000L;
      0xFFFFFFFFFFFFFL;
    ]
  in
  let threshold = Int64.bits_of_float 0x1p-17 in
  List.concat_map
    (fun sign ->
      List.concat_map
        (fun e ->
          List.map (fun m -> Int64.(logor sign (logor (shift_left (of_int e) 52) m))) mantissas)
        (List.init 2048 Fun.id)
      @ List.init 129 (fun k -> Int64.(logor sign (add threshold (of_int (k - 64))))))
    [ 0L; Int64.min_int ]

let () =
  case "HIP fp8 guards against the C codec" @@ fun () ->
  with_probes (fun lib ->
      let open Ctypes in
      let f32 name = Foreign.foreign ~from:lib name (uint32_t @-> returning uint32_t) in
      let f64 name = Foreign.foreign ~from:lib name (uint64_t @-> returning uint32_t) in
      let c32 = f32 "probe_c_single_to_fp8" and c64 = f64 "probe_c_double_to_fp8" in
      let agrees32 guard bits =
        let b = Unsigned.UInt32.of_int bits in
        Unsigned.UInt32.equal (guard b) (c32 b)
      in
      let agrees64 guard bits =
        let b = Unsigned.UInt64.of_int64 bits in
        Unsigned.UInt32.equal (guard b) (c64 b)
      in
      let single, double = ("ocannl_single_to_fp8_uniform", "ocannl_double_to_fp8_uniform") in
      let guard32 = f32 (probe_name ~prefix:"hip_" single)
      and guard64 = f64 (probe_name ~prefix:"hip_" double)
      and broken32 = f32 (probe_name ~prefix:"hipbroken_" single)
      and broken64 = f64 (probe_name ~prefix:"hipbroken_" double) in
      p_all "HIP: the f32 fp8 guard over a correct cast is the C codec, bitwise" f32_patterns
        ~f:(agrees32 guard32);
      p_all "HIP: the f64 fp8 guard over a correct cast is the C codec, bitwise" f64_patterns
        ~f:(agrees64 guard64);
      p_exists "HIP: an f32 guard whose zero drops its sign differs from the C codec" f32_patterns
        ~f:(fun bits -> not (agrees32 broken32 bits));
      p_exists "HIP: an f64 guard whose zero drops its sign differs from the C codec" f64_patterns
        ~f:(fun bits -> not (agrees64 broken64 bits)))
