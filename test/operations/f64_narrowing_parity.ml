(* gh-ocannl-2075: narrowing an f64 to half or bfloat16 rounds ONCE, on the host and in every
   backend's kernels, as gh-ocannl-648 settled for fp8.

   Going through f32 first rounds twice: a double just off an f32 tie lands ON the tie, and the
   second rounding then breaks it by a rule the first has already made wrong. CUDA's [__double2half]
   and its bfloat16 conversion take the double in one step; cc and HIP used to cast to [float]
   first, and so did the host (OCaml's own float16 store narrows through f32, and so did the bf16
   stub), so one tensor read differently depending on who wrote it.

   The corpus is the set of places where a narrowing has a decision to make: for every pair of
   adjacent finite codes, the exact midpoint (a tie, to even), its two f64 neighbours (one f64 ulp
   is far under half an f32 ulp, so these are exactly the inputs f32 rounds ONTO the tie), and the
   lower code itself; the overflow threshold and its neighbours; both signs; plus zeros, f64
   subnormals and extremes, infinities and a NaN. The expectation is an ORACLE written here, not a
   second codec: the format's decode table plus "a code owns the interval between the midpoints to
   its neighbours, ties to the even code, infinity taking part as the code after the largest finite"
   -- the half and bfloat16 midpoints are exact doubles, so every comparison is exact.

   Two writers are checked against it: the device (a cast of a materialized f64 source, which every
   backend with f64 renders as its own conversion -- the emitted text is checked too, so a host fold
   cannot pass for it) and the host ([Ndarray]'s per-cell and bulk writers). Metal has no f64, so
   its device leg skips; its host leg runs. Counts and claims go to stdout; every value printed is
   the oracle's, exact by construction. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Test_utils.Generated.init ~backend_name
let supports_f64 = (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.supports_f64

type format = {
  name : string;
  prec : Ir.Ops.prec;
  max_code : int;  (** The largest finite positive code. *)
  inf_code : int;
  decode : int -> float;  (** A non-negative finite code's exact value. *)
  bits_of_value : float -> int;  (** The code of a value the format holds exactly. *)
  cc_spelling : string;
  hip_spelling : string;
  cuda_spelling : string;
}

let two_pow n = Float.ldexp 1. n

let half =
  {
    name = "half";
    prec = Ir.Ops.half;
    max_code = 0x7BFF;
    inf_code = 0x7C00;
    decode =
      (fun c ->
        let e = c lsr 10 and m = c land 0x3FF in
        if e = 0 then Float.of_int m *. two_pow (-24)
        else Float.of_int (1024 + m) *. two_pow (e - 25));
    bits_of_value = Ir.Ops.single_to_half;
    cc_spelling = "double_to_half(";
    hip_spelling = "ocannl_double_to_half(";
    cuda_spelling = "__double2half(";
  }

let bfloat16 =
  {
    name = "bfloat16";
    prec = Ir.Ops.bfloat16;
    max_code = 0x7F7F;
    inf_code = 0x7F80;
    decode = (fun c -> Int32.float_of_bits (Int32.of_int_exn (c lsl 16)));
    bits_of_value = Ir.Ops.single_to_bfloat16;
    cc_spelling = "double_to_bfloat16(";
    hip_spelling = "ocannl_double_to_bfloat16(";
    (* CUDA's [__nv_bfloat16] takes a double through its own one-step constructor; there is no
       spelling of ours to pin. *)
    cuda_spelling = "";
  }

let sign_bit = 0x8000

(* The oracle. [x] is finite or infinite, not NaN. *)
let oracle fmt x =
  let sign = if Float.ieee_negative x then sign_bit else 0 in
  let a = Float.abs x in
  let code =
    if Float.is_inf a then fmt.inf_code
    else
      (* The largest code whose value does not exceed [a]: [decode] is increasing. *)
      let rec search lo hi =
        if lo >= hi then lo
        else
          let mid = (lo + hi + 1) / 2 in
          if Float.(fmt.decode mid <= a) then search mid hi else search lo (mid - 1)
      in
      let c = search 0 fmt.max_code in
      let lower = fmt.decode c in
      let upper_code, upper =
        if c = fmt.max_code then (fmt.inf_code, lower +. (lower -. fmt.decode (c - 1)))
        else (c + 1, fmt.decode (c + 1))
      in
      let mid = (lower +. upper) /. 2. in
      if Float.(a < mid) then c
      else if Float.(a > mid) then upper_code
      else if c land 1 = 0 then c
      else upper_code
  in
  code lor sign

(* What a narrowing through f32 gives: the double rounded to f32 first. *)
let through_f32 fmt x = oracle fmt (Int32.float_of_bits (Int32.bits_of_float x))

let corpus fmt =
  let positive = ref [] in
  let add x = positive := x :: !positive in
  for c = 0 to fmt.max_code do
    let lower = fmt.decode c in
    let upper =
      if c = fmt.max_code then lower +. (lower -. fmt.decode (c - 1)) else fmt.decode (c + 1)
    in
    let mid = (lower +. upper) /. 2. in
    add lower;
    add mid;
    add (Float.one_ulp `Down mid);
    add (Float.one_ulp `Up mid)
  done;
  List.iter
    [
      two_pow (-1074);
      two_pow (-1022);
      two_pow (-200);
      two_pow (-150);
      1e300;
      Float.max_finite_value;
    ]
    ~f:add;
  let positive = List.rev !positive in
  Array.of_list (positive @ List.map positive ~f:Float.neg @ [ Float.infinity; Float.neg_infinity ])

(* The cells as codes; a NaN cell answers [None]. *)
let codes_of fmt values =
  Array.map values ~f:(fun v -> if Float.is_nan v then None else Some (fmt.bits_of_value v))

let narrow_on_device fmt values =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Specified at f64 and materialized, both: precision inference would otherwise flow the
     destination's precision back into the source (a copy, narrowed on the host when the source was
     filled), and a virtual source inlines as literals the backend compiler folds on the host
     (docs/agent-notes/backend-precision-and-simd.md, "A test that means to exercise a
     NARROWING"). *)
  let src =
    NTDSL.init ~l:"f64_src" ~prec:Ir.Ops.double
      ~o:[ Array.length values ]
      ~f:(fun idcs -> values.(idcs.(0)))
      ()
  in
  Train.set_materialized src.Tensor.value;
  let dst = Operation.cast ~grad_spec:Tensor.Prohibit_grad ~label:[ "narrowed" ] src () in
  Ir.Tnode.update_prec dst.Tensor.value fmt.prec;
  Train.set_materialized dst.Tensor.value;
  let routine = "cast_narrowed_fwd" in
  Test_utils.Generated.arm routine;
  let ctx = Train.forward_once ctx dst in
  (Context.get_values ctx dst.Tensor.value, Test_utils.Generated.read routine)

(* The host's two writers into a narrow buffer: per cell, and the bulk writer behind
   [Context.set_values]. *)
let narrow_on_host fmt values =
  let n = Array.length values in
  let fresh () = Ir.Ndarray.create_array ~debug:"f64_host" fmt.prec ~dims:[| n |] ~padding:None in
  let per_cell = fresh () in
  Array.iteri values ~f:(fun i v -> Ir.Ndarray.set_from_float per_cell [| i |] v);
  let bulk = fresh () in
  Ir.Ndarray.set_flat_values bulk values;
  ( Array.init n ~f:(fun i -> Ir.Ndarray.get_as_float per_cell [| i |]),
    Ir.Ndarray.retrieve_flat_values bulk )

let check fmt =
  let values = corpus fmt in
  let want = Array.map values ~f:(oracle fmt) in
  let n = Array.length values in
  Stdio.printf "%s: %d f64 inputs\n" fmt.name n;
  (* The corpus decides the question: on some of it the double rounding gives another code. *)
  let discriminating = Array.count values ~f:(fun x -> oracle fmt x <> through_f32 fmt x) in
  Stdio.printf "%s: inputs where rounding through f32 gives another code: %d\n" fmt.name
    discriminating;
  pf "%s: the corpus holds inputs that rounding through f32 gets wrong" fmt.name (discriminating > 0);
  let x = 1. +. two_pow (-Ir.Ops.significand_bits fmt.prec) +. two_pow (-40) in
  Stdio.printf "%s: the oracle narrows %h to %h\n" fmt.name x (fmt.decode (oracle fmt x));
  (* Each writer's cells as codes, against the oracle's, cell by cell: [p_all2] refuses an empty or
     short readback. The first few misses also go to stderr, with their inputs. *)
  let want_codes = Array.map want ~f:Option.some in
  let agrees_with_oracle claim label got =
    let got = codes_of fmt got in
    let wrong =
      Array.filter_mapi got ~f:(fun i g ->
          if i < n && Option.equal Int.equal g want_codes.(i) then None else Some i)
    in
    Array.iteri wrong ~f:(fun k i ->
        if k < 8 then
          Stdio.eprintf "%s, %s: %h narrowed to %s, want 0x%04x (not part of the golden)\n" fmt.name
            label values.(i)
            (match got.(i) with Some g -> Printf.sprintf "0x%04x" g | None -> "NaN")
            want.(i));
    Stdio.eprintf "%s, %s: %d of %d inputs miss (not part of the golden)\n" fmt.name label
      (Array.length wrong) n;
    p_all2 ~min:n claim got want_codes ~f:(Option.equal Int.equal)
  in
  let host_cell, host_bulk = narrow_on_host fmt values in
  agrees_with_oracle
    (fmt.name ^ ": the host's per-cell writer narrows every input to the oracle's code")
    "host per-cell" host_cell;
  agrees_with_oracle
    (fmt.name ^ ": the host's bulk writer narrows every input to the oracle's code")
    "host bulk" host_bulk;
  let leg = Verdict.leg ~when_:supports_f64 ~on:backend_name () in
  (* Every device run ahead of the claims, not inside a leg's: a run (its generated-source read) can
     make claims of its own, and a leg's evaluation makes exactly one. *)
  let device, nan_device =
    if supports_f64 then
      let device = narrow_on_device fmt values in
      (device, fst (narrow_on_device fmt [| Float.nan; Float.neg Float.nan |]))
    else (([||], ""), [||])
  in
  in_legf ~leg "%s: the device narrows every input to the oracle's code" fmt.name (fun claim ->
      agrees_with_oracle claim "device" (fst device));
  in_legf ~leg
    "%s: the device's narrowing reaches every finite code of both signs and both infinities"
    fmt.name (fun claim ->
      let reached = Hash_set.create (module Int) in
      Array.iter (codes_of fmt (fst device)) ~f:(Option.iter ~f:(Hash_set.add reached));
      p claim (Hash_set.length reached = 2 * (fmt.max_code + 2)));
  let spelling =
    match backend_name with
    | "cc" | "multidev_cc" -> fmt.cc_spelling
    | "hip" -> fmt.hip_spelling
    | "cuda" -> fmt.cuda_spelling
    | _ -> ""
  in
  let spelled =
    Verdict.sub_leg leg ~when_:(lazy (not (String.is_empty spelling))) ~on:backend_name
  in
  in_legf ~leg:spelled
    "%s: the kernel narrows the f64 source with the backend's one-step conversion" fmt.name
    (fun claim -> p claim (String.is_substring (snd device) ~substring:spelling));
  (* A NaN stays a NaN; its payload and sign are the vendors' to choose. *)
  in_legf ~leg "%s: a NaN narrows to a NaN" fmt.name (fun claim ->
      p claim ((not (Array.is_empty nan_device)) && Array.for_all nan_device ~f:Float.is_nan))

let () =
  check half;
  check bfloat16
