(* The fp8 (e5m2) narrowing codec is ONE rounding, on the host and in every backend's kernels
   (gh-ocannl-646).

   A tensor's fp8 cells get written from two places: the host, through [Ndarray]'s
   [Ops.single_to_fp8] (the C stub compiled from builtins.c), and a kernel, through whatever the
   backend emits — the same software codec on cc and Metal, the native [__nv_fp8_e5m2] cast on CUDA,
   and on HIP [__hip_fp8_e5m2] behind a guard that pre-rounds the underflow window ROCm gets wrong
   (gh-ocannl-647). Nothing makes those agree by construction, so this compares them directly: the
   same values narrowed each way, read back and compared.

   The values are the ones where a codec has a decision to make, because everywhere else agreement
   is uninformative — ties in both directions (round-to-nearest-EVEN, not away from zero), the
   subnormal range (rounded into, not flushed — an e5m2 subnormal is unreachable for a codec that
   flushes), the sign of a zero, and overflow of a finite value (saturating to the largest finite,
   not going to infinity). Those four were the software codec's arbitrary choices and the native
   types' considered ones; they are the native types' now.

   Two inputs are deliberately NOT pinned, because the vendors disagree with each other and OCANNL
   does not emit the conversion itself on those backends: an already-INFINITE input, which CUDA
   saturates to the largest finite while HIP keeps it infinite, and the SIGN of a NaN, which CUDA
   drops and HIP keeps. Each is asserted below only up to the property all four agree on. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let codegen_capabilities = Context.codegen_capabilities (Context.auto ())
let skipped = Verdict.skipped ~backend:backend_name

(* Narrow on the DEVICE. [*. 1.] is the exact identity on every value below — signed zeros,
   infinities and NaNs included — so the only thing between the source and the fp8 buffer is the
   backend's float-to-fp8 conversion. Reading back applies the HOST's widening, which is exact and
   agrees everywhere (all 256 codes, checked against both GPUs), so a difference in what comes back
   is a difference in the narrowing.

   The source is MATERIALIZED, and that is the whole test. Left to virtualize, its cells inline into
   the consumer as literals, the conversion becomes a compile-time constant expression, and the
   backend compiler folds it on the HOST — so the test would compare the host codec against the host
   half of the vendor header and never emit a device conversion at all. Verified rather than
   assumed: with the source materialized, HIP's underflow defect (gh-ocannl-647) is reproducible
   through this test and its guard is observable; without it, the leg passes in both regimes and
   pins nothing. *)
let narrow_on_device ?(src_prec = Ir.Ops.single) values =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let src =
    TDSL.ndarray values ~label:[ "codec_src" ]
      ~output_dims:[ Array.length values ]
      ~top_down_prec:false ()
  in
  (* The source stays f32, EXPLICITLY. Precision inference otherwise flows the destination's fp8
     backwards into it, and then the kernel is [dst[i] = src[i]] over two fp8 buffers — a byte copy,
     with the narrowing having happened on the host when the source was filled. The test then
     compares the host codec with itself and passes on every backend while emitting no conversion at
     all. [~top_down_prec:false] and the explicit [update_prec] together are what make the emitted
     statement a conversion; the generated source is the check ([dst[i] = (fp8)(src[i])], src
     declared [float *]). *)
  Ir.Tnode.update_prec src.Tensor.value src_prec;
  Train.set_materialized src.Tensor.value;
  let dst = TDSL.O.( *. ) src (TDSL.number 1.0) in
  Ir.Tnode.update_prec dst.Tensor.value Ir.Ops.fp8;
  Train.set_materialized dst.Tensor.value;
  let ctx = Train.forward_once ctx dst in
  Context.get_values ctx dst.Tensor.value

(* The same values narrowed on the host: [Ndarray.set_from_float] runs [Ops.single_to_fp8]. *)
let narrow_on_host values =
  Array.map values ~f:(fun v -> Ir.Ops.fp8_to_single (Ir.Ops.single_to_fp8 v))

let two_pow n = Float.(2. ** of_int n)

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

let decisive =
  [|
    (* Ties between adjacent e5m2 values: to even, not away from zero. *)
    1.125;
    1.375;
    1.625;
    2.25;
    -1.125;
    -1.625;
    0.140625;
    (* The subnormal range: a flushing codec returns zero for all but the first. *)
    two_pow (-14);
    two_pow (-15);
    two_pow (-16);
    2.5 *. two_pow (-16);
    1e-5;
    -1e-5;
    (* The tie between zero and the smallest subnormal, which goes to zero (even). *)
    two_pow (-17);
    (* Signed zeros. *)
    0.0;
    -0.0;
    (* Overflow of a FINITE value: saturates to the largest finite. *)
    57344.0;
    60000.0;
    61440.0;
    65504.0;
    1e30;
    -1e30;
    (* Ordinary values, as controls. *)
    1.0;
    -2.0;
    0.5;
    100.0;
    -0.001;
  |]

let () =
  let host = narrow_on_host decisive in
  let device = narrow_on_device decisive in
  (* The values themselves, not just their agreement: on cc and Metal both sides of the comparison
     above run the SAME software codec, so a tie rule changed in both would keep the comparison
     green. The golden is what pins the rule, and it is backend-uniform — every entry here was
     checked against [__nv_fp8_e5m2] and [__hip_fp8_e5m2] on real hardware. Hex float notation keeps
     it platform-independent (AGENTS.md). *)
  Stdio.printf "narrowed (input -> e5m2):\n";
  Array.iteri decisive ~f:(fun i v -> Stdio.printf "  %h -> %h\n" v device.(i));
  let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  let disagreements =
    Array.filter_mapi decisive ~f:(fun i v ->
        if same host.(i) device.(i) then None
        else Some (Printf.sprintf "%h: host %h, device %h" v host.(i) device.(i)))
  in
  Array.iter disagreements ~f:(fun d -> Stdio.eprintf "fp8 narrowing disagreement: %s\n" d);
  p_empty "the device narrows every decisive value exactly as the host does"
    ~over:(Array.to_list decisive) (Array.to_list disagreements);

  (* Reachability, so that "no disagreement" cannot hold vacuously: the flushing codec this replaced
     could never emit the smallest subnormal from a narrowing, whichever side ran it. *)
  p "narrowing reaches the smallest e5m2 subnormal"
    (Array.existsi decisive ~f:(fun i v -> Float.(v > 0.) && Float.(device.(i) = two_pow (-16))));
  p "a finite input above the range saturates rather than going infinite"
    (Array.for_alli decisive ~f:(fun i v -> (not (Float.is_finite v)) || Float.is_finite device.(i)));
  p "the sign of a zero survives narrowing"
    (Array.existsi decisive ~f:(fun i v ->
         Float.(v = 0.) && Float.ieee_negative v && Float.ieee_negative device.(i)));

  (* Infinities and NaNs: only the part all four backends agree on. *)
  let sdev = narrow_on_device [| Float.infinity; Float.neg_infinity; Float.nan |] in
  p "an infinite input narrows to the largest finite magnitude or beyond, keeping its sign"
    (Float.(sdev.(0) >= 57344.) && Float.(sdev.(1) <= -57344.));
  p "a NaN input narrows to a NaN" (Float.is_nan sdev.(2));

  (* f64 -> fp8, which must not round twice (gh-ocannl-648). Each value sits one f64 ulp off an f32
     tie, so a conversion that goes through f32 first lands ON the tie and then breaks it by the
     wrong rule; a one-step conversion sees which side of the midpoint the double is on. The GPU fp8
     types convert straight from the double, so this is the same "match the hardware" question as
     the tie rule itself. Skipped on Metal, which has no f64 storage at all. *)
  let f64_claim = "narrowing f64 to fp8 rounds once, not twice" in
  let eps = Float.(2. ** -40.) in
  let f64_vals = [| 1.125 +. eps; 1.125 -. eps; 1.625 +. eps; 1.625 -. eps |] in
  let f64_host = Array.map f64_vals ~f:(fun v -> Ir.Ops.fp8_to_single (Ir.Ops.double_to_fp8 v)) in
  (* The dump is of the HOST codec, which runs everywhere, so the golden stays backend-uniform:
     [Verdict.skipped] keeps the CLAIM line uniform, it does not cover descriptive output beside it,
     and a dump printed only off Metal is a golden diff on Metal. What the device computed is what
     the claim below compares against these same values, so nothing is lost by printing the host
     side. *)
  Array.iteri f64_vals ~f:(fun i v -> Stdio.printf "  f64 %h -> %h\n" v f64_host.(i));
  (if not codegen_capabilities.Ir.Backend_intf.supports_f64 then skipped f64_claim
   else
     let dev = narrow_on_device ~src_prec:Ir.Ops.double f64_vals in
     p_all2 f64_claim dev f64_host ~f:(fun a b -> Float.equal a b));

  (* The same underflow, but reached through fp8 ARITHMETIC rather than a conversion: an fp8
     operator computes in f32 and narrows its RESULT, which is a second narrowing site, and one a
     guard on the conversions alone would miss (Codex P2 on PR #372). [(2^-16) **. 5] is 2^-80,
     squarely inside ROCm's broken window, and reachable from fp8 operands — a product of two cannot
     get below 2^-32, so the power is what gets there. *)
  let pow_narrowed =
    Tensor.unsafe_reinitialize ();
    let ctx = Context.auto () in
    let base = TDSL.ndarray [| two_pow (-16) |] ~label:[ "codec_pow" ] ~output_dims:[ 1 ] () in
    (* The exponent is a materialized TENSOR, not the literal [**. 5.0] takes. With a literal the
       optimizer expands the power into a multiplication chain, and on a backend that narrows every
       intermediate to fp8 (HIP: its arithmetic renders AT fp8, unlike cc's f32 compute seam) the
       first product already underflows to zero — the chain never reaches the window and the leg
       passes for the wrong reason. One [powf] over runtime operands is what puts a single
       operator's f32 result at 2^-80. *)
    let expo = TDSL.ndarray [| 5.0 |] ~label:[ "codec_exp" ] ~output_dims:[ 1 ] () in
    List.iter [ base; expo ] ~f:(fun t ->
        Ir.Tnode.update_prec t.Tensor.value Ir.Ops.fp8;
        Train.set_materialized t.Tensor.value);
    (* Named, so its debug artifacts do not collide with the narrowing routine's: a same-named
       routine silently overwrites the earlier one's .c/.ll (AGENTS.md), and reading the emitted
       kernel is how this leg was checked to be non-vacuous. *)
    let%cd fp8pow_asn = { fp8pow } =: base ** expo in
    Ir.Tnode.update_prec fp8pow.Tensor.value Ir.Ops.fp8;
    Train.set_materialized fp8pow.Tensor.value;
    let ctx, routine = Context.compile ctx (named "fp8pow" fp8pow_asn) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx fp8pow.Tensor.value).(0)
  in
  Stdio.printf "fp8 arithmetic underflow: (2^-16) ** 5 -> %h\n" pow_narrowed;

  (* Magnitudes far below the smallest subnormal must vanish. This is an unconditional claim on
     every backend, HIP included: ROCm's own float-to-fp8 conversion fails it (gh-ocannl-647, an
     out-of-range shift returning values as large as 2^-14, reported upstream as
     https://github.com/ROCm/rocm-systems/issues/10591), which is exactly why the HIP backend
     narrows through a guarded helper rather than the platform's cast. Nothing configurable is left
     here to skip on -- the leg failing on HIP means the guard stopped being emitted. *)
  let claim = "magnitudes far below the smallest subnormal narrow to zero" in
  let arith_claim = "an fp8 operator's underflowing result narrows to zero too" in
  let tiny = [| 4.1359e-25; 8.27e-25; 1.65e-24; 3.31e-24; -4.1359e-25 |] in
  let tdev = narrow_on_device tiny in
  p_all claim (Array.to_list tdev) ~f:(fun x -> Float.(x = 0.));
  p arith_claim Float.(pow_narrowed = 0.)
