(* Every emitted float constant is a floating literal that parses back to exactly the host value
   (gh-ocannl-623).

   [C_syntax.c_float_literal] renders each [Low_level.Constant] into the kernel source. Three
   properties have to hold of the token it produces, and [%.16g] alone gets all three wrong:

   - It has to be a FLOATING literal. A value with no fractional part printed by [%.16g] has no
   radix point, so it is a C INTEGER literal that the surrounding cast happens to convert. That is
   value-preserving in today's cast contexts for every value but one: [-0.] printed as ["-0"] is the
   integer zero, hence [+0.0] once cast — a live cross-backend data corruption fixed for that single
   value by gh-ocannl-615. The rest of the class is latent, not inert (integer division, integer
   promotion, a digit run outgrowing [long long]), which is what this test closes. - It has to
   ROUND-TRIP. Sixteen significant digits do not recover every double: [0.1 +. 0.2] prints as
   ["0.3"], a different double, and so does [max_float]. Arbitrary host values reach this printer
   (hosted constant inits inline them as scalar stores), so this is a live value-changing bug rather
   than a hypothetical one, and it is invisible to a tolerance-based comparison — hence the bitwise
   checks below. - The specials have to be SPELLED. [INFINITY] / [(-INFINITY)] / [NAN] are C99
   macros MSL also provides and the CUDA and HIP preludes define under [#ifndef]; a NaN's payload
   and sign do not survive any of them, so NaN is checked as "is a NaN" rather than on the bits. - A
   value on an f32 TIE has to be exact, not merely round-tripping. The reasoning that the cast does
   the narrowing holds only where the dialect has a [double]; MSL does not, so Metal rounds the
   decimal itself, at parse time. The two readings agree everywhere except on a tie, where the host
   takes ties-to-even and a decimal near but not equal to the value breaks by whichever side it
   landed on — so the emitted digit count silently decides the value. Ties are therefore spelled as
   exact hexadecimal literals, which no reading has to round.

   Deliberately absent: a precision suffix. The literal is always double-typed and the narrowing is
   [convert_precision]'s cast — one rounding of the exact host double, the same conversion the host
   itself performs. An [f]-suffixed decimal would round the decimal straight to float instead, which
   disagrees for a value on a float tie. The f32 and f16 legs are what pin that the cast is really
   there and really does the narrowing: their oracles are the host's own conversions.

   The twin of this test is [ll_printer_constants], which asks the same two portable questions of
   the IR DUMPS — the [.ll] and [.cd] text a constant bug is actually chased on. Both renderings now
   come out of one helper, [Utils.decimal_float_literal] (gh-ocannl-713); what stays here is what
   only C needs, the specials and the tie's hexadecimal spelling.

   The IR is hand-built so that each constant is stored, untouched, into its own cell of a
   materialized node: through the [Assignments] pipeline a bare constant fill is the backend's
   business to inline or upload, and only an inlined one is about this printer. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let codegen_capabilities = Context.codegen_capabilities (Context.auto ())

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

open Verdict.Claims

let skipped = Verdict.skipped ~backend:backend_name

(* On the bits, because the whole point is values a tolerance — or [Float.equal], which reports [-0.
   = +0.] — cannot tell apart. NaN is the one exception: no dialect spells a payload. *)
let bitwise a b =
  if Float.is_nan a || Float.is_nan b then Float.is_nan a && Float.is_nan b
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* The host's own double -> f32 and double -> f32 -> f16 narrowings: the oracles for the legs whose
   store precision is narrower than the literal's type. [FLOAT_TO_HALF((float)c)] is what the
   backends emit, so the half oracle goes through f32 too. *)
let to_f32 c = Int32.float_of_bits (Int32.bits_of_float c)
let to_f16 c = Ops.half_to_single (Ops.single_to_half (to_f32 c))

(* [value] is what the host holds; [spelling] is the exact literal the printer must produce for it,
   and [was] the pre-fix spelling that must NOT survive — checked with the cast's closing paren
   attached, since ["(float)(7)"] is otherwise a prefix of ["(float)(7.0)"]. [narrow] marks the
   values a half-precision node may hold: [Ops.exceeds_fp16_cutoff] refuses the large ones during
   lowering (the infinities are exempt there by construction, being reduction identities). *)
type case = { value : float; spelling : string; was : string option; narrow : bool }

let case ?was ?(narrow = true) value spelling = { value; spelling; was; narrow }

let cases =
  [
    (* Whole numbers: the class that lost its radix point. *)
    case 7. "7.0" ~was:"7";
    case (-13.) "-13.0" ~was:"-13";
    case 4096. "4096.0" ~was:"4096";
    case 0. "0.0" ~was:"0";
    (* The one value the missing radix point already corrupted (gh-ocannl-615). *)
    case (-0.) "-0.0" ~was:"-0";
    (* Large whole magnitudes: still all digits at [%.16g] (it switches to an exponent at 1e16), so
       this is where an integer literal's type would eventually overflow. *)
    case 1e15 "1000000000000000.0" ~was:"1000000000000000" ~narrow:false;
    case 1e20 "1e+20" ~narrow:false;
    (* Tiny: the smallest subnormal double, which flushes to zero in both narrow legs. *)
    case 5e-324 "4.940656458412465e-324" ~narrow:false;
    (* Values sixteen digits cannot recover. *)
    case (0.1 +. 0.2) "0.30000000000000004";
    case Float.max_finite_value "1.7976931348623157e+308" ~narrow:false;
    (* f32 ties, spelled as exact hex so that rounding the LITERAL and rounding the value agree.
       Three of them, because the decimal spellings fail in three different ways and a single tie
       would pin only one: the first has no round-tripping 16-digit form and its 17-digit form falls
       on the odd side; the second round-trips at 16 digits, so a printer keyed on round-tripping
       alone would leave it a decimal, and it is the value review offered as a counterexample; the
       third is a tie whose 16-digit form happened to land on the EVEN side, i.e. one that the
       decimal spelling got right by luck and the retry would have broken. *)
    case Float.(1. + (2. ** -24.)) "0x1.000001p+0";
    case Float.(0.5 + (1.5 * (2. ** -24.))) "0x1.000003p-1";
    case 1.2853009017203525e+35 "0x1.8c105dp+116" ~narrow:false;
    (* The OVERFLOW midpoint and its negative: halfway between the largest finite f32 and the 2^128
       that would follow it, so the host's ties-to-even overflows to an infinity while any decimal
       spelling sits below the midpoint and answers the largest finite f32 instead. The neighbour
       walk cannot see this tie -- the value above has no f32 to be -- so it is named outright. *)
    case 0x1.ffffffp+127 "0x1.ffffffp+127" ~narrow:false;
    case (-0x1.ffffffp+127) "-0x1.ffffffp+127" ~narrow:false;
    (* Ordinary values, pinning that their spelling did not move. *)
    case 0.5 "0.5";
    case (1. /. 3.) "0.3333333333333333";
    (* The specials. *)
    case Float.infinity "INFINITY";
    case Float.neg_infinity "(-INFINITY)";
    case Float.nan "NAN";
  ]

(* One store per constant, into its own cell of a materialized node. No arithmetic anywhere: an
   identity like [c *. 1.0] would be one more place for a value to be normalized, and the store is
   what the issue is about. Cells are seeded with a sentinel none of the cases equals, so a write
   the lowering dropped fails the value check rather than reading back a plausible number. *)
let leg ~prec ~name ~oracle ~stored =
  let stored = List.filter cases ~f:stored in
  let n = List.length stored in
  let node = Ll_test.node_factory ~prec ~first_id:9600 ~dims:[| n |] () in
  let out = node "flit_out" in
  Ll_test.materialize out;
  let llc =
    List.foldi stored ~init:LL.Noop ~f:(fun i acc { value; _ } ->
        Ll_test.seq acc (Ll_test.set out [| Ll_test.fixed i |] (Ll_test.c value)))
  in
  let o = Ll_test.optimize ~name llc in
  let got = List.hd_exn (Ll_test.execute ~name o ~seed:[ (out, Ll_test.blank n) ] ~read:[ out ]) in
  let want = Array.of_list_map stored ~f:(fun { value; _ } -> oracle value) in
  (stored, got, want)

let report ~claim (stored, got, want) =
  let ok i = i < Array.length got && bitwise got.(i) want.(i) in
  List.iteri stored ~f:(fun i c ->
      if not (ok i) then Stdio.eprintf "  %s: emitted %h, host %h\n" c.spelling got.(i) want.(i));
  p claim
    (Array.length got = Array.length want
    && (not (List.is_empty stored))
    && List.for_alli stored ~f:(fun i _ -> ok i))

(* === The values, at each store precision === *)

(* f64: the literal stands bare, with no cast at all ([convert_precision] is the identity between
   equal precisions) — so this leg is the one where the literal's own C type is the value, and the
   only one that sees the full 17 digits. Metal has no [double] and rejects an f64 node outright, so
   it is gated rather than run there. *)
let () =
  if not codegen_capabilities.Ir.Backend_intf.supports_f64 then
    skipped "f64 constants reach the kernel bit-exactly"
  else
    report ~claim:"f64 constants reach the kernel bit-exactly"
      (leg ~prec:Ops.double ~name:"flit_f64" ~oracle:Fn.id ~stored:(fun _ -> true))

(* f32: the oracle is the host's own double -> f32 rounding, which for the tie cases means
   ties-to-even. This is the leg that runs on Metal, where the literal is not narrowed by the cast
   but rounded at parse time — the two agree only because the ties are spelled exactly. *)
let () =
  report ~claim:"f32 constants reach the kernel bit-exactly"
    (leg ~prec:Ops.single ~name:"flit_f32" ~oracle:to_f32 ~stored:(fun _ -> true))

(* f16, the representative of the narrow-float family (bf16 and fp8 differ only in which codec the
   cast names): the literal is still a plain double literal, never a dialect-specific half literal —
   [0.0h] is valid MSL and a hard error under nvrtc, so no backend may spell one here. *)
let () =
  report ~claim:"f16 constants reach the kernel as the host's own half conversion"
    (leg ~prec:Ops.half ~name:"flit_f16" ~oracle:to_f16 ~stored:(fun c -> c.narrow))

(* === The emitted tokens === *)

(* The executed legs above establish the values; this establishes that they are values of literals
   with the intended shape, so that a future value which happens to be inert cannot quietly go back
   to being an integer literal. Read off the f32 kernel: [(float)(...)] is the double -> f32 cast on
   every C-family backend, which makes the negative checks exact. *)
let () =
  let src = Generated.read "flit_f32" in
  let has s = String.is_substring src ~substring:s in
  let missing =
    List.filter_map cases ~f:(fun { spelling; _ } ->
        if has ("(float)(" ^ spelling ^ ")") then None else Some spelling)
  in
  List.iter missing ~f:(fun s -> Stdio.eprintf "  not emitted as (float)(%s)\n" s);
  p_empty "every constant is emitted as the intended floating literal" ~over:cases missing;
  let surviving =
    List.filter_map cases ~f:(fun { was; _ } ->
        Option.bind was ~f:(fun w -> if has ("(float)(" ^ w ^ ")") then Some w else None))
  in
  List.iter surviving ~f:(fun s -> Stdio.eprintf "  still emitted as (float)(%s)\n" s);
  p_empty "no constant is emitted as an integer literal" ~over:cases surviving
