(* gh-ocannl-2233: no built-in profile selects HIP's f16-accumulate WMMA arm, and every other
   backend resolves the profiles' f16 request as it did before the floor.

   The [performance] and [approximate] payloads set [fp16_arithmetic=prefer_narrow]
   ([Numerics.Fp16_prefer_narrow]): the narrow side of the f16 trade where the backend admits it,
   the backend's own auto where it floors it ([fp16_narrow_floored]). HIP floors it, because gfx11's
   f16-accumulate WMMA misses whole-output checksums by 7-82% and runs slower than the wide arm. An
   explicit [fp16_arithmetic=true] ([Fp16_narrow]) is the escape hatch: it still selects that arm.

   Pure tables, no device and no toolkit: HIP's [Ir.Hip_mma] (arm table and serial-residency
   predicate), CUDA's [Ir.Cuda_mma.resolve_arm], Metal's [Ir.Metal_mma.mma_accumulator] and the CPU
   backends' [Numerics.cpu_compute_prec]/[cpu_accum_prec], each tabulated over every storage
   precision (triple), emission scope and, for the CPU, both native-fp16 answers. Each profile's
   fp16 mode is read from its payload text ([Utils.profile_payloads]) through
   [Numerics.fp16_mode_of_string], the parse [Numerics.default] applies to the setting.

   Claims, per built-in profile. On HIP, the uniform-f16 storage triple's accumulator is f32 and f16
   reduction accumulators reside in f32 (the floor reaches the serial loop too, so width stays
   schedule-uniform). On CUDA, Metal and the CPU backends, the profile's tables equal the tables
   under the setting the profile resolved to before the floor -- [true] for [performance] and
   [approximate], [auto] for [reproducible] -- a judgment list, stated per profile below, that a new
   profile must join. On HIP and CUDA, the seeding gate
   [Backend_intf.advertises_mma_format_in_scope] on a descriptor with no wide-accumulator scope
   admits the uniform-f16 triple exactly where the backend's own arm accumulates at f16, so the
   descriptor's [mma_f16_narrow_floored] and the arm table cannot drift apart. And once, under an
   explicit [fp16_arithmetic=true]: HIP keeps its f16 accumulator, and the seeding gate agrees there
   too.

   The live claims check this process's own resolution: the policy's fp16 mode is the parse of the
   configured value, which comes from the profile the run expects ([--expect-profile=NAME], absent
   meaning no profile supplies it; the dune variants run under [--ocannl_profile=performance] and
   [--ocannl_profile=approximate]), and HIP's uniform-f16 arm is narrow under the live policy only
   when that policy is an explicit [Fp16_narrow] from outside any profile. *)

open Base
open Verdict.Claims
module N = Ir.Numerics
module Ops = Ir.Ops
module BI = Ir.Backend_intf

let with_mode mode f =
  let saved = N.get () in
  N.set_policy { saved with N.fp16_arithmetic = mode };
  Exn.protect ~f ~finally:(fun () -> N.set_policy saved)

let precs = Ops.all_precs

let triples =
  List.concat_map precs ~f:(fun a ->
      List.concat_map precs ~f:(fun b -> List.map precs ~f:(fun d -> (a, b, d))))

let scopes = [ BI.Mma_per_statement; BI.Mma_fragment_scope ]

(* Each backend's tables, rendered so that two modes' tabulations compare as strings. *)
let prec_opt = function None -> "-" | Some p -> Ops.prec_string p

let cuda_table () =
  List.concat_map triples ~f:(fun (a_prec, b_prec, d_prec) ->
      List.map scopes ~f:(fun scope ->
          Option.value_map ~default:"-" (Ir.Cuda_mma.resolve_arm ~a_prec ~b_prec ~d_prec ~scope)
            ~f:(fun arm -> Sexp.to_string (Ir.Cuda_mma.sexp_of_arm arm))))

let metal_table () =
  List.map triples ~f:(fun (a_prec, b_prec, d_prec) ->
      prec_opt (Ir.Metal_mma.mma_accumulator ~a_prec ~b_prec ~d_prec))

let cpu_table () =
  List.concat_map [ false; true ] ~f:(fun native_fp16_arithmetic ->
      List.concat_map precs ~f:(fun prec ->
          [
            Ops.prec_string (N.cpu_compute_prec ~native_fp16_arithmetic prec);
            Ops.prec_string (N.cpu_accum_prec ~native_fp16_arithmetic prec);
          ]))

let non_hip_tables () = [ ("CUDA", cuda_table ()); ("Metal", metal_table ()); ("cc", cpu_table ()) ]
let populated table = List.exists table ~f:(fun cell -> not (String.equal cell "-"))

(* HIP's uniform-f16 arm accumulates at f32 and f16 reduction accumulators reside in f32. *)
let hip_wide () =
  (match Ir.Hip_mma.mma_accumulator ~a_prec:Ops.half ~b_prec:Ops.half ~d_prec:Ops.half with
    | Some (acc, _, _) -> Ops.equal_prec acc Ops.single
    | None -> false)
  && Ir.Hip_mma.fp16_accum_wide ()

let hip_narrow () =
  (match Ir.Hip_mma.mma_accumulator ~a_prec:Ops.half ~b_prec:Ops.half ~d_prec:Ops.half with
    | Some (acc, _, _) -> Ops.equal_prec acc Ops.half
    | None -> false)
  && not (Ir.Hip_mma.fp16_accum_wide ())

let payload_mode name =
  Option.value_exn
    ~message:("the built-in profile " ^ name ^ " states no fp16_arithmetic")
    (N.fp16_mode_of_profile name)

(* The seeding gate ([Backend_intf.advertises_mma_format_in_scope], reading the descriptor's
   [mma_f16_auto_wide] and [mma_f16_narrow_floored]) against the arm table, on a descriptor with its
   wide-accumulator scopes removed: there it must admit the uniform-f16 triple exactly where the arm
   accumulates at f16, in each scope, on HIP and on CUDA (sm_120: every arm's floor met). *)
let f16t = BI.Mma_f16

let seeding_rows () =
  let policy = N.get () in
  let no_wide (mma : BI.mma_capability) =
    { BI.no_hardware_limits with mma = Some { mma with BI.mma_f16_wide_acc_scopes = [] } }
  in
  let hip_arm_narrow ~scope:_ =
    match Ir.Hip_mma.mma_accumulator ~a_prec:Ops.half ~b_prec:Ops.half ~d_prec:Ops.half with
    | Some (acc, _, _) -> Some (Ops.equal_prec acc Ops.half)
    | None -> None
  in
  let cuda_arm_narrow ~scope =
    Option.map (Ir.Cuda_mma.mma_arm ~a_prec:Ops.half ~b_prec:Ops.half ~d_prec:Ops.half ~scope)
      ~f:(fun arm -> Ops.equal_prec arm.BI.arm_accumulator Ops.half)
  in
  let cases =
    [
      (no_wide Ir.Hip_mma.capability, hip_arm_narrow);
      (no_wide (Option.value_exn (Ir.Cuda_mma.capability ~cc:120)), cuda_arm_narrow);
    ]
  in
  List.concat_map cases ~f:(fun (limits, arm_narrow) ->
      List.map scopes ~f:(fun scope ->
          match arm_narrow ~scope with
          | None -> false
          | Some narrow ->
              Bool.equal narrow
                (BI.advertises_mma_format_in_scope limits ~policy ~scope ~a:f16t ~b:f16t ~d:f16t)))

(* The judgment list: the setting each built-in profile's f16 request resolved as on the
   non-flooring backends before gh-ocannl-2233. *)
let resolved_before = [ ("reproducible", "auto"); ("performance", "true"); ("approximate", "true") ]

let () =
  p_all "every built-in profile has a stated pre-floor resolution" Utils.profile_payloads
    ~f:(fun (name, _) -> List.Assoc.mem resolved_before name ~equal:String.equal);
  List.iter Utils.profile_payloads ~f:(fun (name, _) ->
      let mode = payload_mode name in
      pf "%s: HIP's uniform-f16 arm accumulates in f32, and so do its f16 serial accumulators" name
        (with_mode mode hip_wide);
      p_all
        (Printf.sprintf
           "%s: with no wide-accumulator scope, HIP's and CUDA's seeding judgment admits the \
            uniform-f16 triple exactly where the arm accumulates at f16, in both scopes"
           name)
        (with_mode mode seeding_rows) ~f:Fn.id;
      match List.Assoc.find resolved_before name ~equal:String.equal with
      | None -> ()
      | Some before ->
          let ours = with_mode mode non_hip_tables in
          let theirs = with_mode (N.fp16_mode_of_string before) non_hip_tables in
          List.iter2_exn ours theirs ~f:(fun (backend, t1) (_, t2) ->
              pf "%s: %s's tables equal those under fp16_arithmetic=%s, both populated" name backend
                before
                (populated t1 && populated t2 && List.equal String.equal t1 t2)));
  p
    "explicit fp16_arithmetic=true: HIP keeps its f16-accumulate arm and storage-width serial \
     accumulators"
    (with_mode (N.fp16_mode_of_string "true") hip_narrow);
  p_all
    "explicit fp16_arithmetic=true: the seeding judgment agrees with HIP's and CUDA's arm tables \
     there too"
    (with_mode (N.fp16_mode_of_string "true") seeding_rows)
    ~f:Fn.id

(* The live resolution of this process. *)
let expected_profile =
  Array.find_map (Sys.get_argv ()) ~f:(String.chop_prefix ~prefix:"--expect-profile=")

let () =
  let value, source =
    Utils.get_global_arg_with_source ~default:"auto" ~arg_name:"fp16_arithmetic"
  in
  let live = (N.get ()).N.fp16_arithmetic in
  p "live: the policy's fp16 mode is the parse of the configured value"
    (N.equal_fp16_mode live (N.fp16_mode_of_string value));
  let from_profile = match source with Utils.From_profile (_, name) -> Some name | _ -> None in
  p "live: fp16_arithmetic comes from the profile the run expects (none: from no profile)"
    (Option.equal String.equal from_profile expected_profile);
  (* Each mode's answer by an exhaustive match, so a new mode is a compile error here
     (gh-ocannl-2260). *)
  let hip_narrow_mode =
    match live with
    | N.Fp16_narrow -> true
    | N.Fp16_auto | N.Fp16_wide | N.Fp16_prefer_narrow -> false
  in
  let explicit_narrow = hip_narrow_mode && Option.is_none from_profile in
  p "live: HIP's uniform-f16 arm is narrow only under an explicit fp16_arithmetic=true"
    (explicit_narrow || hip_wide ())
