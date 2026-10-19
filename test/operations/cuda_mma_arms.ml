(* CUDA's tensor-core arms resolve once (gh-ocannl-1232).

   [Ir.Cuda_mma.resolve_arm] is the single table from a storage triple, the numerics policy and the
   emission scope to an arm; the backend's [mma_syntax] and [mma_fragment_syntax] hooks match on its
   result, [mma_arm] projects it, and the descriptor's and nvrtc's floors are read from the arm. So
   there is no hook dispatch order left to pin against rendered CUDA: what remains to check is the
   shape of the table itself, which these claims evaluate as pure functions -- no toolkit, no
   device, nothing rendered.

   Before gh-ocannl-1232 the per-statement hook reached wmma's converted wide-f16 arm only by
   falling through from the inline-PTX m16n8k16 arm, and a self-contained statement with a converted
   destination was declined inside the wmma rendering. The table now says it outright: the scopes
   split a triple only where the fragment scope's arm converts its destination, and no per-statement
   arm ever does. A mutation that resolved a per-statement statement to a converted arm, or split
   the scopes anywhere else, fails here rather than in a CUDA rendering on a fleet box. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module N = Ir.Numerics
module M = Ir.Cuda_mma

let policies =
  List.concat_map [ N.Fp16_auto; N.Fp16_narrow; N.Fp16_wide ] ~f:(fun fp16_arithmetic ->
      List.concat_map [ N.Bf16_auto; N.Bf16_narrow; N.Bf16_wide ] ~f:(fun bf16_arithmetic ->
          List.concat_map [ false; true ] ~f:(fun tf32_matmuls ->
              List.map [ false; true ] ~f:(fun narrow_compute_f32 ->
                  { N.tf32_matmuls; narrow_compute_f32; fp16_arithmetic; bf16_arithmetic }))))

let triples =
  List.concat_map Ir.Ops.all_precs ~f:(fun a_prec ->
      List.concat_map Ir.Ops.all_precs ~f:(fun b_prec ->
          List.map Ir.Ops.all_precs ~f:(fun d_prec -> (a_prec, b_prec, d_prec))))

(* Both scopes' arms for every triple under every policy; the resolver reads the installed
   policy. *)
let rows =
  List.concat_map policies ~f:(fun policy ->
      N.set_policy policy;
      List.map triples ~f:(fun (a_prec, b_prec, d_prec) ->
          let resolve scope = M.resolve_arm ~a_prec ~b_prec ~d_prec ~scope in
          ( (policy, (a_prec, b_prec, d_prec)),
            resolve BI.Mma_per_statement,
            resolve BI.Mma_fragment_scope )))

let converted = function
  | M.Wmma kind -> Option.is_some (M.wmma_info kind).M.wc_d_cvt
  | M.Inline_ptx _ -> false

let show_arm = function None -> "none" | Some arm -> M.arm_marker arm

let report what ((policy, (a, b, d)), statement, fragment) =
  Stdio.eprintf "%s: %s*%s>%s under %s: per-statement %s, fragment scope %s\n" what
    (Ir.Ops.prec_string a) (Ir.Ops.prec_string b) (Ir.Ops.prec_string d)
    (Sexp.to_string (N.sexp_of_t policy))
    (show_arm statement) (show_arm fragment)

let () =
  let statement_arms = List.filter_map rows ~f:(fun (_, statement, _) -> statement) in
  let resolved = List.concat_map rows ~f:(fun (_, s, f) -> List.filter_opt [ s; f ]) in
  Stdio.eprintf "%d policy x triple rows, %d per-statement arms (not part of the golden)\n"
    (List.length rows) (List.length statement_arms);
  p_none ~min:100
    "no per-statement arm converts its destination: only a fragment scope's boundary can"
    statement_arms ~f:converted;
  p_all ~min:100
    "the scopes resolve a triple alike, except where the fragment scope's arm converts its \
     destination"
    rows ~f:(fun ((_, statement, fragment) as row) ->
      (match (statement, fragment) with
        | None, None -> true
        | Some s, Some f -> M.equal_arm s f || converted f
        | Some _, None | None, Some _ -> false)
      ||
      (report "scope split" row;
       false));
  p_all "every arm of the variant is resolved for some triple, scope and policy" M.all_of_arm
    ~f:(fun arm -> List.mem resolved arm ~equal:M.equal_arm)
