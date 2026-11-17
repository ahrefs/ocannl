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
   the scopes anywhere else, fails here rather than in a CUDA rendering on a fleet box.

   Since gh-ocannl-2284 the table also reads the device's compute capability: uniform f16 under the
   wide resolution is m16n8k16 per statement where the device meets that arm's floor and m16n8k8
   below it. So the sweep runs at every capability on either side of every arm's floor, and checks
   that the capability only picks among arms: a triple, scope and policy has an arm at every
   capability or at none. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module N = Ir.Numerics
module M = Ir.Cuda_mma

(* Every numerics policy, enumerated by the type, so a field added to it joins the sweep. *)
let policies = N.all

let triples =
  List.concat_map Ir.Ops.all_precs ~f:(fun a_prec ->
      List.concat_map Ir.Ops.all_precs ~f:(fun b_prec ->
          List.map Ir.Ops.all_precs ~f:(fun d_prec -> (a_prec, b_prec, d_prec))))

(* Every capability the resolver can distinguish: 0 (no device), and just below and at every arm's
   floor, read from the arms rather than listed. *)
let ccs =
  0 :: List.concat_map M.all_of_arm ~f:(fun arm -> [ M.arm_floor arm - 1; M.arm_floor arm ])
  |> List.dedup_and_sort ~compare:Int.compare

(* Both scopes' arms for every triple under every policy at every capability; the resolver reads the
   installed policy. *)
let rows =
  List.concat_map policies ~f:(fun policy ->
      N.set_policy policy;
      List.concat_map ccs ~f:(fun cc ->
          List.map triples ~f:(fun (a_prec, b_prec, d_prec) ->
              let resolve scope = M.resolve_arm ~cc ~a_prec ~b_prec ~d_prec ~scope in
              ( (policy, cc, (a_prec, b_prec, d_prec)),
                resolve BI.Mma_per_statement,
                resolve BI.Mma_fragment_scope ))))

let converted = function
  | M.Wmma kind -> Option.is_some (M.wmma_info kind).M.wc_d_cvt
  | M.Inline_ptx _ -> false

let show_arm = function None -> "none" | Some arm -> M.arm_marker arm

let report what ((policy, cc, (a, b, d)), statement, fragment) =
  Stdio.eprintf "%s: sm_%d %s*%s>%s under %s: per-statement %s, fragment scope %s\n" what cc
    (Ir.Ops.prec_string a) (Ir.Ops.prec_string b) (Ir.Ops.prec_string d)
    (Sexp.to_string (N.sexp_of_t policy))
    (show_arm statement) (show_arm fragment)

let () =
  let statement_arms = List.filter_map rows ~f:(fun (_, statement, _) -> statement) in
  let resolved = List.concat_map rows ~f:(fun (_, s, f) -> List.filter_opt [ s; f ]) in
  Stdio.eprintf
    "%d policy x capability x triple rows, %d per-statement arms (not part of the golden)\n"
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
  p_all "every arm of the variant is resolved for some triple, scope, policy and capability"
    M.all_of_arm ~f:(fun arm -> List.mem resolved arm ~equal:M.equal_arm);
  (* The capability picks among arms, never whether there is one; the control below shows it does
     pick somewhere. Each row is compared with its own policy and triple at the top capability. *)
  let top_cc = List.last_exn ccs in
  let key (policy, _, (a, b, d)) =
    String.concat ~sep:" "
      (Sexp.to_string (N.sexp_of_t policy) :: List.map [ a; b; d ] ~f:Ir.Ops.prec_string)
  in
  let at_top = Hashtbl.create (module String) in
  List.iter rows ~f:(fun ((((_, cc, _) as k), _, _) as row) ->
      if cc = top_cc then Hashtbl.set at_top ~key:(key k) ~data:row);
  let top (k, _, _) = Hashtbl.find_exn at_top (key k) in
  p_all ~min:100 "a triple, scope and policy has an arm at every capability or at none" rows
    ~f:(fun ((_, statement, fragment) as row) ->
      let _, top_statement, top_fragment = top row in
      Bool.equal (Option.is_some statement) (Option.is_some top_statement)
      && Bool.equal (Option.is_some fragment) (Option.is_some top_fragment)
      ||
      (report "capability-dependent existence" row;
       false));
  p_exists "some per-statement arm differs between two capabilities, under one policy and triple"
    rows ~f:(fun ((_, statement, _) as row) ->
      let _, top_statement, _ = top row in
      not (Option.equal M.equal_arm statement top_statement))
