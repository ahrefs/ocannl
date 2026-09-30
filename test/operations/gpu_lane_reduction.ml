(* gh-ocannl-1124: the default GPU schedule's lanes over a nest whose preamble holds a reduction --
   the fused attention backward's dK nest, reduced to its shape:

   for b, t, h { for s { dp := 0; for e { dp += A[b, s, h, e] * B[b, t, h, e] }; for d { K[b, t, h,
   d] += dp * Q[b, s, h, d] } } }

   The lanes are the channel loop [d]; the preamble's [dp] is a value-width reduction over [e]. The
   config [gpu_lane_preamble_reduction] (or [default_gpu]'s [?preamble_reduction]) picks the
   treatment: [refused] keeps the plain plan (the gh-ocannl-1003 stage-1 rule), [duplicated] gives
   the nest lanes and every lane recomputes [dp] serially, [cooperative] retypes a reduction whose
   extent is the lane's workgroup [Workgroup_reduce] -- the lane's slot, no lane axis of its own --
   rendered as a simdgroup butterfly all-reduce that leaves the sum in every lane, or as the serial
   loop in every lane where the shuffle cannot render it.

   Legs, each executed on the run's backend against the serial reference (integer-valued operands:
   every partial sum is an exact float, so the comparison is exact whatever the association): 1.
   equal widths at one simdgroup (E = D = 32), cooperative: lanes, the reduction retyped, the
   emitted GPU kernel spells the all-reduce; 2. the same nest duplicated: lanes, the reduction
   serial, no all-reduce emitted; 3. refused: no lanes, the plain plan; 4. unequal widths (E = 16 or
   64 beside D = 32): lanes, the reduction stays serial (it does not span the lane's workgroup); 5.
   a partial simdgroup (E = D = 16): retyped, but the renderer declines the shuffle (lanes outside
   the reduction would be read) and every lane runs the loop; 6. several simdgroups (E = D = 64):
   retyped, declined by this v1 (it would need a shared broadcast and barrier), every lane runs the
   loop; 7. the same nest over an ordinary scope local, not the fused backward's [dp]: the plain
   plan in every mode -- the reassociation's license is the gate that minted [dp], not the shape; 8.
   a [Workgroup_reduce] hand-retyped over such a local keeps the hardware binding a staged reduction
   relies on (compiled, and read on GPU backends). *)

open Base
open Stdio
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name
let gpu = match backend_name with "metal" | "cuda" | "hip" -> true | _ -> false

(* The comment the all-reduce rendering opens with ([C_syntax.try_lane_all_reduce]). *)
let all_reduce_marker = "lane all-reduce into a scope local"
let b_n = 2
let t_n = 16
let h_n = 4
let s_n = 8
let node = L.node_factory ~first_id:1124000 ~dims:[| 1 |] ()

let fill ~dims f =
  let n = Array.fold dims ~init:1 ~f:( * ) in
  Array.init n ~f:(fun i -> f (L.unflat ~dims i))

let a_value b s h e = Float.of_int (1 + ((b + (2 * s) + (3 * h) + (5 * e)) % 7))
let b_value b t h e = Float.of_int (1 + (((7 * b) + (3 * h) + (5 * t) + e) % 11))
let q_value b s h d = Float.of_int (1 + ((b + s + (2 * h) + (3 * d)) % 5))
let k_seed b t h d = Float.of_int (((((b * t_n) + t) * h_n) + h + d) % 13)

type case = {
  name : string;
  opt : LL.optimized;
  e_sym : Ir.Indexing.symbol;
  hw_syms : Ir.Indexing.symbol list;  (** [b], [t], [h] and the lane [d]. *)
  seed : (Ir.Tnode.t * float array) list;
  k_tn : Ir.Tnode.t;
  expected : float array;
  launch_block : int;
}

let dk_nest ?(minted = true) ~name ~e_n ~d_n () =
  let a_dims = [| b_n; s_n; h_n; e_n |]
  and bv_dims = [| b_n; t_n; h_n; e_n |]
  and q_dims = [| b_n; s_n; h_n; d_n |]
  and k_dims = [| b_n; t_n; h_n; d_n |] in
  let a_tn = node ~dims:a_dims (name ^ "_a")
  and bv_tn = node ~dims:bv_dims (name ^ "_b")
  and q_tn = node ~dims:q_dims (name ^ "_q")
  and k_tn = node ~dims:k_dims (name ^ "_k") in
  (* The fused backward's own [dp] local, the only one whose reduction the lanes may reassociate
     ([Online_softmax.reassociable_local]); [~minted:false] gives the same nest an ordinary
     local. *)
  let dp_node =
    if minted then Ir.Online_softmax.dprob_local ~like:a_tn Ir.Ops.single
    else node ~dims:[| 1 |] (name ^ "_dp")
  in
  L.virtualize dp_node;
  List.iter [ a_tn; bv_tn; q_tn; k_tn ] ~f:L.materialize;
  let dp = LL.get_scope dp_node in
  let b = L.sym () and t = L.sym () and h = L.sym () and s = L.sym () in
  let e = L.sym () and d = L.sym () in
  let llc =
    L.loop_n b b_n @@ L.loop_n t t_n @@ L.loop_n h h_n @@ L.loop_n s s_n
    @@ LL.unflat_lines
         [
           LL.Declare_local { id = dp; needs_init = false };
           LL.Set_local (dp, L.c 0.);
           L.loop_n e e_n
             (LL.Set_local
                ( dp,
                  L.add (LL.Get_local dp)
                    (L.mul
                       (L.get a_tn [| L.iter b; L.iter s; L.iter h; L.iter e |])
                       (L.get bv_tn [| L.iter b; L.iter t; L.iter h; L.iter e |])) ));
           (let k_idx = [| L.iter b; L.iter t; L.iter h; L.iter d |] in
            L.loop_n d d_n
              (L.set k_tn k_idx
                 (L.add (L.get k_tn k_idx)
                    (L.mul (LL.Get_local dp)
                       (L.get q_tn [| L.iter b; L.iter s; L.iter h; L.iter d |])))));
         ]
  in
  let opt = L.optimize ~materialized:[ a_tn; bv_tn; q_tn; k_tn ] ~name llc in
  let expected =
    fill ~dims:k_dims (fun i ->
        let b, t, h, d = (i.(0), i.(1), i.(2), i.(3)) in
        List.fold (List.range 0 s_n) ~init:(k_seed b t h d) ~f:(fun acc s ->
            let dp =
              List.fold (List.range 0 e_n) ~init:0. ~f:(fun dp e ->
                  dp +. (a_value b s h e *. b_value b t h e))
            in
            acc +. (dp *. q_value b s h d)))
  in
  {
    name;
    opt;
    e_sym = e;
    hw_syms = [ b; t; h; d ];
    seed =
      [
        (a_tn, fill ~dims:a_dims (fun i -> a_value i.(0) i.(1) i.(2) i.(3)));
        (bv_tn, fill ~dims:bv_dims (fun i -> b_value i.(0) i.(1) i.(2) i.(3)));
        (q_tn, fill ~dims:q_dims (fun i -> q_value i.(0) i.(1) i.(2) i.(3)));
        (k_tn, fill ~dims:k_dims (fun i -> k_seed i.(0) i.(1) i.(2) i.(3)));
      ];
    k_tn;
    expected;
    launch_block = d_n;
  }

(* The axis the scheduled code gives the reduction loop [e]. *)
let axis_of_e case (llc : LL.t) =
  let rec go (llc : LL.t) =
    match llc with
    | LL.For_loop { index; axis; body; _ } ->
        if Ir.Indexing.equal_symbol index case.e_sym then Some axis else go body
    | LL.Seq (a, b) -> Option.first_some (go a) (go b)
    | LL.If { body; _ } | LL.Scan_loop { body; _ } -> go body
    | _ -> None
  in
  go llc

(* Whether a [Workgroup] loop runs inside a [Serial] loop: the lane bound past the serial [s]. *)
let rec lane_under_serial ?(under = false) (llc : LL.t) =
  match llc with
  | LL.For_loop { axis = LL.Workgroup; _ } when under -> true
  | LL.For_loop { axis = LL.Serial; body; _ } -> lane_under_serial ~under:true body
  | LL.For_loop { body; _ } | LL.If { body; _ } | LL.Scan_loop { body; _ } ->
      lane_under_serial ~under body
  | LL.Seq (a, b) -> lane_under_serial ~under a || lane_under_serial ~under b
  | _ -> false

let axis_name = function
  | LL.Serial -> "Serial"
  | LL.Workgroup_reduce -> "Workgroup_reduce"
  | _ -> "another kind"

let run ?(preamble = S.Preamble_cooperative) ?(lanes = true) ~reduction_axis ~emits_all_reduce case
    =
  let scheduled =
    S.apply
      (S.default_gpu ~block_size:256 ~min_parallel:64 ~workgroup_fill:1 ~preamble_reduction:preamble
         case.opt)
      case.opt
  in
  (* What schedule-aware fission reads: the reduce lane is the output lanes' threads, not a
     dimension of its own. *)
  if lanes then
    p
      (Printf.sprintf "%s: the mapping probe counts %d active threads (a reduce lane adds none)"
         case.name
         (b_n * t_n * h_n * case.launch_block))
      (List.equal
         (fun (g1, a1) (g2, a2) -> g1 = g2 && a1 = a2)
         (S.statement_mappings case.opt.llc
            (S.default_gpu ~block_size:256 ~min_parallel:64 ~workgroup_fill:1
               ~preamble_reduction:preamble case.opt))
         [ (b_n * t_n * h_n, b_n * t_n * h_n * case.launch_block) ]);
  if lanes then (
    p
      (Printf.sprintf "%s: Grid (h, t, b) and a %d-wide lane" case.name case.launch_block)
      (Array.equal Int.equal (LL.launch_dims scheduled.llc).LL.block [| case.launch_block; 1; 1 |]
      && Array.equal Int.equal (LL.launch_dims scheduled.llc).LL.grid [| h_n; t_n; b_n |]);
    p
      (Printf.sprintf "%s: the lane sits inside the serial query loop" case.name)
      (lane_under_serial scheduled.llc))
  else
    p
      (Printf.sprintf "%s: no lane inside the serial query loop" case.name)
      (not (lane_under_serial scheduled.llc));
  p
    (Printf.sprintf "%s: the reduction loop is %s" case.name (axis_name reduction_axis))
    (Option.equal LL.equal_axis_type (axis_of_e case scheduled.llc) (Some reduction_axis));
  let got = List.hd_exn (L.execute ~name:case.name scheduled ~seed:case.seed ~read:[ case.k_tn ]) in
  p_all2
    (Printf.sprintf "%s: every cell of K holds the serial reference's exact value" case.name)
    got case.expected ~f:Float.equal;
  let claim =
    Printf.sprintf "%s: the emitted kernel %s the all-reduce" case.name
      (if emits_all_reduce then "spells" else "does not spell")
  in
  if not gpu then skipped ~backend:backend_name claim
  else if emits_all_reduce then
    Generated.assert_emits ~routine:case.name ~contains:all_reduce_marker claim
  else Generated.assert_omits ~routine:case.name ~contains:all_reduce_marker claim

let () =
  eprintf "gpu_lane_reduction backend: %s (not part of the golden)\n%!" backend_name;
  printf "--- leg 1: equal widths at one simdgroup, cooperative ---\n";
  run ~reduction_axis:LL.Workgroup_reduce ~emits_all_reduce:true
    (dk_nest ~name:"lred_coop32" ~e_n:32 ~d_n:32 ());
  printf "--- leg 2: the same nest, duplicated ---\n";
  run ~preamble:S.Preamble_duplicated ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_dup32" ~e_n:32 ~d_n:32 ());
  printf "--- leg 3: refused keeps the plain plan ---\n";
  run ~preamble:S.Preamble_refused ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_refused32" ~e_n:32 ~d_n:32 ());
  printf "--- leg 4: unequal key and value widths ---\n";
  run ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_e16_d32" ~e_n:16 ~d_n:32 ());
  run ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_e64_d32" ~e_n:64 ~d_n:32 ());
  printf "--- leg 5: a partial simdgroup declines the shuffle ---\n";
  run ~reduction_axis:LL.Workgroup_reduce ~emits_all_reduce:false
    (dk_nest ~name:"lred_coop16" ~e_n:16 ~d_n:16 ());
  printf "--- leg 6: several simdgroups decline the shuffle (v1) ---\n";
  run ~reduction_axis:LL.Workgroup_reduce ~emits_all_reduce:false
    (dk_nest ~name:"lred_coop64" ~e_n:64 ~d_n:64 ())

let () =
  printf "--- leg 7: an ordinary local in the same shape keeps the plain plan ---\n";
  run ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~minted:false ~name:"lred_plain32" ~e_n:32 ~d_n:32 ());
  printf "--- leg 8: a hand-retyped Workgroup_reduce over an ordinary local keeps its binding ---\n";
  (* The renderer's all-reduce keys on the local's provenance, not on the local target: a staged
     reduction may bind such a loop to leave each lane its own partial. Compiled only -- under the
     binding each lane holds its own term, which is that reading's meaning, not this nest's. *)
  let case = dk_nest ~minted:false ~name:"lred_bound32" ~e_n:32 ~d_n:32 () in
  let b, t, h, d = match case.hw_syms with [ b; t; h; d ] -> (b, t, h, d) | _ -> assert false in
  let sched =
    [
      S.Retype { axis = b; ty = LL.Grid };
      S.Retype { axis = t; ty = LL.Grid };
      S.Retype { axis = h; ty = LL.Grid };
      S.Retype { axis = d; ty = LL.Workgroup };
      S.Retype { axis = case.e_sym; ty = LL.Workgroup_reduce };
    ]
  in
  let scheduled = S.apply sched case.opt in
  ignore (L.link ~name:case.name scheduled : Context.t * Context.routine);
  let omits = "lred_bound32: the emitted kernel does not spell the all-reduce" in
  let binds = "lred_bound32: the emitted kernel binds the reduction loop instead of looping it" in
  if not gpu then (
    skipped ~backend:backend_name omits;
    skipped ~backend:backend_name binds)
  else (
    Generated.assert_omits ~routine:case.name ~contains:all_reduce_marker omits;
    (* Positively, whatever the dialect's index type: the loop's symbol is assigned from the
       workgroup [.x] register, and it has no serial loop's [= 0] start. *)
    let register = match backend_name with "metal" -> "lid.x" | _ -> "threadIdx.x" in
    let ident = Ir.Indexing.symbol_ident case.e_sym in
    let src = Generated.read case.name in
    p binds
      (List.exists (String.split_lines src) ~f:(fun line ->
           String.is_substring line ~substring:(" " ^ ident ^ " = (")
           && String.is_substring line ~substring:register)
      && not (String.is_substring src ~substring:(" " ^ ident ^ " = 0;"))))
