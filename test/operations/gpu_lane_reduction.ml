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
   64 beside D = 32): cooperative retypes a reduction or declines the lanes, so the plain plan -- a
   reduction every lane would run whole is the duplicated arm, measured as a regression; 5. a
   partial simdgroup (E = D = 16) likewise keeps the plain plan at the devices' 32-lane shuffles,
   and on a device claiming 16-lane ones the retype happens and the renderer (32-lane warps)
   declines the shuffle, every lane running the loop; 6. several simdgroups (gh-ocannl-1168): at the
   default bound of one simdgroup (E = D = 64) the plain plan; allowed two or four simdgroups, E = D
   = 64 and 128 take lanes and the renderer all-reduces across simdgroups through workgroup-shared
   partials between two barriers per pair -- executed exact against the serial reference AND the
   materialized (unscheduled) run, over operands whose per-simdgroup partials discriminate a dropped
   or misrouted partial -- while a width past the bound keeps the plain plan, and a device claiming
   64-lane shuffles gets the retype the renderer renders as two 32-lane simdgroups; 7. the same nest
   over an ordinary scope local, not the fused backward's [dp]: the plain plan in every mode -- the
   reassociation's license is the gate that minted [dp], not the shape; 8. a [Workgroup_reduce]
   hand-retyped over such a local keeps the hardware binding a staged reduction relies on (compiled,
   and read on GPU backends); 9. the configured [auto] resolves from the device's economics --
   cooperative where per-lane recompute is cheap (Metal, CUDA), refused elsewhere (HIP, cc,
   unmeasured) -- and the default schedule equals the resolved explicit mode's. *)

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

(* What the cross-simdgroup form's opening comment adds ([C_syntax.try_lane_all_reduce]). *)
let multi_marker k = Printf.sprintf "%d simdgroups of 32" k
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
  flag_lane : Ir.Indexing.symbol option;
      (** [`Lane_flag]'s workgroup loop computing the lane-dependent flag. *)
  seed : (Ir.Tnode.t * float array) list;
  k_tn : Ir.Tnode.t;
  expected : float array;
  k_with : (int -> int list) -> float array;
  launch_block : int;
}

(* [?guard] wraps the preamble reduction in an [If]: [`Uniform] on the serial query symbol ([s < s_n
   - 2], so the last two queries contribute nothing), [`Lane_flag] on a scope local a workgroup loop
   sets lane-dependently ([flag := a < 32]) -- the shape whose lanes would part ways at the
   cross-simdgroup barriers. *)
let dk_nest ?(minted = true) ?guard ~name ~e_n ~d_n () =
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
    if minted then (
      (* Minted under the approximate-tier gate that licenses the reassociation, as the rewrite
         mints it. *)
      Ir.Online_softmax.set_backward_enabled (Some true);
      let tn = Ir.Online_softmax.dprob_local ~like:a_tn Ir.Ops.single in
      Ir.Online_softmax.set_backward_enabled None;
      tn)
    else node ~dims:[| 1 |] (name ^ "_dp")
  in
  L.virtualize dp_node;
  List.iter [ a_tn; bv_tn; q_tn; k_tn ] ~f:L.materialize;
  let dp = LL.get_scope dp_node in
  let b = L.sym () and t = L.sym () and h = L.sym () and s = L.sym () in
  let e = L.sym () and d = L.sym () in
  let reduction =
    L.loop_n e e_n
      (LL.Set_local
         ( dp,
           L.add (LL.Get_local dp)
             (L.mul
                (L.get a_tn [| L.iter b; L.iter s; L.iter h; L.iter e |])
                (L.get bv_tn [| L.iter b; L.iter t; L.iter h; L.iter e |])) ))
  in
  let flag_lane, guarded =
    match guard with
    | None -> (None, [ reduction ])
    | Some `Uniform -> (None, [ L.if_ (L.lt (L.embed s) (L.ic (s_n - 2))) reduction ])
    | Some `Lane_flag ->
        let flag_node = node ~dims:[| 1 |] (name ^ "_flag") in
        L.virtualize flag_node;
        let flag = LL.get_scope flag_node and a = L.sym () in
        ( Some a,
          [
            LL.Declare_local { id = flag; needs_init = false };
            L.loop_n a e_n (LL.Set_local (flag, L.lt (L.embed a) (L.ic 32)));
            L.if_ (LL.Get_local flag) reduction;
          ] )
  in
  let llc =
    L.loop_n b b_n @@ L.loop_n t t_n @@ L.loop_n h h_n @@ L.loop_n s s_n
    @@ LL.unflat_lines
         ([ LL.Declare_local { id = dp; needs_init = false }; LL.Set_local (dp, L.c 0.) ]
         @ guarded
         @ [
             (let k_idx = [| L.iter b; L.iter t; L.iter h; L.iter d |] in
              L.loop_n d d_n
                (L.set k_tn k_idx
                   (L.add (L.get k_tn k_idx)
                      (L.mul (LL.Get_local dp)
                         (L.get q_tn [| L.iter b; L.iter s; L.iter h; L.iter d |])))));
           ])
  in
  let opt = L.optimize ~materialized:[ a_tn; bv_tn; q_tn; k_tn ] ~name llc in
  (* [K] when lane [d]'s [dp] sums the terms of [e] in [es d]: the whole range is the reference; a
     subset is what a wrong cross-simdgroup rendering would compute ([discriminates]). *)
  let skipped s = match guard with Some `Uniform -> s >= s_n - 2 | _ -> false in
  let k_with es =
    fill ~dims:k_dims (fun i ->
        let b, t, h, d = (i.(0), i.(1), i.(2), i.(3)) in
        List.fold (List.range 0 s_n) ~init:(k_seed b t h d) ~f:(fun acc s ->
            let dp =
              List.fold
                (if skipped s then [] else es d)
                ~init:0.
                ~f:(fun dp e -> dp +. (a_value b s h e *. b_value b t h e))
            in
            acc +. (dp *. q_value b s h d)))
  in
  let expected = k_with (fun _ -> List.range 0 e_n) in
  {
    name;
    opt;
    e_sym = e;
    hw_syms = [ b; t; h; d ];
    flag_lane;
    seed =
      [
        (a_tn, fill ~dims:a_dims (fun i -> a_value i.(0) i.(1) i.(2) i.(3)));
        (bv_tn, fill ~dims:bv_dims (fun i -> b_value i.(0) i.(1) i.(2) i.(3)));
        (q_tn, fill ~dims:q_dims (fun i -> q_value i.(0) i.(1) i.(2) i.(3)));
        (k_tn, fill ~dims:k_dims (fun i -> k_seed i.(0) i.(1) i.(2) i.(3)));
      ];
    k_tn;
    expected;
    k_with;
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

(* Every case [run] schedules, by name: leg 11 reads the memory estimate off them. *)
let scheduled_by_name : (string, LL.optimized) Hashtbl.t = Hashtbl.create (module String)

(* A device whose shuffles are [width] lanes wide: the GPU backends state 32. *)
let simd width = { Ir.Backend_intf.no_hardware_limits with simdgroup_width = Some width }

let run ?(preamble = S.Preamble_cooperative) ?(lanes = true) ?(limits = simd 32) ?(all_reduce = 1)
    ?(marker = all_reduce_marker) ?(materialized = false) ~reduction_axis ~emits_all_reduce case =
  let schedule () =
    S.default_gpu ~block_size:256 ~min_parallel:64 ~workgroup_fill:1 ~preamble_reduction:preamble
      ~all_reduce_simdgroups:all_reduce ~limits case.opt
  in
  let scheduled = S.apply (schedule ()) case.opt in
  Hashtbl.set scheduled_by_name ~key:case.name ~data:scheduled;
  (* What schedule-aware fission reads: the reduce lane is the output lanes' threads, not a
     dimension of its own. *)
  if lanes then
    p
      (Printf.sprintf "%s: the mapping probe counts %d active threads (a reduce lane adds none)"
         case.name
         (b_n * t_n * h_n * case.launch_block))
      (List.equal
         (fun (g1, a1) (g2, a2) -> g1 = g2 && a1 = a2)
         (S.statement_mappings case.opt.llc (schedule ()))
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
  (if materialized then
     let plain =
       List.hd_exn
         (L.execute ~name:(case.name ^ "_mat") case.opt ~seed:case.seed ~read:[ case.k_tn ])
     in
     p_all2
       (Printf.sprintf "%s: every cell of K equals the materialized, unscheduled run's" case.name)
       got plain ~f:Float.equal);
  let claim =
    Printf.sprintf "%s: the emitted kernel %s the all-reduce" case.name
      (if emits_all_reduce then "spells" else "does not spell")
  in
  if not gpu then skipped ~backend:backend_name claim
  else if emits_all_reduce then Generated.assert_emits ~routine:case.name ~contains:marker claim
  else Generated.assert_omits ~routine:case.name ~contains:all_reduce_marker claim

(* The operands discriminate the cross-simdgroup phase: lane [d] summing only its own simdgroup's
   terms (the phase skipped), or only simdgroup 0's terms in every lane (one partial read for all),
   gives a different [K] somewhere. *)
let discriminates case =
  let simdgroup w = List.range (32 * w) (32 * (w + 1)) in
  let differs what wrong =
    p_exists
      (Printf.sprintf "%s: the operands tell %s from the whole sum" case.name what)
      (List.zip_exn (Array.to_list wrong) (Array.to_list case.expected))
      ~f:(fun (a, b) -> not (Float.equal a b))
  in
  differs "each lane's own simdgroup's partial" (case.k_with (fun d -> simdgroup (d / 32)));
  differs "simdgroup 0's partial" (case.k_with (fun _ -> simdgroup 0))

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
  printf "--- leg 4: unequal key and value widths keep the plain plan ---\n";
  run ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_e16_d32" ~e_n:16 ~d_n:32 ());
  run ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_e64_d32" ~e_n:64 ~d_n:32 ());
  printf "--- leg 5: a partial simdgroup keeps the plain plan; the renderer declines it too ---\n";
  run ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_coop16" ~e_n:16 ~d_n:16 ());
  (* A device claiming 16-lane shuffles gets the retype; the renderer, whose warp is 32 lanes, then
     declines the shuffle and every lane runs the loop -- still exact. *)
  run ~limits:(simd 16) ~reduction_axis:LL.Workgroup_reduce ~emits_all_reduce:false
    (dk_nest ~name:"lred_coop16r" ~e_n:16 ~d_n:16 ());
  printf "--- leg 6: several simdgroups, where the device allows them ---\n";
  (* The default bound (one simdgroup, every device until measured): the plain plan. *)
  run ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_coop64" ~e_n:64 ~d_n:64 ());
  List.iter
    [ (64, 2, "lred_multi64"); (128, 4, "lred_multi128") ]
    ~f:(fun (width, k, name) ->
      let case = dk_nest ~name ~e_n:width ~d_n:width () in
      discriminates case;
      run ~all_reduce:k ~marker:(multi_marker k) ~materialized:true
        ~reduction_axis:LL.Workgroup_reduce ~emits_all_reduce:true case);
  (* Past the bound: four simdgroups allowed two keep the plain plan. *)
  run ~all_reduce:2 ~lanes:false ~reduction_axis:LL.Serial ~emits_all_reduce:false
    (dk_nest ~name:"lred_multi128_cap2" ~e_n:128 ~d_n:128 ());
  (* A device claiming 64-lane shuffles retypes at one of its simdgroups; the renderer's 32-lane
     warps make that two. *)
  run ~limits:(simd 64) ~marker:(multi_marker 2) ~reduction_axis:LL.Workgroup_reduce
    ~emits_all_reduce:true
    (dk_nest ~name:"lred_coop64r" ~e_n:64 ~d_n:64 ())

let () =
  printf "--- leg 7: an ordinary local in the same shape keeps the plain plan ---\n";
  let like = node ~dims:[| 1 |] "lred_ungated" in
  Ir.Online_softmax.set_backward_enabled (Some false);
  p "the fused backward's dp local is not minted without online_softmax_backward"
    (match Ir.Online_softmax.dprob_local ~like Ir.Ops.single with
    | _ -> false
    | exception Invalid_argument _ -> true);
  Ir.Online_softmax.set_backward_enabled (Some true);
  let single = Ir.Online_softmax.dprob_local ~like Ir.Ops.single in
  p "the dp local minted for the same node at another precision is a node of its own, at that one"
    (let double = Ir.Online_softmax.dprob_local ~like Ir.Ops.double in
     (not (Ir.Tnode.equal single double))
     && Ir.Ops.equal_prec (Lazy.force double.Ir.Tnode.storage_prec) Ir.Ops.double);
  p "the dp local minted again at its own precision is the same node"
    (Ir.Tnode.equal single (Ir.Online_softmax.dprob_local ~like Ir.Ops.single));
  Ir.Online_softmax.set_backward_enabled None;
  (* The marker is the minting, not the node's public fields: a look-alike built with the rewrite's
     namespace and label is an ordinary local. *)
  let forged =
    Ir.Tnode.create ~namespace:single.Ir.Tnode.namespace (Ir.Tnode.Specified Ir.Ops.single)
      ~id:1124999 ~label:single.Ir.Tnode.label
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  p "a node forged with the dp local's namespace and label is not reassociable"
    (Ir.Online_softmax.reassociable_local single
    && not (Ir.Online_softmax.reassociable_local forged));
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

let () =
  printf "--- leg 9: auto resolves from the device's economics ---\n";
  (* Under the configured [auto]: cooperative where redundant per-lane scalar work is cheap
     (measured on Metal and CUDA), refused elsewhere (HIP, the C backends, anything unmeasured) --
     and what the default schedule then emits IS the explicit mode's schedule, so the Metal and CUDA
     timings of [cooperative] are the default's. *)
  let module BI = Ir.Backend_intf in
  let cheap = { (simd 32) with lane_scalar_recompute_cheap = true } in
  p "auto on a device where per-lane recompute is cheap is cooperative"
    (S.equal_lane_preamble_reduction (S.lane_preamble_reduction_for cheap) S.Preamble_cooperative);
  p "auto on an unmeasured device is refused"
    (S.equal_lane_preamble_reduction
       (S.lane_preamble_reduction_for BI.no_hardware_limits)
       S.Preamble_refused);
  let device = Context.hardware_limits (Lazy.force L.base_ctx) in
  let expected =
    match backend_name with "metal" | "cuda" -> S.Preamble_cooperative | _ -> S.Preamble_refused
  in
  p "auto on the run's backend resolves to its measured mode"
    (S.equal_lane_preamble_reduction (S.lane_preamble_reduction_for device) expected);
  let case = dk_nest ~name:"lred_auto32" ~e_n:32 ~d_n:32 () in
  let sched ?preamble_reduction limits =
    S.sexp_of_schedule
      (S.default_gpu ~block_size:256 ~min_parallel:64 ~workgroup_fill:1 ?preamble_reduction ~limits
         case.opt)
  in
  List.iter
    [
      ("a cheap-recompute device", cheap);
      ("an unmeasured device", BI.no_hardware_limits);
      ("the run's device", device);
    ]
    ~f:(fun (what, limits) ->
      p
        (Printf.sprintf "on %s the default schedule is the resolved mode's" what)
        (Sexp.equal (sched limits)
           (sched ~preamble_reduction:(S.lane_preamble_reduction_for limits) limits)))

(* The sites of a scheduled case, as [Low_level.lane_all_reduce_sites] reads them: the predicate the
   renderer and the schedule's memory estimate share. *)
let sites (opt : LL.optimized) =
  LL.lane_all_reduce_sites ~reassociable:Ir.Online_softmax.reassociable_local opt.llc

let hand_schedule ?(extra_lanes = []) case =
  let b, t, h, d = match case.hw_syms with [ b; t; h; d ] -> (b, t, h, d) | _ -> assert false in
  S.apply
    ([
       S.Retype { axis = b; ty = LL.Grid };
       S.Retype { axis = t; ty = LL.Grid };
       S.Retype { axis = h; ty = LL.Grid };
       S.Retype { axis = d; ty = LL.Workgroup };
       S.Retype { axis = case.e_sym; ty = LL.Workgroup_reduce };
     ]
    @ List.map extra_lanes ~f:(fun axis -> S.Retype { axis; ty = LL.Workgroup }))
    case.opt

let () =
  printf "--- leg 10: barriers only under workgroup-uniform control ---\n";
  (* A guard on the serial query symbol is the same in every lane: the site keeps the
     cross-simdgroup form, executed exact (the last two queries skip the reduction). *)
  let case = dk_nest ~guard:`Uniform ~name:"lred_guard_uniform64" ~e_n:64 ~d_n:64 () in
  let scheduled = hand_schedule case in
  Hashtbl.set scheduled_by_name ~key:case.name ~data:scheduled;
  p "lred_guard_uniform64: a guard on the serial query symbol leaves the site cross-simdgroup"
    (match sites scheduled with [ site ] -> site.LL.lar_cross_simdgroup | _ -> false);
  let got = List.hd_exn (L.execute ~name:case.name scheduled ~seed:case.seed ~read:[ case.k_tn ]) in
  p_all2 "lred_guard_uniform64: every cell of K holds the serial reference's exact value" got
    case.expected ~f:Float.equal;
  let claim = "lred_guard_uniform64: the emitted kernel spells the cross-simdgroup all-reduce" in
  if gpu then Generated.assert_emits ~routine:case.name ~contains:(multi_marker 2) claim
  else skipped ~backend:backend_name claim;
  (* A flag a workgroup loop sets lane-dependently: lanes 32..63 would skip both barriers. The site
     renders serially in every lane instead. Compiled only -- under the flag's per-lane binding the
     reduction runs in half the lanes, which is that program's meaning, not the reference's. *)
  let case = dk_nest ~guard:`Lane_flag ~name:"lred_guard_flag64" ~e_n:64 ~d_n:64 () in
  let scheduled = hand_schedule ~extra_lanes:(Option.to_list case.flag_lane) case in
  Hashtbl.set scheduled_by_name ~key:case.name ~data:scheduled;
  p "lred_guard_flag64: a guard reading a lane-dependent local leaves the site serial"
    (match sites scheduled with [ site ] -> not site.LL.lar_cross_simdgroup | _ -> false);
  ignore (L.link ~name:case.name scheduled : Context.t * Context.routine);
  let omits = "lred_guard_flag64: the emitted kernel does not spell the all-reduce" in
  let loops = "lred_guard_flag64: the emitted kernel loops the reduction in every lane" in
  if not gpu then (
    skipped ~backend:backend_name omits;
    skipped ~backend:backend_name loops)
  else (
    Generated.assert_omits ~routine:case.name ~contains:all_reduce_marker omits;
    let ident = Ir.Indexing.symbol_ident case.e_sym in
    p loops (String.is_substring (Generated.read case.name) ~substring:(" " ^ ident ^ " = 0;")))

(* The workgroup-shared bytes of the [lred_partials_*] arrays a generated kernel declares. An
   element type other than float/double reads as an impossible count, so the claim fails. *)
let declared_partial_bytes src =
  List.sum
    (module Int)
    (String.split_lines src)
    ~f:(fun line ->
      let line = String.strip line in
      match String.substr_index line ~pattern:"lred_partials_" with
      | Some i when String.is_suffix line ~suffix:"];" && not (String.mem line '=') -> (
          let ty = List.last_exn (String.split (String.rstrip (String.prefix line i)) ~on:' ') in
          let lb = String.index_exn line '[' and rb = String.rindex_exn line ']' in
          let slots = Int.of_string (String.sub line ~pos:(lb + 1) ~len:(rb - lb - 1)) in
          match ty with "float" -> 4 * slots | "double" -> 8 * slots | _ -> -1_000_000)
      | _ -> 0)

let () =
  printf "--- leg 11: the schedule's memory estimate counts the partials the kernel declares ---\n";
  let capabilities = Context.codegen_capabilities (Lazy.force L.base_ctx) in
  let estimate name =
    S.workgroup_memory_bytes ~capabilities (Hashtbl.find_exn scheduled_by_name name)
    - S.workgroup_memory_bytes ~capabilities:Ir.Backend_intf.no_codegen_capabilities
        (Hashtbl.find_exn scheduled_by_name name)
  in
  let admits name bytes =
    match
      S.check_hardware_limits ~name
        ~limits:{ Ir.Backend_intf.no_hardware_limits with max_workgroup_memory_bytes = Some bytes }
        ~capabilities
        (Hashtbl.find_exn scheduled_by_name name)
    with
    | () -> true
    | exception Utils.User_error _ -> false
  in
  (* On every backend: the estimate is exactly what the emitted kernel declares (none on cc). *)
  List.iter
    [
      "lred_coop32";
      "lred_dup32";
      "lred_coop16r";
      "lred_multi64";
      "lred_multi128";
      "lred_coop64r";
      "lred_guard_uniform64";
      "lred_guard_flag64";
    ] ~f:(fun name ->
      p
        (Printf.sprintf "%s: the estimate counts exactly the partials the emitted kernel declares"
           name)
        (estimate name = declared_partial_bytes (Generated.read name)));
  (* On the GPU backends, the boundaries: 2 f32 slots for 64 lanes, 4 for 128, none within one
     simdgroup or for a serial site. *)
  List.iter
    [
      ("lred_multi64", 8);
      ("lred_multi128", 16);
      ("lred_guard_uniform64", 8);
      ("lred_coop32", 0);
      ("lred_guard_flag64", 0);
    ]
    ~f:(fun (name, bytes) ->
      gated ~when_:gpu ~on:backend_name
        (Printf.sprintf "%s: %d bytes of partials, admitted at that limit%s" name bytes
           (if bytes > 0 then " and refused one byte below it" else ""))
        (estimate name = bytes && admits name bytes && (bytes = 0 || not (admits name (bytes - 1)))))
