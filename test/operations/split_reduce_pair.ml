(* gh-ocannl-1414: a reduction over two loops, neither of which splits alone, is split with the pair
   merged into one loop -- or, where the merge declines, interchanged so that the inner loop leads.

   Backprop lowers a small-batch parameter gradient as [for b, s, o: g[o] += dy[b,s,o]]. At [b = 8]
   the batch loop is too short to split, and the sequence loop is enclosed by it, a reduction loop
   that pins no component of the cell, so [Autotune.split_reduce_sites] found no site.

   Covered here, on every backend: - Detection: the batch-8 bias gradient is a site over the
   coalesced [(b, s)] pair; an operand read along the sequence only (a positional table, which has
   no flattened address) makes the coalescing decline and the pair is interchanged instead; a
   small-output contraction over two axes ([d[b,s,j] += w[j,h,e] * a[b,s,h,e]], the attention out
   projection's shape) is a site over the coalesced [(h, e)]; a reduction one loop already splits
   keeps its uncoalesced site. - Executed parity: each site's prelude, compiled as the search's
   [F_split] candidate compiles it, computes bitwise the default pipeline's result and the exact
   host sum. The inputs are dyadic (multiples of 1/16 and small enough that every partial sum is
   exact in f32), so the split's reassociation cannot move a bit, and a dropped or doubled row moves
   a cell. - The coalesced prelude round-trips through the schedule cache's saved form. -
   [Autotune.detect_matmul] reads the accumulation form of a write inside a loop whose body is a
   sequence (the shape of a split's first pass) instead of raising. - The untuned CUDA plan splits
   the batch-8 bias gradient at a threshold its 1024-row chain meets, into 32 blocks (not 512: the
   block count stays within the square root of the chain), and the seeded routine keeps the exact
   sums; off CUDA it splits nothing. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module L = Ll_test
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let splits_here = String.is_substring backend_name ~substring:"cuda"
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

let preset opt =
  if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits:Ir.Backend_intf.no_hardware_limits opt
  else if is_cpu then Sched.default_cpu ~min_parallel:1 opt
  else []

let zero_sched _tns = []

(* Multiples of 1/16 in [-1/4, 3/4], over a stride coprime to 17 so neighbouring cells differ; off
   centre, so that long sums do not cancel to nothing. *)
let dyadic ~n ~stride = Array.init n ~f:(fun c -> Float.of_int ((c * stride % 17) - 4) /. 16.)
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

let capture comp =
  let stash = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !stash

let site_of sites tn = List.find sites ~f:(fun s -> Ir.Tnode.equal s.Autotune.sr_target tn)

let extent (opt : LL.optimized) s =
  List.find_map (LL.loop_bounds opt.LL.llc) ~f:(fun (s', (lo, hi)) ->
      Option.some_if (Ir.Indexing.equal_symbol s s') (hi - lo + 1))

(* Whether the code reads a split's partials node: the prelude applied, not merely parsed. *)
let reads_partials (llc : LL.t) =
  List.exists (LL.affine_accesses llc) ~f:(fun a ->
      (not a.Ir.Affine.a_write)
      && List.exists a.Ir.Affine.a_tn.Ir.Tnode.label ~f:(String.equal "partials"))

(* The default pipeline's values of [tn], and those of the compile applying [sites]' prelude
   whole-routine before fission, as the search's [F_split] candidates are compiled. *)
let run_default comp tn =
  let ctx, routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  Context.get_values ctx tn

let run_split comp tn ~sites =
  let applied = ref false in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let opt = Sched.apply (Autotune.split_reduce_prelude sites) opt in
        applied := reads_partials opt.LL.llc;
        let _, tuples = Sched.fission_segmented ~preset ~zero_sched ~static_indices:[] opt in
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx tn, !applied)

(* --- The batch-8 bias gradient: [for b, s, o: g[o] += dy[b,s,o] (* w[s,o] *)]. --- *)

let nb = 8
and ns = 128
and no = 16

(* [~table:true] multiplies the gradient by a [ns x no] table read along the sequence only, as the
   positional embedding is read in gpt2_mini's first layers. *)
let bias_grad ~label ~table =
  let b =
    Operation.init ~l:(label ^ "_b") ~prec:Ir.Ops.single ~o:[ no ]
      ~f:(fun idx -> Float.of_int (idx.(0) % 5) /. 4.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    NTDSL.ndarray
      (dyadic ~n:(nb * ns * no) ~stride:5)
      ~label:[ label ^ "_x" ]
      ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
  in
  let dy_values = dyadic ~n:(nb * ns * no) ~stride:7 in
  let dy =
    NTDSL.ndarray dy_values ~label:[ label ^ "_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
  in
  let w_values = Array.map (dyadic ~n:(ns * no) ~stride:3) ~f:(fun v -> v +. 1.) in
  let w =
    NTDSL.ndarray w_values ~label:[ label ^ "_w" ] ~batch_dims:[ ns ] ~output_dims:[ no ] ()
  in
  let%op y = x + b in
  let%op prod = y *. dy in
  let loss =
    if table then
      let%op l = (prod *. w) ++ "... => 0" in
      l
    else
      let%op l = prod ++ "... => 0" in
      l
  in
  let g = (Option.value_exn ~here:[%here] b.Tensor.diff).Tensor.grad in
  Train.set_materialized g;
  (* Exact in f64 and in f32 alike: every partial sum is a multiple of 1/256 below 2^11. *)
  let exact =
    Array.init no ~f:(fun o ->
        let acc = ref 0. in
        for r = 0 to (nb * ns) - 1 do
          let wv = if table then w_values.((r % ns * no) + o) else 1. in
          acc := !acc +. (dy_values.((r * no) + o) *. wv)
        done;
        !acc)
  in
  (named label (Train.grad_update loss), g, exact)

let parity ~what ~comp ~tn ~exact ~site =
  let default = run_default comp tn in
  let split, applied = run_split comp tn ~sites:[ (site, 32) ] in
  p (what ^ ": the split prelude applied (the compiled code reads its partials)") applied;
  p_exists (what ^ ": the exact result is not trivially zero") (Array.to_list exact) ~f:(fun v ->
      Float.(abs v >= 1.));
  p_alli (what ^ ": the default pipeline computes the exact sums") (Array.to_list default)
    ~f:(fun i v -> bitwise v exact.(i));
  p_all2
    (what ^ ": the split computes bitwise the default pipeline's result")
    split default ~f:bitwise

let () =
  let comp, g, exact = bias_grad ~label:"srp" ~table:false in
  let opt = capture comp in
  let sites = Autotune.split_reduce_sites opt in
  match site_of sites g with
  | None -> p "the batch-8 bias gradient is a split-reduce site" false
  | Some site ->
      p "the batch-8 bias gradient is a split-reduce site" true;
      p "its site merges the batch and sequence loops"
        (match site.sr_coalesce with
        | Some (outer, inner) ->
            Option.equal Int.equal (extent opt outer) (Some nb)
            && Option.equal Int.equal (extent opt inner) (Some ns)
        | None -> false);
      p "its split loop ranges over every row" (site.sr_red = nb * ns);
      p "its split loop is the merged loop, not a loop of the lowering"
        (Option.is_none (extent opt site.sr_axis));
      parity ~what:"coalesced (b, s)" ~comp ~tn:g ~exact ~site;
      (* The prelude the search stores and a later process replays. *)
      let canon = SC.canonicalize opt in
      let prelude = Autotune.split_reduce_prelude [ (site, 32) ] in
      let saved, _ = SC.to_saved (SC.base_registry canon) prelude in
      let replayed, _ = SC.of_saved canon saved in
      let key sched =
        let scratch =
          {
            opt with
            LL.traced_store = Hashtbl.copy opt.LL.traced_store;
            LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
          }
        in
        SC.digest (SC.canonicalize (Sched.apply sched scratch))
      in
      p "the coalesced prelude round-trips through the cache's saved form"
        (String.equal (key prelude) (key replayed))

let () =
  let comp, g, exact = bias_grad ~label:"srt" ~table:true in
  let opt = capture comp in
  match site_of (Autotune.split_reduce_sites opt) g with
  | None -> p "the bias gradient read through a sequence table is a split-reduce site" false
  | Some site ->
      p "the bias gradient read through a sequence table is a split-reduce site" true;
      p "the table's reads decline the coalescing, so its site is the interchanged pair"
        (Option.is_none site.sr_coalesce && site.sr_red = ns
        && Option.equal Int.equal (extent opt site.sr_axis) (Some ns));
      parity ~what:"interchanged (b, s)" ~comp ~tn:g ~exact ~site

(* --- A small-output contraction over two axes: [d[b,s,j] += w[j,h,e] * a[b,s,h,e]]. --- *)

let () =
  let cb = 2 and cs = 4 and cj = 8 and ch = 8 and ce = 32 in
  let w_values = dyadic ~n:(cj * ch * ce) ~stride:3 in
  let a_values = dyadic ~n:(cb * cs * ch * ce) ~stride:11 in
  let w = NTDSL.ndarray w_values ~label:[ "srp_w_o" ] ~output_dims:[ cj; ch; ce ] () in
  let a = NTDSL.ndarray a_values ~label:[ "srp_attn" ] ~output_dims:[ cb; cs; ch; ce ] () in
  let%op d = w +* "jhe;bshe=>bsj" a in
  Train.set_materialized d.Tensor.value;
  let comp = named "srp_he" (Train.forward d) in
  (* Products of two multiples of 1/16 summed 256 at a time: multiples of 1/256 below 2^8. *)
  let exact =
    Array.init
      (cb * cs * cj)
      ~f:(fun c ->
        let bs = c / cj and j = c % cj in
        let acc = ref 0. in
        for he = 0 to (ch * ce) - 1 do
          acc := !acc +. (w_values.((j * ch * ce) + he) *. a_values.((bs * ch * ce) + he))
        done;
        !acc)
  in
  let opt = capture comp in
  match site_of (Autotune.split_reduce_sites opt) d.Tensor.value with
  | None -> p "the two-axis contraction is a split-reduce site" false
  | Some site ->
      p "the two-axis contraction is a split-reduce site" true;
      p "its site merges the two contracted loops"
        (match site.sr_coalesce with
        | Some (outer, inner) ->
            Option.equal Int.equal (extent opt outer) (Some ch)
            && Option.equal Int.equal (extent opt inner) (Some ce)
        | None -> false);
      p "its split loop ranges over the whole contraction" (site.sr_red = ch * ce);
      parity ~what:"coalesced (h, e)" ~comp ~tn:d.Tensor.value ~exact ~site

(* --- A reduction one loop already splits keeps that split. --- *)

let () =
  let b =
    Operation.init ~l:"srl_b" ~prec:Ir.Ops.single ~o:[ no ]
      ~f:(fun idx -> Float.of_int idx.(0))
      ~grad_spec:Tensor.Require_grad ()
  in
  let rows = 128 and cols = 8 in
  let x =
    NTDSL.ndarray
      (dyadic ~n:(rows * cols * no) ~stride:5)
      ~label:[ "srl_x" ] ~batch_dims:[ rows; cols ] ~output_dims:[ no ] ()
  in
  let%op loss = x + b ++ "... => 0" in
  let g = (Option.value_exn ~here:[%here] b.Tensor.diff).Tensor.grad in
  Train.set_materialized g;
  let opt = capture (named "srl" (Train.grad_update loss)) in
  match site_of (Autotune.split_reduce_sites opt) g with
  | None -> p "a 128-row batch reduction is a split-reduce site" false
  | Some site ->
      p "a 128-row batch reduction is a split-reduce site" true;
      p "it is split on its batch loop, uncoalesced"
        (Option.is_none site.sr_coalesce
        && Option.equal Int.equal (extent opt site.sr_axis) (Some rows))

(* --- The matmul detector on a split's first-pass shape: [for i { p[i] := 0; for k { ... } }].
   --- *)

let () =
  let node = L.node_factory ~first_id:14140 ~dims:[| 4 |] () in
  let d = node "srp_fd"
  and a = node ~dims:[| 4; 8 |] "srp_fa"
  and v = node ~dims:[| 8 |] "srp_fv" in
  let i = L.sym () and k = L.sym () in
  let prec = Ir.Ops.single in
  let acc =
    L.set d
      [| L.iter i |]
      (LL.Ternop
         ( Ir.Ops.FMA,
           (L.get a [| L.iter i; L.iter k |], prec),
           (L.get v [| L.iter k |], prec),
           (L.get d [| L.iter i |], prec) ))
  in
  let llc = L.loop_n i 4 (L.seq (L.set d [| L.iter i |] (L.c 0.)) (L.loop_n k 8 acc)) in
  p "the matmul detector reads a write nested in a loop's statement sequence without raising"
    (match Autotune.detect_matmul llc with _ -> true | exception Invalid_argument _ -> false)

(* --- The untuned CUDA plan. --- *)

let () =
  let comp, g, exact = bias_grad ~label:"srs" ~table:true in
  let report = ref None in
  let ctx, routine =
    Autotune.seeded_default ~min_serial:(nb * ns)
      ~report:(fun r -> report := Some r)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let values = Context.get_values ctx g in
  let r = Option.value_exn ~here:[%here] !report in
  List.iter r.Autotune.sd_split ~f:(fun (tn, blocks) ->
      Stdio.eprintf "split: %s into %d blocks (not part of the golden)\n" (Ir.Tnode.debug_name tn)
        blocks);
  Verdict.gated ~when_:splits_here ~on:backend_name
    "on CUDA the untuned plan splits the batch-8 bias gradient into 32 blocks"
    (List.exists r.sd_split ~f:(fun (tn, blocks) -> Ir.Tnode.equal tn g && blocks = 32));
  Verdict.gated ~when_:splits_here ~on:backend_name
    "on CUDA the split routine compiles without falling back to the default" (not r.sd_fell_back);
  Verdict.gated ~when_:(not splits_here) ~on:backend_name "off CUDA the untuned plan splits nothing"
    (List.is_empty r.sd_split);
  p_alli "the untuned plan's bias gradient is the exact sum" (Array.to_list values) ~f:(fun o v ->
      bitwise v exact.(o))
