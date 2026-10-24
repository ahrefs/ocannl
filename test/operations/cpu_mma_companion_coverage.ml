(* gh-ocannl-1561: the CPU tensorized matmul flavors that bind a hardware dimension are not offered
   beside a companion nest they would leave uncovered.

   The whole-triple [Tile_mma] form retypes the zeroing's column loop to a [Workgroup] axis at the
   [Tensorize] lane width [m_nj] (plus a [Grid] row split when [sk_bm > 0]), and the packed Grid
   shapes [Grid]-type their row-block loop. Launch dimensions are kernel-global, so a nest beside
   the matmul writing a materialized node must sit under loops covering those dimensions
   ([Low_level.validate_parallel]); the CPU builders give companions no geometry. The tuner used to
   offer these flavors anyway: the materialize-all arm of tune_footprint_default (an 8x32 by 32x8
   product read through its diagonal) timed [W_sketch[mma-cpu 0x0x0]] -- the whole-triple form, [0]
   meaning the unsplit row -- and paid a failed compile for it at every search. Now the family tree
   refutes those flavors, naming the companion's node, wherever such a companion remains.

   Seeding uses synthetic limits (a 32-byte vector file), so the offered set is machine-independent.

   1. The issue's routine, [a] materialized: every offered seed applies and passes
   [validate_parallel], none is the whole-triple form, the tree names the companion [d] in its
   refutation, and every offered seed computes [d] bit for bit (integer inputs, exact sums). 2. A
   64x32 by 32x64 product read through its diagonal: tall enough for the Grid shapes (four 16-row
   blocks). The same claims, plus: no offered seed is a Grid shape. 3. Control, the 64-row product
   with no companion: the whole-triple form and a packed Grid shape are still offered, and every
   seed validates -- the refutation is the companion's, not the shape's. 4. A bias-add + relu tail
   beside the materialized product: the unfused flavor offers no hardware-binding shape, and its
   seeds validate and compute the output bit for bit; the fused flavor (whose fusion relocates the
   tail) keeps the whole-triple form, which validates and computes too. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
open Verdict.Claims

let accum_prec = Ll_test.backend_accum_prec

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let seeds opt = Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt
let label sp = Autotune.sketch_params_label sp
let whole_triple (sp : Autotune.sketch_params) = sp.sk_mma && sp.sk_bk = 0
let grid_shape (sp : Autotune.sketch_params) = sp.sk_grid || (whole_triple sp && sp.sk_bm > 0)
let grid_shape_or_whole sp = grid_shape sp || whole_triple sp

let refutations opt =
  match fst (Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt) with
  | Autotune.Seeding_matmul { tree; _ } -> List.map (Ir.Schedule_space.refutations tree) ~f:snd
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> []

let site_writes opt (t : Tensor.t) =
  match fst (Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt) with
  | Autotune.Seeding_matmul { site; _ } -> Ir.Tnode.equal site.Autotune.m_d t.Tensor.value
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> false

(* Whether a seed's schedule constructs, applies and passes [validate_parallel]; a failure goes to
   stderr. *)
let validates ~tag opt sp =
  match
    let s = scratch opt in
    let post = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp s) s in
    LL.validate_parallel post.LL.optimize_ctx.placements post.LL.llc
  with
  | () -> true
  | exception exn ->
      Stdio.eprintf "%s: %s failed: %s\n%!" tag (label sp) (Exn.to_string exn);
      false

let ints label ~rows ~cols modulus =
  TDSL.ndarray
    (Array.init (rows * cols)
       ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus ~offset:(-2.) ~stride:1.))
    ~label:[ label ] ~output_dims:[ rows; cols ] ()

let values ?transform ~mat comp (out : Tensor.t) =
  let ctx = Context.decide_materialized (Context.cpu ()) mat in
  let ctx, routine =
    match transform with
    | None -> Context.compile ctx comp Ir.Indexing.Empty
    | Some f -> Context.compile ~lowered_transform:f ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let v = Context.get_values ctx out.Tensor.value in
  Context.release ctx;
  v

(* Every seed of [indexed] (positions in the offered list), compiled and run on the backend, gives
   [out] bit for bit as the untuned compile does. The seed is re-derived by position from the
   compile's own lowering. *)
let p_executed ?(what = "offered seed") ~tag ~mat comp (out : Tensor.t) indexed =
  let reference = values ~mat comp out in
  p_all (Printf.sprintf "%s: every %s computes the output bit for bit" tag what) indexed
    ~f:(fun (idx, sp) ->
      let transform opt =
        match List.nth (seeds opt) idx with
        | Some sp' when String.equal (label sp') (label sp) ->
            [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp' opt) opt ]
        | _ ->
            failwith
              (Printf.sprintf "seed %d (%s) is gone from the compile's lowering" idx (label sp))
      in
      match values ~transform ~mat comp out with
      | got ->
          Array.length got = Array.length reference
          && (not (Array.is_empty got))
          && Array.for_all2_exn got reference ~f:(fun g r ->
              Int64.equal (Int64.bits_of_float g) (Int64.bits_of_float r))
      | exception exn ->
          Stdio.eprintf "%s: seed %d (%s) FAILED %s\n%!" tag idx (label sp) (Exn.to_string exn);
          false)

let companion_leg ~tag ~n ~k ~grid =
  let x = ints (tag ^ "_x") ~rows:n ~cols:k 7 in
  let y = ints (tag ^ "_y") ~rows:k ~cols:n 5 in
  let%op a = x +* "ik; kj => ij" y in
  let%op d = a ++ "ii => i" in
  let comp = Train.forward d in
  let mat = [ a.Tensor.value ] in
  let opt =
    Context.lowered_for_decisions ~materialized:mat (Context.cpu ()) comp Ir.Indexing.Empty
  in
  let offered = seeds opt in
  Stdio.eprintf "%s offered (not part of the golden): %s\n%!" tag
    (String.concat ~sep:"; " (List.map offered ~f:label));
  p (tag ^ ": premise: the matmul site writes the materialized product") (site_writes opt a);
  p_exists (tag ^ ": premise: CPU tensorized seeds are offered") offered ~f:(fun sp -> sp.sk_mma);
  p_all (tag ^ ": every offered seed passes validate_parallel") offered ~f:(validates ~tag opt);
  p_none (tag ^ ": no offered seed is the whole-triple form") offered ~f:whole_triple;
  if grid then p_none (tag ^ ": no offered seed is a Grid shape") offered ~f:grid_shape;
  let named ~what =
    List.exists (refutations opt) ~f:(fun w ->
        String.is_substring w ~substring:"gh-ocannl-1561"
        && String.is_substring w ~substring:what
        && String.is_substring w ~substring:(Ir.Tnode.debug_name d.Tensor.value))
  in
  p
    (tag ^ ": the whole-triple form is refuted naming the companion's node")
    (named ~what:"whole-triple");
  if grid then
    p
      (tag ^ ": the packed Grid shapes are refuted naming the companion's node")
      (named ~what:"packed Grid");
  p_executed ~tag ~mat comp d (List.mapi offered ~f:(fun i sp -> (i, sp)))

(* Leg 1: the issue's routine. *)
let () = companion_leg ~tag:"diag8" ~n:8 ~k:32 ~grid:false

(* Leg 2: tall enough for the Grid shapes. *)
let () = companion_leg ~tag:"diag64" ~n:64 ~k:32 ~grid:true

(* Leg 3: the control, no companion. *)
let () =
  let tag = "alone64" in
  let x = ints "alone_x" ~rows:64 ~cols:32 7 in
  let y = ints "alone_y" ~rows:32 ~cols:64 5 in
  let%op a = x +* "ik; kj => ij" y in
  let comp = Train.forward a in
  let opt = Context.lowered_for_decisions (Context.cpu ()) comp Ir.Indexing.Empty in
  let offered = seeds opt in
  Stdio.eprintf "%s offered (not part of the golden): %s\n%!" tag
    (String.concat ~sep:"; " (List.map offered ~f:label));
  p_exists "control: the whole-triple form is offered" offered ~f:whole_triple;
  p_exists "control: a packed Grid shape is offered" offered ~f:(fun sp -> sp.sk_mma && sp.sk_grid);
  p_all "control: every offered seed passes validate_parallel" offered ~f:(validates ~tag opt);
  p_none "control: no refutation names a companion" (refutations opt)
    ~f:(String.is_substring ~substring:"gh-ocannl-1561")

(* Leg 4: an epilogue tail, the fusion flavors judged apart. The bias-add + relu tail beside the
   materialized product is a companion of the unfused flavor, which loses the hardware-binding
   shapes; the fused flavor's demand skips the tail its fusion relocates into the accumulation nest,
   so it keeps them. The claims cover the unfused flavor whole and the fused flavor's whole-triple
   form, the shape this issue is about: the fused flavor's k-blocked packed twins are a different
   matter -- [Fuse_epilogue] refuses a [Tile_mma] under a k-block loop, whatever the companions. *)
let () =
  let tag = "tail64" in
  let x = ints "tail_x" ~rows:64 ~cols:32 7 in
  let y = ints "tail_y" ~rows:32 ~cols:64 5 in
  let bias =
    TDSL.ndarray
      (Array.init 64 ~f:(fun i -> Float.of_int (i % 5) -. 2.))
      ~label:[ "tail_bias" ] ~output_dims:[ 64 ] ()
  in
  let%op prod = x +* "ik; kj => ij" y in
  let%op out = relu (prod + bias) in
  let comp = Train.forward out in
  let mat = [ prod.Tensor.value ] in
  let opt =
    Context.lowered_for_decisions ~materialized:mat (Context.cpu ()) comp Ir.Indexing.Empty
  in
  let offered = seeds opt in
  Stdio.eprintf "%s offered (not part of the golden): %s\n%!" tag
    (String.concat ~sep:"; " (List.map offered ~f:label));
  let indexed = List.mapi offered ~f:(fun i sp -> (i, sp)) in
  let unfused = List.filter indexed ~f:(fun (_, sp) -> not sp.sk_epilogue) in
  let fused_whole = List.filter indexed ~f:(fun (_, sp) -> sp.sk_epilogue && whole_triple sp) in
  p_none "tail64: no unfused seed binds a hardware dimension" unfused ~f:(fun (_, sp) ->
      grid_shape_or_whole sp);
  p_all "tail64: every unfused seed passes validate_parallel" unfused ~f:(fun (_, sp) ->
      validates ~tag opt sp);
  p_executed ~what:"unfused seed" ~tag ~mat comp out unfused;
  p_all "tail64: the fused flavor keeps the whole-triple form, and it passes validate_parallel"
    fused_whole ~f:(fun (_, sp) -> validates ~tag opt sp);
  p_executed ~what:"fused whole-triple seed" ~tag ~mat comp out fused_whole

let () = Stdio.printf "cpu_mma_companion_coverage: done\n"
