(* gh-ocannl-1609 (after gh-ocannl-1561): the CPU tensorized matmul flavors that bind a hardware
   dimension, beside a companion nest, are valid on the C backends' fork-join model, and the family
   tree EXCLUDES them as search economy rather than refuting them.

   The whole-triple [Tile_mma] form retypes the zeroing's column loop to a [Workgroup] axis at the
   [Tensorize] lane width [m_nj] (plus a [Grid] row split when [sk_bm > 0]); the packed Grid shapes
   [Grid]-type their row-block loop. Under hardware-thread launch semantics a nest beside the matmul
   writing a materialized node would have to sit under loops covering those dimensions, and
   gh-ocannl-1561 refuted the flavors on it. The C backends do not launch: each pool-parallel [Grid]
   loop is one dispatch joined before the next statement, and code outside [Grid] loops runs once
   ([Low_level.validate_parallel ~fork_join], gh-ocannl-1458, which the cc compile path passes). So
   the companions either run serially (the whole-triple form, whose active [Workgroup] lane no
   companion loop covers) or take the default preset's per-nest geometry (the packed Grid shapes,
   [Sketch_families.cpu_companion_geometry]). The tree offers neither by default -- [Excluded] until
   a measurement shows them winning -- and [Autotune.lift_cpu_companions] lifts them.

   Seeding uses synthetic limits (a 32-byte vector file), so the offered set is machine-independent.
   The renderer's launch floor is disabled ([cc_grid_min_updates=0]) so every [Grid] loop of these
   small routines renders on the pool, the companions' included.

   1. diag8: an 8x32 by 32x8 product, [a] materialized, read through its diagonal [d] (the
   companion). The production tree offers no whole-triple seed and excludes the form naming [d]; the
   lift recovers it. Every recovered seed passes [validate_parallel ~fork_join:true], fails the
   hardware-thread mode (the whole-triple form's companion misses the active [Workgroup] slot -- the
   negative control that shows the fork-join mode is what admits it), leaves the companion serial,
   and computes [d] bit for bit across two runs (integer inputs, exact sums; the second run catches
   a missed zeroing). 2. diag64: tall enough for the Grid shapes (four 16-row blocks); the same,
   plus the packed Grid shapes are excluded and recovered, and in each recovered Grid seed the
   companion's own [Grid] loop renders as a pool dispatch. 3. Control, the 64-row product with no
   companion: the whole-triple form and a packed Grid shape are offered, and nothing is excluded on
   companions. 4. A bias-add + relu tail beside the materialized product: the unfused flavor offers
   no hardware-binding shape and its lift recovers them, all computing the output bit for bit; the
   fused flavor (whose fusion relocates the tail) keeps the whole-triple form. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = "cc"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let () = Hashtbl.set Utils.config_file_args ~key:"cc_grid_min_updates" ~data:"0"
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

let seeding opt = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt
let seeds opt = snd (seeding opt)

let matmul_tree opt =
  match fst (seeding opt) with
  | Autotune.Seeding_matmul { tree; _ } -> Some tree
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> None

(* The lifted tree's leaves the production seeding does not offer, in tree order. *)
let recovered opt =
  match matmul_tree opt with
  | None -> []
  | Some tree ->
      let offered = seeds opt in
      List.filter
        (Sspace.leaves (Autotune.lift_cpu_companions tree))
        ~f:(fun sp -> not (List.mem offered sp ~equal:Poly.equal))

let exclusions opt =
  match matmul_tree opt with None -> [] | Some tree -> List.map (Sspace.exclusions tree) ~f:snd

let label sp = Autotune.sketch_params_label sp
let whole_triple (sp : Autotune.sketch_params) = sp.sk_mma && sp.sk_bk = 0
let packed_grid (sp : Autotune.sketch_params) = sp.sk_mma && sp.sk_bk > 0 && sp.sk_grid
let grid_shape (sp : Autotune.sketch_params) = sp.sk_grid || (whole_triple sp && sp.sk_bm > 0)
let grid_shape_or_whole sp = grid_shape sp || whole_triple sp

let site_writes opt (t : Tensor.t) =
  match fst (seeding opt) with
  | Autotune.Seeding_matmul { site; _ } -> Tn.equal site.Autotune.m_d t.Tensor.value
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> false

let applied opt sp =
  let s = scratch opt in
  Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp s) s

(* Whether a seed's schedule constructs, applies and passes [validate_parallel] in the given mode;
   with [~loud] a failure goes to stderr. *)
let validates ?(loud = true) ~fork_join ~tag opt sp =
  match
    let post = applied opt sp in
    LL.validate_parallel ~fork_join post.LL.optimize_ctx.placements post.LL.llc
  with
  | () -> true
  | exception exn ->
      if loud then Stdio.eprintf "%s: %s failed: %s\n%!" tag (label sp) (Exn.to_string exn);
      false

(* The C backends validate in fork-join mode: they bind no hardware [Grid] index
   ([C_syntax.compile_proc]'s [~fork_join]), and the tuner passes [is_cpu]. *)
let validates_cc = validates ~fork_join:true

let written (llc : LL.t) =
  let acc = ref [] in
  Ll_test.walk llc ~on_set:(fun tn -> acc := tn :: !acc);
  !acc

(* The outermost loops of the top-level nests writing [tn] in [llc]. *)
let writer_loops (llc : LL.t) tn =
  List.filter_map (LL.flat_lines [ llc ]) ~f:(function
    | LL.For_loop { index; axis; _ } as nest when List.mem (written nest) tn ~equal:Tn.equal ->
        Some (index, axis)
    | _ -> None)

let grid_written opt sp tn =
  List.exists (writer_loops (applied opt sp).LL.llc tn) ~f:(fun (_, axis) ->
      LL.equal_axis_type axis LL.Grid)

(* Positive integer operands: every product sum is exact in single precision and the diagonal is
   clear of the zero init. *)
let ints label ~rows ~cols modulus =
  TDSL.ndarray
    (Array.init (rows * cols)
       ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus ~offset:1. ~stride:1.))
    ~label:[ label ] ~output_dims:[ rows; cols ] ()

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Compile [comp] (transformed, if given), run it twice -- a missed zeroing accumulates twice -- and
   read [out]. Returns the values and the transformed code. *)
let values ?transform ~name ~mat comp (out : Tensor.t) =
  let ctx = Context.decide_materialized (Context.cpu ()) mat in
  let post = ref None in
  let lowered_transform opt =
    match transform with
    | None -> [ opt ]
    | Some f ->
        let o = f opt in
        post := Some o;
        [ o ]
  in
  let ctx, routine = Context.compile ~lowered_transform ctx (named name comp) Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let ctx = Context.run ctx routine in
  let v = Context.get_values ctx out.Tensor.value in
  Context.release ctx;
  (v, !post)

(* The reference: the untuned compile. Its values must discriminate (vary, and clear the zero init),
   or a seed that zeroes or drops cells could still match. *)
let reference ~tag ~mat comp (out : Tensor.t) =
  let v, _ = values ~name:(tag ^ "_ref") ~mat comp out in
  p
    (tag ^ ": premise: the reference output varies and has no zero cell")
    (Array.length v > 1
    && (not (Array.for_all v ~f:(fun x -> Float.equal x v.(0))))
    && Array.for_all v ~f:(fun x -> not (Float.equal x 0.)));
  v

(* Every seed of [indexed] (positions in [enum opt]), compiled and run on cc, gives [out] bit for
   bit as the untuned compile does. The seed is re-derived by position from the compile's own
   lowering. [on_post] sees each seed's transformed code and its routine name. *)
let p_executed ?(what = "recovered seed") ?(on_post = fun _ _ _ -> ()) ~tag ~enum ~mat ~expected
    comp (out : Tensor.t) indexed =
  p_all (Printf.sprintf "%s: every %s computes the output bit for bit" tag what) indexed
    ~f:(fun (idx, sp) ->
      let transform opt =
        match List.nth (enum opt) idx with
        | Some sp' when String.equal (label sp') (label sp) ->
            Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp' opt) opt
        | _ ->
            failwith
              (Printf.sprintf "seed %d (%s) is gone from the compile's lowering" idx (label sp))
      in
      let name = Printf.sprintf "%s_seed%d" tag idx in
      match values ~transform ~name ~mat comp out with
      | got, post ->
          Option.iter post ~f:(fun post -> on_post sp name post);
          Array.length got = Array.length expected
          && Array.for_all2_exn got expected ~f:(fun g r ->
              Int64.equal (Int64.bits_of_float g) (Int64.bits_of_float r))
      | exception exn ->
          Stdio.eprintf "%s: seed %d (%s) FAILED %s\n%!" tag idx (label sp) (Exn.to_string exn);
          false)

(* The companion's Grid loops of a compiled seed render as pool dispatches: the pool loop's bound is
   the renderer's own chunk-bound name for the loop's identifier. *)
let pool_rendered_writers ~companion name (post : LL.optimized) =
  let src = Generated.read name in
  let _, _, hi = Ir.C_syntax.grid_chunk_derivations in
  let grid_loops =
    List.filter (writer_loops post.LL.llc companion) ~f:(fun (_, axis) ->
        LL.equal_axis_type axis LL.Grid)
  in
  (not (List.is_empty grid_loops))
  && List.for_all grid_loops ~f:(fun (index, _) ->
      let ident = Ir.Indexing.symbol_ident index in
      String.is_substring src
        ~substring:(Printf.sprintf "%s < %s;" ident (Ir.C_syntax.derived_name hi ident)))

let named_exclusion ~what ~companion opt =
  List.exists (exclusions opt) ~f:(fun w ->
      String.is_substring w ~substring:"gh-ocannl-1609"
      && String.is_substring w ~substring:what
      && String.is_substring w ~substring:(Tn.debug_name companion))

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
  let back = recovered opt in
  Stdio.eprintf "%s offered (not part of the golden): %s\n%s recovered: %s\n%!" tag
    (String.concat ~sep:"; " (List.map offered ~f:label))
    tag
    (String.concat ~sep:"; " (List.map back ~f:label));
  let companion = d.Tensor.value in
  p (tag ^ ": premise: the matmul site writes the materialized product") (site_writes opt a);
  p_exists (tag ^ ": premise: CPU tensorized seeds are offered") offered ~f:(fun sp -> sp.sk_mma);
  p_none (tag ^ ": no offered seed is the whole-triple form") offered ~f:whole_triple;
  if grid then p_none (tag ^ ": no offered seed is a Grid shape") offered ~f:grid_shape;
  p
    (tag ^ ": the whole-triple form is excluded naming the companion's node")
    (named_exclusion ~what:"whole-triple" ~companion opt);
  if grid then
    p
      (tag ^ ": the packed Grid shapes are excluded naming the companion's node")
      (named_exclusion ~what:"packed Grid" ~companion opt);
  p_exists (tag ^ ": the lift recovers the whole-triple form") back ~f:whole_triple;
  if grid then p_exists (tag ^ ": the lift recovers a packed Grid shape") back ~f:packed_grid;
  p_all (tag ^ ": every recovered seed binds a hardware dimension") back ~f:grid_shape_or_whole;
  p_all
    (tag ^ ": every recovered seed passes validate_parallel in fork-join mode")
    back ~f:(validates_cc ~tag opt);
  p_none
    (tag ^ ": no recovered whole-triple seed passes the hardware-thread mode")
    (List.filter back ~f:whole_triple)
    ~f:(validates ~loud:false ~fork_join:false ~tag opt);
  p_none (tag ^ ": every recovered whole-triple seed leaves the companion serial")
    (List.filter back ~f:whole_triple) ~f:(fun sp -> grid_written opt sp companion);
  if grid then
    p_all (tag ^ ": every recovered packed Grid seed gives the companion a Grid loop")
      (List.filter back ~f:packed_grid) ~f:(fun sp -> grid_written opt sp companion);
  let expected = reference ~tag ~mat comp d in
  let pool_ok = ref true in
  p_executed ~tag ~enum:recovered ~mat ~expected comp d
    ~on_post:(fun sp name post ->
      if packed_grid sp && not (pool_rendered_writers ~companion name post) then (
        Stdio.eprintf "%s: %s: the companion's Grid loop is not pool-rendered\n%!" tag (label sp);
        pool_ok := false))
    (List.mapi back ~f:(fun i sp -> (i, sp)));
  if grid then
    p (tag ^ ": every recovered packed Grid seed renders the companion as a pool dispatch") !pool_ok

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
  p_exists "control: a packed Grid shape is offered" offered ~f:packed_grid;
  p_all "control: every offered seed passes validate_parallel in fork-join mode" offered
    ~f:(validates_cc ~tag opt);
  p_none "control: no exclusion names a companion" (exclusions opt)
    ~f:(String.is_substring ~substring:"gh-ocannl-1609");
  (* Over the offered seeds, nothing is excluded on companions, so the lift adds nothing. *)
  p_empty "control: the lift recovers nothing" ~over:offered (recovered opt)

(* Leg 4: an epilogue tail, the fusion flavors judged apart. The bias-add + relu tail beside the
   materialized product is a companion of the unfused flavor, whose hardware-binding shapes are
   excluded and recovered by the lift; the fused flavor's demand skips the tail its fusion relocates
   into the accumulation nest, so it keeps them. The fused flavor's k-blocked packed twins are a
   different matter, whatever the companions: [Fuse_epilogue] refuses a [Tile_mma] under a k-block
   loop, and the tree refutes them on it (gh-ocannl-1612, pinned by cpu_seed_validity). *)
let () =
  let tag = "tail64" in
  let x = ints "tail_x" ~rows:64 ~cols:32 7 in
  let y = ints "tail_y" ~rows:32 ~cols:64 5 in
  let bias =
    TDSL.ndarray
      (Array.init 64 ~f:(fun i -> Float.of_int (300 + (i % 5))))
      ~label:[ "tail_bias" ] ~output_dims:[ 64 ] ()
  in
  let%op prod = x +* "ik; kj => ij" y in
  (* The bias lifts every sum above zero, so the relu passes every cell through and the output stays
     discriminating. *)
  let%op out = relu (prod + bias) in
  let comp = Train.forward out in
  let mat = [ prod.Tensor.value ] in
  let opt =
    Context.lowered_for_decisions ~materialized:mat (Context.cpu ()) comp Ir.Indexing.Empty
  in
  let offered = seeds opt in
  let back = recovered opt in
  Stdio.eprintf "%s offered (not part of the golden): %s\n%s recovered: %s\n%!" tag
    (String.concat ~sep:"; " (List.map offered ~f:label))
    tag
    (String.concat ~sep:"; " (List.map back ~f:label));
  let indexed = List.mapi offered ~f:(fun i sp -> (i, sp)) in
  let unfused = List.filter indexed ~f:(fun (_, sp) -> not sp.sk_epilogue) in
  let fused_whole = List.filter indexed ~f:(fun (_, sp) -> sp.sk_epilogue && whole_triple sp) in
  let back_indexed = List.mapi back ~f:(fun i sp -> (i, sp)) in
  p_none "tail64: no unfused seed binds a hardware dimension" unfused ~f:(fun (_, sp) ->
      grid_shape_or_whole sp);
  p_exists "tail64: the lift recovers the unfused whole-triple form" back ~f:(fun sp ->
      (not sp.sk_epilogue) && whole_triple sp);
  p_exists "tail64: the lift recovers an unfused packed Grid shape" back ~f:(fun sp ->
      (not sp.sk_epilogue) && packed_grid sp);
  p_all "tail64: every recovered seed passes validate_parallel in fork-join mode" back
    ~f:(validates_cc ~tag opt);
  p_all "tail64: every unfused seed passes validate_parallel in fork-join mode" unfused
    ~f:(fun (_, sp) -> validates_cc ~tag opt sp);
  let expected = reference ~tag ~mat comp out in
  p_executed ~what:"unfused seed" ~tag ~enum:seeds ~mat ~expected comp out unfused;
  p_executed ~tag:(tag ^ "_back") ~enum:recovered ~mat ~expected comp out back_indexed;
  p_all "tail64: the fused flavor keeps the whole-triple form, and it passes validate_parallel"
    fused_whole ~f:(fun (_, sp) -> validates_cc ~tag opt sp);
  p_executed ~what:"fused whole-triple seed" ~tag:(tag ^ "_fused") ~enum:seeds ~mat ~expected comp
    out fused_whole

let () = Stdio.printf "cpu_mma_companion_coverage: done\n"
