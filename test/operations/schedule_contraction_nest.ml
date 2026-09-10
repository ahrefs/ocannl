(* gh-ocannl-683: a contraction over several axes is a matmul site.

   Attention's out projection [{ w_o } * attn] contracts over the weight's two input axes (head,
   head_dim), so its lowering is [d[b,s,j] += w[j,h,e] * x[b,s,h,e]] -- a reduction NEST of two
   loops. The matmul matcher took the single innermost loop as [k] and required every other loop to
   own an axis of [d], so the head loop refused the site: no matmul family was ever seeded there,
   and the kernel shipped as an untiled global-accumulator nest at 8 blocks (22% of the gpt2_mini
   step on gfx1151 at 9% of sgemm peak).

   Mechanism under test: the contraction nest is the maximal innermost suffix of loops absent from
   the accumulator ([classify_matmul]); its innermost loop is [m_k], the rest are [m_ko] -- k-loops
   lowering has already split. Every pipeline treats them as k-block loops above the one its own
   k-split mints ([Sketch_families.k_blocks]): sunk below the output roles, the staged tiles
   reloaded at, the accumulator privatized over the outermost. Single-axis sites have an empty
   [m_ko] and keep byte-identical schedules, which the existing sketch suites pin.

   Three lowered shapes: the out projection itself (the [*] operator on a weight with two input
   axes, the issue's form); a three-axis contraction whose materialized output feeds a bias+relu
   companion nest (companion coverage and epilogue twins on a multi-axis site); and the out
   projection at bf16, the one operand format every wmma backend and Metal advertise, so the
   tensorized pipelines execute through the real mma hook rather than only constructing.

   The odd-extent section (gh-ocannl-730) runs the same out projection at head_dim 12, which neither
   curated blocktile [bk] divides: the GPU family stages both operands through zero-fringe workgroup
   tiles, so the extent PADS to the block size and every geometry is seeded and executed there,
   where the k gate used to delete the family wholesale. The CPU blocktile pipeline packs outside
   that composition and keeps the gate, so it is where the gh-ocannl-683 extent label is read off.

   Executed assertions compare every candidate against a serial reference computed from the same
   discriminating inputs. For the f32 legs the values vary with every index and keep all partial
   sums exactly representable, so bitwise equality is required regardless of the accumulation order
   a tiling imposes; every compared cell of the contraction's output is required nonzero, so a
   candidate that drops a write and leaves the zero-initialized destination in place cannot pass
   (the companion leg compares the materialized pre-relu result as well as the relu'd tail, which
   legitimately zeros cells). GPU backends execute the blocktile family (workgroup-shared staging)
   and, where the device advertises a uniform-bf16 tile, the tensorized family under the tolerance
   schedule_mma_matmul documents for gfx1151's not-exactly-rounded WMMA; cc executes the CPU
   families (packed and register-tiled pipelines included), so every backend executes a
   multi-axis-contraction sketch. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Every compared cell must differ from the zero sentinel: a candidate that drops a write leaves the
   zero-initialized destination in place, which a reference holding zeros there cannot see. *)
let nonzero name (a : float array) =
  if Array.exists a ~f:(fun x -> Float.(x = 0.)) then
    failwith (name ^ ": the reference holds a zero cell -- the parity check there is vacuous");
  a

let values ctx t = Context.get_values ctx t.Tensor.value

(* A serial compile of [fwd]; returns the values of every tensor in [outs]. *)
let run_serial ~name fwd outs =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.auto ()) (named name fwd) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  List.map outs ~f:(values ctx)

let capture fwd =
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

(* The unfused seeds of a family, via the public seeding API. Synthetic no-limits keep the
   enumeration machine-independent. *)
let unfused_seeds ~is_gpu ~is_cpu ~limits opt =
  Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt
  |> List.filter ~f:(fun q -> not q.Autotune.sk_epilogue)

(* A synthetic f32 mma capability makes the tensorized GPU branch seedable machine-independently (an
   f32 tile is not a hardware format on the wmma backends, so the real capability would refute it);
   one pipelined depth so the pipelined staged twins construct over a k-block nest too. *)
let mma_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles =
            [
              ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                (8, 8, 8) );
            ];
          mma_f16_wide_acc_scopes = [];
          mma_staged_layouts = [];
          mma_pipeline_depths = [ 2 ];
        };
  }

(* Every seed's schedule applied as the pure IR transform it is -- backend-independent. Returns the
   seeds whose schedule constructs and validates, and the ones that construct but fail
   [validate_parallel]; a construction failure is reported and counted against the claim. *)
let constructs_and_validates ~tag ~what seeds opt =
  let ok = ref true in
  let valid, invalid =
    List.partition_tf seeds ~f:(fun q ->
        match Sched.apply (Autotune.sketch_schedule ~p:q opt) opt with
        | o -> (
            match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
            | () -> true
            | exception _ -> false)
        | exception exn ->
            Stdio.eprintf "%s/%s: construct FAILED: %s\n" tag what (Exn.to_string exn);
            ok := false;
            false)
  in
  p (Printf.sprintf "%s: %s seeds are proposed" tag what) (not (List.is_empty seeds));
  p (Printf.sprintf "%s: every %s seed's schedule constructs" tag what) !ok;
  (valid, invalid)

let binds_hardware q opt =
  let o = Sched.apply (Autotune.sketch_schedule ~p:q opt) opt in
  let dims = LL.launch_dims o.LL.llc in
  let product = Array.fold ~init:1 ~f:( * ) in
  (product dims.LL.grid, product dims.LL.block)

(* Execute every seed against the serial reference, each under its own armed artifact; [outs] and
   [wants] pair the compared tensors with their reference values, [close] is the per-cell agreement.
   Returns how many ran and how many matched on every compared tensor. *)
let execute_seeds ?(on_routine = fun (_ : Context.routine) -> ()) ~tag ~routine ~fwd ~outs ~wants
    ~close seeds =
  let n_ran = ref 0 and n_match = ref 0 in
  List.iter seeds ~f:(fun q ->
      Generated.arm routine;
      match
        let ctx, r =
          Context.compile
            ~lowered_transform:(fun o -> [ Sched.apply (Autotune.sketch_schedule ~p:q o) o ])
            (Context.auto ()) fwd Ir.Indexing.Empty
        in
        let ctx = Context.run ctx r in
        on_routine r;
        List.map outs ~f:(values ctx)
      with
      | gots ->
          Int.incr n_ran;
          if List.for_all2_exn gots wants ~f:(fun got want -> Array.for_all2_exn got want ~f:close)
          then Int.incr n_match
      | exception exn -> Stdio.eprintf "%s: seed FAILED: %s\n" tag (Exn.to_string exn));
  (!n_ran, !n_match)

(* One f32 leg. [build ()] returns the tensor to forward and the contraction's own output (the same
   tensor unless a companion consumes it); both are compared, and the contraction's output is the
   one required nonzero everywhere. [ko_extents] are the expected extents of the outer contraction
   loops (nest order), [nk] the innermost contraction extent. *)
let leg ~tag ~ko_extents ~nk ?(companion = false) ~build () =
  let outs_of (y, z) = if phys_equal y z then [ y ] else [ y; z ] in
  let want =
    let ((y, _) as built) = build () in
    let vals = run_serial ~name:(tag ^ "_serial") (Train.forward y) (outs_of built) in
    ignore (nonzero (tag ^ "_serial") (List.last_exn vals));
    vals
  in
  let ((cand, _) as built) = build () in
  let outs = outs_of built in
  let fwd = named (tag ^ "_sched") (Train.forward cand) in
  let opt = capture fwd in
  (match Autotune.detect_matmul opt.LL.llc with
  | None ->
      p (tag ^ ": the multi-axis contraction is detected as a matmul site") false;
      p (tag ^ ": the outer contraction loops carry the expected extents") false;
      p (tag ^ ": m_k is the innermost contraction loop") false;
      p (tag ^ ": the accumulation is in fused form") false
  | Some site ->
      p (tag ^ ": the multi-axis contraction is detected as a matmul site") true;
      p
        (tag ^ ": the outer contraction loops carry the expected extents")
        (List.equal Int.equal (List.map site.Autotune.m_ko ~f:snd) ko_extents);
      p (tag ^ ": m_k is the innermost contraction loop") (site.Autotune.m_nk = nk);
      p (tag ^ ": the accumulation is in fused form") site.Autotune.m_fma);
  (* --- GPU families: structure everywhere. --- *)
  let gpu_seeds =
    unfused_seeds ~is_gpu:true ~is_cpu:false ~limits:Ir.Backend_intf.no_hardware_limits opt
  in
  let _, gpu_invalid = constructs_and_validates ~tag ~what:"GPU blocktile" gpu_seeds opt in
  p_empty (tag ^ ": every GPU blocktile seed validates") ~over:gpu_seeds gpu_invalid;
  (* The geometry the untiled kernel never had: every seed tiles the output across a workgroup, and
     the batch-grid twins spread the batch across blocks (a 64x64 block tile of a 64x64 site is one
     block, legitimately). *)
  p_all (tag ^ ": every GPU blocktile seed binds a multi-thread workgroup") gpu_seeds ~f:(fun q ->
      snd (binds_hardware q opt) > 1);
  p
    (tag ^ ": every GPU batch-grid twin launches more than one block")
    (List.exists gpu_seeds ~f:(fun q -> q.Autotune.sk_batch_grid)
    && List.for_all gpu_seeds ~f:(fun q ->
        (not q.Autotune.sk_batch_grid) || fst (binds_hardware q opt) > 1));
  let mma_seeds =
    unfused_seeds ~is_gpu:true ~is_cpu:false ~limits:mma_limits opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_mma)
  in
  let _, mma_invalid = constructs_and_validates ~tag ~what:"GPU tensorized" mma_seeds opt in
  p_empty (tag ^ ": every GPU tensorized seed validates") ~over:mma_seeds mma_invalid;
  p
    (tag ^ ": the tensorized seeds include unstaged, staged and pipelined-staged geometries")
    (List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_bk = 0)
    && List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_bk > 0 && q.Autotune.sk_depth = 1)
    && List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_depth = 2));
  (* --- CPU families: structure everywhere, executed on cc. --- *)
  let cpu_limits =
    if on_gpu then Ir.Backend_intf.no_hardware_limits else Context.hardware_limits (Context.auto ())
  in
  let cpu_seeds = unfused_seeds ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt in
  let cpu_valid, cpu_invalid = constructs_and_validates ~tag ~what:"CPU" cpu_seeds opt in
  (* The CPU pipelines carry no companion coverage ([companion_geometry] is consulted by the GPU
     pipelines only), so on a site with a companion nest the pool-parallel CPU shapes -- the ones
     binding a Grid dimension -- decline at validation and are skipped, exactly as on a
     single-axis site; the all-serial shapes validate. *)
  (* Quantified over the whole CPU family rather than over the declines (gh-ocannl-729): on a leg
     with no companion nest nothing declines, and "every decline binds a Grid dimension" is then a
     claim about an empty set -- a golden line byte-identical to a checked one. Over [cpu_seeds] it
     says the same thing and is evaluated on every leg. The membership test comes first so that
     [binds_hardware], which re-applies the schedule, still runs only on the declines. *)
  p_none (tag ^ ": every CPU seed binding no hardware dimension validates") cpu_seeds ~f:(fun q ->
      List.mem cpu_invalid q ~equal:phys_equal && fst (binds_hardware q opt) = 1);
  if not companion then p_empty (tag ^ ": every CPU seed validates") ~over:cpu_seeds cpu_invalid
  else if on_gpu then
    (* Built under no-limits here, the CPU family has no Grid shapes to decline. *)
    skipped (tag ^ ": the Grid-bound CPU shapes decline on the uncovered companion")
  else
    p
      (tag ^ ": the Grid-bound CPU shapes decline on the uncovered companion")
      (not (List.is_empty cpu_invalid));
  let execute ~what seeds =
    let n_ran, n_match =
      execute_seeds ~tag ~routine:(tag ^ "_sched") ~fwd ~outs ~wants:want ~close:Float.equal seeds
    in
    p
      (Printf.sprintf "%s: every %s seed compiles and runs" tag what)
      (n_ran = List.length seeds && n_ran > 0);
    p
      (Printf.sprintf "%s: every %s candidate matches the serial reference bitwise" tag what)
      (n_ran = n_match)
  in
  if on_gpu then begin
    execute ~what:"GPU blocktile" gpu_seeds;
    (* The seed that shipped untiled before: its kernel now carries a workgroup-shared tile. *)
    (* This is intentionally dialect identity: [assert_emits] below searches for the language's
       literal shared-address-space qualifier; tensorization itself is read from the routine. *)
    let shared =
      if String.is_substring backend_name ~substring:"metal" then "threadgroup " else "__shared__"
    in
    Generated.arm (tag ^ "_sched");
    let q = List.hd_exn gpu_seeds in
    let _ctx, _r =
      Context.compile
        ~lowered_transform:(fun o -> [ Sched.apply (Autotune.sketch_schedule ~p:q o) o ])
        (Context.auto ()) fwd Ir.Indexing.Empty
    in
    Generated.assert_emits ~routine:(tag ^ "_sched") ~contains:shared
      (tag ^ ": the blocktiled kernel stages operands through workgroup-shared tiles");
    Stdio.eprintf "%s: %s executes the GPU families; the CPU families are structural here\n" tag
      backend_name;
    skipped (tag ^ ": the CPU seeds include the register-tiled packed pipelines");
    skipped (tag ^ ": every CPU seed compiles and runs");
    skipped (tag ^ ": every CPU candidate matches the serial reference bitwise")
  end
  else begin
    Stdio.eprintf "%s: %s cannot execute workgroup-shared staging -- GPU execution legs skipped\n"
      tag backend_name;
    skipped (tag ^ ": every GPU blocktile seed compiles and runs");
    skipped (tag ^ ": every GPU blocktile candidate matches the serial reference bitwise");
    skipped (tag ^ ": the blocktiled kernel stages operands through workgroup-shared tiles");
    (* The whole-triple form is refuted on the weight's transposed storage ([j, ..., k]), as on any
       [j,k]-stored weight; the packed forms normalize the layout and are seeded. *)
    p
      (tag ^ ": the CPU seeds include the register-tiled packed pipelines")
      (List.exists cpu_seeds ~f:(fun q -> q.Autotune.sk_mma && q.Autotune.sk_bk > 0));
    execute ~what:"CPU" cpu_valid
  end

(* The tensorized pipelines through the real mma hook: an out projection at bf16 -- the one operand
   format the wmma backends and Metal all advertise in the uniform combination -- seeded against
   [Context.hardware_limits], executing one candidate per tensorized shape this site seeds
   (unstaged, staged, pipelined-staged, batch-grid) under the 5% tolerance schedule_batched_mma uses
   for gfx1151's not-exactly-rounded WMMA, and checking that the emitted source reaches the
   backend's intrinsic rather than the scalar fallback. Where the device advertises no such tile
   (cc; an f32-only capability) the leg is reported skipped. *)
let bf16_leg ~tag ~build =
  let real_limits = Context.hardware_limits (Context.auto ()) in
  let has_uniform_bf16_tile =
    Ir.Backend_intf.advertises_mma_format real_limits ~a:Ir.Backend_intf.Mma_bf16
      ~b:Ir.Backend_intf.Mma_bf16 ~d:Ir.Backend_intf.Mma_bf16
  in
  let shapes =
    [
      ("unstaged", fun q -> q.Autotune.sk_bk = 0);
      ( "staged",
        fun q -> q.Autotune.sk_bk > 0 && q.Autotune.sk_depth = 1 && not q.Autotune.sk_batch_grid );
      ("pipelined-staged", fun q -> q.Autotune.sk_depth = 2);
      ("batch-grid", fun q -> q.Autotune.sk_batch_grid && q.Autotune.sk_depth = 1);
    ]
  in
  let skip_shape what =
    skipped (tag ^ " bf16: the " ^ what ^ " candidate compiles and runs");
    skipped (tag ^ " bf16: the " ^ what ^ " candidate agrees with the serial twin");
    skipped (tag ^ " bf16: the " ^ what ^ " candidate renders the tensor-core intrinsic")
  in
  if not (on_gpu && has_uniform_bf16_tile) then begin
    Stdio.eprintf
      "%s: %s advertises no uniform-bf16 mma tile -- the tensorized execution leg is skipped\n" tag
      backend_name;
    skipped (tag ^ " bf16: the multi-axis site seeds the backend's advertised tile");
    List.iter shapes ~f:(fun (what, _) -> skip_shape what)
  end
  else begin
    let close a b = Float.(abs (a - b) <= 0.05 * max 1. (abs b)) in
    let ref_t = build () in
    let want =
      List.hd_exn (run_serial ~name:(tag ^ "_bf16_serial") (Train.forward ref_t) [ ref_t ])
      |> nonzero (tag ^ "_bf16_serial")
    in
    let cand = build () in
    let routine = tag ^ "_bf16_mma" in
    let fwd = named routine (Train.forward cand) in
    let opt = capture fwd in
    let seeds =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:real_limits opt
      |> List.filter ~f:(fun q -> q.Autotune.sk_mma && not q.Autotune.sk_epilogue)
    in
    p
      (tag ^ " bf16: the multi-axis site seeds the backend's advertised tile")
      (not (List.is_empty seeds));
    List.iter shapes ~f:(fun (what, pick) ->
        match List.find seeds ~f:pick with
        | None ->
            Stdio.eprintf "%s: no %s tensorized seed for this site on %s\n" tag what backend_name;
            skip_shape what
        | Some q ->
            let tensorized = ref false in
            let n_ran, n_match =
              execute_seeds
                ~on_routine:(fun r ->
                  tensorized :=
                    Ir.C_syntax.equal_tensorization r.mma.Ir.C_syntax.tensorization
                      Ir.C_syntax.Tensorized)
                ~tag ~routine ~fwd ~outs:[ cand ] ~wants:[ want ] ~close [ q ]
            in
            p (tag ^ " bf16: the " ^ what ^ " candidate compiles and runs") (n_ran = 1);
            p (tag ^ " bf16: the " ^ what ^ " candidate agrees with the serial twin") (n_match = 1);
            p
              (tag ^ " bf16: the " ^ what ^ " candidate renders the tensor-core intrinsic")
              (n_ran = 1 && !tensorized))
  end

(* What a tile-geometry refutation calls the extent it judged (gh-ocannl-683). The divisibility
   gates compare a tile's k-extent against the INNERMOST contraction loop's extent [m_nk] alone --
   the outer contraction loops are k-block loops every pipeline inherits already split -- so on a
   multi-axis site a bare "b=16 does not divide k=12" names a number that is not the site's K (48
   here), misleading whoever reads a refutation log or [Ir.Schedule_space.refutations]. The
   witnesses are collected off the real family tree, never from literals; single-axis sites must
   keep the bare "k=%d" text the sketch-family goldens quote, which the control leg pins.

   Read off the CPU blocktile pipeline: since gh-ocannl-730 the GPU blocktile family PADS its
   k-extent instead of gating on it (both operands are staged through zero-fringe workgroup tiles),
   so it no longer produces a k-divisibility witness at all -- which the padded leg below asserts.
   The CPU blocktile pipeline packs into stack scratch outside that composition and keeps the gate,
   so its witnesses are where the label is still rendered. *)
let k_refutations ~name ~is_gpu build =
  let opt = capture (named name (Train.forward (build ()))) in
  match
    Autotune.matmul_sketch_tree ~is_gpu ~is_cpu:(not is_gpu)
      ~limits:Ir.Backend_intf.no_hardware_limits opt
  with
  | None -> []
  | Some tree -> Sspace.refutations tree |> List.map ~f:snd

let k_witnesses ~name ~is_gpu ~prefix build =
  k_refutations ~name ~is_gpu build |> List.filter ~f:(String.is_prefix ~prefix)

(* The padded GPU blocktile family at an awkward contraction extent (gh-ocannl-730): head_dim 12,
   which neither curated [bk] (8, 16) divides. The pipeline stages BOTH operands through zero-fringe
   workgroup tiles, so the extent pads to the block size and the family is seeded where it used to
   refute wholesale; the leaf guards the pad leaves behind are what [Schedule.Privatize] classifies
   as an iteration mask. Structure everywhere; executed against the serial reference on GPU, where
   every compared cell must agree bitwise -- the padded k slots contribute exact zeros, so a padded
   candidate is not merely close to the serial twin, it is equal to it. *)
let padded_leg ~tag ~nk ~build () =
  let ref_t = build () in
  let want =
    List.hd_exn (run_serial ~name:(tag ^ "_serial") (Train.forward ref_t) [ ref_t ])
    |> nonzero (tag ^ "_serial")
  in
  let cand = build () in
  let fwd = named (tag ^ "_sched") (Train.forward cand) in
  let opt = capture fwd in
  (match Autotune.detect_matmul opt.LL.llc with
  | None ->
      p (tag ^ ": the awkward-extent site is detected with the expected innermost extent") false
  | Some site ->
      p
        (tag ^ ": the awkward-extent site is detected with the expected innermost extent")
        (site.Autotune.m_nk = nk));
  let seeds =
    unfused_seeds ~is_gpu:true ~is_cpu:false ~limits:Ir.Backend_intf.no_hardware_limits opt
    |> List.filter ~f:(fun q -> not q.Autotune.sk_mma)
  in
  p
    (tag ^ ": the GPU blocktile family is seeded at an extent no menu bk divides")
    (not (List.is_empty seeds));
  p_all (tag ^ ": every seeded geometry pads rather than gating on the contraction extent") seeds
    ~f:(fun q ->
      List.exists (Autotune.sketch_schedule ~p:q opt) ~f:(function
        | Sched.Pad _ -> true
        | _ -> false));
  let _, invalid = constructs_and_validates ~tag ~what:"padded GPU blocktile" seeds opt in
  p_empty (tag ^ ": every padded GPU blocktile seed validates") ~over:seeds invalid;
  if on_gpu then begin
    let n_ran, n_match =
      execute_seeds ~tag ~routine:(tag ^ "_sched") ~fwd ~outs:[ cand ] ~wants:[ want ]
        ~close:Float.equal seeds
    in
    p
      (tag ^ ": every padded GPU blocktile seed compiles and runs")
      (n_ran = List.length seeds && n_ran > 0);
    p (tag ^ ": every padded candidate matches the serial reference bitwise") (n_ran = n_match)
  end
  else begin
    Stdio.eprintf
      "%s: %s cannot execute workgroup-shared staging -- padded execution legs skipped\n" tag
      backend_name;
    skipped (tag ^ ": every padded GPU blocktile seed compiles and runs");
    skipped (tag ^ ": every padded candidate matches the serial reference bitwise")
  end

(* [offset + stride * (flat index mod modulus)] over row-major [dims]: varies along every axis whose
   extent is not a multiple of [modulus]. *)
let cycle ~dims ~modulus ~offset ~stride idcs =
  let flat = Array.foldi dims ~init:0 ~f:(fun i acc d -> (acc * d) + (idcs.(i) % d)) in
  offset +. (stride *. Float.of_int (flat % modulus))

let () =
  (* --- The out projection: [{ w_o } * attn] with two input axes on the weight. --- *)
  (* Discriminating inputs: values vary with every index (the moduli are coprime with every axis
     extent), the weight is strictly negative and the activation strictly positive so no product
     and no sum is zero, and every product is a small multiple of 1/8 with partial sums far below
     2^24, so f32 addition is exact in any order. *)
  let bb = 2 and ss = 64 and jj = 64 and hh = 4 and ee = 16 in
  let w () =
    NTDSL.init ~l:"cn_w" ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee ]
      ~f:(cycle ~dims:[| jj; hh; ee |] ~modulus:11 ~offset:(-5.5) ~stride:0.5)
      ()
  in
  let att () =
    NTDSL.init ~l:"cn_att" ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee ]
      ~f:(cycle ~dims:[| bb; ss; hh; ee |] ~modulus:13 ~offset:0.25 ~stride:0.25)
      ()
  in
  leg ~tag:"out_proj" ~ko_extents:[ hh ] ~nk:ee
    ~build:(fun () ->
      let wv = w () and av = att () in
      let%op out = wv * av in
      (out, out))
    ();

  (* The same out projection at a head_dim the blocktile menu's k-extents do not divide, so the [bk]
     gate actually refutes, plus a single-axis control contracting over the same 12. *)
  let ee_odd = 12 in
  let w_odd () =
    NTDSL.init ~l:"cn_w_odd" ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee_odd ]
      ~f:(cycle ~dims:[| jj; hh; ee_odd |] ~modulus:11 ~offset:(-5.5) ~stride:0.5)
      ()
  in
  let att_odd () =
    NTDSL.init ~l:"cn_att_odd" ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee_odd ]
      ~f:(cycle ~dims:[| bb; ss; hh; ee_odd |] ~modulus:13 ~offset:0.25 ~stride:0.25)
      ()
  in
  let multi =
    k_witnesses ~name:"out_proj_witness" ~is_gpu:false ~prefix:"b=" (fun () ->
        let wv = w_odd () and av = att_odd () in
        let%op out = wv * av in
        out)
  in
  let gpu_refuted =
    k_refutations ~name:"out_proj_gpu_witness" ~is_gpu:true (fun () ->
        let wv = w_odd () and av = att_odd () in
        let%op out = wv * av in
        out)
  in
  let gpu_bk = List.filter gpu_refuted ~f:(String.is_prefix ~prefix:"bk=") in
  let single =
    k_witnesses ~name:"single_axis_witness" ~is_gpu:false ~prefix:"b=" (fun () ->
        let wv =
          NTDSL.init ~l:"cn_w1" ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ ee_odd ]
            ~f:(cycle ~dims:[| jj; ee_odd |] ~modulus:11 ~offset:(-5.5) ~stride:0.5)
            ()
        in
        let av =
          NTDSL.init ~l:"cn_att1" ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ ee_odd ]
            ~f:(cycle ~dims:[| bb; ss; ee_odd |] ~modulus:13 ~offset:0.25 ~stride:0.25)
            ()
        in
        let%op out = wv * av in
        out)
  in
  p "out_proj: the CPU k gate refutes on the site whose innermost extent it does not divide"
    (not (List.is_empty multi));
  p_all
    "out_proj: a refuted k-extent names the innermost contraction extent, not the site's whole K"
    multi ~f:(fun wit ->
      String.is_substring wit
        ~substring:"does not divide innermost contraction extent k=12 (of K=48 over 2 loops)");
  p_all "single-axis: the same refuted k-extent keeps the bare k= witness" single ~f:(fun wit ->
      String.is_suffix wit ~suffix:"does not divide k=12");
  (* gh-ocannl-730: the GPU blocktile family stages both operands, so its k-extent pads and the gate
     that used to delete the whole family here produces no witness at all. *)
  p_empty
    "out_proj: the GPU blocktile family raises no bk-divisibility refutation on the awkward extent"
    ~over:gpu_refuted gpu_bk;
  padded_leg ~tag:"out_proj_odd" ~nk:ee_odd
    ~build:(fun () ->
      let wv = w_odd () and av = att_odd () in
      let%op out = wv * av in
      out)
    ();

  (* --- A three-axis contraction with a materialized output feeding a bias+relu companion. --- *)
  let bb2 = 2 and ss2 = 32 and jj2 = 64 and gg = 2 and hh2 = 2 and ee2 = 16 in
  let x () =
    NTDSL.init ~l:"cn_x" ~prec:Ir.Ops.single ~o:[ bb2; ss2; gg; hh2; ee2 ]
      ~f:(cycle ~dims:[| bb2; ss2; gg; hh2; ee2 |] ~modulus:13 ~offset:0.25 ~stride:0.25)
      ()
  in
  let w3 () =
    NTDSL.init ~l:"cn_w3" ~prec:Ir.Ops.single ~o:[ jj2; gg; hh2; ee2 ]
      ~f:(cycle ~dims:[| jj2; gg; hh2; ee2 |] ~modulus:11 ~offset:(-5.5) ~stride:0.5)
      ()
  in
  let bias () =
    NTDSL.init ~l:"cn_bias" ~prec:Ir.Ops.single ~o:[ jj2 ]
      ~f:(fun idcs -> (Float.of_int (idcs.(0) % 3) -. 1.) *. 0.5)
      ()
  in
  leg ~tag:"three_axis_companion" ~ko_extents:[ gg; hh2 ] ~nk:ee2 ~companion:true
    ~build:(fun () ->
      let xv = x () and wv = w3 () and bv = bias () in
      let%op z = xv +* "bsghe;jghe=>bsj" wv in
      Train.set_materialized z.Tensor.value;
      let%op y = relu (z + bv) in
      (y, z))
    ();

  (* --- The out projection at bf16, through the real tensorized pipelines. --- Products are
     multiples of 1/16 of one sign with sums of typical magnitude ~12 over the 32-term contraction,
     so bf16 accumulation in the serial twin rounds little and the 5% tolerance is dominated by the
     tensor core's own rounding; the strictly negative weight against a strictly positive activation
     keeps every cell nonzero. *)
  let hb = 2 and eb = 16 in
  let wb () =
    NTDSL.init ~l:"cn_wb" ~prec:Ir.Ops.bfloat16 ~o:[ jj ] ~i:[ hb; eb ]
      ~f:(cycle ~dims:[| jj; hb; eb |] ~modulus:5 ~offset:(-2.5) ~stride:0.5)
      ()
  in
  let attb () =
    NTDSL.init ~l:"cn_attb" ~prec:Ir.Ops.bfloat16 ~b:[ bb; ss ] ~o:[ hb; eb ]
      ~f:(cycle ~dims:[| bb; ss; hb; eb |] ~modulus:3 ~offset:0.125 ~stride:0.125)
      ()
  in
  bf16_leg ~tag:"out_proj" ~build:(fun () ->
      let wv = wb () and av = attb () in
      let%op out = wv * av in
      Tn.update_prec out.Tensor.value Ir.Ops.bfloat16;
      out)
