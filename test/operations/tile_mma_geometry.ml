(* The register-tile geometry as a schedule decision (gh-ocannl-619).

   The C-tile geometry of the register-tiled [Tile_mma] rendering — [rm] rows of [rn] vectors of
   [lanes], held across the k-loop — used to be picked inside the renderer alone: not part of the
   schedule, not visible to the tuner, sweepable only by patching the renderer (gh-ocannl-575), and
   the dimension that decided whether gcc's accumulator spill fired (gh-ocannl-614). It now travels
   on [Schedule.Tensorize]'s [tile] into [Low_level.Tile_mma], where the C backends honour it
   exactly or decline to the scalar fallback with a named rule; the ranking model that picks for an
   unrequested site and the fit rules a request must pass live in [Ir.Register_tile], and the sketch
   seeding twins every CPU tensorized leaf with that module's [alternatives]. This test pins:

   - The model, backend-independently: the default is a geometry the fit rules accept, the
   alternatives never repeat it and all pass the rules, a column extent below one vector yields
   nothing, and a descriptive table of what the model says on the shapes the issue argued over (a
   change detector, not a claim).

   - The emission, on the C backends (every leg is [Verdict.skipped] elsewhere — the register tiling
   is theirs): a requested geometry renders as requested (the emitted header names it and says it
   came from the schedule) and matches the serial twin bitwise; a request the rules reject — over
   the live-register budget, or wider than the column extent — declines to the scalar fallback, so
   the routine's label says [Scalar_fallback] rather than crediting a tile that never ran, and the
   values still match; the unrequested rendering carries exactly the geometry
   [Register_tile.default] computes for the site (the relationship, not a restated number).

   - The cache format: a saved [Tensorize] with a geometry round-trips through its sexp, and one
   without omits the field, so entries written before gh-ocannl-619 keep parsing. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module RT = Ir.Register_tile
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let skipped = Verdict.skipped ~backend:backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* Zeros compare equal to zeros; every reference is pinned nonzero where it is produced so the
   parity claims have content (gh-ocannl-481 item 3). *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* === The model === *)
let () =
  let shapes =
    (* (vector bytes, element bytes, m, n): NEON/AVX2/AVX-512 f32 at the n = 512 the gh-575 sweeps
       measured, the 64-column site the tree test seeds, gh-575's n = 40 step-down, pure-fp16 on
       32-byte vectors, and a two-row site. *)
    [
      (16, 4, 64, 512);
      (32, 4, 64, 512);
      (64, 4, 64, 512);
      (32, 4, 64, 64);
      (64, 4, 64, 40);
      (32, 2, 64, 512);
      (32, 4, 2, 48);
    ]
  in
  Stdio.printf "model: vector_bytes elt_bytes m n -> default | alternatives\n";
  List.iter shapes ~f:(fun (vector_bytes, elt_bytes, m, n) ->
      let dflt = RT.default ~vector_bytes ~elt_bytes ~m ~n in
      let alts = RT.alternatives ~vector_bytes ~elt_bytes ~m ~n in
      Stdio.printf "  %2d %d %3d %4d -> %s | %s\n" vector_bytes elt_bytes m n
        (Option.value_map dflt ~default:"none" ~f:RT.to_string)
        (if List.is_empty alts then "none"
         else String.concat ~sep:", " (List.map alts ~f:RT.to_string)));
  let accepted t (vector_bytes, elt_bytes, m, n) =
    Result.is_ok (RT.check ~vector_bytes ~elt_bytes ~m ~n t)
  in
  p_all "the default geometry passes the fit rules on every shape" shapes ~f:(fun shape ->
      let vector_bytes, elt_bytes, m, n = shape in
      match RT.default ~vector_bytes ~elt_bytes ~m ~n with
      | Some t -> accepted t shape
      | None -> false);
  p_all "no alternative repeats the default" shapes ~f:(fun (vector_bytes, elt_bytes, m, n) ->
      let dflt = RT.default ~vector_bytes ~elt_bytes ~m ~n in
      List.for_all (RT.alternatives ~vector_bytes ~elt_bytes ~m ~n) ~f:(fun t ->
          not (Option.equal RT.equal (Some t) dflt)));
  p_all "every alternative passes the fit rules" shapes ~f:(fun shape ->
      let vector_bytes, elt_bytes, m, n = shape in
      List.for_all (RT.alternatives ~vector_bytes ~elt_bytes ~m ~n) ~f:(fun t -> accepted t shape));
  p_all "every alternative is peel-free, or the register-budget cap peeling at most one vector"
    shapes ~f:(fun (vector_bytes, elt_bytes, m, n) ->
      List.for_all (RT.alternatives ~vector_bytes ~elt_bytes ~m ~n) ~f:(fun t ->
          (* The cap: the largest [rn] the budget admits beside [rm] rows, or the column extent. *)
          let cap = min ((RT.budget ~vector_bytes - t.rm) / (t.rm + 1)) (n / t.lanes) in
          n % RT.width t = 0 || (t.rn = cap && n % RT.width t <= t.lanes)));
  p "a column extent below one vector has no default and no alternatives"
    (Option.is_none (RT.default ~vector_bytes:32 ~elt_bytes:4 ~m:64 ~n:7)
    && List.is_empty (RT.alternatives ~vector_bytes:32 ~elt_bytes:4 ~m:64 ~n:7));
  p "the budget is the widest default tile's live-register count"
    (RT.budget ~vector_bytes:32 = RT.live_registers { rm = 4; rn = 3; lanes = 8 }
    && RT.budget ~vector_bytes:16 = RT.live_registers { rm = 4; rn = 6; lanes = 4 })

(* === The emission === *)
let n = 64

type leg = Serial | Tensorized of RT.t option

(* Whole-triple tensorization over the standard layout (the shape tile_mma_declines pins as
   register-tiled), the zeroing's column loop as the lane axis; [Tensorized None] leaves the
   geometry to the renderer. *)
let whole_triple ~tile ~(out : Tn.t) (opt : LL.optimized) : Sched.schedule =
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
  let rec path (llc : LL.t) =
    match llc with
    | LL.For_loop { index; body; _ } -> (
        match
          List.filter (LL.flat_lines [ body ]) ~f:(function
            | LL.Noop | LL.Comment _ -> false
            | _ -> true)
        with
        | [ single ] -> index :: path single
        | _ -> [ index ])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  let i, j, k =
    match
      List.find_exn
        (List.map (LL.flat_lines [ opt.LL.llc ]) ~f:path)
        ~f:(fun p -> List.length p = 3)
    with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let tz, _lane = Sched.tensorize ?tile ~i ~j ~k ~simd_width:n () in
  [ ez; Sched.Retype { axis = zj; ty = LL.Workgroup }; tz ]

let compile_run ~name ~leg (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let transform (opt : LL.optimized) =
    match leg with
    | Serial -> opt
    | Tensorized tile -> Sched.apply (whole_triple ~tile ~out:out.Tensor.value opt) opt
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out.Tensor.value, routine.Context.mma)

let renderings (summary : Ir.C_syntax.mma_summary) = List.map summary.renderings ~f:snd

let register_tiled s =
  List.equal Ir.C_syntax.equal_mma_rendering (renderings s) [ Ir.C_syntax.Mma_register_tiled ]

let scalar_fallback s =
  List.equal Ir.C_syntax.equal_mma_rendering (renderings s) [ Ir.C_syntax.Mma_scalar_fallback ]

let header t =
  Printf.sprintf "Tile_mma register tiling: %dx%d C-tile of %d-lane" t.RT.rm t.RT.rn t.RT.lanes

let provenance = "geometry from the schedule"
let elt_bytes = 4

let () =
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "tmg_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tmg_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op serial = ma * mb in
  let want, _ = compile_run ~name:"tmg_serial" ~leg:Serial serial in
  let want = nonzero "tmg_serial" want in
  if not on_cpu then
    List.iter
      [
        "a requested geometry renders register-tiled";
        "the emitted header names the requested geometry and its provenance";
        "the requested-geometry rendering matches the serial twin bitwise";
        "an over-budget request is refused by the fit rules";
        "an over-budget request declines to the scalar fallback";
        "the declined routine is labeled scalar-fallback, not tensorized";
        "the declined routine still computes the serial values";
        "an over-width request declines to the scalar fallback";
        "the unrequested rendering carries the model's default geometry";
        "the unrequested header does not claim a schedule provenance";
      ]
      ~f:skipped
  else begin
    let limits = Context.hardware_limits (Context.auto ()) in
    let vector_bytes = limits.Ir.Backend_intf.simd_vector_bytes in
    let ladder = RT.simd_lane_ladder ~vector_bytes ~elt_bytes in
    let lanes = List.hd_exn ladder in
    (* Two vector columns: within the budget on every file (4*2 + 4 + 2 = 14 <= 19) and, at 16
       lanes, still no wider than n = 64. *)
    let req = { RT.rm = 4; rn = 2; lanes } in
    (* Forward code is consumed once per compile: one tensor per leg. *)
    let%op honoured = ma * mb in
    let%op ob_t = ma * mb in
    let%op ow_t = ma * mb in
    let%op dflt_t = ma * mb in
    let got, census = compile_run ~name:"tmg_requested" ~leg:(Tensorized (Some req)) honoured in
    p "a requested geometry renders register-tiled" (register_tiled census);
    let src = Generated.read "tmg_requested" in
    p "the emitted header names the requested geometry and its provenance"
      (String.is_substring src ~substring:(header req)
      && String.is_substring src ~substring:provenance);
    p_all2 "the requested-geometry rendering matches the serial twin bitwise" got want
      ~f:Float.equal;
    (* One vector column past what the budget admits beside four rows, at the narrowest width the
       file renders, so that the budget rule — not the width rule — is the one that fires. *)
    let over_budget =
      { RT.rm = 4; rn = ((RT.budget ~vector_bytes - 4) / 5) + 1; lanes = List.last_exn ladder }
    in
    p "an over-budget request is refused by the fit rules"
      (match RT.check ~vector_bytes ~elt_bytes ~m:n ~n over_budget with
      | Error why -> String.is_substring why ~substring:"budget"
      | Ok () -> false);
    let got_ob, census_ob =
      compile_run ~name:"tmg_over_budget" ~leg:(Tensorized (Some over_budget)) ob_t
    in
    p "an over-budget request declines to the scalar fallback" (scalar_fallback census_ob);
    p "the declined routine is labeled scalar-fallback, not tensorized"
      (Ir.C_syntax.equal_tensorization census_ob.Ir.C_syntax.tensorization
         Ir.C_syntax.Scalar_fallback);
    p_all2 "the declined routine still computes the serial values" got_ob want ~f:Float.equal;
    let over_width = { RT.rm = 1; rn = (n / lanes) + 1; lanes } in
    let _, census_ow =
      compile_run ~name:"tmg_over_width" ~leg:(Tensorized (Some over_width)) ow_t
    in
    p "an over-width request declines to the scalar fallback" (scalar_fallback census_ow);
    (* The relationship: the unrequested rendering IS [Register_tile.default] of the site. *)
    let dflt = Option.value_exn (RT.default ~vector_bytes ~elt_bytes ~m:n ~n) in
    let _, census_d = compile_run ~name:"tmg_default" ~leg:(Tensorized None) dflt_t in
    let src_d = Generated.read "tmg_default" in
    p "the unrequested rendering carries the model's default geometry"
      (register_tiled census_d && String.is_substring src_d ~substring:(header dflt));
    p "the unrequested header does not claim a schedule provenance"
      (not (String.is_substring src_d ~substring:provenance))
  end

(* === The cache format === *)
let () =
  let module SC = Ir.Schedule_cache in
  let with_tile =
    SC.Tensorize
      {
        i = SC.Base 0;
        j = SC.Base 1;
        k = SC.Base 2;
        simd_width = 8;
        tile = Some { RT.rm = 4; rn = 2; lanes = 8 };
      }
  in
  let without =
    SC.Tensorize { i = SC.Base 0; j = SC.Base 1; k = SC.Base 2; simd_width = 8; tile = None }
  in
  let round_trip op = SC.saved_optop_of_sexp (SC.sexp_of_saved_optop op) in
  p "a saved Tensorize with a geometry round-trips through its sexp"
    (SC.equal_saved_optop (round_trip with_tile) with_tile);
  let bare = Sexp.to_string (SC.sexp_of_saved_optop without) in
  Stdio.printf "saved Tensorize without a geometry: %s\n" bare;
  p "a saved Tensorize without a geometry omits the field (pre-gh-619 entries keep parsing)"
    ((not (String.is_substring bare ~substring:"tile"))
    && SC.equal_saved_optop (round_trip without) without)
