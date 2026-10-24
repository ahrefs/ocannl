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
   nothing, the price tie between two tail-bearing tiles goes to the smaller one while the larger
   stays seeded (gh-ocannl-1180), and a descriptive table of what the model says on the shapes the
   issue argued over (a change detector, not a claim).

   - The emission, on the C backends (every leg is [Verdict.skipped] elsewhere — the register tiling
   is theirs): a requested geometry renders as requested (the emitted header names it and says it
   came from the schedule) and matches the serial twin bitwise; a request the rules reject — over
   the live-register budget, or wider than the column extent — declines to the scalar fallback, so
   the routine's label says [Scalar_fallback] rather than crediting a tile that never ran, and the
   values still match; the unrequested rendering carries exactly the geometry
   [Register_tile.default] computes for the site (the relationship, not a restated number).

   - The tails (gh-ocannl-620), on the C backends: a site the geometry divides in neither dimension
   renders its leftover columns and rows as narrower register tiles — the last column vector PARTIAL
   — with no scalar code, bitwise against the serial twin, at f32 and through the narrow-storage
   bridges at partial width; at the file's own widest lanes and again at every narrower file's width
   the file renders (gh-ocannl-1455).

   - The seeding, on the pre-schedule lowering under synthetic limits: the whole-triple seeds carry
   exactly [Register_tile.alternatives] of the site's extents beside one auto seed, and one seeded
   alternative, instantiated through [Autotune.sketch_schedule], renders that geometry.

   - The two-row twin (gh-ocannl-947): [Register_tile.rm_twin] exists exactly where the default
   takes four rows, is the widest two-row tile the budget admits, renders register-tiled and bitwise
   against the serial twin, and joins the seeds only under [autotune_register_tile_rm_twin], after
   the alternatives.

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

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec
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

(* A vector register file, its width and its register count together (gh-ocannl-1399). *)
let vf bytes registers = { Ir.Vector_file.bytes; registers }

(* === The model === *)
let () =
  let shapes =
    (* (vector bytes, vector registers, element bytes, m, n): NEON/AVX2/AVX-512 f32 at the n = 512
       the gh-575 sweeps measured, the 64-column site the tree test seeds, gh-575's n = 40
       step-down, pure-fp16 on 32-byte vectors, a two-row site, a three-row site where the
       float-priced model broke a tie by rounding (gh-ocannl-947), the two tail-bearing ties
       gh-ocannl-1180 timed (AVX2 n = 28, NEON n = 56), and SSE's 16-byte file on 16 registers at n
       = 512 and at the tail-free n = 96 AVX-512's 4 x 6 won (gh-ocannl-1285). *)
    [
      (16, 32, 4, 64, 512);
      (32, 16, 4, 64, 512);
      (64, 32, 4, 64, 512);
      (32, 16, 4, 64, 64);
      (64, 32, 4, 64, 40);
      (32, 16, 2, 64, 512);
      (32, 16, 4, 2, 48);
      (16, 32, 4, 3, 53);
      (32, 16, 4, 64, 28);
      (16, 32, 4, 64, 56);
      (16, 16, 4, 64, 512);
      (64, 32, 4, 64, 96);
    ]
  in
  Stdio.printf "model: vector_bytes vector_registers elt_bytes m n -> default | alternatives\n";
  List.iter shapes ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      let dflt = RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n in
      let alts = RT.alternatives ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n in
      Stdio.printf "  %2d %2d %d %3d %4d -> %s | %s\n" vector_bytes vector_registers elt_bytes m n
        (Option.value_map dflt ~default:"none" ~f:RT.to_string)
        (if List.is_empty alts then "none"
         else String.concat ~sep:", " (List.map alts ~f:RT.to_string)));
  let accepted t (vector_bytes, vector_registers, elt_bytes, m, n) =
    Result.is_ok (RT.check ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n t)
  in
  p_all "the default geometry passes the fit rules on every shape" shapes ~f:(fun shape ->
      let vector_bytes, vector_registers, elt_bytes, m, n = shape in
      match RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n with
      | Some t -> accepted t shape
      | None -> false);
  p_all "no alternative repeats the default" shapes
    ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      let dflt = RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n in
      List.for_all
        (RT.alternatives ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
        ~f:(fun t -> not (Option.equal RT.equal (Some t) dflt)));
  p_all "every alternative passes the fit rules" shapes ~f:(fun shape ->
      let vector_bytes, vector_registers, elt_bytes, m, n = shape in
      List.for_all
        (RT.alternatives ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
        ~f:(fun t -> accepted t shape));
  (* Pass counts at one width: what the price ranks by, so "ties the default" is "takes as many
     passes". *)
  let passes ~n (t : RT.t) = (n + RT.width t - 1) / RT.width t in
  p_all
    "every alternative is the largest tail-free width, the register-budget cap with a tail of at \
     most one vector, or the largest tile tying a tail-bearing default"
    shapes ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      let dflt =
        Option.value_exn (RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
      in
      List.for_all
        (RT.alternatives ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
        ~f:(fun t ->
          (* The cap: the largest [rn] the budget admits beside [rm] rows, or the column extent. *)
          let cap = min ((RT.budget ~vector_registers - t.rm) / (t.rm + 1)) (n / t.lanes) in
          let tail_free rn = n % (rn * t.lanes) = 0 in
          (tail_free t.rn && not (List.exists (List.range (t.rn + 1) (cap + 1)) ~f:tail_free))
          || (t.rn = cap && n % RT.width t <= t.lanes)
          || n % RT.width dflt <> 0
             && t.lanes = dflt.lanes
             && passes ~n t = passes ~n dflt
             && t.rn = min (RT.rn_cap ~vector_registers) (n / t.lanes)));
  (* The tie rule (gh-ocannl-1180), as the structure it implies rather than restated picks: on every
     site a tail-bearing default takes the FEWEST vector columns that reach its pass count (any
     fewer columns would take another pass), and wherever a larger tile within the model's ceiling
     ties it -- the pick of the larger-tile rule this replaced -- that larger tile is still seeded.
     Swept at every row count below four and at four. *)
  let tie_sites =
    List.concat_map
      [ (16, 32, 4); (16, 32, 2); (16, 16, 4); (32, 16, 4); (32, 16, 2); (64, 32, 4); (64, 32, 2) ]
      ~f:(fun (vb, vr, eb) ->
        List.concat_map [ 1; 2; 3; 64 ] ~f:(fun m ->
            List.map (List.range 4 401) ~f:(fun n -> (vb, vr, eb, m, n))))
  in
  p_all "a tail-bearing default takes the fewest vector columns its pass count admits" tie_sites
    ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      match RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n with
      | None -> true
      | Some d -> n % RT.width d = 0 || d.rn = 1 || passes ~n { d with rn = d.rn - 1 } > passes ~n d);
  p_all "the largest tile tying a tail-bearing default is seeded" tie_sites
    ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      match RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n with
      | None -> true
      | Some d ->
          let top = { d with rn = min (RT.rn_cap ~vector_registers) (n / d.lanes) } in
          (* The default holds the fewest passes at its width, so the top tile always ties it. *)
          n % RT.width d = 0
          || top.rn = d.rn
          || passes ~n top = passes ~n d
             && List.mem
                  (RT.alternatives ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
                  top ~equal:RT.equal);
  (* The ranking's own structure, not a restated pick: every candidate's vector columns cost the
     same [1 + 1/rm] whatever its [rn], so at ONE width the row count scales the columns' term and
     cannot reorder the candidates -- the column choice is the same at every [m]. A 16-byte file
     renders one width per element size. The float-priced model broke equal prices by rounding at rm
     = 3 and rm = 1 and failed this at (3, 53), (1, 100) and a few dozen other sites
     (gh-ocannl-947). *)
  let sites =
    List.concat_map [ 4; 2 ] ~f:(fun elt_bytes ->
        List.concat_map (List.range 1 9) ~f:(fun m ->
            List.map (List.range 4 401) ~f:(fun n -> (elt_bytes, m, n))))
  in
  p_all "at a single-width file the row count never changes the chosen columns" sites
    ~f:(fun (elt_bytes, m, n) ->
      let cols m =
        Option.map (RT.default ~file:(vf 16 32) ~elt_bytes ~m ~n) ~f:(fun t -> (t.rn, t.lanes))
      in
      Option.equal (fun (a, b) (c, d) -> a = c && b = d) (cols m) (cols 64));
  (* The two-row twin (gh-ocannl-947), proposed only under [autotune_register_tile_rm_twin]. *)
  Stdio.printf "rm twins: %s\n"
    (String.concat ~sep:", "
       (List.map shapes ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
            Option.value_map
              (RT.rm_twin ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n)
              ~default:"none" ~f:RT.to_string)));
  p_all "the rm twin takes two rows at the default's width exactly where the default takes four"
    shapes ~f:(fun (vector_bytes, vector_registers, elt_bytes, m, n) ->
      match
        ( RT.default ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n,
          RT.rm_twin ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n )
      with
      | Some d, Some t -> d.rm = RT.rm_cap && t.rm = 2 && t.lanes = d.lanes
      | Some d, None -> d.rm < RT.rm_cap
      | None, twin -> Option.is_none twin);
  p_all "the rm twin passes the fit rules, and one more vector column would not" shapes
    ~f:(fun shape ->
      let vector_bytes, vector_registers, elt_bytes, m, n = shape in
      match RT.rm_twin ~file:(vf vector_bytes vector_registers) ~elt_bytes ~m ~n with
      | None -> true
      | Some t -> accepted t shape && not (accepted { t with rn = t.rn + 1 } shape));
  let dividing = RT.coverage ~m:64 ~n:64 { rm = 4; rn = 2; lanes = 8 } in
  p "the coverage of a dividing geometry is all full blocks"
    (dividing.m_full = 64 && dividing.n_full = 64);
  p_empty "the coverage of a dividing geometry has no column tail" ~over:[ dividing ]
    dividing.tail_widths;
  p "the coverage of a non-dividing geometry ends in a partial vector and a row band"
    (let cov = RT.coverage ~m:7 ~n:19 { rm = 4; rn = 1; lanes = 8 } in
     cov.m_full = 4 && cov.n_full = 16 && List.equal Int.equal cov.tail_widths [ 3 ]);
  p "a column tail of whole vectors is whole"
    (let cov = RT.coverage ~m:64 ~n:512 { rm = 4; rn = 6; lanes = 8 } in
     cov.n_full = 480 && List.equal Int.equal cov.tail_widths [ 8; 8; 8; 8 ]);
  p "a column extent below one vector has no default and no alternatives"
    (Option.is_none (RT.default ~file:(vf 32 16) ~elt_bytes:4 ~m:64 ~n:7)
    && List.is_empty (RT.alternatives ~file:(vf 32 16) ~elt_bytes:4 ~m:64 ~n:7));
  (* The column cap is fitted to the register COUNT by the resident floor (gh-ocannl-1285): the
     widest four-row tile keeps every operand resident and one more column would not. *)
  p_all "the column cap is the widest four-row tile resident on the register file"
    [ -1; 0; 5; 6; 8; 16; 20; 21; 24; 32; 64; 1_000_003 ] ~f:(fun vector_registers ->
      let cap = RT.rn_cap ~vector_registers in
      let floor rn = RT.resident_floor { rm = RT.rm_cap; rn; lanes = 1 } in
      (cap = 0 || floor cap <= vector_registers) && floor (cap + 1) > vector_registers);
  p
    "the budget is the widest default tile's live-register count: tinyBLAS's 4 x 3 on 16 \
     registers, 4 x 6 on 32"
    (RT.budget ~vector_registers:16 = RT.live_registers { rm = 4; rn = 3; lanes = 8 }
    && RT.budget ~vector_registers:32 = RT.live_registers { rm = 4; rn = 6; lanes = 4 });
  p "a file that holds no four-row tile has no default"
    (Option.is_none (RT.default ~file:(vf 16 4) ~elt_bytes:4 ~m:64 ~n:512));
  (* A geometry minted under a narrower vector file renders on a wider one of the same register
     count (gh-ocannl-1317): a seed population or a tuned schedule from a 16-byte file keeps its
     register tile at 32 or 64 bytes instead of declining to the scalar fallback. Every geometry the
     narrower file proposes -- the default, the alternatives, the two-row twin -- over the model's
     shapes, at both element sizes and both register counts, is one the wider file's fit rules
     accept. The converse is refused: the wider file's own widths do not fit the narrower one. *)
  let nested = [ (8, 16); (16, 32); (16, 64); (32, 64) ] in
  let minted =
    List.concat_map nested ~f:(fun (narrow, wide) ->
        List.concat_map [ 16; 32 ] ~f:(fun registers ->
            List.concat_map [ 4; 2 ] ~f:(fun elt_bytes ->
                List.concat_map shapes ~f:(fun (_, _, _, m, n) ->
                    let file = vf narrow registers in
                    Option.to_list (RT.default ~file ~elt_bytes ~m ~n)
                    @ RT.alternatives ~file ~elt_bytes ~m ~n
                    @ Option.to_list (RT.rm_twin ~file ~elt_bytes ~m ~n)
                    |> List.map ~f:(fun t -> (vf wide registers, elt_bytes, m, n, t))))))
  in
  p_all ~min:(List.length shapes)
    "every geometry a narrower vector file proposes passes the fit rules of a wider one" minted
    ~f:(fun (file, elt_bytes, m, n, t) -> Result.is_ok (RT.check ~file ~elt_bytes ~m ~n t));
  let wide_defaults =
    List.concat_map nested ~f:(fun (narrow, wide) ->
        List.filter_map shapes ~f:(fun (_, registers, elt_bytes, m, n) ->
            Option.bind
              (RT.default ~file:(vf wide registers) ~elt_bytes ~m ~n)
              ~f:(fun t ->
                Option.some_if
                  (t.lanes * elt_bytes > narrow)
                  (vf narrow registers, elt_bytes, m, n, t))))
  in
  p_all ~min:(List.length shapes)
    "a wider file's default whose vector exceeds a narrower file is refused by it, naming the width"
    wide_defaults ~f:(fun (file, elt_bytes, m, n, t) ->
      match RT.check ~file ~elt_bytes ~m ~n t with
      | Error why -> String.is_substring why ~substring:"is not a width"
      | Ok () -> false)

(* === The emission === *)
let n = 64

type leg = Serial | Tensorized of RT.t option

(* Whole-triple tensorization over the standard layout (the shape tile_mma_declines pins as
   register-tiled), the zeroing's column loop as the lane axis; [Tensorized None] leaves the
   geometry to the renderer. *)
let whole_triple ~tile ~simd_width ~(out : Tn.t) (opt : LL.optimized) : Sched.schedule =
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let tz, _lane = Sched.tensorize ?tile ~i ~j ~k ~simd_width () in
  [ ez; Sched.Retype { axis = zj; ty = LL.Workgroup }; tz ]

(* [simd_width] is the zeroing's column extent (the coverage rule), [n] unless the site says
   otherwise. *)
let compile_run ?(simd_width = n) ~name ~leg (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let transform (opt : LL.optimized) =
    match leg with
    | Serial -> opt
    | Tensorized tile -> Sched.apply (whole_triple ~tile ~simd_width ~out:out.Tensor.value opt) opt
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
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
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
        "a geometry minted for a narrower vector file renders register-tiled under its own header";
        "the narrower-file rendering matches the serial twin bitwise";
        "an over-budget request is refused by the fit rules";
        "an over-budget request declines to the scalar fallback";
        "the declined routine is labeled scalar-fallback, not tensorized";
        "the declined routine still computes the serial values";
        "an over-width request declines to the scalar fallback";
        "the unrequested rendering carries the model's default geometry";
        "the unrequested header does not claim a schedule provenance";
        "the two-row twin renders register-tiled under its own header";
        "the two-row twin matches the serial twin bitwise";
      ]
      ~f:skipped
  else begin
    let limits = Context.hardware_limits (Context.auto ()) in
    let file = limits.Ir.Backend_intf.simd_vector in
    let ladder = RT.simd_lane_ladder ~vector_bytes:file.bytes ~elt_bytes in
    let lanes = List.hd_exn ladder in
    (* Two vector columns where the file holds them (4*2 + 4 + 2 = 14 <= 19, the 16-register budget)
       and, at 16 lanes, still no wider than n = 64; one where it holds only one -- a configured
       width wider than a native register divides the count, e.g. 8 registers for 64-byte vectors on
       NEON. *)
    let rn = min 2 (RT.rn_cap ~vector_registers:file.registers) in
    let req = { RT.rm = 4; rn; lanes } in
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
    (* A geometry minted for a narrower file (gh-ocannl-1317): half the narrowest width the file's
       own ladder ranks -- 16 bytes on a 32- or 64-byte file, 8 on a 16-byte one -- a width the
       renderer's model never picks here, which only a request minted elsewhere names. An 8-byte
       file, the narrowest the renderer accepts, contains no narrower one. *)
    let narrower = List.last_exn ladder / 2 in
    let n_tiled =
      "a geometry minted for a narrower vector file renders register-tiled under its own header"
    and n_parity = "the narrower-file rendering matches the serial twin bitwise" in
    if narrower * elt_bytes < 8 then (
      skipped n_tiled;
      skipped n_parity)
    else begin
      let minted = { RT.rm = 4; rn; lanes = narrower } in
      let%op minted_t = ma * mb in
      let got_n, census_n =
        compile_run ~name:"tmg_narrower" ~leg:(Tensorized (Some minted)) minted_t
      in
      p n_tiled
        (register_tiled census_n
        && String.is_substring (Generated.read "tmg_narrower") ~substring:(header minted));
      p_all2 n_parity got_n want ~f:Float.equal
    end;
    (* One vector column past what the budget admits beside four rows, at the narrowest width the
       file's ladder ranks, so that the budget rule — not the width rule — is the one that fires. *)
    let over_budget =
      {
        RT.rm = 4;
        rn = ((RT.budget ~vector_registers:file.registers - 4) / 5) + 1;
        lanes = List.last_exn ladder;
      }
    in
    p "an over-budget request is refused by the fit rules"
      (match RT.check ~file ~elt_bytes ~m:n ~n over_budget with
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
    let dflt = Option.value_exn (RT.default ~file ~elt_bytes ~m:n ~n) in
    let _, census_d = compile_run ~name:"tmg_default" ~leg:(Tensorized None) dflt_t in
    let src_d = Generated.read "tmg_default" in
    p "the unrequested rendering carries the model's default geometry"
      (register_tiled census_d && String.is_substring src_d ~substring:(header dflt));
    p "the unrequested header does not claim a schedule provenance"
      (not (String.is_substring src_d ~substring:provenance));
    (* The two-row twin of this site's default (gh-ocannl-947): the widest two-row tile the file's
       budget admits: 2x10 of 4 lanes on NEON and 2x5 of 8 on AVX2, each with a 24-column tail at n
       = 64, and 2x4 of 16 on AVX-512, where the column extent caps it tail-free. *)
    let twin = Option.value_exn (RT.rm_twin ~file ~elt_bytes ~m:n ~n) in
    let%op twin_t = ma * mb in
    let got_tw, census_tw = compile_run ~name:"tmg_rm_twin" ~leg:(Tensorized (Some twin)) twin_t in
    let src_tw = Generated.read "tmg_rm_twin" in
    p "the two-row twin renders register-tiled under its own header"
      (register_tiled census_tw && String.is_substring src_tw ~substring:(header twin));
    p_all2 "the two-row twin matches the serial twin bitwise" got_tw want ~f:Float.equal
  end

(* === The tails (gh-ocannl-620) === *)
(* Until gh-ocannl-620 the columns [rn * lanes] did not cover, and the rows [rm] did not, were
   peeled to scalar code — a scalar column costing about a vector slot per k step, which made the
   width a divisibility question (gh-ocannl-575) and cost 3.6x at n = 512 on a 48-wide fp16 tile.
   Now both edges are narrower register tiles: the row band at [m mod rm] rows, the column tail as
   whole vectors plus one PARTIAL vector whose lanes past its width read as zero and are not
   stored. A 6 x 19 site divides neither way at every width the renderer accepts (19 is odd, so no
   even lane count divides it; 6 is 2 mod 4): the header names the tails, no scalar accumulator is
   emitted, the partial column is a zeroed register filled at exactly its width — at f32 an
   initializer and stores of exactly the partial's lanes (gh-ocannl-1071), and for bf16 and half
   (widened to f32) the same around a whole vector of storage bits the bridges convert at full
   width (gh-ocannl-1102) — and the values match the serial twin bitwise.

   The partial's width is read from {!RT.coverage} like the rest of the geometry, not restated: it
   is three lanes at the 4-, 8- and 16-lane tiles of a 16-, 32- or 64-byte file, but ONE lane at the
   2-lane tile of an 8-byte file (gh-ocannl-1455).

   The same site is also rendered at every narrower file's width the renderer honours on this file
   (gh-ocannl-1317's [Register_tile.renderable_lanes] below the file's own ladder: 4 and 2 lanes on
   a 32- or 64-byte file, 2 on a 16-byte one, none on an 8-byte one), at f32 and through the bf16
   and half bridges: a geometry a narrower file minted keeps its tile on a wider one, so its bridges
   run at a lane count the wider file never picks for itself -- the 4-lane bridges inside an AVX2 or
   AVX-512 kernel, the 2-lane portable ones everywhere (gh-ocannl-1455). *)

(* What one tiled rendering of the tail site showed: the five tail claims, evaluated. *)
type tail_result = { tiled : bool; header : bool; no_scalar : bool; partial : bool; parity : bool }

let () =
  let m, nt, k = (6, 19, 5) in
  let claims =
    [
      "a site the tile divides in neither dimension renders register-tiled";
      "the header names the column tail and the two-row band";
      "no scalar accumulator is emitted";
      "the partial column is a zeroed register filled at exactly its width";
      "the tails match the serial twin bitwise";
    ]
  in
  let names tag = List.map claims ~f:(fun c -> tag ^ ": " ^ c) in
  let narrower_tag tag = tag ^ " at every narrower file's width" in
  let tags = [ "f32"; "bf16"; "half"; "half_ns" ] and narrower_tags = [ "f32"; "bf16"; "half" ] in
  if not on_cpu then (
    List.iter tags ~f:(fun tag -> List.iter (names tag) ~f:skipped);
    List.iter narrower_tags ~f:(fun tag -> List.iter (names (narrower_tag tag)) ~f:skipped))
  else begin
    let limits = Context.hardware_limits (Context.auto ()) in
    let file = limits.Ir.Backend_intf.simd_vector in
    let ladder = RT.simd_lane_ladder ~vector_bytes:file.bytes ~elt_bytes in
    (* A 16-column width at [lanes] -- [4x1] of 16, [4x2] of 8, [4x4] of 4, [4x8] of 2 -- capped at
       the widest four-row tile the register file holds: a 16-register file at 4 lanes (SSE, or x86
       at a forced 16 bytes) takes [4x3] of 4, whose 12 columns leave a two-vector tail rather than
       one (gh-ocannl-1285), and a 32-register file at 2 lanes takes [4x6]. Whatever the tile, the
       19 - [n_full] leftover columns end in a PARTIAL vector and 6 - 4 = 2 rows form the band; the
       header the tile must state, and the partial's width, are derived from {!RT.coverage}. *)
    let tile_at lanes =
      { RT.rm = 4; rn = min (16 / lanes) (RT.rn_cap ~vector_registers:file.registers); lanes }
    in
    let tile = tile_at (List.hd_exn ladder) in
    (* The f32 lane counts of every narrower file this one contains -- each halving of its width
       down to the 8-byte floor every rendering file meets -- that its own ladder does not rank.
       Derived from the files, not read from [RT.renderable_lanes]: that is the rule under test, and
       a gate reading it would turn its regression into a skip. *)
    let narrower =
      let rec halvings bytes = if bytes < 8 then [] else bytes :: halvings (bytes / 2) in
      halvings (file.bytes / 2)
      |> List.map ~f:(fun bytes -> bytes / elt_bytes)
      |> List.filter ~f:(fun lanes -> lanes < List.last_exn ladder)
    in
    (* Exact in bf16 and half: cells are the multiples of 1/2 in [-1.5, 1.5], so products are
       multiples of 1/4 and the five-term sums at most 11.25 — six significand bits. That both
       operands vary with both axes is not argued here but ENFORCED: [Ll_test.cycle] raises when the
       modulus divides a row-major stride, i.e. when an axis cancels, and 7 is coprime to both row
       strides (A's k = 5, B's n = 19). The hand-rolled fixture this replaces cancelled along A's
       rows once, and a row-band pass reading the wrong A row would still have matched the serial
       twin. *)
    let av = Ll_test.cycle ~dims:[| m; k |] ~modulus:7 ~offset:(-3.) ~stride:0.5 in
    let bv = Ll_test.cycle ~dims:[| k; nt |] ~modulus:7 ~offset:(-3.) ~stride:0.5 in
    (* The tail geometry [tile] renders at this site, as the emission spells it: the partial vector
       is the tail pass's last column -- its accumulator, and its address as the tail's first column
       plus the column's lane offset -- and [width] is how many of its lanes are real. *)
    let geometry (t : RT.t) =
      let cov = RT.coverage ~m ~n:nt t in
      let tail_vectors = List.length cov.tail_widths in
      let acc = Printf.sprintf "tmma_c_0_%d__" (tail_vectors - 1) in
      let at = Printf.sprintf "%d + %d" cov.n_full ((tail_vectors - 1) * t.lanes) in
      let width = Option.value (List.last cov.tail_widths) ~default:0 in
      let header =
        Printf.sprintf "; column tail %d in %d vector%s; row tail %d)" (nt - cov.n_full)
          tail_vectors
          (if tail_vectors = 1 then "" else "s")
          (m - cov.m_full)
      in
      (acc, at, width, header)
    in
    (* At f32: a zeroed register initialized from exactly [width] lanes and storing exactly
       those. *)
    let f32_partial (t : RT.t) has =
      let acc, at, width, _ = geometry t in
      let lane l = Printf.sprintf "(&tmma_d__[(tmma_i__ + 0) * %d + %s])[%d]" nt at l in
      let last = width - 1 in
      0 < width && width < t.lanes
      && has (acc ^ " = {" ^ lane 0)
      && has (lane last ^ "};")
      && has (Printf.sprintf "%s = %s[%d];" (lane last) acc last)
      && not (has (lane width))
    in
    (* Through the bridges: the partial column's storage bits are staged in a whole vector, [width]
       lanes in and [width] lanes out, and the bridge macro converts it at the full lane count --
       for half, bit-cast to a vector of [HALF_T] first, so the macro's per-lane fallback reads
       [HALF_T]s. *)
    let narrow_partial ~bridged ~via (t : RT.t) has =
      let acc, at, width, _ = geometry t in
      let bits ~qual l =
        Printf.sprintf "((%socannl_u16_alias *)&tmma_d__[(tmma_i__ + 0) * %d + %s])[%d]" qual nt at
          l
      in
      let last = width - 1 in
      0 < width && width < t.lanes
      && has ("ocannl_pv__ = {" ^ bits ~qual:"const " 0)
      && has (bits ~qual:"const " last ^ "};")
      && has (Printf.sprintf "%s = ocannl_pv__[%d];" (bits ~qual:"" last) last)
      && has (Printf.sprintf "%d, %s, (const %s *)&%s);" t.lanes acc bridged via)
      && not (has (bits ~qual:"const " width) || has (bits ~qual:"" width))
    in
    (* One serial twin per precision, then one tiled rendering per requested tile. *)
    let leg ?b_label ~tag ~prec ~partial tiles =
      let a = NTDSL.init ~l:("tmt_a_" ^ tag) ~prec ~i:[ k ] ~o:[ m ] ~f:av () in
      let b_label = Option.value b_label ~default:("tmt_b_" ^ tag) in
      let b = NTDSL.init ~l:b_label ~prec ~i:[ nt ] ~o:[ k ] ~f:bv () in
      let%op serial = a * b in
      Tn.update_prec serial.Tensor.value prec;
      let want, _ =
        compile_run ~simd_width:nt ~name:("tmt_" ^ tag ^ "_serial") ~leg:Serial serial
      in
      let want = nonzero ("tmt_" ^ tag ^ "_serial") want in
      List.map tiles ~f:(fun (name, (t : RT.t)) ->
          let%op tiled = a * b in
          Tn.update_prec tiled.Tensor.value prec;
          let got, census = compile_run ~simd_width:nt ~name ~leg:(Tensorized (Some t)) tiled in
          let src = Generated.read name in
          let has s = String.is_substring src ~substring:s in
          let _, _, _, header = geometry t in
          ( {
              tiled = register_tiled census;
              header = has header;
              no_scalar = not (has "tmma_acc__");
              partial = partial t has;
              parity = Array.equal Float.equal got want;
            },
            got,
            want ))
    in
    (* The host-width legs: one rendering each, claimed as it ran, parity with its first differing
       cell named. *)
    let host_leg ?b_label ~tag ~prec ~partial () =
      match leg ?b_label ~tag ~prec ~partial [ ("tmt_" ^ tag, tile) ] with
      | [ (r, got, want) ] -> (
          match names tag with
          | [ n_tiled; n_header; n_scalar; n_partial; n_parity ] ->
              p n_tiled r.tiled;
              p n_header r.header;
              p n_scalar r.no_scalar;
              p n_partial r.partial;
              p_all2 n_parity got want ~f:Float.equal
          | _ -> assert false)
      | _ -> assert false
    in
    (* The narrower-width legs: one rendering per narrower width, each claim over all of them. *)
    let narrower_leg ~tag ~prec ~partial () =
      let results =
        if List.is_empty narrower then []
        else
          List.map
            ~f:(fun (r, _, _) -> r)
            (leg ~tag:(tag ^ "_nw") ~prec ~partial
               (List.map narrower ~f:(fun lanes ->
                    (Printf.sprintf "tmt_%s_nw%d" tag lanes, tile_at lanes))))
      in
      let on = Printf.sprintf "a %d-byte vector file, which contains no narrower one" file.bytes in
      List.iter2_exn
        (names (narrower_tag tag))
        [
          (fun r -> r.tiled);
          (fun r -> r.header);
          (fun r -> r.no_scalar);
          (fun r -> r.partial);
          (fun r -> r.parity);
        ]
        ~f:(fun label f ->
          gated_all ~aggregation:`Environment ~min:(List.length narrower)
            ~when_:(not (List.is_empty narrower))
            ~on label results ~f)
    in
    host_leg ~tag:"f32" ~prec:Ir.Ops.single ~partial:f32_partial ();
    (* The narrow legs request the f32 leg's geometry at the same lanes: the lane count follows the
       COMPUTE precision, f32 for bf16 and — under the default policy — for half too, so a width the
       narrow storage's element size picked would be declined. They used to leave the geometry to
       the renderer, which stopped meaning a one-vector tail when NEON's 19-column default became
       the 4x3 tile of a two-vector tail (gh-ocannl-1180). *)
    let bf16_partial = narrow_partial ~bridged:"unsigned short" ~via:"ocannl_pv__" in
    let half_partial = narrow_partial ~bridged:"HALF_T" ~via:"ocannl_pc__" in
    host_leg ~tag:"bf16" ~prec:Ir.Ops.bfloat16 ~partial:bf16_partial ();
    host_leg ~tag:"half" ~prec:Ir.Ops.half ~partial:half_partial ();
    (* The same half leg with its B operand labeled like the staging typedef the tile declares
       ([ocannl_vec<lanes>hs]) ahead of its operand pointers: [ocannl_] is the emitter's namespace,
       so the node's code name takes the [n<id>_] form and the kernel compiles (gh-ocannl-1102). A
       bare name there would be shadowed by the typedef and the pointer's initializer would not
       parse. *)
    host_leg
      ~b_label:(Printf.sprintf "ocannl_vec%dhs" tile.lanes)
      ~tag:"half_ns" ~prec:Ir.Ops.half ~partial:half_partial ();
    narrower_leg ~tag:"f32" ~prec:Ir.Ops.single ~partial:f32_partial ();
    narrower_leg ~tag:"bf16" ~prec:Ir.Ops.bfloat16 ~partial:bf16_partial ();
    narrower_leg ~tag:"half" ~prec:Ir.Ops.half ~partial:half_partial ()
  end

(* === The seeding === *)
let () =
  let m, nn, k = (64, 512, 64) in
  let av =
    Array.init (m * k) ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:11 ~offset:0. ~stride:0.5)
  in
  let bv =
    Array.init (k * nn)
      ~f:(Ll_test.cycle_flat ~dims:[| k; nn |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let a = TDSL.ndarray av ~label:[ "tmg_sa" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  let b = TDSL.ndarray bv ~label:[ "tmg_sb" ] ~input_dims:[ nn ] ~output_dims:[ k ] () in
  let%op sc = a * b in
  let captured = ref None in
  let ctx = Context.auto () in
  let (_ : Context.t * Context.routine) =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx
      (named "tmg_seed_site" (Train.forward sc))
      Ir.Indexing.Empty
  in
  let opt = Option.value_exn !captured in
  (* The rm-twin gate is read per call; pin it off for the ungated claims whatever the ambient
     environment says, and on only around the gated ones. *)
  let rm_twin_gate on =
    Unix.putenv "OCANNL_AUTOTUNE_REGISTER_TILE_RM_TWIN" (if on then "true" else "false")
  in
  rm_twin_gate false;
  let whole_tiles ~limits =
    Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt
    |> List.filter ~f:(fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk = 0 && p.Autotune.sk_bm = 0)
    |> List.map ~f:(fun p -> p.Autotune.sk_tile)
  in
  (* Synthetic 16-byte vectors on 32 registers (NEON, four f32 lanes, the 6-column cap): the n = 512
     site of the gh-575 sweep. Enumerated on the pre-schedule lowering, so it holds on every
     backend. *)
  let synthetic =
    {
      Ir.Backend_intf.no_hardware_limits with
      simd_vector = { Ir.Vector_file.bytes = 16; registers = 32 };
    }
  in
  let file = synthetic.simd_vector in
  let tiles = whole_tiles ~limits:synthetic in
  Stdio.printf "seeds: whole-triple bm0 tiles on 16-byte vectors at %dx%dx%d: %s\n" m nn k
    (String.concat ~sep:", " (List.map tiles ~f:(Option.value_map ~default:"auto" ~f:RT.to_string)));
  p "the whole-triple seeds carry one auto geometry then exactly the model's alternatives"
    (List.equal (Option.equal RT.equal) tiles
       (None :: List.map (RT.alternatives ~file ~elt_bytes ~m ~n:nn) ~f:Option.some));
  p_exists "the site offers at least one alternative to time" tiles ~f:Option.is_some;
  (* A register file too small for any four-row tile is refused at the seeding, through the
     capability judgment the family tree refutes its tensorized branch with, rather than seeded and
     declined at the renderer (gh-ocannl-1285). *)
  let refutation registers =
    Autotune.tensorized_capability_refutation ~is_gpu:false ~is_cpu:true
      ~limits:{ synthetic with simd_vector = { file with registers } }
      ~a_prec:Ir.Ops.single ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single
  in
  p "a register file holding no four-row tile withholds the tensorized family"
    (Option.is_some (refutation (RT.resident_floor { rm = RT.rm_cap; rn = 1; lanes = 1 } - 1))
    && Option.is_none (refutation (RT.resident_floor { rm = RT.rm_cap; rn = 1; lanes = 1 })));
  rm_twin_gate true;
  let gated = whole_tiles ~limits:synthetic in
  rm_twin_gate false;
  Stdio.printf "seeds with the rm twin gate on: %s\n"
    (String.concat ~sep:", " (List.map gated ~f:(Option.value_map ~default:"auto" ~f:RT.to_string)));
  p "the gate adds exactly the model's two-row twin after the alternatives"
    (List.equal (Option.equal RT.equal) gated (tiles @ [ RT.rm_twin ~file ~elt_bytes ~m ~n:nn ])
    && Option.is_some (RT.rm_twin ~file ~elt_bytes ~m ~n:nn));
  if not on_cpu then
    skipped "a seeded alternative instantiates through the sketch and renders its geometry"
  else begin
    (* Under the machine's own limits the alternative must fit its register file. *)
    let limits = Context.hardware_limits (Context.auto ()) in
    let seeds = Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt in
    match
      List.find seeds ~f:(fun p ->
          p.Autotune.sk_mma && p.Autotune.sk_bk = 0 && p.Autotune.sk_bm = 0
          && Option.is_some p.Autotune.sk_tile)
    with
    | None ->
        (* Only where the machine's width divides 512 into a single affordable geometry. *)
        skipped "a seeded alternative instantiates through the sketch and renders its geometry"
    | Some seed ->
        let t = Option.value_exn seed.Autotune.sk_tile in
        let%op sc2 = a * b in
        let comp = named "tmg_seeded" (Train.forward sc2) in
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun o ->
              [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:seed o) o ])
            ctx comp Ir.Indexing.Empty
        in
        let _ctx = Context.run ctx routine in
        let src = Generated.read "tmg_seeded" in
        p "a seeded alternative instantiates through the sketch and renders its geometry"
          (register_tiled routine.Context.mma
          && String.is_substring src ~substring:(header t)
          && String.is_substring src ~substring:provenance)
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
