(* gh-ocannl-1477: under seeding, [Autotune.model_default] never displaces the seeded plan with a
   sketch that only ties the seeded plan's own sketch.

   The ranking score sums [max (compute, memory)] per kernel. The seeded plan runs its zero-fill as
   a kernel of its own, which the score charges separately, while a family leaf keeps the zero nest
   inside its one kernel, where those writes vanish under the compute leg. Every scalar leaf of a
   site ties on the compute leg however its work is launched, so the walk's first leaf -- a plain
   one, its batch loops serial inside one block -- "strictly" beat the plan, and the `performance`
   and `approximate` profiles shipped it: measured 2.6-35x slower at gpt2_mini's q/k/v and out
   projections, and 2x slower at a weight gradient (a geometry tie, no batch loops). The walks are
   now anchored at the seeded plan's own sketch on the program they search
   ([Autotune.model_seeded_anchor]).

   Four computations, the shapes [bench_rank_site] measures (b8): the q/k/v projection (interior and
   outer batch loops: the plain/[bgrid] tie), the out projection (outer batch loops), the weight
   gradient (no batch loops: geometry ties), and a two-matmul chain whose materialized sums cut it
   into keyed segments, so the per-segment substitution is anchored too. Where [model_default]
   starts from the seeded plan (CUDA and HIP, [Autotune.model_seeding]; the gate is pinned at 32
   here so the claims do not rest on the [auto] gate's device constants), the claims are: - each
   single-site contraction is seeded, and [model_explain] marks exactly one anchor in its
   whole-routine walk: the leaf of the seeded plan's own sketch, as the plan's report names it; -
   [model_default]'s pick is no fallback, and is the seeded plan or a candidate ranking STRICTLY
   below its walk's anchor (each entry of a per-segment pick below its segment's anchor). On the
   code before this fix the pick ranked equal to the anchor at every single site: a tie; - the
   picked routine computes bitwise what [Autotune.seeded_default]'s does, on inputs whose partial
   sums round (the pick is a scalar blocktile, which keeps every cell's summation order). Elsewhere
   [model_default] starts from the default pipeline, and no walk is anchored. Ranks go to stderr:
   they are device constants' quotients, not part of the golden. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The bitwise claims compare two kernel shapes over the same summation order: every product
   accumulation is an explicit FMA (the contraction licence, as [seeded_default] pins it), so a GPU
   compiler's per-kernel contraction choice cannot round them apart. The seeding gate is a
   threshold, so the seeded plan is the one gh-ocannl-1278's measurements describe. *)
let () =
  Unix.putenv "OCANNL_FP_CONTRACT" "true";
  Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "32"

let seeds_here =
  let ctx = Context.auto () in
  Option.is_some
    (Autotune.model_seeding ~backend:(Context.backend_name ctx)
       ~limits:(Context.hardware_limits ctx))

(* Full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

type site = { tag : string; single : bool; comp : Ir.Assignments.comp; out : Tensor.t }

let site ~tag ~single out =
  Train.set_materialized out.Tensor.value;
  { tag; single; comp = Train.forward out; out }

(* [d[b,s,h,e] += w[h,e,k] * x[b,s,k]]: heads are interior batch loops, the batch outer ones. *)
let qkv () =
  let nb, ns, nh, ne, nk = (8, 128, 8, 32, 256) in
  let w =
    NTDSL.ndarray
      (values ~n:(nh * ne * nk) ~stride:389)
      ~label:[ "msa_wqkv" ] ~input_dims:[ nk ] ~output_dims:[ nh; ne ] ()
  in
  let x =
    NTDSL.ndarray
      (values ~n:(nb * ns * nk) ~stride:577)
      ~label:[ "msa_xqkv" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nk ] ()
  in
  let%op d = w * x in
  site ~tag:"qkv_b8" ~single:true d

(* [d[b,s,j] += w[j,h,e] * x[b,s,h,e]]: the contraction spans two axes, the batch loops are
   outer. *)
let outproj () =
  let nb, ns, nj, nh, ne = (8, 128, 256, 8, 32) in
  let w =
    NTDSL.ndarray
      (values ~n:(nj * nh * ne) ~stride:389)
      ~label:[ "msa_wout" ] ~input_dims:[ nh; ne ] ~output_dims:[ nj ] ()
  in
  let x =
    NTDSL.ndarray
      (values ~n:(nb * ns * nh * ne) ~stride:577)
      ~label:[ "msa_xout" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nh; ne ] ()
  in
  let%op d = w * x in
  site ~tag:"outproj_b8" ~single:true d

(* [dw[o,i] += dy[k,o] * x[k,i]] over [k = b * s] rows: no batch loops, so only geometries tie. *)
let wgrad () =
  let k, no, ni = (8 * 128, 256, 1024) in
  let dy =
    NTDSL.ndarray (values ~n:(k * no) ~stride:389) ~label:[ "msa_dy" ] ~output_dims:[ k; no ] ()
  in
  let x =
    NTDSL.ndarray (values ~n:(k * ni) ~stride:577) ~label:[ "msa_x" ] ~output_dims:[ k; ni ] ()
  in
  let%op dw = dy +* "ko; ki => oi" x in
  site ~tag:"wgrad_b8" ~single:true dw

(* Two products of different shapes behind materialized sums: keyed matmul segments. *)
let chain () =
  let q = 128 in
  let mat ~label ~stride =
    NTDSL.ndarray
      (values ~n:(q * q) ~stride)
      ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()
  in
  let qa = mat ~label:"msa_a" ~stride:389 and qb = mat ~label:"msa_b" ~stride:577 in
  let qc = mat ~label:"msa_c" ~stride:211 and qf = mat ~label:"msa_f" ~stride:701 in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qf * qe in
  site ~tag:"chain" ~single:false qg

type outcome = {
  o_site : site;
  o_seeded : Autotune.seeded_choice;
  o_explained : Autotune.explanation;
  o_pick : Autotune.model_choice;
  o_seeded_values : float array;
  o_model_values : float array;
}

let run_site s =
  let compile_run ~what compile =
    let ctx, routine =
      compile
        ~name:(Printf.sprintf "msa_%s_%s" s.tag what)
        (Context.auto ()) s.comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.sync ctx;
    Context.get_values ctx s.out.Tensor.value
  in
  let seeded = ref None and pick = ref None in
  let o_seeded_values =
    compile_run ~what:"seeded" (fun ~name ctx comp bindings ->
        Autotune.seeded_default ~name ~report:(fun r -> seeded := Some r) ctx comp bindings)
  in
  let o_explained =
    Autotune.model_explain
      ~name:("msa_" ^ s.tag ^ "_explain")
      (Context.auto ()) s.comp Ir.Indexing.Empty
  in
  let o_model_values =
    compile_run ~what:"model" (fun ~name ctx comp bindings ->
        Autotune.model_default ~name ~report:(fun r -> pick := Some r) ctx comp bindings)
  in
  {
    o_site = s;
    o_seeded = Option.value_exn ~here:[%here] !seeded;
    o_explained;
    o_pick = Option.value_exn ~here:[%here] !pick;
    o_seeded_values;
    o_model_values;
  }

let anchors (cands : (string * Autotune.explain) list) =
  List.filter cands ~f:(fun (_, ex) -> ex.Autotune.ex_anchor)

let rank_of (cands : (string * Autotune.explain) list) label =
  Option.bind (List.Assoc.find cands label ~equal:String.equal) ~f:(fun ex ->
      ex.Autotune.ex_rank_ms)

(* The leaf label of a sketch in the whole-routine walk, as [model_explain] and the pick print
   it. *)
let whole_label p = "W_sketch[" ^ Autotune.sketch_params_label p ^ "]"

(* Whether [entry] is one of the comma-separated entries of an [F_sketch[...]] pick label. A leaf
   label carries its segment's key tag, so it names one segment; nothing here parses a label. *)
let is_entry ~pick entry =
  List.exists
    [ "[" ^ entry ^ ","; "[" ^ entry ^ "]"; "," ^ entry ^ ","; "," ^ entry ^ "]" ]
    ~f:(fun bracketed -> String.is_substring pick ~substring:bracketed)

let strictly_below_anchor (o : outcome) =
  let pick = o.o_pick.Autotune.mc_label in
  let below cands rank =
    match (rank, anchors cands) with
    | Some r, [ (_, anchor) ] -> (
        match anchor.Autotune.ex_rank_ms with Some a -> Float.(r < a) | None -> false)
    | Some _, [] -> true
    | _ -> false
  in
  if String.equal pick "seeded" then true
  else if String.is_prefix pick ~prefix:"W_sketch[" then
    below o.o_explained.x_routine (rank_of o.o_explained.x_routine pick)
  else if String.is_prefix pick ~prefix:"F_sketch[" then
    (* Every entry against its own segment's anchor; at least one entry. *)
    let entries =
      List.concat_map o.o_explained.x_segments ~f:(fun xs ->
          List.filter_map (List.tl_exn xs.Autotune.xs_candidates) ~f:(fun (label, ex) ->
              if is_entry ~pick label then Some (xs.Autotune.xs_candidates, ex.Autotune.ex_rank_ms)
              else None))
    in
    (not (List.is_empty entries)) && List.for_all entries ~f:(fun (cands, rank) -> below cands rank)
  else false

let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let () =
  let sites = [ qkv (); outproj (); wgrad (); chain () ] in
  let outcomes = if seeds_here then List.map sites ~f:run_site else [] in
  let singles = List.filter outcomes ~f:(fun o -> o.o_site.single) in
  List.iter outcomes ~f:(fun o ->
      let routine = o.o_explained.Autotune.x_routine in
      Stdio.eprintf "%s: seeded [%s], anchors [%s], pick %s rank %s (not part of the golden)\n"
        o.o_site.tag
        (String.concat ~sep:"; "
           (List.map o.o_seeded.Autotune.sd_seeded ~f:(fun (at, p) ->
                Printf.sprintf "u%d %s" at (Autotune.sketch_params_label p))))
        (String.concat ~sep:"; "
           (List.map
              (anchors routine
              @ List.concat_map o.o_explained.x_segments ~f:(fun xs ->
                  anchors xs.Autotune.xs_candidates))
              ~f:(fun (l, ex) ->
                Printf.sprintf "%s %s" l
                  (Option.value_map ex.Autotune.ex_rank_ms ~default:"n/a" ~f:Float.to_string))))
        o.o_pick.Autotune.mc_label
        (Option.value_map o.o_pick.mc_rank_ms ~default:"n/a" ~f:Float.to_string));
  gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded plan seeds each single-site contraction" singles ~f:(fun o ->
      List.length o.o_seeded.Autotune.sd_seeded = 1);
  gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP each single-site walk is anchored exactly once, at the seeded plan's own sketch"
    singles ~f:(fun o ->
      match (anchors o.o_explained.Autotune.x_routine, o.o_seeded.Autotune.sd_seeded) with
      | [ (label, ex) ], [ (_, p) ] ->
          String.equal label (whole_label p) && Option.is_some ex.Autotune.ex_rank_ms
      | _ -> false);
  gated_exists ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the chain's per-segment substitution anchors a segment"
    (List.filter outcomes ~f:(fun o -> not o.o_site.single))
    ~f:(fun o ->
      List.exists o.o_explained.Autotune.x_segments ~f:(fun xs ->
          List.length (anchors xs.Autotune.xs_candidates) = 1));
  gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP model_default's pick is no fallback to the default pipeline" outcomes ~f:(fun o ->
      not (String.equal o.o_pick.Autotune.mc_label "default"));
  gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the pick is the seeded plan, or ranks strictly below its walk's anchor (no tie)"
    outcomes ~f:strictly_below_anchor;
  gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the picked routine computes bitwise what the seeded plan's does" outcomes
    ~f:(fun o ->
      Array.length o.o_model_values = Array.length o.o_seeded_values
      && Array.for_all2_exn o.o_model_values o.o_seeded_values ~f:bitwise);
  gated_all ~when_:seeds_here ~on:backend_name "on CUDA/HIP the outputs are not trivially zero"
    outcomes ~f:(fun o -> Array.exists o.o_seeded_values ~f:(fun v -> Float.(abs v > 1e-3)));
  (* Off the seeded backends: the incumbent is the default pipeline, and no walk is anchored. *)
  let unseeded =
    if seeds_here then []
    else
      let s = List.hd_exn sites in
      let x =
        Autotune.model_explain ~name:"msa_unseeded_explain" (Context.auto ()) s.comp
          Ir.Indexing.Empty
      in
      x.Autotune.x_routine
      @ List.concat_map x.Autotune.x_segments ~f:(fun xs -> xs.Autotune.xs_candidates)
  in
  gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the explanation lists candidates and anchors none"
    ((not (List.is_empty unseeded)) && List.is_empty (anchors unseeded))
