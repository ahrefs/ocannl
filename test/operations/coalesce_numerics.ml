(* gh-ocannl-1278: do the coalesced tuner branch's layouts change what a matmul site computes?

   [Autotune.model_admits] keeps the coalesced branch out of [model_default]'s numerics-preserving
   candidate set beside the tensorized and conv families, on the ground that it reorders the
   reduction. Neither layout touches a cell's summation order on its face: [Coalesce_columns]
   (gh-ocannl-1165) merges two OUTPUT loops, the q/k/v site's head and column, and leaves the
   contraction alone; [Coalesce_contraction] (gh-ocannl-1279) merges the out projection's two
   contraction loops [(h, e)] into one loop over [h * e] in the same lexicographic order, so a cell
   still accumulates its products in the order the uncoalesced nest visits them.

   This test executes that claim on inputs where the order matters. Each value is a multiple of 1/37
   rounded to f32, so products and partial sums round: a host replay of one cell's f32 accumulation
   in the reverse order changes its bits, which is the control that bitwise agreement below is not
   an accident of exact arithmetic (schedule_coalesce's inputs are exact by design, so its bitwise
   claims could not tell). On a GPU backend, per layout: every scalar blocktile seed of the ordinary
   family ([Autotune.scalar_blocktile], the set the seeded default and [model_default] draw from) is
   executed, and they agree among themselves bitwise; then every coalesced seed of the layout is
   executed and must compute that same output bitwise. Seeds are enumerated under the device's own
   limits, so every one of them can run. Off GPU the coalesced seeds cannot execute and only the
   structural claims hold. *)

(* cc-fp-contract: not-cc -- the executed bitwise claims are gated on a hardware grid, so cc skips
   them *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

(* Where the GPU compiler holds a kernel's arithmetic still across kernels, so equal summation
   orders mean equal bits. Not on HIP: hiprtc compiles under [-ffast-math], which licenses
   contraction, and no configuration turns it off there (CUDA's is [cuda_fast_math]); whether an
   [acc + a * b] fuses follows the surrounding kernel, and partitioning the staged copies
   (gh-ocannl-1661) moved the 2x2 register-tile seeds' rounding at 2% of qkv's cells and 1% of
   outproj's, identically across their geometries. There the bounded claims below stand in. *)
let bitwise_gpu = on_gpu && not (String.equal backend_name "hip")

(* Within 2^-16 of the output's largest magnitude: wide of contraction drift (a rounding per
   product), narrow of a lost, stale or misplaced product, which moves a cell by a product. *)
let within ~what_ran name ~want got =
  let top = Array.fold want ~init:0. ~f:(fun m v -> Float.max m (Float.abs v)) in
  Verdict.within ~what_ran name got want ~tol:(fun _ -> top /. 65536.)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

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

let run_with fwd tensor transform =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Context.get_values (Context.run ctx routine) tensor.Tensor.value

let r32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* How a differing output differs (gh-ocannl-1661), for the record: how many cells, the largest
   absolute difference and the largest distance in f32 units in the last place. A rounding
   difference (a contraction) moves cells by an ulp or two; a lost or stale copy moves them by the
   magnitude of a product. *)
let difference got want =
  if Array.length got <> Array.length want then "of another length"
  else
    let ulps a b = Int32.(abs (bits_of_float a - bits_of_float b)) |> Int32.to_int_exn in
    let n, abs_max, ulp_max =
      Array.fold2_exn got want ~init:(0, 0., 0) ~f:(fun (n, m, u) g w ->
          if Float.equal g w then (n, m, u)
          else (n + 1, Float.max m (Float.abs (g -. w)), Int.max u (ulps g w)))
    in
    Printf.sprintf "in %d of %d cells, by at most %g (%d ulp)" n (Array.length want) abs_max ulp_max

(* A value per cell that varies with every index (1009 is prime and above every row-major stride's
   residue period here) and is inexact in binary: a multiple of 1/37, rounded to f32 as stored. *)
let value ~dims idcs =
  let flat = Array.fold2_exn dims idcs ~init:0 ~f:(fun acc d i -> (acc * d) + i) in
  r32 (Float.of_int ((((flat * 7919) + 13) % 1009) - 504) /. 37.)

(* [rev] lists the axes whose index is read reversed: the same values, met in the reverse order
   along those axes, for the order control below. *)
let init ~l ?(rev = []) ?(b = []) ?(i = []) ~o () =
  let dims = Array.of_list (b @ o @ i) in
  let f idcs =
    value ~dims
      (Array.mapi idcs ~f:(fun ax ix ->
           if List.mem rev ax ~equal:Int.equal then dims.(ax) - 1 - ix else ix))
  in
  NTDSL.init ~l ~prec:Ir.Ops.single ~b ~o ~i ~f ()

(* The device's limits where seeds execute; elsewhere the enumeration is structural only. *)
let limits =
  if on_gpu then Context.hardware_limits (Context.auto ()) else Ir.Backend_intf.no_hardware_limits

let layout_tag = function
  | Autotune.Coalesce_columns -> "coal-n"
  | Autotune.Coalesce_contraction -> "coal-k"

(* [cand] is the site; [reversed] the same contraction over inputs whose contraction axes are read
   reversed, so each cell sums the same products in the reverse order. *)
let leg ~tag ~layout cand reversed =
  let fwd = named tag (Train.forward cand) in
  let opt = capture fwd in
  let lt = layout_tag layout in
  p
    (Printf.sprintf "%s: the %s prefix applies" tag lt)
    (Option.is_some (Autotune.coalesce_prefix ~layout opt));
  let family =
    List.filter ~f:Autotune.scalar_blocktile
      (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt)
  in
  let coalesced =
    List.filter (Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits opt) ~f:(fun q ->
        Poly.equal q.Autotune.sk_coalesce (Some layout))
  in
  p_exists (Printf.sprintf "%s: the family offers a scalar blocktile seed" tag) family ~f:(fun _ ->
      true);
  p_exists (Printf.sprintf "%s: the %s layout offers seeds" tag lt) coalesced ~f:(fun _ -> true);
  (* The columns layout seeds scalar blocktile leaves only. The contraction layout also seeds staged
     tensorized leaves and fused twins (gh-ocannl-1459), which reround or relocate the tail by
     design and are pinned in schedule_coalesce; the question here is the scalar leaves'. *)
  let scalar =
    List.filter coalesced ~f:(fun q ->
        Autotune.scalar_blocktile { q with Autotune.sk_coalesce = None })
  in
  (match layout with
  | Autotune.Coalesce_columns ->
      p_all (Printf.sprintf "%s: every %s seed is a scalar blocktile but for its coalescing" tag lt)
        coalesced ~f:(fun q -> Autotune.scalar_blocktile { q with Autotune.sk_coalesce = None })
  | Autotune.Coalesce_contraction ->
      p_exists
        (Printf.sprintf "%s: the %s layout offers a scalar blocktile seed but for its coalescing"
           tag lt) scalar ~f:(fun _ -> true));
  let run q =
    run_with fwd cand (fun o -> Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o)
  in
  let label q = Autotune.sketch_params_label q in
  Stdio.eprintf
    "coalesce_numerics: %s on %s: %d family seed(s), %d %s seed(s) (not part of the golden)\n%!" tag
    backend_name (List.length family) (List.length coalesced) lt;
  (* The executed populations exist only on a GPU; off it they are empty and the gated claims skip
     without reading them. *)
  let outs = if on_gpu then List.map family ~f:(fun q -> (q, run q)) else [] in
  let coal_outs =
    if on_gpu then
      List.map scalar ~f:(fun q ->
          match run q with
          | got -> (q, Some got)
          | exception exn ->
              Stdio.eprintf "coalesce_numerics: %s %s FAILED: %s\n%!" tag (label q)
                (Exn.to_string exn);
              (q, None))
    else []
  in
  let want = match outs with (_, w) :: _ -> w | [] -> [||] in
  gated ~when_:on_gpu ~on:backend_name
    (Printf.sprintf "%s: the family's output is not all zeros" tag)
    (Array.exists want ~f:(fun v -> Float.(v <> 0.)));
  (let claim =
     Printf.sprintf "%s: the family's scalar blocktile seeds agree within 2^-16 of its scale" tag
   in
   gated_all ~when_:on_gpu ~on:backend_name claim outs ~f:(fun (q, got) ->
       within ~what_ran:(label q) claim ~want got));
  gated_all ~when_:bitwise_gpu ~on:backend_name
    (Printf.sprintf "%s: the family's scalar blocktile seeds agree bitwise among themselves" tag)
    outs ~f:(fun (q, got) ->
      let ok = Array.equal Float.equal got want in
      if not ok then
        Stdio.eprintf "coalesce_numerics: %s family %s differs %s\n%!" tag (label q)
          (difference got want);
      ok);
  (* The executed control: the first family seed run on the reversed inputs sums every cell's
     products in the reverse order, and some cell's bits change -- so the bitwise claims here would
     see a layout that reordered a cell's summation. *)
  let rev_out =
    match family with
    | q :: _ when on_gpu ->
        let fwd_rev = named (tag ^ "_rev") (Train.forward reversed) in
        run_with fwd_rev reversed (fun o ->
            Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q o) o)
    | _ -> [||]
  in
  gated ~when_:on_gpu ~on:backend_name
    (Printf.sprintf "%s: control: summing every cell's products in reverse changes some cell" tag)
    (Array.length rev_out = Array.length want && not (Array.equal Float.equal rev_out want));
  (* Informational: the serial reference, unscheduled. Its accumulation form differs from a
     register-tiled one (a backend may contract one and not the other), so it is not claimed. *)
  if on_gpu then
    Stdio.eprintf
      "coalesce_numerics: %s: the serial reference %s the family (not part of the golden)\n%!" tag
      (if Array.equal Float.equal (run_with fwd cand Fn.id) want then "matches" else "differs from");
  (let claim =
     Printf.sprintf "%s: every scalar blocktile %s seed computes the family's values within 2^-16"
       tag lt
   in
   gated_all ~when_:on_gpu ~on:backend_name claim coal_outs ~f:(fun (q, got) ->
       Option.exists got ~f:(within ~what_ran:(label q) claim ~want)));
  gated_all ~when_:bitwise_gpu ~on:backend_name
    (Printf.sprintf "%s: every scalar blocktile %s seed computes the family's values bitwise" tag lt)
    coal_outs ~f:(fun (q, got) ->
      match got with
      | Some got ->
          let ok = Array.equal Float.equal got want in
          if not ok then
            Stdio.eprintf "coalesce_numerics: %s %s differs %s\n%!" tag (label q)
              (difference got want);
          ok
      | None -> false)

(* {1 The control: the inputs make summation order visible} *)

let bb = 2
and ss = 64
and hh = 4
and ee = 32
and kk = 64

let () =
  let xv bi si ki = value ~dims:[| bb; ss; kk |] [| bi; si; ki |]
  and wv hi ei ki = value ~dims:[| hh; ee; kk |] [| hi; ei; ki |] in
  let cells = List.init ss ~f:(fun si -> (0, si, si % hh, si % ee)) in
  p_exists "control: some q/k/v cell's f32 sum changes bits when its products are summed in reverse"
    cells ~f:(fun (bi, si, hi, ei) ->
      let prods = List.init kk ~f:(fun ki -> r32 (xv bi si ki *. wv hi ei ki)) in
      let sum l = List.fold l ~init:0. ~f:(fun acc v -> r32 (acc +. v)) in
      not (Float.equal (sum prods) (sum (List.rev prods))))

(* {1 The columns layout: the q/k/v projection [d[b,s,h,e] += x[b,s,k] * w[h,e,k]]} *)

let () =
  let site ?rev tag =
    let xv = init ~l:(tag ^ "_x") ?rev ~o:[ bb; ss; kk ] ()
    and wv = init ~l:(tag ^ "_w") ?rev ~o:[ hh; ee; kk ] () in
    let%op out = xv +* "bsk;hjk=>bshj" wv in
    out
  in
  leg ~tag:"qkv" ~layout:Autotune.Coalesce_columns (site "cn") (site ~rev:[ 2 ] "cn_rev")

(* {1 The contraction layout: the out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]]} *)

let () =
  let jj = 64 and ss = 32 in
  let site ?(rev = false) tag =
    let w = init ~l:(tag ^ "_w") ~rev:(if rev then [ 1; 2 ] else []) ~o:[ jj ] ~i:[ hh; ee ] () in
    let att =
      init ~l:(tag ^ "_att") ~rev:(if rev then [ 2; 3 ] else []) ~b:[ bb; ss ] ~o:[ hh; ee ] ()
    in
    let%op out = w * att in
    out
  in
  leg ~tag:"outproj" ~layout:Autotune.Coalesce_contraction (site "ck") (site ~rev:true "ck_rev")
