(* gh-ocannl-1183: backward contractions are matmul sites through an enabling interchange.

   Backprop lowers a weight gradient as [for b, s, o, i: dw[o,i] += dy[b,s,o] * x[b,s,i]] -- the
   contraction loops OUTERMOST -- and a data gradient as [for b, s, o, i: dx[b,s,i] += dy[b,s,o] *
   w[o,i]] -- the contraction loop between write loops. [classify_matmul] reads the contraction nest
   off the innermost end, so neither was ever a matmul site: on the gpt2_mini training step not one
   of the backward contractions was detected, and no tiled, register-tiled or tensorized sketch was
   ever seeded for the weight-gradient kernels that dominate it at large batch.

   Mechanism under test: [Autotune.detect_matmul_canonical] sinks the contraction loops below the
   write loops by an adjacent-[Swap] chain, each [Op_legal], re-detects on the interchanged code,
   and every schedule built from the site carries the chain as its prefix -- so the schedule applies
   to the ORIGINAL code, and a schedule persisted against one compile replays against another.

   Hand-built nests (each built twice, with fresh symbols, for the replay leg), in the exact loop
   orders the training step's lowering produces: - wgrad: the weight gradient, two outer contraction
   loops (b, s); - dgrad: the data gradient, the contraction loop third of four; - fwd: the forward
   product, contraction innermost -- the control the interchange leaves alone.

   Executed assertions compare against a serial reference computed on the host from the same
   discriminating inputs: every cell is a small dyadic, so every partial sum is exact in f32 and
   bitwise equality is required whatever the accumulation order; the inputs are strictly positive,
   so every reference cell is nonzero and a candidate that drops a write cannot pass. GPU backends
   execute the GPU sketch families, cc the CPU ones; the interchange alone executes everywhere. *)

open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Idx = Ir.Indexing
module Tn = Ir.Tnode
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name
let skipped = Verdict.skipped ~backend:backend_name

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

let nb = 2
and ns = 32
and no = 64
and ni = 64

let node = Ll_test.node_factory ~first_id:118300 ~dims:[| 1 |] ()

(* Strictly positive dyadic cells (offset 1): every product and partial sum is exact in f32. *)
let values ~dims ~modulus ~stride =
  let n = Array.fold dims ~init:1 ~f:( * ) in
  Array.init n ~f:(Ll_test.cycle_flat ~dims ~modulus ~offset:1. ~stride)

let dy_dims = [| nb; ns; no |]
let x_dims = [| nb; ns; ni |]
let w_dims = [| no; ni |]
let dy_v = values ~dims:dy_dims ~modulus:7 ~stride:0.25
let x_v = values ~dims:x_dims ~modulus:5 ~stride:0.125
let w_v = values ~dims:w_dims ~modulus:5 ~stride:0.125
let at dims v idcs = v.(Ll_test.flat ~dims idcs)

type case = {
  tag : string;
  build : unit -> LL.t;  (** Fresh symbols per call. *)
  out : Tn.t;
  seed : (Tn.t * float array) list;
  want : float array;
}

let fma a b acc =
  LL.Ternop (Ir.Ops.FMA, (a, Ll_test.single), (b, Ll_test.single), (acc, Ll_test.single))

let it = Ll_test.iter

(* dw[o,i] = sum_{b,s} dy[b,s,o] * x[b,s,i], loops (b, s, o, i). *)
let wgrad =
  let dy = node ~dims:dy_dims "mi_w_dy" and x = node ~dims:x_dims "mi_w_x" in
  let dw = node ~dims:w_dims "mi_w_dw" in
  List.iter [ dy; x; dw ] ~f:Ll_test.materialize;
  let build () =
    let b = Ll_test.sym () and s = Ll_test.sym () and o = Ll_test.sym () and i = Ll_test.sym () in
    Ll_test.seq (Ll_test.zero dw)
      (Ll_test.loop_n b nb
         (Ll_test.loop_n s ns
            (Ll_test.loop_n o no
               (Ll_test.loop_n i ni
                  (Ll_test.set dw
                     [| it o; it i |]
                     (fma
                        (Ll_test.get dy [| it b; it s; it o |])
                        (Ll_test.get x [| it b; it s; it i |])
                        (Ll_test.get dw [| it o; it i |])))))))
  in
  let want =
    Array.init (no * ni) ~f:(fun c ->
        let cell = Ll_test.unflat ~dims:w_dims c in
        let o = cell.(0) and i = cell.(1) in
        let acc = ref 0. in
        for b = 0 to nb - 1 do
          for s = 0 to ns - 1 do
            acc := !acc +. (at dy_dims dy_v [| b; s; o |] *. at x_dims x_v [| b; s; i |])
          done
        done;
        !acc)
  in
  { tag = "wgrad"; build; out = dw; seed = [ (dy, dy_v); (x, x_v) ]; want }

(* dx[b,s,i] = sum_o dy[b,s,o] * w[o,i], loops (b, s, o, i). *)
let dgrad =
  let dy = node ~dims:dy_dims "mi_d_dy" and w = node ~dims:w_dims "mi_d_w" in
  let dx = node ~dims:x_dims "mi_d_dx" in
  List.iter [ dy; w; dx ] ~f:Ll_test.materialize;
  let build () =
    let b = Ll_test.sym () and s = Ll_test.sym () and o = Ll_test.sym () and i = Ll_test.sym () in
    Ll_test.seq (Ll_test.zero dx)
      (Ll_test.loop_n b nb
         (Ll_test.loop_n s ns
            (Ll_test.loop_n o no
               (Ll_test.loop_n i ni
                  (Ll_test.set dx
                     [| it b; it s; it i |]
                     (fma
                        (Ll_test.get dy [| it b; it s; it o |])
                        (Ll_test.get w [| it o; it i |])
                        (Ll_test.get dx [| it b; it s; it i |])))))))
  in
  let want =
    Array.init
      (nb * ns * ni)
      ~f:(fun c ->
        let cell = Ll_test.unflat ~dims:x_dims c in
        let b = cell.(0) and s = cell.(1) and i = cell.(2) in
        let acc = ref 0. in
        for o = 0 to no - 1 do
          acc := !acc +. (at dy_dims dy_v [| b; s; o |] *. at w_dims w_v [| o; i |])
        done;
        !acc)
  in
  { tag = "dgrad"; build; out = dx; seed = [ (dy, dy_v); (w, w_v) ]; want }

(* y[b,s,o] = sum_i x[b,s,i] * w[o,i], loops (b, s, o, i): already a plain site. *)
let fwd_build () =
  let x = node ~dims:x_dims "mi_f_x" and w = node ~dims:w_dims "mi_f_w" in
  let y = node ~dims:dy_dims "mi_f_y" in
  List.iter [ x; w; y ] ~f:Ll_test.materialize;
  let b = Ll_test.sym () and s = Ll_test.sym () and o = Ll_test.sym () and i = Ll_test.sym () in
  Ll_test.seq (Ll_test.zero y)
    (Ll_test.loop_n b nb
       (Ll_test.loop_n s ns
          (Ll_test.loop_n o no
             (Ll_test.loop_n i ni
                (Ll_test.set y
                   [| it b; it s; it o |]
                   (fma
                      (Ll_test.get x [| it b; it s; it i |])
                      (Ll_test.get w [| it o; it i |])
                      (Ll_test.get y [| it b; it s; it o |])))))))

let hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let digest o = SC.digest (SC.canonicalize ~with_placements:false o)
let is_swap = function Sched.Swap _ -> true | _ -> false

let unfused_seeds ~is_gpu ~limits opt =
  Autotune.sketch_seed_params ~is_gpu ~is_cpu:(not is_gpu) ~limits opt
  |> List.filter ~f:(fun q -> not q.Autotune.sk_epilogue)

let execute ~name (o : LL.optimized) (c : case) =
  match Ll_test.execute ~name o ~seed:c.seed ~read:[ c.out ] with
  | [ got ] -> Some got
  | _ -> None
  | exception exn ->
      Stdio.eprintf "%s: execution FAILED: %s\n" name (Exn.to_string exn);
      None

let leg (c : case) ~expect_prefix ~ko_extents ~nk =
  let tag = c.tag in
  let o = Ll_test.optimize ~name:("mi_" ^ tag) (c.build ()) in
  p
    (tag ^ ": the plain matcher does not see the backward contraction")
    (Option.is_none (Autotune.detect_matmul o.LL.llc));
  match Autotune.detect_matmul_canonical o with
  | None -> p (tag ^ ": the interchanged contraction is a matmul site") false
  | Some (site, prefix, swapped) ->
      p (tag ^ ": the interchanged contraction is a matmul site") true;
      p_all (tag ^ ": the enabling prefix is a chain of Swaps") prefix ~f:is_swap;
      p (tag ^ ": the enabling prefix has the expected length") (List.length prefix = expect_prefix);
      p (tag ^ ": the site accumulates into the gradient node") (phys_equal site.Autotune.m_d c.out);
      p
        (tag ^ ": the outer contraction loops carry the expected extents")
        (List.equal Int.equal (List.map site.Autotune.m_ko ~f:snd) ko_extents);
      p (tag ^ ": m_k is the innermost contraction loop") (site.Autotune.m_nk = nk);
      p
        (tag ^ ": the prefix replayed on the original code is the code the site was detected on")
        (String.equal (digest (Sched.apply prefix (hermetic o))) (digest swapped));
      (* The interchange alone reorders whole cells' sequences, never one cell's: executed bitwise
         against the host reference on every backend. *)
      let interchanged = Sched.apply prefix (hermetic o) in
      (match execute ~name:("mi_" ^ tag ^ "_swapped") interchanged c with
      | Some got ->
          p
            (tag ^ ": the interchanged code matches the reference bitwise")
            (Array.equal Float.equal got c.want)
      | None -> p (tag ^ ": the interchanged code matches the reference bitwise") false);
      let limits =
        if on_gpu then Context.hardware_limits (Context.auto ())
        else Ir.Backend_intf.no_hardware_limits
      in
      let gpu_seeds = unfused_seeds ~is_gpu:true ~limits:Ir.Backend_intf.no_hardware_limits o in
      let cpu_seeds = unfused_seeds ~is_gpu:false ~limits o in
      p (tag ^ ": GPU sketch seeds are proposed") (not (List.is_empty gpu_seeds));
      p (tag ^ ": CPU sketch seeds are proposed") (not (List.is_empty cpu_seeds));
      let sched q = Autotune.sketch_schedule ~accum_prec ~p:q o in
      p_all (tag ^ ": every seed's schedule starts with the enabling prefix")
        (gpu_seeds @ cpu_seeds) ~f:(fun q ->
          let s = sched q in
          List.length s > List.length prefix
          && Sexp.equal
               (Sched.sexp_of_schedule (List.take s (List.length prefix)))
               (Sched.sexp_of_schedule prefix));
      (* Replay from the original code: persisted against this compile, decoded against a fresh
         lowering of the same nest (fresh symbols), the schedule rebuilds the same program. *)
      let fresh = Ll_test.optimize ~name:("mi_" ^ tag ^ "_fresh") (c.build ()) in
      p
        (tag ^ ": a fresh lowering has the same structural identity")
        (String.equal (digest o) (digest fresh));
      p_all (tag ^ ": every seed's schedule replays from a fresh lowering of the original code")
        (gpu_seeds @ cpu_seeds) ~f:(fun q ->
          match
            let s = sched q in
            let saved, _ = SC.to_saved (SC.base_registry (SC.canonicalize o)) s in
            let replayed, _ =
              SC.of_saved (SC.canonicalize fresh)
                (SC.saved_schedule_of_sexp (SC.sexp_of_saved_schedule saved))
            in
            (digest (Sched.apply s (hermetic o)), digest (Sched.apply replayed (hermetic fresh)))
          with
          | a, b -> String.equal a b
          | exception exn ->
              Stdio.eprintf "%s: replay FAILED: %s\n" tag (Exn.to_string exn);
              false);
      let run_family ~what seeds =
        Stdio.eprintf "%s: executing %d %s seeds on %s\n%!" tag (List.length seeds) what
          backend_name;
        let results =
          List.mapi seeds ~f:(fun n q ->
              match Sched.apply (sched q) (hermetic o) with
              | scheduled -> execute ~name:(Printf.sprintf "mi_%s_%s%d" tag what n) scheduled c
              | exception exn ->
                  Stdio.eprintf "%s/%s: construct FAILED: %s\n" tag what (Exn.to_string exn);
                  None)
        in
        p_all
          (Printf.sprintf "%s: every %s candidate compiles, runs and matches the reference bitwise"
             tag what) results ~f:(function
          | Some got -> Array.equal Float.equal got c.want
          | None -> false)
      in
      if on_gpu then (
        run_family ~what:"gpu" gpu_seeds;
        skipped (tag ^ ": every cpu candidate compiles, runs and matches the reference bitwise"))
      else (
        skipped (tag ^ ": every gpu candidate compiles, runs and matches the reference bitwise");
        run_family ~what:"cpu" cpu_seeds)

let () =
  p_all "reference: every weight-gradient cell is nonzero" (Array.to_list wgrad.want) ~f:(fun v ->
      Float.(v <> 0.));
  p_all "reference: every data-gradient cell is nonzero" (Array.to_list dgrad.want) ~f:(fun v ->
      Float.(v <> 0.));
  (* Sink (b, s) below (o, i): each of the two passes moves one contraction loop across both write
     loops. *)
  leg wgrad ~expect_prefix:4 ~ko_extents:[ nb ] ~nk:ns;
  (* Sink o below i: one swap. *)
  leg dgrad ~expect_prefix:1 ~ko_extents:[] ~nk:no;
  let f = Ll_test.optimize ~name:"mi_fwd" (fwd_build ()) in
  p "fwd: a forward product is a plain site, with no prefix"
    (match Autotune.detect_matmul_canonical f with
    | Some (_, [], same) -> phys_equal same f
    | _ -> false)
