(* gh-ocannl-2080. An isolated f16 recipe site: two virtual narrowing twins over f32 masters. Report
   the real autotuner's crown, then compare the best tile and non-tile whole-site seeds in fresh,
   interleaved queued windows. No fixture IO. --dry-run compiles and checks every seed without
   ranking or reporting timings. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let args = Bench_args.create "twin_tile_bench"
let n = Bench_args.int args 0 ~name:"size" ~default:256
let dry = Bench_args.flag args ~name:"dry-run"
let p fmt = Bench_out.p fmt
let backend = Utils.get_global_arg ~arg_name:"backend" ~default:""
let root = Context.auto ()
let facts = Context.backend_facts_of_name backend
let caps = Context.codegen_capabilities root
let limits = Context.hardware_limits root

let () =
  if not (Ir.Backend_intf.binds_hardware_grid facts) then
    failwith "twin_tile_bench needs an explicitly selected GPU backend"

let product () =
  let input l f =
    let t = NTDSL.init ~l ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f () in
    let c = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
    Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
    Train.set_virtual c.Tensor.value;
    c
  in
  let a =
    input "ttb_a" (fun ix ->
        let i = ix.(0) and k = ix.(1) in
        1.
        +. (Float.of_int (i % 7) *. 0.125)
        +. (if k % 2 = 0 then 0.000244140625 else 0.)
        +. if k = 0 then 0.5 +. (Float.of_int (i % 7) *. 0.125) else 0.)
  in
  let b =
    input "ttb_b" (fun ix ->
        ((if ix.(0) % 2 = 0 then 1. else -1.) *. (1. +. (Float.of_int (ix.(1) % 11) *. 0.125)))
        +. if ix.(0) % 2 = 0 then 0.000244140625 else 0.)
  in
  let%op c = a * b in
  Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
  c

let output = product ()
let comp = Train.forward output

let compile name transform =
  Context.compile ~name ~lowered_transform:(fun o -> [ transform o ]) root comp Ir.Indexing.Empty

let check (ctx, routine) =
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx output.Tensor.value in
  Array.iteri got ~f:(fun index value ->
      let i = index / n and j = index % n in
      let want =
        (0.5 +. (Float.of_int (i % 7) *. 0.125)) *. (1. +. (Float.of_int (j % 11) *. 0.125))
      in
      if not (Float.is_finite value && Float.(abs (value -. want) <= 0.001)) then
        failwith (Printf.sprintf "wrong output at %d: %g vs %g" index value want))

let measure (ctx, routine) =
  match
    Autotune.admitted_timing_ms
      (Autotune.time_routine ~timing:Autotune.Queued ~repeats:3 ctx routine)
  with
  | Some ms -> ms
  | None -> failwith "timing window refused; do not report this cell"

let () =
  if n % 32 <> 0 then failwith "size must be a multiple of 32";
  p "backend=%s n=%d placement=virtual memory=backend-default dry_run=%b\n" backend n dry;
  let captured = ref None in
  let baseline =
    compile "ttb_capture" (fun o ->
        captured := Some o;
        o)
  in
  check baseline;
  Context.release (fst baseline);
  let opt = Option.value_exn !captured in
  let seeds =
    Autotune.sketch_seed_params ~capabilities:caps ~is_gpu:true ~is_cpu:false ~limits opt
  in
  p "seeds=%d tiles=%d\n" (List.length seeds) (List.count seeds ~f:(fun q -> q.Autotune.sk_mma));
  let variants =
    List.map seeds ~f:(fun q ->
        ( Autotune.sketch_params_label q,
          fun o -> Ir.Schedule.apply (Autotune.sketch_schedule ~accum_prec:caps.accum_prec ~p:q o) o
        ))
    @ List.map [ 64; 128; 256; 512 ] ~f:(fun block_size ->
        ( "default-" ^ Int.to_string block_size,
          fun o -> Ir.Schedule.apply (Ir.Schedule.default_gpu ~block_size o) o ))
  in
  let best_tile = ref None and best_scalar = ref None in
  List.iteri variants ~f:(fun index (label, transform) ->
      let compiled = compile ("ttb_seed_" ^ Int.to_string index) transform in
      check compiled;
      let mma = (snd compiled).Context.mma in
      if mma.scalar_fallbacks > 0 then failwith (label ^ ": scalar fallback");
      let tiled = Ir.C_syntax.equal_tensorization mma.tensorization Ir.C_syntax.Tensorized in
      if not dry then (
        let ms = measure compiled in
        p "seed %s ms=%.6f %s\n" label ms (Ir.C_syntax.mma_summary_string mma);
        let best = if tiled then best_tile else best_scalar in
        if Option.value_map !best ~default:true ~f:(fun (old, _, _) -> Float.(ms < old)) then
          best := Some (ms, label, transform));
      Context.release (fst compiled));
  (* The f16 benchmark recipe tunes with zero beam rounds as well. This independent search includes
     fission/presets; its crown is not inferred from the seed sweep. *)
  let ctx, routine =
    Train.tune_placements ~name:"ttb_tuned" ~cache_dir:"" ~placement_store:false ~inline_flips:0
      ~rounds:0
      ~report:(fun r ->
        p "crown=%s asked_tile=%b census=%s mma_timed=%d mma_best_ms=%.6f best_ms=%.6f\n"
          r.Autotune.best_label r.best_tensorized
          (Option.value_map r.best_tensorization ~default:"none" ~f:Ir.C_syntax.tensorization_name)
          r.mma_timed r.mma_best_ms r.best_ms)
      root output comp Ir.Indexing.Empty
  in
  p "shipped_census=%s\n" (Ir.C_syntax.mma_summary_string routine.Context.mma);
  check (ctx, routine);
  Context.release ctx;
  if not dry then (
    let _, tile_label, tile = Option.value_exn !best_tile in
    let _, scalar_label, scalar = Option.value_exn !best_scalar in
    let tile = compile "ttb_final_tile" tile and scalar = compile "ttb_final_scalar" scalar in
    check tile;
    check scalar;
    p "finalists tile=%s non_tile=%s\n" tile_label scalar_label;
    for rep = 0 to 6 do
      let tile_ms, scalar_ms =
        if rep % 2 = 0 then
          let t = measure tile in
          (t, measure scalar)
        else
          let s = measure scalar in
          (measure tile, s)
      in
      p "paired rep=%d tile_ms=%.6f non_tile_ms=%.6f ratio=%.6f\n" rep tile_ms scalar_ms
        (tile_ms /. scalar_ms)
    done;
    Context.release (fst tile);
    Context.release (fst scalar))
