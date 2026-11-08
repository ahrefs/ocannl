(* gh-ocannl-2080: both matmul operands round virtual f32 masters to f16. Seeding must judge the
   reads' formats, and the actual GPU compile must emit a tile, not its fallback. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let n = 64
let backend = Utils.get_global_arg ~arg_name:"backend" ~default:""
let ctx = Context.auto ()
let facts = Context.backend_facts_of_name backend
let capabilities = Context.codegen_capabilities ctx
let limits = Context.hardware_limits ctx

(* Each adjacent k pair cancels after rounding. The k=0 term leaves a nonzero dyadic answer varying
   across both output axes. Omitting A's rounding leaves 32 * 2^-12 times B's column scale, well
   beyond the value tolerance. *)
let a_values ix =
  let i = ix.(0) and k = ix.(1) in
  1.
  +. (Float.of_int (i % 7) *. 0.125)
  +. (if k % 2 = 0 then 0.000244140625 else 0.)
  +. if k = 0 then 0.5 +. (Float.of_int (i % 7) *. 0.125) else 0.

let b_values ix =
  ((if ix.(0) % 2 = 0 then 1. else -1.) *. (1. +. (Float.of_int (ix.(1) % 11) *. 0.125)))
  +. if ix.(0) % 2 = 0 then 0.000244140625 else 0.

let product ~materialized =
  let input label f =
    let t = NTDSL.init ~l:label ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f () in
    let c = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
    Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
    (if materialized then Train.set_materialized else Train.set_virtual) c.Tensor.value;
    c
  in
  let a = input "gtt_a" a_values and b = input "gtt_b" b_values in
  let%op c = a * b in
  Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
  c

let compile ?(transform = Fn.id) name output =
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (Train.forward output) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx output.Tensor.value, routine)

let () =
  Stdio.eprintf "gpu_twin_tiles: backend %s\n" backend;
  let opt = ref None in
  let _, _ =
    compile "gtt_capture" (product ~materialized:false) ~transform:(fun o ->
        opt := Some o;
        o)
  in
  let opt = Option.value_exn !opt in
  let half x = Ir.Ops.half_to_single (Ir.Ops.single_to_half x) in
  p_all "omitting either operand's rounding changes the answer beyond tolerance"
    [ (Fn.id, half); (half, Fn.id) ]
    ~f:(fun (round_a, round_b) ->
      let raw = ref 0. in
      for k = 0 to n - 1 do
        raw := !raw +. (round_a (a_values [| 0; k |]) *. round_b (b_values [| k; 0 |]))
      done;
      Float.(abs (!raw -. 0.5) > 0.001));
  let synthetic =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.minimal_mma_capability with
            mma_tile = (8, 8, 8);
            mma_format_tiles = [ ((Mma_f16, Mma_f16, Mma_f16), (8, 8, 8)) ];
          };
    }
  in
  let seeds ?capabilities limits =
    Autotune.sketch_seed_params ?capabilities ~is_gpu:true ~is_cpu:false ~limits opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_mma)
  in
  p_exists "two virtual f16 twins seed a GPU tile using their read formats" (seeds synthetic)
    ~f:(fun _ -> true);
  p_all "rounded GPU operands seed only staged tiles" (seeds synthetic) ~f:(fun q ->
      q.Autotune.sk_bk > 0);
  let backing_formats =
    {
      synthetic with
      mma =
        Some
          {
            (Option.value_exn synthetic.mma) with
            mma_format_tiles = [ ((Mma_f32, Mma_f32, Mma_f16), (8, 8, 8)) ];
          };
    }
  in
  p_empty "the backing masters' formats cannot admit a rounded tile" ~over:(seeds synthetic)
    (seeds backing_formats);
  let gpu_leg =
    Verdict.leg
      ~when_:(Ir.Backend_intf.binds_hardware_grid facts && Option.is_some limits.mma)
      ~on:backend ()
  in
  let actual = lazy (seeds ~capabilities limits) in
  in_leg ~leg:gpu_leg "the advertised tile capability seeds tiles at the twin site" (fun label ->
      p_exists label (Lazy.force actual) ~f:(fun _ -> true));
  let executions =
    lazy
      (let reference, _ = compile "gtt_materialized" (product ~materialized:true) in
       List.mapi (Lazy.force actual) ~f:(fun index q ->
           let got, routine =
             compile
               ("gtt_staged_" ^ Int.to_string index)
               (product ~materialized:false)
               ~transform:(fun o ->
                 Ir.Schedule.apply
                   (Autotune.sketch_schedule ~accum_prec:capabilities.accum_prec ~p:q o)
                   o)
           in
           Stdio.eprintf "%s: %s %s\n" backend (Autotune.sketch_params_label q)
             (Ir.C_syntax.mma_summary_string routine.Context.mma);
           (got, reference, routine)))
  in
  in_leg ~leg:gpu_leg "the seeded twin-site pipelines compile to tensor code" (fun label ->
      p_all label (Lazy.force executions) ~f:(fun (_, _, r) ->
          Ir.C_syntax.equal_tensorization r.Context.mma.tensorization Ir.C_syntax.Tensorized
          && r.Context.mma.scalar_fallbacks = 0));
  in_leg ~leg:gpu_leg "tiled twin reads agree with materialized twins" (fun label ->
      p_all label (Lazy.force executions) ~f:(fun (got, reference, r) ->
          Array.length got = n * n
          && Verdict.within ~what_ran:r.Context.name label got reference ~tol:(fun _ -> 0.001)))
