(* gh-ocannl-2105: a conv whose input and kernel are virtual f16 twins of f32 masters, executed on
   the active GPU through every conv tile the backend's own seeder admits at the twin site.

   gh-ocannl-2078 made conv seeding judge an operand by the precision its read reaches the site at
   (the twin's f16), not the master's backing storage (f32): the GPU format and tile selection, and
   the staged traffic it prices. [conv_twin_packing] checks that selection against a synthetic
   matrix unit and executes only CPU contexts. Here the backend's real capability seeds the tiles,
   each seed compiles to tensor code (the routine's mma census: tensorized, no scalar fallback), and
   its executed values equal a materialized-twin reference bitwise.

   The data discriminate: every value varies with every index of its operand, and every master
   carries a 2^-15 component f16 cannot hold, so omitting either operand's rounding moves every
   output cell, and a host oracle computes the rounded conv. *)

(* cc-fp-contract: exact-operands -- Ll_test.exact_partials derives the f32 reduction premise from
   the initialized operands after their required rounding. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend = Utils.get_global_arg ~arg_name:"backend" ~default:""
let ctx = Context.auto ()
let facts = Context.backend_facts_of_name backend
let capabilities = Context.codegen_capabilities ctx
let limits = Context.hardware_limits ctx

(* A 2-D conv: batch 2, an 8 x 8 output over 16 in-channels, a 3 x 3 window, 32 out-channels. No
   extent is pad-masked by an 8- or 16-wide tile; masked accumulators are
   [masked_persistent_fragment]'s subject. *)
let batch = 2
let rows = 8
let ic = 16
let oc = 32
let window = 3

(* Index order: batch, then output axes (rows, columns, channels) for the input; output axis (out
   channel), then input axes (kh, kw, in channel) for the kernel. *)
let x_values =
  Ll_test.operand ~b:[ batch ]
    ~o:[ rows + window - 1; rows + window - 1; ic ]
    ~i:[]
    (Ll_test.weighted ~weights:[| 1; 2; 3; 5 |] ~modulus:7 ~offset:1.000244140625 ~stride:0.125)

let k_values =
  Ll_test.operand ~b:[] ~o:[ oc ] ~i:[ window; window; ic ]
    (Ll_test.weighted ~weights:[| 1; 3; 2; 4 |] ~modulus:5 ~offset:1.000244140625 ~stride:0.125)

let half x = Ir.Ops.half_to_single (Ir.Ops.single_to_half x)

(* The conv's two f16 operands: [`Virtual] and [`Materialized] twins of f32 masters, or [`Stored]:
   f16 nodes initialized with the rounded values, whose site reads its operands' own storage. *)
let product operands =
  let twin label ~b ~i ~o f =
    match operands with
    | `Stored -> NTDSL.init ~l:label ~prec:Ir.Ops.half ~b ~i ~o ~f:(fun ix -> half (f ix)) ()
    | (`Virtual | `Materialized) as mode ->
        let t = NTDSL.init ~l:label ~prec:Ir.Ops.single ~b ~i ~o ~f () in
        let c = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
        Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
        (match mode with
        | `Virtual -> Train.set_virtual c.Tensor.value
        | `Materialized -> Train.set_materialized c.Tensor.value);
        c
  in
  let x =
    twin "gct_x" ~b:[ batch ] ~i:[] ~o:[ rows + window - 1; rows + window - 1; ic ] x_values
  in
  let k = twin "gct_k" ~b:[] ~i:[ window; window; ic ] ~o:[ oc ] k_values in
  let%op y =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." k
  in
  Ir.Tnode.update_prec y.Tensor.value Ir.Ops.single;
  y

let compile ?(transform = Fn.id) name output =
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (Train.forward output) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx output.Tensor.value, routine)

(* The output cells, row-major over (batch, oh, ow, oc), each as the products its reduction sums. *)
let cells ~round_x ~round_k =
  Ll_test.reduction_cells ~out:[| batch; rows; rows; oc |] ~over:[| window; window; ic |]
    (fun o r ->
      round_x (x_values [| o.(0); o.(1) + r.(0); o.(2) + r.(1); r.(2) |])
      *. round_k (k_values [| o.(3); r.(0); r.(1); r.(2) |]))

let sum_cells = List.map ~f:(List.sum (module Float) ~f:Fn.id)

let () =
  Stdio.eprintf "gpu_conv_twin_tiles: backend %s\n" backend;
  let rounded = cells ~round_x:half ~round_k:half in
  (* Rounded, the products are multiples of 2^-6 whose partial sums stay exact in f32, so every
     correct association returns the oracle's bits: compared bitwise below. *)
  p_all "the rounded products have exact partial sums at f32" rounded
    ~f:(Ll_test.exact_partials ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
  let oracle = Array.of_list (sum_cells rounded) in
  p_all "omitting either operand's rounding moves every output cell"
    [ (Fn.id, half); (half, Fn.id) ]
    ~f:(fun (round_x, round_k) ->
      let raw = Array.of_list (sum_cells (cells ~round_x ~round_k)) in
      Array.for_all2_exn raw oracle ~f:(fun r o -> Float.(r <> o)));
  let capture cell o =
    cell := Some o;
    o
  in
  let reference, _ = compile "gct_materialized" (product `Materialized) in
  p_all2 "the materialized-twin reference equals the host oracle bitwise" reference oracle
    ~f:Float.equal;
  let opt = ref None in
  let _, _ = compile "gct_capture" (product `Virtual) ~transform:(capture opt) in
  let opt = Option.value_exn !opt in
  let stored_opt = ref None in
  let _, _ = compile "gct_stored" (product `Stored) ~transform:(capture stored_opt) in
  (* A tile the environment withholds (routine logging, say) is not the backend's missing coverage:
     its skip carries the capability's own aggregation. *)
  let on_gpu = Ir.Backend_intf.binds_hardware_grid facts in
  let capability =
    Ll_test.tensorized_capability ~is_gpu:true ~is_cpu:false ~limits ~a_prec:Ir.Ops.half
      ~b_prec:Ir.Ops.half ~d_prec:Ir.Ops.single
  in
  let gpu_leg =
    Verdict.leg
      ~aggregation:
        (match capability with
        | `Withheld aggregation when on_gpu -> aggregation
        | `Withheld _ | `Advertised -> `Backend)
      ~when_:(on_gpu && match capability with `Advertised -> true | `Withheld _ -> false)
      ~on:(backend ^ " (no f16 x f16 -> f32 tile)")
      ()
  in
  let conv_seeds opt =
    Autotune.sketch_seed_params ~capabilities ~is_gpu:true ~is_cpu:false ~limits opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_conv)
  in
  let seeds = lazy (conv_seeds opt) in
  in_leg ~leg:gpu_leg "the backend's capability seeds conv tiles at the twin site" (fun label ->
      p_exists label (Lazy.force seeds) ~f:(fun q -> q.Autotune.sk_mma));
  (* The seeder must judge the twins' reads as f16 operands stored at f16, in format, tile and
     staged-traffic pricing alike: the same seeds, field for field. (The materialized twins'
     lowering, which computes the casts beside the conv, seeds no conv tile at all.) *)
  in_leg ~leg:gpu_leg "the twin site seeds exactly an f16-storage site's conv tiles" (fun label ->
      let labels seeds = List.map seeds ~f:Autotune.sketch_params_label in
      let got = labels (Lazy.force seeds)
      and want = labels (conv_seeds (Option.value_exn !stored_opt)) in
      let ok = List.equal String.equal got want in
      if not ok then
        Stdio.eprintf "%s: twin seeds [%s], f16-storage seeds [%s] (not part of the golden)\n" label
          (String.concat ~sep:"; " got) (String.concat ~sep:"; " want);
      p label ok);
  (* Recompile a fresh graph for each seed: node identity belongs to that lowering. *)
  let executions =
    lazy
      (List.mapi (Lazy.force seeds) ~f:(fun index q ->
           let got, routine =
             compile
               ("gct_seed_" ^ Int.to_string index)
               (product `Virtual)
               ~transform:(fun o ->
                 Ir.Schedule.apply
                   (Autotune.sketch_schedule ~accum_prec:capabilities.accum_prec ~p:q o)
                   o)
           in
           Stdio.eprintf "%s: %s %s (not part of the golden)\n" backend
             (Autotune.sketch_params_label q)
             (Ir.C_syntax.mma_summary_string routine.Context.mma);
           (got, routine)))
  in
  in_leg ~leg:gpu_leg "every seeded twin-site conv compiles to tensor code, no scalar fallback"
    (fun label ->
      p_all label (Lazy.force executions) ~f:(fun (_, r) ->
          Ll_test.all_tensorized ~what:r.Context.name (Ll_test.renderings r.Context.mma)));
  in_leg ~leg:gpu_leg "every seeded twin-site conv equals the materialized twin bitwise"
    (fun label ->
      p_all label (Lazy.force executions) ~f:(fun (got, r) ->
          let ok =
            Array.length got = Array.length reference
            && Array.for_all2_exn got reference ~f:Float.equal
          in
          if not ok then
            Stdio.eprintf "%s: values differ from the materialized twin (not part of the golden)\n"
              r.Context.name;
          ok))
