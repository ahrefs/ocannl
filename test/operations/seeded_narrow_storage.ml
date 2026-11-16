(* gh-ocannl-1405, gh-ocannl-2195: the untuned CUDA/HIP default seeds the sites whose accumulator
   precision is wider than their storage precision in every profile, and sinks them only under the
   numerics gate [gpu_default_seed_narrow_storage] ([Autotune.seeded_default]'s [seed_narrow]).

   Such a site is a bf16 gradient on CUDA and HIP (where [accum_prec] widens bf16 to f32), an f16
   one on CUDA and HIP under the default [fp16_arithmetic] (auto resolves wide there since
   gh-ocannl-2195), and fp8 on both. Backprop lowers a weight or bias gradient with the batch rows
   outermost, so the default reads the narrow cell, adds one contribution at f32 and rounds back to
   storage once per row. The seeding and the sink treat those sites differently: - a seeded
   contraction (the weight gradient) keeps its result bitwise: the sketch's [Privatize] mints its
   tile wide only where the default's serial rendering widens too (one accumulator-width decision,
   gh-ocannl-1151), so the tile stays at storage precision and narrows once per row, as the default
   does -- so it is seeded with the gate off, in every profile; - a sunk reduction (the bias
   gradient) does not: below its write loops the accumulator is one code generation widens, so the
   whole reduction runs at f32 and rounds once -- the summation order is the default's, only the
   number of roundings changes, towards the exact sum -- so only the gate admits it.

   Four parts. 1. Payloads: the [approximate] profile turns the gate on, and [reproducible] pins it
   off, at its default (part 4 executes the unset default). 2. Structure, on every backend: the
   seeding rule takes a narrow matmul output whatever the accumulator, and the plan derived for CUDA
   under a widening [accum_prec] (bf16, f16, fp8) seeds the weight gradient with the gate off, while
   only the gate adds the sink. 3. Executed, per precision (bf16, f16, fp8): one dense layer with
   narrow weight and bias gradients, the inputs NOT dyadic, compiled on the default pipeline and
   through the seeded default, under the default arithmetic (fast math on, explicit FMAs) and under
   the [reproducible] profile's (no contraction anywhere). Where the seeded default runs and
   [accum_prec] widens the precision (the backend's own resolution), the weight gradient is seeded
   with the gate off and stays bitwise the default's, and some default cell is further than one
   rounding from the f64 reference, so a seeded schedule narrowing elsewhere than the default would
   show. For bf16 the gate on also sinks the bias gradient, which then is closer to an f64 host
   reference than the default's, and within one bf16 rounding of it in every cell, which the
   default's is not. Where nothing widens, the gate changes nothing. 4. Configuration: the key
   reaches the seeded default and [model_default] when no override is passed. *)

(* cc-fp-contract: same-shape -- on cc both seeded and model defaults compile the plain default
   pipeline *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let key = "gpu_default_seed_narrow_storage"

(* The two arithmetics the seeding's parity is executed under (part 3). [`Default]: the default
   user's -- the GPU compilers' fast math on, as it defaults -- with every product accumulation an
   explicit FMA ([fp_contract=true], as seeded_default pins it), so that the bitwise claims are
   about summation order and rounding count, not the GPU compiler's per-kernel contraction.
   [`Reproducible]: the [reproducible] profile's ([fp_contract=false], both fast-math keys off), no
   contraction anywhere -- gh-ocannl-2195's contract. Stated key by key rather than through the
   profile, which would also pin the gate under test; read per compile, so a later setting governs
   the later compiles. *)
let set_arithmetic arith =
  let fp_contract, fast_math =
    match arith with `Default -> ("true", "true") | `Reproducible -> ("false", "false")
  in
  List.iter
    [
      ("OCANNL_FP_CONTRACT", fp_contract);
      ("OCANNL_CUDA_FAST_MATH", fast_math);
      ("OCANNL_HIP_FAST_MATH", fast_math);
    ]
    ~f:(fun (k, v) -> Unix.putenv k v)

let () = set_arithmetic `Default
let bf16 = Ir.Ops.bfloat16
let precs = [ bf16; Ir.Ops.half; Ir.Ops.fp8 ]
let name prec = Ir.Ops.prec_string prec

(* {2 1. Payloads} *)

let () =
  let payload name =
    Utils.parse_config_lines ~source:("profile " ^ name)
      (String.split_lines (List.Assoc.find_exn Utils.profile_payloads name ~equal:String.equal))
  in
  let stated name = List.Assoc.find (payload name) key ~equal:String.equal in
  p "the approximate profile turns the gate on"
    (Option.equal String.equal (stated "approximate") (Some "true"));
  p "the reproducible profile pins the gate off, its default"
    (Option.equal String.equal (stated "reproducible") (Some "false"))

(* {2 2. The seeding rule} *)

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* [accum_prec] widening [prec] to f32, as CUDA's resolution does. *)
let widen_of prec p = if Ir.Ops.equal_prec p prec then Ir.Ops.single else p

let () =
  let a =
    TDSL.ndarray
      (Array.create ~len:(128 * 64) 1.)
      ~label:[ "sns_a" ] ~input_dims:[ 64 ] ~output_dims:[ 128 ] ()
  in
  let b =
    TDSL.ndarray
      (Array.create ~len:(64 * 128) 1.)
      ~label:[ "sns_b" ] ~input_dims:[ 128 ] ~output_dims:[ 64 ] ()
  in
  let%op c = a * b in
  Tn.update_prec c.Tensor.value bf16;
  let lowered =
    Context.lowered_for_decisions ~name:"sns_matmul" (Context.auto ()) (Train.forward c)
      Ir.Indexing.Empty
  in
  p "the seeding rule seeds a bf16 matmul output, whatever its accumulator precision"
    (Option.is_some
       (Autotune.seeded_params ~limits:Ir.Backend_intf.no_hardware_limits ~min_reduction:1
          (scratch lowered)))

(* {2 3. Executed: a dense layer with narrow gradients, per precision} *)

let nb = 4
and ns = 64
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let x_values = values ~n:(nb * ns * ni) ~stride:389
let dy_values = values ~n:(nb * ns * no) ~stride:577

type layer = { tag : string; comp : Ir.Assignments.comp; wgrad : Tn.t; bgrad : Tn.t }

(* One dense layer whose weight and bias gradients are stored at [prec]. *)
let layer prec =
  let tag = name prec in
  let w =
    Operation.init ~l:("sns_w_" ^ tag) ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
      ~f:(fun idx ->
        Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
          idx)
      ~grad_spec:Tensor.Require_grad ()
  in
  let b =
    Operation.init ~l:("sns_b_" ^ tag) ~prec:Ir.Ops.single ~o:[ no ]
      ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    NTDSL.ndarray x_values ~label:[ "sns_x_" ^ tag ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
  in
  let dy =
    NTDSL.ndarray dy_values ~label:[ "sns_dy_" ^ tag ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
  in
  let%op y = (w * x) + b in
  let%op loss = (y *. dy) ++ "... => 0" in
  let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  let wgrad = grad_of w and bgrad = grad_of b in
  Tn.update_prec wgrad prec;
  Tn.update_prec bgrad prec;
  Train.set_materialized wgrad;
  Train.set_materialized bgrad;
  { tag; comp = Train.grad_update loss; wgrad; bgrad }

let layers = List.map precs ~f:(fun prec -> (prec, layer prec))

(* The f64 host references: [y.grad] is [dy], exactly, so [w.grad[o,i] = sum_r dy[r,o] * x[r,i]] and
   [b.grad[o] = sum_r dy[r,o]], over the f32 inputs. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)
let rows = nb * ns

let wgrad_ref =
  Array.init (no * ni) ~f:(fun c ->
      let o = c / ni and i = c % ni in
      let acc = ref 0. in
      for r = 0 to rows - 1 do
        acc := !acc +. (f32 dy_values.((r * no) + o) *. f32 x_values.((r * ni) + i))
      done;
      !acc)

let bgrad_ref =
  Array.init no ~f:(fun o ->
      let acc = ref 0. in
      for r = 0 to rows - 1 do
        acc := !acc +. f32 dy_values.((r * no) + o)
      done;
      !acc)

let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let error got want = Array.fold2_exn got want ~init:0. ~f:(fun acc g r -> acc +. Float.abs (g -. r))

(* Within one [prec] rounding of the reference: half a unit in the last place is at most [2^-(p-1) /
   2 = 2^-p] of the magnitude for a format with [p] significant bits (bf16 8, f16 11, fp8 e5m2 3);
   the f32 accumulation's own error is about 2^-24 per row, so [2^-p * |ref| + 2^-16] bounds a
   single rounding of a near-exact sum and no more. *)
let one_rounding prec want =
  Float.((abs want /. (2. ** Float.of_int (Ir.Ops.significand_bits prec))) +. (1. /. 65536.))

let run l compile =
  let ctx, routine = compile (Context.auto ()) l.comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx l.wgrad, Context.get_values ctx l.bgrad)

let seeded ?seed_narrow l =
  let report = ref None in
  let values =
    run l (fun ctx comp bindings ->
        Autotune.seeded_default ~min_reduction:128 ?seed_narrow
          ~report:(fun r -> report := Some r)
          ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

(* Where the seeding of a widening site can show: the seeded default runs here, and the backend's
   own accumulator resolution widens [prec]. *)
let widens_here prec =
  let ctx = Context.auto () in
  Autotune.seeded_default_backend (Context.backend_facts_of_name backend_name)
  && not
       (Ir.Ops.equal_prec ((Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec prec) prec)

let census (r : Autotune.seeded_choice) =
  List.map r.sd_seeded ~f:(fun (at, p) ->
      Printf.sprintf "u%d %s" at (Autotune.sketch_params_label p))
  @ List.map r.sd_sunk ~f:(fun (at, n) -> Printf.sprintf "u%d sunk %d" at n)

(* The structural leg (gh-ocannl-1626): per-PR CI runs cc, where the claims gated on [widens_here]
   below are skipped. Which segments the plan seeds and sinks is the seeded plan's decision, derived
   on the host ([Autotune.seeded_plan]), so it is derived here for "cuda" whatever the host, on the
   lowering this host captures, with [widen_of prec] standing in for CUDA's resolution (the
   [widens_here] premise). The control: under an accumulator that does not widen, the gate changes
   no decision, so the sink census moves with the premise, not with the flag alone. Runs before part
   4 states the key. *)
let () =
  List.iter layers ~f:(fun (prec, l) ->
      let opt = Context.lowered_for_decisions (Context.auto ()) l.comp Ir.Indexing.Empty in
      let decisions ?seed_narrow accum_prec =
        match
          Autotune.seeded_plan ~min_reduction:128 ?seed_narrow ~facts:Ir.Backend_intf.cuda_facts
            ~limits:Ir.Backend_intf.no_hardware_limits ~accum_prec ~static_indices:[] opt
        with
        | Some { Autotune.sp_segments = Some (_, ds); _ } ->
            List.map ds ~f:(function
              | Autotune.Seeded_sketch { params; _ } ->
                  "sketch " ^ Autotune.sketch_params_label params
              | Autotune.Seeded_sunk { swaps; _ } -> Printf.sprintf "sunk %d" swaps
              | Autotune.Seeded_preset -> "preset")
        | _ -> []
      in
      let count prefix ds = List.count ds ~f:(String.is_prefix ~prefix) in
      let sketches ds = List.filter ds ~f:(String.is_prefix ~prefix:"sketch") in
      let widen = widen_of prec in
      let off = decisions ~seed_narrow:false widen and on = decisions ~seed_narrow:true widen in
      Stdio.eprintf "structural %s: off [%s] on [%s] (not part of the golden)\n%!" l.tag
        (String.concat ~sep:"; " off) (String.concat ~sep:"; " on);
      pf
        "structural, %s: under a widening accumulator the plan derived for cuda seeds the weight \
         gradient with the gate off"
        l.tag
        (count "sketch" off >= 1);
      pf "structural, %s: the gate leaves the seeded sketches as they are" l.tag
        (List.equal String.equal (sketches off) (sketches on));
      pf "structural, %s: the gate adds the sink of the bias gradient" l.tag
        (count "sunk" off < count "sunk" on);
      pf "structural, %s: with the key unset the plan decides as with the gate off" l.tag
        (List.equal String.equal (decisions widen) off);
      pf
        "structural control, %s: under an accumulator that does not widen, the gate changes nothing"
        l.tag
        ((not (List.is_empty (decisions Fn.id)))
        && List.equal String.equal
             (decisions ~seed_narrow:true Fn.id)
             (decisions ~seed_narrow:false Fn.id)))

(* The seeding's parity, per precision, the gate off (its default): where [accum_prec] widens the
   precision the weight gradient is seeded, and both gradients are bitwise the default pipeline's
   everywhere. The power of the weight gradient's claim: some default cell is further than one
   rounding from the f64 reference, so the default rounds more than once per cell, and a seeded
   schedule rounding at other points (a tile held wide across the whole reduction, say) would differ
   there. *)
let parity ~arith =
  set_arithmetic arith;
  (* The reproducible run's labels are the ones the golden has always carried. *)
  let tag l =
    match arith with `Reproducible -> l.tag | `Default -> l.tag ^ " (default arithmetic)"
  in
  List.map layers ~f:(fun (prec, l) ->
      let widens = widens_here prec in
      Stdio.eprintf "seeded_narrow_storage: %s widens=%b on %s (not part of the golden)\n%!" (tag l)
        widens backend_name;
      let ((w_default, b_default) as default) = run l Context.compile in
      let (w_off, b_off), r_off = seeded ~seed_narrow:false l in
      List.iter (census r_off) ~f:(fun line ->
          Stdio.eprintf "%s, gate off: %s (not part of the golden)\n" (tag l) line);
      p_exists
        (tag l ^ ": the weight gradient is not trivially zero")
        (Array.to_list w_default)
        ~f:(fun v -> Float.(abs v > 1e-3));
      Verdict.gated ~when_:widens ~on:backend_name
        (Printf.sprintf
           "where accum_prec widens %s, with the gate off the weight gradient is seeded" (tag l))
        (not (List.is_empty r_off.sd_seeded));
      Verdict.gated ~when_:widens ~on:backend_name
        (Printf.sprintf
           "where accum_prec widens %s, the seeded routine compiles without falling back" (tag l))
        (not r_off.sd_fell_back);
      p_all2
        (tag l ^ ": with the gate off the weight gradient is bitwise the default pipeline's")
        w_off w_default ~f:bitwise;
      p_all2
        (tag l ^ ": with the gate off the bias gradient is bitwise the default pipeline's")
        b_off b_default ~f:bitwise;
      Verdict.gated_exists ~when_:widens ~on:backend_name
        (Printf.sprintf
           "where accum_prec widens %s, some default weight-gradient cell is further than one %s \
            rounding from the f64 reference (the parity's power)"
           (tag l) l.tag)
        (List.zip_exn (Array.to_list w_default) (Array.to_list wgrad_ref))
        ~f:(fun (g, r) -> Float.(abs (g -. r) > one_rounding prec r));
      Stdio.eprintf "%s abs error vs f64: weight %g, bias %g (not part of the golden)\n%!" (tag l)
        (error w_default wgrad_ref) (error b_default bgrad_ref);
      (prec, (l, default, r_off)))

let (_ : (Ir.Ops.prec * (layer * (float array * float array) * Autotune.seeded_choice)) list) =
  parity ~arith:`Default

let defaults = parity ~arith:`Reproducible

(* The gate, on bf16: it adds the sink of the bias gradient, which changes it towards the exact sum,
   and leaves the seeded weight gradient as it was. *)
let () =
  let l, (w_default, b_default), r_off =
    List.Assoc.find_exn defaults bf16 ~equal:Ir.Ops.equal_prec
  in
  let narrow_here = widens_here bf16 in
  let (w_on, b_on), r_on = seeded ~seed_narrow:true l in
  List.iter (census r_on) ~f:(fun line ->
      Stdio.eprintf "admitted: %s (not part of the golden)\n" line);
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the gate off the seeded default sinks fewer segments than \
     with it on, and seeds the same"
    (List.length r_off.sd_sunk < List.length r_on.sd_sunk
    && List.equal String.equal
         (List.map r_off.sd_seeded ~f:(fun (_, p) -> Autotune.sketch_params_label p))
         (List.map r_on.sd_seeded ~f:(fun (_, p) -> Autotune.sketch_params_label p)));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the gate on the bias gradient is sunk"
    (not (List.is_empty r_on.sd_sunk));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the admitted routine compiles without falling back"
    (not r_on.sd_fell_back);
  Stdio.eprintf
    "abs error vs f64: weight default %g admitted %g; bias default %g admitted %g (not part of the \
     golden)\n\
     %!"
    (error w_default wgrad_ref) (error w_on wgrad_ref) (error b_default bgrad_ref)
    (error b_on bgrad_ref);
  p_all2 "with the gate on the seeded weight gradient is bitwise the default pipeline's" w_on
    w_default ~f:bitwise;
  Verdict.gated_exists ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the gate changes the bias gradient in some cell"
    (List.zip_exn (Array.to_list b_on) (Array.to_list b_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the admitted bias gradient is closer to the f64 reference"
    Float.(error b_on bgrad_ref < error b_default bgrad_ref);
  Verdict.gated_all ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, every admitted bias-gradient cell is one bf16 rounding of the \
     reference"
    [ (b_on, bgrad_ref) ]
    ~f:(fun (got, want) ->
      Verdict.within
        "where accum_prec widens bf16, every admitted bias-gradient cell is one bf16 rounding of \
         the reference"
        got want ~tol:(one_rounding bf16));
  Verdict.gated_exists ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, some default bias-gradient cell is not (the control)"
    (List.zip_exn (Array.to_list b_default) (Array.to_list bgrad_ref))
    ~f:(fun (g, r) -> Float.(abs (g -. r) > one_rounding bf16 r));
  Verdict.gated_all ~when_:(not narrow_here) ~on:backend_name
    "where nothing widens, the gate leaves the bias gradient bitwise the default's"
    (List.zip_exn (Array.to_list b_on) (Array.to_list b_default))
    ~f:(fun (a, b) -> bitwise a b);
  (* {2 4. The configuration key} Unset (the test config does not state it), the seeded default
     decides as with the gate off; stated true, as with it on. *)
  let _, r_unset = seeded l in
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the key unset the seeded default decides as with the gate \
     off, short of what the gate takes"
    (List.equal String.equal (census r_unset) (census r_off)
    && List.length (census r_unset) < List.length (census r_on));
  Unix.putenv ("OCANNL_" ^ String.uppercase key) "true";
  let _, r_keyed = seeded l in
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the key stated true decides what the gate decides"
    ((not (List.is_empty (census r_on))) && List.equal String.equal (census r_keyed) (census r_on));
  (* Under the configured [auto] gate, for the record: the ranking prices a launch at its fill
     (gh-ocannl-1479), which on a device it leaves idle can decline the contraction's sketch, and
     the admitted reduction sink then takes the weight gradient as it takes the bias gradient. *)
  let auto_report = ref None in
  ignore
    (run l (fun ctx comp bindings ->
         Autotune.seeded_default ~report:(fun r -> auto_report := Some r) ctx comp bindings)
      : float array * float array);
  Option.iter !auto_report ~f:(fun r ->
      List.iter (census r) ~f:(fun line ->
          Stdio.eprintf "auto gate, key true: %s (not part of the golden)\n" line));
  (* [model_default] starts from the seeded plan under the configured gate, and its
     numerics-preserving set admits the narrow contraction as the seeding does: whichever scalar
     blocktile it picks, the weight gradient keeps the default's narrowing points. The seeding
     threshold is pinned to the one the seeded runs above use, under which the plan seeds the
     contraction: under [auto] whether it does depends on the device's launch fill, and where it
     does not, the plan sinks the weight gradient and the model's incumbent rounds it once, as the
     bias's (gh-ocannl-1599's triage). *)
  Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "128";
  let w_model, _ =
    run l (fun ctx comp bindings ->
        Autotune.model_default
          ~report:(fun (c : Autotune.model_choice) ->
            Stdio.eprintf "model pick with the key true: %s (not part of the golden)\n%!" c.mc_label)
          ctx comp bindings)
  in
  p_all2 "with the key true the model's weight gradient is bitwise the default pipeline's" w_model
    w_default ~f:bitwise
