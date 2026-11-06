(* Collapsing a scope keeps the scope's rounding (gh-ocannl-1788).

   A [Local_scope] converts its value into its own precision before its slot reads it. [simplify]
   collapses a single-assignment scope into the expression around it, which drops that step, so the
   collapse is exact only when the slot evaluates at the scope's precision, or when the value is a
   load whose storage precision the scope's widens exactly, and it is never exact for a cast
   (gh-ocannl-1866): a narrow scope over a read it does not widen exactly keeps its rounding even in
   a slot of its own precision, which can render wider, as an [Ops.Round_prec] in the expression
   (gh-ocannl-1975) or, over anything but a plain load, as a scope code generation gives its node's
   precision whatever the compute precision. Before gh-ocannl-1788 it collapsed every float scope: a
   bfloat16 scope over a single product, read in a single sum, lost its rounding wherever the
   scope's local is declared at bfloat16 -- Metal, CUDA and HIP, and cc under
   [narrow_compute_f32=false]. (Under cc's default [narrow_compute_f32=true] the local is single
   anyway, so the collapse happened to be exact there.)

   Each shape is executed in four forms: the scoped form through the [?prelowered] seam, its
   assignment under an always-true guard (the uncollapsed oracle: see {!kept_scope}); the scope
   through a schedule whose trailing [simplify] decides the collapse; a virtual node read in the
   sum, inlined by [optimize], whose own [simplify] decides it; and the value written straight into
   the sum, which shows that the oracle's rounding is observable where the claim needs it to be. A
   cast also runs as a materialized node, the value every form of it has to give. All run under each
   narrow-compute policy ([Ir.Numerics.set_policy] in-process, distinct routine names per policy).
   The single inputs and their squares lie off the bfloat16 grid; the bfloat16 inputs lie on it, so
   seeding them is exact.

   A scope [simplify] keeps can still collapse at render time, where the backend's compute precision
   makes it exact (gh-ocannl-1887): [C_syntax.collapse_exact_scopes], so that the explicit-SIMD
   renderings can take the loop. Those shapes run retyped [Vectorized] at an extent of several
   vector groups and a remainder, against the oracle bitwise, and the emitted kernel says which of
   the two happened. The collapse is predicted from the backend's own [compute_prec] ({!collapses}).
   The rest of the file is the shapes where the three precisions agree and it still must not
   happen. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated
open Verdict.Claims

(* The forms compared bitwise are different kernel shapes, so the C compiler's contraction must not
   fuse the sum into a product in one of them only. *)
let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Generated.init ~backend_name
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)
let cc_leg = Verdict.leg ~when_:on_cpu ~on:backend_name ()
let capabilities = lazy (Context.codegen_capabilities (Context.auto ()))
let compute p = (Lazy.force capabilities).Ir.Backend_intf.compute_prec p
let bf = Ops.bfloat16
let f32 = Ops.single
let f64 = Ops.double
let fp8 = Ops.fp8
let half = Ops.half
let n = 4

(* Several 4-lane groups and a remainder. *)
let long = 11

let factories =
  List.concat_mapi [ bf; f32; f64; fp8; half ] ~f:(fun p prec ->
      List.mapi [ n; long ] ~f:(fun e extent ->
          ( (prec, extent),
            Ll_test.node_factory ~prec
              ~first_id:(24700 + (100 * ((2 * p) + e)))
              ~dims:[| extent |] () )))

let node ?(extent = n) prec =
  snd (List.find_exn factories ~f:(fun ((p, e), _) -> Ops.equal_prec p prec && e = extent))

let bnode label = node bf label
let fnode label = node f32 label

(* 1 + (2k + 1/4) * 2^-10: off the bfloat16 grid (8 significant bits), and so are their squares. *)
let f32_inputs extent =
  Array.init extent ~f:(fun k -> 1. +. ((Float.of_int (2 * k) +. 0.25) *. 0.0009765625))

(* Exactly representable in bfloat16, so seeding a bfloat16 node is exact. A product of two is exact
   in single and off the bfloat16 grid, so no compiler's contraction can move it, and a rounding to
   bfloat16 can. *)
let bf_inputs extent = Array.init extent ~f:(fun k -> 1. +. (Float.of_int (k + 1) *. 0.0078125))

(* The assignment sits under an always-true guard: outside the single-assignment shape both
   [simplify] and {!Ir.C_syntax}'s render-time collapse act on, so the local is really declared. A
   plain [?prelowered] scope no longer shows that, whichever way the claims around it read: the
   render-time collapse would compare the collapse with itself. [`Oracle] is checked to hold the
   local in its emitted kernel. *)
let kept_scope ~orig_indices tn v =
  let id = LL.get_scope tn in
  Ll_test.scope_of ~orig_indices id
    (LL.If { cond = (Ll_test.c 1., f32); body = LL.Set_local (id, v) })

let plain_scope ~orig_indices tn v = Ll_test.scope ~orig_indices tn v
let routine_name label tag policy = Printf.sprintf "scr_%s_%s_%s" label tag policy

(* Whether a kernel still declares the local of the node suffixed [mid]. The node is the scope's and
   nothing else's, so its name occurs in the kernel exactly when the local does. *)
let holds_local ?(mid = "_mid") routine =
  String.is_substring (Generated.read routine) ~substring:mid

type shape = {
  label : string;
  slot : Ops.prec;  (** the destination's, hence the sum's evaluation precision *)
  scope_prec : Ops.prec;
  cast : bool;  (** a read the scope does not widen exactly: its local takes its node's precision *)
  value : x:LL.scalar_arg -> y:LL.scalar_arg -> LL.scalar_t;  (** the scope's value *)
}

(* Whether the render-time collapse should take a kept scope: the three precisions of [T_R v =
   conv(V->R)(e@V); ... conv(R->S)(v)] agree. A cast's [R] and [V] are its node's precision,
   anything else's are the compute precision. *)
let collapses shape =
  let scope_prec = if shape.cast then shape.scope_prec else compute shape.scope_prec in
  Ops.equal_prec scope_prec (compute shape.slot)

(* [out[i] = x[i] + scope(value)], with [x] at the slot's precision and [y] at the other one.
   [`Oracle] runs it under {!kept_scope}, [`Scheduled] through [Schedule.apply]'s trailing simplify,
   [`Simd] retyped [Vectorized] the same way, and [`Unscoped] runs the value written straight into
   the sum -- the collapse done by hand, without the scope's rounding. [`Inlined] reaches the scope
   the ordinary way: a virtual node computing the value, read in the sum and inlined by [optimize],
   whose own simplify decides the collapse. Returns the values, the number of scopes left in the
   program that ran, and whether the scope's node is virtual there. *)
let run ?(extent = n) shape ~form ~policy =
  let tag =
    match form with
    | `Oracle -> if extent = n then "pre" else "keep"
    | `Scheduled -> "sch"
    | `Simd -> "vec"
    | `Unscoped -> "raw"
    | `Inlined -> "inl"
    | `Materialized -> "mat"
  in
  let name = routine_name shape.label tag policy in
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let node prec suffix = node ~extent prec (name ^ suffix) in
  let other = if Ops.is_narrow_float shape.slot then f32 else bf in
  let x_tn = node shape.slot "_x" and y_tn = node other "_y" in
  let mid = node shape.scope_prec "_mid" and out = node shape.slot "_out" in
  List.iter [ x_tn; y_tn; out ] ~f:Ll_test.materialize;
  let x = (Ll_test.get x_tn i, shape.slot) and y = (Ll_test.get y_tn i, other) in
  let v = shape.value ~x ~y in
  let sum operand = Ll_test.loop_n s extent (Ll_test.set out i (LL.Binop (Ops.Add, x, operand))) in
  let raw = sum (v, shape.slot) in
  let o =
    match form with
    | `Unscoped -> Ll_test.optimize_scoped ~name ~raw raw
    | (`Inlined | `Materialized) as placed ->
        let materialized =
          match placed with
          | `Inlined ->
              Ll_test.virtualize mid;
              [ x_tn; y_tn; out ]
          | `Materialized ->
              Ll_test.materialize mid;
              [ x_tn; y_tn; out; mid ]
        in
        let s1 = Ll_test.sym () in
        let i1 = [| Ll_test.iter s1 |] in
        let v1 = shape.value ~x:(Ll_test.get x_tn i1, shape.slot) ~y:(Ll_test.get y_tn i1, other) in
        Ll_test.optimize ~materialized ~name
          (LL.unflat_lines
             [
               Ll_test.loop_n s1 extent (Ll_test.set mid i1 v1);
               sum (Ll_test.get mid i, shape.scope_prec);
             ])
    | `Oracle ->
        Ll_test.optimize_scoped ~name ~raw
          (sum (kept_scope mid ~orig_indices:i v, shape.scope_prec))
    | `Scheduled | `Simd ->
        Ll_test.optimize_scoped ~name ~raw
          (sum (Ll_test.scope mid ~orig_indices:i v, shape.scope_prec))
  in
  let o =
    match form with
    | `Scheduled -> Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o
    | `Simd -> Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Vectorized } ] o
    | `Oracle | `Unscoped | `Inlined | `Materialized -> o
  in
  let seed tn =
    ( tn,
      (if Ops.is_narrow_float (Lazy.force tn.Ir.Tnode.storage_prec) then bf_inputs else f32_inputs)
        extent )
  in
  let values = Ll_test.execute ~name o ~seed:[ seed x_tn; seed y_tn ] ~read:[ out ] in
  (List.hd_exn values, Ll_test.count_scopes o.LL.llc, Ll_test.known_virtual o mid)

let bits vs = Array.map vs ~f:(fun v -> Int64.bits_of_float v)
let same_bits a b = Array.equal Int64.equal (bits a) (bits b)

let narrow_product =
  (* A bfloat16 scope over a single product of single loads, in a single sum. *)
  {
    label = "narrow";
    slot = f32;
    scope_prec = bf;
    cast = false;
    value = (fun ~x ~y:_ -> LL.Binop (Ops.Mul, x, x));
  }

let wide_product =
  (* A single scope over a single product of single loads, in a bfloat16 sum. *)
  {
    label = "wide";
    slot = bf;
    scope_prec = f32;
    cast = false;
    value = (fun ~x:_ ~y -> LL.Binop (Ops.Mul, y, y));
  }

let fp8_product =
  (* An fp8 scope over a product of bfloat16 loads, in a single sum. Every backend computes fp8 in
     single by default, so this is the shape that collapses on the GPUs too. *)
  {
    label = "fp8";
    slot = f32;
    scope_prec = fp8;
    cast = false;
    value = (fun ~x:_ ~y -> LL.Binop (Ops.Mul, y, y));
  }

let narrow_load =
  (* A bfloat16 scope over a bfloat16 load, in a single sum: the scope's rounding is the
     identity. *)
  { label = "nload"; slot = f32; scope_prec = bf; cast = false; value = (fun ~x:_ ~y -> fst y) }

let widening_load =
  (* A single scope over a bfloat16 load, in a bfloat16 sum: the widening is exact. *)
  { label = "wload"; slot = bf; scope_prec = f32; cast = false; value = (fun ~x ~y:_ -> fst x) }

let cast_into_own_prec =
  (* A bfloat16 scope over a single load, in a bfloat16 sum: the shape a cast twin has in a model of
     its own precision. *)
  { label = "cast_own"; slot = bf; scope_prec = bf; cast = true; value = (fun ~x:_ ~y -> fst y) }

let cast_into_own_half =
  (* The same at half, which a backend can compute at its own width and vectorize. *)
  { cast_into_own_prec with label = "cast_half"; slot = half; scope_prec = half }

let cast_into_wider =
  (* A bfloat16 scope over a single load, in a single sum. *)
  { label = "cast_wide"; slot = f32; scope_prec = bf; cast = true; value = (fun ~x ~y:_ -> fst x) }

(* A narrowing init does not make a scope a cast. [mid] is a bfloat16 accumulator opened by a single
   load and then updated: it keeps the residency of a reduction ([accum_prec]), so increments of
   [2^-9], a quarter of its ulp at 1, add up where it resides wide and round away where it resides
   at bfloat16. Its init, 1, is on the grid either way. *)
let accumulator_keeps_residency ~policy =
  let name = "scr_accum_" ^ policy in
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] and at0 = [| Ll_test.fixed 0 |] in
  let x_tn = fnode (name ^ "_x") and y_tn = fnode (name ^ "_y") in
  let mid = bnode (name ^ "_mid") and out = bnode (name ^ "_out") in
  List.iter [ x_tn; y_tn; out ] ~f:Ll_test.materialize;
  let id = LL.get_scope mid in
  let body =
    LL.Seq
      ( LL.Set_local (id, Ll_test.get y_tn at0),
        Ll_test.loop_n s n
          (LL.Set_local (id, LL.Binop (Ops.Add, (LL.Get_local id, bf), (Ll_test.get x_tn i, bf))))
      )
  in
  let scoped =
    Ll_test.set out at0
      (LL.Local_scope { id; body; orig_indices = at0; mint = LL.Inlined_computation })
  in
  let raw =
    Ll_test.loop_n s n
      (Ll_test.set out at0
         (LL.Binop (Ops.Add, (Ll_test.get y_tn at0, f32), (Ll_test.get x_tn i, f32))))
  in
  let o = Ll_test.optimize_scoped ~name ~raw scoped in
  let seed = [ (x_tn, Array.create ~len:n 0.001953125); (y_tn, Array.create ~len:n 1.) ] in
  let values = Ll_test.execute ~name o ~seed ~read:[ out ] in
  let accum = (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.accum_prec bf in
  let expected = if Ops.equal_prec accum bf then 1. else 1.0078125 in
  p "a narrowing init leaves an accumulator at its residency"
    Float.((List.hd_exn values).(0) = expected)

(* A kept scope retyped [Vectorized], against the oracle. A value parity alone passes with the
   collapse switched off: the claims on the kernel are what fail then. *)
let render_time_collapse shape ~policy =
  let keep_name = routine_name shape.label "keep" policy in
  let vec_name = routine_name shape.label "vec" policy in
  let kept, _, _ = run ~extent:long shape ~form:`Oracle ~policy in
  let simd, simd_scopes, _ = run ~extent:long shape ~form:`Simd ~policy in
  pf "%s: the oracle holds its local in the kernel" shape.label (holds_local keep_name);
  pf "%s: a scope survives lowering, a cast rounding in the expression instead" shape.label
    (simd_scopes = if shape.cast then 0 else 1);
  pf "%s: the kernel holds the local unless it rounds in the expression or compute makes it exact"
    shape.label
    (Bool.equal (holds_local vec_name) (not (shape.cast || collapses shape)));
  pf "%s: the retyped form computes the oracle's values bitwise" shape.label (same_bits kept simd);
  (* A cast rounds in lanes where its load computes at the lanes' width: at a narrower one,
     [vec_expr] would be narrowing what it had not loaded yet. *)
  in_legf ~leg:cc_leg
    "%s: explicit SIMD exactly where the expression has no scope and fits the lanes" shape.label
    (fun label ->
      p label
        (Bool.equal
           (String.is_substring (Generated.read vec_name) ~substring:"ocannl_vec")
           (if shape.cast then Ops.equal_prec (compute shape.slot) Ops.single else collapses shape)))

let every_backend = Verdict.leg ~when_:true ~on:backend_name ()

(* One statement built twice, its scopes kept and left to the render-time collapse, which must give
   the same values whatever it decides. [collapses]: whether it should take this one. *)
let must_agree ?(leg = every_backend) ?(collapses = false) label ~policy build =
  let compile_and_run tag scope =
    let name = routine_name label tag policy in
    let o, seed, out = build ~name ~scope in
    (name, List.hd_exn (Ll_test.execute ~name o ~seed ~read:[ out ]))
  in
  let runs =
    lazy
      (let _, kept = compile_and_run "keep" kept_scope in
       let name, scoped = compile_and_run "col" plain_scope in
       Stdio.eprintf "(not part of the golden) %s %s: %s\n" policy label
         (String.concat ~sep:" " (Array.to_list (Array.map scoped ~f:(Printf.sprintf "%h"))));
       (kept, name, scoped))
  in
  in_legf ~leg "%s: the kept form's values, bitwise" label (fun claim ->
      let kept, _, scoped = Lazy.force runs in
      p claim (same_bits kept scoped));
  in_legf ~leg "%s: the local stays unless the collapse is exact" label (fun claim ->
      let _, name, _ = Lazy.force runs in
      p claim (Bool.equal (holds_local name) (not collapses)))

(* gh-ocannl-1958: a scope of the slot's own precision over a value whose rendering is not typed at
   that precision. The precisions agree, yet the local is what rounds the value: [T v = e; ... v
   ...] is a conversion [e] alone does not make, so collapsed, the arithmetic around it runs in the
   wider type (or, for two truths under [/], divides C integers). [simplify] keeps such a scope
   ([keeps_scopes]) by {!Ir.Low_level.typed_at}. Each value is computed into virtual nodes, read in
   the expression [read], and compared with the scope kept, with the schedule's trailing simplify
   and, unless wide compute takes the node's rounding from the scope, with a materialized node.
   [observable]: the leg of the backends rendering the value wider, where it written straight into
   the expression has to come out different, so the claims above can fail. *)
let wider_rendering ?(materialized_agrees = true) ?observable ?(keeps_scopes = true) label ~policy
    ~prec ~inputs ~values ~read =
  let run form =
    let tag =
      match form with
      | `Oracle -> "keep"
      | `Scheduled -> "sch"
      | `Unscoped -> "raw"
      | `Inlined -> "inl"
      | `Materialized -> "mat"
    in
    let name = routine_name label tag policy in
    let s = Ll_test.sym () in
    let i = [| Ll_test.iter s |] in
    let in_tns = List.mapi inputs ~f:(fun k _ -> node prec (Printf.sprintf "%s_in%d" name k)) in
    let out = node prec (name ^ "_out") in
    let observed = out :: in_tns in
    List.iter observed ~f:Ll_test.materialize;
    let operands i = List.map in_tns ~f:(fun tn -> (Ll_test.get tn i, prec)) in
    let vs = values (operands i) in
    let mids = List.mapi vs ~f:(fun k _ -> node prec (Printf.sprintf "%s_mid%d" name k)) in
    let at_prec v = (v, prec) in
    let write reads =
      Ll_test.loop_n s n (Ll_test.set out i (read (List.map reads ~f:at_prec) (operands i)))
    in
    let raw = write vs in
    let scoped place =
      write (List.map2_exn mids vs ~f:(fun mid v -> place ~orig_indices:i mid v))
    in
    let o =
      match form with
      | `Unscoped -> Ll_test.optimize_scoped ~name ~raw raw
      | `Oracle -> Ll_test.optimize_scoped ~name ~raw (scoped kept_scope)
      | `Scheduled ->
          Ir.Schedule.apply
            [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ]
            (Ll_test.optimize_scoped ~name ~raw (scoped plain_scope))
      | (`Inlined | `Materialized) as placed ->
          let materialized =
            match placed with
            | `Inlined ->
                List.iter mids ~f:Ll_test.virtualize;
                observed
            | `Materialized ->
                List.iter mids ~f:Ll_test.materialize;
                mids @ observed
          in
          let fill k mid =
            let s1 = Ll_test.sym () in
            let i1 = [| Ll_test.iter s1 |] in
            Ll_test.loop_n s1 n (Ll_test.set mid i1 (List.nth_exn (values (operands i1)) k))
          in
          Ll_test.optimize ~materialized ~name
            (LL.unflat_lines
               (List.mapi mids ~f:fill @ [ write (List.map mids ~f:(fun mid -> Ll_test.get mid i)) ]))
    in
    let seed = List.zip_exn in_tns inputs in
    let result = List.hd_exn (Ll_test.execute ~name o ~seed ~read:[ out ]) in
    (name, result, Ll_test.count_scopes o.LL.llc, List.map mids ~f:(Ll_test.known_virtual o))
  in
  let show vs = String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h"))) in
  let keep_name, kept, _, _ = run `Oracle in
  let _, sch, _, _ = run `Scheduled in
  let _, inl, inl_scopes, inlined = run `Inlined in
  Stdio.eprintf "(not part of the golden) %s %s: kept %s; inlined %s\n" policy label (show kept)
    (show inl);
  pf "%s: the oracle holds its local in the kernel" label (holds_local keep_name);
  pf "%s: the schedule's simplify keeps the oracle's values bitwise" label (same_bits kept sch);
  p_all (Printf.sprintf "%s: optimize inlines the virtual nodes" label) inlined ~f:Fn.id;
  pf "%s: optimize keeps a scope exactly where the collapse would not be exact" label
    (inl_scopes = if keeps_scopes then List.length inlined else 0);
  pf "%s: the inlined form computes the oracle's values bitwise" label (same_bits kept inl);
  (if materialized_agrees then
     let _, mat, _, _ = run `Materialized in
     pf "%s: every placement computes the materialized node's values bitwise" label
       (same_bits mat kept && same_bits mat inl));
  Option.iter observable ~f:(fun leg ->
      in_legf ~leg "%s: the value written straight into the expression differs" label (fun claim ->
          let _, raw, _, _ = run `Unscoped in
          p claim (not (same_bits kept raw))))

(* What a single holds of a double: the value a [float] local would. *)
let single_of v = Stdlib.Int32.(float_of_bits (bits_of_float v))
let two_to_minus_30 = 0.000000000931322574615478515625

let rendered_wider ~policy =
  let tiny = Array.create ~len:n two_to_minus_30 in
  let unary op = function [ x; _ ] -> [ LL.Unop (op, x) ] | _ -> assert false in
  let sum reads operands =
    match (reads, operands) with [ v ], [ _; y ] -> LL.Binop (Ops.Add, v, y) | _ -> assert false
  in
  (* [(gate + tiny) - y], the gate passing [y]. Single gives 0: [y + tiny] rounds back to [y]. In
     double, [tiny] survives. The sum alone could not tell: 53 bits make double rounding of one
     operator exact. *)
  let gated op =
    ( (function [ x; y; _ ] -> [ LL.Binop (op, x, y) ] | _ -> assert false),
      fun reads operands ->
        match (reads, operands) with
        | [ v ], [ _; y; tiny ] -> LL.Binop (Ops.Sub, (LL.Binop (Ops.Add, v, tiny), f32), y)
        | _ -> assert false )
  in
  let x = [| 3.; 7.; 11.; 13. |] in
  wider_rendering "f32_recip" ~policy ~prec:f32 ~observable:cc_leg
    ~inputs:[ x; Array.map x ~f:(fun v -> -.single_of (1. /. v)) ]
    ~values:(unary Ops.Recip) ~read:sum;
  wider_rendering "f32_recip_sqrt" ~policy ~prec:f32 ~observable:cc_leg
    ~inputs:[ x; Array.map x ~f:(fun v -> -.single_of (1. /. single_of (Float.sqrt v))) ]
    ~values:(unary Ops.Recip_sqrt) ~read:sum;
  let y = [| 1.; 1.5; 2.; 2.5 |] in
  let values, read = gated Ops.Relu_gate in
  wider_rendering "f32_relu_gate" ~policy ~prec:f32 ~observable:cc_leg ~inputs:[ x; y; tiny ]
    ~values ~read;
  let values, read = gated Ops.Satur01_gate in
  wider_rendering "f32_satur01_gate" ~policy ~prec:f32 ~observable:cc_leg
    ~inputs:[ [| 0.125; 0.25; 0.5; 0.75 |]; y; tiny ]
    ~values ~read;
  (* [Not] is no truth at a float precision: cc spells it [x == 0.0 ? 1.0 : 0.0], so it reads as the
     gates do. *)
  wider_rendering "f32_not" ~policy ~prec:f32 ~observable:cc_leg
    ~inputs:[ [| 0.; 0.; 1.; 0. |]; [| 1.; 1.; 0.; 1. |]; tiny ]
    ~values:(function [ x; _; _ ] -> [ LL.Unop (Ops.Not, x) ] | _ -> assert false)
    ~read:(snd (gated Ops.Relu_gate));
  (* Two truths are C [int], so the collapsed quotient divides integers. 1/0 traps on x86, which is
     the negative control there. *)
  wider_rendering "f32_truth" ~policy ~prec:f32
    ~inputs:[ [| 1.; 1.5; 3.; 4. |]; Array.create ~len:n 2. ]
    ~values:(function
      | [ x; two ] -> [ LL.Binop (Ops.Cmplt, x, two); LL.Binop (Ops.Cmplt, two, x) ]
      | _ -> assert false)
    ~read:(fun reads _ ->
      match reads with [ a; b ] -> LL.Binop (Ops.Div, a, b) | _ -> assert false);
  (* Outside a division a truth converts exactly, and it has to collapse: the select would test a
     local, and the one-hot gather would not see its [Cmpeq]. *)
  wider_rendering "f32_truth_tested" ~policy ~prec:f32 ~keeps_scopes:false
    ~inputs:[ [| 1.; 1.5; 3.; 4. |]; Array.create ~len:n 2. ]
    ~values:(function [ x; two ] -> [ LL.Binop (Ops.Cmplt, x, two) ] | _ -> assert false)
    ~read:(fun reads operands ->
      match (reads, operands) with
      | [ test ], [ x; two ] -> LL.Ternop (Ops.Where, test, x, two)
      | _ -> assert false);
  (* A library call at a narrow precision is typed either way: at storage width its own text rounds
     it ([FLOAT_TO_HALF(sinf(..))]), and wide, local and slot agree on f32. So it collapses. Under
     wide compute no local narrows it, so a materialized node, whose store does, is not the
     oracle. *)
  let x = [| 1.; 1.25; 1.5; 1.75 |] in
  wider_rendering "half_sin" ~policy ~prec:half ~materialized_agrees:false ~keeps_scopes:false
    ~inputs:[ x; Array.map x ~f:(fun v -> -.(Float.round (Float.sin v *. 2048.) /. 2048.)) ]
    ~values:(unary Ops.Sin) ~read:sum

(* The loop's symbol and index, and materialized nodes for a statement's inputs and result. *)
let statement_nodes ~name ?(extent = n) precs =
  let s = Ll_test.sym () in
  let tns =
    List.mapi precs ~f:(fun k prec ->
        let tn = node ~extent prec (Printf.sprintf "%s_n%d" name k) in
        Ll_test.materialize tn;
        tn)
  in
  (s, [| Ll_test.iter s |], tns)

let ( *: ) a b = LL.Binop (Ops.Mul, a, b)

(* A scope read twice: CSE leaves the second read a [Get_local], so the local has to exist. *)
let shared_read ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; f32 ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = (Ll_test.get x_tn i, f32) in
  let shared = scope ~orig_indices:i (bnode (name ^ "_mid")) (x *: x) in
  let id = match shared with LL.Local_scope { id; _ } -> id | _ -> assert false in
  let write operands =
    Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Add, fst operands, snd operands)))
  in
  let raw = write ((x *: x, f32), (x *: x, f32)) in
  let o = Ll_test.optimize_scoped ~name ~raw (write ((shared, f32), (LL.Get_local id, f32))) in
  (o, [ (x_tn, f32_inputs n) ], out)

(* A truth alone is C [int]: the division of two collapsed scopes would divide integers. The kept
   form divides 1 by 0 in float. Off this file's other inputs: the scope's value must decide. *)
let truth_quotient ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; f32 ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = (Ll_test.get x_tn i, f32) in
  let less = LL.Binop (Ops.Cmplt, x, (Ll_test.c 2., f32)) in
  let more = LL.Binop (Ops.Cmplt, (Ll_test.c 2., f32), x) in
  let write a b = Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Div, a, b))) in
  let raw = write (less, f32) (more, f32) in
  let scoped =
    write
      (scope ~orig_indices:i (bnode (name ^ "_mid")) less, f32)
      (scope ~orig_indices:i (bnode (name ^ "_mid2")) more, f32)
  in
  (Ll_test.optimize_scoped ~name ~raw scoped, [ (x_tn, f32_inputs n) ], out)

(* A value whose rendering is wider than its precision: [Recip] is [1.0 / x], a double at single, so
   the local is what rounds it. Kept, this subtracts the single nearest a third from itself. *)
let reciprocal ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; f32; f32 ] in
  let x_tn, y_tn, out = match tns with [ a; b; c ] -> (a, b, c) | _ -> assert false in
  let inverse prec = LL.Unop (Ops.Recip, (Ll_test.get x_tn i, prec)) in
  let write v =
    Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Add, (v, f32), (Ll_test.get y_tn i, f32))))
  in
  let o =
    Ll_test.optimize_scoped ~name
      ~raw:(write (inverse f32))
      (write (scope ~orig_indices:i (bnode (name ^ "_mid")) (inverse bf)))
  in
  let third = Stdlib.Int32.(float_of_bits (bits_of_float (1. /. 3.))) in
  (o, [ (x_tn, Array.create ~len:n 3.); (y_tn, Array.create ~len:n (-.third)) ], out)

(* A draw in a scope wider than the narrow store around it: [mentions_rng_conversion] stops at the
   scope, so collapsed, the statement would render at the store's precision, and draw there. *)
let wide_draw ~name ~scope =
  let s, i, tns = statement_nodes ~name [ half; half ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let u4 = Ops.uint4x32 in
  let block = LL.Binop (Ops.Threefry4x32_light, (LL.Constant_bits 7L, u4), (Ll_test.embed s, u4)) in
  let draw = LL.Unop (Ops.Uint4x32_to_prec_uniform1, (block, u4)) in
  let x = (Ll_test.get x_tn i, half) in
  let write v = Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Add, x, (v, half)))) in
  let o =
    Ll_test.optimize_scoped ~name ~raw:(write draw)
      (write (scope ~orig_indices:i (fnode (name ^ "_mid")) draw))
  in
  (o, [ (x_tn, bf_inputs n) ], out)

(* A scope in a [Where] condition: the condition has a precision of its own, so the walk does not
   follow it, and the scope stays whatever the arms do. *)
let where_condition ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; f32 ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = (Ll_test.get x_tn i, f32) in
  let write cond =
    Ll_test.loop_n s n
      (Ll_test.set out i (LL.Ternop (Ops.Where, (cond, f32), x, (Ll_test.c 0.5, f32))))
  in
  let o =
    Ll_test.optimize_scoped ~name
      ~raw:(write (x *: x))
      (write (scope ~orig_indices:i (bnode (name ^ "_mid")) (x *: x)))
  in
  (o, [ (x_tn, f32_inputs n) ], out)

(* A cast read by an update, where the accumulator resides wider than the arithmetic: the update can
   localize at the residency, which would read the cast unrounded. Run under the policy that widens
   the bfloat16 accumulator and computes bfloat16 narrow. The inputs cancel, so the sum is 0 when
   each rounds to its neighbour on the grid, and the sum of what rounding drops otherwise: without
   the cancellation that remainder scales with the total, and so does the final rounding. *)
let cast_in_update ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; bf ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let at0 = [| Ll_test.fixed 0 |] in
  let x = Ll_test.get x_tn i in
  let write v =
    Ll_test.loop_n s n
      (Ll_test.set out at0 (LL.Binop (Ops.Add, (Ll_test.get out at0, bf), (v, bf))))
  in
  let o =
    Ll_test.optimize_scoped ~name ~raw:(write x)
      (write (scope ~orig_indices:i (bnode (name ^ "_mid")) x))
  in
  let cancelling = Array.init n ~f:(fun k -> if k % 2 = 0 then 1.0029296875 else -1.) in
  (o, [ (x_tn, cancelling); (out, Array.create ~len:n 0.) ], out)

(* A collapse that would make an update: not one with the scope around it, so the store narrows
   every iteration, and with increments a quarter of the ulp at 1 the value never moves. Collapsed,
   it would localize at the residency and narrow once, and they would add up. On cc's default too,
   whose accumulator is its compute precision. *)
let exposed_update ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f32; bf ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let at0 = [| Ll_test.fixed 0 |] in
  let sum = LL.Binop (Ops.Add, (Ll_test.get out at0, bf), (Ll_test.get x_tn i, bf)) in
  let write v = Ll_test.loop_n s n (Ll_test.set out at0 v) in
  let o =
    Ll_test.optimize_scoped ~name ~raw:(write sum)
      (write (scope ~orig_indices:i (bnode (name ^ "_mid")) sum))
  in
  (o, [ (x_tn, Array.create ~len:n 0.001953125); (out, Array.create ~len:n 1.) ], out)

(* A scope at the root of a store. Kept, a double is converted to the scope's single and then to
   fp8; collapsed, [store_precs] would see a copy and convert it in one step, and [double_to_fp8]
   does not round twice: 1.125 + 2^-40 is an f32 tie, broken to 1, and above the fp8 tie, so
   1.25. *)
let double_at_store_root ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f64; fp8 ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = Ll_test.get x_tn i in
  let write v = Ll_test.loop_n s n (Ll_test.set out i v) in
  let o =
    Ll_test.optimize_scoped ~name ~raw:(write x)
      (write (scope ~orig_indices:i (fnode (name ^ "_mid")) x))
  in
  (o, [ (x_tn, Array.create ~len:n (1.125 +. Float.ldexp 1. (-40))) ], out)

(* The same at a cast's own precision: the three precisions are the store's on every backend, so the
   scope goes, and the kept form has to give the copy's one step (gh-ocannl-1956). *)
let cast_at_store_root ~name ~scope =
  let s, i, tns = statement_nodes ~name [ f64; fp8 ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = Ll_test.get x_tn i in
  let write v = Ll_test.loop_n s n (Ll_test.set out i v) in
  let o =
    Ll_test.optimize_scoped ~name ~raw:(write x)
      (write (scope ~orig_indices:i (node fp8 (name ^ "_mid")) x))
  in
  (o, [ (x_tn, Array.create ~len:n (1.125 +. Float.ldexp 1. (-40))) ], out)

(* Arithmetic at the same root renders at [comp_prec] either way, and the scope goes. *)
let product_at_store_root ~name ~scope =
  let s, i, tns = statement_nodes ~name [ bf; bf ] in
  let x_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
  let x = (Ll_test.get x_tn i, bf) in
  let write v = Ll_test.loop_n s n (Ll_test.set out i v) in
  let o =
    Ll_test.optimize_scoped ~name
      ~raw:(write (x *: x))
      (write (scope ~orig_indices:i (fnode (name ^ "_mid")) (x *: x)))
  in
  (o, [ (x_tn, bf_inputs n) ], out)

(* A scope inside a collapsing scope's value. *)
let nested ~name ~scope =
  let s, i, tns = statement_nodes ~name [ bf; f32; f32 ] in
  let y_tn, x_tn, out = match tns with [ a; b; c ] -> (a, b, c) | _ -> assert false in
  let x = (Ll_test.get x_tn i, f32) and y = (Ll_test.get y_tn i, bf) in
  let write v = Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Add, x, (v, f32)))) in
  let inner = scope ~orig_indices:i (bnode (name ^ "_inner")) (y *: y) in
  let o =
    Ll_test.optimize_scoped ~name
      ~raw:(write ((y *: y, bf) *: y))
      (write (scope ~orig_indices:i (bnode (name ^ "_mid")) ((inner, bf) *: y)))
  in
  (o, [ (y_tn, bf_inputs n); (x_tn, f32_inputs n) ], out)

(* A reduction over a kept scope, retyped [Vectorized]. That annotation licenses reassociating the
   sum, which a collapse lets it act on, so the data is dyadic: every order gives one value. *)
let reduction ~policy =
  let run_it tag ~ty scope =
    let name = routine_name "reduce" tag policy in
    let s, i, tns = statement_nodes ~name ~extent:long [ bf; f32 ] in
    let y_tn, out = match tns with [ a; b ] -> (a, b) | _ -> assert false in
    let at0 = [| Ll_test.fixed 0 |] in
    let y = (Ll_test.get y_tn i, bf) in
    let write v =
      Ll_test.loop_n s long
        (Ll_test.set out at0 (LL.Binop (Ops.Add, (Ll_test.get out at0, f32), (v, f32))))
    in
    let mid = node ~extent:long bf (name ^ "_mid") in
    let o =
      Ll_test.optimize_scoped ~name
        ~raw:(write (y *: y))
        (write (scope ~orig_indices:i mid (y *: y)))
    in
    let o = Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty } ] o in
    let seed = [ (y_tn, bf_inputs long); (out, Array.create ~len:long 0.) ] in
    (name, (List.hd_exn (Ll_test.execute ~name o ~seed ~read:[ out ])).(0))
  in
  let _, kept = run_it "keep" ~ty:LL.Serial kept_scope in
  let name, reduced = run_it "vec" ~ty:LL.Vectorized plain_scope in
  pf "reduce: the retyped form computes the oracle's value" Float.(kept = reduced);
  let collapsed = collapses narrow_product in
  pf "reduce: the kernel holds the local unless the compute precision makes it exact"
    (Bool.equal (holds_local name) (not collapsed));
  in_leg ~leg:cc_leg "reduce: a collapsed scope renders the SIMD reduction" (fun label ->
      p label
        (Bool.equal
           (String.is_substring (Generated.read name) ~substring:"Vectorized reduction rendering")
           collapsed))

(* What a [Round_prec] rounds to is an annotation (gh-ocannl-1975), so it has to count wherever a
   read's precision does: as the source of a cast of a cast, where a simplifier revisiting the outer
   scope sees the inner one as an expression, and in the code's identity. *)
let () =
  let x = fnode "_rounded_x" and out = fnode "_rounded_out" and at0 = [| Ll_test.fixed 0 |] in
  let rounded t = LL.Unop (Ops.Round_prec, (Ll_test.get x at0, t)) in
  p "a rounding reads as what it rounded to, so the cast around it still rounds"
    (LL.narrowing_read ~scope_prec:bf (rounded half));
  let digest t = LL.alpha_digest (Ll_test.set out at0 (rounded t)) in
  p "roundings to two precisions are two programs" (not (String.equal (digest half) (digest bf)));
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let update =
    LL.Binop
      (Ops.Add, (LL.Unop (Ops.Round_prec, (Ll_test.get out i, half)), f32), (Ll_test.get x i, f32))
  in
  p "zero forwarding leaves a rounded self-read alone: a rounding holds a read, never a select"
    (Option.is_none (LL.zero_seed_candidate out (Ll_test.loop_n s n (Ll_test.set out i update))))

let () =
  let saved = Ir.Numerics.get () in
  List.iter [ true; false ] ~f:(fun narrow_compute_f32 ->
      Ir.Numerics.set_policy { saved with Ir.Numerics.narrow_compute_f32 };
      let policy = if narrow_compute_f32 then "nc_on" else "nc_off" in
      Verdict.case
        (if narrow_compute_f32 then "narrow floats computed in single"
         else "narrow floats computed at storage width")
        (fun () ->
          List.iter [ narrow_product; wide_product ] ~f:(fun shape ->
              let pre, pre_scopes, _ = run shape ~form:`Oracle ~policy in
              let sch, sch_scopes, _ = run shape ~form:`Scheduled ~policy in
              let raw, _, _ = run shape ~form:`Unscoped ~policy in
              let inl, inl_scopes, inlined = run shape ~form:`Inlined ~policy in
              (* The rounding is observable wherever the scope's local is declared at the scope's
                 own precision, which every backend does under [narrow_compute_f32=false]; cc's
                 default declares the local single, so there the two may agree. *)
              if not narrow_compute_f32 then
                p
                  (Printf.sprintf
                     "%s: the oracle's rounding is observable -- the value written straight into \
                      the sum computes different values"
                     shape.label)
                  (not (same_bits pre raw));
              p
                (Printf.sprintf
                   "%s: the uncollapsed oracle runs its scope, and the schedule's simplify keeps it"
                   shape.label)
                (pre_scopes = 1 && sch_scopes = 1);
              p
                (Printf.sprintf "%s: the scheduled form computes the oracle's values bitwise"
                   shape.label)
                (same_bits pre sch);
              p
                (Printf.sprintf
                   "%s: optimize inlines the virtual node and keeps its scope, computing the \
                    oracle's values bitwise"
                   shape.label)
                (inlined && inl_scopes = 1 && same_bits pre inl);
              let show vs =
                String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h")))
              in
              Stdio.eprintf "(not part of the golden) %s %s: oracle %s; unscoped %s\n" policy
                shape.label (show pre) (show raw));
          List.iter [ narrow_load; widening_load ] ~f:(fun shape ->
              let pre, _, _ = run shape ~form:`Oracle ~policy in
              let sch, sch_scopes, _ = run shape ~form:`Scheduled ~policy in
              let inl, inl_scopes, inlined = run shape ~form:`Inlined ~policy in
              p
                (Printf.sprintf "%s: an exact scope over a load is still collapsed" shape.label)
                (sch_scopes = 0);
              p
                (Printf.sprintf "%s: the collapsed form computes the oracle's values bitwise"
                   shape.label)
                (same_bits pre sch);
              p
                (Printf.sprintf
                   "%s: optimize inlines the virtual node and collapses its scope, computing the \
                    oracle's values bitwise"
                   shape.label)
                (inlined && inl_scopes = 0 && same_bits pre inl));
          List.iter [ cast_into_own_prec; cast_into_wider ] ~f:(fun shape ->
              let pre, _, _ = run shape ~form:`Oracle ~policy in
              let sch, sch_scopes, _ = run shape ~form:`Scheduled ~policy in
              let inl, inl_scopes, inlined = run shape ~form:`Inlined ~policy in
              let mat, _, _ = run shape ~form:`Materialized ~policy in
              let raw, _, _ = run shape ~form:`Unscoped ~policy in
              p
                (Printf.sprintf "%s: a cast leaves no scope, scheduled or inlined" shape.label)
                (sch_scopes = 0 && inlined && inl_scopes = 0);
              p
                (Printf.sprintf
                   "%s: every form of the cast computes the materialized values bitwise" shape.label)
                (same_bits mat pre && same_bits mat sch && same_bits mat inl);
              let show vs =
                String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h")))
              in
              Stdio.eprintf "(not part of the golden) %s %s: materialized %s; unscoped %s\n" policy
                shape.label (show mat) (show raw));
          accumulator_keeps_residency ~policy;
          List.iter
            [
              narrow_product;
              wide_product;
              fp8_product;
              cast_into_own_prec;
              cast_into_own_half;
              cast_into_wider;
            ]
            ~f:(render_time_collapse ~policy);
          reduction ~policy;
          rendered_wider ~policy;
          must_agree "shared" ~policy shared_read;
          must_agree "truth" ~policy truth_quotient;
          must_agree "recip" ~policy reciprocal;
          must_agree "exposed_update" ~policy exposed_update;
          must_agree "draw" ~policy wide_draw;
          must_agree "where" ~policy where_condition;
          must_agree "nested" ~policy ~collapses:(collapses narrow_product) nested;
          must_agree "root_product" ~policy ~collapses:(collapses wide_product)
            product_at_store_root;
          let f64_leg =
            Verdict.leg ~when_:(Lazy.force capabilities).Ir.Backend_intf.supports_f64
              ~on:backend_name ()
          in
          must_agree ~leg:f64_leg "root_double" ~policy double_at_store_root;
          must_agree ~leg:f64_leg "root_cast" ~policy ~collapses:true cast_at_store_root));
  Ir.Numerics.set_policy
    { saved with Ir.Numerics.narrow_compute_f32 = false; bf16_arithmetic = Bf16_wide };
  Verdict.case "a bfloat16 accumulator wider than the arithmetic" (fun () ->
      must_agree "update" ~policy:"acc_wide" cast_in_update);
  Ir.Numerics.set_policy saved
