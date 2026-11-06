(* gh-ocannl-1866: a mixed-precision cast twin rounds its master at the twin's precision, whichever
   placement it gets, on every backend and under either [narrow_compute_f32].

   Placement is the compiler's choice, so it must not change what a read of the twin gives. (Other
   nodes are another matter: gh-ocannl-1965's narrow_placement_contract.) A materialized twin rounds
   at its store. A virtual one is inlined as a scope over a load of the master, and two things used
   to skip its rounding wherever compute is wider than the twin (cc's default for f16 and bf16,
   every backend for fp8). A scope read in an equally narrow slot collapsed into the slot
   (gh-ocannl-1788's rule held that collapse exact), and the slot then renders at the wider compute
   precision; and a scope kept in a wider slot declared its local at that wider precision. Both are
   exercised here: [slot]'s two cases are the two arms of that sentence, each with a value that only
   the rounding decides.

   Every expectation is exact by construction, so the claims are equalities and their digits go to
   stderr. The master holds [1 + d] with [d] just under half the twin's ulp at 1, so a rounded twin
   reads 1. In a slot at the twin's own precision a single factor cannot show the difference, since
   the store rounds the product anyway; so the consumer multiplies by 3, which carries the unrounded
   offset past half an ulp of [3]: rounded [3], unrounded [3 + 2 ulp(1)].

   A wider master has one rounding more to get wrong (gh-ocannl-1956): an f64 master can sit just
   off an f32 tie, where a virtual twin's read used to go through f32 on its way to the twin's
   precision wherever compute is f32 (cc's default) while a materialized one converts as its store
   does. {!f64_master} holds that value. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode
module IDX = Train.IDX
module Ops = Ir.Ops
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Test_utils.Generated.init ~backend_name

let placements =
  [ ("materialized", MP.Twin_materialized); ("virtual", MP.Twin_virtual); ("auto", MP.Twin_auto) ]

(* Half of [prec]'s ulp at 1, less a margin clear of a tie. *)
let under_half_ulp prec = (2. **. Float.of_int (-Ops.significand_bits prec)) -. (2. **. -20.)

let twin_of ~placement ?master_prec ~prec ?input_dims values =
  MP.with_master_weights ~placement ?master_prec ~prec (fun () ->
      TDSL.param ~values ?input_dims ~output_dims:[ 1 ] "w" ())

(* The placement the compile resolved, from the context that ran it: {!Tn.Placements.get} falls back
   to declared intent, so the forced placements are read as decisions the compile honoured, and
   [Twin_auto] only as decided one way or the other: which way is the heuristics' call, and no value
   below depends on it. *)
let placed ~label ctx twin placement =
  let plc = Context.placements ctx and tn = twin.Tensor.value in
  Stdio.eprintf "%s: twin placed %s (not part of the golden)\n" label
    (Tn.debug_memory_mode (Tn.Placements.get plc tn));
  match placement with
  | MP.Twin_virtual -> Tn.Placements.known_virtual plc tn
  | Twin_materialized -> Tn.Placements.known_non_virtual plc tn
  | Twin_auto -> Tn.Placements.known_virtual plc tn || Tn.Placements.known_non_virtual plc tn

(* One placement's run: whether it was placed as asked, and the value. [run ~label t] answers the
   value and the context that computed it. *)
let run_placed ~label ~twin run placement =
  Tensor.unsafe_reinitialize ();
  let t = twin ~placement in
  let v, ctx = run ~label t in
  Stdio.eprintf "%s: y = %h (not part of the golden)\n" label v;
  (placed ~label ctx t placement, v)

(* Every placement's run where [when_] holds. Ahead of the claims, not inside a leg's: a [run] can
   make claims of its own, and a leg's evaluation makes exactly one. *)
let run_placements ~when_ name ~twin run =
  if not when_ then fun _ -> (false, Float.nan)
  else
    let runs =
      List.map placements ~f:(fun (where, placement) ->
          (where, run_placed ~label:(name ^ ", " ^ where) ~twin run placement))
    in
    List.Assoc.find_exn runs ~equal:String.equal

let case ?(when_ = true) name ~expected ~twin run =
  Stdio.printf "%s: every placement gives %h\n" name expected;
  let leg = Verdict.leg ~when_ ~on:backend_name () in
  let outcome = run_placements ~when_ name ~twin run in
  List.iter placements ~f:(fun (where, _) ->
      let label = name ^ ", " ^ where in
      in_legf ~leg "%s: placed as asked" label (fun claim -> p claim (fst (outcome where)));
      in_legf ~leg "%s: the twin's rounding reaches the value" label (fun claim ->
          p claim Float.(snd (outcome where) = expected)))

(* [case] where the value is the backend's to choose: every placement reads what the materialized
   twin does. *)
let agrees_with_materialized ~when_ name ~twin run =
  let leg = Verdict.leg ~when_ ~on:backend_name () in
  let outcome = run_placements ~when_ name ~twin run in
  List.iter placements ~f:(fun (where, _) ->
      let label = name ^ ", " ^ where in
      in_legf ~leg "%s: placed as asked" label (fun claim -> p claim (fst (outcome where)));
      in_legf ~leg "%s: reads what the materialized twin reads" label (fun claim ->
          p claim Float.(snd (outcome where) = snd (outcome "materialized"))))

let forward ?bindings ?launch_values y ~out_prec =
  Tn.update_prec y.Tensor.value out_prec;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once ?bindings ?launch_values (Context.auto ()) y in
  ((Context.get_values ctx y.Tensor.value).(0), ctx)

(* The two arms of the header's sentence, at one twin precision. *)
let slot prec =
  let name = Ops.prec_string prec in
  let master = 1. +. under_half_ulp prec in
  let twin ~placement = twin_of ~placement ~prec [| master |] in
  case (name ^ " twin read in an f32 slot") ~expected:1. ~twin (fun ~label:_ t ->
      let%op y = t *. 1.0 in
      forward y ~out_prec:Ops.single);
  case (name ^ " twin read in a slot of its own precision") ~expected:3. ~twin (fun ~label:_ t ->
      let%op y = t *. 3.0 in
      forward y ~out_prec:prec)

(* gh-ocannl-1956: an f64 master just above the tie between 1 and the next value at [prec], and
   close enough to it that f32 rounds onto the tie. One step rounds up; through f32, the tie goes to
   even, 1. Placement must not choose between them. fp8 is one step on every backend with an f64
   (gh-ocannl-648), so its value is claimed: 1.25, and 3.75 at the store, a tie that goes to 4 where
   the double rounding's 1 gives 3. A conversion to half or bfloat16 is the backend's own (cc and
   HIP go through f32, CUDA converts in one step), so those claim the parity. Metal has no f64. *)
let f64_master prec =
  let name = Ops.prec_string prec ^ " twin of an f64 master" in
  let when_ = (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.supports_f64 in
  let master = 1. +. (2. **. Float.of_int (-Ops.significand_bits prec)) +. (2. **. -40.) in
  let twin ~placement = twin_of ~placement ~master_prec:Ops.double ~prec [| master |] in
  let in_single ~label:_ t =
    let%op y = t *. 1.0 in
    forward y ~out_prec:Ops.single
  in
  let in_own_prec ~label:_ t =
    let%op y = t *. 3.0 in
    forward y ~out_prec:prec
  in
  if Ops.equal_prec prec Ops.fp8 then (
    case ~when_ (name ^ " read in an f32 slot") ~expected:1.25 ~twin in_single;
    case ~when_ (name ^ " read in a slot of its own precision") ~expected:4. ~twin in_own_prec)
  else (
    agrees_with_materialized ~when_ (name ^ " read in an f32 slot") ~twin in_single;
    agrees_with_materialized ~when_
      (name ^ " read in a slot of its own precision")
      ~twin in_own_prec)

(* A reduction over the twin: the accumulation stays wide, and its reads of the twin round. *)
let reduction () =
  let prec = Ops.half in
  let master = 1. +. under_half_ulp prec in
  case "half twin read by a matmul" ~expected:3.
    ~twin:(fun ~placement ->
      twin_of ~placement ~prec ~input_dims:[ 3 ] [| master; master; master |])
    (fun ~label:_ t ->
      let x = NTDSL.init ~l:"x" ~prec:Ops.single ~o:[ 3 ] ~f:(fun _ -> 1.) () in
      let%op y = t * x in
      forward y ~out_prec:prec)

(* The backward pass reads the twin too: d(twin * x)/dx is the twin. *)
let backward () =
  let prec = Ops.half in
  let master = 1. +. under_half_ulp prec in
  case "half twin read by the backward pass" ~expected:1.
    ~twin:(fun ~placement -> twin_of ~placement ~prec [| master |])
    (fun ~label:_ t ->
      let x = TDSL.param ~value:2. ~output_dims:[ 1 ] "x" () in
      let%op y = t *. x in
      Tn.update_prec y.Tensor.value Ops.single;
      let ctx = Train.update_once (Context.auto ()) y in
      ((Context.get_values ctx (Option.value_exn x.Tensor.diff).grad).(0), ctx))

(* Two reads of the twin, and a routine run again after the master moved: each run reads the
   master's current value, rounded, whatever the scopes were merged into. *)
let refresh () =
  let prec = Ops.half in
  let d = under_half_ulp prec in
  case "half twin read twice, master refreshed" ~expected:3.
    ~twin:(fun ~placement -> twin_of ~placement ~prec [| 1. +. d |])
    (fun ~label t ->
      let%op y = (t *. 1.0) + (t *. 1.0) in
      Tn.update_prec y.Tensor.value Ops.single;
      Train.set_materialized y.Tensor.value;
      let master = Set.choose_exn y.Tensor.params in
      let ctx = Train.init_params (Context.auto ()) IDX.empty y in
      let ctx, routine = Train.to_routine ctx IDX.empty (Train.forward y) in
      let ctx = Context.run ctx routine in
      let first = (Context.get_values ctx y.Tensor.value).(0) in
      Stdio.eprintf "%s: before the refresh y = %h (not part of the golden)\n" label first;
      pf "%s: the first run rounds" label Float.(first = 2.);
      let ctx = Context.set_values ctx master.Tensor.value [| 1.5 +. d |] in
      let ctx = Context.run ctx routine in
      ((Context.get_values ctx y.Tensor.value).(0), ctx))

(* The rule is the read, not the operation that made it: a narrow slice of wider data rounds as a
   cast does. That is what [Precision_policy] makes of a model's input batch, a virtual node stored
   narrow over f32 data, so a model's placement-parity reaches it too. The launch reads cell 1; the
   other cell holds a value that would show if the wrong one were read. *)
let slice () =
  let prec = Ops.half in
  let master = 1. +. under_half_ulp prec in
  let batch_n, bindings = IDX.get_static_symbol IDX.empty in
  case "half slice of f32 data" ~expected:3.
    ~twin:(fun ~placement ->
      let xs =
        NTDSL.init ~l:"xs" ~prec:Ops.single ~b:[ 2 ] ~o:[ 1 ]
          ~f:(fun idcs -> if idcs.(0) = 1 then master else 7.)
          ()
      in
      let%op batch = xs @| batch_n in
      Tn.update_prec batch.Tensor.value prec;
      (match placement with
      | MP.Twin_auto -> ()
      | Twin_virtual -> Train.set_virtual batch.Tensor.value
      | Twin_materialized -> Train.set_materialized batch.Tensor.value);
      batch)
    (fun ~label:_ t ->
      let%op y = t *. 3.0 in
      forward ~bindings ~launch_values:[ (batch_n, 1) ] y ~out_prec:prec)

(* What the virtual path emits: no buffer for the twin, which is what a materialized twin's accesses
   show the text would otherwise hold, and on cc the rounding is the conversion of the master into a
   local of the twin's precision. *)
let emission () =
  let module G = Test_utils.Generated in
  let prec = Ops.half in
  let ident = G.node_ident ~facts:(Context.backend_facts_of_name backend_name) in
  let routine = "emitted_fwd" in
  let compile placement =
    Tensor.unsafe_reinitialize ();
    let t = twin_of ~placement ~prec [| 1. +. under_half_ulp prec |] in
    let%op emitted = t *. 1.0 in
    G.arm routine;
    ignore (forward emitted ~out_prec:Ops.single : float * Context.t)
  in
  let twin_access = ident "cast" ^ "[" in
  compile MP.Twin_materialized;
  G.assert_emits ~routine ~contains:twin_access "a materialized twin reads its buffer";
  compile MP.Twin_virtual;
  G.assert_omits ~routine ~contains:twin_access "the virtual twin reads no buffer of its own";
  gated ~when_:(String.equal backend_name "cc") ~on:"cc"
    "the virtual twin narrows its master into a local at its own precision"
    (String.is_substring (G.read routine) ~substring:("= FLOAT_TO_HALF(" ^ ident "w" ^ "[0])"))

let () =
  let saved = Ir.Numerics.get () in
  List.iter [ true; false ] ~f:(fun narrow_compute_f32 ->
      Ir.Numerics.set_policy { saved with Ir.Numerics.narrow_compute_f32 };
      Verdict.case
        (if narrow_compute_f32 then "narrow floats computed in single"
         else "narrow floats computed at storage width")
        (fun () ->
          List.iter [ Ops.half; Ops.bfloat16; Ops.fp8 ] ~f:(fun prec ->
              slot prec;
              f64_master prec)));
  Ir.Numerics.set_policy saved;
  reduction ();
  backward ();
  refresh ();
  slice ();
  emission ()
