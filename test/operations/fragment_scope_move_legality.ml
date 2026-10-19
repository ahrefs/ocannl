(* gh-ocannl-1342: a move applied on top of a tensorized schedule whose accumulator is a resident
   simdgroup fragment is refused by the op-legality oracle when it leaves a [Tile_mma] into that
   fragment outside every scope the renderer recognizes, and admitted when it does not.

   The scope: [Tensorize]'s accumulator contraction brackets the contracted loop chain with a lane-0
   load of the fragment and a lane-0 store-back, and marks the fragment. [C_syntax] maps a bracket
   holding exactly one [Tile_mma] to persistent intrinsic fragments when it meets it at the head of
   a statement sequence ([Schedule.recognize_fragment_scope], shared by both sides). A [Tile_mma]
   outside such a scope still compiles: it declines to the lane-0 scalar fallback. #1229's CUDA
   probe on gpt2_mini timed 17 such round-1 candidates at 4.1-7.5 ms against the 2.21 ms head, all a
   materializing [Unroll]: of the attention out-projections' [Base] contraction axis, and of the
   logits tile's [Split_outer] contraction block.

   The fixtures, on two tensorized half-precision bases (the half operands are what every GPU
   backend tensorizes, so the compile legs below run on Metal, CUDA and HIP alike):

   - A, a 64x64x64 matmul whose contraction is split [k_o x k_i] with the scope around [k_o], inside
   a serial row loop [i_m]. [Unroll k_o ~materialize:true] (the [Split_outer] shape) replicates the
   scope's [Tile_mma] four times; [Unroll i_m ~materialize:true] copies the whole scope, and the
   second copy trails the first in one statement sequence; a [Vectorized] retype of the transfers'
   column copy loop (a GPU menu move: innermost, serial, non-accumulating) leaves no recognizable
   load or store-back. Admitted on the same schedule: both in-place unrolls, and a split of [k_o]. -
   B, [ihd;hdj=>ij] with the scope around the [Base] contraction axis [h] (the out-projection's
   shape). [Unroll h ~materialize:true] replicates the [Tile_mma]; its in-place twin is admitted.

   The structural legs run on every backend: each fixture's applied IR has exactly its expected
   violation (or none), the fixtures exercise every reason [fragment_scope_violation] has (derived
   from the type), and [op_legality] (the beam menu's gate) and [schedule_legality] (seeding's walk)
   refuse each refused fixture with the message of the violation it adds, and admit the rest.

   The compile legs relate the oracle to the renderer on a GPU backend that tensorizes the half
   triple: every fixture, compiled anyway as a candidate, renders exactly as many lane-0 scalar
   fallbacks as the oracle finds violations in the IR it was compiled from, and at least one
   tensorized statement besides where it finds none — and computes the serial twin's values bitwise
   (small integer operands: every partial sum is exact in half). On the C backends a scope and a
   bare [Tile_mma] render alike, so there is nothing to relate there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let hermetic (opt : LL.optimized) : LL.optimized =
  {
    opt with
    LL.traced_store = Hashtbl.copy opt.LL.traced_store;
    optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
  }

let n = 64
let bm = 16
let simd_width = 32

(* The zeroing nest's geometry, shared by both bases: rows Grid(2) x Serial(32), aligned with the
   accumulation's row blocks; the column loop Serial(2) x Workgroup(32), the lane width (one
   workgroup extent per kernel). *)
let zero_ops ~out =
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:(2 * bm) ~outer:LL.Grid ~inner:LL.Serial in
  let rz, _, _ = Sched.split ~axis:zj ~factor:simd_width ~outer:LL.Serial ~inner:LL.Workgroup in
  [ ez; sp_zi; rz ]

type syms = {
  outer_red : Ir.Indexing.symbol;  (** The contracted chain's outermost loop: the scope's body. *)
  around : Ir.Indexing.symbol option;  (** A serial loop around the scope (base A). *)
}

(* Base A: [i -> Grid(2) x i_m(2) x i_i(16)], [k -> k_o(4) x k_i(16)], [k_o] swapped above [i_i] and
   [j]; the contraction lands around [k_o] (the outermost serial loop not indexing the accumulator),
   inside [i_m]. *)
let matmul_schedule ~out (opt : LL.optimized) : Sched.schedule * syms =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let sp_i, _, i_r = Sched.split ~axis:i ~factor:(2 * bm) ~outer:LL.Grid ~inner:LL.Serial in
  let sp_r, i_m, i_i = Sched.split ~axis:i_r ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
  ( zero_ops ~out
    @ [
        sp_i;
        sp_r;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        tz;
      ],
    { outer_red = k_o; around = Some i_m } )

(* Base B: [ihd;hdj=>ij] lowers to [i { j { h { d } } }]; [h] is swapped above [i_i] and [j], and
   the contraction lands around it. *)
let two_axis_schedule ~out (opt : LL.optimized) : Sched.schedule * syms =
  let i, j, h, d =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 4) with
    | [ i; j; h; d ] -> (i, j, h, d)
    | _ -> assert false
  in
  let sp_i, _, i_r = Sched.split ~axis:i ~factor:(2 * bm) ~outer:LL.Grid ~inner:LL.Serial in
  let sp_r, _, i_i = Sched.split ~axis:i_r ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:d ~simd_width () in
  ( zero_ops ~out
    @ [ sp_i; sp_r; Sched.Swap { outer = j; inner = h }; Sched.Swap { outer = i_i; inner = h }; tz ],
    { outer_red = h; around = None } )

type expect = Replicated | Not_leading | Unbracketed | Admitted

(* A move names loops of the base's schedule, or loops its application mints (the transfers' copy
   loops, fresh at each application): the latter is no op of a chain built before application, so
   seeding cannot propose it, while the beam's menu (built from the applied IR) can. *)
type move = Of_schedule of (syms -> Sched.optop) | Of_applied of (LL.optimized -> Sched.optop)

let move_op move s applied = match move with Of_schedule f -> f s | Of_applied f -> f applied

type base = {
  b_name : string;
  b_tensor : unit -> Tensor.t;  (** A fresh twin of the base's computation. *)
  b_schedule : out:Tn.t -> LL.optimized -> Sched.schedule * syms;
  b_moves : (string * expect * move) list;
}

let split2 sym =
  let op, _, _ = Sched.split ~axis:sym ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
  op

let unroll ~materialize sym = Sched.Unroll { axis = sym; materialize }
let around s = Option.value_exn s.around

(* The fragment load's column copy loop in [applied]: the innermost loop of the lane-0 transfer
   nests, shared by the load and the store-back ([contract_tensorized_accumulator] mints one pair of
   copy symbols for both). *)
let copy_column (applied : LL.optimized) =
  let fragments = applied.LL.simdgroup_fragments in
  let rec find (c : LL.t) =
    match c with
    | LL.Seq (c1, c2) -> (
        match Sched.recognize_fragment_scope ~fragments c with
        | Some fs -> Some (List.hd_exn fs.Sched.fs_load.Sched.ft_syms)
        | None -> Option.first_some (find c1) (find c2))
    | LL.For_loop { body; _ } | LL.If { body; _ } -> find body
    | _ -> None
  in
  Option.value_exn (find applied.LL.llc)

let violation_matches expect (vs : Sched.fragment_scope_violation list) =
  match expect with
  | Admitted -> List.is_empty vs
  | Replicated ->
      (not (List.is_empty vs))
      && List.for_all vs ~f:(function Sched.Tile_replicated _ -> true | _ -> false)
  | Not_leading ->
      (not (List.is_empty vs))
      && List.for_all vs ~f:(function Sched.Scope_not_leading _ -> true | _ -> false)
  | Unbracketed ->
      (not (List.is_empty vs))
      && List.for_all vs ~f:(function Sched.Tile_unbracketed _ -> true | _ -> false)

(* An op verdict states the violations the op adds: refused with the first one's message when the
   fixture is expected refused, otherwise admitted (proven or "compile and see"). *)
let verdict_states expect (vs : Sched.fragment_scope_violation list) (v : Sched.op_verdict) =
  match (expect, vs, v) with
  | Admitted, _, (Sched.Op_legal | Sched.Op_unknown _) -> true
  | (Replicated | Not_leading | Unbracketed), first :: _, Sched.Op_illegal msg ->
      String.equal msg (Sched.fragment_scope_violation_message first)
  | _ -> false

let reasons =
  List.sort ~compare:String.compare
    (List.map Sched.Variants_of_fragment_scope_violation.descriptions ~f:fst)

let () =
  let operand ~l ~dims ~modulus ~offset =
    NTDSL.init ~l ~prec:Ir.Ops.half ~o:(Array.to_list dims)
      ~f:(Ll_test.cycle ~dims ~modulus ~offset ~stride:1.)
      ()
  in
  let ma = operand ~l:"ma" ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) in
  let mb = operand ~l:"mb" ~dims:[| n; n |] ~modulus:3 ~offset:(-1.) in
  let x = operand ~l:"x" ~dims:[| n; 2; 32 |] ~modulus:5 ~offset:(-2.) in
  let w = operand ~l:"w" ~dims:[| 2; 32; n |] ~modulus:3 ~offset:(-1.) in
  let half (t : Tensor.t) =
    Tn.update_prec t.Tensor.value Ir.Ops.half;
    t
  in
  let bases =
    [
      {
        b_name = "A";
        b_tensor = (fun () -> half [%op ma +* "ik;kj=>ij" mb]);
        b_schedule = matmul_schedule;
        b_moves =
          [
            ( "Unroll k_o ~materialize:true (the contracted loop)",
              Replicated,
              Of_schedule (fun s -> unroll ~materialize:true s.outer_red) );
            ( "Unroll i_m ~materialize:true (a loop around the scope)",
              Not_leading,
              Of_schedule (fun s -> unroll ~materialize:true (around s)) );
            ( "Vectorized retype of the transfers' column copy loop",
              Unbracketed,
              Of_applied
                (fun applied -> Sched.Retype { axis = copy_column applied; ty = LL.Vectorized }) );
            ( "Unroll k_o in place",
              Admitted,
              Of_schedule (fun s -> unroll ~materialize:false s.outer_red) );
            ( "Unroll i_m in place",
              Admitted,
              Of_schedule (fun s -> unroll ~materialize:false (around s)) );
            ("Split k_o by 2", Admitted, Of_schedule (fun s -> split2 s.outer_red));
          ];
      };
      {
        b_name = "B";
        b_tensor = (fun () -> half [%op x +* "ihd;hdj=>ij" w]);
        b_schedule = two_axis_schedule;
        b_moves =
          [
            ( "Unroll h ~materialize:true (a Base contraction axis)",
              Replicated,
              Of_schedule (fun s -> unroll ~materialize:true s.outer_red) );
            ( "Unroll h in place",
              Admitted,
              Of_schedule (fun s -> unroll ~materialize:false s.outer_red) );
          ];
      };
    ]
  in
  let compile ?(transform = fun o -> o) name tensor =
    Test_utils.with_fp16_auto (fun () ->
        Context.compile_outcome
          ~lowered_transform:(fun o -> [ transform o ])
          ~provenance:SO.Candidate ~candidate:name (Context.auto ())
          (named name (Train.forward tensor))
          Ir.Indexing.Empty)
  in
  let values tensor = function
    | Ok (ctx, routine) ->
        let ctx = Context.run ctx routine in
        Ok (Context.get_values ctx tensor.Tensor.value, routine.Context.mma)
    | Error (SO.Classified c) -> Error (Sexp.to_string_hum (SO.sexp_of_classified_cause c))
    | Error (SO.Fatal { exn; _ }) -> Error (Exn.to_string exn)
  in
  (* Per base: the serial reference values, and (via the transform) the lowered code the structural
     legs schedule — captured unscheduled, hermetically, so each leg applies its own copy. *)
  let prepared =
    List.map bases ~f:(fun base ->
        let t = base.b_tensor () in
        let lowered = ref None in
        let reference =
          values t
            (compile ("fsm_serial_" ^ base.b_name) t ~transform:(fun opt ->
                 lowered := Some (hermetic opt);
                 opt))
        in
        (base, t, Option.value_exn !lowered, fst (Result.ok_or_failwith reference)))
  in
  p_all "each serial reference is not all zeros" prepared ~f:(fun (_, _, _, got) ->
      Array.exists got ~f:(fun v -> Float.(v <> 0.)));
  let applied_base (base, t, lowered, _) =
    let schedule, s = base.b_schedule ~out:t.Tensor.value (hermetic lowered) in
    (schedule, s, Sched.apply schedule (hermetic lowered))
  in
  p_all "each base marks a simdgroup fragment and has no fragment-scope violation" prepared
    ~f:(fun prep ->
      let _, _, applied = applied_base prep in
      (not (Set.is_empty applied.LL.simdgroup_fragments))
      && List.is_empty (Sched.fragment_scope_violations applied));
  let fixtures =
    List.concat_map prepared ~f:(fun ((base, _, _, _) as prep) ->
        let schedule, s, applied = applied_base prep in
        List.map base.b_moves ~f:(fun (label, expect, move) ->
            (prep, label, expect, schedule, applied, move, move_op move s applied)))
  in
  let violations_after applied op =
    Sched.fragment_scope_violations (Sched.apply [ op ] (hermetic applied))
  in
  p_all "each move's applied IR has exactly its expected violations (or none)" fixtures
    ~f:(fun (_, label, expect, _, applied, _, op) ->
      let vs = violations_after applied op in
      Stdio.eprintf "%s: [%s]\n%!" label
        (String.concat ~sep:"; " (List.map vs ~f:Sched.fragment_scope_violation_message));
      violation_matches expect vs);
  p "the fixtures' violations exercise every fragment_scope_violation reason"
    (let exercised =
       List.dedup_and_sort ~compare:String.compare
         (List.concat_map fixtures ~f:(fun (_, _, _, _, applied, _, op) ->
              List.map (violations_after applied op)
                ~f:Sched.Variants_of_fragment_scope_violation.to_name))
     in
     Stdio.eprintf "reasons: [%s]; exercised by the fixtures: [%s]\n%!"
       (String.concat ~sep:"; " reasons)
       (String.concat ~sep:"; " exercised);
     List.equal String.equal exercised reasons);
  p_all
    "op_legality refuses each refused fixture with its first violation's message, admits the rest"
    fixtures ~f:(fun (_, _, expect, _, applied, _, op) ->
      verdict_states expect (violations_after applied op) (Sched.op_legality applied op));
  (* Seeding's walk: the whole chain from the unscheduled code, the move last. The base schedule
     itself is admitted op by op, and a refused move is the chain's last verdict. A move of a loop
     the application mints is no op of such a chain ({!move}). *)
  let seedable =
    List.filter fixtures ~f:(fun (_, _, _, _, _, move, _) ->
        match move with Of_schedule _ -> true | Of_applied _ -> false)
  in
  p_all "schedule_legality refuses exactly the refused fixtures, at their own position" seedable
    ~f:(fun ((_, _, lowered, _), _, expect, schedule, applied, _, op) ->
      let verdicts = Sched.schedule_legality (hermetic lowered) (schedule @ [ op ]) in
      List.length verdicts = List.length schedule + 1
      && List.for_all (List.drop_last_exn verdicts) ~f:(fun (_, v) ->
          match v with Sched.Op_illegal _ -> false | _ -> true)
      && verdict_states expect (violations_after applied op) (snd (List.last_exn verdicts)));

  (* --- The compile legs: each fixture compiled anyway, as the beam compiles a round candidate,
     with the IR handed to the renderer kept so the oracle can be asked about that very IR. --- *)
  let when_, aggregation =
    match
      Ll_test.tensorized_matmul_capability ~is_gpu:on_gpu ~is_cpu:(not on_gpu)
        ~limits:(Context.hardware_limits (Context.auto ()))
        ~a:ma.Tensor.value ~b:mb.Tensor.value
        ~d:(match prepared with (_, t, _, _) :: _ -> t.Tensor.value | [] -> assert false)
    with
    | `Advertised -> (on_gpu, `Backend)
    | `Withheld aggregation -> (false, aggregation)
  in
  let compiled =
    if not when_ then []
    else
      List.mapi fixtures ~f:(fun idx ((base, _, _, reference), label, _, _, _, move, _) ->
          let t = base.b_tensor () in
          let compiled_ir = ref None in
          let transform opt =
            let schedule, s = base.b_schedule ~out:t.Tensor.value opt in
            let base_applied = Sched.apply schedule opt in
            let applied = Sched.apply [ move_op move s base_applied ] base_applied in
            compiled_ir := Some (hermetic applied);
            applied
          in
          let outcome = values t (compile ~transform ("fsm_move" ^ Int.to_string idx) t) in
          let vs = Option.value_map !compiled_ir ~default:[] ~f:Sched.fragment_scope_violations in
          (label, reference, vs, outcome))
  in
  gated_all ~aggregation ~when_ ~on:backend_name
    "each compiled fixture renders one lane-0 scalar fallback per violation the oracle finds in \
     its IR, and a tensorized statement besides where it finds none"
    compiled ~f:(fun (label, _, vs, outcome) ->
      match outcome with
      | Error why ->
          Stdio.eprintf "%s: did not compile: %s\n%!" label why;
          false
      | Ok (_, (mma : Ir.C_syntax.mma_summary)) ->
          Stdio.eprintf "%s: %d violation(s); %d of %d statement(s) on the fallback\n%!" label
            (List.length vs) mma.scalar_fallbacks mma.statements;
          mma.scalar_fallbacks = List.length vs
          && ((not (List.is_empty vs)) || mma.scalar_fallbacks < mma.statements));
  gated_all ~aggregation ~when_ ~on:backend_name
    "each compiled fixture computes the serial twin's values bitwise" compiled
    ~f:(fun (label, reference, _, outcome) ->
      match outcome with
      | Error _ -> false
      | Ok (got, _) ->
          let mismatches =
            List.filter_mapi (Array.to_list got) ~f:(fun i g ->
                if Float.equal g reference.(i) then None else Some (i, g, reference.(i)))
          in
          (match mismatches with
          | [] -> ()
          | (i, g, r) :: _ ->
              Stdio.eprintf "%s: %d of %d cells differ; first at %d: %g vs the twin's %g\n%!" label
                (List.length mismatches) (Array.length got) i g r);
          List.is_empty mismatches)
