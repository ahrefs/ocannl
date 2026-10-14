(* gh-ocannl-1231: a move applied on top of a pipelined (Stage pipeline_depth = 2) schedule is
   refused by the op-legality oracle when it breaks the staging scope's invariant — before it costs
   a candidate compile — and admitted when it does not.

   The invariant (Schedule.pipeline_violation): every loop binding a pipelined tile's rotor is
   Serial, and every read of the tile sits inside one. They are the two positional facts C_syntax
   checks while rendering the buffer rotation; #1137's CUDA probe lost 14 round-1 candidates per
   search to them (same three moves of each beam entry's menu, deterministically).

   The fixtures are those moves, as the beam menu proposes them for the rotor loop k_o (extent 4, so
   the menu offers a split by 2 and both unrolls): [Split] by 2 and [Unroll ~materialize:true]
   remove the rotor binder, leaving the tile's reads outside any rotor loop; [Unroll
   ~materialize:false] keeps the loop but makes it Unrolled. Each is refused with its typed reason,
   by [op_legality] (the beam menu's gate) and by [schedule_legality] at the move's own position
   (seeding's walk). Two moves on the same schedules are still admitted: a split of a loop inside
   the staging scope, and a split of a loop outside it.

   The base is the staged + tensorized matmul composition of schedule_pipelined_matmul (both stages
   anchored at k_o). The legality claims are structural and run on every backend. On a GPU backend
   the compile legs also show the oracle agrees with the renderer: each refused move, compiled
   anyway as a candidate, is C_syntax's typed Backend_codegen decline, while the admitted move
   compiles and computes the serial twin's values. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false }

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 64
let bm = 16
let simd_width = 32

type syms = {
  k_o : Ir.Indexing.symbol;  (** The rotor: both stages anchor here. *)
  j : Ir.Indexing.symbol;  (** A compute loop inside the staging scope (before Tensorize). *)
  zi_i : Ir.Indexing.symbol;  (** The zeroing nest's serial loop, outside the staging scope. *)
}

(* The staged (+ optionally tensorized) schedule of schedule_pipelined_matmul at depth 2. *)
let staged_schedule ~tensorize ~ma ~mb ~out (opt : LL.optimized) : Sched.schedule * syms =
  let paths = Ll_test.nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun path -> List.length path = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  (* The zeroing's workgroup axis matches the tensorization lane's width (one workgroup extent per
     kernel). *)
  let rz, _, _ = Sched.split ~axis:zj ~factor:simd_width ~outer:LL.Serial ~inner:LL.Workgroup in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
  let stage source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 2;
        tile_prec = None;
      }
  in
  ( ([
       ez;
       sp_zi;
       rz;
       sp_i;
       sp_k;
       Sched.Swap { outer = j; inner = k_o };
       Sched.Swap { outer = i_i; inner = k_o };
       stage ma [ i_i; k_i ];
       stage mb [ k_i; j ];
     ]
    @ if tensorize then [ tz ] else []),
    { k_o; j; zi_i } )

(* The moves, as the beam menu builds them (serial splits; unrolls in both representations). *)
type expect = Refused_read_outside | Refused_not_serial | Admitted

let moves : (string * expect * (syms -> Sched.optop)) list =
  [
    ( "Split k_o by 2 (the rotor)",
      Refused_read_outside,
      fun s ->
        let op, _, _ = Sched.split ~axis:s.k_o ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
        op );
    ( "Unroll k_o ~materialize:true (the rotor)",
      Refused_read_outside,
      fun s -> Sched.Unroll { axis = s.k_o; materialize = true } );
    ( "Unroll k_o ~materialize:false (the rotor)",
      Refused_not_serial,
      fun s -> Sched.Unroll { axis = s.k_o; materialize = false } );
    ( "Split zi_i by 2 (outside the staging scope)",
      Admitted,
      fun s ->
        let op, _, _ = Sched.split ~axis:s.zi_i ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
        op );
  ]

(* Inside the staging scope: only expressible before Tensorize consumes the compute loops. *)
let inner_move s =
  let op, _, _ = Sched.split ~axis:s.j ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
  op

let hermetic (opt : LL.optimized) : LL.optimized =
  {
    opt with
    LL.traced_store = Hashtbl.copy opt.LL.traced_store;
    optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
  }

let violation_matches expect (v : Sched.pipeline_violation option) =
  match (expect, v) with
  | Refused_read_outside, Some (Sched.Read_outside_rotor _) -> true
  | Refused_not_serial, Some (Sched.Rotor_not_serial { axis = LL.Unrolled; _ }) -> true
  | Admitted, None -> true
  | _ -> false

let verdict_matches expect (v : Sched.op_verdict) =
  match (expect, v) with
  | Refused_read_outside, Sched.Op_illegal msg ->
      String.is_substring msg ~substring:"read outside its rotor loop"
  | Refused_not_serial, Sched.Op_illegal msg ->
      String.is_substring msg ~substring:"no longer Serial"
  | Admitted, (Sched.Op_legal | Sched.Op_unknown _) -> true
  | _ -> false

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* The serial twin: the reference values, and (via the transform) the lowered code the structural
     legs schedule — captured unscheduled, hermetically, so each leg applies its own copy. *)
  let%op mc0 = ma * mb in
  let lowered = ref None in
  let ctx_s, routine_s =
    Context.compile
      ~lowered_transform:(fun opt ->
        lowered := Some (hermetic opt);
        [ opt ])
      (Context.auto ())
      (named "pml_serial" (Train.forward mc0))
      Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = Context.get_values ctx_s mc0.Tensor.value in
  p "the serial reference is not all zeros" (Array.exists got_serial ~f:(fun x -> Float.(x <> 0.)));
  let lowered = Option.value_exn !lowered in
  let out = mc0.Tensor.value in

  (* --- The structural legs (every backend). --- *)
  let base ~tensorize =
    let opt = hermetic lowered in
    let schedule, s = staged_schedule ~tensorize ~ma:ma.Tensor.value ~mb:mb.Tensor.value ~out opt in
    (schedule, s, Sched.apply schedule (hermetic opt))
  in
  let schedule, s, applied = base ~tensorize:true in
  p "the base records both tiles pipelined at depth 2 with k_o as rotor"
    (Map.length applied.LL.pipelined = 2
    && Map.for_all applied.LL.pipelined ~f:(fun { LL.pt_depth; pt_rotor } ->
        pt_depth = 2 && Ir.Indexing.equal_symbol pt_rotor s.k_o));
  p "the base itself satisfies the staging scope's invariant"
    (Option.is_none (Sched.pipeline_violation applied));
  p_all "each move's applied IR has exactly its expected violation (or none)" moves
    ~f:(fun (_, expect, mk) ->
      violation_matches expect (Sched.pipeline_violation (Sched.apply [ mk s ] (hermetic applied))));
  p_all "op_legality refuses each fixture move with its reason, admits the legal one" moves
    ~f:(fun (_, expect, mk) -> verdict_matches expect (Sched.op_legality applied (mk s)));
  (* Seeding's walk: the whole chain from the unscheduled code, the move last. A refused move is the
     chain's last verdict and the only illegal one (the base schedule itself is not refused). *)
  p_all "schedule_legality refuses exactly the fixture move at its own position" moves
    ~f:(fun (_, expect, mk) ->
      let verdicts = Sched.schedule_legality (hermetic lowered) (schedule @ [ mk s ]) in
      List.length verdicts = List.length schedule + 1
      && List.for_all (List.drop_last_exn verdicts) ~f:(fun (_, v) ->
          match v with Sched.Op_illegal _ -> false | _ -> true)
      && verdict_matches expect (snd (List.last_exn verdicts)));
  (* Inside the staging scope, before Tensorize: splitting a compute loop keeps every read of both
     tiles under k_o. *)
  (let _, s_plain, applied_plain = base ~tensorize:false in
   p "a split inside the staging scope keeps the invariant and is admitted"
     (Option.is_none
        (Sched.pipeline_violation (Sched.apply [ inner_move s_plain ] (hermetic applied_plain)))
     && verdict_matches Admitted (Sched.op_legality applied_plain (inner_move s_plain))));

  (* --- The compile legs (GPU): the oracle agrees with the renderer. Each move is compiled as a
     candidate on top of the base, the way the beam compiles a round candidate. --- *)
  let compile_move ~idx mk =
    let%op mcm = ma * mb in
    let transform opt =
      let schedule, s =
        staged_schedule ~tensorize:true ~ma:ma.Tensor.value ~mb:mb.Tensor.value
          ~out:mcm.Tensor.value opt
      in
      Sched.apply (schedule @ [ mk s ]) opt
    in
    ( mcm,
      Context.compile_outcome
        ~lowered_transform:(fun o -> [ transform o ])
        ~provenance:SO.Candidate
        ~candidate:("pml_move" ^ Int.to_string idx)
        (Context.auto ())
        (named ("pml_move" ^ Int.to_string idx) (Train.forward mcm))
        Ir.Indexing.Empty )
  in
  let renderer_agrees ~idx (_, expect, mk) =
    let mcm, outcome = compile_move ~idx mk in
    match (expect, outcome) with
    | (Refused_read_outside | Refused_not_serial), Error (SO.Classified { phase; cause; _ }) -> (
        SO.equal_phase phase SO.Backend_codegen
        &&
        match (expect, cause) with
        | Refused_read_outside, SO.Unsupported { feature; _ } ->
            String.equal feature "pipelined tile read outside its rotor loop"
        | Refused_not_serial, SO.Unsupported { feature; _ } ->
            String.equal feature "pipelined tile whose rotor loop is not Serial"
        | _ -> false)
    | Admitted, Ok (ctx, routine) ->
        let ctx = Context.run ctx routine in
        let got = Context.get_values ctx mcm.Tensor.value in
        Array.for_all2_exn got got_serial ~f:(fun a b -> Float.(abs (a - b) < 1e-2))
    | _, Error (SO.Classified c) ->
        Stdio.eprintf "move %d: unexpected decline: %s\n%!" idx
          (Sexp.to_string_hum (SO.sexp_of_classified_cause c));
        false
    | _, Error (SO.Fatal { exn; _ }) ->
        Stdio.eprintf "move %d: fatal: %s\n%!" idx (Exn.to_string exn);
        false
    | _, Ok _ ->
        Stdio.eprintf "move %d: compiled, but a decline was expected\n%!" idx;
        false
  in
  if on_gpu then
    p_alli
      "compiled anyway, each refused move is the renderer's typed decline; the admitted one runs"
      moves ~f:(fun idx move -> renderer_agrees ~idx move)
  else
    (* The C backends reject the workgroup-shared composition before either renderer check. *)
    skipped ~backend:backend_name
      "compiled anyway, each refused move is the renderer's typed decline; the admitted one runs"
