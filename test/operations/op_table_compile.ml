(* gh-ocannl-1731: every (operator, precision) arm of the selected backend's operator tables renders
   and compiles. Two defects found by gh-ocannl-1684 shipped because nothing compiled the arms no
   value test happened to run: float [Mod] on cc rendered C's [%], which does not compile on floats,
   and half [Mod] on CUDA/HIP was an ambiguous [fmod] call.

   The sweep goes through the backend's real pipeline, the [?prelowered] seam of
   {!Context.compile_outcome}: one hand-built routine per storage precision applies every in-domain
   operator to that precision's operands and stores each result in a cell of its own, and the
   backend renders and compiles it. Nothing runs. The code reaching the backend is the hand-built
   [Low_level.t] AS WRITTEN (the optimizer only builds the record around it), so no simplification
   can drop an arm before it renders.

   - A table arm is an operator at a COMPUTE precision, and which compute precision a storage
   precision renders at is the numerics policy's choice ({!Ir.Numerics.t}: on cc, narrow storage
   computes in f32 unless [narrow_compute_f32 = false], or f16 natively under [Fp16_narrow]). So the
   sweep takes every policy in {!Ir.Numerics.all}, reads the backend's own resolution off
   {!Context.codegen_capabilities}, and compiles each distinct (storage, compute) pair once, under
   the first policy that reaches it. Arms at a compute precision no policy reaches are unreachable
   on this backend, and are listed on stderr rather than compiled. - An arm the backend REFUSES to
   render (an [Invalid_argument] or a typed refusal raised during code generation, such as [ToPowOf]
   at an integer precision) is a decline, not a defect: the refusal is a loud, attributed error.
   What must never happen is an arm that renders and then fails the backend's compiler -- that is
   the silent class, a kernel that only a user meets. Refusals are listed on stderr; the claims
   below bound where they may occur. - A routine that fails is split in halves until each failure is
   one arm, so a defect is named by its operator and precision, and a refused arm costs no compiler
   run (refusal happens before the compiler is invoked).

   The operator domain is read off {!Ops}, not listed: every derived enumeration of [unop], [binop],
   [ternop] and [vec_unop], minus three classes no lowering can emit, each excluded by a predicate
   over the operator rather than a list. The projections [Arg1]/[Arg2] have no spelling (the
   renderer emits the selected operand alone, {!Ops.binop_conditionality}). The Threefry operators
   are defined only on packed [uint4x32] state, so at a scalar precision they are out of domain, and
   at [uint4x32] they are the only arithmetic in domain ({!Ops.prec_family}: packed RNG state, not a
   scalar). The heterogeneous [uint4x32] consumers ({!Ops.is_homogeneous_prec_unop} and friends)
   take a [uint4x32] block and produce every scalar precision. Each homogeneous arithmetic arm is
   swept twice, over buffer reads and with an inlined [a + b] as its first operand ([Inlined]),
   since integer promotion types the two differently; at an integer storage precision, a third time
   over an inlined real-valued [Sqrt(a)] ([Inlined_real], gh-ocannl-1785), and a fourth as the first
   operand of [Mod] ([Under_mod], gh-ocannl-1883), which takes integer-typed operands only.
   Conversions are every ordered pair of distinct storage precisions the backend can store
   ({!Ir.Backend_intf.codegen_capabilities}'s [supports_f64]).

   gh-ocannl-1768 extended the sweep in three directions.

   - The explicit-SIMD renderings: at each float compute precision, every arm the backend's vector
   style renders ({!vector_arms}: cc's GCC vector arithmetic and SIMD reductions, the GPU dialects'
   packed lanes) as a [Vectorized] loop of its own. A loop the rendering declines compiles serially,
   so each compiled routine's source is read for the rendering's banner: at one precision the
   rendering takes all of its arms or none, and on every backend some precision takes them. - The
   value-select blends ([select_blend_syntax], gh-ocannl-1622): the [Where] and gate arms already
   reach them; the sources now show it, every select arm blending or none at a precision, and on cc
   and HIP some precision blending. - Builtin parameter guards ({!Ir.C_syntax.Param_guards}): the
   whole sweep compiles with every builtin call on cc and Metal passing its arguments through a
   guard that fails the compile when the operand's type narrows into the parameter's, so the compile
   claim covers what gh-ocannl-1704 was (a [half] passed by value to a [uint16_t]). Its negative
   control plants gh-ocannl-1704 back in the backend's own table, and must compile without the
   guards and fail with them. Which functions the guards read is checked over both tables: an
   unreadable function is named, with why. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Outcome = Ir.Schedule_outcome
module Guards = Ir.C_syntax.Param_guards
module Generated = Test_utils.Generated
open Verdict.Claims

type arm =
  | Unary of Ops.unop
  | Binary of Ops.binop
  | Ternary of Ops.ternop
  | Vector of Ops.vec_unop
  | Convert_from of Ops.prec
  | Inlined of arm
      (** A homogeneous arithmetic arm whose first operand is an inlined [a + b] rather than a
          buffer read: C-family integer promotion types that sum [int] where the read is [uchar], so
          an arm that compiles over a read can still be an ambiguous call over an expression. *)
  | Inlined_real of arm
      (** A homogeneous arithmetic arm at an integer storage precision whose first operand is an
          inlined real-valued [Sqrt(a)]: a float-valued subexpression at an integer compute
          precision, which the renderer computes in float and converts once (gh-ocannl-1785), so an
          integer arm never meets a float operand -- no mixed [min(uchar, double)], no [%] over a
          double. *)
  | Under_mod of arm
      (** A homogeneous arithmetic arm at an integer storage precision as the first operand of
          [Mod], the one operator whose integer spelling takes integers only: an arm that is
          float-TYPED over its integral value (cc's [fmax], [truncf] and [? 1.0 : 0.0], Metal's
          [clamp((float)...)]) compiles into a store and fails here (gh-ocannl-1883). *)
  | Lanes of arm
      (** A homogeneous arithmetic arm applied cell by cell over a [Vectorized] loop, every operand
          read at the loop index: what the explicit-SIMD rendering takes (gh-ocannl-1768). *)
  | Lanes_uniform of arm
      (** {!Lanes} with the last operand a loop-free read: the rendering's splat of a lane-uniform
          value, and the only form a gated operand may take in a lane. *)
  | Reduce of Ops.binop
      (** [acc = op(acc, x[i])] over a [Vectorized] loop: the SIMD accumulator grid. *)
  | Reduce_fma  (** [acc = fma(x[i], y[i], acc)] over a [Vectorized] loop. *)

let rec arm_name = function
  | Unary op -> "unop " ^ Sexp.to_string (Ops.sexp_of_unop op)
  | Binary op -> "binop " ^ Sexp.to_string (Ops.sexp_of_binop op)
  | Ternary op -> "ternop " ^ Sexp.to_string (Ops.sexp_of_ternop op)
  | Vector op -> "vec_unop " ^ Sexp.to_string (Ops.sexp_of_vec_unop op)
  | Convert_from q -> "convert from " ^ Ops.prec_string q
  | Inlined arm -> arm_name arm ^ " over an inlined operand"
  | Inlined_real arm -> arm_name arm ^ " over an inlined real-valued operand"
  | Under_mod arm -> arm_name arm ^ " under Mod"
  | Lanes arm -> arm_name arm ^ " over vector lanes"
  | Lanes_uniform arm -> arm_name arm ^ " over vector lanes, its last operand uniform"
  | Reduce op -> "vector reduction " ^ Sexp.to_string (Ops.sexp_of_binop op)
  | Reduce_fma -> "vector fma reduction"

let is_vector_arm = function
  | Lanes _ | Lanes_uniform _ | Reduce _ | Reduce_fma -> true
  | Unary _ | Binary _ | Ternary _ | Vector _ | Convert_from _ | Inlined _ | Inlined_real _
  | Under_mod _ ->
      false

(* A value select: an operator that yields one of its operands, the other evaluated only
   conditionally, which the backend's [select_blend_syntax] may render branchless (gh-ocannl-1622).
   The logical [And] and [Or] gate their second operand too, but yield a truth value, not an
   operand: they are not selects. *)
let rec is_select = function
  | Ternary op -> (
      match Ops.ternop_conditionality op with
      | Ops.Cond_and_one_arm -> true
      | Ops.All_three -> false)
  | Binary (Ops.And | Ops.Or) -> false
  | Binary op -> (
      match Ops.binop_conditionality op with
      | Ops.Gated_second -> true
      | Ops.Both_operands | Ops.Only_first | Ops.Only_second -> false)
  | Inlined arm | Inlined_real arm | Under_mod arm -> is_select arm
  | Unary _ | Vector _ | Convert_from _ | Lanes _ | Lanes_uniform _ | Reduce _ | Reduce_fma -> false

type verdict =
  | Compiled
  | Refused of string  (** Refused while lowering or rendering: before any compiler ran. *)
  | Failed of string  (** Rendered, then the backend's compiler or loader rejected it. *)

let is_packed prec = match Ops.prec_family prec with Ops.Packed_integer -> true | _ -> false
let is_threefry = function Ops.Threefry4x32_crypto | Ops.Threefry4x32_light -> true | _ -> false

let is_projection op =
  match Ops.binop_conditionality op with Ops.Only_first | Ops.Only_second -> true | _ -> false

(* The extent of every [Vectorized] loop: a multiple of every pack and chain width a backend
   renders, so neither rendering declines for want of lanes. *)
let lanes_extent = 128

(* The vector arms at compute precision [compute], in the backend's explicit-SIMD style (both
   renderings decline anything but a float compute precision).

   - [`Vec_extensions] (cc's GCC/Clang vector types) renders vector arithmetic itself, so its arms
   are the operators [C_syntax]'s [vec_expr] spells over vectors -- the four infix operators, a
   fused [FMA], [Neg] and [Identity] -- each also over a lane-uniform last operand where it has one
   (the splat), and the accumulations its SIMD reduction grid folds ({!Ir.Low_level}'s
   [accum_local_update_parts]: [Add], [Mul], [Max], [Min], and the fused form). - [`Packed_struct]
   (the GPU dialects) moves whole packs and computes each lane through the scalar operator tables,
   so its arms are every homogeneous arithmetic arm, a gated operand uniform (a lane may not load
   one, see [try_vectorize]). It renders no reduction. *)
let vector_arms ~style ~compute =
  if not (Ops.is_float compute) then []
  else
    match style with
    | `Vec_extensions ->
        List.map
          [
            Binary Ops.Add;
            Binary Ops.Sub;
            Binary Ops.Mul;
            Binary Ops.Div;
            Ternary Ops.FMA;
            Unary Ops.Neg;
            Unary Ops.Identity;
          ] ~f:(fun arm -> Lanes arm)
        @ [ Lanes_uniform (Binary Ops.Add); Lanes_uniform (Ternary Ops.FMA) ]
        @ List.map [ Ops.Add; Ops.Mul; Ops.Max; Ops.Min ] ~f:(fun op -> Reduce op)
        @ [ Reduce_fma ]
    | `Packed_struct ->
        List.filter_map Ops.all_of_unop ~f:(fun op ->
            if Ops.is_homogeneous_prec_unop op then Some (Lanes (Unary op)) else None)
        @ List.filter_map Ops.all_of_binop ~f:(fun op ->
            if is_projection op || is_threefry op || not (Ops.is_homogeneous_prec_binop op) then
              None
            else
              match Ops.binop_conditionality op with
              | Ops.Gated_second -> Some (Lanes_uniform (Binary op))
              | Ops.Both_operands | Ops.Only_first | Ops.Only_second -> Some (Lanes (Binary op)))
        @ List.filter_map Ops.all_of_ternop ~f:(fun op ->
            if Ops.is_homogeneous_prec_ternop op then Some (Lanes (Ternary op)) else None)

(* The in-domain arms at storage precision [store], over the storage precisions [precs] the backend
   can hold. *)
let arms_at ~precs store =
  let conversions =
    List.filter_map precs ~f:(fun q ->
        if Ops.equal_prec q store then None else Some (Convert_from q))
  in
  if is_packed store then
    List.filter_map Ops.all_of_binop ~f:(fun op ->
        if is_threefry op then Some (Binary op) else None)
    @ conversions
  else
    let arithmetic =
      List.map Ops.all_of_unop ~f:(fun op -> Unary op)
      @ List.filter_map Ops.all_of_binop ~f:(fun op ->
          if is_projection op || is_threefry op then None else Some (Binary op))
      @ List.map Ops.all_of_ternop ~f:(fun op -> Ternary op)
    in
    let homogeneous = function
      | Unary op -> Ops.is_homogeneous_prec_unop op
      | Binary op -> Ops.is_homogeneous_prec_binop op
      | Ternary op -> Ops.is_homogeneous_prec_ternop op
      | Vector _ | Convert_from _ | Inlined _ | Inlined_real _ | Under_mod _ | Lanes _
      | Lanes_uniform _ | Reduce _ | Reduce_fma ->
          false
    in
    arithmetic
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm then Some (Inlined arm) else None)
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm && Ops.is_integer store then Some (Inlined_real arm) else None)
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm && Ops.is_integer store then Some (Under_mod arm) else None)
    @ List.map Ops.all_of_vec_unop ~f:(fun op -> Vector op)
    @ conversions

(* Node ids: this executable's range. *)
let next_id = ref 17_310_000

let node ~prec ?(dims = [| 1 |]) label =
  Int.incr next_id;
  let tn =
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  Ll_test.materialize tn;
  tn

(* The operands and result cells of one storage precision's routine. Every operand is its own node,
   so no operator sees the same value twice and no read folds into another. *)
type operands = {
  store : Ops.prec;
  a : Tn.t;
  b : Tn.t;
  c : Tn.t;
  block : Tn.t;  (** A packed [uint4x32] block, for the heterogeneous consumers. *)
  lane : Tn.t;  (** An [int32] lane index. *)
  sources : (Ops.prec * Tn.t) list;  (** A source node per conversion. *)
  lanes : Tn.t array;  (** Three operands of [lanes_extent] cells, for the vector arms. *)
  out : Tn.t;  (** One cell per scalar arm. *)
  vec_out : Tn.t;  (** The lanes of the vector arm. *)
}

let operands ~precs ~n_arms store =
  let tag = Ops.prec_string store in
  let mk label = node ~prec:store (tag ^ "_" ^ label) in
  {
    store;
    a = mk "a";
    b = mk "b";
    c = mk "c";
    block = node ~prec:Ops.uint4x32 (tag ^ "_block");
    lane = node ~prec:Ops.int32 (tag ^ "_lane");
    sources =
      List.filter_map precs ~f:(fun q ->
          if Ops.equal_prec q store then None
          else Some (q, node ~prec:q (tag ^ "_from_" ^ Ops.prec_string q)));
    lanes =
      Array.init 3 ~f:(fun k ->
          node ~prec:store ~dims:[| lanes_extent |] (Printf.sprintf "%s_lanes%d" tag k));
    out = node ~prec:store ~dims:[| n_arms |] (tag ^ "_out");
    vec_out =
      (if is_packed store then mk "vec_out_unused"
       else node ~prec:store ~dims:[| Ops.vec_unop_lanes store |] (tag ^ "_vec_out"));
  }

let at0 = [| Idx.Fixed_idx 0 |]
let read tn prec : LL.scalar_arg = (LL.Get (tn, at0), prec)

(* The statement applying a vector arm, and the nodes it writes: each its own, so no two loops of a
   routine write one node. *)
let vector_statement o arm : LL.t * Tn.t list =
  let s = o.store in
  let i = Ll_test.sym () in
  let at_i = [| Idx.Iterator i |] in
  let lane k = (LL.Get (o.lanes.(k), at_i), s) in
  let loop body = Ll_test.loop ~axis:LL.Vectorized ~upto:(lanes_extent - 1) i body in
  let tag = Ops.prec_string s in
  match arm with
  | Lanes inner | Lanes_uniform inner ->
      let uniform = match arm with Lanes_uniform _ -> true | _ -> false in
      let operand k ~last = if uniform && last then read o.c s else lane k in
      let llsc : LL.scalar_t =
        match inner with
        | Unary op -> Unop (op, operand 0 ~last:true)
        | Binary op -> Binop (op, operand 0 ~last:false, operand 1 ~last:true)
        | Ternary op ->
            Ternop (op, operand 0 ~last:false, operand 1 ~last:false, operand 2 ~last:true)
        | _ -> invalid_arg ("op_table_compile: no vector rendering of " ^ arm_name inner)
      in
      let out = node ~prec:s ~dims:[| lanes_extent |] (tag ^ "_vout") in
      (loop (Ll_test.set out at_i llsc), [ out ])
  | Reduce op ->
      let acc = node ~prec:s (tag ^ "_vacc") in
      (loop (Ll_test.set acc at0 (LL.Binop (op, read acc s, lane 0))), [ acc ])
  | Reduce_fma ->
      let acc = node ~prec:s (tag ^ "_vacc") in
      (loop (Ll_test.set acc at0 (LL.Ternop (Ops.FMA, lane 0, lane 1, read acc s))), [ acc ])
  | _ -> invalid_arg ("op_table_compile: not a vector arm: " ^ arm_name arm)

(* The statement applying [arm] (the [k]th of its routine). *)
let rec statement o k arm : LL.t =
  let s = o.store in
  let store llsc = Ll_test.set o.out [| Idx.Fixed_idx k |] llsc in
  let sum = (LL.Binop (Ops.Add, read o.a s, read o.b s), s) in
  let real = (LL.Unop (Ops.Sqrt, read o.a s), s) in
  match arm with
  | Lanes _ | Lanes_uniform _ | Reduce _ | Reduce_fma ->
      invalid_arg "op_table_compile: a vector arm is a loop of its own (vector_statement)"
  | Inlined (Unary op) -> store (LL.Unop (op, sum))
  | Inlined (Binary op) -> store (LL.Binop (op, sum, read o.c s))
  | Inlined (Ternary op) -> store (LL.Ternop (op, sum, read o.b s, read o.c s))
  | Inlined arm -> statement o k arm
  | Inlined_real (Unary op) -> store (LL.Unop (op, real))
  | Inlined_real (Binary op) -> store (LL.Binop (op, real, read o.c s))
  | Inlined_real (Ternary op) -> store (LL.Ternop (op, real, read o.b s, read o.c s))
  | Inlined_real arm -> statement o k arm
  | Under_mod (Unary op) -> store (LL.Binop (Ops.Mod, (LL.Unop (op, read o.a s), s), read o.c s))
  | Under_mod (Binary op) ->
      store (LL.Binop (Ops.Mod, (LL.Binop (op, read o.a s, read o.b s), s), read o.c s))
  | Under_mod (Ternary op) ->
      store
        (LL.Binop (Ops.Mod, (LL.Ternop (op, read o.a s, read o.b s, read o.c s), s), read o.c s))
  | Under_mod arm -> statement o k arm
  | Unary op when Ops.is_homogeneous_prec_unop op -> store (LL.Unop (op, read o.a s))
  | Unary op -> store (LL.Unop (op, read o.block Ops.uint4x32))
  | Binary op when Ops.is_homogeneous_prec_binop op -> store (LL.Binop (op, read o.a s, read o.b s))
  | Binary op -> store (LL.Binop (op, read o.block Ops.uint4x32, read o.lane Ops.int32))
  | Ternary op -> store (LL.Ternop (op, read o.a s, read o.b s, read o.c s))
  | Vector vec_unop ->
      Ll_test.set_from_vec ~length:(Ops.vec_unop_lanes s) ~vec_unop o.vec_out at0
        (read o.block Ops.uint4x32)
  | Convert_from q ->
      let src = List.Assoc.find_exn o.sources ~equal:Ops.equal_prec q in
      store (LL.Get (src, at0))

let nodes o =
  [ o.a; o.b; o.c; o.block; o.lane; o.out; o.vec_out ]
  @ Array.to_list o.lanes @ List.map o.sources ~f:snd

(* The vector and blend coverage claims read the emitted kernels. *)
let () = Utils.settings.output_debug_files_in_build_directory <- true
let base_ctx = Lazy.force Ll_test.base_ctx
let backend_name = String.lowercase (Context.backend_name base_ctx)
let () = Generated.init ~backend_name
let compiles = ref 0

(* A refusal is the renderer raising [Invalid_argument] or a [User_error] (or a typed [Unsupported]
   decline) -- the attributed errors a table or the code generator raises for a pair it does not
   render. The PHASE does not tell a refusal from a compiler rejection: cc renders inside its
   compile step, so both report [Backend_compile]. Anything else -- the backend's compiler or loader
   rejecting the rendered code ([Backend_rejected]), or an exception nothing classified -- is a
   failure, so an unrecognized outcome can only fail the claim, never pass it. *)
let is_refusal_constructor name =
  String.equal name "Invalid_argument" || String.is_suffix name ~suffix:"User_error"

let verdict_of_failure : Outcome.failure -> verdict =
  let phase_name phase = Sexp.to_string (Outcome.sexp_of_phase phase) in
  function
  | Outcome.Fatal { phase; exn; _ } ->
      let detail = phase_name phase ^ ": " ^ Exn.to_string exn in
      if is_refusal_constructor (Stdlib.Printexc.exn_slot_name exn) then Refused detail
      else Failed detail
  | Outcome.Classified { phase; cause; _ } -> (
      let detail = phase_name phase ^ ": " ^ Sexp.to_string (Outcome.sexp_of_cause cause) in
      match cause with
      | Outcome.Unsupported _ -> Refused detail
      | Outcome.Unclassified { exn_constructor; _ } when is_refusal_constructor exn_constructor ->
          Refused detail
      | _ -> Failed detail)

(* The routines that compiled, with the arms each holds: the compiled arms, partitioned, since a
   routine that compiles is never split. *)
let compiled_routines : (string * arm list) list ref = ref []

(* Compiles the [arms] (each with its routine-wide position) as one routine; [None] when it
   compiled. *)
let try_compile ~policy ~name o arms =
  let llc, written =
    List.fold_right arms ~init:(LL.Noop, []) ~f:(fun (k, arm) (acc, written) ->
        if is_vector_arm arm then
          let stmt, nodes = vector_statement o arm in
          (LL.Seq (stmt, acc), nodes @ written)
        else (LL.Seq (statement o k arm, acc), written))
  in
  Int.incr compiles;
  let name = Printf.sprintf "%s_%d" name !compiles in
  let compiled () = compiled_routines := (name, List.map arms ~f:snd) :: !compiled_routines in
  Test_utils.with_policy
    (fun _ -> policy)
    (fun () ->
      match Ll_test.optimize ~materialized:(nodes o @ written) ~name llc with
      | exception exn ->
          let detail = "lowering: " ^ Exn.to_string exn in
          Some
            (if is_refusal_constructor (Stdlib.Printexc.exn_slot_name exn) then Refused detail
             else Failed detail)
      | optimized -> (
          match
            Context.compile_outcome ~name ~provenance:Outcome.Candidate
              ~prelowered:{ optimized with LL.llc }
              ~lowered_transform:(fun x -> [ x ])
              base_ctx Ir.Assignments.empty_comp Idx.Empty
          with
          | Ok (_ctx, _routine) ->
              compiled ();
              None
          | Error failure -> Some (verdict_of_failure failure)))

(* Routines that failed although every one of their arms compiled on its own: a failure of the
   combination (a binding limit, say), which no single arm explains. Splitting would otherwise drop
   it, and the sweep would report the arms it could not compile together as compiled. *)
let unexplained : (string * verdict) list ref = ref []

(* Splits a failing routine until each failure is one arm. *)
let rec probe ~policy ~name o arms =
  match try_compile ~policy ~name o arms with
  | None -> List.map arms ~f:(fun (_, arm) -> (arm, Compiled))
  | Some verdict -> (
      match arms with
      | [ (_, arm) ] -> [ (arm, verdict) ]
      | _ ->
          let left, right = List.split_n arms (List.length arms / 2) in
          let results = probe ~policy ~name o left @ probe ~policy ~name o right in
          if List.for_all results ~f:(fun (_, v) -> match v with Compiled -> true | _ -> false)
          then
            unexplained :=
              ( Printf.sprintf "%s over %d arms from %s" name (List.length arms)
                  (arm_name (snd (List.hd_exn arms))),
                verdict )
              :: !unexplained;
          results)

(* Whether the C reference refuses [arm] at these precisions too: the shared C spellings in {!Ops}
   raise for it, or it is [FMA] at a non-float precision, which the C-family emission seam
   ({!Ir.C_syntax}'s [pp_scalar], gh-ocannl-873) refuses for every backend before any table is
   consulted. *)
let rec c_reference_refuses ~store ~compute arm =
  let raises f = match f () with _ -> false | exception Invalid_argument _ -> true in
  match arm with
  | Unary op -> raises (fun () -> Ops.unop_c_syntax compute op)
  | Binary op -> raises (fun () -> Ops.binop_c_syntax compute op)
  | Ternary Ops.FMA -> not (Ops.is_float compute)
  | Ternary op -> raises (fun () -> Ops.ternop_c_syntax compute op)
  | Vector op -> raises (fun () -> Ops.vec_unop_c_syntax compute op)
  | Convert_from from -> raises (fun () -> Ops.c_convert_precision ~from ~to_:store)
  | Inlined arm | Inlined_real arm | Under_mod arm | Lanes arm | Lanes_uniform arm ->
      c_reference_refuses ~store ~compute arm
  | Reduce op -> c_reference_refuses ~store ~compute (Binary op)
  | Reduce_fma -> c_reference_refuses ~store ~compute (Ternary Ops.FMA)

type case = {
  store : Ops.prec;
  compute : Ops.prec;
  results : (arm * verdict) list;
  routines : (string * arm list) list;  (** The routines that compiled, with their arms. *)
}

(* The C reference's explicit-SIMD style: the backends rendering through cc's syntax config use GCC
   vector extensions, the GPU dialects pack structs. *)
let on_cc = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts base_ctx)
let style = if on_cc then `Vec_extensions else `Packed_struct

(* How often [pattern] occurs in [text]. *)
let occurrences text ~pattern =
  List.length (String.substr_index_all text ~may_overlap:false ~pattern)

(* Over the routines of [case] that compiled an arm satisfying [kind]: how many such arms they hold,
   and how often their sources show [pattern]. *)
let witnessed case ~kind ~patterns =
  List.fold case.routines ~init:(0, 0) ~f:(fun (arms, seen) (name, routine_arms) ->
      let k = List.count routine_arms ~f:kind in
      if k = 0 then (arms, seen)
      else
        let source = Generated.read name in
        ( arms + k,
          seen + List.sum (module Int) patterns ~f:(fun pattern -> occurrences source ~pattern) ))

(* A defect the parameter guards must catch: gh-ocannl-1704 reverted, in this backend's own table.
   On cc the conversion's call site passed a [half] node bare -- [HALF_TO_UINT16] the identity -- to
   [half_to_uint4x32]'s [uint16_t]; on Metal the definition took [uint16_t] where its call site
   passes [half]. Both converted by VALUE. *)
let reverted_1704 =
  if on_cc then [ ("HALF_TO_UINT16", "#define HALF_TO_UINT16(x) (x)") ]
  else
    [
      ( "half_to_uint4x32",
        "uint4 half_to_uint4x32(uint16_t x) {\n    return uint4(uint32_t(x), 0, 0, 0);\n}" );
    ]

let () =
  let caps = Context.codegen_capabilities base_ctx in
  let precs =
    List.filter Ops.storage_precs ~f:(fun p ->
        caps.supports_f64 || not (Ops.equal_prec p Ops.double))
  in
  (* Each distinct (storage, compute) pair, with the first policy reaching it. *)
  let pairs =
    List.concat_map Ir.Numerics.all ~f:(fun policy ->
        Test_utils.with_policy
          (fun _ -> policy)
          (fun () -> List.map precs ~f:(fun store -> (store, caps.compute_prec store, policy))))
    |> List.fold ~init:[] ~f:(fun seen ((s, c, _) as pair) ->
        if List.exists seen ~f:(fun (s', c', _) -> Ops.equal_prec s s' && Ops.equal_prec c c') then
          seen
        else pair :: seen)
    |> List.rev
    |> List.stable_sort ~compare:(fun (s1, _, _) (s2, _, _) -> Ops.compare_prec s1 s2)
  in
  let cases =
    Guards.with_guards (fun () ->
        List.map pairs ~f:(fun (store, compute, policy) ->
            (* The vector arms compile apart from the scalar ones, so a routine's source witnesses
               one rendering: a packed lane renders its gate's blend too, once per lane loop. *)
            let arms = arms_at ~precs store in
            let vectors = vector_arms ~style ~compute in
            let o = operands ~precs ~n_arms:(List.length arms) store in
            let name =
              Printf.sprintf "optbl_%s_at_%s" (Ops.prec_string store) (Ops.prec_string compute)
            in
            let before = !compiled_routines in
            let positioned = List.mapi ~f:(fun k a -> (k, a)) in
            let results =
              probe ~policy ~name o (positioned arms)
              @
              if List.is_empty vectors then []
              else probe ~policy ~name:(name ^ "_vec") o (positioned vectors)
            in
            let routines =
              List.take !compiled_routines (List.length !compiled_routines - List.length before)
            in
            { store; compute; results; routines }))
  in
  let label { store; compute; _ } =
    Printf.sprintf "%s storage at %s compute" (Ops.prec_string store) (Ops.prec_string compute)
  in
  let all_results = List.concat_map cases ~f:(fun c -> List.map c.results ~f:(fun r -> (c, r))) in
  Stdio.eprintf "op_table_compile on %s (not part of the golden): %d compile attempts\n"
    (Context.backend_name base_ctx) !compiles;
  List.iter cases ~f:(fun case ->
      let count f = List.count case.results ~f:(fun (_, v) -> f v) in
      Stdio.eprintf "  %s: %d compiled, %d refused, %d failed\n" (label case)
        (count (function Compiled -> true | _ -> false))
        (count (function Refused _ -> true | _ -> false))
        (count (function Failed _ -> true | _ -> false)));
  let reached = List.map cases ~f:(fun c -> c.compute) in
  List.iter precs ~f:(fun p ->
      if not (List.mem reached p ~equal:Ops.equal_prec) then
        Stdio.eprintf "  no policy computes at %s: its arms are unreachable here\n"
          (Ops.prec_string p));
  List.iter all_results ~f:(fun (case, (arm, verdict)) ->
      match verdict with
      | Compiled -> ()
      | Refused why -> Stdio.eprintf "  refused: %s, %s -- %s\n" (label case) (arm_name arm) why
      | Failed why -> Stdio.eprintf "  FAILED: %s, %s -- %s\n" (label case) (arm_name arm) why);
  List.iter !unexplained ~f:(fun (what, verdict) ->
      Stdio.eprintf "  UNEXPLAINED: %s -- %s\n" what
        (match verdict with Compiled -> "" | Refused why | Failed why -> why));
  p_none ~min:200 "no in-domain arm renders and then fails the backend's compiler" all_results
    ~f:(fun (_, (_, verdict)) -> match verdict with Failed _ -> true | _ -> false);
  p_empty "every failing routine's failure is pinned to one arm" ~over:cases !unexplained;
  (* A backend may render MORE than the C reference (every backend renders integer [Recip] in float,
     gh-ocannl-1785, where the C table refuses it), never refuse more: a refusal outside the C
     reference's is an operator a user of this backend alone cannot use, or an arm the sweep built
     wrongly. At a float precision the reference refuses nothing but a conversion out of packed RNG
     state. *)
  p_all "every refused arm is one the C reference refuses too" all_results
    ~f:(fun (case, (arm, v)) ->
      match v with
      | Refused _ -> c_reference_refuses ~store:case.store ~compute:case.compute arm
      | Compiled | Failed _ -> true);
  (* gh-ocannl-1785: a plain conversion out of packed RNG state is refused on every backend, with
     one message naming the operators that do convert random bits -- where Metal used to read it as
     the uniform conversion and CUDA/HIP as lane 0. The rule above only bounds refusals, so it
     cannot say this. *)
  let out_of_packed =
    List.filter all_results ~f:(fun (_, (arm, _)) ->
        match arm with Convert_from q -> is_packed q | _ -> false)
  in
  p_all ~min:1 "every plain conversion out of uint4x32 is refused, with the shared message"
    out_of_packed ~f:(fun (case, (_, v)) ->
      match v with
      | Refused why ->
          String.is_substring why
            ~substring:(Ops.no_plain_uint4x32_conversion_message ~to_:case.store)
      | Compiled | Failed _ -> false);
  p_all "every storage precision the backend holds compiled at least one arm" precs ~f:(fun p ->
      List.exists all_results ~f:(fun (case, (_, v)) ->
          Ops.equal_prec case.store p && match v with Compiled -> true | _ -> false));
  (* gh-ocannl-1768: the explicit-SIMD renderings. A [Vectorized] loop the rendering declines
     compiles as a serial loop, so compiling says nothing unless the rendering was taken: each
     compiled loop's source carries the rendering's banner. Whether a rendering takes a loop is
     decided per precision, its operators being its domain ({!vector_arms}), so at one precision it
     takes all of them or none -- an operator it silently stopped vectorizing breaks that. *)
  let vectorized case =
    witnessed case ~kind:is_vector_arm
      ~patterns:[ "/* Vectorized rendering:"; "/* Vectorized reduction rendering:" ]
  in
  List.iter cases ~f:(fun case ->
      let arms, seen = vectorized case in
      if arms > 0 then
        Stdio.eprintf "  %s: %d of %d compiled vector arms took the explicit-SIMD rendering\n"
          (label case) seen arms);
  p_all "at each precision, the explicit-SIMD rendering takes every vector arm or none" cases
    ~f:(fun case ->
      let arms, seen = vectorized case in
      seen = 0 || seen = arms);
  let vector_leg =
    Verdict.leg ~aggregation:`Environment
      ~when_:((not on_cc) || Context.Cc_backend.vector_bytes_setting () >= 8)
      ~on:"cc with cc_vector_bytes below 8" ()
  in
  Verdict.in_leg ~leg:vector_leg "some precision's vector arms take the explicit-SIMD rendering"
    (fun label ->
      p_exists label cases ~f:(fun case ->
          let _, seen = vectorized case in
          seen > 0));
  (* The value-select blends (gh-ocannl-1622): a select the backend may speculate renders as its
     [select_blend_syntax], which binds the arms to locals first. Whether it does is decided per
     compute precision, so at one precision every select arm blends or none does. *)
  let blended case =
    witnessed case ~kind:is_select ~patterns:[ Ir.C_syntax.blend_local_a ^ " =" ]
  in
  List.iter cases ~f:(fun case ->
      let arms, seen = blended case in
      if arms > 0 then
        Stdio.eprintf "  %s: %d select blends over %d compiled select arms\n" (label case) seen arms);
  p_all "at each precision, the select blend renders every select arm or none" cases ~f:(fun case ->
      let arms, seen = blended case in
      seen = 0 || seen = arms);
  (* cc and HIP blend; CUDA's ptxas if-converts the short-circuiting spelling itself, and Metal's
     compiler no tool here can read, so both keep [?:] (C_syntax's [select_blend_syntax]). Which of
     the two a backend does is its [blends_value_selects] fact, which the inline fan-in guard reads
     before codegen (gh-ocannl-1775): two-sided, on every backend, against the emitted kernels. *)
  let blends = (Context.backend_facts base_ctx).Ir.Backend_intf.blends_value_selects in
  (* Non-empty first: a sweep that compiled no select arm would make the short-circuiting reading
     pass vacuously. *)
  p_exists "the sweep compiled select arms" cases ~f:(fun case ->
      let arms, _ = blended case in
      arms > 0);
  let renders_blend case =
    let _, seen = blended case in
    seen > 0
  in
  if blends then
    p_exists "the select blend renders at some precision exactly where the backend's facts say so"
      cases ~f:renders_blend
  else
    p_none "the select blend renders at some precision exactly where the backend's facts say so"
      cases ~f:renders_blend;
  (* gh-ocannl-1768: the builtin parameter guards were in force over the whole sweep, so the claim
     above that no arm fails to compile includes that no arm passes a builtin an operand its
     parameter narrows. The guards exist on cc and Metal. *)
  let guard_leg =
    Verdict.leg ~when_:(on_cc || String.equal backend_name "metal") ~on:backend_name ()
  in
  Verdict.in_leg ~leg:guard_leg "the sweep's kernels carried builtin parameter guards" (fun label ->
      p label (Guards.emitted () > 0));
  (* The negative control: gh-ocannl-1704 reverted compiles unguarded, as it shipped, and fails the
     guards. On a cc compiler without [_Float16], [HALF_T] is the storage bits and there is no
     defect to revert. *)
  let control_leg =
    Verdict.sub_leg guard_leg ~aggregation:`Environment
      ~when_:
        (lazy
          ((not on_cc)
          || not
               (String.equal "none"
                  (Option.value ~default:"none"
                     (List.Assoc.find
                        (Context.Cc_backend.host_capabilities ())
                        ~equal:String.equal "fp16")))))
      ~on:"a cc compiler without _Float16"
  in
  let control ~guard =
    let o = operands ~precs ~n_arms:1 Ops.uint4x32 in
    let policy = List.hd_exn Ir.Numerics.all in
    Guards.with_guards ~guard ~overrides:reverted_1704 (fun () ->
        try_compile ~policy ~name:"optbl_reverted_1704" o [ (0, Convert_from Ops.half) ])
  in
  Verdict.in_leg ~leg:control_leg "gh-ocannl-1704 reverted compiles without the guards"
    (fun label -> p label (Option.is_none (control ~guard:false)));
  Verdict.in_leg ~leg:control_leg "gh-ocannl-1704 reverted fails the guards" (fun label ->
      match control ~guard:true with
      | Some (Failed why) ->
          Stdio.eprintf "  the reverted gh-ocannl-1704 under the guards: %s\n" why;
          p label true
      | Some (Refused why) ->
          Stdio.eprintf "  the reverted gh-ocannl-1704 was refused: %s\n" why;
          p label false
      | None | Some Compiled -> p label false)

(* The executed legs. Compiling an arm says nothing about its value, and the integer-precision table
   fixes here had values to get right. Each case is one the wrong fix gets wrong, with every operand
   and result small enough for a byte, so no out-of-range conversion is involved on any backend.

   [Min(a + b, c)] over [200 + 100, 250] is 250: C compares the integer-promoted sum, where
   narrowing the operands to [uchar] first, to resolve the overload, compares 44 with 250.

   [two * Sin(a)] over [a = 1] is 1: the sine stays a float until the store, where casting a math
   result back to the integer precision first gives 2 * 0.

   [two * Min(Sin(a), Sin(b))] over [a = b = 1] is 1: the comparison happens in the operands' common
   type, where casting them to an integer type to resolve the overload gives 2 * 0.

   [two * Min(Sin(b), Sqrt(Neg(a)))] over [a = b = 1] is 1: a floating comparison keeps [fmin]'s NaN
   handling, where a plain [<] selects the NaN, an invalid conversion at the byte store. (The mixed
   case, an integer read against a float intermediate, is gh-ocannl-1785's, below.)

   [two * Trunc(Sin(a))] over [a = 1] is 0: an operand at an integer precision can be float-valued,
   so integer [Trunc] is not the identity (that gives 2 * 0.84 = 1.68, stored 1). *)
let () =
  let byte = Ops.byte in
  let mk label = node ~prec:byte ("value_" ^ label) in
  let a = mk "a" and b = mk "b" and c = mk "c" and two = mk "two" in
  let cases =
    [
      ( "byte Min over an integer-promoted sum compares before narrowing",
        LL.Binop (Ops.Min, (LL.Binop (Ops.Add, read a byte, read b byte), byte), read c byte),
        [ (a, 200.); (b, 100.); (c, 250.) ],
        250. );
      ( "byte math over an integer operand keeps its float result until the store",
        LL.Binop (Ops.Mul, read two byte, (LL.Unop (Ops.Sin, read a byte), byte)),
        [ (a, 1.); (two, 2.) ],
        1. );
      ( "byte Min compares float-valued operands as floats",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            ( LL.Binop
                ( Ops.Min,
                  (LL.Unop (Ops.Sin, read a byte), byte),
                  (LL.Unop (Ops.Sin, read b byte), byte) ),
              byte ) ),
        [ (a, 1.); (b, 1.); (two, 2.) ],
        1. );
      ( "byte Min of a float and a NaN intermediate is the float, as fmin's is",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            ( LL.Binop
                ( Ops.Min,
                  (LL.Unop (Ops.Sin, read b byte), byte),
                  (LL.Unop (Ops.Sqrt, (LL.Unop (Ops.Neg, read a byte), byte)), byte) ),
              byte ) ),
        [ (a, 1.); (b, 1.); (two, 2.) ],
        1. );
      ( "byte Trunc truncates a float-valued operand",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            (LL.Unop (Ops.Trunc, (LL.Unop (Ops.Sin, read a byte), byte)), byte) ),
        [ (a, 1.); (two, 2.) ],
        0. );
    ]
  in
  List.iteri cases ~f:(fun i (claim, llsc, seed, want) ->
      let out = mk (Printf.sprintf "out%d" i) in
      let llc = Ll_test.set out at0 llsc in
      let name = Printf.sprintf "optbl_value_%d" i in
      let inputs = List.map seed ~f:fst in
      let got =
        Ll_test.execute ~name
          { (Ll_test.optimize ~materialized:(out :: inputs) ~name llc) with LL.llc }
          ~seed:(List.map seed ~f:(fun (tn, v) -> (tn, [| v |])))
          ~read:[ out ]
      in
      p claim (Ll_test.same got [ [| want |] ]))

(* gh-ocannl-1785: the rule for a float-valued subexpression at an integer compute precision, made
   explicit on every backend. It is computed in float -- through the operators that follow their
   operands, as C's usual arithmetic conversions do -- and converted to the integer precision once:
   at the store or scope holding it, or at an operand of [Mod], whose integer spelling takes
   integers only. Comparisons and [Not] yield a truth value, an integer. Each case below is
   discriminating: its float intermediate has a fractional part a premature conversion would drop,
   or that a different conversion point would keep.

   [Mod(c, Sqrt(a))] over [10, 8] is 0: the divisor converts at the operand, 10 % 2, where a float
   remainder gives fmod(10, 2.83) = 1.51, stored 1 -- and C's [%] over a double does not compile.

   [two * Min(Sin(a), b)] over [a = b = 1] is 1: a float against an integer read compares in float
   (CUDA's integer [min] did not resolve the mixed call), where converting the sine first gives 0.

   [three * Recip(two)] is 1: an integer [Recip] divides in float, 3 * 0.5, where integer division
   gives 0 (refused on cc and at byte on CUDA/HIP, integer division at int32 there, float on Metal).

   [Exp(Sqrt(a))] over [a = 2] is 4: a chain of math stays float, exp(1.41) = 4.11, where converting
   between the two calls gives exp(1) = 2.72, stored 2.

   [two * Where(cond, Sin(a), b)] over [1, 1, 5] is 1: a float-valued select arm keeps the select
   float, 2 * 0.84, where converting the arm gives 0.

   [three * (Not(Sin(a)) / two)] over [a = 0] is 0: [Not] yields the truth value 1, an integer, so 1
   / 2 is integer division, where a float-typed truth gives 3 * 0.5 = 1.5, stored 1.

   int32 [two * Sin(a)] over [a = -1] is -1: the conversion truncates toward zero (-1.68), not to
   the floor (-2). *)
let () =
  let run ~prec ~name claim llsc seed want =
    let out = node ~prec (name ^ "_out") in
    let llc = Ll_test.set out at0 llsc in
    let inputs = List.map seed ~f:fst in
    let got =
      Ll_test.execute ~name
        { (Ll_test.optimize ~materialized:(out :: inputs) ~name llc) with LL.llc }
        ~seed:(List.map seed ~f:(fun (tn, v) -> (tn, [| v |])))
        ~read:[ out ]
    in
    p claim (Ll_test.same got [ [| want |] ])
  in
  let byte = Ops.byte in
  let mk label = node ~prec:byte ("realv_" ^ label) in
  let unop op v = (LL.Unop (op, v), byte) in
  let binop op v1 v2 = (LL.Binop (op, v1, v2), byte) in
  let a = mk "a" and b = mk "b" and c = mk "c" and cond = mk "cond" in
  let two = mk "two" and three = mk "three" in
  run ~prec:byte ~name:"optbl_realv_mod" "byte Mod converts a float-valued operand at the operand"
    (fst (binop Ops.Mod (read c byte) (unop Ops.Sqrt (read a byte))))
    [ (c, 10.); (a, 8.) ]
    0.;
  run ~prec:byte ~name:"optbl_realv_min" "byte Min of a float and an integer compares in float"
    (fst (binop Ops.Mul (read two byte) (binop Ops.Min (unop Ops.Sin (read a byte)) (read b byte))))
    [ (two, 2.); (a, 1.); (b, 1.) ]
    1.;
  run ~prec:byte ~name:"optbl_realv_recip" "byte Recip divides in float"
    (fst (binop Ops.Mul (read three byte) (unop Ops.Recip (read two byte))))
    [ (three, 3.); (two, 2.) ]
    1.;
  run ~prec:byte ~name:"optbl_realv_chain" "byte math over math stays float until the store"
    (fst (unop Ops.Exp (unop Ops.Sqrt (read a byte))))
    [ (a, 2.) ]
    4.;
  run ~prec:byte ~name:"optbl_realv_where" "byte Where with a float-valued arm selects in float"
    (fst
       (binop Ops.Mul (read two byte)
          (LL.Ternop (Ops.Where, read cond byte, unop Ops.Sin (read a byte), read b byte), byte)))
    [ (two, 2.); (cond, 1.); (a, 1.); (b, 5.) ]
    1.;
  run ~prec:byte ~name:"optbl_realv_not" "byte Not of a float-valued operand is an integer truth"
    (fst
       (binop Ops.Mul (read three byte)
          (binop Ops.Div (unop Ops.Not (unop Ops.Sin (read a byte))) (read two byte))))
    [ (three, 3.); (a, 0.); (two, 2.) ]
    0.;
  let int32 = Ops.int32 in
  let i_a = node ~prec:int32 "realv_i_a" and i_two = node ~prec:int32 "realv_i_two" in
  run ~prec:int32 ~name:"optbl_realv_int32" "int32 conversion of a float truncates toward zero"
    (LL.Binop (Ops.Mul, read i_two int32, (LL.Unop (Ops.Sin, read i_a int32), int32)))
    [ (i_two, 2.); (i_a, -1.) ]
    (-1.)

(* gh-ocannl-1785: the rule holds through the optimizer, not only over hand-built trees.

   The constant case runs the SIMPLIFIED code (the cases above replace it with the hand-built tree):
   [Mod(10, Sqrt(8))] over constants folds only where folding agrees with the renderer, so it is 0
   as over runtime inputs, where folding the math in double gives [Float.rem 10 2.83] = 1.51, stored
   1.

   The parity legs execute one program twice, its producer virtualized and then materialized, and
   compare both with the value worked out by hand. Each producer is discriminating, varying with its
   cell and carrying a fractional part a misplaced conversion drops or keeps:

   - a single-precision [s = x + y] read by a byte product [t = s * k]: inlined and collapsed, the
   sum would be computed at byte, [(uchar)0.5 + (uchar)0.75 = 0] where the materialized [s] holds
   1.25, read as 1; - a byte [s = Sqrt(a)] read by a byte product [t = s * k]: inlined and
   collapsed, the root would reach the product unrounded, [3 * 2.83 = 8] where the materialized [s]
   holds 2, so [t = 6]. *)
let () =
  let byte = Ops.byte in
  let out = node ~prec:byte "realv_fold_out" in
  let name = "optbl_realv_fold" in
  let c v = (LL.Constant v, byte) in
  let llc = Ll_test.set out at0 (LL.Binop (Ops.Mod, c 10., (LL.Unop (Ops.Sqrt, c 8.), byte))) in
  let got =
    Ll_test.execute ~name (Ll_test.optimize ~materialized:[ out ] ~name llc) ~seed:[] ~read:[ out ]
  in
  p "byte Mod over folded constants agrees with the renderer" (Ll_test.same got [ [| 0. |] ]);
  let n = 4 in
  let leg ~label ~tag ~producer_prec ~producer ~inputs ~k ~want =
    let fresh ?(prec = byte) suffix =
      Int.incr next_id;
      Tn.create (Tn.Specified prec) ~id:!next_id
        ~label:[ "realv_" ^ tag ^ "_" ^ suffix ]
        ~unpadded_dims:(lazy [| n |])
        ~padding:(lazy None)
        ()
    in
    let s = fresh ~prec:producer_prec "s" and t = fresh "t" in
    let kn = node ~prec:byte ("realv_" ^ tag ^ "_k") in
    let ins =
      List.map inputs ~f:(fun (prec, suffix, values) -> (fresh ~prec suffix, prec, values))
    in
    List.iter ins ~f:(fun (tn, _, _) -> Ll_test.materialize tn);
    Ll_test.materialize t;
    let i = Ll_test.sym () in
    let at_i = [| Idx.Iterator i |] in
    let reads = List.map ins ~f:(fun (tn, prec, _) -> (LL.Get (tn, at_i), prec)) in
    let llc =
      Ll_test.loop_n i n
        (LL.Seq
           ( Ll_test.set s at_i (producer reads),
             Ll_test.set t at_i
               (LL.Binop (Ops.Mul, (LL.Get (s, at_i), producer_prec), read kn byte)) ))
    in
    let seed = (kn, [| k |]) :: List.map ins ~f:(fun (tn, _, values) -> (tn, values)) in
    let materialized = t :: kn :: List.map ins ~f:(fun (tn, _, _) -> tn) in
    let run ~name ~materialized =
      let o = Ll_test.optimize ~materialized ~name llc in
      (o, Ll_test.execute ~name o ~seed ~read:[ t ])
    in
    let inlined_o, inlined = run ~name:("optbl_realv_" ^ tag ^ "_inl") ~materialized in
    let _, materialized_run =
      run ~name:("optbl_realv_" ^ tag ^ "_mat") ~materialized:(s :: materialized)
    in
    p (label ^ ": the producer was virtualized") (Ll_test.known_virtual inlined_o s);
    p
      (label ^ ": the materialized run gives the worked value")
      (Ll_test.same materialized_run [ want ]);
    p
      (label ^ ": the virtualized run agrees with the materialized one")
      (Ll_test.same inlined [ want ])
  in
  leg ~label:"float sum under a byte product" ~tag:"sum" ~producer_prec:Ops.single
    ~producer:(function [ x; y ] -> LL.Binop (Ops.Add, x, y) | _ -> assert false)
    ~inputs:
      [
        (Ops.single, "x", [| 0.5; 1.5; 2.25; 3.5 |]); (Ops.single, "y", [| 0.75; 0.75; 0.5; 0.25 |]);
      ]
    ~k:4. ~want:[| 4.; 8.; 8.; 12. |];
  leg ~label:"byte root under a byte product" ~tag:"root" ~producer_prec:byte
    ~producer:(function [ a ] -> LL.Unop (Ops.Sqrt, a) | _ -> assert false)
    ~inputs:[ (byte, "a", [| 8.; 15.; 3.; 24. |]) ]
    ~k:3. ~want:[| 6.; 9.; 3.; 12. |]

(* gh-ocannl-1883: the integer table arms are integer-typed, so their values are exact and an
   integer-only consumer takes them. Every case runs at each integer precision the backend holds,
   over operands and results small enough for a byte, and is exact by construction.

   Each arm under [Mod], over [a = 7, b = 5, c = 4, z = 0]: [Max] 7 % 4 = 3, [Min] 5 % 4 = 1, [Relu]
   7 % 4 = 3, [Satur01] 1 % 4 = 1, [Trunc] 7 % 4 = 3, [Not(z)] 1 % 4 = 1. On cc these did not
   compile ([%] over [fmax]'s double); Metal's [Satur01] and [Trunc] were floats too.

   [b * (Not(z) / c)] over [z = 0, c = 4] is 0: [Not] is an integer truth, so 1 / 4 is integer
   division, where cc's [? 1.0 : 0.0] divided in double, 5 * 0.25 = 1.25, stored 1.

   The wide cases have operands a float-typed arm rounds. int32 [Trunc(k * k + one) - k * k] over [k
   = 4096] is 1: [truncf] of 2^24 + 1 is 2^24. int64 [Max(m * m + one, m * m) - m * m] and [Relu(m *
   m + one) - m * m] over [m = 2^27] are 1: [fmax] of 2^54 + 1 is 2^54. The operands are built in
   the kernel because a seed is a float. *)
let () =
  let run ~prec ~name claim llsc seed want =
    let out = node ~prec (name ^ "_out") in
    let llc = Ll_test.set out at0 llsc in
    let inputs = List.map seed ~f:fst in
    let got =
      Ll_test.execute ~name
        { (Ll_test.optimize ~materialized:(out :: inputs) ~name llc) with LL.llc }
        ~seed:(List.map seed ~f:(fun (tn, v) -> (tn, [| v |])))
        ~read:[ out ]
    in
    p claim (Ll_test.same got [ [| want |] ])
  in
  List.iter Ops.integer_precs ~f:(fun prec ->
      let tag = Ops.prec_string prec in
      let mk label = node ~prec (Printf.sprintf "intv_%s_%s" tag label) in
      let a = mk "a" and b = mk "b" and c = mk "c" and z = mk "z" in
      let r tn = read tn prec in
      let unop op v = (LL.Unop (op, v), prec) in
      let binop op v1 v2 = (LL.Binop (op, v1, v2), prec) in
      let seed = [ (a, 7.); (b, 5.); (c, 4.); (z, 0.) ] in
      List.iter
        [
          ("max", "Max", binop Ops.Max (r a) (r b), 3.);
          ("min", "Min", binop Ops.Min (r a) (r b), 1.);
          ("relu", "Relu", unop Ops.Relu (r a), 3.);
          ("sat01", "Satur01", unop Ops.Satur01 (r a), 1.);
          ("trunc", "Trunc", unop Ops.Trunc (r a), 3.);
          ("not", "Not", unop Ops.Not (r z), 1.);
        ]
        ~f:(fun (short, op, arm, want) ->
          run ~prec
            ~name:(Printf.sprintf "optbl_intv_%s_mod_%s" tag short)
            (Printf.sprintf "%s %s under Mod is an integer" tag op)
            (fst (binop Ops.Mod arm (r c)))
            seed want);
      run ~prec
        ~name:(Printf.sprintf "optbl_intv_%s_not_div" tag)
        (Printf.sprintf "%s Not is an integer truth, dividing as an integer" tag)
        (fst (binop Ops.Mul (r b) (binop Ops.Div (unop Ops.Not (r z)) (r c))))
        seed 0.);
  let wide ~prec ~base =
    let tag = Ops.prec_string prec in
    let k = node ~prec (Printf.sprintf "intv_%s_k" tag) in
    let one = node ~prec (Printf.sprintf "intv_%s_one" tag) in
    let r tn = read tn prec in
    let binop op v1 v2 = (LL.Binop (op, v1, v2), prec) in
    let square = binop Ops.Mul (r k) (r k) in
    let above = binop Ops.Add square (r one) in
    let seed = [ (k, base); (one, 1.) ] in
    fun short claim arm ->
      run ~prec
        ~name:(Printf.sprintf "optbl_intv_%s_wide_%s" tag short)
        claim
        (fst (binop Ops.Sub (arm ~square ~above) square))
        seed 1.
  in
  let wide32 = wide ~prec:Ops.int32 ~base:4096. in
  wide32 "trunc" "int32 Trunc is exact above 2^24" (fun ~square:_ ~above ->
      (LL.Unop (Ops.Trunc, above), Ops.int32));
  let wide64 = wide ~prec:Ops.int64 ~base:134217728. in
  wide64 "max" "int64 Max is exact above 2^53" (fun ~square ~above ->
      (LL.Binop (Ops.Max, above, square), Ops.int64));
  wide64 "relu" "int64 Relu is exact above 2^53" (fun ~square:_ ~above ->
      (LL.Unop (Ops.Relu, above), Ops.int64))

(* gh-ocannl-1768: what the parameter guards cover in the two tables they guard. An entry the guard
   cannot read is either not a function -- a macro, a type, a constant, which have no parameter
   types -- or a function it skips, which has to be named here with why. *)
let unguarded_functions =
  [
    ( ("cc", "OCANNL_CHECKED_LANE"),
      "the precision's name passed as a string, for the abort's diagnostic (gh-ocannl-1867); only \
       rendered under debug_check_lane_bounds, around an index-precision lane operand" );
    ( ("metal", "ocannl_max"),
      "a template: integer Max in the operands' common type, which it is instantiated at" );
    ( ("metal", "ocannl_min"),
      "a template: integer Min in the operands' common type, which it is instantiated at" );
    ( ("metal", "threefry_round"),
      "its state passed by reference, internal to the threefry functions, which are guarded" );
  ]

let () =
  let not_a_function why =
    String.equal why Guards.a_macro || String.equal why Guards.not_a_function
  in
  let skipped =
    List.concat_map
      [ ("cc", Context.Builtins_cc.builtins); ("metal", Builtins_metal.builtins) ]
      ~f:(fun (table, entries) ->
        List.filter_map entries ~f:(fun (key, definition, _) ->
            match Guards.parameter_types ~key definition with
            | Ok _ -> None
            | Error why when not_a_function why -> None
            | Error why ->
                Stdio.eprintf "  %s's %s is unguarded: %s\n" table key why;
                Some (table, key)))
  in
  let named = List.map unguarded_functions ~f:fst in
  let equal (t1, k1) (t2, k2) = String.equal t1 t2 && String.equal k1 k2 in
  p_all "every function of the cc and Metal tables is guarded or named as skipped" skipped
    ~f:(List.mem named ~equal);
  p_all ~min:1 "every function named as skipped is one the guards skip" named
    ~f:(List.mem skipped ~equal)
