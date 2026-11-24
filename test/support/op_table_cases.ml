(* The operator domain and hand-built IR shared by the device compiler sweep and the CUDA-like host
   guard preflight. No arm list is maintained in the host test. *)
open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode

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
  | Under_integer of Ops.binop * [ `Left | `Right ] * arm
      (** A homogeneous arithmetic arm at an integer storage precision as either operand of [Mod] or
          [Div]. An integer-VALUED float would fail to compile under [Mod] and silently change the
          quotient under [Div] (gh-ocannl-1883). *)
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
  | Under_integer (op, side, arm) ->
      let side = match side with `Left -> "left" | `Right -> "right" in
      arm_name arm ^ " as the " ^ side ^ " operand of " ^ Sexp.to_string (Ops.sexp_of_binop op)
  | Lanes arm -> arm_name arm ^ " over vector lanes"
  | Lanes_uniform arm -> arm_name arm ^ " over vector lanes, its last operand uniform"
  | Reduce op -> "vector reduction " ^ Sexp.to_string (Ops.sexp_of_binop op)
  | Reduce_fma -> "vector fma reduction"

let is_vector_arm = function
  | Lanes _ | Lanes_uniform _ | Reduce _ | Reduce_fma -> true
  | Unary _ | Binary _ | Ternary _ | Vector _ | Convert_from _ | Inlined _ | Inlined_real _
  | Under_integer _ ->
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
  | Inlined arm | Inlined_real arm | Under_integer (_, _, arm) -> is_select arm
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
      | Vector _ | Convert_from _ | Inlined _ | Inlined_real _ | Under_integer _ | Lanes _
      | Lanes_uniform _ | Reduce _ | Reduce_fma ->
          false
    in
    arithmetic
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm then Some (Inlined arm) else None)
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm && Ops.is_integer store then Some (Inlined_real arm) else None)
    @ (if Ops.is_integer store then
         List.concat_map (List.filter arithmetic ~f:homogeneous) ~f:(fun arm ->
             List.concat_map [ Ops.Mod; Ops.Div ] ~f:(fun op ->
                 List.map [ `Left; `Right ] ~f:(fun side -> Under_integer (op, side, arm))))
       else [])
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
  | Under_integer (consumer, side, arm) -> (
      let consume value =
        let arg = (value, s) and other = read o.c s in
        let left, right = match side with `Left -> (arg, other) | `Right -> (other, arg) in
        store (LL.Binop (consumer, left, right))
      in
      match arm with
      | Unary op -> consume (LL.Unop (op, read o.a s))
      | Binary op -> consume (LL.Binop (op, read o.a s, read o.b s))
      | Ternary op -> consume (LL.Ternop (op, read o.a s, read o.b s, read o.c s))
      | _ -> statement o k arm)
  | Unary op when Ops.is_homogeneous_prec_unop op -> store (LL.Unop (op, read o.a s))
  | Unary Ops.Round_prec -> store (LL.Unop (Ops.Round_prec, read o.a s))
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

let rec c_reference_refuses ~store ~compute arm =
  let raises f = match f () with _ -> false | exception Invalid_argument _ -> true in
  match arm with
  | Unary op -> raises (fun () -> Ops.unop_c_syntax compute op)
  | Binary op -> raises (fun () -> Ops.binop_c_syntax compute op)
  | Ternary Ops.FMA -> not (Ops.is_float compute)
  | Ternary op -> raises (fun () -> Ops.ternop_c_syntax compute op)
  | Vector op -> raises (fun () -> Ops.vec_unop_c_syntax compute op)
  | Convert_from from -> raises (fun () -> Ops.c_convert_precision ~from ~to_:store)
  | Inlined arm | Inlined_real arm | Under_integer (_, _, arm) | Lanes arm | Lanes_uniform arm ->
      c_reference_refuses ~store ~compute arm
  | Reduce op -> c_reference_refuses ~store ~compute (Binary op)
  | Reduce_fma -> c_reference_refuses ~store ~compute (Ternary Ops.FMA)

(* The same planted half-by-value defect is used by the host and both device compiler legs. *)
let reverted_1704_cuda_like =
  "__device__ uint4x32_t half_to_uint4x32(unsigned short x) { uint4x32_t r = {{x, 0, 0, 0}}; \
   return r; }"
