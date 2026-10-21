(* gh-ocannl-1474: the [Rev_sides] scatter (a [++^] concatenation's gradient) establishes the
   margins of every padded target it WRITES, on both of its paths, and leaves the node it READS
   alone.

   The audit question was whether a padded [Rev_sides] target can be built at all. It can, from
   ordinary [%op] code: a gradient shares its value's shape, so a concatenation operand that a
   padded convolution also consumes has a padded gradient, and that gradient is a scatter target.
   Below, [x] is concatenated with [v] and also read by [y = x +* "s, o=+k; k => s, o" kernel]; the
   concatenation [cat] is itself read by a padded convolution [z], so the node the scatter reads
   ([cat]'s gradient) is padded too.

   The padding contract ([Assignments.loop_accum]) is that every writer of a padded node carries the
   idempotent fill of its margins, because device buffers are allocated raw. The scatter used to
   reset the margins of the node it reads instead -- a node [Assignments.collect_written] says it
   does not write -- and, on the non-initializing path every backprop takes, none of its targets'.
   That was harmless only through the target's own initializer ([zero_grads]' [Fetch] fills the
   whole buffer): nothing in the scatter kept the contract.

   Executed legs: - The scatter as the front end emits it (non-initializing), lowered and run alone
   over PHYSICAL buffers seeded with a per-cell sentinel everywhere, margins included: [x]'s
   gradient holds the committed padding value in its margins and sentinel-plus-slice in its semantic
   cells; the read side keeps every sentinel. On the base commit the target's margins kept their
   sentinels and the read side's were overwritten. - The same record flipped to initializing, the
   flavour whose elided init used to carry the target's margin reset itself: margins hold the
   padding value, semantic cells the slice. - A whole training step: every gradient's semantic cells
   equal the oracle computed here from the definitions of both convolutions and the concatenation.

   Not claimed: what a gradient's margins hold after a whole backward pass. A padded window's
   transpose -- here [y]'s and [z]'s backward -- writes the contributions of the padding positions
   into the operand gradient's margins, after its own resets, so those margins hold that data, not
   the padding value, whatever the scatter does; nothing reads a gradient through a window
   (docs/agent-notes/lowering-and-analysis.md). *)

open Base
open Ocannl
open Operation.DSL_modules
module Asgns = Ir.Assignments
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
open Verdict.Claims

let rows = 2
let cols = 4
let extra = 2
let kernel_vals = [| 1.; 2.; 4. |]

let rec find_rev_sides = function
  | Asgns.Accum_op { rhs = Asgns.Rev_sides _; _ } as a -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_rev_sides a) (find_rev_sides b)
  | Asgns.Block_comment (_, body) -> find_rev_sides body
  | Asgns.Accum_op _ | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

let ndarray ~grad dims values =
  Tensor.ndarray
    ~grad_spec:(if grad then Tensor.Require_grad else Tensor.Prohibit_grad)
    values ~batch_dims:[] ~input_dims:[] ~output_dims:dims ()

let src_vals = Array.init (rows * cols) ~f:(fun i -> Float.of_int (i + 1))
let v_vals = Array.init (rows * extra) ~f:(fun i -> Float.of_int (20 + i))
let wy_vals = Array.init (rows * cols) ~f:(fun i -> Float.of_int (100 * (i + 1)))
let wz_vals = Array.init (rows * (cols + extra)) ~f:(fun i -> Float.of_int (10000 * (i + 1)))
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let build () =
  let src = ndarray ~grad:true [ rows; cols ] src_vals in
  let v = ndarray ~grad:true [ rows; extra ] v_vals in
  let kernel = ndarray ~grad:false [ 3 ] kernel_vals in
  let wy = ndarray ~grad:false [ rows; cols ] wy_vals in
  let wz = ndarray ~grad:false [ rows; cols + extra ] wz_vals in
  (* [x] is computed, so its layout is not committed at creation and the padded convolution can
     grant it margins (padding_lifecycle). *)
  let%op x = src ++ "s, c => s, c" in
  let%op y = x +* "s, o=+k; k => s, o" kernel in
  let%op cat = (x, v) ++^ "s, c; s, d => s, c^d" in
  let%op z = cat +* "s, o=+k; k => s, o" kernel in
  let%op loss = (y *. wy) ++ "...|... => |->0" + ((z *. wz) ++ "...|... => |->0") in
  (src, x, v, cat, loss)

(* The padding of axis 1 (the convolved one), [(left, right)]; axis 0 is unpadded. *)
let margins tn =
  match Tn.get_padding tn with
  | Some (pads, value) ->
      let Ir.Ops.{ left; right } = pads.(1) in
      Some (left, right, value, pads.(0).Ir.Ops.left + pads.(0).Ir.Ops.right)
  | None -> None

let physical_width tn = (Lazy.force tn.Tn.dims).(1)

(* Whether physical cell [(s, j)] of [tn] is a margin cell. *)
let is_margin tn j =
  match margins tn with
  | Some (left, right, _, _) -> j < left || j >= physical_width tn - right
  | None -> false

let sentinel ~base i = -.(base +. Float.of_int i)

(* Uploads [cells] as [tn]'s whole physical buffer, margins included. *)
let upload_physical ctx tn cells =
  let dims = Lazy.force tn.Tn.dims in
  let nd = Nd.create_array ~debug:(Tn.debug_name tn) Ir.Ops.single ~dims ~padding:None in
  let width = dims.(1) in
  Array.iteri cells ~f:(fun i value -> Nd.set_from_float nd [| i / width; i % width |] value);
  Context.from_host ctx tn nd

let num_physical tn = Array.fold (Lazy.force tn.Tn.dims) ~init:1 ~f:( * )
let physical ctx tn = Nd.retrieve_flat_values (Context.to_host ctx tn)
let left_margin tn = Option.value_map (margins tn) ~default:0 ~f:(fun (left, _, _, _) -> left)

(* The semantic cells of a physical [tn] buffer, row-major. *)
let semantic tn phys =
  let width = physical_width tn in
  Array.filteri phys ~f:(fun i _ -> not (is_margin tn (i % width)))

(* [cat]'s gradient as seeded: one value per semantic cell, distinct from every sentinel. *)
let gcat_semantic = Array.init (rows * (cols + extra)) ~f:(fun i -> Float.of_int (1 + i))

let scatter_legs () =
  Tensor.unsafe_reinitialize ();
  let _src, x, v, cat, _loss = build () in
  let gx = grad_of x and gv = grad_of v and gcat = grad_of cat in
  let non_initializing, initializing =
    match find_rev_sides (Option.value_exn ~here:[%here] cat.Tensor.diff).Tensor.backprop.asgns with
    | Some (Asgns.Accum_op r as a) ->
        p "the concatenation's gradient scatter is emitted non-initializing"
          (not r.initialize_neutral);
        (a, Asgns.Accum_op { r with initialize_neutral = true })
    | _ -> failwith "expected the concatenation's Rev_sides gradient"
  in
  (* The audit's premise, from the front end: the scatter's target and its read side are padded,
     along the convolved axis only, and commit the convolution's neutral element. *)
  List.iter
    [ ("x's gradient (a scatter target)", gx); ("cat's gradient (the scatter's read side)", gcat) ]
    ~f:(fun (name, tn) ->
      p
        (name ^ " is padded along the convolved axis, committing 0")
        (match margins tn with
        | Some (left, right, value, other_axis) ->
            left > 0 && right > 0 && other_axis = 0 && Float.equal value 0.
        | None -> false));
  p "v's gradient (the other target) is unpadded" (Option.is_none (Tn.get_padding gv));
  let constant_stores tn llc =
    Ll_test.count_stmt llc ~f:(function
      | LL.Set { tn = t; llsc = LL.Constant _; _ } -> Tn.equal t tn
      | _ -> false)
  in
  let writes tn llc =
    Ll_test.count_stmt llc ~f:(function LL.Set { tn = t; _ } -> Tn.equal t tn | _ -> false)
  in
  List.iter [ gx; gv; gcat ] ~f:Ll_test.materialize;
  let materialized = [ gx; gv; gcat ] in
  let run ~name record =
    let llc = Asgns.to_low_level record in
    (* Structural: the margins of the padded target are stored, the read side is never written. *)
    p
      (name ^ ": the scatter stores constants into x's gradient (its margin resets)")
      (constant_stores gx llc > 0);
    p (name ^ ": the scatter writes no cell of the node it reads") (writes gcat llc = 0);
    let ctx, routine = Ll_test.link ~name (Ll_test.optimize ~materialized ~name llc) in
    let gx_seeded = Array.init (num_physical gx) ~f:(sentinel ~base:100.) in
    (* The read side: [gcat_semantic] in its semantic cells, a sentinel in each margin cell. *)
    let gcat_seeded =
      let width = physical_width gcat and left = left_margin gcat in
      Array.init (num_physical gcat) ~f:(fun i ->
          let s = i / width and j = i % width in
          if is_margin gcat j then sentinel ~base:500. i
          else gcat_semantic.((s * (cols + extra)) + j - left))
    in
    let gv_prior = Array.init (rows * extra) ~f:(fun i -> -1. -. Float.of_int i) in
    let ctx = upload_physical ctx gx gx_seeded in
    let ctx = upload_physical ctx gcat gcat_seeded in
    let ctx = Context.set_values ctx gv gv_prior in
    (Context.run ctx routine, gx_seeded, gcat_seeded, gv_prior)
  in
  let gcat_slice s c = gcat_semantic.((s * (cols + extra)) + c) in
  (* {1 Non-initializing: accumulate into what the target held.} *)
  let ctx, gx_seeded, gcat_physical, gv_prior = run ~name:"non_initializing" non_initializing in
  let gx_after = physical ctx gx in
  let width = physical_width gx in
  let left = left_margin gx in
  pf_all2 "non-initializing: x's gradient holds 0 in its margins and sentinel + slice elsewhere"
    gx_after
    (Array.mapi gx_seeded ~f:(fun i seeded ->
         let s = i / width and j = i % width in
         if is_margin gx j then 0. else seeded +. gcat_slice s (j - left)))
    ~f:Float.equal;
  pf_all2 "non-initializing: the read side keeps its sentinel margins and its values"
    (physical ctx gcat) gcat_physical ~f:Float.equal;
  pf_all2 "non-initializing: v's gradient is its prior plus its slice" (Context.get_values ctx gv)
    (Array.mapi gv_prior ~f:(fun i prior -> prior +. gcat_slice (i / extra) (cols + (i % extra))))
    ~f:Float.equal;
  (* {1 Initializing: the targets' cells are stored, not accumulated.} *)
  let ctx, _, gcat_physical, _ = run ~name:"initializing" initializing in
  pf_all2 "initializing: x's gradient holds 0 in its margins and its slice elsewhere"
    (physical ctx gx)
    (Array.init (rows * width) ~f:(fun i ->
         let s = i / width and j = i % width in
         if is_margin gx j then 0. else gcat_slice s (j - left)))
    ~f:Float.equal;
  pf_all2 "initializing: the read side keeps its sentinel margins and its values"
    (physical ctx gcat) gcat_physical ~f:Float.equal

(* {1 A whole training step, against the definitions.} With a padded convolution [out[s, o] = sum_k
   in[s, o + k - left] * kernel[k]] (out-of-range reads are the padding value 0) and loss [sum (y *.
   wy) + sum (z *. wz)]: the upstream gradient of a convolution's input cell [c] is the sum over
   [(o, k)] with [o + k - left = c] of [w[s, o] * kernel[k]]; [x]'s gradient adds [cat]'s gradient
   at [c], [v]'s is [cat]'s at [cols + d], and [src]'s is [x]'s. *)
let training_leg () =
  Tensor.unsafe_reinitialize ();
  let src, x, v, cat, loss = build () in
  let gsrc = grad_of src and gx = grad_of x and gv = grad_of v and gcat = grad_of cat in
  List.iter [ gsrc; gx; gv; gcat ] ~f:Train.set_materialized;
  let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
  let conv_input_grad ~left ~weights ~out_width s c =
    let acc = ref 0. in
    for o = 0 to out_width - 1 do
      for k = 0 to 2 do
        if o + k - left = c then acc := !acc +. (weights.((s * out_width) + o) *. kernel_vals.(k))
      done
    done;
    !acc
  in
  let gcat_oracle s j =
    conv_input_grad ~left:(left_margin gcat) ~weights:wz_vals ~out_width:(cols + extra) s j
  in
  let gx_oracle s c =
    conv_input_grad ~left:(left_margin gx) ~weights:wy_vals ~out_width:cols s c +. gcat_oracle s c
  in
  let semantic_cells tn = semantic tn (physical ctx tn) in
  pf_all2 "training step: cat's gradient is z's input gradient" (semantic_cells gcat)
    (Array.init
       (rows * (cols + extra))
       ~f:(fun i -> gcat_oracle (i / (cols + extra)) (i % (cols + extra))))
    ~f:Float.equal;
  pf_all2 "training step: x's gradient sums y's input gradient and its slice of cat's"
    (semantic_cells gx)
    (Array.init (rows * cols) ~f:(fun i -> gx_oracle (i / cols) (i % cols)))
    ~f:Float.equal;
  pf_all2 "training step: v's gradient is its slice of cat's" (Context.get_values ctx gv)
    (Array.init (rows * extra) ~f:(fun i -> gcat_oracle (i / extra) (cols + (i % extra))))
    ~f:Float.equal;
  pf_all2 "training step: src's gradient is x's" (Context.get_values ctx gsrc)
    (Array.init (rows * cols) ~f:(fun i -> gx_oracle (i / cols) (i % cols)))
    ~f:Float.equal

let () =
  scatter_legs ();
  training_leg ()
