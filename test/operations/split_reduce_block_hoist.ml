(* gh-ocannl-1412: a split reduction's first pass runs its block loop OUTSIDE the cell loops, so the
   GPU presets put the lanes on the cell axis.

   [Sched.Split_reduce] puts the block loop where the reduction loop was, below the cell loops its
   pinning discipline requires: [for o: for blk: { init; for inner: for s: .. }]. The default GPU
   preset maps the outer loop of such a chain to the grid and the inner one to the lanes, so the
   lanes walked the blocks, [chunk] rows apart, and every lane of a wavefront loaded its own cache
   line on every row. On gpt2_mini_train b256 that made the untuned split 1.42x SLOWER on HIP's
   unified-memory gfx1151, where the first passes of the bias gradients cost about 400 ms of the
   step. [Autotune.split_reduce_sites] now records the cell loops ([sr_block_hoist]) and
   [Autotune.split_reduce_prelude] interchanges the block loop above them.

   Covered here, on every backend, for a bias gradient [for b, s, o: g[o] += dy[b,s,o]] over 8192
   rows: - Detection: the site records exactly its cell loop. - Structure: in the prelude's code the
   accumulation into the partials sits under the block loop first, then the cell loop; the same site
   with an empty [sr_block_hoist] (the control) keeps the cell loop outermost. - Lanes (GPU
   backends): under the default GPU preset with this device's limits, the first pass's lane loops
   index the partials' cell component and not its block component; the control's lanes index the
   block component. - Executed parity: the hoisted split computes bitwise the unhoisted split's
   gradient (the interchange keeps each cell's summation order) and both stay within an f32
   tolerance of an f64 host reference. The inputs are not dyadic, so a hoist that reordered a cell's
   contributions would move bits. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module L = Ll_test
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

let nb = 128
and ns = 64
and no = 64

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let b =
  Operation.init ~l:"sbh_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let dy_values = values ~n:(nb * ns * no) ~stride:577

let x =
  NTDSL.ndarray
    (values ~n:(nb * ns * no) ~stride:389)
    ~label:[ "sbh_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

let dy = NTDSL.ndarray dy_values ~label:[ "sbh_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
let%op y = x + b
let%op loss = (y *. dy) ++ "... => 0"
let g = (Option.value_exn ~here:[%here] b.Tensor.diff).Tensor.grad
let () = Train.set_materialized g

let comp =
  let c = Train.grad_update loss in
  { c with asgns = Ir.Assignments.Block_comment ("sbh", c.asgns) }

(* The bias gradient in f64 on the host: every cell sums the f32-rounded [dy] over the rows. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let exact =
  Array.init no ~f:(fun o ->
      let acc = ref 0. in
      for r = 0 to (nb * ns) - 1 do
        acc := !acc +. f32 dy_values.((r * no) + o)
      done;
      !acc)

(* Either summation order lands far inside this on the fixture; one dropped or doubled block of 256
   rows moves a cell by orders of magnitude more. *)
let tolerance = 1e-3
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let is_partials (tn : Ir.Tnode.t) = List.exists tn.Ir.Tnode.label ~f:(String.equal "partials")

(* The loops enclosing the first pass's accumulation into the partials, outermost first, with their
   axis types, and that write's index; [None] when the code has no such write. *)
let first_pass (llc : LL.t) =
  let stack = ref [] and found = ref None in
  L.walk llc
    ~on_stmt:(fun s ->
      match s with
      | LL.For_loop { index; axis; _ } -> stack := (index, axis) :: !stack
      | LL.Set { tn; idcs; llsc; _ } when Option.is_none !found && is_partials tn ->
          let reads = ref false in
          L.walk_scalar llsc ~on_get:(fun t -> if Ir.Tnode.equal t tn then reads := true);
          if !reads then found := Some (List.rev !stack, idcs)
      | _ -> ())
    ~on_exit:(fun s -> match s with LL.For_loop _ -> stack := List.tl_exn !stack | _ -> ());
  !found

let capture () =
  let stash = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !stash

let limits = Context.hardware_limits (Context.auto ())

let preset opt =
  if is_gpu then Sched.default_gpu ~limits opt
  else if is_cpu then Sched.default_cpu ~min_work:0 opt
  else []

let zero_sched _tns = []

(* The fissioned, scheduled segments of the code [site]'s prelude produces, as the search's
   [F_split] candidates compile it. *)
let split_segments opt site =
  let opt = Sched.apply (Autotune.split_reduce_prelude [ (site, 32) ]) opt in
  let _, tuples = Sched.fission_segmented ~preset ~zero_sched ~static_indices:[] opt in
  List.map tuples ~f:(fun (_, _, _, post) -> post)

let run_split site =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> split_segments opt site)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  Context.get_values ctx g

let () =
  let opt = capture () in
  match
    List.find (Autotune.split_reduce_sites opt) ~f:(fun s -> Ir.Tnode.equal s.Autotune.sr_target g)
  with
  | None -> p "the bias gradient is a split-reduce site" false
  | Some site ->
      p "the bias gradient is a split-reduce site" true;
      let control = { site with Autotune.sr_block_hoist = [] } in
      (* The one cell loop: the loop the gradient's write indexes, outside the split loop. *)
      p "the site records one cell loop to hoist the block loop above"
        (List.length site.sr_block_hoist = 1);
      let structure s =
        first_pass (Sched.apply (Autotune.split_reduce_prelude [ (s, 32) ]) opt).LL.llc
      in
      let mentions idx (sym, _) = Idx.axis_index_mentions_symbol sym idx in
      (match (structure site, structure control) with
      | Some (outer :: next :: _, idcs), Some (c_outer :: c_next :: _, c_idcs) ->
          p "hoisted: the first pass's outermost loop is its block loop" (mentions idcs.(0) outer);
          p "hoisted: the cell loop runs directly inside it" (mentions idcs.(1) next);
          p "control: the cell loop stays outermost" (mentions c_idcs.(1) c_outer);
          p "control: the block loop runs inside it" (mentions c_idcs.(0) c_next)
      | _ -> p "both preludes leave a first pass of two or more loops" false);
      (* The lanes, under this device's default GPU preset. *)
      let lanes s =
        List.find_map (split_segments opt s) ~f:(fun post -> first_pass post.LL.llc)
        |> Option.map ~f:(fun (loops, idcs) ->
            (List.filter loops ~f:(fun (_, ty) -> LL.equal_axis_type ty LL.Workgroup), idcs))
      in
      let lane_claim ~what ~component s =
        Verdict.gated ~when_:is_gpu ~on:backend_name what
          (match lanes s with
          | Some ((_ :: _ as wg), idcs) ->
              List.exists wg ~f:(mentions idcs.(component))
              && not (List.exists wg ~f:(mentions idcs.(1 - component)))
          | _ -> false)
      in
      lane_claim ~component:1 site
        ~what:"hoisted: the first pass's lanes walk the cell axis, not the blocks";
      lane_claim ~component:0 control
        ~what:"control: the unhoisted first pass's lanes walk the blocks, not the cell axis";
      let hoisted = run_split site and unhoisted = run_split control in
      Stdio.eprintf "max error vs f64: hoisted %g (not part of the golden)\n"
        (Array.foldi hoisted ~init:0. ~f:(fun o m v -> Float.max m (Float.abs (v -. exact.(o)))));
      p_exists "the bias gradient is not trivially zero" (Array.to_list exact) ~f:(fun v ->
          Float.(abs v > 1.));
      p_alli "the hoisted split's gradient is within the f32 tolerance of the f64 reference"
        (Array.to_list hoisted) ~f:(fun o v -> Float.(abs (v - exact.(o)) <= tolerance));
      p_all2 "the hoisted split computes bitwise the unhoisted split's gradient" hoisted unhoisted
        ~f:bitwise
