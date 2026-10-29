(* gh-ocannl-1712: every seeded schedule's [Fuse_epilogue] twin, executed against the unfused
   reference, on sites whose seeds produce single-trip loops and partitioned segments.

   [Fuse_epilogue] relocates an elementwise tail to right after the accumulator's store-back, so it
   is sound only where that store-back runs exactly once per output cell, after the whole reduction.
   Its exactly-once check looks for an enclosing loop of extent > 1 that does not index the site; a
   schedule that copies the reduction into several single-trip loops hides the second write from it.
   gh-ocannl-1670 (staging#1476) found the first such producer: the packed seeds' k-block
   [Partition] with ONE full k block leaves both segments' k-block loops single-trip, the twin fused
   after the first segment's store-back, and its tail read the first k block alone
   (transformer_names dropped half its vocabulary, with every loss claim still holding). That PR
   made [fuse_at_fragment] refuse any write of the accumulator after the fused site. This sweep pins
   the class, not the instance: on each site below, the fused twin of EVERY whole-routine seed the
   search would propose ([Autotune.search_seeding]'s [ss_whole], sketch and coalesced, each seed's
   flag-flipped twin, which is how the family tree mints its fused flavor) is compiled and run, and
   every twin that is admitted must equal the untuned two-kernel run; every other one must be
   refused typed (a crash is not a quiet exclusion from the population).

   The sites are chosen for their single-trip producers: one full k block and an edge (the k-split,
   batched as in transformer_names and not), four full blocks and an edge, k exactly one block (a
   single-trip k-block loop with no Partition), k below one block (a padded single block), one full
   row block and an edge, one full column block and an edge, and two convolutions (a 3x3 window, and
   a 1x1 window whose kernel loops are single-trip).

   Inputs are DISCRIMINATING and EXACT. Every operand varies with each of its axes
   ([Ll_test.cycle_flat] refuses a blind axis), so a tail computed from part of the reduction
   differs from the full one; and every value is a small dyadic, every sum stays far below 2^24, so
   the f32 accumulation is exact in any order, contracted or not, on any backend and at any packing
   precision a seed picks (f16 and bf16 hold the operands exactly). The comparison is therefore
   BITWISE everywhere, with no tolerance for a dropped block to hide in. The tails are plain sums
   (no relu but one) so no clamp masks a difference.

   A point-in-time check of whole-routine seeds on these graphs: per-segment candidates of a fission
   search are bench_seed_parity's (gh-ocannl-1490), and beam moves applied after a fused twin
   operate on already-fused code. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Zeros compare equal to zeros: a reference that collapsed to all zeros would let every parity
   claim pass while covering nothing. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let has_partition sched = List.exists sched ~f:(function Sched.Partition _ -> true | _ -> false)

type outcome =
  | Ran of float array  (** Admitted: the readback. *)
  | Refused of string
      (** A typed refusal ([Invalid_argument], what [Context.compile] turns a schedule's typed cause
          into): the candidate a search would have counted as declined. *)
  | Crashed of string  (** Any other exception: never a quiet exclusion from the population. *)

type twin = {
  partitioned : bool;  (** The twin's schedule carries a [Partition]. *)
  outcome : outcome;
}

(* Every twin of every site, for the cross-site claims. *)
let all_twins : twin list ref = ref []

(* One site: [build ()] makes the graph and returns the reduction output (materialized, the fusion
   target) and the tail's output. *)
let sweep tag build =
  Tensor.unsafe_reinitialize ();
  let z, y = build () in
  Train.set_materialized z.Tensor.value;
  let fwd = named ("ep1712_" ^ tag) (Train.forward y) in
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let capture = ref None in
  (* The untuned compile doubles as the reference: two kernels, the tail after the whole
     reduction. *)
  let rctx, rroutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        [ opt ])
      ctx fwd Ir.Indexing.Empty
  in
  let want =
    nonzero (tag ^ "_ref") (Context.get_values (Context.run rctx rroutine) y.Tensor.value)
  in
  let opt = Option.value_exn ~here:[%here] !capture in
  let seeding =
    Autotune.search_seeding ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices:[] opt
  in
  (* The twins, keyed by label (which renders every field, gh-ocannl-1230's [autotune_labels]): each
     seed's own twin, and the twin of each unfused seed, deduplicated in seeding order. *)
  let twin_params =
    List.fold seeding.ss_whole ~init:[] ~f:(fun acc q ->
        let t = { q with Autotune.sk_epilogue = true } in
        let label = Autotune.sketch_params_label t in
        if List.Assoc.mem acc label ~equal:String.equal then acc else (label, t) :: acc)
    |> List.rev
  in
  let run (label, q) =
    let partitioned = ref false in
    let outcome =
      match
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun opt ->
              let sched = Autotune.sketch_schedule ~accum_prec ~p:q opt in
              partitioned := has_partition sched;
              [ Sched.apply sched opt ])
            (Context.auto ()) fwd Ir.Indexing.Empty
        in
        Context.get_values (Context.run ctx routine) y.Tensor.value
      with
      | got -> Ran got
      | exception Invalid_argument msg -> Refused msg
      | exception exn -> Crashed (Exn.to_string exn)
    in
    let one_line msg = String.prefix (String.tr ~target:'\n' ~replacement:' ' msg) 400 in
    (match outcome with
    | Ran got when not (Array.equal Float.equal got want) ->
        eprintf "%s: twin %s DIFFERS from the unfused reference (not part of the golden)\n%!" tag
          label
    | Ran _ -> ()
    | Refused msg ->
        eprintf "%s: twin %s refused: %s (not part of the golden)\n%!" tag label (one_line msg)
    | Crashed msg ->
        eprintf "%s: twin %s CRASHED: %s (not part of the golden)\n%!" tag label (one_line msg));
    { partitioned = !partitioned; outcome }
  in
  let twins = List.map twin_params ~f:run in
  let ran =
    List.filter_map twins ~f:(fun t -> match t.outcome with Ran got -> Some got | _ -> None)
  in
  eprintf "%s: %d twins swept, %d ran, %d partitioned (not part of the golden)\n%!" tag
    (List.length twins) (List.length ran)
    (List.count twins ~f:(fun t -> t.partitioned));
  all_twins := !all_twins @ twins;
  (* The convolution family seeds its GPU leg only through a tensorized pipeline (an advertised tile
     for its f32 operands: none on CUDA outside tf32, nor on HIP), so a site the seeder judges a
     conv has no twins where the backend withholds one. The gate is that capability, the seeder's
     own judgment, never the seed list under test (gh-ocannl-1115); a site the matmul family takes
     (a 1x1 window) keeps its claims everywhere. Every operand here is created single-precision. *)
  let capability =
    match fst (Autotune.family_seeding ~capabilities ~is_gpu ~is_cpu ~limits opt) with
    | Autotune.Seeding_conv _ ->
        Ll_test.tensorized_capability ~is_gpu ~is_cpu ~limits ~a_prec:Ir.Ops.single
          ~b_prec:Ir.Ops.single
          ~d_prec:(Lazy.force z.Tensor.value.Ir.Tnode.storage_prec)
    | Autotune.Seeding_matmul _ | Autotune.Seeding_none -> `Advertised
  in
  let when_, aggregation =
    match capability with
    | `Advertised -> (true, None)
    | `Withheld aggregation -> (false, Some aggregation)
  in
  let claim label xs ~f = gated_all ?aggregation ~when_ ~on:backend_name label xs ~f in
  claim (tag ^ ": every fused twin swept runs or is refused typed") twins ~f:(fun t ->
      match t.outcome with Ran _ | Refused _ -> true | Crashed _ -> false);
  claim (tag ^ ": every admitted fused twin equals the unfused reference bitwise") ran
    ~f:(fun got -> Array.equal Float.equal got want)

(* A GEMM [z = x . w] over [b] batch rows ([b = 0]: unbatched) with an elementwise tail [z + pos],
   [pos] broadcast along the rows. Values: x in [0, 3] by quarters, w and pos small integers. *)
let gemm ?(b = 0) ?(with_relu = false) ~m ~k ~n tag () =
  let rows = Int.max b 1 * m in
  let xv =
    Array.init (rows * k)
      ~f:(Ll_test.cycle_flat ~dims:[| rows; k |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init (k * n) ~f:(Ll_test.cycle_flat ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let pv = Array.init n ~f:(fun j -> Float.of_int (j % 5) -. 2.) in
  let x =
    if b = 0 then TDSL.ndarray xv ~label:[ tag ^ "_x" ] ~output_dims:[ m; k ] ()
    else TDSL.ndarray xv ~label:[ tag ^ "_x" ] ~batch_dims:[ b ] ~output_dims:[ m; k ] ()
  in
  let w = TDSL.ndarray wv ~label:[ tag ^ "_w" ] ~output_dims:[ k; n ] () in
  let pos = TDSL.ndarray pv ~label:[ tag ^ "_pos" ] ~output_dims:[ n ] () in
  let z =
    if b = 0 then
      let%op z = x +* "ik;kj=>ij" w in
      z
    else
      let%op z = x +* "b|ik;kj=>b|ij" w in
      z
  in
  let y =
    if with_relu then
      let%op y = relu (z + pos) in
      y
    else
      let%op y = z + pos in
      y
  in
  (z, y)

(* A stride-1 convolution [b, h, w, ic] * [kh, kw, ic -> oc] with a per-channel bias tail. *)
let conv ~hw ~ic ~kw ~oc tag () =
  let x =
    NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ hw; hw; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "_k") ~prec:Ir.Ops.single ~i:[ kw; kw; ic ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let bias =
    NTDSL.init ~l:(tag ^ "_b") ~prec:Ir.Ops.single ~o:[ oc ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
      ()
  in
  let%op z =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  let%op y = z + bias in
  (z, y)

let () =
  (* One full 16-deep k block and an edge, batched with 15 rows: transformer_names' one-hot
     embedding site, where the gh-ocannl-1670 split met the exactly-once check. *)
  sweep "ksplit1b" (gemm ~b:4 ~m:15 ~k:28 ~n:16 "ksplit1b");
  (* The same split unbatched, at a dividing row count. *)
  sweep "ksplit1" (gemm ~m:16 ~k:20 ~n:32 "ksplit1");
  (* Four full k blocks and an edge: the partitioned segments are not single-trip. *)
  sweep "ksplit4" (gemm ~m:24 ~k:70 ~n:40 "ksplit4");
  (* k exactly one block: a single-trip k-block loop and no Partition. *)
  sweep "kone" (gemm ~m:24 ~k:16 ~n:32 "kone");
  (* k below one block: one padded k block. *)
  sweep "ksub" (gemm ~m:24 ~k:12 ~n:32 "ksub");
  (* One full row block and an edge. *)
  sweep "rowedge" (gemm ~m:20 ~k:32 ~n:32 "rowedge");
  (* One full column block and an edge, with a relu tail. *)
  sweep "coledge" (gemm ~with_relu:true ~m:32 ~k:32 ~n:20 "coledge");
  (* A 3x3 window. *)
  sweep "conv3" (conv ~hw:11 ~ic:4 ~kw:3 ~oc:8 "conv3");
  (* A 1x1 window: the kernel loops are single-trip. *)
  sweep "conv1" (conv ~hw:8 ~ic:8 ~kw:1 ~oc:16 "conv1");
  (* The CPU packed seeds partition their padded k-block loop (gh-ocannl-1670), so on a C backend
     the sweep reaches Partition-carrying twins: the class this sweep exists for is in its
     population, admitted or refused. *)
  gated ~when_:is_cpu ~on:backend_name "the sweep reaches Partition-carrying fused twins"
    (List.exists !all_twins ~f:(fun t -> t.partitioned));
  printf "\nDone.\n%!"
