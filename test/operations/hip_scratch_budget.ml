(* gh-ocannl-533: an over-budget private (scratch) segment must be rejected BEFORE launch.

   Launching a kernel whose per-work-item scratch exceeds what the device can back does not fail
   cleanly: ROCm aborts the HSA queue ("[UpdateScratch] scratch_size overflow!" /
   HSA_STATUS_ERROR_INVALID_ARGUMENT) and what surfaces is an undifferentiated hipErrorInvalidValue
   out of synchronize, on a stream that is already dead. During the gh-ocannl-484 sweep one autotune
   candidate in that state took the whole benchmark process down.

   So the fix is prediction, not recovery (archive/docs/proposals/gh-ocannl-536.md): the HIP backend
   reads the linked kernel's private segment size and declines it as a typed [Resource_exceeded
   Thread_scratch] at [Backend_link]. This test pins both halves of that claim: the over-budget
   kernel is declined with that cause, and — the part that would be worthless to assert structurally
   alone — the device is still usable afterwards, with a following reduction computing the right
   values.

   Gated behind the [slow] alias: it is meaningful only on HIP, and it deliberately compiles a
   kernel with a ~260 KB per-work-item stack frame. The decline claims run in a [Verdict.leg] open
   only on HIP when the selected device's enforced limit ([Context.thread_scratch_limit]) puts the
   fixture over budget; off HIP they are counted backend skips, and on a HIP device whose enforced
   limit covers the fixture (or with [hip_scratch_validation=false]) they are counted environment
   skips (gh-ocannl-2404). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module SO = Ir.Schedule_outcome
open Verdict.Claims

(* Near the largest frame hipcc will accept: it refuses a stack frame over 262136 B, and 65024
   floats lands at ~260 KB, leaving ~2 KB of headroom for whatever else codegen puts on the frame
   (at 65528 the margin is 8 bytes — thin enough that an unrelated codegen change would turn this
   test into a compile error).

   Near-maximal rather than merely over gfx1151's ~104 KB budget, because the budget rises on
   smaller devices: a fixed 128 KiB frame would fall under budget on a device with fewer CUs. At
   this size only devices whose enforced per-work-item scratch limit exceeds the fixture skip the
   decline leg via [Context.thread_scratch_limit]. *)
let scratch_floats = 65024
let scratch_bytes = scratch_floats * 4

(* Replaces the routine body with: per Grid thread, fill a [Local] array of [scratch_floats]
   forward, then read it back in REVERSE into an accumulation. The reversed second loop is what
   keeps the array alive — a forward read in the same order would let the compiler forward each
   store to its load and drop the array entirely, and then there would be no scratch to reject. *)
let over_budget_transform ~out_tn ~src_tn ~rows (opt : LL.optimized) : LL.optimized =
  let prec = Lazy.force out_tn.Tn.storage_prec in
  let scratch =
    Tn.create ~namespace:"hipscratch" (Tn.Specified prec) ~id:0 ~label:[ "scratch"; "budget" ]
      ~unpadded_dims:(lazy [| scratch_floats |])
      ~padding:(lazy None)
      ()
  in
  Tn.Placements.update opt.LL.optimize_ctx.LL.placements scratch Tn.Local (Site "999:test-setup");
  ignore (LL.get_node opt.LL.traced_store scratch : LL.traced_array);
  let i = Idx.get_symbol () and x = Idx.get_symbol () and y = Idx.get_symbol () in
  let fill =
    LL.For_loop
      {
        index = x;
        from_ = 0;
        to_ = scratch_floats - 1;
        axis = LL.Serial;
        body =
          LL.Set
            {
              tn = scratch;
              idcs = [| Idx.Iterator x |];
              llsc = LL.Get (src_tn, [| Idx.Iterator i |]);
              debug = "";
            };
      }
  in
  let drain =
    LL.For_loop
      {
        index = y;
        from_ = 0;
        to_ = scratch_floats - 1;
        axis = LL.Serial;
        body =
          LL.Set
            {
              tn = out_tn;
              idcs = [| Idx.Iterator i |];
              llsc =
                LL.Get (scratch, [| Idx.affine ~symbols:[ (-1, y) ] ~offset:(scratch_floats - 1) |]);
              debug = "";
            };
      }
  in
  let llc =
    LL.For_loop
      { index = i; from_ = 0; to_ = rows - 1; axis = LL.Grid; body = LL.Seq (fill, drain) }
  in
  { opt with LL.llc }

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let on_hip = String.equal backend "hip" in
  let scratch_limit = Context.thread_scratch_limit ctx in
  if on_hip then
    Stdio.eprintf
      "scratch: enforced thread scratch limit %s B, fixture array %d B (not part of the golden)\n%!"
      (Option.value_map scratch_limit ~default:"(none)" ~f:Int.to_string)
      scratch_bytes;
  let hip_leg = Verdict.leg ~when_:on_hip ~on:backend () in
  let scratch_leg =
    Verdict.sub_leg ~aggregation:`Environment hip_leg
      ~when_:(lazy (Option.exists scratch_limit ~f:(fun limit -> scratch_bytes > limit)))
      ~on:
        "a hip device whose enforced thread scratch limit covers the fixture, or \
         hip_scratch_validation=false"
  in

  let rows = 8 in
  let src_values = Array.init rows ~f:(fun i -> Float.of_int (i + 1)) in
  let src = Tensor.term_init src_values ~grad_spec:Tensor.Prohibit_grad () in
  Train.set_materialized src.Tensor.value;

  (* Establish [src] on the device before the rejected compile: the declined routine never runs, so
     nothing it references would be initialized by it. This also fixes a baseline — the same
     reduction is recomputed in step 2, after the decline. *)
  let%op warm = src ++ "... => 0" in
  let ctx = Train.forward_once ctx warm in
  let expected = Array.fold src_values ~init:0.0 ~f:( +. ) in
  let warm_got = Context.get_values ctx warm.Tensor.value in
  p "scratch: the reduction is correct before the decline"
    (Array.length warm_got = 1 && Float.(abs (warm_got.(0) - expected) < 1e-4));

  (* --- 1. The over-budget candidate. --- *)
  let over_budget_outcome =
    lazy
      ((* Built here, not alongside [warm]: two live forward roots sharing [src] as a non-embedded
          descendant is a [consume_forward_code] conflict. *)
       let%op out = src *. 2.0 in
       let comp = Train.forward out in
       let outcome =
         Context.compile_outcome ~name:"hip_scratch_over_budget"
           ~lowered_transform:(fun o ->
             [ (over_budget_transform ~out_tn:out.Tensor.value ~src_tn:src.Tensor.value ~rows) o ])
           ~provenance:SO.User_schedule ~candidate:"over-budget scratch" ctx comp Idx.Empty
       in
       (* Release any unexpectedly linked routine without launching it: launching an over-budget
          kernel on HIP can abort the HSA queue (gh-ocannl-533). *)
       (match outcome with Ok (ctx', _) -> Context.release ctx' | Error _ -> ());
       outcome)
  in
  in_leg ~leg:scratch_leg
    "scratch: over-budget kernel declined as Resource_exceeded Thread_scratch at link" (fun label ->
      p label
        (match Lazy.force over_budget_outcome with
        | Error
            (SO.Classified
               {
                 phase = SO.Backend_link;
                 cause = SO.Resource_exceeded { resource = SO.Thread_scratch; requested; limit; _ };
                 _;
               }) ->
            (* The requested figure must be the kernel's real frame, not a placeholder, and the
               reported limit must agree with [Context.thread_scratch_limit]. *)
            requested >= scratch_bytes && Option.equal Int.equal limit scratch_limit
        | Ok _ | Error (SO.Classified _ | SO.Fatal _) -> false));
  in_leg ~leg:scratch_leg "scratch: the decline claims no device writes" (fun label ->
      p label
        (match Lazy.force over_budget_outcome with
        | Error (SO.Classified { execution_effect; _ }) ->
            SO.equal_execution_effect execution_effect SO.No_device_writes
        | Ok _ | Error (SO.Fatal _) -> false));

  (* --- 2. The device is still usable: a reduction afterwards computes the right values. This is
     the half a structural assertion cannot cover — a validator that rejected by poisoning the
     stream would pass check 1 and fail here. --- *)
  let%op total = src ++ "... => 0" in
  let ctx = Train.forward_once ctx total in
  let got = Context.get_values ctx total.Tensor.value in
  p "scratch: a reduction after the decline is still correct"
    (Array.length got = 1 && Float.(abs (got.(0) - expected) < 1e-4))
