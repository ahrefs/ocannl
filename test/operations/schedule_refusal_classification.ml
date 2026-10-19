(* gh-ocannl-1323: only an op's deliberate precondition refusal is a declined candidate.

   [Schedule.apply_classified] used to turn every [Invalid_argument] escaping schedule application
   into a typed [Illegal_schedule], so a genuine transform bug (an array bound, a [List.zip_exn])
   was skipped by the tuner as one more decline. The ops now raise their refusals typed at their own
   sites ([Schedule.refuse]), the classified path passes those through and nothing else, and the
   public [Schedule.apply] renders the refusal as the [Invalid_argument] its direct callers have
   always seen. This test pins the three readings at the boundary a search classifies through,
   [Schedule_outcome.protect], under both strictness settings: a [Fuse_epilogue] refusal is a
   classified decline; a plain [Invalid_argument] a transform raises (injected through
   [Schedule.with_transform_fault], since a real bug has no stable trigger) is fatal when strict and
   an [Unclassified] decline only when permissive; and the injection leaves the public surface's
   exception unchanged. No backend runs: application is pure IR.

   gh-ocannl-1390: the same three readings one level up, at the sketch builder
   ([Autotune.sketch_schedule]), which used to convert every [Invalid_argument] escaping it into an
   [Illegal_schedule "Autotune.sketch"] decline. A builder precondition (a conv sketch on code with
   no conv site) refuses typed, with that very cause; an [Invalid_argument] the builder raises
   (injected through [Autotune.with_sketch_builder_fault]) escapes untyped -- fatal when strict, an
   [Unclassified] decline only when permissive. Building is pure too. *)

open Base
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Outcome = Ir.Schedule_outcome

let node = Ll_test.node_factory ~first_id:1_323_000_000 ~dims:[| 4 |] ()

(* acc[i] += a[i, k] * b[k], and nothing after it: a reduction with no elementwise tail, so
   [Fuse_epilogue] on [acc] is refused by its own precondition ("no statement follows the
   reduction"). *)
let program () =
  let open Ll_test in
  let a = node ~dims:[| 4; 4 |] "a" and b = node "b" and acc = node "acc" in
  List.iter [ a; b; acc ] ~f:materialize;
  let i = sym () and k = sym () in
  let body =
    set acc
      [| iter i |]
      (add (get acc [| iter i |]) (mul (get a [| iter i; iter k |]) (get b [| iter k |])))
  in
  let llc = seq (zero acc) (loop_n i 4 (loop_n k 4 body)) in
  (optimize ~name:"refusal_classification" llc, acc, i)

let classify ~strict f =
  Outcome.protect ~strict
    ~classify_backend:(fun _ _ -> None)
    ~provenance:Outcome.Candidate ~phase:Outcome.Transform f

let injected = "injected transform bug (gh-ocannl-1323)"
let fault (_ : Sched.optop) = invalid_arg injected

let () =
  let opt, acc, i = program () in
  let refused = [ Sched.Fuse_epilogue { target = acc; shared = false } ] in
  (* Applies cleanly, so the only failure in the injected legs is the injection. *)
  let benign = [ Sched.Retype { axis = i; ty = LL.Serial } ] in
  let refusal_prefix = "Schedule.Fuse_epilogue: " in
  let refusal_detail =
    match Sched.apply_classified refused opt with
    | (_ : LL.optimized) -> None
    | exception Outcome.Cause_at (Outcome.Transform, Outcome.Illegal_schedule { check; detail })
      when String.equal check "Schedule.apply" ->
        Some detail
  in
  p "the Fuse_epilogue precondition refusal is raised typed, as Illegal_schedule at Transform"
    (Option.exists refusal_detail ~f:(String.is_prefix ~prefix:refusal_prefix));
  List.iter [ true; false ] ~f:(fun strict ->
      pf "strict=%b: the refusal reads as a classified decline carrying the refusal" strict
        (match classify ~strict (fun () -> Sched.apply_classified refused opt) with
        | Error
            (Outcome.Classified
               { phase = Outcome.Transform; cause = Outcome.Illegal_schedule { detail; _ }; _ }) ->
            Option.equal String.equal (Some detail) refusal_detail
        | Ok _ | Error _ -> false));
  p "the public Schedule.apply renders the same refusal as Invalid_argument"
    (match Sched.apply refused opt with
    | (_ : LL.optimized) -> false
    | exception Invalid_argument detail -> Option.equal String.equal (Some detail) refusal_detail);
  let applies () =
    match Sched.apply_classified benign opt with (_ : LL.optimized) -> true | exception _ -> false
  in
  p "the benign schedule applies without the injection" (applies ());
  p "an injected Invalid_argument escapes apply_classified untyped"
    (Sched.with_transform_fault fault (fun () ->
         match Sched.apply_classified benign opt with
         | (_ : LL.optimized) -> false
         | exception Invalid_argument detail -> String.equal detail injected));
  p "strict: an untyped transform exception is fatal, its own exception and no cause"
    (Sched.with_transform_fault fault (fun () ->
         match classify ~strict:true (fun () -> Sched.apply_classified benign opt) with
         | Error (Outcome.Fatal { exn = Invalid_argument detail; cause = None; phase; _ }) ->
             String.equal detail injected && Outcome.equal_phase phase Outcome.Transform
         | Ok _ | Error _ -> false));
  p "permissive: the same exception is contained as Unclassified, never as Illegal_schedule"
    (Sched.with_transform_fault fault (fun () ->
         match classify ~strict:false (fun () -> Sched.apply_classified benign opt) with
         | Error
             (Outcome.Classified
                { cause = Outcome.Unclassified { exn_constructor = "Invalid_argument"; _ }; _ }) ->
             true
         | Ok _ | Error _ -> false));
  p "the public Schedule.apply still raises the injected Invalid_argument itself"
    (Sched.with_transform_fault fault (fun () ->
         match Sched.apply benign opt with
         | (_ : LL.optimized) -> false
         | exception Invalid_argument detail -> String.equal detail injected));
  p "the fault seam is scoped: it is gone once with_transform_fault returns" (applies ())

(* A conv sketch's parameters: on [program]'s reduction, which is no convolution, the builder's site
   precondition refuses. *)
let conv_params : Autotune.sketch_params =
  {
    sk_gpu = false;
    sk_mma = true;
    sk_simd = 0;
    sk_bm = 0;
    sk_bn = 0;
    sk_bk = 0;
    sk_tm = 0;
    sk_tn = 0;
    sk_hoist = false;
    sk_grid = false;
    sk_pack_rest = false;
    sk_conv = true;
    sk_epilogue = false;
    sk_batch_grid = false;
    sk_batch_inner = false;
    sk_coalesce = None;
    sk_swizzle = None;
    sk_depth = 1;
    sk_pack_prec = None;
    sk_tile = None;
  }

let builder_injected = "injected sketch-builder bug (gh-ocannl-1390)"
let builder_fault (_ : Autotune.sketch_params) = invalid_arg builder_injected

let () =
  let opt, _, _ = program () in
  let build () = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:conv_params opt in
  let refusal () =
    match build () with
    | (_ : Sched.schedule) -> None
    | exception Outcome.Cause_at (Outcome.Transform, Outcome.Illegal_schedule { check; detail })
      when String.equal check "Autotune.sketch" ->
        Some detail
    | exception (_ : exn) -> None
  in
  let builder_detail = refusal () in
  p "the builder's no-conv-site precondition refuses typed, as Illegal_schedule \"Autotune.sketch\""
    (Option.exists builder_detail ~f:(String.is_substring ~substring:"no convolution site"));
  List.iter [ true; false ] ~f:(fun strict ->
      pf "strict=%b: the builder refusal reads as a classified decline carrying the refusal" strict
        (match classify ~strict build with
        | Error
            (Outcome.Classified
               { phase = Outcome.Transform; cause = Outcome.Illegal_schedule { detail; _ }; _ }) ->
            Option.equal String.equal (Some detail) builder_detail
        | Ok _ | Error _ -> false));
  p "an injected Invalid_argument escapes the builder untyped"
    (Autotune.with_sketch_builder_fault builder_fault (fun () ->
         match build () with
         | (_ : Sched.schedule) -> false
         | exception Invalid_argument detail -> String.equal detail builder_injected
         | exception (_ : exn) -> false));
  p "strict: an untyped builder exception is fatal, its own exception and no cause"
    (Autotune.with_sketch_builder_fault builder_fault (fun () ->
         match classify ~strict:true build with
         | Error (Outcome.Fatal { exn = Invalid_argument detail; cause = None; phase; _ }) ->
             String.equal detail builder_injected && Outcome.equal_phase phase Outcome.Transform
         | Ok _ | Error _ -> false));
  p "permissive: the same exception is contained as Unclassified, never as Illegal_schedule"
    (Autotune.with_sketch_builder_fault builder_fault (fun () ->
         match classify ~strict:false build with
         | Error
             (Outcome.Classified
                { cause = Outcome.Unclassified { exn_constructor = "Invalid_argument"; _ }; _ }) ->
             true
         | Ok _ | Error _ -> false));
  p "the builder fault seam is scoped: the typed refusal is back once it returns"
    (Option.equal String.equal (refusal ()) builder_detail)
