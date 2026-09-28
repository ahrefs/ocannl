(* The autotuner's privatized preset extension (Autotune.extend_with_privatize), used by the
   privatized fission-candidate flavor: detection of a materialized read-modify-write accumulator
   and executed parity of the privatized schedule.

   [mc = ma * mb] with a materialized output lowers to [Zero_out mc] plus a serial nest accumulating
   [mc[i,j] += ma[i,k] * mb[k,j]]. The extension over the backend's default preset (which may well
   be empty here — the whole-routine annotator bails on the materialized [Zero_out]; the fission
   pipeline separates zeros into their own segments) must append exactly one [Privatize] targeting
   [mc] over the serial reduction loop, and applying the extended schedule must compute the same
   values as the identity-transform twin.

   The appended [Privatize] mints its tile at the backend's accumulator residency (gh-ocannl-1116),
   pinned structurally on a bf16 twin (accum_width's Privatize legs pin the executed width), and the
   saved form carries that precision as a REQUIRED field: a cache entry written before it (the field
   absent) must fail to decode, which is what makes its lookup miss and the routine re-tune rather
   than replay a winner timed under the narrowing rendering. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module SC = Ir.Schedule_cache
open Verdict.Claims

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let approx a b = Float.(abs (a -. b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 16

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Identity-transform twin --- *)
  let%op mc0 = ma * mb in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      ctx_s
      (named "priv_naive" (Train.forward mc0))
      Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_naive = nonzero "apz_naive" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- The backend's default preset, extended with privatization --- *)
  let%op mc1 = ma * mb in
  let n_privatize = ref (-1) in
  let transform (opt : LL.optimized) =
    let preset =
      if Sched.backend_is_gpu backend_name then Sched.default_gpu ~min_parallel:1 opt
      else if Sched.backend_is_cpu backend_name then Sched.default_cpu ~min_parallel:1 opt
      else []
    in
    let extended = Autotune.extend_with_privatize ~accum_prec ~static_indices:[] preset opt in
    n_privatize :=
      List.count extended ~f:(function
        | Sched.Privatize { target; _ } -> Ir.Tnode.equal target mc1.Tensor.value
        | _ -> false);
    Sched.apply extended opt
  in
  let ctx_a = Context.auto () in
  let ctx_a, routine_a =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx_a
      (named "priv_tuned" (Train.forward mc1))
      Ir.Indexing.Empty
  in
  let ctx_a = Context.run ctx_a routine_a in
  let got_priv = Context.get_values ctx_a mc1.Tensor.value in
  p "extension appends exactly one Privatize targeting the accumulator" (!n_privatize = 1);
  p_all2 "privatized preset matches the identity twin" got_priv got_naive ~f:approx;

  (* --- The bf16 twin: the precision the extension mints at, and its saved form --- *)
  let bf16 = Ir.Ops.bfloat16 in
  let ma16 =
    NTDSL.init ~l:"ma16" ~prec:bf16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> mav.((idcs.(0) * n) + idcs.(1)))
      ()
  in
  let mb16 =
    NTDSL.init ~l:"mb16" ~prec:bf16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> mbv.((idcs.(0) * n) + idcs.(1)))
      ()
  in
  let%op mc16 = ma16 * mb16 in
  Ir.Tnode.update_prec mc16.Tensor.value bf16;
  let minted = ref [] and saved_forms = ref None in
  let transform16 (opt : LL.optimized) =
    let extended = Autotune.extend_with_privatize ~accum_prec ~static_indices:[] [] opt in
    minted :=
      List.filter_map extended ~f:(function
        | Sched.Privatize { target; acc_prec; _ } when Ir.Tnode.equal target mc16.Tensor.value ->
            Some acc_prec
        | _ -> None);
    let canon = SC.canonicalize ~static_indices:[] opt in
    let saved, _ = SC.to_saved (SC.base_registry canon) extended in
    saved_forms := Some (canon, saved);
    Sched.apply extended opt
  in
  let ctx16, routine16 =
    Context.compile
      ~lowered_transform:(fun o -> [ transform16 o ])
      (Context.auto ())
      (named "priv_bf16" (Train.forward mc16))
      Ir.Indexing.Empty
  in
  ignore (Context.run ctx16 routine16 : Context.t);
  p
    "the extension mints the bf16 accumulator's Privatize tile at the backend's accumulator \
     residency"
    (List.equal Ir.Ops.equal_prec !minted [ accum_prec bf16 ]);
  let canon, saved = Option.value_exn !saved_forms in
  let sexp = SC.sexp_of_saved_schedule saved in
  let replayed, _ = SC.of_saved canon (SC.saved_schedule_of_sexp sexp) in
  p "a saved Privatize replays with the precision it was minted at"
    (List.equal Ir.Ops.equal_prec
       (List.filter_map replayed ~f:(function
         | Sched.Privatize { acc_prec; _ } -> Some acc_prec
         | _ -> None))
       !minted);
  (* The pre-gh-ocannl-1116 spelling of the same entry: every [(acc_prec ...)] field dropped. *)
  let rec strip_acc_prec (s : Sexp.t) : Sexp.t =
    match s with
    | Sexp.Atom _ -> s
    | Sexp.List l ->
        Sexp.List
          (List.filter_map l ~f:(function
            | Sexp.List [ Sexp.Atom "acc_prec"; _ ] -> None
            | x -> Some (strip_acc_prec x)))
  in
  let pre_1116 = strip_acc_prec sexp in
  p
    "a saved Privatize without its accumulator precision (a pre-gh-ocannl-1116 entry) does not \
     decode"
    ((not (Sexp.equal pre_1116 sexp))
    && Result.is_error (Result.try_with (fun () -> SC.saved_schedule_of_sexp pre_1116)))
