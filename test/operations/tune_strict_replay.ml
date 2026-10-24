(* gh-ocannl-1560: a searched crown whose split-reduce prelude runs before fission replays as timed,
   and [autotune_strict_replay] checks every fissioned candidate's saved form replays.

   The CUDA shape of gh-ocannl-1480, end to end. A split-reduce prelude applies whole-routine ahead
   of fission, and [Schedule.apply]'s trailing simplification interval-folds guards, a gather's
   in-range guard against its index tensor's bounds candidate -- which a [Host_inits] upload
   proposes when a routine first links. On CUDA and HIP the serial baseline is never dispatched
   (gh-ocannl-532), so the search's first candidate, the seeded untuned plan with its split prelude,
   compiles before anything has linked, and its own link uploads the ids. Its saved segments then
   carry the guard, and before the prelude applied blind to node bounds
   ([Low_level.without_node_bounds]) a replay folded it: the crown's replay was refused and the
   untuned default shipped in its place (gpt2_mini_train's arm A crown).

   The fixture: an embedding gather [table * one_hot(ids)] (the gh-343 rewrite gives it a guarded
   dynamic read) summed over 8192 tokens into 64 cells -- a chain of 8192 per cell, which the seeded
   plan splits ([Autotune.seeded_split_backend]). The search runs FIRST in the process, on tensors
   nothing has uploaded yet: a reference computed on the device beforehand would upload the ids and
   fold the guard in the base lowering already, so nothing would be left for a replay to fold -- the
   reason a small gather fixture tried earlier did not reproduce the shape. Each leg builds fresh
   tensors for the same reason. Scripted timings crown the seeded plan; the reference is computed on
   the host, and every value is a multiple of 1/4 small enough that any summation order is exact.

   Legs: (1) without strict replay, the seeded crown (split included) ships unsubstituted and
   computes the reference bit for bit; (2) the same search under [autotune_strict_replay] completes
   -- every fissioned candidate, the seeded plan right after the link that uploaded the ids among
   them, replays from its saved form -- and ships unsubstituted; (3) strict replay over the
   single-kernel head-zero form of gh-ocannl-1480's cc shape ([tune_arm_a_replay]'s fixture), whose
   segmentation a replay once read back as a folded zero. The crown claims are the seeded backends'
   (stderr records a skip elsewhere); the completion, bounds and output claims hold everywhere. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let backend = Context.backend_name (Context.auto ())
let () = Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend
let seeded = Autotune.seeded_split_backend backend
let tokens = 8192
let vocab = 16
let embed = 64
let id_of t = ((t * 7) + (t / 5)) % vocab

(* Multiples of 1/4 in [-1, 1]: a sum over the 8192 tokens stays within 2^13 in steps of 1/4, exact
   in single precision whatever the order. *)
let table_value c = Float.of_int ((c * 5 % 9) - 4) /. 4.

(* [out.(o) = sum_t table.(o, ids.(t))]: the table's output axis outermost, its input axis
   rightmost. *)
let reference =
  Array.init embed ~f:(fun o ->
      let acc = ref 0. in
      for t = 0 to tokens - 1 do
        acc := !acc +. table_value ((o * vocab) + id_of t)
      done;
      !acc)

type fixture = { ids : Tensor.t; out : Tensor.t; comp : Ir.Assignments.comp }

let gather_table label =
  let ids = Nn_blocks.class_ids_of_int_list ~label:(label ^ "_ids") (List.init tokens ~f:id_of) in
  let table =
    TDSL.ndarray
      (Array.init (embed * vocab) ~f:table_value)
      ~label:[ label ^ "_table" ]
      ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  (ids, table)

(* Two copies under distinct names, so the two legs' routines never share a name. *)
let plain_fixture () =
  let ids, table = gather_table "plain" in
  let%op plain_emb = table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let%op plain_out = plain_emb ++ "b|o => o" in
  { ids; out = plain_out; comp = Train.forward plain_out }

let strict_fixture () =
  let ids, table = gather_table "strict" in
  let%op strict_emb = table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let%op strict_out = strict_emb ++ "b|o => o" in
  { ids; out = strict_out; comp = Train.forward strict_out }

(* The seeded plan wins whenever it is timed; every other candidate is far slower. *)
let measure (w : Scripted_search.window) =
  if String.is_prefix w.label ~prefix:"F_seeded[" then 1. else 50. +. (Float.of_int w.ordinal /. 64.)

type leg = {
  report : Autotune.report option;
  substituted : bool;
  output : float array;
  failure : string option;  (** The search's exception, if it raised. *)
}

let tune ~strict (comp : Ir.Assignments.comp) (out : Tensor.t) =
  let report = ref None in
  match
    Scripted_search.scripted
      ~settings:[ ("autotune_strict_replay", Bool.to_string strict) ]
      ~measure
      (fun () ->
        let deferred =
          Autotune.tune_deferred ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Ir.Indexing.Empty
        in
        let ctx, routine = Autotune.compile_deferred deferred in
        (deferred, ctx, routine))
  with
  | exception exn ->
      { report = !report; substituted = false; output = [||]; failure = Some (Exn.to_string exn) }
  | (deferred, ctx, routine), _record ->
      let ctx = Context.run ctx routine in
      let output = Context.get_values ctx out.Tensor.value in
      Context.release ctx;
      {
        report = !report;
        substituted = Autotune.shipped_substitute deferred;
        output;
        failure = None;
      }

let best_label leg = Option.value_map leg.report ~default:"" ~f:(fun r -> r.Autotune.best_label)

let describe tag leg =
  Stdio.eprintf "%s (not part of the golden): crowned %S, substituted %b%s\n%!" tag (best_label leg)
    leg.substituted
    (Option.value_map leg.failure ~default:"" ~f:(fun e -> ", raised " ^ e))

(* The crown claims, on the backends whose search seeds the split plan first. *)
let seeded_claims tag leg =
  Verdict.gated ~when_:seeded ~on:backend
    (tag ^ ": the search crowned the seeded plan, its split prelude included")
    (String.is_prefix (best_label leg) ~prefix:"F_seeded["
    && String.is_substring (best_label leg) ~substring:", 1 split,");
  Verdict.gated ~when_:seeded ~on:backend
    (tag ^ ": the seeded crown replays -- no untuned default shipped in its place")
    (Option.is_none leg.failure && not leg.substituted)

let gather_leg ~strict tag (f : fixture) =
  p
    (tag ^ ": nothing had proposed the ids' bounds before the search")
    (Option.is_none (Ir.Tnode.bounds_candidate f.ids.Tensor.value));
  let leg = tune ~strict f.comp f.out in
  describe tag leg;
  p (tag ^ ": the search completed and shipped a routine") (Option.is_none leg.failure);
  p
    (tag ^ ": a link during the search proposed the ids' bounds")
    (Option.is_some (Ir.Tnode.bounds_candidate f.ids.Tensor.value));
  seeded_claims tag leg;
  p
    (tag ^ ": the shipped routine computes the host reference bit for bit")
    (Array.equal Float.equal leg.output reference)

(* Legs (1) and (2), each first to touch its own tensors. *)
let () = gather_leg ~strict:false "prelude" (plain_fixture ())
let () = gather_leg ~strict:true "strict prelude" (strict_fixture ())

(* --- Leg (3): strict replay over gh-ocannl-1480's single-kernel head-zero form. --- *)

let n = 8

let nd label modulus offset =
  TDSL.ndarray
    (Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus ~offset ~stride:0.5))
    ~label:[ label ] ~output_dims:[ n; n ] ()

let a = nd "hz_a" 7 (-3.)
let b = nd "hz_b" 5 (-2.)
let%op hz_y = relu (a *. b)
let%op hz_loss = hz_y ++ "ij => 0"

let () =
  let comp = Train.forward hz_loss in
  let leg = tune ~strict:true comp hz_loss in
  describe "strict head zero" leg;
  p "strict head zero: every fissioned candidate replays from its saved form, the search completes"
    (Option.is_none leg.failure);
  p "strict head zero: the crown replays -- no untuned default shipped in its place"
    (not leg.substituted);
  let expected = Scripted_search.untuned ~ctx:(Context.auto ()) hz_loss comp in
  p "strict head zero: the shipped routine computes the untuned output bit for bit"
    (Array.equal Float.equal leg.output expected)

let () = Stdio.printf "tune_strict_replay: done\n"
