(* gh-ocannl-2165: checking a shape scheme must leave the training step's emitted source (up to
   minted identifiers) and placements unchanged. Fresh processes exercise the configuration setting
   itself rather than its test override, and start each compile with fresh id/uid counters. This is
   a cc codegen test, not a value test. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module Generated = Test_utils.Generated
module Tn = Ir.Tnode
module LL = Ir.Low_level

type observation = { source : string; placements : (int * string) list; traces : int; checks : int }

let emit ~changed =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.fixed_state_for_init <- Some 3;
  Generated.init ~backend_name:"cc";
  Tensor.unsafe_reinitialize ();
  let traces = ref 0 and checks = ref 0 in
  let ctx = Context.auto () in
  let comp, loss =
    Tensor.For_testing.observe_schemes
      (fun () ->
        let xs =
          NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ 3 ] ~o:[ 4 ]
            ~f:(fun i -> Float.of_int (1 + (4 * i.(0)) + i.(1)))
            ()
        in
        let%op layer () x = relu ({ b; o = [ 4 ] } + ({ w } * x)) in
        let layer = layer () in
        let%op err = layer (layer xs) - 1.0 in
        let%op loss = ((err *. err) ++ "...|... => |->0") /. !..3 in
        let rate = if changed then 0.125 else 0.25 in
        ( Ir.Assignments.sequence
            [ Train.grad_update loss; Train.sgd_update ~learning_rate:(TDSL.O.( !. ) rate) loss ],
          loss ))
      ~on_event:(function
        | Tensor.Scheme.Traced _ -> Int.incr traces
        | Checked _ -> Int.incr checks
        | Unschemed _ -> ())
  in
  let ctx = Train.init_params ctx Train.IDX.empty loss in
  let name = "scheme_step" in
  let compiled, routine = Autotune.untuned_compile ~name ctx comp Train.IDX.empty in
  let nodes =
    List.concat_map routine.Context.segments ~f:(fun seg -> Hashtbl.keys seg.LL.traced_store)
    |> List.dedup_and_sort ~compare:Tn.compare
    |> List.sort ~compare:(fun a b -> Int.compare a.Tn.id b.Tn.id)
  in
  let source_name = if List.length routine.segments > 1 then name ^ "__seg" else name in
  let result =
    {
      source =
        Generated.normalize_minted ~facts:(Context.backend_facts compiled)
          (Generated.read source_name);
      placements =
        List.map nodes ~f:(fun tn ->
            (tn.Tn.id, Tn.debug_memory_mode (Tn.Placements.get (Context.placements compiled) tn)));
      traces = !traces;
      checks = !checks;
    }
  in
  Context.release compiled;
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  Stdlib.output_string Stdlib.stdout (Stdlib.Marshal.to_string result []);
  Stdlib.flush Stdlib.stdout

let observe mode ~changed =
  let result =
    Fresh_process.run
      ([ "--emit"; "--ocannl_backend=cc"; "--ocannl_op_shape_schemes=" ^ mode ]
      @ if changed then [ "--changed" ] else [])
  in
  let status, stdout, stderr = result in
  Stdio.eprintf "%s, changed=%b (not part of the golden):\n%s%!" mode changed stderr;
  if not (Poly.equal status (Unix.WEXITED 0)) then (
    Fresh_process.report ~label:mode result;
    failwith "training step child failed");
  (Stdlib.Marshal.from_string stdout 0 : observation)

let same_placements a b =
  List.equal
    (fun (id1, mode1) (id2, mode2) -> id1 = id2 && String.equal mode1 mode2)
    a.placements b.placements

let same_source a b = String.equal a.source b.source

let () =
  let args = Array.to_list (Sys.get_argv ()) in
  if List.mem args "--emit" ~equal:String.equal then
    emit ~changed:(List.mem args "--changed" ~equal:String.equal)
  else
    let off = observe "off" ~changed:false in
    let check = observe "check" ~changed:false in
    p "control: off neither traces nor checks" (off.traces = 0 && off.checks = 0);
    p "control: check traces the shared layer and checks both applications"
      (check.traces = 1 && check.checks = 2);
    p "control: the training step emits source and has placements"
      ((not (String.is_empty off.source)) && not (List.is_empty off.placements));
    if not (same_source off check) then
      Stdio.eprintf "off source:\n%s\ncheck source:\n%s\n%!" off.source check.source;
    p "check and off emit identical normalized training-step source" (same_source off check);
    p "check and off place every compiled node identically" (same_placements off check);
    let changed = observe "check" ~changed:true in
    p "control: changing the SGD rate changes the normalized source" (not (same_source off changed))
