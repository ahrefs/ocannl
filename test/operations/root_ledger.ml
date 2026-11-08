(* gh-ocannl-2063: fixed-size live root, many discarded compile children. Table reachability
   attributes retention (shared values overlap); Gc measures the whole live heap independently. Warm
   the compiler first, then clear its bounded analysis cache before every heap reading. *)
open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let settle () =
  Ir.Low_level.clear_analysis_cache ();
  Gc_witness.settle ();
  (Stdlib.Gc.stat ()).live_words

let compile root comp = Train.to_routine root Train.IDX.empty comp

let named name (comp : Ir.Assignments.comp) =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

let claim_empty label root =
  p_all label (Context.ledger_census root) ~f:(fun (_, count, _) -> count = 0)

let claim_nonempty label root =
  p_exists label (Context.ledger_census root) ~f:(fun (_, count, _) -> count > 0)

let loop ?(restore_history = false) root comp =
  let cycle () =
    let child, routine = compile root comp in
    let child = Context.run child routine in
    Context.sync child;
    let restore =
      if restore_history then Context.save_accumulation_history child else fun () -> ()
    in
    Context.release child;
    restore ()
  in
  for _ = 1 to 8 do
    cycle ()
  done;
  let before = settle () in
  let tables = Context.ledger_census root in
  for _ = 1 to 256 do
    cycle ()
  done;
  let after = settle () in
  let deltas =
    List.map2_exn tables (Context.ledger_census root)
      ~f:(fun (name, count, words) (_, count', words') ->
        Stdio.eprintf "%s: entries %+d, reachable words %+d (not part of the golden)\n%!" name
          (count' - count) (words' - words);
        (count' - count, words' - words))
  in
  Stdio.eprintf "live words delta: %+d (not part of the golden)\n%!" (after - before);
  p_all "every table has bounded retained growth" deltas ~f:(fun (count, words) ->
      count = 0 && words >= -64 && words <= 64);
  p "released compiles keep root heap growth bounded"
    (after - before >= -4096 && after - before <= 4096)

let () =
  let root = Context.cpu () in
  let%op output = [ 1.; 2.; 3. ] + [ 4.; 5.; 6. ] in
  Train.set_materialized output.value;
  let comp = Train.forward output in
  case "forward compile/release" (fun () -> loop root comp);
  let w = TDSL.param ~values:[| 1.; 2.; 3. |] "ledger_w" ~output_dims:[ 3 ] () in
  let%op loss = w *. w in
  let update = Train.grad_update loss in
  case "gradient compile/release" (fun () ->
      let grad_root = Context.set_values (Context.cpu ()) w.value [| 1.; 2.; 3. |] in
      loop grad_root update;
      Context.release grad_root);
  case "accumulating into a surviving buffer" (fun () ->
      let grad = (Option.value_exn w.diff).grad in
      let grad_root = Context.set_values (Context.cpu ()) w.value [| 1.; 2.; 3. |] in
      let grad_root = Context.set_values grad_root grad [| 0.; 0.; 0. |] in
      loop ~restore_history:true grad_root (Train.grad_update ~accum_steps:1 loss);
      p "retirement preserves writes to surviving ancestor buffers"
        (Context.buffer_written grad_root grad
        && Poly.equal (Context.get_values grad_root grad) [| 528.; 1056.; 1584. |]);
      Context.release grad_root);
  case "merge source after sibling writer release" (fun () ->
      let a, ra = compile root comp in
      let b, _ = compile a (named "rl_b" [%cd output =: 7]) in
      let c, rc = compile a (named "rl_c" [%cd output =: 9]) in
      (* Discard the run results: readiness must come from the shared buffer ledger. *)
      ignore (Context.run a ra);
      ignore (Context.run c rc);
      Context.release c;
      let destination_root = Context.cpu () in
      let destination, initial = compile destination_root comp in
      let destination = Context.run destination initial in
      let destination = Context.copy ~into_merge_buffer:Copy ~src:b ~dst:destination output.value in
      let consumer, routine = compile destination (named "rl_merge" [%cd output =: output.merge]) in
      let consumer = Context.run consumer routine in
      p "a released sibling's surviving writes remain ready for merge copy"
        (Poly.equal (Context.get_values consumer output.value) [| 9.; 9.; 9. |]);
      Context.release consumer;
      Context.release destination;
      Context.release destination_root;
      Context.release b;
      Context.release a);
  let child, routine = compile root comp in
  let y =
    Operation.init ~l:"ledger_fork" ~prec:Ir.Ops.single ~o:[ 3 ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  Train.set_materialized y.value;
  let chain = Context.set_values child y.value [| 1.; 2.; 3. |] in
  let fork = Context.set_values child y.value [| 4.; 5.; 6. |] in
  Context.release chain;
  claim_nonempty "an upload fork retains the routine records" root;
  let fork = Context.run fork routine in
  p "the surviving fork still executes the routine"
    (Poly.equal (Context.get_values fork output.value) [| 5.; 7.; 9. |]);
  Context.release fork;
  claim_empty "the last fork release retires the routine records" root;
  Context.release fork;
  claim_empty "repeated release is idempotent" root;
  let child, _ = compile root comp in
  let raised =
    try
      Ir.Resource_fault_injection.with_callback
        (fun point ->
          if
            Ir.Resource_fault_injection.equal_point point
              Ir.Resource_fault_injection.Finalize_before_free
          then failwith "ledger release fault")
        ~f:(fun () -> Context.release child);
      false
    with Failure message -> String.equal message "ledger release fault"
  in
  p "the release fault was injected" raised;
  claim_nonempty "failed release retains its routine records for retry" root;
  Context.release child;
  claim_empty "successful retry retires the records" root;
  Context.release root
