(* gh-ocannl-1326: a GPU backend's pool allocator reports a refused device allocation typed, as
   [Ir.Schedule_outcome.Device_memory_exhausted], which a compile classifies as a contained
   [Resource_exceeded Device_memory] decline at [Backend_link] carrying the refused bytes.

   The real driver refuses here, not a simulated device: the routine materializes a 1 TiB
   intermediate on the device (two 2^19-element inputs, their outer product), which no device has,
   so the allocation is refused outright rather than by putting the device under memory pressure —
   the same trick arrayjit/test/test_cuda_classify_failure plays with a raw 32 TiB allocation. The
   routine is compiled, never run. The other backends are skipped: the C backends allocate host
   memory, and a unified-memory Metal device answers a refusal through a different API. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SO = Ir.Schedule_outcome
open Verdict.Claims

let side = 1 lsl 19

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let typed_claim = "the refused compile is a contained device-memory decline at link"
  and bytes_claim = "the decline carries the refused bytes, at least the intermediate's"
  and free_claim = "the driver's free bytes are read, and fewer than were requested" in
  let claims = [ typed_claim; bytes_claim; free_claim ] in
  if not (List.mem [ "cuda"; "hip" ] backend ~equal:String.equal) then
    List.iter claims ~f:(fun what -> Verdict.skipped ~backend what)
  else
    let xv = Array.create ~len:side 1. in
    let x = TDSL.ndarray xv ~label:[ "dmr_x" ] ~output_dims:[ side ] () in
    let y = TDSL.ndarray xv ~label:[ "dmr_y" ] ~output_dims:[ side ] () in
    let%op big = x +* "a; b => ab" y in
    let%op total = big ++ "ab => 0" in
    Train.set_materialized big.Tensor.value;
    let comp = Train.forward total in
    (* The refused bytes and the driver's free bytes, where the outcome is the typed decline. *)
    let typed =
      match Context.compile_outcome ~provenance:SO.User_schedule ctx comp Ir.Indexing.Empty with
      | Ok _ ->
          Stdio.eprintf "the 1 TiB compile linked (not part of the golden)\n%!";
          None
      | Error (SO.Fatal f) ->
          Stdio.eprintf "fatal (not part of the golden): %s\n%!" (Exn.to_string f.SO.exn);
          None
      | Error (SO.Classified c) -> (
          Stdio.eprintf "decline (not part of the golden): %s\n%!" (SO.detail_of_cause c.SO.cause);
          match c with
          | {
           SO.phase = SO.Backend_link;
           cause = SO.Resource_exceeded { resource = SO.Device_memory; requested; limit; _ };
           execution_effect = SO.No_device_writes;
          } ->
              Some (requested, limit)
          | _ -> None)
    in
    p typed_claim (Option.is_some typed);
    p bytes_claim (Option.exists typed ~f:(fun (requested, _) -> requested >= side * side * 4));
    p free_claim
      (Option.exists typed ~f:(fun (requested, limit) ->
           Option.exists limit ~f:(fun free -> free < requested)))
