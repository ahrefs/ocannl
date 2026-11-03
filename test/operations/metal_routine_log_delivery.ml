(* gh-ocannl-1594: routine logs reach the stream log by the synchronization after the routine, each
   exactly once, and a logging routine does not crash the host.

   Metal hands shader log messages to an [MTLLogState] handler on its own dispatch threads. The
   backend used to register an OCaml closure there; the ctypes block wrapping it lived in memory the
   OCaml GC reclaimed while Metal still pointed at it, and a logging routine crashed the process.
   The handler is now ocaml-metal's native one, which buffers the messages C-side; [await] drains
   them on the host thread. Survival alone would not tell a working capture from one that drops
   every message, so this counts them: after a routine of [n] logged stores and a [sync], the stream
   log holds exactly [n] of that routine's store lines; a second [sync] adds none; a second
   routine's lines arrive the same way, exactly once, and leave the first routine's alone.

   The C backends write the same stream log (each routine's log file is appended to it after the
   run), so they run the same claims. CUDA and HIP deliver device-side [printf] lines through the
   capture window each logged launch opens (gh-ocannl-1636), a mechanism
   test/operations/routine_log_stream_files pins; this test, whose counts were established on cc and
   Metal only, skips them. *)

open Base
module LL = Ir.Low_level
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let n = 8

(* The stream log's lines that record a store into [label]: the value lines, ["<label>[<i>]..."],
   not the ["# <label>[...] := ..."] comments some backends also log. *)
let store_lines ~label =
  let dir = Utils.log_files_dir () in
  let files =
    Stdlib.Sys.readdir dir |> Array.to_list
    |> List.filter ~f:(String.is_suffix ~suffix:".log")
    |> List.sort ~compare:String.compare
  in
  List.concat_map files ~f:(fun f -> Stdio.In_channel.read_lines (Stdlib.Filename.concat dir f))
  |> List.filter ~f:(String.is_prefix ~prefix:(label ^ "["))

let count ~label = List.length (store_lines ~label)

(* A routine storing [n] values into a fresh node named [label], each store logged. *)
let logged_stores ~first_id ~label =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk (label ^ "_src") and x = mk label in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add (Ll_test.get a [| at |]) (Ll_test.c 1.)))
  in
  let o = Ll_test.optimize ~materialized:[ a; x ] ~name:label llc in
  (o, a, x)

(* The context after the run and its [sync], with the values the routine stored and the ones it
   should have. *)
let run_logged ?ctx ~first_id ~label () =
  let o, a, x = logged_stores ~first_id ~label in
  let av = Array.init n ~f:Float.of_int in
  let ctx, _routine =
    Ll_test.run_routine ?ctx ~name:label o ~seed:[ (a, av); (x, Ll_test.blank n) ]
  in
  let got = Context.get_values ctx x in
  Context.sync ctx;
  (ctx, got, Array.map av ~f:(fun v -> v +. 1.))

let claim_count what ~label ~expected =
  let got = count ~label in
  Stdio.eprintf "%s: %d %s store lines in the stream log (not part of the golden)\n%!" what got
    label;
  p what (got = expected)

(* CUDA and HIP skip every claim, by name, in the order this one sequence makes them
   (gh-ocannl-1703). *)
let leg =
  Verdict.leg
    ~when_:(not (List.mem [ "cuda"; "hip" ] backend_name ~equal:String.equal))
    ~on:backend_name ()

let () =
  if not (Utils.debug_log_from_routines ()) then
    fail "this test runs under --ocannl_debug_log_from_routines=true (see its dune rule)";
  let first = lazy (run_logged ~first_id:1000 ~label:"mrld_first" ()) in
  in_leg ~leg "mrld_first: the routine stores its values" (fun label ->
      let _, got, want = Lazy.force first in
      p_all2 label got want ~f:Float.equal);
  in_leg ~leg "the first routine's stores are all in the stream log after its sync" (fun what ->
      claim_count what ~label:"mrld_first" ~expected:n);
  in_leg ~leg "a second sync adds none of them again" (fun what ->
      let ctx, _, _ = Lazy.force first in
      Context.sync ctx;
      claim_count what ~label:"mrld_first" ~expected:n);
  let second =
    lazy
      (let ctx, _, _ = Lazy.force first in
       run_logged ~ctx ~first_id:2000 ~label:"mrld_second" ())
  in
  in_leg ~leg "mrld_second: the routine stores its values" (fun label ->
      let _, got, want = Lazy.force second in
      p_all2 label got want ~f:Float.equal);
  in_leg ~leg "the second routine's stores are all in the stream log after its sync" (fun what ->
      claim_count what ~label:"mrld_second" ~expected:n);
  in_leg ~leg "the first routine's lines are still there exactly once" (fun what ->
      claim_count what ~label:"mrld_first" ~expected:n)
