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
   run), so they run the same claims. CUDA and HIP log by device-side [printf] through stdout, a
   different channel ([Utils.capture_stdout_logs]); their claims are skipped here. *)

open Base
module LL = Ir.Low_level
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
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

let run_logged ?ctx ~first_id ~label () =
  let o, a, x = logged_stores ~first_id ~label in
  let av = Array.init n ~f:Float.of_int in
  let ctx, _routine =
    Ll_test.run_routine ?ctx ~name:label o ~seed:[ (a, av); (x, Ll_test.blank n) ]
  in
  let got = Context.get_values ctx x in
  Context.sync ctx;
  p_all2
    (label ^ ": the routine stores its values")
    got
    (Array.map av ~f:(fun v -> v +. 1.))
    ~f:Float.equal;
  ctx

let claim_count what ~label ~expected =
  let got = count ~label in
  Stdio.eprintf "%s: %d %s store lines in the stream log (not part of the golden)\n%!" what got
    label;
  p what (got = expected)

let () =
  if not (Utils.debug_log_from_routines ()) then
    fail "this test runs under --ocannl_debug_log_from_routines=true (see its dune rule)";
  let labels = [ "mrld_first"; "mrld_second" ] in
  match backend_name with
  | "cuda" | "hip" ->
      List.iter labels ~f:(fun label ->
          skipped ~backend:backend_name (label ^ ": the routine stores its values"));
      List.iter
        [
          "the first routine's stores are all in the stream log after its sync";
          "a second sync adds none of them again";
          "the second routine's stores are all in the stream log after its sync";
          "the first routine's lines are still there exactly once";
        ]
        ~f:(skipped ~backend:backend_name)
  | _ ->
      let ctx = run_logged ~first_id:1000 ~label:"mrld_first" () in
      claim_count "the first routine's stores are all in the stream log after its sync"
        ~label:"mrld_first" ~expected:n;
      Context.sync ctx;
      claim_count "a second sync adds none of them again" ~label:"mrld_first" ~expected:n;
      let (_ : Context.t) = run_logged ~ctx ~first_id:2000 ~label:"mrld_second" () in
      claim_count "the second routine's stores are all in the stream log after its sync"
        ~label:"mrld_second" ~expected:n;
      claim_count "the first routine's lines are still there exactly once" ~label:"mrld_first"
        ~expected:n
