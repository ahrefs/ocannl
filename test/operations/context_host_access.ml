(* Regression test for gh-ocannl-333: tensor data lives only on devices, and all CPU-side value
   access is an on-demand, context-mediated device-to-host transfer.

   The key invariants pinned here: - A large (>16-element) ndarray-backed literal carries no host
   copy on its tensor node; its data is uploaded into each context that links it. Compiling/running
   the SAME literal-bearing graph into two INDEPENDENT contexts initializes both identically (the
   [Host_inits] table is read, not consumed) — this is the assertion that would fail if init data
   were a one-shot global cache. - [Context.set_values] / [Context.get_values] round-trip values
   through the device buffer. - The context-aware [At] accessors read/write individual elements. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let big = Array.init 20 ~f:Float.of_int

let () =
  (* --- Multi-context initialization of one ndarray-backed literal --- [t] is created once; each
     call builds a fresh graph reading it and runs it in an independent context, so the same literal
     is uploaded into both. *)
  let t = TDSL.ndarray big ~label:[ "lit" ] ~output_dims:[ 20 ] () in
  let read_in_fresh_context () =
    let%op out = t + t in
    let ctx = Context.cpu () in
    let ctx = Train.forward_once ctx out in
    Context.get_values ctx t.Tensor.value
  in
  let v1 = read_in_fresh_context () in
  let v2 = read_in_fresh_context () in
  Stdio.printf "literal matches source in ctx1: %b\n" (Array.equal Float.equal v1 big);
  Stdio.printf "literal matches across two independent contexts: %b\n"
    (Array.equal Float.equal v1 v2);

  (* --- set_values / get_values round-trip through the device buffer --- *)
  let%op out = t + t in
  let ctx = Context.cpu () in
  let ctx = Train.forward_once ctx out in
  let doubled = Array.map big ~f:(fun x -> 2. *. x) in
  Stdio.printf "out = t + t computed on device: %b\n"
    (Array.equal Float.equal (Context.get_values ctx out.Tensor.value) doubled);
  let fresh = Array.init 20 ~f:(fun i -> Float.of_int (100 + i)) in
  let ctx = Context.set_values ctx out.Tensor.value fresh in
  Stdio.printf "set_values/get_values round-trip: %b\n"
    (Array.equal Float.equal (Context.get_values ctx out.Tensor.value) fresh);

  (* --- context-aware At accessors --- *)
  Stdio.printf "At get element 3: %.1f\n" Operation.At.((ctx, out).@[3]);
  let ctx = Operation.At.((ctx, out).@{[| 0 |]} <- 42.0) in
  Stdio.printf "At set then get element 0: %.1f\n" Operation.At.((ctx, out).@[0]);
  ignore (ctx : Context.t)
