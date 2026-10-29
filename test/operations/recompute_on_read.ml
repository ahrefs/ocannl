(* gh-ocannl-777: a virtual node is read from a Context by recomputation, and a node that cannot be
   read says why.

   Executed parity: one program, built twice -- once with its intermediates left to the pipeline
   (virtual, read back by [Context.get_values]'s throwaway recompute routine), once with them
   materialized (read back from their buffers). The producers discriminate: [x] holds a distinct
   dyadic value in every cell, and each intermediate depends on its cell's value, so a read of the
   wrong cell, a broadcast, or an unrun copy cannot match. Elementwise arithmetic on these values is
   exact in single precision, so the elementwise claims are bitwise; the reduction gets a two-sided
   tolerance.

   Around it: the parent context is left as it was (no buffer for the node, no ledger growth, still
   usable); a read evaluates the stored computation against the leaves as they are at the read
   (recompute-at-read, gh-617); a failure after the recompute ran propagates and still releases the
   throwaway context's pools; and each refusal names its reason. Runs on the backend [Context.auto]
   picks. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX
module Tn = Ir.Tnode
open Verdict.Claims

let rows = 3
let cols = 4

(* Distinct per cell, dyadic: [x.(i, j) = (8i + j + 1) / 8]. *)
let x_values = Array.init (rows * cols) ~f:(fun n -> Float.of_int (n + 1) /. 8.)
let x_values' = Array.map x_values ~f:(fun v -> v +. 0.5)

let is_mode m ctx (t : Tensor.t) =
  Option.equal Tn.equal_memory_mode
    (Option.map ~f:fst (Tn.Placements.get (Context.placements ctx) t.value))
    (Some m)

let refusal f =
  match f () with (_ : float array) -> None | exception Utils.User_error msg -> Some msg

(* Non-empty by construction: a claim built on it names at least one substring. *)
let says msg substrings =
  (not (List.is_empty substrings))
  && List.for_all substrings ~f:(fun s -> String.is_substring msg ~substring:s)

(* The program; [~materialize] is the one difference between the two arms. *)
let build ~materialize =
  let x = TDSL.ndarray x_values ~label:[ "ror_x" ] ~output_dims:[ rows; cols ] () in
  let x1 =
    TDSL.ndarray
      (Array.sub x_values ~pos:0 ~len:cols)
      ~label:[ "ror_x1" ] ~output_dims:[ 1; cols ] ()
  in
  let%op v = (x *. 0.5) + x in
  let%op v2 = (v *. v) + 1 in
  let%op r = v2 ++ "ij => j" in
  let%op w = x1 *. 3 in
  let%op out = (r *. 2) + (w ++ "ij => j") in
  if materialize then
    List.iter [ v; v2; r; w ] ~f:(fun (t : Tensor.t) -> Train.set_materialized t.value);
  let ctx = Train.forward_once (Context.auto ()) out in
  (ctx, x, v, v2, r, w)

let () =
  let ctx, x, v, v2, r, w = build ~materialize:false in
  let mctx, _, mv, mv2, mr, mw = build ~materialize:true in
  p "the pipeline leaves v, v2 and w virtual"
    (is_mode Tn.Virtual ctx v && is_mode Tn.Virtual ctx v2 && is_mode Tn.Virtual ctx w);
  p "the materialized arm stores them"
    (Context.mem mctx mv.value && Context.mem mctx mv2.value && Context.mem mctx mw.value);
  (* A refused read comes back empty, so the claim reading it fails as a claim (an empty population
     fails [p_all2]) rather than ending the run before the later claims. *)
  let get ctx (t : Tensor.t) =
    try Context.get_values ctx t.value
    with Utils.User_error msg ->
      Stdio.eprintf "read of %s refused: %s (not part of the golden)\n%!" (Tn.debug_name t.value)
        msg;
      [||]
  in
  p_all2 "v recomputed equals v stored, bitwise" (get ctx v) (get mctx mv) ~f:Float.equal;
  p_all2 "v2 (reading the virtual v) recomputed equals v2 stored, bitwise" (get ctx v2)
    (get mctx mv2) ~f:Float.equal;
  p_all2 "w (with a dimension-1 axis) recomputed equals w stored, bitwise" (get ctx w) (get mctx mw)
    ~f:Float.equal;
  (* [r] reads [v2] whether or not the pipeline keeps [r] itself virtual; either way the read
     matches the materialized arm. *)
  p_close "the reduction r read back matches r stored" (get ctx r) (get mctx mr) ~tol:(fun want ->
      1e-6 *. Float.max 1. (Float.abs want));
  (* The oracle is independent of both arms: the closed form of v. *)
  p_all2 "v holds 1.5 x, cell by cell" (get ctx v)
    (Array.map x_values ~f:(fun a -> 1.5 *. a))
    ~f:Float.equal;
  let nd = Context.to_host ctx v.value in
  p "to_host returns the node's own layout"
    (Array.equal Int.equal (Ir.Ndarray.dims nd) (Lazy.force v.value.Tn.dims));
  (* --- The parent is left as it was --- *)
  let ledger = Context.ledger_entries ctx in
  let pools = (Ir.Alloc_census.snapshot ()).live_working_pools in
  let again = get ctx v2 in
  p "a repeated read agrees" (Array.equal Float.equal again (get mctx mv2));
  p "a read gives the parent no buffer for the node" (not (Context.mem ctx v2.value));
  p "repeated reads leave the parent's ledger as it was" (Context.ledger_entries ctx = ledger);
  p "and free the throwaway context's pools"
    ((Ir.Alloc_census.snapshot ()).live_working_pools = pools);
  (* --- Recompute-at-read --- a write to the leaf after the routine that deferred [v] is observed
     by the next read, where the materialized reading snapshots it at the routine's run. *)
  let ctx' = Context.set_values ctx x.value x_values' in
  p_all2 "after a write to x, v reads 1.5 x of the new x" (get ctx' v)
    (Array.map x_values' ~f:(fun a -> 1.5 *. a))
    ~f:Float.equal;
  p_all2 "the materialized arm keeps the value its run computed" (get mctx mv)
    (Array.map x_values ~f:(fun a -> 1.5 *. a))
    ~f:Float.equal;
  (* --- A failure after the recompute ran propagates, and the pools are still released --- *)
  let pools = (Ir.Alloc_census.snapshot ()).live_working_pools in
  let injected =
    Ir.Resource_fault_injection.with_callback
      (function Ir.Resource_fault_injection.To_host_before_copy -> failwith "injected" | _ -> ())
      ~f:(fun () ->
        match get ctx' v with _ -> false | exception Failure msg -> String.equal msg "injected")
  in
  p "a transfer failure during the recompute propagates" injected;
  p "the throwaway context's pools are released on that failure too"
    ((Ir.Alloc_census.snapshot ()).live_working_pools = pools);
  p "the parent context still reads afterwards" (Array.length (get ctx' v) = rows * cols);
  (* --- A release of the throwaway context that fails before freeing is retried --- the caller
     never held that context, so the next recompute retries it, and its ledger records are retired
     once it commits. *)
  let pools = (Ir.Alloc_census.snapshot ()).live_working_pools in
  let ledger = Context.ledger_entries ctx' in
  let failed_release =
    Ir.Resource_fault_injection.with_callback
      (function Ir.Resource_fault_injection.Finalize_before_free -> failwith "injected" | _ -> ())
      ~f:(fun () ->
        match get ctx' v with _ -> false | exception Failure msg -> String.equal msg "injected")
  in
  p "a release failing before it frees propagates from the read" failed_release;
  p "the next read succeeds" (Array.length (get ctx' v) = rows * cols);
  p "and its retry frees the pools the failed release kept"
    ((Ir.Alloc_census.snapshot ()).live_working_pools = pools);
  p "and retires the failed read's ledger records" (Context.ledger_entries ctx' = ledger)

(* --- Refusals, each naming its reason --- *)
let () =
  let inner = 32 in
  let x =
    TDSL.ndarray (Array.init inner ~f:Float.of_int) ~label:[ "rorl_x" ] ~output_dims:[ inner ] ()
  in
  let a =
    TDSL.ndarray
      (Array.init (4 * inner) ~f:(fun n -> Float.of_int ((n / inner) + 1)))
      ~label:[ "rorl_a" ] ~input_dims:[ inner ] ~output_dims:[ 4 ] ()
  in
  let%op h = a * x in
  let%op u = h + 3 in
  let%op out = u *. h in
  let ctx = Train.forward_once (Context.auto ()) out in
  p "the pipeline placed h Local and u virtual" (is_mode Tn.Local ctx h && is_mode Tn.Virtual ctx u);
  (match refusal (fun () -> Context.get_values ctx u.value) with
  | None -> p "reading u, which depends on the Local h, is refused" false
  | Some msg ->
      p "reading u, which depends on the Local h, is refused, naming the path and h's provenance"
        (says msg [ "u -> h"; "placed Local"; "39:inline-reduction-cap" ]));
  let fresh = TDSL.range_of_shape ~label:[ "rorl_fresh" ] ~output_dims:[ 3 ] () in
  (match refusal (fun () -> Context.get_values ctx fresh.value) with
  | None -> p "reading a node no routine of the lineage computes is refused" false
  | Some msg ->
      p "reading a node no routine of the lineage computes is refused as not computed"
        (says msg [ "not present in context"; "no computation stored" ]));
  let k, bindings = IDX.get_static_symbol IDX.empty in
  let%op e = !@k + 0.5 in
  let%op ey = e *. 2 in
  let lctx = Train.forward_once ~bindings ~launch_values:[ (k, 3) ] (Context.auto ()) ey in
  match refusal (fun () -> Context.get_values lctx e.value) with
  | None -> p "reading e, which embeds a launch index, is refused" false
  | Some msg ->
      p "reading e, which embeds a launch index, is refused, naming the index"
        (says msg [ "launch index"; Ir.Indexing.symbol_ident k.Ir.Indexing.static_symbol ])
