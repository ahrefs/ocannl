(* gh-ocannl-1737: printing a tree recomputes all of its virtual nodes through ONE throwaway
   routine, where gh-ocannl-777's per-read recompute compiled one per node.

   The routine count is read through [Context.backend_compiles], against a unit measured here rather
   than assumed: one single-node [Context.to_host] of a virtual node is exactly one recompute
   routine, whatever a routine costs on this backend (a Metal routine also counts its link-time
   library compile). Separate reads of the tree's virtual nodes cost that unit once per node -- the
   negative control that shows the counter tells one routine from several -- and the printed tree
   costs it once.

   Values: one program, built twice -- intermediates left to the pipeline (virtual, recomputed at
   the read) and materialized (read from their buffers). [x] holds a distinct integer in every cell
   and each intermediate is a distinct polynomial of its cell's value, so a wrong cell, a broadcast
   or an unrun copy cannot match; the arithmetic is exact in single precision, so the claims are
   bitwise. The printed tree (the golden) holds the same exact values a node-by-node read printed
   before this change. [~depth] bounds only the boxed layout -- the levels below it are drawn as an
   indented tree, values included -- so a depth-1 print names, and reads, every node, still in one
   routine. A write to [x] after the run is seen by the next shared read (recompute-at-read,
   gh-617).

   Refusals keep their classification: a node depending on [Local] scratch prints the same reason
   tag inside the shared read, and [Context.to_host_all] raises the very refusal [Context.to_host]
   raises for it, before compiling anything. Runs on the backend [Context.auto] picks. *)

(* cc-fp-contract: exact-operands -- integer inputs and small integer polynomials, exact in f32 *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
open Verdict.Claims

let rows = 2
let cols = 3

(* Distinct per cell: [x.(i, j) = 3i + j + 1]. Every intermediate is an integer below 1000, so the
   printed tree shows its values exactly (three significant digits): no decimal tie to round. *)
let x_values = Array.init (rows * cols) ~f:(fun n -> Float.of_int (n + 1))
let x_values' = Array.map x_values ~f:(fun v -> v +. 1.)

let is_mode m ctx (t : Tensor.t) =
  Option.equal Tn.equal_memory_mode
    (Option.map ~f:fst (Tn.Placements.get (Context.placements ctx) t.value))
    (Some m)

(* The program; [~materialize] is the one difference between the two arms. *)
let build ~materialize =
  (* Data-backed: the host writes it after the forward is compiled, which a literal this small, a
     fetch, refuses (gh-ocannl-1959). *)
  let x =
    Operation.init ~l:"pt_x" ~prec:Ir.Ops.single ~o:[ rows; cols ]
      ~f:(function [| i; j |] -> x_values.((i * cols) + j) | _ -> assert false)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  (* 3x, 3x^2 + 1, 3x^2 - 2x + 1, 3x^2 + x + 1, 3x^2 + 2x + 1. *)
  let%op v1 = (x *. 2) + x in
  let%op v2 = (v1 *. x) + 1 in
  let%op v3 = v2 - (x *. 2) in
  let%op v4 = v3 + v1 in
  let%op v5 = (v4 *. 2) - v2 in
  let%op out = v5 + 1 in
  let chain = [ v1; v2; v3; v4; v5 ] in
  if materialize then List.iter chain ~f:(fun (t : Tensor.t) -> Train.set_materialized t.value);
  let ctx = Train.forward_once (Context.auto ()) out in
  (ctx, x, chain, out)

let compiles_of f =
  let before = Context.backend_compiles () in
  let result = f () in
  (result, Context.backend_compiles () - before)

(* Every cell of every array, in order: one population for a bitwise [p_all2]. *)
let cells nds = Array.concat (List.map nds ~f:Ir.Ndarray.retrieve_flat_values)

let () =
  let ctx, x, chain, out = build ~materialize:false in
  let mctx, _, mchain, _ = build ~materialize:true in
  let virtual_ = List.filter chain ~f:(is_mode Tn.Virtual ctx) in
  p "the pipeline leaves at least four of the tree's five intermediates virtual"
    (List.length virtual_ >= 4);
  p_all "each of them is read by recompute" virtual_ ~f:(fun (t : Tensor.t) ->
      match Context.observability ctx t.value with Recomputable -> true | _ -> false);
  (* The unit: one recompute routine, as a single-node read compiles it. *)
  let (_ : Ir.Ndarray.t), unit_ =
    compiles_of (fun () -> Context.to_host ctx (List.hd_exn virtual_).value)
  in
  p "a single-node recompute compiles" (unit_ > 0);
  let separate, separate_cost =
    compiles_of (fun () -> List.map virtual_ ~f:(fun (t : Tensor.t) -> Context.to_host ctx t.value))
  in
  p "separate reads compile one recompute routine per virtual node"
    (separate_cost = unit_ * List.length virtual_);
  let ledger = Context.ledger_entries ctx in
  let pools = (Ir.Alloc_census.snapshot ()).live_working_pools in
  let shared, shared_cost =
    compiles_of (fun () -> Context.to_host_all ctx (List.map chain ~f:(fun t -> t.Tensor.value)))
  in
  p "to_host_all over the chain compiles one recompute routine" (shared_cost = unit_);
  let separate_all = List.map chain ~f:(fun t -> Context.to_host ctx t.Tensor.value) in
  p_all2 "the shared read equals the separate reads, bitwise" (cells shared) (cells separate_all)
    ~f:Float.equal;
  p_all2 "and the separate reads of the virtual nodes agree with it, bitwise" (cells separate)
    (cells
       (List.filter_map (List.zip_exn chain shared) ~f:(fun (t, nd) ->
            if List.mem virtual_ t ~equal:phys_equal then Some nd else None)))
    ~f:Float.equal;
  let materialized =
    Array.concat (List.map mchain ~f:(fun t -> Context.get_values mctx t.Tensor.value))
  in
  p_all2 "the shared read equals the materialized arm, bitwise" (cells shared) materialized
    ~f:Float.equal;
  p "the shared read leaves the parent's ledger as it was" (Context.ledger_entries ctx = ledger);
  p "and frees the throwaway context's pools"
    ((Ir.Alloc_census.snapshot ()).live_working_pools = pools);
  p_all "and gives the parent no buffer for any virtual node" virtual_ ~f:(fun (t : Tensor.t) ->
      not (Context.mem ctx t.value));
  (* The printed tree: one routine for the whole of it, and the values the golden pins. *)
  let (), print_cost = compiles_of (fun () -> Train.printf_tree ~with_grad:false ctx out) in
  Stdio.print_endline "";
  p "printing the tree compiles one recompute routine" (print_cost = unit_);
  (* [~depth] bounds only the boxed layout: the levels below it are drawn as an indented tree, with
     their values, so a shallow print shows -- and reads -- every node, in the same one routine. *)
  let shallow, shallow_cost =
    compiles_of (fun () ->
        PrintBox_text.to_string
          (Tensor.to_printbox ~with_id:true ~ctx ~with_grad:false ~depth:1 out))
  in
  p "printing at depth 1 compiles one recompute routine too" (shallow_cost = unit_);
  p_all "and names every node of the graph, past depth 1 included (the golden shows their values)"
    (x :: out :: chain) ~f:(fun (t : Tensor.t) ->
      String.is_substring shallow ~substring:(Printf.sprintf "#%d %s" t.value.id (Tn.label t.value)));
  Stdio.print_string shallow;
  Stdio.print_endline "";
  (* Recompute-at-read: a write to the leaf is seen by the next shared read. *)
  let ctx' = Context.set_values ctx x.value x_values' in
  let v1 = List.hd_exn chain in
  p_all2 "after a write to x, the shared read sees v1 = 3 x of the new x"
    (cells (Context.to_host_all ctx' [ v1.value ]))
    (Array.map x_values' ~f:(fun a -> 3. *. a))
    ~f:Float.equal

(* --- A refusal keeps its classification inside a shared read --- *)
let () =
  let inner = 32 in
  let x =
    TDSL.ndarray (Array.create ~len:inner 0.25) ~label:[ "ptl_x" ] ~output_dims:[ inner ] ()
  in
  let a =
    TDSL.ndarray
      (Array.init (2 * inner) ~f:(fun n -> Float.of_int ((n / inner) + 1)))
      ~label:[ "ptl_a" ] ~input_dims:[ inner ] ~output_dims:[ 2 ] ()
  in
  let%op h = a * x in
  let%op u = h + 3 in
  let y = TDSL.ndarray [| 1.; 2. |] ~label:[ "ptl_y" ] ~output_dims:[ 2 ] () in
  let%op w = (y *. 2) + y in
  let%op out = (u *. h) + w in
  let ctx = Train.forward_once (Context.auto ()) out in
  p "the pipeline placed h Local and u, w virtual"
    (is_mode Tn.Local ctx h && is_mode Tn.Virtual ctx u && is_mode Tn.Virtual ctx w);
  let refusal f = match f () with _ -> None | exception Utils.User_error msg -> Some msg in
  let alone = refusal (fun () -> Context.to_host ctx u.value) in
  let before = Context.backend_compiles () in
  let shared = refusal (fun () -> Context.to_host_all ctx [ w.value; u.value ]) in
  p "to_host_all refuses u with the refusal to_host gives it"
    (Option.is_some alone && Option.equal String.equal alone shared);
  p "and refuses before compiling anything" (Context.backend_compiles () = before);
  (* The printed tree shows [u]'s and [h]'s reasons and [w]'s recomputed values. *)
  Stdio.print_endline "";
  Train.printf_tree ~with_grad:false ctx out;
  Stdio.print_endline ""
