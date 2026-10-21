(* gh-ocannl-1473: two fresh builds of one program emit the same kernel, up to the numbers minted
   into its identifiers.

   A fresh build of a program mints new node uids and new loop-symbol numbers, and nothing else
   about the program changes, so its kernel must differ from the first build's only in identifiers
   spelled with those numbers. Before gh-ocannl-1473 it did not: the kernel's parameters, Metal's
   pool-slot table and the local declarations followed a traversal of the kernel's node hash table,
   whose order follows the uids, and the per-parameter backends (cc, CUDA, HIP) then sorted their
   parameters by spelling, which reorders [n9999_w] and [n10006_w] against [n1000_w] and [n1007_w].
   [C_syntax.store_order] now emits them in the order the code first touches them.

   The probe is a gradient update over a small graph with two parameters sharing the label [w], so
   their identifiers carry their ids, and with every parameter's gradient materialized, so the
   kernel has a handful of pointer parameters -- enough for a hash-table order to show. Three fresh
   builds are compiled under one routine name: the first with the two [w]s at ids that share a digit
   count, the second with them straddling a power of ten (a string sort of their names flips there),
   the third wherever the id counter left off. The kernels are compared after renaming every minted
   identifier by its first appearance ([Test_utils.Generated.normalize_minted]). *)

open Base
open Stdio
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true

let build () =
  let layer x =
    let%op y = relu (({ w = [ 2 ] } *. x) + x) in
    (y, w)
  in
  let%op c = { a = [ -4 ] } + { b = [ 2 ] } in
  let c1, w1 = layer c in
  let c2, w2 = layer c1 in
  let%op g = (c2 - b) **. 2 in
  List.iter [ a; b; w1; w2 ] ~f:(fun p ->
      Option.iter p.Tensor.diff ~f:(fun diff -> Train.set_materialized diff.Tensor.grad));
  (g, w1, w2)

let digits id = String.length (Int.to_string id)

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  eprintf "fresh_build_kernel_text: on the %s backend (not part of the golden)\n%!" backend;
  Test_utils.Generated.init ~backend_name:backend;
  (* Where the first [w] lands relative to the id counter, and how far the second is from it: a dry
     build, never compiled. *)
  let offset, gap =
    let start = Tensor.get_next_id () in
    let _, w1, w2 = build () in
    (w1.value.id - start, w2.value.id - w1.value.id)
  in
  let routine = "fbk_step" in
  let compile_fresh ?first_w_at () =
    Option.iter first_w_at ~f:(fun id ->
        (* [bump_next_id k] makes the next id at least [k + 1]. *)
        Tensor.bump_next_id (id - offset - 1));
    let g, w1, w2 = build () in
    Test_utils.Generated.arm routine;
    let ctx = Train.init_params ctx Train.IDX.empty g in
    let _ctx, _routine =
      Context.compile ~name:routine ctx (Train.grad_update g) Ir.Indexing.Empty
    in
    ((w1.value.id, w2.value.id), Test_utils.Generated.read routine)
  in
  (* Sequenced by [let]: a list literal's elements are evaluated right to left. *)
  let builds =
    let shared_digits = compile_fresh ~first_w_at:1000 () in
    let straddling = compile_fresh ~first_w_at:9999 () in
    let left_off = compile_fresh () in
    [ shared_digits; straddling; left_off ]
  in
  let ids = List.map builds ~f:fst in
  p
    (Printf.sprintf
       "the first build's two w ids share a digit count, and the second's straddle a power of ten \
        (the second w is %d ids after the first)"
       gap)
    (match ids with
    | [ (a1, a2); (b1, b2); _ ] -> digits a1 = digits a2 && digits b1 < digits b2
    | _ -> false);
  let srcs = List.map builds ~f:snd in
  let normalized = List.map srcs ~f:Test_utils.Generated.normalize_minted in
  let first = List.hd_exn normalized in
  List.iteri normalized ~f:(fun i s ->
      if not (String.equal s first) then
        eprintf "fresh build 0 and fresh build %d emit different kernels:\n%s\n----\n%s\n" i
          (List.hd_exn srcs) (List.nth_exn srcs i));
  p_all ~min:3
    "in every build, both w parameters reach the kernel under identifiers spelling their ids" builds
    ~f:(fun ((w1, w2), src) ->
      let spelled id = String.is_substring src ~substring:(Printf.sprintf "n%d_w" id) in
      spelled w1 && spelled w2);
  p_all ~min:3 "every fresh build emits the same kernel up to minted identifiers" normalized
    ~f:(String.equal first)
