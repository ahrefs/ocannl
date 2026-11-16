(* Regression test for forward-code fragment ordering (computed-after-use): three matmuls consume
   one shared layer_norm output. The layer_norm chain is embedded in the forward code of its
   first-constructed consumer (q), while sibling subtree ids follow construction order -- which for
   `(f q + f k) + f v` is v, k, q (OCaml applications evaluate right-to-left). Before the
   topological ordering of forward fragments in Tensor.op, the id-ascending emission ran the k and v
   matmuls before the layer_norm chain, so they read zeros: k and v printed all zeros. Correct
   output: k = 2*q and v = 3*q (the weights are scaled copies), with q nonzero. *)
open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let%op layer_norm ~epsilon () ~gamma ~beta x =
  let mean = (x ++ " ... | ..d.. => ... | 0 " [ "d" ]) /. dim d in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ " ... | ... => ... | 0 ") /. dim d in
  (gamma *. (centered /. sqrt (variance + !.epsilon))) + beta

let () =
  let seq = 3 and dm = 4 and heads = 2 and dh = 2 in
  let x0 =
    NTDSL.init ~l:"x0" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; m |] -> if m = t % dm then 0.1 *. Float.of_int (t + 1) else 0.05 | _ -> assert false)
      ()
  in
  let g = NTDSL.init ~l:"g" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 1.) () in
  let bta = NTDSL.init ~l:"bta" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 0.) () in
  let mk_w name c =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[ dm ] ~o:[ heads; dh ]
      ~f:(function
        | [| h; j; m |] ->
            c *. (1. +. (0.1 *. Float.of_int ((h * 2) + j)) +. (0.01 *. Float.of_int m))
        | _ -> assert false)
      ()
  in
  let wq = mk_w "wq" 0.1 and wk = mk_w "wk" 0.2 and wv = mk_w "wv" 0.3 in
  let l = layer_norm ~epsilon:1e-5 () ~gamma:g ~beta:bta x0 in
  let%op q = wq * l in
  let%op k = wk * l in
  let%op v = wv * l in
  Train.set_materialized q.Tensor.value;
  Train.set_materialized k.Tensor.value;
  Train.set_materialized v.Tensor.value;
  let%op total = q ++ "...|... => |->0" + (k ++ "...|... => |->0") + (v ++ "...|... => |->0") in
  let ctx = Train.forward_once (Context.auto ()) total in
  let p name t =
    let vals = Context.get_values ctx t.Tensor.value in
    printf "%s:" name;
    Array.iter vals ~f:(fun v -> printf " %.6f" v);
    printf "\n%!"
  in
  p "q" q;
  p "k" k;
  p "v" v

(* Fragment-level dependency cycle: a2 embeds the computed l's code (via a1) but reads the computed
   p; b2 embeds p's code (via b1) but reads l. Statement-level interleaving (l; a1; p; b1; a2; b2)
   could schedule this, but whole-fragment ordering cannot -- any order reads some value before it
   is computed. The compiler errs on the sound side and raises at construction. Note the shared
   nodes must be COMPUTED tensors: data-backed terminals are uploaded at link time and correctly
   create no ordering constraints. *)
let () =
  let mk name =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 2 ]
      ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
      ()
  in
  let base = mk "base2" and base' = mk "base2b" in
  let open NTDSL.O in
  let l = base *. base in
  let p = relu base' in
  let a1 = l *. l in
  let b1 = relu p in
  let a2 = a1 *. p in
  let b2 = b1 *. l in
  match a2 + b2 with
  | exception Ocannl_tensor.Tensor.Session_error (msg, _) ->
      Verdict.p "fragment cycle raises Session_error"
        (String.is_substring msg ~substring:"dependency cycle")
  | _ -> printf "fragment cycle raises Session_error: false\n"

(* A slice reads its parent's value (gh-ocannl-2205). [parent]'s code is embedded in [owner], its
   first consumer, then in [later]'s fragment; the slice [reader] is constructed before [later], so
   by id its fragment comes first. Its only read of [parent] is the [Slice] itself: the ordering
   must still put [later]'s fragment, which computes [parent], first. The slice is forced to copy (a
   precision mismatch) and both it and [parent] are materialized, so the copy runs as its own loop
   reading [parent]'s buffer, rather than being inlined into [out] past [later]'s fragment. *)
let () =
  let index, bindings = Train.IDX.get_static_symbol ~static_range:2 Train.IDX.empty in
  let data =
    NTDSL.init ~l:"ffo_slice_data" ~prec:Ir.Ops.single ~b:[ 2 ] ~i:[] ~o:[ 4 ]
      ~f:(function [| b; j |] -> Float.of_int ((10 * b) + j + 1) | _ -> assert false)
      ()
  in
  let parent = NTDSL.pointmul data (NTDSL.number 2.) () in
  let owner = NTDSL.pointmul parent (NTDSL.number 3.) () in
  let reader = NTDSL.slice index parent () in
  let later = NTDSL.add owner (NTDSL.number 1.) () in
  let out = NTDSL.add reader later () in
  Ir.Tnode.update_prec reader.Tensor.value Ir.Ops.half;
  List.iter [ parent; reader; out ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let base = Context.auto () in
  let ctx, routine =
    Context.compile ~name:"ffo_slice_order" base (Tensor.consume_forward_code out) bindings
  in
  Train.IDX.find_exn routine.bindings index := 1;
  let ctx = Context.run ctx routine in
  Verdict.p "the slice copies rather than aliasing its parent"
    (not
       (Ir.Assignments.slice_alias_eligible ~array:reader.Tensor.value ~sliced:parent.Tensor.value));
  (* out[b, j] = 2 * (10 + j + 1) + 3 * 2 * (10 * b + j + 1) + 1 = 29 + 60 b + 8 j. *)
  Verdict.p "a slice reader runs after the fragment computing its parent"
    (Array.equal Float.equal
       (Context.get_values ctx out.Tensor.value)
       (Array.init 8 ~f:(fun i -> Float.of_int (29 + (60 * (i / 4)) + (8 * (i % 4))))));
  Context.release ctx;
  Context.release base
