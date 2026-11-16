open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* gh-ocannl-2185: a random draw sized only by its use site takes the use site's size. A dropout
   mask behind the literal of its comparison ([!.rate < uniform_at step]) closed to one value
   broadcast over its input once the dimension solver stopped equating two variables that bound only
   each other (gh-ocannl-2141): the mask kept or dropped every entry, and nothing executed noticed.
   Each case draws over an input of ones and claims the mask keeps some entries and drops others,
   and that a raw draw under the same use site is distinct per element. *)

let values ~seed build =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed ();
  let y = build () in
  let ctx = Ocannl.Train.forward_once (Context.auto ()) y in
  let v = Context.get_values ctx y.Tensor.value in
  Context.release ctx;
  v

let ones dims =
  Tensor.term_init ~grad_spec:Tensor.Prohibit_grad
    (Array.create ~len:(List.fold dims ~init:1 ~f:( * )) 1.0)
    ()

let ones_bo ~b ~o =
  Ocannl.Operation.init ~l:"x" ~prec:Ir.Ops.single ~b:[ b ] ~o:[ o ]
    ~f:(fun _ -> 1.0)
    ~grad_spec:Tensor.Prohibit_grad ()

let step () = NTDSL.number 3.
let dropout x = Ocannl.Nn_blocks.dropout ~rate:0.5 () ~train_step:(Some (step ())) x

let mixed name v =
  p (name ^ ": at least 8 entries") (Array.length v >= 8);
  p_exists (name ^ ": the mask keeps some entries") (Array.to_list v) ~f:(fun x -> Float.(x > 0.));
  p_exists (name ^ ": the mask drops some entries") (Array.to_list v) ~f:(fun x -> Float.(x = 0.))

let distinct name v =
  Stdio.eprintf "%s: %d distinct of %d (not part of the golden)\n" name
    (Set.length (Set.of_array (module Float) v))
    (Array.length v);
  p
    (name ^ ": every element draws its own value")
    (Array.length v >= 8 && Set.length (Set.of_array (module Float) v) = Array.length v)

let () =
  Verdict.case "Nn_blocks.dropout over a 16-vector" (fun () ->
      List.iter [ 7; 8 ] ~f:(fun seed ->
          mixed ("seed " ^ Int.to_string seed) (values ~seed (fun () -> dropout (ones [ 16 ])))));
  Verdict.case "Nn_blocks.dropout over a batch of 4 by 8 outputs" (fun () ->
      mixed "batch 4, output 8" (values ~seed:7 (fun () -> dropout (ones_bo ~b:4 ~o:8))));
  Verdict.case "the comparison behind dropout, written out" (fun () ->
      mixed "x *. (0.5 < uniform_at step)"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. (0.5 < uniform_at (step ())) in
             y)));
  Verdict.case "a raw draw sized by its use site" (fun () ->
      distinct "x *. uniform_at step"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. uniform_at (step ()) in
             y));
      (* Not a whole number of blocks: the counter takes 3 blocks of 4, the last one partly. *)
      distinct "x[10] *. uniform_at step"
        (values ~seed:7 (fun () ->
             let x = ones [ 10 ] in
             let%op y = x *. uniform_at (step ()) in
             y));
      distinct "x *. (uniform_at step + 1)"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. (uniform_at (step ()) + 1) in
             y));
      (* Siblings: a draw behind another literal, or behind a scalar operand. *)
      distinct "x *. (uniform_at step - 0.5)"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. (uniform_at (step ()) - 0.5) in
             y));
      distinct "x *. (2 *. uniform_at step)"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. (2. *. uniform_at (step ())) in
             y)))

(* gh-ocannl-2180: a normal draw over an unbatched vector closed to one value broadcast over its use
   site: [box_muller] draws behind the literals of its clamp, a packed draw's row took its bound's
   axes, and those closed minimal, as the results' axes they were. A parameter drawing the same way
   is the control. *)
let () =
  Verdict.case "a normal draw sized by its use site" (fun () ->
      distinct "x *. normal_at step"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. normal_at (step ()) in
             y));
      distinct "x *. normal ()"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. normal () in
             y));
      distinct "x *. { w = normal () }"
        (values ~seed:7 (fun () ->
             let x = ones [ 16 ] in
             let%op y = x *. { w = normal () } in
             y)))

(* gh-ocannl-2141: what the raw draw's values rest on. [uniform_at] hashes each block's offset
   ([range_over_offsets], a leaf) with the key into the block counter, which the draw sizes from its
   use site only in the final stage. The leaf, bounded below by the counter alone, must close after
   it: closed first, it closed to 1, every block read offset 0, and 16 elements drew 4 values. Built
   here by hand, as [Operation.uniform_at] builds it, to see the leaf and the counter. *)
let () =
  Verdict.case "a draw's offsets take its counter's size" (fun () ->
      List.iter [ 16; 10 ] ~f:(fun n ->
          Tensor.unsafe_reinitialize ();
          Tensor.set_random_seed ~seed:7 ();
          let x = ones [ n ] in
          let key =
            Ocannl.Operation.threefry4x32
              (Ocannl.Operation.threefry4x32
                 (Ocannl.Operation.embed_self_id ())
                 (Tensor.get_random_seed ()) ())
              (step ()) ()
          in
          let offsets =
            Tensor.term ~fetch_op:Range_over_offsets ~grad_spec:Prohibit_grad
              ~label:[ "range_over_offsets" ] ()
          in
          let counter = Ocannl.Operation.threefry4x32 key offsets () in
          let y =
            Ocannl.Operation.pointmul x (Ocannl.Operation.uint4x32_to_prec_uniform counter ()) ()
          in
          let ctx = Ocannl.Train.forward_once (Context.auto ()) y in
          let numel t = Ir.Tnode.num_elems t.Tensor.value in
          let name = "over " ^ Int.to_string n in
          p (name ^ ": the counter holds one block per 4 elements") (numel counter = (n + 3) / 4);
          p (name ^ ": the offsets leaf takes the counter's size") (numel offsets = numel counter);
          Context.release ctx))
