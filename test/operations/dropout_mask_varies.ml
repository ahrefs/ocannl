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

(* Printed, not claimed: a normal draw over an unbatched vector closes to one value broadcast over
   its use site, guess or not (also at 380ea6528, before gh-ocannl-2141): [box_muller] draws behind
   the literals of its clamp, and its equality spec ties the result's row to theirs
   (gh-ocannl-2180). *)
let () =
  let count name build =
    let v = values ~seed:7 build in
    Stdio.printf "%s: %d distinct values of %d\n" name
      (Set.length (Set.of_array (module Float) v))
      (Array.length v)
  in
  count "x *. normal_at step" (fun () ->
      let x = ones [ 16 ] in
      let%op y = x *. normal_at (step ()) in
      y);
  count "x *. normal ()" (fun () ->
      let x = ones [ 16 ] in
      let%op y = x *. normal () in
      y);
  count "x *. { w = normal () }" (fun () ->
      let x = ones [ 16 ] in
      let%op y = x *. { w = normal () } in
      y)
