(* gh-ocannl-2030: [~deduced:Input_equals_output] equates a fresh shape's input row with its output
   row, and [Row.unify_row] can defer part of that equation: an unpadded window whose kernel is
   unsolved, or a window pair held for the Stage2+ fixpoint (gh-ocannl-2038). Those deferrals used
   to trip [assert (List.is_empty more_ineqs)] in [Shape.of_spec] (and [Shape.make]); now they join
   the active constraints and the closure solves them, or refuses them naming the deduction.

   Each case ties the deduced shape's output to concrete extents through [Shape.infer_equal] and
   reads the closed dims. The input's windowed or strided axis is then sized ONLY by the deferred
   equation: nothing else constrains its variables, so a dropped deferral leaves them free to close
   to 1, which the claims tell apart from the extent the equation gives. *)

open Base
open Ocannl
open Verdict.Claims

type outcome = Closed of int list | Refused of string

let outcome_to_string = function
  | Closed ds -> String.concat ~sep:"," (List.map ds ~f:Int.to_string)
  | Refused _ -> "refused"

(* Constructs the deduced shape for [spec], equates it with a shape of output [output] (and no batch
   axes) and closes it: [batch, output, input] dims (matrix convention), or the shape error's
   message. *)
let close ~spec ~output =
  Tensor.unsafe_reinitialize ();
  let result =
    match
      let sh = Shape.of_spec ~deduced:Input_equals_output ~debug_name:"deduced" ~id:(-1) spec in
      Shape.infer_equal sh
        (Shape.make ~batch_dims:[] ~output_dims:output ~debug_name:"other" ~id:(-1) ());
      (* A step over the shape, so that the closure applies its solution to the shape's rows. *)
      Shape.propagate_shapes
        (Shape.make_update_step ~shape:sh
           ~logic:(Shape.Terminal { is_param = false; logic = Fetch (Constant 0.) })
           ~neutral_elem:None);
      Shape.to_dims sh
    with
    | ds -> Closed (Array.to_list ds)
    | exception Row.Shape_error (msg, trace) -> Refused (Shape.error_to_string_hum msg trace)
  in
  Stdio.printf "%s, output %s: %s\n%!" spec
    (String.concat ~sep:"," (List.map output ~f:Int.to_string))
    (outcome_to_string result);
  (match result with
  | Refused msg -> Stdio.eprintf "(not part of the golden) %s: %s\n%!" spec msg
  | Closed _ -> ());
  result

let closes_to ~spec ~output expected =
  pf "%s: the input is sized by the deferred equation" spec
    (match close ~spec ~output with
    | Closed ds -> List.equal Int.equal ds expected
    | Refused _ -> false)

let refused_naming_the_deduction ~spec ~output what =
  pf "%s: %s, naming the deduction" spec what
    (match close ~spec ~output with
    | Refused msg -> String.is_substring msg ~substring:"input_equals_output"
    | Closed _ -> false)

let () =
  (* An unpadded window over a free kernel, [x = 2*(o-1) + k]: the plain [o] fixes [o = c = 4], the
     output's 9 then gives [k = 3], and [x] is 9 by the deferred equation alone. *)
  closes_to ~spec:"o, x -> c, 2*o<+k" ~output:[ 4; 9 ] [ 4; 9; 4; 9 ];
  (* Two unpadded windows: [2*(a-1) + j = 2*(o-1) + k], with [a = 3], [o = 4], [k = 3] from the
     output, so [j = 5] and the input window spans 9 (5 if [j] were left free). *)
  closes_to ~spec:"a, o, 2*a<+j -> d, c, 2*o<+k" ~output:[ 3; 4; 9 ] [ 3; 4; 9; 3; 4; 9 ];
  (* A held pair of strided axes (gh-ocannl-2038): [2*a = 3*o], [o = 4], so [a = 6]. *)
  closes_to ~spec:"2*a -> 3*o" ~output:[ 12 ] [ 12; 12 ];
  (* The same pair over an extent the input's stride does not divide: the deferred equation is
     refused at the closure, as a shape error, not an assertion failure. *)
  refused_naming_the_deduction ~spec:"2*a -> 3*o" ~output:[ 9 ]
    "the closure refuses a non-dividing stride";
  (* Nothing sizes the kernel nor the window's iterator, and the closure closes [x] before it
     settles the window: it refuses where [o = 1], [k = 9] would do. A loud refusal, not the
     assertion failure it used to be; which constraint it names is the closure's business. *)
  p "x -> 2*o<+k over 9: an underdetermined window is a shape error"
    (match close ~spec:"x -> 2*o<+k" ~output:[ 9 ] with Refused _ -> true | Closed _ -> false)
