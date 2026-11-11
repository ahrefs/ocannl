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
open Operation.DSL_modules
open Verdict.Claims

type outcome = Closed of int list | Refused of string

let outcome_to_string = function
  | Closed ds -> String.concat ~sep:"," (List.map ds ~f:Int.to_string)
  | Refused _ -> "refused"

(* Constructs the deduced shape for [spec], equates it with a shape of output [output] (and no batch
   axes) and closes it: [batch, output, input] dims (matrix convention), or the shape error's
   message. *)
let close ?(is_param = false) ~spec ~output () =
  Tensor.unsafe_reinitialize ();
  let result =
    match
      let sh = Shape.of_spec ~deduced:Input_equals_output ~debug_name:"deduced" ~id:(-1) spec in
      Shape.infer_equal sh
        (Shape.make ~batch_dims:[] ~output_dims:output ~debug_name:"other" ~id:(-1) ());
      (* A step over the shape, so that the closure applies its solution to the shape's rows. *)
      Shape.propagate_shapes
        (Shape.make_update_step ~shape:sh
           ~logic:(Shape.Terminal { is_param; logic = Fetch (Constant 0.) })
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
    (match close ~spec ~output () with
    | Closed ds -> List.equal Int.equal ds expected
    | Refused _ -> false)

let refused_naming_the_deduction ~spec ~output what =
  pf "%s: %s, naming the deduction" spec what
    (match close ~spec ~output () with
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
  (* Neither the kernel nor the iterator is sized explicitly: closing the iterator at 1 lets the
     extent 9 infer the kernel as 9. The plain input must wait for that equation. *)
  closes_to ~spec:"x -> 2*o<+k" ~output:[ 9 ] [ 9; 9 ];
  closes_to ~spec:"2*o<+k -> x" ~output:[ 9 ] [ 9; 9 ];
  closes_to ~spec:"x -> 2*o<+2*k" ~output:[ 9 ] [ 9; 9 ]

(* Execute the same deduction through a real terminal, then reduce its inferred input axis. Its
   row-major offsets discriminate both axes: a wrongly broadcast input cannot pass. The context is
   explicitly CPU because the regression is shape closure and projection derivation. *)
let () =
  case "a forward run uses both inferred axes" (fun () ->
      Tensor.unsafe_reinitialize ();
      let sh =
        Shape.of_spec ~deduced:Input_equals_output ~debug_name:"deduced_forward" ~id:(-1)
          "x -> 2*o<+k"
      in
      let t =
        NTDSL.term ~label:[ "deduced_forward" ] ~batch_dims:[] ~fetch_op:Range_over_offsets ()
      in
      Shape.infer_equal t.Tensor.shape sh;
      Shape.infer_equal sh
        (Shape.make ~batch_dims:[] ~output_dims:[ 9 ] ~debug_name:"other" ~id:(-1) ());
      p "the terminal closes with two axes of 9"
        (Array.equal Int.equal (Shape.to_dims t.shape) [| 9; 9 |]);
      let reduced = NTDSL.einsum1 "i -> j => i" t () in
      Train.set_materialized reduced.Tensor.value;
      let ctx = Train.forward_once (Context.cpu ()) reduced in
      pf_close "each output sums its own nine input offsets"
        (Context.get_values ctx reduced.value)
        (Array.init 9 ~f:(fun i -> Float.of_int (324 + (9 * i))))
        ~tol:(fun _ -> 0.))

(* A kernel can also be its iterator, or two windows can use each other's kernels as iterators.
   These stores already close on the base solver: all extents carry the broadcast top explicitly, so
   guessing the cycle's parts at 1 satisfies sizes AND bases. *)
let closes_cycle ~fixed_extent ~mutual ~reverse =
  Tensor.unsafe_reinitialize ();
  let x = Row.get_var () and a = Row.get_var () and b = Row.get_var () in
  let y = if mutual then Row.get_var () else x in
  let b = if mutual then b else a in
  let dim v = Row.Var v in
  let window over kernel =
    Row.Affine
      {
        stride = 2;
        over = dim over;
        conv = Some { dilation = 1; kernel = dim kernel; use_padding = false };
        stride_offset = 0;
      }
  in
  let eq d1 d2 = Row.Dim_eq { d1; d2; origin = [] } in
  let top = Row.get_bcast_dim ~d:1 () in
  let vars = if mutual then [ x; y; a; b ] else [ x; a ] in
  let row : Row.t =
    {
      beg_dims = [];
      dims = List.map vars ~f:dim;
      bcast = Broadcastable;
      prov = Row.provenance ~sh_id:(-1) ~kind:`Output;
    }
  in
  let kernel_x, kernel_y = if fixed_extent then (b, a) else (y, x) in
  let first = window a kernel_x and second = window b kernel_y in
  let constraints =
    [ Row.Shape_row (row, []); eq (dim x) first ]
    @ (if mutual then [ eq (dim y) second ] else [])
    @ if fixed_extent then eq first top :: (if mutual then [ eq second top ] else []) else []
  in
  let constraints = if reverse then List.rev constraints else constraints in
  match
    List.fold [ Row.Stage1; Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ]
      ~init:(constraints, Row.empty_env) ~f:(fun (held, env) stage ->
        Row.solve_inequalities ~stage held env)
  with
  | held, env ->
      List.is_empty held
      && List.for_all vars ~f:(fun v ->
          match Row.get_dim_from_env env v with
          | Some (Dim { d = 1; basis; _ }) -> String.equal basis Row.bcast_if_1
          | _ -> false)
  | exception Row.Shape_error _ -> false

let () =
  p_all "a self-dependent kernel still closes, in either constraint order" [ false; true ]
    ~f:(fun reverse -> closes_cycle ~fixed_extent:true ~mutual:false ~reverse);
  p_all "mutually dependent kernels still close, in either constraint order" [ false; true ]
    ~f:(fun reverse -> closes_cycle ~fixed_extent:true ~mutual:true ~reverse);
  p_all "a variable-extent self-cycle still closes, in either constraint order" [ false; true ]
    ~f:(fun reverse -> closes_cycle ~fixed_extent:false ~mutual:false ~reverse);
  p_all "a variable-extent mutual cycle still closes, in either constraint order" [ false; true ]
    ~f:(fun reverse -> closes_cycle ~fixed_extent:false ~mutual:true ~reverse)

let parameter_window ?output_dims () =
  let x = NTDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ 9 ] () in
  let w = TDSL.param "unsized_kernel" () in
  (TDSL.einsum "2*o<+k; k => o" x w ?output_dims (), w)

let () =
  p "a parameter extent tied to an open window still names its hidden dimension"
    (match close ~is_param:true ~spec:"x -> 2*o<+k" ~output:[ 9 ] () with
    | Refused msg -> String.is_substring msg ~substring:"hidden dimension"
    | Closed _ -> false);
  case "an unsized parameter kernel" (fun () ->
      Tensor.unsafe_reinitialize ();
      let outcome =
        match
          let r, _ = parameter_window () in
          Shape.to_dims r.Tensor.shape
        with
        | _ -> false
        | exception Row.Shape_error (msg, trace) ->
            Stdio.eprintf "(not part of the golden) parameter kernel: %s\n%!"
              (Shape.error_to_string_hum msg trace);
            String.is_substring msg ~substring:"hidden dimension"
      in
      p "an open iterator cannot choose a parameter kernel's hidden dimension" outcome);
  case "a sized iterator determines a parameter kernel" (fun () ->
      Tensor.unsafe_reinitialize ();
      let r, w = parameter_window ~output_dims:[ 4 ] () in
      p "a specified iterator still infers the parameter kernel as 3"
        (Array.equal Int.equal (Shape.to_dims r.Tensor.shape) [| 4 |]
        && Array.equal Int.equal (Shape.to_dims w.Tensor.shape) [| 3 |]))
