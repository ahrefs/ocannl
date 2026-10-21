(* gh-ocannl-1301: a closed shape's committed content -- rows (modulo projection ids), paddings and
   padding neutral -- does not change after the [finish_inference] that closed it. A later operation
   whose inference would change it is refused before anything is written, whether the closed tensor
   was compiled or only closed (its dims never forced); equal writes stay allowed. Operand writes
   beyond the shape follow the same rule: a closed operand's precision, and a closed tensor's
   parameterization. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row

let outcome f =
  match f () with
  | (_ : 'a) -> Ok ()
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg
  | exception Utils.User_error msg -> Error msg
  | exception Failure msg -> Error msg

let refused_with f ~substrings =
  match outcome f with
  | Ok () -> false
  | Error msg ->
      Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
      (* Every refusal names something: an empty list would pass any refusal. *)
      (not (List.is_empty substrings))
      && List.for_all substrings ~f:(fun substring -> String.is_substring msg ~substring)

let leaf name values =
  Tensor.term_init values ~label:[ name ] ~grad_spec:Tensor.Prohibit_grad
    ~output_dims:[ Array.length values ]
    ()

let input name = leaf name (Array.init 8 ~f:(fun i -> Float.of_int (i + 1)))
let kernel name = leaf name [| 1.; 1.; 1. |]

(* Everything the guard protects, in a comparable form: rows modulo projection ids
   ([to_string_hum]'s default style omits them), the three paddings and the neutral. *)
let snapshot (t : Tensor.t) =
  let sh = t.Tensor.shape in
  String.concat ~sep:" | "
    [
      Shape.to_string_hum sh;
      Sexp.to_string (Shape.sexp_of_padding sh.Shape.batch_padding);
      Sexp.to_string (Shape.sexp_of_padding sh.Shape.input_padding);
      Sexp.to_string (Shape.sexp_of_padding sh.Shape.output_padding);
      Sexp.to_string (Option.sexp_of_t (Option.sexp_of_t Float.sexp_of_t) sh.Shape.padding_elem);
    ]

let padded_conv x k =
  let%op c = x +* "o=+k; k => o" k in
  c

(* [close] finalizes the padded convolution's inference: by compiling it, or (to pin the guard at
   the closure itself, before any lowering forces the operand's node) by reading its dims. *)
let refused_unchanged name ~close ~substrings ~ctx x k =
  let before = snapshot x in
  pf "%s: a padded convolution over it is refused" name
    (refused_with (fun () -> close ctx (padded_conv x k)) ~substrings);
  pf "%s: its rows, paddings and padding neutral are unchanged" name
    (String.equal before (snapshot x));
  Tensor.rewind_to_last_closure ();
  let%op d = x *. 2. in
  let ctx = Train.forward_once ctx d in
  pf_all2 "%s: after the rewind, a valid consumer compiles and runs" name
    (Context.get_values ctx d.Tensor.value)
    (Array.init 8 ~f:(fun i -> 2. *. Float.of_int (i + 1)))
    ~f:Float.equal;
  ctx

let () =
  let ctx = Context.auto () in
  (* (a) A compiled operand: its buffer layout is committed. *)
  let x = input "cg_x_compiled" in
  let ctx = Train.forward_once ctx x in
  let ctx =
    refused_unchanged "compiled operand"
      ~close:(fun ctx c -> ignore (Train.forward_once ctx c : Context.t))
      ~substrings:[ "already committed" ] ~ctx x (kernel "cg_k1")
  in
  (* (b) A closed operand whose dims were never forced: the closure alone commits it. *)
  let x2 = input "cg_x_closed" in
  ignore (Shape.to_dims x2.Tensor.shape : int array);
  p "the closed operand's node was never forced" (not (Lazy.is_val x2.Tensor.value.Ir.Tnode.dims));
  let ctx =
    refused_unchanged "closed, uncompiled operand"
      ~close:(fun _ctx c -> ignore (Shape.to_dims c.Tensor.shape : int array))
      ~substrings:[ "is closed (committed by an earlier finish_inference)"; "cg_x_closed" ]
      ~ctx x2 (kernel "cg_k2")
  in
  (* (c) Equal writes: a padded consumer committed in the same closure as its operand, then a second
     identical padded consumer of the now-closed operand -- the margins it needs are already
     there. *)
  let x3 = input "cg_x_padded" in
  Train.set_materialized x3.value;
  let c1 = padded_conv x3 (kernel "cg_k3") in
  let ctx = Train.forward_once ctx c1 in
  let c2 = padded_conv x3 (kernel "cg_k4") in
  let ctx = Train.forward_once ctx c2 in
  p_all2 "a second padded consumer of a closed, already-padded operand runs (equal writes)"
    (Context.get_values ctx c2.Tensor.value)
    (Context.get_values ctx c1.Tensor.value)
    ~f:Float.equal;
  let%op e = x3 + 1. in
  let ctx = Train.forward_once ctx e in
  p_all2 "a pointwise consumer of the closed operand runs (equal row writes)"
    (Context.get_values ctx e.Tensor.value)
    (Array.init 8 ~f:(fun i -> Float.of_int (i + 2)))
    ~f:Float.equal;
  (* (d) Operand precision: a [uint4x32] operation over a closed float operand is refused before
     writing, and does not taint the session. *)
  let f = leaf "cg_float" [| 1.; 2. |] in
  ignore (Shape.to_dims f.Tensor.shape : int array);
  let prec_before = Ir.Tnode.get_specified_prec f.Tensor.value in
  p "a uint4x32 operation over a closed float operand is refused"
    (refused_with
       (fun () -> Ocannl.Operation.uint4x32_to_prec_uniform1 f ())
       ~substrings:[ "is closed"; "uint4x32" ]);
  p "the operand's precision is unchanged"
    (Option.equal Ir.Ops.equal_prec prec_before (Ir.Tnode.get_specified_prec f.Tensor.value));
  let open_operand = leaf "cg_open" [| 1.; 2. |] in
  let open_before = Ir.Tnode.get_specified_prec open_operand.Tensor.value in
  p "a two-operand uint4x32 operation with a closed float operand is refused"
    (refused_with
       (fun () -> Ocannl.Operation.threefry4x32_crypto open_operand f ())
       ~substrings:[ "is closed"; "uint4x32" ]);
  p "and its open operand's precision was not written either"
    (Option.equal Ir.Ops.equal_prec open_before
       (Ir.Tnode.get_specified_prec open_operand.Tensor.value));
  let open2 = leaf "cg_open2" [| 1.; 2. |] in
  let open2_before = Ir.Tnode.get_specified_prec open2.Tensor.value in
  p "with a compiled (settled-precision) float operand too, before writing the open one"
    (refused_with
       (fun () -> Ocannl.Operation.threefry4x32_crypto open2 x ())
       ~substrings:[ "uint4x32" ]
    && Option.equal Ir.Ops.equal_prec open2_before (Ir.Tnode.get_specified_prec open2.Tensor.value)
    );
  p "the refusal did not taint the session"
    (Result.is_ok (outcome (fun () -> Shape.to_dims (TDSL.O.relu f).Tensor.shape)));
  (* (e) Parameterizing a closed tensor is refused before any write. *)
  let closed = leaf "cg_param_init" [| 1.; 2. |] in
  ignore (Shape.to_dims closed.Tensor.shape : int array);
  let t ?label:_ ?top_down_prec:_ ?batch_dims:_ ?batch_axes:_ ?input_dims:_ ?output_dims:_
      ?input_axes:_ ?output_axes:_ ?deduced:_ () =
    closed
  in
  p "a parameter whose initializer returns a closed tensor is refused"
    (refused_with (fun () -> Tensor.param ~t "cg_p" ()) ~substrings:[ "already closed" ]);
  ignore (ctx : Context.t)
