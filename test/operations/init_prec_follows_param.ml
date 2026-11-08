(* An initializer's precision follows its parameter in inference (gh-ocannl-1967), with no
   [Precision_policy] applied: the initializer fills the parameter, so a draw narrower than the
   parameter quantizes its initial values. The parameters here are double over the single session
   default, each for a different reason.

   The structural claims read the settled precisions of the initializer's nodes; the executed ones
   are the load-bearing half: a double parameter's initial values are not all f32-representable.
   Precision inference precedes every backend, so the test names cc, which evaluates double. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module IDX = Train.IDX
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode

let wide = Ir.Ops.double

open Verdict.Claims

let n = 16
let prec_of (t : Tensor.t) = Lazy.force t.value.storage_prec
let is_prec t prec = Ir.Ops.equal_prec (prec_of t) prec

(* The initializer's nodes: the parameter's subgraph, stopping at other parameters. *)
let init_nodes (param : Tensor.t) =
  let rec loop acc (t : Tensor.t) =
    if Set.mem t.params t || List.mem acc t ~equal:phys_equal then acc
    else List.fold t.children ~init:(t :: acc) ~f:(fun acc sub -> loop acc sub.subtensor)
  in
  List.fold param.children ~init:[] ~f:(fun acc sub -> loop acc sub.subtensor)

let float_init_nodes param =
  List.filter (init_nodes param) ~f:(fun t -> Ir.Ops.is_float (prec_of t))

let f32_representable x = Float.(Int32.float_of_bits (Int32.bits_of_float x) = x)

let initial_values base_ctx (root : Tensor.t) (param : Tensor.t) =
  let ctx = Train.init_params base_ctx IDX.empty root in
  Array.to_list (Context.get_values ctx param.value)

let structural_claims name ~(param : Tensor.t) =
  p (name ^ ": the parameter is double") (is_prec param wide);
  p_all (name ^ ": every float node of the initializer is double") (float_init_nodes param)
    ~f:(fun t -> is_prec t wide)

let claims name base_ctx ~root ~(param : Tensor.t) =
  structural_claims name ~param;
  let initial = initial_values base_ctx root param in
  p_all (name ^ ": the initial values lie in the initializer's range") initial ~f:(fun v ->
      Float.(v >= -4. && v < 4.));
  p_exists (name ^ ": some initial value is not f32-representable") initial ~f:(fun v ->
      not (f32_representable v))

let () =
  Tensor.unsafe_reinitialize ();
  let base_ctx = Context.cpu () in

  (* A precision the parameter holds explicitly, set after its construction. *)
  let w = TDSL.param "w_explicit" ~output_dims:[ n ] () in
  Tn.update_prec w.value wide;
  let%op root = w ++ "i=>0" in
  claims "explicit" base_ctx ~root ~param:w;

  (* A precision the parameter infers top-down from a consumer. *)
  let x = NTDSL.init ~l:"x" ~prec:wide ~o:[ n ] ~f:(fun _ -> 1.) () in
  let w = TDSL.param "w_inferred" () in
  let%op root = (w *. x) ++ "i=>0" in
  claims "inferred from a consumer" base_ctx ~root ~param:w;

  (* A double master weight behind a single cast twin: the model reads the twin. *)
  let twin =
    MP.with_master_weights ~master_prec:wide ~prec:Ir.Ops.single (fun () ->
        TDSL.param "w_master" ~output_dims:[ n ] ())
  in
  let%op root = twin ++ "i=>0" in
  let w = Set.choose_exn root.params in
  p "master weight: the model reads a single twin" (is_prec twin Ir.Ops.single);
  claims "master weight" base_ctx ~root ~param:w;

  (* An initializer built before [Tensor.param] sees it, with bottom-up nodes of its own. Structural
     only: its scale factor makes the values of a single draw non-f32-representable too. *)
  let w =
    TDSL.param ~param_init:(TDSL.kaiming TDSL.O.uniform ()) "w_kaiming" ~output_dims:[ n ] ()
  in
  Tn.update_prec w.value wide;
  structural_claims "kaiming" ~param:w;

  (* A bottom-up node both an initializer and the model read, under a consumer of the parameter: the
     parameter follows that consumer, which reads the shared node, which follows the parameter. The
     consumer reads the precision the shared node had before that link, so nothing is circular and
     the forcing order does not matter. *)
  let shared_case ~shared_first =
    let shared = NTDSL.pointmul (NTDSL.number 3.) (NTDSL.number 2.) () in
    let w = TDSL.param ~param_init:(NTDSL.pointmul shared (NTDSL.number 2.)) "w_shared" () in
    let x = NTDSL.init ~l:"x_shared" ~prec:wide ~o:[ n ] ~f:(fun _ -> 1.) () in
    let%op y = w *. (shared *. x) in
    if shared_first then ignore (prec_of shared : Ir.Ops.prec);
    (prec_of y, prec_of w, prec_of shared, Tn.bottom_up_prec shared.value)
  in
  let show (y, w, shared, read) =
    String.concat ~sep:" " (List.map [ y; w; shared; read ] ~f:Ir.Ops.prec_string)
  in
  let consumer_first = shared_case ~shared_first:false in
  Stdio.printf "shared node (consumer, parameter, shared, shared as its consumers read it): %s\n"
    (show consumer_first);
  p "shared node: the forcing order does not change the precisions"
    (String.equal (show consumer_first) (show (shared_case ~shared_first:true)));

  (* Reading a linked node's pre-link precision memoizes it like settling the node does, so a rewind
     past that read is refused as one past a settlement is (the consumer forced here is one the
     rewind would discard). *)
  let shared = NTDSL.pointmul (NTDSL.number 3.) (NTDSL.number 2.) () in
  let _w : Tensor.t =
    TDSL.param ~param_init:(NTDSL.pointmul shared (NTDSL.number 2.)) "w_rewind" ()
  in
  let ck = Tensor.checkpoint ~label:"ck_linked" () in
  let%op y = relu shared in
  ignore (prec_of y : Ir.Ops.prec);
  p "a linked node read bottom-up after a checkpoint refuses the rewind"
    (match Tensor.rewind ck with
    | () -> false
    | exception Tensor.Session_error (msg, _) ->
        Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
        String.is_substring msg ~substring:"began settling");
  Tensor.rewind_to_last_closure ();

  (* A parameter narrower than its initializer does not narrow it: the link promotes. Only the
     constants that are the parameter's own top-down operands follow it down, as they did. *)
  let w = TDSL.param "w_narrow" ~output_dims:[ n ] () in
  Tn.update_prec w.value Ir.Ops.half;
  p "narrow parameter: the parameter is half" (is_prec w Ir.Ops.half);
  let draw_and_above =
    List.filter (float_init_nodes w) ~f:(fun t -> not (List.is_empty t.children))
  in
  p_all "narrow parameter: the draw and the operations over it stay single" draw_and_above
    ~f:(fun t -> is_prec t Ir.Ops.single)
