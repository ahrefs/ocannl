(* A free node -- a bottom-up operation nothing in whose bottom-up dependencies states a precision
   -- follows its consumers' precision, promoting and never narrowing (gh-ocannl-2028). Before, an
   operation over only top-down operands nobody set ([uniform () - 0.5]) settled at the session
   default and was widened afterwards, so model noise under a double consumer was an f32 draw.

   The structural claims read settled precisions: inference precedes every backend. The executed
   claims are the load-bearing half, each with a value a narrower node cannot produce: a double leg
   (gated on f64 support: Metal has none) and a leg under a half session default that runs on every
   backend. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module Tn = Ir.Tnode
module Ops = Ir.Ops
open Verdict.Claims

let n = 64
let prec_of (t : Tensor.t) = Lazy.force t.value.storage_prec
let is_prec t prec = Ops.equal_prec (prec_of t) prec
let grad_of (t : Tensor.t) = (Option.value_exn t.diff).grad
let grad_is_prec t prec = Ops.equal_prec (Lazy.force (grad_of t).storage_prec) prec
let f32_representable x = Float.(Int32.float_of_bits (Int32.bits_of_float x) = x)
let f16_representable x = Float.(Ops.round_to_prec Ops.half x = x)
let zeros ~prec l = NTDSL.init ~l ~prec ~o:[ n ] ~f:(fun _ -> 0.) ()
let ctx0 = Context.auto ()
let backend = Context.backend_name ctx0
let f64_leg = Verdict.leg ~when_:(Context.codegen_capabilities ctx0).supports_f64 ~on:backend ()

(* Each executed case builds its graph in a fresh session, so the unconsumed roots of the structural
   cases (sharing the random seed) do not conflict with its forward code. *)
let fresh_session () = Tensor.unsafe_reinitialize ()

let values (t : Tensor.t) =
  let ctx = Train.forward_once (Context.auto ()) t in
  Array.to_list (Context.get_values ctx t.value)

let () =
  Tensor.unsafe_reinitialize ();

  (* The issue's case. *)
  Verdict.case "noise" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let u = TDSL.O.uniform () in
      let%op noise = u - 0.5 in
      let%op y = x + noise in
      p "noise: the sum is double" (is_prec y Ops.double);
      p "noise: the subtraction follows it to double" (is_prec noise Ops.double);
      p "noise: so does the draw" (is_prec u Ops.double));

  (* Freeness is transitive: the [mul] over the free [sub] and a literal is free too. *)
  Verdict.case "chain" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let u = TDSL.O.uniform () in
      let%op centered = u - 0.5 in
      let%op scaled = centered *. 2.0 in
      let%op _y = x + scaled in
      p_all "chain: every node of the free chain is double" [ scaled; centered; u ] ~f:(fun t ->
          is_prec t Ops.double));

  (* Promote, never narrow: under a half consumer the free noise keeps the session default, and the
     sum promotes over it as before. *)
  Verdict.case "narrower consumer" (fun () ->
      let x = zeros ~prec:Ops.half "x" in
      let u = TDSL.O.uniform () in
      let%op noise = u - 0.5 in
      let%op y = x + noise in
      p_all "narrower consumer: the sum, the noise and the draw stay single" [ y; noise; u ]
        ~f:(fun t -> is_prec t Ops.single));

  (* A stated operand constrains the node: it is not free and keeps its bottom-up precision. *)
  Verdict.case "stated operand" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let c = NTDSL.init ~l:"c" ~prec:Ops.single ~o:[ n ] ~f:(fun _ -> 0.5) () in
      let u = TDSL.O.uniform () in
      let%op noise = u - c in
      let%op y = x + noise in
      p "stated operand: the sum is double" (is_prec y Ops.double);
      p "stated operand: the subtraction over a single operand stays single"
        (is_prec noise Ops.single));

  (* A precision specified after the consumer was built still decides: freeness is read when the
     node settles. *)
  Verdict.case "late specification" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let u = TDSL.O.uniform () in
      let%op noise = u - 0.5 in
      let%op y = x + noise in
      Tn.update_prec u.value Ops.half;
      p "late specification: the sum is double" (is_prec y Ops.double);
      p "late specification: the subtraction over the half draw stays half" (is_prec noise Ops.half));

  (* A precision pinned on the free node itself wins over the link. *)
  Verdict.case "pinned node" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let%op noise = uniform () - 0.5 in
      let%op _y = x + noise in
      Tn.update_prec noise.value Ops.single;
      p "pinned node: the pinned subtraction stays single" (is_prec noise Ops.single));

  Verdict.case "no consumer" (fun () ->
      let%op noise = uniform () - 0.5 in
      p "no consumer: a free node settles at the session default" (is_prec noise Ops.single));

  (* Shared by a single and a double consumer: the link promotes over both, in either forcing
     order. *)
  Verdict.case "shared" (fun () ->
      let shared ~node_first =
        let xd = zeros ~prec:Ops.double "xd" and xs = zeros ~prec:Ops.single "xs" in
        let%op noise = uniform () - 0.5 in
        let%op ys = xs *. noise in
        let%op yd = xd + noise in
        let order = if node_first then [ noise; ys; yd ] else [ yd; ys; noise ] in
        List.iter order ~f:(fun t -> ignore (prec_of t : Ops.prec));
        List.map [ noise; ys; yd ] ~f:(fun t -> Ops.prec_string (prec_of t))
      in
      let node_first = shared ~node_first:true in
      Stdio.printf "shared (noise, single consumer, double consumer): %s\n"
        (String.concat ~sep:" " node_first);
      p "shared: the forcing order does not change the precisions"
        (List.equal String.equal node_first (shared ~node_first:false)));

  (* A free node that settled before its consumer was built keeps its precision, and the consumer
     builds without a refusal. *)
  Verdict.case "settled node" (fun () ->
      let x = zeros ~prec:Ops.double "x" in
      let%op noise = uniform () - 0.5 in
      ignore (prec_of noise : Ops.prec);
      let%op y = x + noise in
      p "settled node: the consumer is double" (is_prec y Ops.double);
      p "settled node: the settled free node keeps single" (is_prec noise Ops.single));

  (* Gradients follow the same rule against the gradient default: [h]'s gradient is free (the
     differentiable leaf [w] states nothing), and the consumer's is double. *)
  Verdict.case "gradient" (fun () ->
      let x =
        Ocannl.Operation.init ~l:"xg" ~prec:Ops.double ~o:[ n ]
          ~f:(fun _ -> 1.)
          ~grad_spec:Tensor.Require_grad ()
      in
      Tn.update_prec (grad_of x) Ops.double;
      let w = Tensor.term_init (Array.create ~len:n 0.25) ~grad_spec:Require_grad () in
      let%op h = w - 0.5 in
      let%op y = x *. h in
      p "gradient: the consumer's gradient is double" (grad_is_prec y Ops.double);
      p "gradient: the free node's gradient follows it" (grad_is_prec h Ops.double);
      p "gradient: so does the leaf's" (grad_is_prec w Ops.double));

  (* Executed, at double: with [x] zero the sum is the noise itself, so an f32 draw would make every
     value f32-representable. The control pins the noise single. *)
  Verdict.case "double values" (fun () ->
      let noise_values ~pin =
        lazy
          (fresh_session ();
           let x = zeros ~prec:Ops.double "x" in
           let%op noise = uniform () - 0.5 in
           let%op y = x + noise in
           Option.iter pin ~f:(Tn.update_prec noise.value);
           Train.set_materialized noise.value;
           values y)
      in
      let free = noise_values ~pin:None and pinned = noise_values ~pin:(Some Ops.single) in
      in_leg ~leg:f64_leg "double values: every value lies in [-0.5, 0.5)" (fun label ->
          p_all label (Lazy.force free) ~f:(fun v -> Float.(v >= -0.5 && v < 0.5)));
      in_leg ~leg:f64_leg "double values: some value of the noise is not f32-representable"
        (fun label -> p_exists label (Lazy.force free) ~f:(fun v -> not (f32_representable v)));
      in_leg ~leg:f64_leg "double values: the noise pinned single is f32-representable"
        (fun label -> p_all label (Lazy.force pinned) ~f:f32_representable));

  (* Executed, a free node over literals only: [(1 + 2^-30) - 1] beneath a double consumer, the sum
     materialized so its store rounds. At single that store is 1 and the difference 0. *)
  Verdict.case "literal values" (fun () ->
      let eps = Float.ldexp 1. (-30) in
      let got =
        lazy
          (fresh_session ();
           let x = zeros ~prec:Ops.double "x" in
           let one = NTDSL.number 1. and tiny = NTDSL.number eps in
           let%op sum = one + tiny in
           let%op diff = sum - 1.0 in
           let%op y = x + diff in
           Train.set_materialized sum.value;
           values y)
      in
      in_leg ~leg:f64_leg
        "literal values: (1 + 2^-30) - 1 is exactly 2^-30 beneath a double consumer" (fun label ->
          p_all label (Lazy.force got) ~f:(fun v -> Float.(v = eps))));

  (* A literal takes its consumer's precision under every session default, so one past the fp16
     cutoff beneath a half consumer is refused the same way under the single and the half default;
     before, a bump to single fired only under the half default and switched the lowering guard off
     (a silent [-inf] for 70000). Beneath a single consumer it computes. *)
  Verdict.case "large literal" (fun () ->
      let run ~default ~consumer c =
        fresh_session ();
        Tensor.default_value_prec := default;
        Exn.protect
          ~finally:(fun () -> Tensor.default_value_prec := Ops.single)
          ~f:(fun () ->
            let x = NTDSL.init ~l:"x" ~prec:consumer ~o:[ 4 ] ~f:(fun _ -> 1.) () in
            let k = NTDSL.number c in
            let%cd y = x - k in
            match values y with
            | vs -> Ok vs
            | exception e ->
                let msg = Exn.to_string e in
                Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
                Error msg)
      in
      let refused = function
        | Error msg -> String.is_substring msg ~substring:"too big for FP16"
        | Ok _ -> false
      in
      p_all "large literal: 20000 and 70000 beneath a half consumer are refused under both defaults"
        (List.concat_map [ Ops.single; Ops.half ] ~f:(fun default ->
             List.map [ 20000.; 70000. ] ~f:(fun c -> (default, c))))
        ~f:(fun (default, c) -> refused (run ~default ~consumer:Ops.half c));
      let computed =
        match run ~default:Ops.half ~consumer:Ops.single 20000. with Ok vs -> vs | Error _ -> []
      in
      p_all "large literal: beneath a single consumer under the half default it computes" computed
        ~f:(fun v -> Float.(v = -19999.)));

  (* Executed on every backend: under a half session default, noise beneath a single consumer is a
     single draw, which some value shows by not being f16-representable. *)
  Verdict.case "half default" (fun () ->
      Tensor.default_value_prec := Ops.half;
      Exn.protect
        ~finally:(fun () -> Tensor.default_value_prec := Ops.single)
        ~f:(fun () ->
          fresh_session ();
          let x = zeros ~prec:Ops.single "x" in
          let u = TDSL.O.uniform () in
          let%op noise = u - 0.5 in
          let%op y = x + noise in
          Train.set_materialized noise.value;
          p "half default: the noise and the draw follow the single consumer"
            (is_prec noise Ops.single && is_prec u Ops.single);
          let vs = values y in
          p_all "half default: every value lies in [-0.5, 0.5]" vs ~f:(fun v ->
              Float.(v >= -0.5 && v <= 0.5));
          p_exists "half default: some value of the noise is not f16-representable" vs ~f:(fun v ->
              not (f16_representable v))))
