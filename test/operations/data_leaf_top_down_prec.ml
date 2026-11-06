(* gh-ocannl-2011: a tensor whose data fixes its precision takes no top-down precision, so a
   consumer built after that precision was read links nothing and refuses nothing. A leaf with no
   data of its own does take it, and still refuses a link that would contradict a read.

   The refusal fired on any forced node, a [Specified] one included, where the update is a no-op:
   [Tensor.op] went past it for an [init] leaf only because that node's [dims] were forced at
   creation. Every value below is a small integer, exact at half precision on every backend. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode

let n = 2
let leaf_prec = Ir.Ops.half
let result_prec = Ir.Ops.single
let cell idcs = Float.of_int ((n * idcs.(0)) + idcs.(1) + 1)
let squared = [| 7.; 10.; 15.; 22. |]
let data prec = Ir.Ndarray.init_array ~debug:"leaf" prec ~dims:[| n; n |] ~padding:None ~f:cell

let leaves =
  [
    ("an init", fun prec -> NTDSL.init ~l:"leaf" ~prec ~i:[ n ] ~o:[ n ] ~f:cell ());
    ("a reshape", fun prec -> NTDSL.reshape ~l:"leaf" ~b:[] ~i:[ n ] ~o:[ n ] (data prec) ());
    ("a wrap", fun prec -> NTDSL.wrap ~l:"leaf" ~b:[] ~i:[ n ] ~o:[ n ] (data prec) ());
  ]

let storage t = Lazy.force t.Tensor.value.Tn.storage_prec
let is prec t = Ir.Ops.equal_prec (storage t) prec

let () =
  List.iter leaves ~f:(fun (name, leaf) ->
      List.iter
        [ ("read first", true); ("unread", false) ]
        ~f:(fun (reading, read_first) ->
          Verdict.case (Printf.sprintf "%s leaf under a wider consumer, %s" name reading) (fun () ->
              Tensor.unsafe_reinitialize ();
              let a = leaf leaf_prec and b = leaf leaf_prec in
              if read_first then
                p "the leaf holds its stated precision" (is leaf_prec a && is leaf_prec b);
              let%op t = a * b in
              Tn.update_prec t.Tensor.value result_prec;
              Train.set_materialized t.Tensor.value;
              let ctx = Train.forward_once (Context.auto ()) t in
              p "the leaf keeps its stated precision" (is leaf_prec a && is leaf_prec b);
              p "the result has its own" (is result_prec t);
              pf_all2 "the product is computed from the leaf's data"
                (Context.get_values ctx t.Tensor.value)
                squared ~f:Float.equal;
              Context.release ctx));
      Verdict.case (Printf.sprintf "a result over %s leaf and its twin" name) (fun () ->
          Tensor.unsafe_reinitialize ();
          let a = leaf leaf_prec and b = leaf leaf_prec in
          let%op t = a * b in
          p "the result reads the leaves' precision bottom-up" (is leaf_prec t)));
  let dataless () = NTDSL.range_of_shape ~input_dims:[ n ] ~output_dims:[ n ] () in
  Verdict.case "a leaf with no data, unread" (fun () ->
      Tensor.unsafe_reinitialize ();
      let a = dataless () and b = dataless () in
      let%op t = a * b in
      Tn.update_prec t.Tensor.value leaf_prec;
      p "it takes its consumer's precision" (is leaf_prec a && is leaf_prec b));
  Verdict.case "a leaf with no data, read first" (fun () ->
      Tensor.unsafe_reinitialize ();
      let a = dataless () and b = dataless () in
      let (_ : Ir.Ops.prec) = storage a in
      match
        let%op t = a * b in
        t
      with
      | exception Utils.User_error msg ->
          p "the consumer is refused, naming the forced precision"
            (String.is_substring msg ~substring:"already been forced")
      | (_ : Tensor.t) -> p "the consumer is refused, naming the forced precision" false)
