(* gh-ocannl-1959: a host write to a node a forward fetch of the lineage re-establishes.

   A fetch-defined node -- a [Tensor.term_init] or literal with at most [limit_constant_fill_size]
   values, a range -- is code, not data: every run of the routine carrying its forward code fetches
   it again. Before this, [Context.set_values] on one was accepted, read back, and undone by the
   next run ([x] went 1 2, 9 10, 1 2 over a write and a run). Now, once a routine of the lineage
   carries that fetch, the host write is refused, naming the node and the routine, and the lineage
   carries on as if it was never attempted.

   Only a write after the forward is compiled is refused, and these legs pin what stays accepted: a
   seed, which a routine reading the node runs on until the forward's fetch takes over; a sibling's
   write; a data-backed node, a host scalar and a parameter, whose writes last; and a device copy,
   which brings the node's own value.

   Every value is a small integer and every operation pointwise, so readbacks are exact. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX

let fetched ?(values = [| 1.; 2. |]) ?(materialize = true) name =
  let x =
    Tensor.term_init values ~label:[ name ]
      ~output_dims:[ Array.length values ]
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  if materialize then Train.set_materialized x.Tensor.value;
  x

let data_backed name =
  Operation.init ~l:name ~prec:Ir.Ops.single ~o:[ 2 ]
    ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
    ~grad_spec:Tensor.Prohibit_grad ()

let double x =
  let%op doubled = x *. 2 in
  doubled

(* The routine carrying the forward code of the node labeled [x_<leg>]: named per leg, since a
   same-named routine overwrites the earlier one's debug artifacts. *)
let routine_of node = "fwd_" ^ String.drop_prefix node 2

let compile_forward ~node ctx t =
  Context.compile ~name:(routine_of node) ctx (Train.forward t) IDX.empty

let values ctx t = Context.get_values ctx t.Tensor.value
let write ctx t = Context.set_values ctx t.Tensor.value [| 9.; 10. |]

(* The refusal, naming the write's entry point, the node, the routine, the issue and the remedy. *)
let refused name ~fn ~node f =
  let msg = LC.refusal name f in
  List.iter
    [ "Context." ^ fn; node; routine_of node; "gh-ocannl-1959"; "Operation.init" ]
    ~f:(fun substring -> p (name ^ ": the refusal names " ^ substring) (LC.has msg substring))

let () =
  case "a host write after the forward is compiled" (fun () ->
      let x = fetched "x_after" in
      let doubled = double x in
      let ctx, routine = compile_forward ~node:"x_after" (Context.auto ()) doubled in
      let tn = x.Tensor.value in
      refused "set_values, compiled and not yet run" ~fn:"set_values" ~node:"x_after" (fun () ->
          write ctx x);
      let ctx = Context.run ctx routine in
      refused "set_values, after a run" ~fn:"set_values" ~node:"x_after" (fun () -> write ctx x);
      refused "set_value" ~fn:"set_value" ~node:"x_after" (fun () ->
          Context.set_value ctx tn [| 0 |] 9.);
      refused "from_host" ~fn:"from_host" ~node:"x_after" (fun () ->
          Context.from_host ctx tn (Context.to_host ctx tn));
      (* Nothing of a refused write landed, on the device or for the next run. *)
      LC.exact "x reads as the fetch left it" (values ctx x) [| 1.; 2. |];
      let ctx = Context.run ctx routine in
      LC.exact "x after the next run" (values ctx x) [| 1.; 2. |];
      LC.exact "the forward's result after the next run" (values ctx doubled) [| 2.; 4. |]);
  case "every kind of fetch-defined node" (fun () ->
      (* A fetch the routine inlined has no buffer to read; the write would be lost the sooner. *)
      let inlined = fetched ~materialize:false "x_inlined" in
      let ctx, _ = compile_forward ~node:"x_inlined" (Context.auto ()) (double inlined) in
      refused "an inlined fetch" ~fn:"set_values" ~node:"x_inlined" (fun () -> write ctx inlined);
      let literal = fetched ~values:[| 3. |] "x_literal" in
      let ctx, _ = compile_forward ~node:"x_literal" (Context.auto ()) (double literal) in
      refused "a one-element literal" ~fn:"set_values" ~node:"x_literal" (fun () ->
          Context.set_values ctx literal.Tensor.value [| 9. |]);
      let range =
        Tensor.term ~fetch_op:Ir.Assignments.Range_over_offsets ~label:[ "x_range" ]
          ~output_dims:[ 2 ] ~grad_spec:Tensor.Prohibit_grad ()
      in
      let ctx, _ = compile_forward ~node:"x_range" (Context.auto ()) (double range) in
      refused "a range" ~fn:"set_values" ~node:"x_range" (fun () -> write ctx range);
      (* Lowering drops this one's in-kernel fetch for a link-time upload (gh-ocannl-633), so the
         write would stay in the buffer until a schedule hoists the literal's registered values. *)
      let literal = TDSL.ndarray [| 1.; 2. |] ~label:[ "x_ndarray" ] ~output_dims:[ 2 ] () in
      Train.set_materialized literal.Tensor.value;
      let ctx, _ = compile_forward ~node:"x_ndarray" (Context.auto ()) (double literal) in
      refused "a materialized literal the routine reads" ~fn:"set_values" ~node:"x_ndarray"
        (fun () -> write ctx literal));
  case "a slice of a fetched parent" (fun () ->
      let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
      let parent =
        Tensor.term_init [| 1.; 2.; 3.; 4. |] ~label:[ "x_parent" ] ~batch_dims:[ 2 ]
          ~output_dims:[ 2 ] ~grad_spec:Tensor.Prohibit_grad ()
      in
      let%op view = parent @| batch_n in
      let%op doubled = view *. 2 in
      let ctx, _ =
        Context.compile ~name:(routine_of "x_parent") (Context.auto ()) (Train.forward doubled)
          bindings
      in
      refused "the parent" ~fn:"set_values" ~node:"x_parent" (fun () ->
          Context.set_values ctx parent.Tensor.value [| 9.; 10.; 11.; 12. |]);
      (* The view keeps the refusal it had, which says where the write belongs. *)
      let msg = LC.refusal "the view" (fun () -> write ctx view) in
      p "the view's refusal names the parent" (LC.has msg "write its parent"));
  case "a seed before the forward is compiled" (fun () ->
      let x = fetched "x_seed" in
      (* [doubled], constructed first, takes [x]'s forward code, so [reader] carries no fetch. *)
      let doubled = double x in
      let%cd reader = { y } =: x in
      Train.set_materialized y.Tensor.value;
      match LC.accepted "the seed" (fun () -> write (Context.auto ()) x) with
      | None -> ()
      | Some ctx ->
          let ctx, read = Context.compile ~name:"reader" ctx reader IDX.empty in
          let ctx = Context.run ctx read in
          LC.exact "a routine that does not carry the fetch runs on the seed" (values ctx y)
            [| 9.; 10. |];
          let ctx, routine = compile_forward ~node:"x_seed" ctx doubled in
          let ctx = Context.run ctx routine in
          LC.exact "the forward's run re-establishes the node" (values ctx x) [| 1.; 2. |];
          refused "a write once it is compiled" ~fn:"set_values" ~node:"x_seed" (fun () ->
              write ctx x);
          let ctx = Context.run ctx read in
          LC.exact "the reader follows the fetch from there" (values ctx y) [| 1.; 2. |]);
  case "a sibling of the forward's routine" (fun () ->
      let x = fetched "x_sibling" in
      let ctx = Context.auto () in
      let compiled, _ = compile_forward ~node:"x_sibling" ctx (double x) in
      refused "the forward's lineage" ~fn:"set_values" ~node:"x_sibling" (fun () ->
          write compiled x);
      Option.iter
        (LC.accepted "a context that does not descend from it" (fun () -> write ctx x))
        ~f:(fun ctx -> LC.exact "the sibling's write reads back" (values ctx x) [| 9.; 10. |]));
  case "nodes whose host writes last" (fun () ->
      let lasts name ~node t =
        let doubled = double t in
        let ctx, routine =
          compile_forward ~node (Train.init_params (Context.auto ()) IDX.empty doubled) doubled
        in
        let ctx = Context.run ctx routine in
        Option.iter
          (LC.accepted (name ^ ": written after the forward's run") (fun () -> write ctx t))
          ~f:(fun ctx ->
            let ctx = Context.run ctx routine in
            LC.exact (name ^ ": the write lasts over a run") (values ctx t) [| 9.; 10. |];
            LC.exact (name ^ ": the run computes from it") (values ctx doubled) [| 18.; 20. |])
      in
      lasts "a data-backed node" ~node:"x_data" (data_backed "x_data");
      lasts "a parameter" ~node:"x_param"
        (TDSL.param ~values:[| 1.; 2. |] "x_param" ~output_dims:[ 2 ] ());
      let scalar = Train.host_scalar ~l:"x_scalar" 1. in
      let doubled = double scalar in
      let ctx, routine = compile_forward ~node:"x_scalar" (Context.auto ()) doubled in
      let ctx = Context.run ctx routine in
      Option.iter
        (LC.accepted "a host scalar: written after the forward's run" (fun () ->
             Context.set_values ctx scalar.Tensor.value [| 9. |]))
        ~f:(fun ctx ->
          let ctx = Context.run ctx routine in
          LC.exact "a host scalar: the run computes from it" (values ctx doubled) [| 18. |]));
  case "a device copy" (fun () ->
      (* [src] holds no buffer of the node, so the copy takes the host round trip, uploading the
         node's literal into [dst]. *)
      let x = TDSL.ndarray [| 1.; 2. |] ~label:[ "x_copy" ] ~output_dims:[ 2 ] () in
      Train.set_materialized x.Tensor.value;
      let ctx, routine = compile_forward ~node:"x_copy" (Context.auto ()) (double x) in
      let ctx = Context.run ctx routine in
      Option.iter
        (LC.accepted "the host round trip" (fun () ->
             Context.copy ~src:(Context.auto ()) ~dst:ctx x.Tensor.value))
        ~f:(fun ctx -> LC.exact "the node reads as its literal" (values ctx x) [| 1.; 2. |]))
