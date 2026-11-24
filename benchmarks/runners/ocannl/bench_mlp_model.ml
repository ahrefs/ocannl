(* Graph construction shared by the MLP runner and the benchmark shape-scheme test. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
module St = Safetensors
module H = Bench_harness

type graph = {
  batch_loss : Tensor.t;
  logits : Tensor.t;
  params : (Tensor.t * Tensor.t) list;
  batch_n : Ir.Indexing.static_symbol option;
  bindings : Ir.Indexing.unit_bindings;
  n_batches : int;
}

let build_graph ~mp_prec ~no_slice ~materialize ~twin_placement st =
  let n_layers = H.meta_int st "n_layers" in
  let batch_size = H.meta_int st "batch_size" in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_x, batch_y, batch_n, bindings =
    if no_slice then (
      assert (n_batches = 1);
      (xs, ys, None, IDX.empty))
    else
      let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
      let%op batch_x = xs @| batch_n in
      let%op batch_y = ys @| batch_n in
      (batch_x, batch_y, Some batch_n, bindings)
  in
  let make_params () =
    List.init n_layers ~f:(fun idx ->
        let i = idx + 1 in
        let w_name = Printf.sprintf "w%d" i in
        let b_name = Printf.sprintf "b%d" i in
        let info = Option.value_exn ~here:[%here] (St.info st w_name) in
        let din, dout =
          match info.St.shape with
          | [ o; i_ ] -> (i_, o)
          | _ -> failwith ("bench_mlp: weight is not a matrix: " ^ w_name)
        in
        let w = TDSL.wrap_param ~l:w_name ~i:[ din ] ~o:[ dout ] (St.to_ndarray st w_name) () in
        let b = TDSL.wrap_param ~l:b_name ~o:[ dout ] (St.to_ndarray st b_name) () in
        (w, b))
  in
  (* Under a reduced precision the fixture weights stay f32 masters (the sgd targets); the graph
     reads their cast twins. wrap_param routes through Tensor.param, so the hook applies. *)
  let params =
    match mp_prec with
    | Some prec -> Mixed_prec.with_master_weights ~placement:twin_placement ~prec make_params
    | None -> make_params ()
  in
  let logits =
    List.foldi params ~init:batch_x ~f:(fun idx acc (w, b) ->
        let last = Int.equal idx (n_layers - 1) in
        let open TDSL.O in
        let z = b + (w * acc) in
        (* Explicit variant: store pre-activations for the backward pass (what the other frameworks
           do) instead of the default fully-Virtual recompute-in-backward. *)
        if materialize then Train.set_materialized z.Tensor.value;
        if last then z else relu z)
  in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits
      ~targets:batch_y
  in
  (* Storage-precision policy: activations and gradients of the MLP body (the logits subtree) go
     reduced; the softmax-CE loss head and its gradients are PINNED at f32 via [except] — the
     "softmax and losses stay f32" AMP default. Merely not assigning the head is not enough:
     bottom-up precision inference would promote it to the reduced precision from the logits operand
     (e.g. the stable-softmax max_logits, whose -inf init the f16 constant guard rejects). Masters
     and twins already carry Specified precisions and are untouched. *)
  Option.iter mp_prec ~f:(fun prec ->
      let body = Hash_set.create (module Int) in
      let rec walk t =
        if not (Hash_set.mem body t.Tensor.value.Ir.Tnode.id) then (
          Hash_set.add body t.Tensor.value.Ir.Tnode.id;
          Option.iter t.Tensor.diff ~f:(fun d -> Hash_set.add body d.Tensor.grad.Ir.Tnode.id);
          List.iter t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
      in
      walk logits;
      let except tn = not (Hash_set.mem body tn.Ir.Tnode.id) in
      Precision_policy.apply ~except
        { Precision_policy.param_prec = None; activation_prec = Some prec; grad_prec = Some prec }
        batch_loss);
  { batch_loss; logits; params; batch_n; bindings; n_batches }
