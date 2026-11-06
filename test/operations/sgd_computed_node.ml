(* gh-ocannl-1998: [Train.sgd_update] refuses a stepped node the loss's forward recomputes, or a
   view of one, since the next forward undoes the step; [Assignments.forward_recomputed] tells that
   from a read-modify-write, whose step lasts.

   1. The classification, over hand-built leaves. 2. What is refused, and with which tensor. 3. Why:
   the same step through bare [sgd_one], which nothing refuses, does not accumulate. 4. What is
   accepted, each against an independent recurrence over several runs: a value the forward builds
   on, directly, through a temporary or through a view; a parameter with a computed initializer; a
   value computed once, outside the loss's forward. *)

open Base
open Stdio
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

(* ----- 1. The classification ----- *)

let projections = lazy (failwith "the classification forces no projections")

let accum ~initialize_neutral ?(op = Ir.Ops.Add) lhs rhs =
  Asgns.Accum_op { initialize_neutral; accum = op; lhs; rhs; projections; projections_debug = "" }

let copy rhs = Asgns.Unop { op = Ir.Ops.Identity; rhs = Asgns.Node rhs }

(* The assignment operators: [=:], [=:+] and [=+]. *)
let assign lhs rhs = accum ~initialize_neutral:false ~op:Ir.Ops.Arg2 lhs (copy rhs)
let reset_sum lhs rhs = accum ~initialize_neutral:true lhs (copy rhs)
let accumulate lhs rhs = accum ~initialize_neutral:false lhs (copy rhs)

let binary ?(op = Ir.Ops.Add) lhs rhs1 rhs2 =
  accum ~initialize_neutral:false ~op:Ir.Ops.Arg2 lhs
    (Asgns.Binop { op; rhs1 = Node rhs1; rhs2 = Node rhs2 })

let slice array sliced =
  let batch_idx, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  Asgns.Fetch
    {
      array;
      fetch_op = Slice { batch_idx; sliced };
      dims = lazy (failwith "the classification forces no dims");
    }

let node label = (TDSL.term ~label:[ label ] ()).Tensor.value

let classified name leaves ~recomputed:expected =
  let code = List.reduce_exn leaves ~f:(fun a b -> Asgns.Seq (a, b)) in
  let got = Asgns.forward_recomputed (Asgns.to_comp code) in
  Verdict.p ("classified: " ^ name) (Set.equal got (Set.of_list (module Tn) expected))

let classification () =
  let s = node "s" and x = node "x" and y = node "y" and tmp = node "tmp" in
  let v = node "v" and w = node "w" in
  classified "=: from another node recomputes" [ assign s x ] ~recomputed:[ s ];
  classified "=:+ recomputes" [ reset_sum s x ] ~recomputed:[ s ];
  classified "=+ builds on the old value" [ accumulate s x ] ~recomputed:[];
  classified "=: reading itself builds on the old value" [ binary s s x ] ~recomputed:[];
  classified "a fetch recomputes"
    [ Asgns.Fetch { array = s; fetch_op = Constant 0.; dims = lazy [||] } ]
    ~recomputed:[ s ];
  classified "a read after the write uses the new value"
    [ assign s x; assign y s ]
    ~recomputed:[ s; y ];
  classified "reset, then accumulate" [ assign s x; accumulate s y ] ~recomputed:[ s ];
  classified "the old value reaches the result through a temporary"
    [ assign tmp s; binary s tmp x ]
    ~recomputed:[ tmp ];
  classified "the old value does not survive a delay line"
    [ assign y s; assign s x ]
    ~recomputed:[ y; s ];
  classified "a state carried in a temporary across the run is still recomputed"
    [ binary s tmp x; assign tmp s ]
    ~recomputed:[ s ];
  classified "a projection operator evaluates only its operand"
    [ binary ~op:Ir.Ops.Arg2 s s x ]
    ~recomputed:[ s ];
  classified "the first-operand projection, likewise"
    [ binary ~op:Ir.Ops.Arg1 s x s ]
    ~recomputed:[ s ];
  classified "an accumulation keeping the old value ignores the rhs"
    [ accum ~initialize_neutral:false ~op:Ir.Ops.Arg1 s (copy x) ]
    ~recomputed:[];
  classified "a set_vec_unop writes from its rhs"
    [
      Asgns.Set_vec_unop
        {
          op = Ir.Ops.Uint4x32_to_prec_uniform;
          lhs = s;
          rhs = Node x;
          projections;
          projections_debug = "";
        };
    ]
    ~recomputed:[ s ];
  classified "a merge buffer is no read of the node"
    [
      accum ~initialize_neutral:false ~op:Ir.Ops.Arg2 s
        (Unop { op = Identity; rhs = Merge_buffer s });
    ]
    ~recomputed:[ s ];
  classified "Rev_sides writes its buffers from the lhs"
    [
      assign s x;
      accum ~initialize_neutral:true y (Rev_sides { op = Identity; lhses = [| Node v; Node w |] });
    ]
    ~recomputed:[ s; v; w ];
  classified "Rev_sides reads the lhs and accumulates into its buffers"
    [ accum ~initialize_neutral:false s (Rev_sides { op = Identity; lhses = [| Node v |] }) ]
    ~recomputed:[];
  classified "a Rev_sides write is built from what it read"
    [
      accum ~initialize_neutral:true ~op:Ir.Ops.Add v
        (Rev_sides { op = Identity; lhses = [| Node s |] });
      binary v v x;
    ]
    ~recomputed:[ s ];
  classified "a view of a recomputed parent follows it"
    [ assign s x; slice v s ]
    ~recomputed:[ s; v ];
  classified "a view of a view follows it"
    [ assign s x; slice v s; slice w v ]
    ~recomputed:[ s; v; w ];
  classified "a view of data does not count" [ slice v x ] ~recomputed:[];
  classified "a view of a persistent parent does not count"
    [ accumulate s x; slice v s ]
    ~recomputed:[];
  classified "accumulating through a view of data builds on the old value"
    [ slice v x; accumulate v y ]
    ~recomputed:[];
  classified "overwriting through a view of data recomputes the parent"
    [ slice v x; assign v y ]
    ~recomputed:[ x; v ];
  classified "a read through a view of the old value is a read of it"
    [ slice v s; binary s v x ]
    ~recomputed:[]

(* ----- Shared by the executed sections ----- *)

let learning_rate () =
  let%op learning_rate = 0.125 in
  Train.every_non_literal_materialized learning_rate;
  learning_rate

let lr = 0.125
let one t = Set.singleton (module Tensor) t

let leaf ~l ?(grad_spec = Tensor.Require_grad) values =
  Operation.init ~l ~prec:Ir.Ops.single
    ~o:[ Array.length values ]
    ~f:(function [| i |] -> values.(i) | _ -> assert false)
    ~grad_spec ()

(* A two-row data leaf: [[1, 2], [3, 4]]. *)
let rows ~l =
  Operation.init ~l ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 2 ]
    ~f:(function [| i; j |] -> Float.of_int ((2 * i) + j + 1) | _ -> assert false)
    ~grad_spec:Tensor.Require_grad ()

let values ctx t = Context.get_values ctx t.Tensor.value
let matches name got want = Verdict.p_close name got want ~tol:(fun _ -> 1e-6)

(* [run] advances the context by one step; [steps] compares [watched] after each of three runs with
   the successive [oracle] iterates from [start]. *)
let tracks name ~run ~ctx ~watched ~start ~oracle =
  let _, _ =
    List.fold [ 1; 2; 3 ] ~init:(ctx, start) ~f:(fun (ctx, state) k ->
        let ctx = run ctx and state = oracle state in
        matches (Printf.sprintf "%s: run %d" name k) (values ctx watched) state;
        (ctx, state))
  in
  ()

(* A forward [v =+ v1]: the node accumulates its operand, keeping any step taken on it. *)
let accumulating x ~label =
  let%cd op_asn ~t ~t1 ~projections = v =+ v1 in
  let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ g in
  Tensor.unop ~op_label:"accum" ~op_asn ~grad_asn x ~label:[ label ] ()

(* ----- 2. Refused ----- *)

let refused name ?params ~node ~stepped loss =
  match Train.sgd_update ~learning_rate:(learning_rate ()) ?params loss with
  | _ -> Verdict.p (name ^ ": refused") false
  | exception Tensor.Session_error (msg, culprit) ->
      eprintf "%s: %s (not part of the golden)\n" name msg;
      Verdict.p (name ^ ": refused") true;
      Verdict.p
        (name ^ ": the refusal carries the stepped tensor")
        (Option.exists culprit ~f:(fun t -> Tn.equal t.Tensor.value stepped.Tensor.value));
      List.iter [ node; "recomputes"; "Operation.init" ] ~f:(fun substring ->
          Verdict.p (name ^ ": the refusal names " ^ substring) (String.is_substring msg ~substring))

let computed_refused () =
  let x = leaf ~l:"xc" [| 1.; 2. |] in
  let%op hc = x *. 2 in
  let%op l = hc *. hc in
  refused "a computed node" ~params:(one hc) ~node:"hc" ~stepped:hc l;
  (* After the forward was consumed too: the refusal reads the code, not the consumption. *)
  let x = leaf ~l:"xd" [| 1.; 2. |] in
  let%op hd = x *. 2 in
  let%op l = hd *. hd in
  let (_ : Asgns.comp) = Train.grad_update l in
  refused "a computed node, after grad_update" ~params:(one hd) ~node:"hd" ~stepped:hd l

let views_refused () =
  let batch_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p = rows ~l:"pv" in
  let%op hv = p *. 2 in
  let%op aliased_view = hv @| batch_n in
  let%op l = aliased_view *. aliased_view in
  refused "a view of a computed node" ~params:(one aliased_view) ~node:"aliased_view"
    ~stepped:aliased_view l;
  (* A view of a view. *)
  let outer, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let inner, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p =
    Operation.init ~l:"pn" ~prec:Ir.Ops.single ~b:[ 2; 2 ] ~o:[ 2 ]
      ~f:(fun _ -> 1.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op hn = p *. 2 in
  let%op row = hn @| outer in
  let%op nested_view = row @| inner in
  let%op l = nested_view *. nested_view in
  refused "a view of a view of a computed node" ~params:(one nested_view) ~node:"nested_view"
    ~stepped:nested_view l;
  (* A padded view is a copy: refused here all the same, with no need of its shapes. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let images =
    Operation.init ~l:"imgs" ~prec:Ir.Ops.single ~b:[ 2; 1 ] ~o:[ 3; 3; 1 ]
      ~f:(fun _ -> 1.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op hp = images *. 2 in
  let%op copied_view = hp @| batch_n in
  let processed =
    Nn_blocks.conv2d ~label:[ "processed" ] ~use_padding:true ~out_channels:1 () copied_view
  in
  refused "a copying view of a computed node" ~params:(one copied_view) ~node:"copied_view"
    ~stepped:copied_view processed;
  let (_ : Context.t) = Train.init_params (Context.auto ()) bindings processed in
  Verdict.p "that view cannot alias its parent"
    (not (Asgns.slice_alias_eligible ~array:copied_view.Tensor.value ~sliced:hp.Tensor.value))

let plain_forward_write_refused () =
  (* A write the forward makes owns no fragment of the node it writes, and undoes a step the same:
     here over a parameter, in the default stepped set. *)
  let x = leaf ~l:"xw" ~grad_spec:Tensor.Prohibit_grad [| 1.; 2. |] in
  let overwritten = TDSL.param ~value:1. "overwritten" () in
  Shape.infer_equal overwritten.Tensor.shape x.Tensor.shape;
  let%op l = overwritten *. overwritten in
  let reset = [%cd overwritten =: x] in
  let l = { l with Tensor.forward = Asgns.sequence [ reset; l.Tensor.forward ] } in
  refused "a parameter its forward overwrites" ~node:"overwritten" ~stepped:overwritten l

(* ----- 3. Why ----- *)

let bare_step_does_not_accumulate () =
  let x = leaf ~l:"xb" [| 1.; 2. |] in
  let%op hb = x *. 2 in
  let%op l = hb *. hb in
  Train.set_materialized hb.Tensor.value;
  let step = Train.sgd_one ~learning_rate:(learning_rate ()) hb in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, routine = Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty in
  (* Every run recomputes (2, 4) and takes the one step d = 2 h: h - h / 4. *)
  tracks "bare sgd_one on a computed node is undone"
    ~run:(fun ctx -> Context.run ctx routine)
    ~ctx ~watched:hb ~start:[||]
    ~oracle:(fun _ -> [| 1.5; 3. |])

(* ----- 4. Accepted ----- *)

(* Compiles [grad_update l] and [step] into one routine, the update of each cell of [watched] by the
   forward being [forward]; a step on [n *. n] is then [n - lr * 2 n]. *)
let builds_on name ~l ~watched ~params ~start ~forward =
  match Train.sgd_update ~learning_rate:(learning_rate ()) ?params l with
  | exception Tensor.Session_error (msg, _) ->
      Verdict.p (name ^ ": accepted (refused: " ^ msg ^ ")") false
  | step ->
      Verdict.p (name ^ ": accepted") true;
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx, routine =
        Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty
      in
      tracks name
        ~run:(fun ctx -> Context.run ctx routine)
        ~ctx ~watched ~start
        ~oracle:(fun state ->
          Array.mapi state ~f:(fun i v ->
              let v = forward i v in
              v -. (lr *. 2. *. v)))

let accumulating_node () =
  let xs = [| 1.; 2. |] in
  let x = leaf ~l:"xa" xs in
  let s = accumulating x ~label:"sa" in
  let%op l = s *. s in
  Train.set_materialized s.Tensor.value;
  builds_on "a node the forward accumulates into" ~l ~watched:s
    ~params:(Some (one s))
    ~start:[| 0.; 0. |]
    ~forward:(fun i v -> v +. xs.(i))

(* A differentiable statistic the forward updates after reading it, the [batch_norm_state] shape. *)
let statistic name ~update ~forward =
  let xs = [| 2.; 4. |] in
  let x = leaf ~l:("x" ^ name) ~grad_spec:Tensor.Prohibit_grad xs in
  let s = TDSL.param ~value:1. ("stat" ^ name) () in
  Shape.infer_equal s.Tensor.shape x.Tensor.shape;
  let%op l = s *. s in
  let l = { l with Tensor.forward = Asgns.sequence [ l.Tensor.forward; update s x ] } in
  builds_on ("a statistic reading itself " ^ name) ~l ~watched:s ~params:None ~start:[| 1.; 1. |]
    ~forward:(fun i v -> forward xs.(i) v)

let statistics () =
  statistic "directly" ~update:(fun s x -> [%cd s =: s + x]) ~forward:(fun x v -> v +. x);
  statistic "through a temporary"
    ~update:(fun s x -> [%cd s =: (!.0.5 *. s) + (!.0.25 *. x)])
    ~forward:(fun x v -> (0.5 *. v) +. (0.25 *. x))

let view_of_persistent_parent () =
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let x = rows ~l:"xp" in
  let s = accumulating x ~label:"sp" in
  let%op kept_view = s @| batch_n in
  let%op l = kept_view *. kept_view in
  let step = Train.sgd_update ~learning_rate:(learning_rate ()) ~params:(one kept_view) l in
  Verdict.p "a view of a persistent parent: accepted" true;
  let ctx, routine =
    Train.to_routine (Context.auto ()) bindings (Asgns.sequence [ Train.grad_update l; step ])
  in
  Verdict.p "that view is an alias" (Tn.is_alias kept_view.Tensor.value);
  IDX.find_exn routine.Context.bindings batch_n := 0;
  (* Row 0 takes the step; both rows accumulate. *)
  tracks "a step through a view of a persistent parent lasts"
    ~run:(fun ctx -> Context.run ctx routine)
    ~ctx ~watched:s ~start:[| 0.; 0.; 0.; 0. |]
    ~oracle:(fun state ->
      Array.mapi state ~f:(fun i v ->
          let v = v +. Float.of_int (i + 1) in
          if i < 2 then v -. (lr *. 2. *. v) else v))

let computed_initializer () =
  (* The initializer runs once: the forward that reads the parameter does not carry it. *)
  let leg name ~tag ~default =
    let xs = [| 1.; 2. |] in
    let xi = leaf ~l:("xi" ^ tag) ~grad_spec:Tensor.Prohibit_grad xs in
    let doubled ?label ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims ?input_axes
        ?output_axes ?deduced () =
      let two = Tensor.number ~grad_spec:Tensor.Prohibit_grad 2. in
      NTDSL.pointmul two xi ?label ?top_down_prec ?batch_dims ?batch_axes ?input_dims ?output_dims
        ?input_axes ?output_axes ?deduced ()
    in
    let w = TDSL.param ~param_init:doubled ("wi" ^ tag) ~output_dims:[ 2 ] () in
    let%op l = w *. xi in
    let params = if default then None else Some (one w) in
    let step = Train.sgd_update ~learning_rate:(learning_rate ()) ?params l in
    Verdict.p (name ^ ": accepted") true;
    let ctx = Train.init_params (Context.auto ()) IDX.empty l in
    let ctx, routine =
      Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty
    in
    tracks name
      ~run:(fun ctx -> Context.run ctx routine)
      ~ctx ~watched:w
      ~start:(Array.map xs ~f:(fun x -> 2. *. x))
      ~oracle:(Array.mapi ~f:(fun i v -> v -. (lr *. xs.(i))))
  in
  leg "a parameter with a computed initializer, by default" ~tag:"d" ~default:true;
  leg "a parameter with a computed initializer, in ?params" ~tag:"p" ~default:false

let computed_once () =
  (* Computed by a routine of its own before the loss is built: the loss's forward does not carry
     it, and the value is the caller's from here on. *)
  let x = leaf ~l:"xo" [| 1.; 2. |] in
  let%op ho = x *. 2 in
  Train.set_materialized ho.Tensor.value;
  let ctx, fwd = Train.to_routine (Context.auto ()) IDX.empty (Train.forward ho) in
  let ctx = Context.run ctx fwd in
  let%op l = ho *. ho in
  let step = Train.sgd_update ~learning_rate:(learning_rate ()) ~params:(one ho) l in
  Verdict.p "a node computed once, outside the loss's forward: accepted" true;
  let ctx, routine = Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty in
  tracks "a step on a node computed once lasts" ~ctx
    ~run:(fun ctx -> Context.run ctx routine)
    ~watched:ho ~start:[| 2.; 4. |]
    ~oracle:(Array.map ~f:(fun v -> v -. (lr *. 2. *. v)));
  (* The same parent reached through a view built after. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p = rows ~l:"po" in
  let%op hq = p *. 2 in
  Train.set_materialized hq.Tensor.value;
  let ctx, fwd = Train.to_routine (Context.auto ()) IDX.empty (Train.forward hq) in
  let ctx = Context.run ctx fwd in
  let%op late_view = hq @| batch_n in
  let%op l = late_view *. late_view in
  let step = Train.sgd_update ~learning_rate:(learning_rate ()) ~params:(one late_view) l in
  Verdict.p "a view of a parent computed once: accepted" true;
  let ctx, routine = Train.to_routine ctx bindings (Asgns.sequence [ Train.grad_update l; step ]) in
  Verdict.p "that view is an alias" (Tn.is_alias late_view.Tensor.value);
  IDX.find_exn routine.Context.bindings batch_n := 0;
  tracks "a step through a view of a parent computed once lasts" ~ctx
    ~run:(fun ctx -> Context.run ctx routine)
    ~watched:hq ~start:[| 2.; 4.; 6.; 8. |]
    ~oracle:(Array.mapi ~f:(fun i v -> if i < 2 then v -. (lr *. 2. *. v) else v))

let () =
  classification ();
  computed_refused ();
  views_refused ();
  plain_forward_write_refused ();
  bare_step_does_not_accumulate ();
  accumulating_node ();
  statistics ();
  view_of_persistent_parent ();
  computed_initializer ();
  computed_once ()
