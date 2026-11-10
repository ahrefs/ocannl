(* gh-ocannl-1998: a step on a node the loss's forward recomputes, or on a view of one, is undone by
   the next forward; [Assignments.forward_recomputed] tells that from a read-modify-write, whose
   step lasts.

   gh-ocannl-2027 moved the refusal from [sgd_update] to the step's mark ([Asgns.field-persists]):
   [Context.run] refuses the run that would undo a step, and [Asgns.sequence] a forward sequenced
   after one.

   1. The classification, over hand-built leaves. 2. What is refused: the second run of a routine
   carrying the forward and the step (the first is a forward run once), and a step sequenced before
   its node's forward. 3. Why: bare [sgd_one], which nothing refused before, each run losing the
   previous step. 4. What is accepted, each against an independent recurrence over several runs: a
   value the forward builds on, directly, through a temporary or through a view; a parameter with a
   computed initializer; a value computed once, outside the loss's forward. 5. Across routines: a
   forward re-run after the steps, refused through each way of marking them (an unmarked step is
   not), and a deliberate replacement -- a host write, a replacing run, re-initialization --
   superseding the steps. *)

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

(* ----- 2. Refused, at the run that would undo the step ----- *)

(* A refusal's stage and message, from building [step l] and compiling it after [grad_update l] into
   one routine, as a training loop does, through two runs of it: the first is a forward run once
   plus one step, which is legal; the second would recompute the stepped node over the step
   (gh-ocannl-2027). *)
let fused_runs ?(bindings = IDX.empty) ?(symbols = []) ~step l =
  let stage = ref "construction" in
  match
    let step = step l in
    stage := "compile";
    let ctx = Train.init_params (Context.auto ()) bindings l in
    let ctx, routine =
      Train.to_routine ctx bindings (Asgns.sequence [ Train.grad_update l; step ])
    in
    List.iter symbols ~f:(fun s -> IDX.find_exn routine.Context.bindings s := 0);
    stage := "run 1";
    let ctx = Context.run ctx routine in
    stage := "run 2";
    ignore (Context.run ctx routine : Context.t)
  with
  | () -> None
  | exception (Utils.User_error msg | Tensor.Session_error (msg, _)) -> Some (!stage, msg)

let refused_with name ~at ~names outcome =
  match outcome with
  | None -> Verdict.p (name ^ ": refused") false
  | Some (stage, msg) ->
      eprintf "%s: refused at %s: %s (not part of the golden)\n" name stage msg;
      Verdict.p (name ^ ": refused") true;
      Verdict.p (name ^ ": refused at " ^ at) (String.equal stage at);
      List.iter names ~f:(fun substring ->
          Verdict.p (name ^ ": the refusal names " ^ substring) (String.is_substring msg ~substring))

let sgd ?params l = Train.sgd_update ~learning_rate:(learning_rate ()) ?params l

let refused name ?bindings ?symbols ?params ~node loss =
  refused_with name ~at:"run 2"
    ~names:[ node; "gh-ocannl-2027"; "Operation.init" ]
    (fused_runs ?bindings ?symbols ~step:(sgd ?params) loss)

let computed_refused () =
  let x = leaf ~l:"xc" [| 1.; 2. |] in
  let%op hc = x *. 2 in
  let%op l = hc *. hc in
  Train.set_materialized hc.Tensor.value;
  refused "a computed node" ~params:(one hc) ~node:"hc" l

let views_refused () =
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p = rows ~l:"pv" in
  let%op hv = p *. 2 in
  let%op aliased_view = hv @| batch_n in
  let%op l = aliased_view *. aliased_view in
  refused "a view of a computed node" ~bindings ~symbols:[ batch_n ] ~params:(one aliased_view)
    ~node:"hv" l;
  Verdict.p "that view is an alias" (Tn.is_alias aliased_view.Tensor.value);
  (* A view of a view. *)
  let outer, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let inner, bindings = IDX.get_static_symbol ~static_range:2 bindings in
  let p =
    Operation.init ~l:"pn" ~prec:Ir.Ops.single ~b:[ 2; 2 ] ~o:[ 2 ]
      ~f:(fun _ -> 1.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op hn = p *. 2 in
  let%op row = hn @| outer in
  let%op nested_view = row @| inner in
  let%op l = nested_view *. nested_view in
  refused "a view of a view of a computed node" ~bindings ~symbols:[ outer; inner ]
    ~params:(one nested_view) ~node:"hn" l;
  Verdict.p "that view is an alias" (Tn.is_alias nested_view.Tensor.value);
  (* A padded view is a copy, and a copy is a fetch its forward re-establishes: [Context.compile]
     refuses the step on it (gh-ocannl-1876), before any run. *)
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
  refused_with "a copying view of a computed node" ~at:"compile"
    ~names:[ "copied_view"; "gh-ocannl-1876" ]
    (fused_runs ~bindings ~symbols:[ batch_n ] ~step:(sgd ~params:(one copied_view)) processed);
  Verdict.p "that view cannot alias its parent"
    (not (Asgns.slice_alias_eligible ~array:copied_view.Tensor.value ~sliced:hp.Tensor.value))

let plain_forward_write_refused () =
  (* A write the forward makes owns no fragment of the node it writes, and undoes a step the same:
     here over a parameter, in the default stepped set. The routine persists the node it
     re-establishes, so its own next run would undo its own step. *)
  let x = leaf ~l:"xw" ~grad_spec:Tensor.Prohibit_grad [| 1.; 2. |] in
  let overwritten = TDSL.param ~value:1. "overwritten" () in
  Shape.infer_equal overwritten.Tensor.shape x.Tensor.shape;
  let%op l = overwritten *. overwritten in
  let reset = [%cd overwritten =: x] in
  let l = { l with Tensor.forward = Asgns.sequence [ reset; l.Tensor.forward ] } in
  refused "a parameter its forward overwrites" ~node:"overwritten" l

let step_before_forward_refused () =
  (* The step sequenced before the forward recomputing its node is undone within the same run:
     [Asgns.sequence] refuses it as the routine is assembled. *)
  let x = leaf ~l:"xf" [| 1.; 2. |] in
  let%op hf = x *. 2 in
  let%op l = hf *. hf in
  Train.set_materialized hf.Tensor.value;
  let step = sgd ~params:(one hf) l in
  match Asgns.sequence [ step; Train.grad_update l ] with
  | _ -> Verdict.p "a step sequenced before its node's forward: refused" false
  | exception Utils.User_error msg ->
      eprintf "a step sequenced before its node's forward: %s (not part of the golden)\n" msg;
      Verdict.p "a step sequenced before its node's forward: refused" true;
      List.iter [ "hf"; "member 1"; "member 2"; "gh-ocannl-2027" ] ~f:(fun substring ->
          Verdict.p
            ("a step sequenced before its node's forward: the refusal names " ^ substring)
            (String.is_substring msg ~substring))

(* ----- 3. Why ----- *)

let bare_step_refused () =
  (* Bare [sgd_one] sees no loss: before gh-ocannl-2027 nothing refused it, and each run recomputed
     (2, 4) and took the one step d = 2 h: h - h / 4, losing the previous one. The first run is that
     step; the second is refused before it launches, so the first run's value stays. *)
  let x = leaf ~l:"xb" [| 1.; 2. |] in
  let%op hb = x *. 2 in
  let%op l = hb *. hb in
  Train.set_materialized hb.Tensor.value;
  let step = Train.sgd_one ~learning_rate:(learning_rate ()) hb in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, routine = Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty in
  let ctx = Context.run ctx routine in
  matches "bare sgd_one on a computed node: run 1" (values ctx hb) [| 1.5; 3. |];
  (match Context.run ctx routine with
  | _ -> Verdict.p "bare sgd_one on a computed node: run 2 refused" false
  | exception Utils.User_error msg ->
      eprintf "bare sgd_one: %s (not part of the golden)\n" msg;
      Verdict.p "bare sgd_one on a computed node: run 2 refused" true;
      Verdict.p "bare sgd_one on a computed node: the refusal names hb"
        (String.is_substring msg ~substring:"hb"));
  matches "bare sgd_one on a computed node: the refused run left the step" (values ctx hb)
    [| 1.5; 3. |]

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

(* ----- 5. Across routines (gh-ocannl-2027) ----- *)

(* [ho = x *. 2] computed by a forward routine of its own and run once, then stepped by [step] three
   times, each step the one [sgd_one] takes on [ho *. ho]: [ho - lr * 2 ho]. Returns the context,
   the forward routine and the steps' value. *)
let stepped_after_forward ~tag ~step =
  let x = leaf ~l:("xs" ^ tag) [| 1.; 2. |] in
  let%op hs = x *. 2 in
  Train.set_materialized hs.Tensor.value;
  let ctx, fwd = Train.to_routine (Context.auto ()) IDX.empty (Train.forward hs) in
  let ctx = Context.run ctx fwd in
  let%op l = hs *. hs in
  let ctx, routine =
    Context.compile ctx (Asgns.sequence [ Train.grad_update l; step hs l ]) IDX.empty
  in
  let ctx = List.fold [ 1; 2; 3 ] ~init:ctx ~f:(fun ctx _ -> Context.run ctx routine) in
  let stepped =
    Array.fold [| 1; 2; 3 |] ~init:[| 2.; 4. |] ~f:(fun v _ ->
        Array.map v ~f:(fun v -> v -. (lr *. 2. *. v)))
  in
  (ctx, fwd, hs, stepped)

let rerun_refused name ~ctx ~fwd ~node ~stepped ~watched =
  matches (name ^ ": the steps accumulate") (values ctx watched) stepped;
  (match Context.run ctx fwd with
  | _ -> Verdict.p (name ^ ": re-running the forward refused") false
  | exception Utils.User_error msg ->
      eprintf "%s: %s (not part of the golden)\n" name msg;
      Verdict.p (name ^ ": re-running the forward refused") true;
      List.iter [ node; "gh-ocannl-2027" ] ~f:(fun substring ->
          Verdict.p (name ^ ": the refusal names " ^ substring) (String.is_substring msg ~substring)));
  matches (name ^ ": the refused run left the steps") (values ctx watched) stepped

let separate_forward_rerun () =
  (* The forward run once before the steps is legal ([computed_once]); running it again after them
     would undo them, and is refused, through [sgd_update], bare [sgd_one] and a hand-written step
     marked [Asgns.persisting]. *)
  let marked_by name step =
    let ctx, fwd, hs, stepped = stepped_after_forward ~tag:name ~step in
    rerun_refused ("a forward re-run after " ^ name) ~ctx ~fwd ~node:"hs" ~stepped ~watched:hs
  in
  marked_by "sgd_update" (fun hs l -> sgd ~params:(one hs) l);
  marked_by "sgd_one" (fun hs _ -> Train.sgd_one ~learning_rate:(learning_rate ()) hs);
  let hand_written hs =
    let learning_rate = learning_rate () in
    [%cd hs =- learning_rate * hs.grad ~logic:"."]
  in
  marked_by "a marked hand-written step" (fun hs _ ->
      Asgns.persisting [ hs.Tensor.value ] (hand_written hs));
  (* Unmarked, the same step states no intent: the re-run is not refused, and undoes the steps. *)
  let ctx, fwd, hs, stepped =
    stepped_after_forward ~tag:"unmarked" ~step:(fun hs _ -> hand_written hs)
  in
  matches "an unmarked hand-written step: the steps accumulate" (values ctx hs) stepped;
  let ctx = Context.run ctx fwd in
  matches "an unmarked hand-written step: a re-run forward undoes them" (values ctx hs) [| 2.; 4. |];
  (* Through a view built after its parent's forward routine ran: the step reaches the parent's
     buffer, which the parent's forward re-establishes. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p = rows ~l:"pr" in
  let%op hr = p *. 2 in
  Train.set_materialized hr.Tensor.value;
  let ctx, fwd = Train.to_routine (Context.auto ()) IDX.empty (Train.forward hr) in
  let ctx = Context.run ctx fwd in
  let%op late_view = hr @| batch_n in
  let%op l = late_view *. late_view in
  let ctx, routine =
    Train.to_routine ctx bindings
      (Asgns.sequence [ Train.grad_update l; sgd ~params:(one late_view) l ])
  in
  Verdict.p "a late view of a re-run forward's node is an alias"
    (Tn.is_alias late_view.Tensor.value);
  IDX.find_exn routine.Context.bindings batch_n := 0;
  let ctx = List.fold [ 1; 2; 3 ] ~init:ctx ~f:(fun ctx _ -> Context.run ctx routine) in
  let row0 =
    Array.fold [| 1; 2; 3 |] ~init:[| 2.; 4. |] ~f:(fun v _ ->
        Array.map v ~f:(fun v -> v -. (lr *. 2. *. v)))
  in
  rerun_refused "a forward re-run after a step through a late view" ~ctx ~fwd ~node:"hr"
    ~stepped:(Array.append row0 [| 6.; 8. |])
    ~watched:hr

let replaced_values () =
  (* A deliberate replacement supersedes the steps: a host write, and a run replacing the value
     without carrying the node's forward code. The forward then runs again, and recomputes. *)
  let ctx, fwd, hs, _ =
    stepped_after_forward ~tag:"host" ~step:(fun hs l -> sgd ~params:(one hs) l)
  in
  let ctx = Context.set_values ctx hs.Tensor.value [| 5.; 7. |] in
  (match Context.run ctx fwd with
  | ctx ->
      Verdict.p "after a host write, the forward re-runs: accepted" true;
      matches "after a host write, the forward recomputes" (values ctx hs) [| 2.; 4. |]
  | exception Utils.User_error msg ->
      Verdict.p ("after a host write, the forward re-runs: accepted (refused: " ^ msg ^ ")") false);
  let ctx, fwd, hs, _ =
    stepped_after_forward ~tag:"replaced" ~step:(fun hs l -> sgd ~params:(one hs) l)
  in
  let y = leaf ~l:"yr" ~grad_spec:Tensor.Prohibit_grad [| 3.; 5. |] in
  let ctx, replace = Context.compile ~name:"replace_hs" ctx [%cd hs =: y] IDX.empty in
  let ctx = Context.run ctx replace in
  matches "a run replacing the stepped value" (values ctx hs) [| 3.; 5. |];
  (match Context.run ctx fwd with
  | ctx ->
      Verdict.p "after a replacing run, the forward re-runs: accepted" true;
      matches "after a replacing run, the forward recomputes" (values ctx hs) [| 2.; 4. |]
  | exception Utils.User_error msg ->
      Verdict.p
        ("after a replacing run, the forward re-runs: accepted (refused: " ^ msg ^ ")")
        false);
  (* Re-initializing stepped parameters replaces them deliberately too. *)
  let xs = [| 1.; 2. |] in
  let xi = leaf ~l:"xri" ~grad_spec:Tensor.Prohibit_grad xs in
  let w = TDSL.param ~value:1. "wri" ~output_dims:[ 2 ] () in
  let%op l = w *. xi in
  let step = sgd l in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let ctx, routine = Context.compile ctx (Asgns.sequence [ Train.grad_update l; step ]) IDX.empty in
  let ctx = Context.run (Context.run ctx routine) routine in
  matches "a stepped parameter" (values ctx w) (Array.map xs ~f:(fun x -> 1. -. (2. *. lr *. x)));
  match Train.init_params ~reinit_all:true ctx IDX.empty l with
  | ctx ->
      Verdict.p "re-initializing stepped parameters: accepted" true;
      matches "re-initializing stepped parameters restores them" (values ctx w) [| 1.; 1. |];
      let ctx = Context.run ctx routine in
      matches "a step after re-initializing" (values ctx w)
        (Array.map xs ~f:(fun x -> 1. -. (lr *. x)))
  | exception Utils.User_error msg ->
      Verdict.p ("re-initializing stepped parameters: accepted (refused: " ^ msg ^ ")") false

let kept_through_members () =
  (* A step followed, as separate members of one flat sequence, by a copy out to a temporary and
     back: the members after the step carry its value, so it lasts. (A [%cd] body nests each
     statement with the ones after it, so the members are built one by one.) *)
  let xs = [| 1.; 2. |] in
  let x = leaf ~l:"xk" ~grad_spec:Tensor.Prohibit_grad xs in
  let w = TDSL.param ~value:1. "wk" ~output_dims:[ 2 ] () in
  let%op l = w *. x in
  let step = sgd l in
  let keep = NTDSL.term ~label:[ "keep" ] () in
  let out = [%cd keep =: w] and back = [%cd w =: keep] in
  match Asgns.sequence [ step; out; back ] with
  | exception Utils.User_error msg ->
      Verdict.p
        ("a step kept through a temporary across members: accepted (refused: " ^ msg ^ ")")
        false
  | update ->
      Verdict.p "a step kept through a temporary across members: accepted" true;
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx, routine =
        Context.compile ctx (Asgns.sequence [ Train.grad_update l; update ]) IDX.empty
      in
      tracks "a step kept through a temporary across members"
        ~run:(fun ctx -> Context.run ctx routine)
        ~ctx ~watched:w ~start:[| 1.; 1. |]
        ~oracle:(Array.mapi ~f:(fun i v -> v -. (lr *. xs.(i))))

let partial_replacement () =
  (* Replacing one row of a stepped computed node, through a view, leaves the steps in the other
     row: re-running the node's forward is still refused. *)
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let p = rows ~l:"pz" in
  let%op hz = p *. 2 in
  Train.set_materialized hz.Tensor.value;
  let ctx, fwd = Train.to_routine (Context.auto ()) IDX.empty (Train.forward hz) in
  let ctx = Context.run ctx fwd in
  let%op l = hz *. hz in
  let ctx, routine =
    Context.compile ctx (Asgns.sequence [ Train.grad_update l; sgd ~params:(one hz) l ]) IDX.empty
  in
  let ctx = List.fold [ 1; 2; 3 ] ~init:ctx ~f:(fun ctx _ -> Context.run ctx routine) in
  let%op part = hz @| batch_n in
  let y = leaf ~l:"yz" ~grad_spec:Tensor.Prohibit_grad [| 5.; 7. |] in
  let ctx, replace = Context.compile ~name:"replace_part" ctx [%cd part =: y] bindings in
  Verdict.p "the replaced part is an alias" (Tn.is_alias part.Tensor.value);
  IDX.find_exn replace.Context.bindings batch_n := 0;
  let ctx = Context.run ctx replace in
  (* Three steps of 1 - 2 lr on the second row, (6, 8): 0.75^3 = 0.421875. *)
  rerun_refused "a forward re-run after a part of the stepped node was replaced" ~ctx ~fwd
    ~node:"hz" ~stepped:[| 5.; 7.; 2.53125; 3.375 |] ~watched:hz

let () =
  classification ();
  computed_refused ();
  views_refused ();
  plain_forward_write_refused ();
  step_before_forward_refused ();
  bare_step_refused ();
  accumulating_node ();
  statistics ();
  view_of_persistent_parent ();
  computed_initializer ();
  computed_once ();
  separate_forward_rerun ();
  replaced_values ();
  kept_through_members ();
  partial_replacement ()
