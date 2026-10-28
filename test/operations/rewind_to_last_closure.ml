(* gh-ocannl-1301: a construction that fails after mutating the inference session taints it, and
   [Tensor.rewind_to_last_closure] discards everything opened since the last successful closure,
   keeping closed and compiled work.

   One process runs REPL-style sequences without [unsafe_reinitialize]: fail, use (refused, naming
   the original failure), rewind, use the discarded handles (refused, naming the rewind), and build
   on the closed model again -- compiled, run, and compared against the same valid construction run
   in a FRESH process (this executable re-run with [fresh]). All data are dyadic constants, so the
   values do not depend on node ids, and the comparison is exact. *)

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
  | exception exn -> Error (Exn.to_string exn)

let refused_with f ~substrings =
  match outcome f with
  | Ok () -> false
  | Error msg ->
      Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
      (* Every refusal names something: an empty list would pass any refusal. *)
      (not (List.is_empty substrings))
      && List.for_all substrings ~f:(fun substring -> String.is_substring msg ~substring)

let tainted = "the session is inconsistent after an earlier shape error at"
let discarded = "discarded by the rewind after"

let leaf ?(grad_spec = Tensor.Prohibit_grad) name values =
  Tensor.term_init values ~label:[ name ] ~grad_spec ~output_dims:[ Array.length values ] ()

(* Model A: y = w *. x, w differentiable. *)
let model_a () =
  let x = leaf "rw_x" [| 1.; 2.; 3. |] in
  let w = leaf ~grad_spec:Tensor.Require_grad "rw_w" [| 0.5; -1.; 2. |] in
  let%op y = w *. x in
  (w, y)

(* Construction C over model A: loss = sum (y - t)^2, forward and backprop. With y = [0.5; -2; 6]
   and t = 1: loss = 34.25, dloss/dw = 2 (y - t) x = [-1; -12; 30]. *)
let construction_c ctx (w, y) =
  let t = leaf "rw_t" [| 1.; 1.; 1. |] in
  let%op d = y - t in
  let%op loss = (d *. d) ++ "i => 0" in
  let grad = (Option.value_exn ~here:[%here] w.Tensor.diff).grad in
  Train.set_materialized grad;
  let ctx = Train.update_once ctx loss in
  (Context.get_values ctx loss.Tensor.value, Context.get_values ctx grad)

let hex xs = String.concat ~sep:" " (Array.to_list (Array.map xs ~f:(Printf.sprintf "%h")))

let fresh_leg () =
  let ctx = Context.auto () in
  let loss, grad = construction_c ctx (model_a ()) in
  Stdio.printf "RESULT loss %s\nRESULT grad %s\n" (hex loss) (hex grad)

let parse_fresh stdout =
  let field key =
    List.find_map (String.split_lines stdout) ~f:(fun line ->
        String.chop_prefix line ~prefix:("RESULT " ^ key ^ " "))
    |> Option.value_map ~default:[||] ~f:(fun s ->
        Array.of_list_map (String.split s ~on:' ') ~f:Float.of_string)
  in
  (field "loss", field "grad")

let scenario () =
  let ctx = Context.auto () in
  (* 1. A failure before the first closure: the checkpoint is the empty session. *)
  let t0 = leaf "rw_t0" [| 1.; 2. |] in
  let t3 = leaf "rw_t3" [| 1.; 2.; 3. |] in
  let first_failure = outcome (fun () -> TDSL.O.( + ) t0 t3) in
  let first_msg = match first_failure with Error msg -> msg | Ok () -> "" in
  p "before any closure: a pointwise op over incompatible dims fails"
    (Result.is_error first_failure);
  p "the failure taints the session: the next construction is refused, naming it"
    (refused_with (fun () -> NTDSL.O.relu t0) ~substrings:[ tainted; first_msg ]);
  Tensor.rewind_to_last_closure ();
  p "after the rewind, an operand from the failed span is refused, naming the rewind"
    (refused_with (fun () -> NTDSL.O.relu t0) ~substrings:[ discarded; first_msg ]);
  p "a stale-handle refusal fails before mutating anything: the session stays usable"
    (Result.is_ok (outcome (fun () -> NTDSL.O.relu (leaf "rw_ok" [| 1. |]))));
  (* 2. Model B, compiled: closed and consumed. *)
  let b = leaf "rw_b" [| 1.; 2.; 3. |] in
  let%op bz = b *. 2. in
  let ctx = Train.forward_once ctx bz in
  let bz_values = Context.get_values ctx bz.Tensor.value in
  p_all2 "model B computes 2b" bz_values [| 2.; 4.; 6. |] ~f:Float.equal;
  (* 3. Model A, closed without compiling: [y] stays a forward and a backprop root. *)
  let ((w, y) as model) = model_a () in
  ignore (Lazy.force y.Tensor.value.Ir.Tnode.dims : int array);
  p "model A is closed" y.Tensor.shape.Shape.closed;
  p "model A's result is a forward and a backprop root"
    (Tensor.is_fwd_root y && Tensor.is_bprop_root y);
  (* 4. Open work over the closed models: a [%cd] embedding of [y] (handout marker [Taken]), a
     consumer [h] of [y] (drops its backprop root), a second consumer of [bz]; then a failure. *)
  let acc = leaf "rw_acc" [| 0.; 0.; 0. |] in
  let acc_code = [%cd acc =+ y] in
  (* A program that only WRITES a discarded node (reads the live, compiled [bz]). *)
  let out = leaf "rw_out" [| 0.; 0.; 0. |] in
  let out_code = [%cd out =: bz] in
  let%op h = y *. 3. in
  let%op bh = bz + 1. in
  let big =
    TDSL.ndarray (Array.init 20 ~f:Float.of_int) ~label:[ "rw_big" ] ~output_dims:[ 20 ] ()
  in
  p "the open work consumed model A's roots"
    ((not (Tensor.is_fwd_root y)) && not (Tensor.is_bprop_root y));
  let bad = leaf "rw_bad" [| 1.; 2. |] in
  let failure = outcome (fun () -> TDSL.O.( + ) h bad) in
  let msg = match failure with Error msg -> msg | Ok () -> "" in
  p "a pointwise op over incompatible dims fails" (Result.is_error failure);
  (* 5. While tainted: constructions, finalization, code handout and compilation are refused. *)
  p "tainted: a construction over the closed model is refused, naming the first failure"
    (refused_with (fun () -> NTDSL.O.relu y) ~substrings:[ tainted; msg ]);
  p "tainted: finalizing inference (an open tensor's dims) is refused"
    (refused_with (fun () -> Shape.to_dims h.Tensor.shape) ~substrings:[ tainted ]);
  p "tainted: a closed tensor's committed dims stay readable"
    (Result.is_ok (outcome (fun () -> Shape.to_dims y.Tensor.shape)));
  p "tainted: Train.forward_once is refused"
    (refused_with (fun () -> Train.forward_once ctx y) ~substrings:[ tainted ]);
  p "tainted: compiling a program handed out earlier is refused"
    (refused_with
       (fun () -> Context.compile ctx bz.Tensor.forward Ir.Indexing.Empty)
       ~substrings:[ tainted ]);
  (* 6. The rewind. *)
  let next_before = Tensor.get_next_id () in
  Tensor.rewind_to_last_closure ();
  p "after the rewind, model A's result is a forward and a backprop root again"
    (Tensor.is_fwd_root y && Tensor.is_bprop_root y);
  p "model B stays consumed" (not (Tensor.is_fwd_root bz));
  p "ids are never reused" (Tensor.get_next_id () >= next_before);
  List.iter
    [ ("h", h); ("bh", bh); ("acc", acc); ("big", big) ]
    ~f:(fun (name, t) ->
      pf "discarded %s: used as an operand, refused naming the rewind and the failure" name
        (refused_with (fun () -> NTDSL.O.relu t) ~substrings:[ discarded; msg ]);
      pf "discarded %s: compiled, refused" name
        (refused_with (fun () -> Train.forward_once ctx t) ~substrings:[ discarded ]);
      pf "discarded %s: read through the context, refused" name
        (refused_with (fun () -> Context.get_values ctx t.value) ~substrings:[ discarded ]);
      pf "discarded %s: printed, refused" name
        (refused_with
           (fun () -> Tensor.print ~force:false ~with_code:false ~with_grad:false `Inline t)
           ~substrings:[ discarded ]));
  p "a program that only writes a discarded node: compiled, refused"
    (refused_with
       (fun () -> Context.compile ctx out_code Ir.Indexing.Empty)
       ~substrings:[ discarded ]);
  p "a [%cd] assignment into a discarded tensor is refused at construction"
    (refused_with (fun () -> [%cd out =: bz]) ~substrings:[ discarded; "%cd assignment" ]);
  p "discarded [%cd] program: compiled, refused"
    (refused_with
       (fun () -> Context.compile ctx acc_code Ir.Indexing.Empty)
       ~substrings:[ discarded ]);
  p "model B's context values survive the rewind"
    (Array.equal Float.equal (Context.get_values ctx bz.Tensor.value) bz_values);
  let%op bz1 = bz + 1. in
  let ctx = Train.forward_once ctx bz1 in
  p_all2 "a new consumer of the compiled model B runs"
    (Context.get_values ctx bz1.Tensor.value)
    [| 3.; 5.; 7. |] ~f:Float.equal;
  (* 7. Construction C over the closed, uncompiled model A, against the fresh process. *)
  let loss, grad = construction_c ctx model in
  let child = Fresh_process.run [ "fresh" ] in
  let status, child_out, _ = child in
  p "the fresh-process reference ran" (match status with Unix.WEXITED 0 -> true | _ -> false);
  if not (match status with Unix.WEXITED 0 -> true | _ -> false) then
    Fresh_process.report ~label:"fresh leg" child;
  let fresh_loss, fresh_grad = parse_fresh child_out in
  p_all2 "construction C after the rewind: the loss equals the fresh process's" loss fresh_loss
    ~f:Float.equal;
  p_all2 "construction C after the rewind: w's gradient equals the fresh process's" grad fresh_grad
    ~f:Float.equal;
  p_all2 "and the fresh process computed the closed form" fresh_grad [| -1.; -12.; 30. |]
    ~f:Float.equal;
  ignore (w : Tensor.t)

(* Mutation timing and repeated recovery. *)
let timing () =
  let fail_after_inner ~t:_ ~t1 ~projections:_ =
    (* An inner construction registers its step, then the outer [op_asn] fails. *)
    ignore (NTDSL.O.relu t1 : Tensor.t);
    failwith "rw: op_asn failure after an inner construction"
  in
  let fail_before ~t:_ ~t1:_ ~projections:_ = failwith "rw: op_asn failure before any mutation" in
  let%cd copy_asn ~t ~t1 ~projections = v =: v1 in
  let fail_grad ~t:_ ~g:_ ~t1:_ ~projections:_ = failwith "rw: grad_asn failure" in
  let no_grad ~t:_ ~g:_ ~t1:_ ~projections:_ = Ir.Assignments.empty_comp in
  let unop ~op_asn ~grad_asn ?(grad_spec = Tensor.Prohibit_grad) t1 =
    Tensor.unop ~op_label:"rw_probe" ~op_asn ~grad_asn ~grad_spec t1 ()
  in
  let closed = leaf ~grad_spec:Tensor.Require_grad "rw_closed" [| 1.; 2. |] in
  ignore (Lazy.force closed.Tensor.value.Ir.Tnode.dims : int array);
  let case name ~expect_taint build =
    let failure = outcome build in
    pf "%s: fails" name (Result.is_error failure);
    let next = outcome (fun () -> NTDSL.O.relu closed) in
    if expect_taint then
      pf "%s: taints the session (the next construction is refused)" name
        (match next with Error msg -> String.is_substring msg ~substring:tainted | Ok () -> false)
    else
      pf "%s: does not taint the session (the next construction succeeds)" name (Result.is_ok next);
    Tensor.rewind_to_last_closure ();
    pf "%s: after the rewind, the closed tensor is usable" name
      (Result.is_ok (outcome (fun () -> NTDSL.O.relu closed)))
  in
  case "op_asn failing after an inner construction" ~expect_taint:true (fun () ->
      unop ~op_asn:fail_after_inner ~grad_asn:no_grad closed);
  case "op_asn failing before any mutation" ~expect_taint:false (fun () ->
      unop ~op_asn:fail_before ~grad_asn:no_grad closed);
  case "grad_asn failing after registration" ~expect_taint:true (fun () ->
      unop ~op_asn:copy_asn ~grad_asn:fail_grad ~grad_spec:Tensor.Require_grad closed);
  (* An [op_asn] that embeds a closed root through [%cd] ([take_forward_code]: a root-map change
     only, no solver write) and then fails before registering: the session is tainted, and the
     rewind gives the root back. *)
  let root = leaf "rw_root" [| 1.; 2. |] in
  ignore (Lazy.force root.Tensor.value.Ir.Tnode.dims : int array);
  let embed_then_fail ~t:_ ~t1:_ ~projections:_ =
    (* What [%cd] generates for a tensor filler that is a forward root. *)
    if Tensor.is_fwd_root root then ignore (Tensor.take_forward_code root : Ir.Assignments.comp);
    failwith "rw: op_asn failure after a %cd embedding"
  in
  p "a closed root is a forward root" (Tensor.is_fwd_root root);
  case "op_asn failing after a %cd embedding" ~expect_taint:true (fun () ->
      unop ~op_asn:embed_then_fail ~grad_asn:no_grad closed);
  p "after the rewind, the root the failed op_asn embedded is a forward root again"
    (Tensor.is_fwd_root root);
  (* Repeated recovery: an empty rewind, and the random-seed cache. *)
  let next = Tensor.get_next_id () in
  Tensor.rewind_to_last_closure ();
  p "an empty rewind allocates nothing and leaves closed work usable"
    (Tensor.get_next_id () = next && Result.is_ok (outcome (fun () -> NTDSL.O.relu closed)));
  Tensor.rewind_to_last_closure ();
  let _draw = TDSL.uniform () ~output_dims:[ 2 ] () in
  let seed = Tensor.get_random_seed () in
  Tensor.rewind_to_last_closure ~reason:"rw: discarding a draw" ();
  p "a seed created in the discarded span is dropped from the cache"
    (refused_with (fun () -> NTDSL.O.relu seed) ~substrings:[ discarded; "rw: discarding a draw" ]
    && Result.is_ok
         (outcome (fun () -> Tensor.check_not_discarded ~what:"rw" (Tensor.get_random_seed ()))))

(* Static-symbol roles are claimed by open work and committed by a closure: a discarded slice use
   leaves the symbol free for an extent use, a committed one does not. *)
let symbol_roles () =
  let extent_use sym =
    (* A parameter with an inferred shape: the captured axis is still open. *)
    let%op wa = { wa = 0.5 } in
    let%op _wa2 = wa ++ "u=>u" [ "u" ] in
    Shape.set_sym_dim u sym
  in
  let slice_use sym =
    let mb = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
    let%op sl = mb @| sym in
    sl
  in
  Tensor.rewind_to_last_closure ();
  let discarded_sym, _ = Ir.Indexing.get_static_symbol ~static_range:2 Ir.Indexing.Empty in
  ignore (slice_use discarded_sym : Tensor.t);
  Tensor.rewind_to_last_closure ~reason:"rw: discarding a slice use" ();
  p "a slice role claimed by discarded work does not stick: the symbol can be an extent"
    (Result.is_ok (outcome (fun () -> extent_use discarded_sym)));
  Tensor.rewind_to_last_closure ();
  let committed_sym, _ = Ir.Indexing.get_static_symbol ~static_range:2 Ir.Indexing.Empty in
  let sl = slice_use committed_sym in
  ignore (Shape.to_dims sl.Tensor.shape : int array);
  p "a slice role committed by a closure sticks"
    (committed_sym.Ir.Indexing.used_as_slice
    && refused_with (fun () -> extent_use committed_sym) ~substrings:[ "already slices" ])

(* Caller-retained handles beyond tensors: a variable reference the discarded work bound (and
   inferred a dimension into), and the random-seed singleton replaced since the closure. *)
let retained_handles () =
  Tensor.rewind_to_last_closure ();
  let x = leaf "rw_cap" [| 1.; 2.; 3. |] in
  let%op _s = x ++ "a=>0" [ "a" ] in
  Tensor.rewind_to_last_closure ~reason:"rw: discarding a capture" ();
  p "a variable reference bound by discarded work is refused, naming the rewind"
    (refused_with
       (fun () -> Shape.set_dim a 3)
       ~substrings:[ "capture it afresh"; "rw: discarding a capture" ]);
  let fresh = leaf "rw_fresh" [| 1.; 2.; 3. |] in
  let fresh_prec = Ir.Tnode.get_specified_prec fresh.Tensor.value in
  p "a two-operand precision setter over a discarded operand is refused before writing the other"
    (refused_with
       (fun () -> Ocannl.Operation.threefry4x32_crypto fresh x ())
       ~substrings:[ discarded ]
    && Option.equal Ir.Ops.equal_prec fresh_prec (Ir.Tnode.get_specified_prec fresh.Tensor.value));
  let init ?label:_ ?top_down_prec:_ ?batch_dims:_ ?batch_axes:_ ?input_dims:_ ?output_dims:_
      ?input_axes:_ ?output_axes:_ ?deduced:_ () =
    x
  in
  p "a parameter whose initializer returns a discarded tensor is refused"
    (refused_with (fun () -> Tensor.param ~t:init "rw_p" ()) ~substrings:[ discarded ]);
  (* A low-level node created after the rewind, in the session namespace with an id inside the
     discarded range (arrayjit clients choose their own ids), is not a discarded node. *)
  let low =
    Ir.Tnode.create (Ir.Tnode.Default Ir.Ops.single) ~id:x.Tensor.value.Ir.Tnode.id
      ~label:[ "rw_lowlevel" ]
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  p "a node created after the rewind with a discarded id is not stale"
    (Result.is_ok
       (outcome (fun () -> Option.iter !Ir.Tnode.validity_check ~f:(fun check -> check low))));
  let seed = Tensor.get_random_seed () in
  ignore (Shape.to_dims seed.Tensor.shape : int array);
  Tensor.set_random_seed ~seed:5 ();
  Tensor.rewind_to_last_closure ();
  p "a seed replaced since the closure is restored to the closure's"
    (phys_equal seed (Tensor.get_random_seed ()));
  (* A scoped seed installed by [with_saved_random_seed] around a closure is not the closed state's
     seed. *)
  Tensor.with_saved_random_seed (fun () ->
      Tensor.set_random_seed ~seed:9 ();
      let scoped = Tensor.get_random_seed () in
      ignore (Shape.to_dims scoped.Tensor.shape : int array));
  Tensor.rewind_to_last_closure ();
  p "a scoped seed closed inside with_saved_random_seed does not come back on a rewind"
    (phys_equal seed (Tensor.get_random_seed ()));
  (* A closed root embedded by [%cd] (handout marker [Taken]) and then handed out with
     [consume_forward_code]: the caller holds its code, so a rewind does not make it a root
     again. *)
  let y2 = leaf "rw_y2" [| 1.; 2. |] in
  ignore (Lazy.force y2.Tensor.value.Ir.Tnode.dims : int array);
  let acc2 = leaf "rw_acc2" [| 0.; 0. |] in
  let _embedding = [%cd acc2 =+ y2] in
  ignore (Tensor.consume_forward_code y2 : Ir.Assignments.comp);
  Tensor.rewind_to_last_closure ();
  p "a root handed out after a %cd embedding stays handed out across a rewind"
    (not (Tensor.is_fwd_root y2));
  (* Inference-set writes count as mutations: a spec failing inside the solver's constraint
     derivation over an open operand, after marking its row variable, taints. *)
  let%op xo = { xo = 0.5 } in
  let failure = outcome (fun () -> TDSL.einsum1 "..a.. a => a" xo ()) in
  pf "a spec failing after marking an open operand's row (%s) taints the session"
    (match failure with Error _ -> "it fails" | Ok () -> "it did not fail")
    (Result.is_error failure
    &&
    match outcome (fun () -> NTDSL.O.relu (leaf "rw_after" [| 1. |])) with
    | Error msg -> String.is_substring msg ~substring:tainted
    | Ok () -> false);
  Tensor.rewind_to_last_closure ();
  (* A [%cd] assignment to the gradient of a tensor that has none is refused BEFORE its step is
     registered: no phantom step constrains the open target afterwards. *)
  let b2 = leaf "rw_b2" [| 1.; 2. |] in
  ignore (Lazy.force b2.Tensor.value.Ir.Tnode.dims : int array);
  let ng =
    Tensor.term_init [| 1.; 2.; 3. |] ~label:[ "rw_ng" ] ~grad_spec:Tensor.Prohibit_grad ()
  in
  p "a %cd assignment to a missing gradient is refused"
    (Result.is_error (outcome (fun () -> [%cd ng.grad =: b2])));
  p "and leaves no step behind: the target's own shape still closes as its data says"
    (match outcome (fun () -> Shape.to_dims ng.Tensor.shape) with
    | Ok () -> Array.equal Int.equal (Shape.to_dims ng.Tensor.shape) [| 3 |]
    | Error _ -> false);
  (* A [%cd] assignment reading a discarded tensor into a live one is refused before registering,
     and taints nothing. *)
  let stale = leaf "rw_stale_src" [| 1.; 2. |] in
  Tensor.rewind_to_last_closure ~reason:"rw: discarding a source" ();
  p "a %cd assignment reading a discarded tensor is refused, naming the rewind"
    (refused_with
       (fun () -> [%cd b2 =: stale])
       ~substrings:[ discarded; "rw: discarding a source" ]);
  p "and the refusal taints nothing" (Result.is_ok (outcome (fun () -> NTDSL.O.relu b2)));
  (* Parameterization is a taint boundary: a failing post-processing hook taints. *)
  Tensor.rewind_to_last_closure ();
  let saved_hook = !Tensor.param_postprocess in
  (Tensor.param_postprocess := fun _ -> failwith "rw: param post-processing failed");
  let init ?label ?top_down_prec:_ ?batch_dims:_ ?batch_axes:_ ?input_dims:_ ?output_dims:_
      ?input_axes:_ ?output_axes:_ ?deduced:_ () =
    Tensor.term_init [| 1.; 2. |] ?label ~grad_spec:Tensor.Prohibit_grad ~output_dims:[ 2 ] ()
  in
  let failed = outcome (fun () -> Tensor.param ~t:init "rw_pp" ()) in
  Tensor.param_postprocess := saved_hook;
  p "a failing parameter post-processing hook taints the session"
    (Result.is_error failed
    &&
    match outcome (fun () -> NTDSL.O.relu b2) with
    | Error msg -> String.is_substring msg ~substring:tainted
    | Ok () -> false);
  Tensor.rewind_to_last_closure ()

(* gh-ocannl-1505: every [Context] entry point that forces a node's dims runs the stale check first.
   The discarded tensor here leaves its dims to the closure (an inferred-shape draw), so forcing
   them after the rewind fails as a shape error that names neither the node nor the rewind. *)
let context_entry_points () =
  Tensor.rewind_to_last_closure ();
  let ctx = Context.auto () in
  let live = leaf "rw_live" [| 1.; 2. |] in
  let ctx = Train.forward_once ctx live in
  let open_draw = TDSL.uniform () () in
  let%cd open_code = live =: open_draw in
  Tensor.rewind_to_last_closure ~reason:"rw: discarding an open draw" ();
  let tn = open_draw.Tensor.value in
  let stale =
    [
      discarded;
      "rw: discarding an open draw";
      (Ir.Tnode.debug_name tn
       [@rendered_name "the refusal names the node as the library renders it"]);
    ]
  in
  p "the discarded draw's dims were never forced" (not (Lazy.is_val tn.Ir.Tnode.dims));
  p "Context.set_values on a discarded node is refused, naming the rewind"
    (refused_with (fun () -> Context.set_values ctx tn [| 1. |]) ~substrings:stale);
  p "Context.copy of a discarded node is refused, naming the rewind"
    (refused_with (fun () -> Context.copy ~src:ctx ~dst:ctx tn) ~substrings:stale);
  (* Any of the program's discarded nodes may be the one named: the draw embeds its own. *)
  p "Context.decision_surface over a discarded program is refused, naming the rewind"
    (refused_with
       (fun () -> Context.decision_surface ctx open_code Ir.Indexing.Empty)
       ~substrings:[ discarded; "rw: discarding an open draw" ]);
  p "no refusal computed the discarded node's dims" (not (Lazy.is_val tn.dims));
  (* The analyze-only lowering takes the session gate like a compile: a closed program, lowerable
     before the failure, is refused while the session is tainted. *)
  let lower () =
    ignore
      (Context.lowered_for_decisions ~name:"rw_decisions" ctx live.Tensor.forward Ir.Indexing.Empty
        : Ir.Low_level.optimized)
  in
  p "Context.lowered_for_decisions lowers a closed program" (Result.is_ok (outcome lower));
  let bad = outcome (fun () -> TDSL.O.( + ) live (leaf "rw_bad3" [| 1.; 2.; 3. |])) in
  let msg = match bad with Error msg -> msg | Ok () -> "" in
  p "tainted: Context.lowered_for_decisions over the closed program is refused, naming the failure"
    (Result.is_error bad && refused_with lower ~substrings:[ tainted; msg ]);
  Tensor.rewind_to_last_closure ()

(* gh-ocannl-1505: a rewind discards exactly the nodes the frontend allocated since the closure. A
   node created in the open window by anyone else -- here a hand-made one, as an arrayjit client or
   [Persistence.load] makes them -- in the session namespace with an id inside the window's id range
   is not open work: an id/uid-range record of the window refused it. *)
let exact_allocated_nodes () =
  Tensor.rewind_to_last_closure ();
  let ctx = Context.auto () in
  let open_t = leaf "rw_open" [| 1.; 2. |] in
  let%op open_u = open_t *. 2. in
  let hand_made =
    Ir.Tnode.create (Ir.Tnode.Default Ir.Ops.single) ~id:open_t.Tensor.value.Ir.Tnode.id
      ~label:[ "rw_hand_made" ]
      ~unpadded_dims:(lazy [| 2 |])
      ~padding:(lazy None)
      ()
  in
  p "the hand-made node shares an open tensor's id and namespace"
    (hand_made.Ir.Tnode.id = open_t.value.id
    && String.equal hand_made.namespace open_t.value.namespace);
  Tensor.rewind_to_last_closure ~reason:"rw: discarding beside a hand-made node" ();
  p "the open tensors' nodes are discarded"
    (refused_with
       (fun () -> Context.get_values ctx open_u.Tensor.value)
       ~substrings:[ discarded; "rw: discarding beside a hand-made node" ]
    && refused_with (fun () -> Context.get_values ctx open_t.Tensor.value) ~substrings:[ discarded ]
    );
  let read_back = ref [||] in
  let written =
    outcome (fun () ->
        let ctx = Context.set_values ctx hand_made [| 0.5; -2. |] in
        read_back := Context.get_values ctx hand_made)
  in
  p "the hand-made node created in the window is not: it is written through the context"
    (match written with
    | Ok () -> true
    | Error msg ->
        Stdio.eprintf "(not part of the golden) hand-made node refused: %s\n%!" msg;
        false);
  p_all2 "and its values read back" !read_back [| 0.5; -2. |] ~f:Float.equal

let () =
  match Sys.get_argv () with
  | [| _; "fresh" |] -> fresh_leg ()
  | _ ->
      scenario ();
      timing ();
      symbol_roles ();
      retained_handles ();
      context_entry_points ();
      exact_allocated_nodes ()
