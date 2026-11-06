(* gh-ocannl-903: explicit checkpoints over the open inference session. [Tensor.checkpoint]
   snapshots the open frontier, [Tensor.rewind] puts it back and discards what the window created.

   The oracles are independent of the snapshot code: what a test can see of the tensors it holds
   (their rows, precision cells, memory-mode intents, ownership and root status), whether a capture
   reference made before the checkpoint takes a new solution, whether the window's tensors are
   refused, and -- the behavioural one -- whether the same valid suffix closes to the same dims as
   in a control session that never took the detour (and, for one fixed scenario, computes the same
   values as a FRESH process). Each restore category is then omitted in turn
   ([Tensor.For_testing.rewind_omitting]): one of those oracles must fail, or the category is
   restoring nothing the oracles see. For a step's projections, which only a closure writes, the
   oracle is the dims they iterate, after a closure that fails having published some
   (gh-ocannl-1827), and [Shape.get_projections] refuses the stale ones (gh-ocannl-1970).
   [For_testing.checkpoint_differences], the machinery's own observer, is checked too, but only as a
   diagnostic: it reads the snapshots it is judging. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row
module Tn = Ir.Tnode

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
      (not (List.is_empty substrings))
      && List.for_all substrings ~f:(fun substring -> String.is_substring msg ~substring)

let discarded = "discarded by the rewind after"
let tainted = "the session is inconsistent after an earlier shape error at"
let cannot = "Tensor.rewind: cannot rewind to"

let leaf ?(grad_spec = Tensor.Prohibit_grad) name n =
  Tensor.term_init
    (Array.init n ~f:(fun i -> Float.of_int (i + 1)))
    ~label:[ name ] ~grad_spec ~output_dims:[ n ] ()

(* {2 The family of open prefixes} *)

type prefix = {
  x : Tensor.t;
  y : Tensor.t;
  r : Shape.delayed_var_ref;  (** Made before the checkpoint, unbound and unsolved. *)
  r_sym : Shape.delayed_var_ref;  (** Likewise, for a symbolic extent. *)
  sym : Ir.Indexing.static_symbol;  (** A static symbol no inference has claimed a role of. *)
  held : Tensor.t list;
}

let build_prefix rng =
  let k = 2 + Random.State.int rng 2 in
  let x = leaf "ck_x" k in
  let%op w = { w = 0.5 } in
  let%op y = w * x in
  let y = if Random.State.bool rng then TDSL.O.relu y else y in
  let extra =
    if Random.State.bool rng then
      let%op e = y *. 2. in
      [ e ]
    else []
  in
  let r = Shape.get_variable_ref "ck_r" in
  let r_sym = Shape.get_variable_ref "ck_r_sym" in
  let sym = Ir.Indexing.make_static_symbol ~static_range:4 (Ir.Indexing.get_symbol ()) in
  (* [w]: a parameter whose output axis is open until a consumer sizes it. *)
  { x; y; r; r_sym; sym; held = [ x; w; y ] @ extra }

(* What the test sees of a tensor it holds, without forcing anything. *)
type seen = {
  rows : string;
  prec : Tn.delayed_prec;
  intent : (Tn.memory_mode * Tn.provenance) option;
  owned : bool;
  observable : bool;
  roots : bool * bool;
}

let see (t : Tensor.t) =
  let sh = t.shape in
  {
    rows =
      String.concat ~sep:" | "
        (List.map [ sh.batch; sh.input; sh.output ] ~f:(fun r -> Sexp.to_string (Row.sexp_of_t r)));
    prec = Tn.Prec_cell.get t.value.delayed_prec_unsafe;
    intent = t.value.memory_mode_intent;
    owned = t.value.context_owned;
    observable = t.value.observable;
    roots = (Tensor.is_fwd_root t, Tensor.is_bprop_root t);
  }

let same_seen a b =
  String.equal a.rows b.rows && phys_equal a.prec b.prec && phys_equal a.intent b.intent
  && Bool.equal a.owned b.owned
  && Bool.equal a.observable b.observable
  && Poly.equal a.roots b.roots

(* The window: writes to every kind of frontier object, then optionally fails. Returns the tensors
   it created that the test can try to reuse, and the capture reference it made and bound. *)
let detour pre ~m ~fail =
  let made = ref [] in
  let window_ref = ref None in
  let note t =
    made := t :: !made;
    t
  in
  (match
     Shape.set_dim pre.r 5;
     (* A role claim on the symbol (Row's pending roles) and a symbolic solution. *)
     Shape.set_sym_dim pre.r_sym pre.sym;
     Train.set_materialized pre.y.value;
     Tn.update_prec pre.x.value Ir.Ops.half;
     ignore (note (Tensor.get_random_seed ()) : Tensor.t);
     let a = note (leaf "ck_a" m) in
     (* Sizes [w]'s open output axis to [m]: rows of [y] (and the solver's environment) rewritten in
        place, over objects older than the checkpoint. *)
     let y = pre.y in
     let%op s = y +* "...|o; ...|o => ...|o" a in
     let s = note s in
     (* A capture reference the window makes and binds. *)
     let%op q = s ++ "...|v => ...|v" [ "v" ] in
     ignore (note q : Tensor.t);
     window_ref := Some v;
     if fail then ignore (note (TDSL.O.( + ) s (leaf "ck_b" (m + 1))) : Tensor.t)
   with
  | () -> ()
  | exception _ -> ());
  (!made, !window_ref)

(* The valid suffix: closes everything; the dims of every held tensor and of its result. *)
let suffix pre ~j =
  match
    let c = leaf "ck_c" j in
    let out = TDSL.O.( + ) pre.y c in
    List.map (pre.held @ [ out ]) ~f:(fun t -> Shape.to_dims t.Tensor.shape)
  with
  | dims -> Ok dims
  | exception exn -> Error (Exn.to_string exn)

type leg = {
  restored : bool;  (** The held tensors look as they did before the checkpoint. *)
  stale : bool;  (** Every tensor the window created is refused, naming the rewind. *)
  seed_live : bool;  (** The cached random seed is usable. *)
  ref_unsolved : bool;  (** The capture reference made before the checkpoint takes a new size. *)
  no_role : bool;  (** The symbol holds no role claim. *)
  window_ref_refused : bool;  (** The capture reference the window made is refused. *)
  dims : (int array list, string) Result.t;
  differences : string list;
}

let rewound_leg ~seed ~fail ?omit () =
  Tensor.unsafe_reinitialize ();
  let pre = build_prefix (Random.State.make [| seed |]) in
  let rng = Random.State.make [| seed; 903 |] in
  let m = 3 + Random.State.int rng 2 and j = 5 + Random.State.int rng 2 in
  let before = List.map pre.held ~f:see in
  let ck = Tensor.checkpoint ~label:"ck_soak" () in
  let made, window_ref = detour pre ~m ~fail in
  (match omit with
  | None -> Tensor.rewind ck
  | Some omit -> Tensor.For_testing.rewind_omitting ~omit ck);
  let differences = Tensor.For_testing.checkpoint_differences ck in
  let restored =
    (not (List.is_empty before)) && List.for_all2_exn before (List.map pre.held ~f:see) ~f:same_seen
  in
  let stale =
    (not (List.is_empty made))
    && List.for_all made ~f:(fun t ->
        match outcome (fun () -> Tensor.check_not_discarded ~what:"ck" t) with
        | Error msg -> String.is_substring msg ~substring:discarded
        | Ok () -> false)
  in
  let seed_live =
    Result.is_ok
      (outcome (fun () -> Tensor.check_not_discarded ~what:"ck" (Tensor.get_random_seed ())))
  in
  let ref_unsolved = Result.is_ok (outcome (fun () -> Shape.set_dim pre.r 7)) in
  let no_role = not (Row.is_used_as_extent pre.sym) in
  let window_ref_refused =
    match window_ref with
    | None -> false
    | Some v -> refused_with (fun () -> Shape.set_dim v 3) ~substrings:[ "capture it afresh" ]
  in
  {
    restored;
    stale;
    seed_live;
    ref_unsolved;
    no_role;
    window_ref_refused;
    dims = suffix pre ~j;
    differences;
  }

let control_dims ~seed =
  Tensor.unsafe_reinitialize ();
  let pre = build_prefix (Random.State.make [| seed |]) in
  let rng = Random.State.make [| seed; 903 |] in
  let _m = 3 + Random.State.int rng 2 and j = 5 + Random.State.int rng 2 in
  ignore (Tensor.get_random_seed () : Tensor.t);
  Shape.set_dim pre.r 7;
  suffix pre ~j

let same_dims a b =
  match (a, b) with Ok a, Ok b -> List.equal (Array.equal Int.equal) a b | _ -> false

let leg_holds ~control leg =
  leg.restored && leg.stale && leg.seed_live && leg.ref_unsolved && leg.no_role
  && leg.window_ref_refused && same_dims leg.dims control

(* {2 The projections a failed closure published} *)

(* gh-ocannl-1827: the oracle of the "steps" restore category. *)

(* A product-accumulating window: its neutral element is 1, finite, so unlike the max family
   (gh-504) its padded window demands margins on the operand, committed to 1. *)
let prod_window spec t1 t2 =
  let%cd op_asn ~t ~t1 ~t2 ~projections = v =:* v1 + v2 in
  let grad_asn ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Ir.Assignments.empty_comp in
  Tensor.binop ~op_label:"ck_prod_window"
    ~compose_op:(Shape.Einsum (spec, []))
    ~op_asn ~grad_asn ~grad_spec:Tensor.Prohibit_grad t1 t2 ()

(* The tensor's own accumulation: its forward code ends with it, after its operands' code. *)
let accum_projections (t : Tensor.t) =
  Ir.Assignments.fold_leaves t.forward.asgns ~init:None ~f:(fun found leaf ->
      match leaf with Ir.Assignments.Accum_op { projections; _ } -> Some projections | _ -> found)

type steps_leg = {
  closure_failed : bool;  (** The window's closure failed on the older step's neutral. *)
  published : bool;
      (** The checkpoint's own observer saw projections published before it failed. *)
  proj_dims : (int array, string) Result.t;  (** The dims the newer step's projections iterate. *)
  operand_proj_dims : (int array, string) Result.t;
      (** The dims a reduction over it iterates on its operand: its result does not change size. *)
  shape_dims : (int array, string) Result.t;  (** The dims its shape closed to. *)
}

(* A closure derives projections newest step first, and fails partway: the window's product window
   commits the margins of [x] to 1, the step of [a] -- made before the checkpoint -- gets its
   projections, over the size the window gave its captured axis, and then the older sum window over
   [x], demanding margins of 0, fails. With [detour = false], the control: no window. *)
let window_size = 5

let steps_leg ?omit ~detour ~size () =
  Tensor.unsafe_reinitialize ();
  let x = leaf "ck_sx" 6 in
  let b = TDSL.einsum "o=+k; k => o" x (leaf "ck_sk" 3) () in
  let%op wq = { wq = 0.5 } in
  let%op a = wq ++ "u => u" [ "u" ] in
  let%op reduced = a ++ "i => 0" in
  let ck = Tensor.checkpoint ~label:"ck_steps" () in
  let closure_failed, published =
    if not detour then (false, false)
    else (
      Shape.set_dim u window_size;
      let c = prod_window "o=+k; k => o" x (leaf "ck_sk2" 3) in
      let closure_failed =
        refused_with
          (fun () -> Shape.to_dims c.shape)
          ~substrings:[ "Conflicting padding neutral elements" ]
      in
      let published =
        List.mem
          (Tensor.For_testing.checkpoint_differences ck)
          "an update step's projections" ~equal:String.equal
      in
      (match omit with
      | None -> Tensor.rewind ck
      | Some omit -> Tensor.For_testing.rewind_omitting ~omit ck);
      (closure_failed, published))
  in
  let read f = match f () with dims -> Ok dims | exception exn -> Error (Exn.to_string exn) in
  let shape_dims =
    read (fun () ->
        Shape.set_dim u size;
        ignore (Shape.to_dims b.shape : int array);
        Shape.to_dims a.shape)
  in
  let read_projections t ~f =
    read (fun () ->
        match accum_projections t with
        | Some projections -> f (Lazy.force projections)
        | None -> failwith "no accumulation in the forward code")
  in
  let proj_dims = read_projections a ~f:(fun proj -> proj.Ir.Indexing.lhs_dims) in
  let operand_proj_dims = read_projections reduced ~f:(fun proj -> proj.rhs_dims.(0)) in
  { closure_failed; published; proj_dims; operand_proj_dims; shape_dims }

let dims_to_string = function
  | Ok dims -> String.concat ~sep:" " (List.map (Array.to_list dims) ~f:Int.to_string)
  | Error msg -> "ERROR " ^ msg

let same_dims1 a b = match (a, b) with Ok a, Ok b -> Array.equal Int.equal a b | _ -> false

let steps () =
  let control = steps_leg ~detour:false ~size:3 () in
  Stdio.printf "steps, control: the step's projections iterate %s, its shape closes to %s\n"
    (dims_to_string control.proj_dims)
    (dims_to_string control.shape_dims);
  let l = steps_leg ~detour:true ~size:3 () in
  p "steps: the window's closure fails on the older step's padding neutral" l.closure_failed;
  p "steps: the failed closure had published projections of a step older than the checkpoint"
    l.published;
  p "steps, a failed closure rewound: the step's projections iterate the control's dims"
    (same_dims1 l.proj_dims control.proj_dims);
  p "steps, a failed closure rewound: a reduction's projections iterate the control's dims"
    (same_dims1 l.operand_proj_dims control.operand_proj_dims);
  p "steps, a failed closure rewound: its shape closes to the control's dims"
    (same_dims1 l.shape_dims control.shape_dims)

(* The "steps" category restores only projections, which a window cannot write without a closure:
   the soak's constructions never reach it (gh-ocannl-1827). Its oracle is the projections' own
   dims; omitted, the step keeps the projections the failed closure derived, over the window's size,
   since a closure derives none for a step that has them, and reading them is refused, naming both
   sizes (gh-ocannl-1970). *)
let steps_negative_control ~omit =
  let size = 3 in
  let control = steps_leg ~detour:false ~size () in
  let l = steps_leg ~omit ~detour:true ~size () in
  Stdio.eprintf
    "(not part of the golden) omitting %S: projections over %s, on the reduction's operand %s, \
     shape %s\n\
     %!"
    omit (dims_to_string l.proj_dims)
    (dims_to_string l.operand_proj_dims)
    (dims_to_string l.shape_dims);
  p
    (Printf.sprintf "negative control: omitting %S fails an independent oracle" omit)
    (l.closure_failed && not (same_dims1 l.proj_dims control.proj_dims));
  let refused_over ~what = function
    | Ok _ -> false
    | Error msg ->
        let says substring = String.is_substring msg ~substring in
        says "are stale"
        && says (Printf.sprintf "dims (%d) of %s" window_size what)
        && says (Printf.sprintf "closed to dims (%d)" size)
  in
  p
    (Printf.sprintf
       "negative control: omitting %S leaves projections over the window's size, which are refused"
       omit)
    (refused_over ~what:"its result" l.proj_dims && same_dims1 l.shape_dims control.shape_dims);
  p
    (Printf.sprintf
       "negative control: omitting %S leaves a reduction's operand projections, which are refused"
       omit)
    (refused_over ~what:"operand 0" l.operand_proj_dims)

let soak () =
  let seeds = List.init 24 ~f:(fun i -> i + 1) in
  let controls = List.map seeds ~f:(fun seed -> (seed, control_dims ~seed)) in
  p_all "control: every prefix closes with the suffix" controls ~f:(fun (_, d) -> Result.is_ok d);
  List.iter [ true; false ] ~f:(fun fail ->
      let what = if fail then "a failing construction" else "a succeeding construction" in
      let legs =
        List.map controls ~f:(fun (seed, control) -> (rewound_leg ~seed ~fail (), control))
      in
      p_all (Printf.sprintf "soak, %s rewound: the held tensors look as before the checkpoint" what)
        legs ~f:(fun (l, _) -> l.restored);
      p_all
        (Printf.sprintf "soak, %s rewound: the window's tensors are refused, naming the rewind" what)
        legs ~f:(fun (l, _) -> l.stale);
      p_all (Printf.sprintf "soak, %s rewound: the seed cache holds no discarded seed" what) legs
        ~f:(fun (l, _) -> l.seed_live);
      p_all
        (Printf.sprintf "soak, %s rewound: the reference made before the checkpoint is unsolved"
           what) legs ~f:(fun (l, _) -> l.ref_unsolved);
      p_all (Printf.sprintf "soak, %s rewound: the symbol holds no role claim" what) legs
        ~f:(fun (l, _) -> l.no_role);
      p_all (Printf.sprintf "soak, %s rewound: the reference the window made is refused" what) legs
        ~f:(fun (l, _) -> l.window_ref_refused);
      p_all (Printf.sprintf "soak, %s rewound: the valid suffix closes to the control's dims" what)
        legs ~f:(fun (l, control) -> same_dims l.dims control);
      p_all
        (Printf.sprintf "soak, %s rewound: the checkpoint's own observer sees no difference" what)
        legs ~f:(fun (l, _) -> List.is_empty l.differences));
  (* Negative controls: each restore category omitted in turn must fail an independent oracle: on
     some seed here, or for "steps" -- which no construction of the soak writes -- in the
     failed-closure scenario of [steps_negative_control]. *)
  List.iter Tensor.For_testing.restore_categories ~f:(fun omit ->
      if String.equal omit "steps" then steps_negative_control ~omit
      else
        p_exists (Printf.sprintf "negative control: omitting %S fails an independent oracle" omit)
          controls ~f:(fun (seed, control) ->
            not (leg_holds ~control (rewound_leg ~seed ~fail:true ~omit ()))))

(* {2 Lifecycle} *)

let lifecycle () =
  Tensor.unsafe_reinitialize ();
  let pre = build_prefix (Random.State.make [| 7 |]) in
  (* Nested checkpoints: rewinding to the outer abandons the inner, and the outer can be rewound to
     again. *)
  let outer = Tensor.checkpoint ~label:"ck_outer" () in
  let%op t1 = pre.y *. 3. in
  let inner = Tensor.checkpoint ~label:"ck_inner" () in
  let%op _t2 = t1 *. 3. in
  Tensor.rewind outer;
  p "nested: the window's tensor is discarded by the outer rewind"
    (refused_with (fun () -> NTDSL.O.relu t1) ~substrings:[ discarded; "ck_outer" ]);
  p "nested: the inner checkpoint is abandoned by the outer rewind"
    (refused_with (fun () -> Tensor.rewind inner) ~substrings:[ cannot; "ck_inner"; "abandoned" ]);
  let%op _t3 = pre.y *. 5. in
  Tensor.rewind outer;
  p_empty "repeated: the outer checkpoint can be rewound to again"
    ~over:Tensor.For_testing.restore_categories
    (Tensor.For_testing.checkpoint_differences outer);
  (* A closure commits: it ends the checkpoints taken before it. *)
  let ck = Tensor.checkpoint ~label:"ck_closed" () in
  p "the suffix closes inference" (Result.is_ok (suffix pre ~j:2));
  p "a closure since the checkpoint refuses the rewind"
    (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ cannot; "inference closed" ]);
  (* The closure's checkpoint is always available, and ends the explicit ones. *)
  let x2 = leaf "ck_x2" 2 in
  let ck = Tensor.checkpoint ~label:"ck_before_coarse" () in
  Tensor.rewind_to_last_closure ();
  p "rewind_to_last_closure abandons the explicit checkpoints"
    (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ cannot; "abandoned" ]);
  p "and discards the open work, as before"
    (refused_with (fun () -> NTDSL.O.relu x2) ~substrings:[ discarded ]);
  (* A tainted session takes no checkpoint; a rewind clears the taint. *)
  let x3 = leaf "ck_x3" 2 in
  let ck = Tensor.checkpoint ~label:"ck_taint" () in
  let failure = outcome (fun () -> TDSL.O.( + ) x3 (leaf "ck_x4" 3)) in
  p "a failing construction fails" (Result.is_error failure);
  p "a tainted session takes no checkpoint"
    (refused_with (fun () -> Tensor.checkpoint ()) ~substrings:[ tainted ]);
  Tensor.rewind ck;
  p "the rewind clears the taint: the open tensor older than the checkpoint is usable"
    (Result.is_ok (outcome (fun () -> NTDSL.O.relu x3)));
  (* A raising committed-state read of a frontier shape memoizes the failure in its lazy: no
     explicit rewind can reset it, the coarse rewind still can. *)
  Tensor.rewind ck;
  ignore (outcome (fun () -> TDSL.O.( + ) x3 (leaf "ck_x5" 3)) : (unit, string) Result.t);
  p "tainted: forcing an open node's dims raises"
    (refused_with (fun () -> Lazy.force x3.value.dims) ~substrings:[ tainted ]);
  p "a failed read of a frontier shape refuses the explicit rewind"
    (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ cannot; "raised after it" ]);
  Tensor.rewind_to_last_closure ();
  (* The same for a projections read of a step pending over closed shapes: a [%cd] assignment to a
     closed tensor, made before the checkpoint. *)
  let cl = leaf "ck_cl" 2 and src = leaf "ck_src" 2 in
  ignore (Shape.to_dims cl.shape : int array);
  ignore (Shape.to_dims src.shape : int array);
  let code = [%cd cl =: src] in
  let ck = Tensor.checkpoint ~label:"ck_pending_step" () in
  ignore
    (outcome (fun () -> TDSL.O.( + ) (leaf "ck_p2" 2) (leaf "ck_p3" 3)) : (unit, string) Result.t);
  let find_accum asgns =
    Ir.Assignments.fold_leaves asgns ~init:None ~f:(fun found leaf ->
        match leaf with
        | Ir.Assignments.Accum_op { projections; _ } -> Option.first_some found (Some projections)
        | _ -> found)
  in
  p "tainted: forcing the pending assignment's projections raises"
    (match find_accum code.Ir.Assignments.asgns with
    | Some projections -> refused_with (fun () -> Lazy.force projections) ~substrings:[ tainted ]
    | None -> false);
  p "a failed projections read of a frontier step refuses the explicit rewind"
    (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ cannot; "committed projections" ]);
  Tensor.rewind_to_last_closure ();
  (* A failed closure forced through a shape the window made reads no frontier shape: the explicit
     rewind stays available, and puts back what the failed closure wrote. *)
  let%op wq = { wq = 0.5 } in
  let%op _wq_u = wq ++ "u => u" [ "u" ] in
  let ck = Tensor.checkpoint ~label:"ck_failed_closure" () in
  let z = TDSL.O.( + ) (leaf "ck_z" 2) wq in
  Shape.set_dim u 5;
  p "a contradictory set_dim makes the closure fail, and taints the session"
    (Result.is_error (outcome (fun () -> Shape.to_dims z.shape))
    && Option.exists (Shape.taint_provenance ()) ~f:(String.is_prefix ~prefix:"finish_inference"));
  Tensor.rewind ck;
  p "after rewinding a failed closure, the reference it solved takes a new size"
    (Result.is_ok (outcome (fun () -> Shape.set_dim u 3)));
  p "and the prefix closes to it"
    (match Shape.to_dims wq.shape with
    | dims -> Array.equal Int.equal dims [| 3 |]
    | exception _ -> false);
  (* A precision that began settling after the checkpoint cannot be unsettled. *)
  let x7 = leaf "ck_x7" 2 in
  let%op y7 = x7 *. 2. in
  let ck = Tensor.checkpoint ~label:"ck_settle" () in
  ignore (Lazy.force y7.value.storage_prec : Ir.Ops.prec);
  p "a precision settled after the checkpoint refuses the rewind"
    (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ cannot; "began settling" ]);
  Tensor.rewind_to_last_closure ();
  (* Mid-construction: no checkpoint, no rewind. *)
  let x8 = leaf "ck_x8" 2 in
  let in_op_asn = ref (Ok ()) in
  let probe ~t:_ ~t1:_ ~projections:_ =
    in_op_asn := outcome (fun () -> Tensor.checkpoint ());
    Ir.Assignments.empty_comp
  in
  let no_grad ~t:_ ~g:_ ~t1:_ ~projections:_ = Ir.Assignments.empty_comp in
  ignore
    (Tensor.unop ~op_label:"ck_probe" ~op_asn:probe ~grad_asn:no_grad
       ~grad_spec:Tensor.Prohibit_grad x8 ()
      : Tensor.t);
  p "a checkpoint inside an op_asn is refused"
    (match !in_op_asn with
    | Error msg ->
        String.is_substring msg ~substring:"Tensor.checkpoint: called from within [op_asn]"
    | Ok () -> false);
  Tensor.rewind_to_last_closure ()

(* {2 The wrappers, and constraints that do not stack} *)

let wrappers () =
  Tensor.unsafe_reinitialize ();
  let pre = build_prefix (Random.State.make [| 11 |]) in
  let before = List.map pre.held ~f:see in
  let looks_as_before () =
    (not (List.is_empty before)) && List.for_all2_exn before (List.map pre.held ~f:see) ~f:same_seen
  in
  (* with_checkpoint: a raise is rewound and re-raised as is. *)
  let raised =
    match
      Tensor.with_checkpoint ~label:"ck_with" (fun () ->
          let s = TDSL.O.( + ) pre.y (leaf "ck_wa" 3) in
          TDSL.O.( + ) s (leaf "ck_wb" 4))
    with
    | (_ : Tensor.t) -> None
    | exception Row.Shape_error (msg, _) -> Some msg
  in
  p "with_checkpoint re-raises the construction's own shape error" (Option.is_some raised);
  p "with_checkpoint rewound the failure: the held tensors look as before" (looks_as_before ());
  p "with_checkpoint rewound the taint" (Result.is_ok (outcome (fun () -> NTDSL.O.relu pre.x)));
  Tensor.rewind_to_last_closure ();
  let pre = build_prefix (Random.State.make [| 11 |]) in
  let kept = Tensor.with_checkpoint (fun () -> TDSL.O.( + ) pre.y (leaf "ck_wk" 3)) in
  p "with_checkpoint keeps the work of a normal return"
    (Result.is_ok (outcome (fun () -> Tensor.check_not_discarded ~what:"ck" kept)));
  (* with_rewind: #404's trace -- read the open state inside, leave nothing behind. *)
  let ck = Tensor.checkpoint () in
  let traced_rows =
    Tensor.with_rewind ~label:"ck_trace" (fun () ->
        let tracer = Tensor.term ~label:[ "ck_tracer" ] ~grad_spec:Tensor.Prohibit_grad () in
        let%op applied = (kept *. tracer) + tracer in
        ( applied,
          Sexp.to_string (Row.sexp_of_t applied.shape.output),
          Tensor.For_testing.checkpoint_sizes (Tensor.checkpoint ()) ))
  in
  let applied, rows, sizes_inside = traced_rows in
  p "with_rewind returns what the trace read" (not (String.is_empty rows));
  p "with_rewind: the trace's tensors are discarded"
    (refused_with (fun () -> NTDSL.O.relu applied) ~substrings:[ discarded; "ck_trace" ]);
  p_empty "with_rewind leaves no trace in the session" ~over:Tensor.For_testing.restore_categories
    (Tensor.For_testing.checkpoint_differences ck);
  p "with_rewind: the trace had grown the open frontier"
    (List.exists2_exn (Tensor.For_testing.checkpoint_sizes ck) sizes_inside
       ~f:(fun (_, before) (_, inside) -> inside > before));
  p "with_rewind over a trace that closes inference raises, loudly"
    (refused_with
       (fun () -> Tensor.with_rewind (fun () -> ignore (Shape.to_dims kept.shape : int array)))
       ~substrings:[ cannot; "inference closed" ]);
  (* A re-run after a rewind does not stack constraints. *)
  Tensor.unsafe_reinitialize ();
  let pre = build_prefix (Random.State.make [| 13 |]) in
  let ck = Tensor.checkpoint () in
  let construct () =
    let%op z = (pre.y *. 2.) + pre.y in
    ignore (z : Tensor.t);
    Tensor.For_testing.checkpoint_sizes (Tensor.checkpoint ())
  in
  let first = construct () in
  Tensor.rewind ck;
  let second = construct () in
  let third = construct () in
  p "a re-run after a rewind has the first run's frontier sizes"
    (List.equal Poly.equal first second);
  p "negative control: a re-run without the rewind grows them"
    (not (List.equal Poly.equal second third))

(* {2 External handouts} *)

let handouts () =
  Tensor.unsafe_reinitialize ();
  (* A parameter over an open differentiable initializer shares its value node but owns a fresh
     gradient: handing out the parameter's backprop does not hand out the initializer's. *)
  let v = leaf ~grad_spec:Tensor.Require_grad "ck_hv" 2 in
  let init_tensor = TDSL.O.relu v in
  p "the initializer is a backprop root" (Tensor.is_bprop_root init_tensor);
  let ck = Tensor.checkpoint () in
  let reuse ?label:_ ?top_down_prec:_ ?batch_dims:_ ?batch_axes:_ ?input_dims:_ ?output_dims:_
      ?input_axes:_ ?output_axes:_ ?deduced:_ () =
    init_tensor
  in
  let parameter = Tensor.param ~t:reuse "ck_hp" () in
  ignore (Tensor.consume_backprop_code parameter : Ir.Assignments.comp);
  Tensor.rewind ck;
  p "rewound: the initializer's backprop root is back, the parameter's handout notwithstanding"
    (Tensor.is_bprop_root init_tensor);
  (* A handout inside [with_unchanged_roots] after a closure there is undone with the maps. *)
  Tensor.rewind_to_last_closure ();
  let y = leaf "ck_hy" 2 in
  p "a fresh leaf is a forward root" (Tensor.is_fwd_root y);
  Tensor.with_unchanged_roots ~f:(fun () ->
      ignore (Shape.to_dims y.shape : int array);
      ignore (Tensor.consume_forward_code y : Ir.Assignments.comp));
  Tensor.rewind_to_last_closure ();
  p "with_unchanged_roots undid a handout made after a closure inside it" (Tensor.is_fwd_root y)

(* {2 Against a fresh process: compiled, run} *)

let fixed_suffix pre =
  let c = leaf "ck_fc" 2 in
  TDSL.O.( + ) pre.y c

let fixed_prefix () =
  let x = leaf "ck_fx" 2 in
  let%op w = { w = 0.5 } in
  let%op y = w * x in
  {
    x;
    y;
    r = Shape.get_variable_ref "ck_fr";
    r_sym = Shape.get_variable_ref "ck_fr_sym";
    sym = Ir.Indexing.make_static_symbol ~static_range:4 (Ir.Indexing.get_symbol ());
    held = [ x; w; y ];
  }

let values_of out =
  let ctx = Train.forward_once (Context.auto ()) out in
  Context.get_values ctx out.Tensor.value

let hex xs = String.concat ~sep:" " (Array.to_list (Array.map xs ~f:(Printf.sprintf "%h")))

let fresh_leg () =
  let pre = fixed_prefix () in
  Stdio.printf "RESULT %s\n" (hex (values_of (fixed_suffix pre)))

let fresh_comparison () =
  Tensor.unsafe_reinitialize ();
  let pre = fixed_prefix () in
  let ck = Tensor.checkpoint ~label:"ck_fixed" () in
  ignore (detour pre ~m:3 ~fail:true : Tensor.t list * Shape.delayed_var_ref option);
  Tensor.rewind ck;
  let got = values_of (fixed_suffix pre) in
  let child = Fresh_process.run [ "fresh" ] in
  let status, child_out, _ = child in
  let ok = match status with Unix.WEXITED 0 -> true | _ -> false in
  p "the fresh-process reference ran" ok;
  if not ok then Fresh_process.report ~label:"fresh leg" child;
  let want =
    List.find_map (String.split_lines child_out) ~f:(String.chop_prefix ~prefix:"RESULT ")
    |> Option.value ~default:""
  in
  p "compiled and run after the rewind: the values equal the fresh process's"
    (String.equal (hex got) want && not (String.is_empty want));
  p_all2 "and they are the closed form 0.5 (1 + 2) + [1; 2]" got [| 2.5; 3.5 |] ~f:Float.equal

let () =
  match Array.to_list (Sys.get_argv ()) with
  | [ _; "fresh" ] -> fresh_leg ()
  | _ ->
      soak ();
      steps ();
      lifecycle ();
      wrappers ();
      handouts ();
      fresh_comparison ()
