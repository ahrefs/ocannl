(* gh-ocannl-1789: what a rewind records of the work it discarded -- the nodes, the capture
   references and the update steps the window made, each refused afterwards with the rewind's
   provenance -- is bounded by what is still reachable, not by how many rewinds the session ran.
   Shape-scheme traces rewind once per schemed closure, and a REPL or notebook re-running cells
   rewinds explicitly, so a long session runs thousands of them.

   The claim is a count, read after full major collections:
   [Tensor.For_testing.discarded_registry_sizes] reports, per kind, the reachable discarded objects
   the session can still name (on the pre-gh-1789 tables, their entries). Three kinds of rewind run
   many times (explicit [Tensor.rewind], [Tensor.rewind_to_last_closure], a schemed closure's
   trace), each window dropping what it made; every count must come back to its value after the
   first round. The control holds one window's stale handles: each count must see them (so the flat
   counts are not those of a counter that reads nothing), and each handle must stay refused -- after
   the rewinds, and after a closure -- naming the rewind that discarded it. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let outcome f =
  match f () with
  | (_ : 'a) -> Ok ()
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg
  | exception Utils.User_error msg -> Error msg
  | exception exn -> Error (Exn.to_string exn)

let refused_with f ~substrings =
  match outcome f with
  | Ok () -> false
  | Error msg ->
      Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
      (not (List.is_empty substrings))
      && List.for_all substrings ~f:(fun substring -> String.is_substring msg ~substring)

let () = Tensor.For_testing.set_op_shape_schemes (Some `Check)

let leaf name n =
  Tensor.term_init
    (Array.init n ~f:(fun i -> Float.of_int (i + 1)))
    ~label:[ name ] ~grad_spec:Tensor.Prohibit_grad ~output_dims:[ n ] ()

(* A schemed closure: building it traces its scheme in a rewound window. *)
let%op layer () x = ({ w; o = [ 4 ] } * x) + 1.0

(* One window's stale handles: a tensor, the capture reference an einsum bound, and an update step
   made over the window's shape without being registered. *)
type stale = { tensor : Tensor.t; capture : Shape.delayed_var_ref; step : Shape.update_step }

let window_work () =
  let x = leaf "bounded_x" 3 in
  let%op s = x *. 2. in
  let%op q = s ++ "...|v => ...|v" [ "v" ] in
  let step =
    Shape.make_update_step ~shape:q.shape
      ~logic:(Shape.Terminal { is_param = false; logic = Fetch (Constant 0.) })
      ~neutral_elem:None
  in
  { tensor = q; capture = v; step }

(* Each round runs in a function of its own that keeps nothing it made. *)
let[@inline never] explicit_round () =
  let ck = Tensor.checkpoint ~label:"bounded" () in
  ignore (window_work () : stale);
  Tensor.rewind ck

let[@inline never] closure_round () =
  ignore (window_work () : stale);
  Tensor.rewind_to_last_closure ()

let[@inline never] trace_round () = ignore (layer () : Tensor.t -> Tensor.t)

let rounds =
  [
    ("explicit rewinds", explicit_round, 1000);
    ("rewinds to the last closure", closure_round, 200);
    ("schemed closures' traces", trace_round, 200);
  ]

let sizes () =
  Gc_witness.settle ();
  Tensor.For_testing.discarded_registry_sizes ()

let show sizes =
  String.concat ~sep:", " (List.map sizes ~f:(fun (kind, n) -> kind ^ " " ^ Int.to_string n))

let[@inline never] held_window () =
  let ck = Tensor.checkpoint ~label:"the held window" () in
  let held = window_work () in
  Tensor.rewind ck;
  held

let refused_stale held ~substring =
  [
    ( "the tensor",
      refused_with
        (fun () -> Tensor.check_not_discarded ~what:"probe" held.tensor)
        ~substrings:[ "discarded by the rewind after"; substring ] );
    ( "the capture reference",
      refused_with
        (fun () -> Shape.set_dim held.capture 3)
        ~substrings:[ "capture it afresh"; substring ] );
    ( "the update step",
      refused_with
        (fun () -> ignore (Shape.get_projections held.step : Ir.Indexing.projections))
        ~substrings:[ "re-run its definition"; substring ] );
  ]

let () =
  Verdict.case "many rewinds leave the discard registries at the live window's size" (fun () ->
      Tensor.unsafe_reinitialize ();
      List.iter rounds ~f:(fun (_, round, _) -> round ());
      let first = sizes () in
      List.iter rounds ~f:(fun (_, round, n) ->
          for _ = 1 to n do
            round ()
          done);
      let after = sizes () in
      Stdio.eprintf "(not part of the golden) after the first round: %s; after %d more: %s\n%!"
        (show first)
        (List.sum (module Int) rounds ~f:(fun (_, _, n) -> n))
        (show after);
      p_all ~min:3 "every registry holds as many records as after the first round"
        (List.zip_exn first after) ~f:(fun ((_, n0), (_, n1)) -> n1 = n0);
      (* The control: one window's stale handles, held. *)
      let held = held_window () in
      let holding = sizes () in
      Stdio.eprintf "(not part of the golden) holding one window's handles: %s\n%!" (show holding);
      p_all ~min:3 "holding one window's stale handles, every registry counts them"
        (List.zip_exn after holding) ~f:(fun ((_, n0), (_, n1)) -> n1 > n0);
      p_all ~min:3 "each held handle is refused, naming its rewind"
        (refused_stale held ~substring:"the held window")
        ~f:snd;
      List.iter rounds ~f:(fun (_, round, n) ->
          for _ = 1 to n do
            round ()
          done);
      p_all ~min:3 "and still refused, naming its rewind, after as many rewinds again"
        (refused_stale held ~substring:"the held window")
        ~f:snd;
      (* A closure: inference completes on a fresh tensor, and a checkpoint from before is
         refused. *)
      let ck = Tensor.checkpoint ~label:"before the closure" () in
      ignore (Shape.to_dims (leaf "bounded_closed" 2).shape : int array);
      p "control: the closure happened"
        (refused_with (fun () -> Tensor.rewind ck) ~substrings:[ "inference closed" ]);
      p_all ~min:3 "and still refused, naming its rewind, after a closure"
        (refused_stale held ~substring:"the held window")
        ~f:snd)
