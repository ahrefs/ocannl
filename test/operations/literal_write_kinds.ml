(* One literal predicate for [Context]'s host and routine refusals (gh-ocannl-2104 follow-up).

   [Context.check_no_literal_write] (a routine write) asked [Tn.known_host_constant], through slice
   views; [Context.check_host_write_lasts] (a host write) asked the bare [host_constant] marker and
   skipped views. Both now ask [Context.literal_of]. What that changed, recorded on the base:

   - A host write to a [TDSL.number] before any routine carries its fetch was accepted and silently
   lost: [set_values] of 9 into the number 3, then the forward of [3 *. 2], read 6. It is now
   refused as a literal (gh-ocannl-2021), like a [TDSL.ndarray] is. - A host write to an [\@|] view
   of a literal was refused, but by [upload]'s "an @| slice view; write its parent instead"
   ([set_value]: [to_host]'s "read its parent instead"), a remedy that is refused in its turn. It is
   now refused as a view of the literal, naming it.

   [Context.check_copy_lasts]' literal exemption keeps a narrower question, which no view and no
   number answers: whether an unrun source's buffer holds the registered values. The copy legs pin
   that a literal's copy from such a source stays accepted and that a number's stays refused.

   Every value is a small integer and every operation pointwise, so readbacks are exact. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX

let double x =
  let%op doubled = x *. 2 in
  doubled

let values ctx t = Context.get_values ctx t.Tensor.value
let count n ~from = Array.init n ~f:(fun i -> Float.of_int (from + i))

(* The literal refusal: the write's entry point, the node, the literal it lands in, the issue and
   the remedy. *)
let refused_as_literal name ~fn ~node ~literal f =
  let msg = LC.refusal name f in
  List.iter
    [ "Context." ^ fn; node; literal; "gh-ocannl-2021"; "Operation.init" ]
    ~f:(fun substring -> p (name ^ ": the refusal names " ^ substring) (LC.has msg substring))

let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
      let literal = Printf.sprintf "lit_parent_%d" n in
      let registered = count (2 * n) ~from:1 in
      let parent =
        TDSL.ndarray registered ~label:[ literal ] ~batch_dims:[ 2 ] ~output_dims:[ n ] ()
      in
      let%op view = parent @| batch_n in
      let node = "view" in
      let writes stage ctx =
        let tn = view.Tensor.value in
        let leg fn = Printf.sprintf "%d, %s: %s to a view of a literal" n stage fn in
        refused_as_literal (leg "set_values") ~fn:"set_values" ~node ~literal (fun () ->
            Context.set_values ctx tn (Array.create ~len:n 99.));
        refused_as_literal (leg "set_value") ~fn:"set_value" ~node ~literal (fun () ->
            Context.set_value ctx tn [| 0 |] 99.);
        refused_as_literal (leg "from_host") ~fn:"from_host" ~node ~literal (fun () ->
            Context.from_host ctx tn
              (Ir.Ndarray.create_array ~debug:"literal_write_kinds" ~dims:[| n |] ~padding:None
                 Ir.Ops.single))
      in
      writes "before compilation" (Context.auto ());
      let doubled = double view in
      let ctx, routine =
        Context.compile ~name:(Printf.sprintf "view_fwd_%d" n) (Context.auto ())
          (Train.forward doubled) bindings
      in
      writes "compiled" ctx;
      IDX.find_exn routine.Context.bindings batch_n := 1;
      let ctx = Context.run ctx routine in
      LC.exact
        (Printf.sprintf "%d: the literal keeps its registered values" n)
        (values ctx parent) registered;
      LC.exact
        (Printf.sprintf "%d: the forward of the view reads them" n)
        (values ctx doubled)
        (Array.map (count n ~from:(n + 1)) ~f:(fun v -> 2. *. v)))

let () =
  let num = TDSL.number ~label:[ "num_seed" ] 3. in
  let doubled = double num in
  refused_as_literal "a number before compilation" ~fn:"set_values" ~node:"num_seed"
    ~literal:"a literal" (fun () -> Context.set_values (Context.auto ()) num.Tensor.value [| 9. |]);
  let ctx, routine =
    Context.compile ~name:"num_seed_fwd" (Context.auto ()) (Train.forward doubled) IDX.empty
  in
  (* Once a routine carries the number's fetch, the fetch-defined refusal answers first, as for any
     small literal. *)
  let msg =
    LC.refusal "a number once compiled" (fun () -> Context.set_values ctx num.Tensor.value [| 9. |])
  in
  p "a number once compiled: the refusal names gh-ocannl-1959" (LC.has msg "gh-ocannl-1959");
  let ctx = Context.run ctx routine in
  LC.exact "the forward of the number reads its value" (values ctx doubled) [| 6. |];
  (* A fresh context compiles nothing of it: still a literal there. *)
  refused_as_literal "a number in a fresh context" ~fn:"set_values" ~node:"num_seed"
    ~literal:"a literal" (fun () -> Context.set_values (Context.auto ()) num.Tensor.value [| 9. |])

(* A copy into a context whose forward fetches the node, from a source compiled with the same
   forward and not run. *)
let copy_from_unrun ~tag t =
  let compile suffix =
    Context.compile ~name:(tag ^ suffix) (Context.auto ()) (Train.forward t) IDX.empty
  in
  let dst, routine = compile "_dst" in
  let dst = Context.run dst routine in
  let src, _ = compile "_src" in
  (dst, fun () -> Context.copy ~src ~dst t.Tensor.value)

let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      let registered = count n ~from:1 in
      let lit =
        TDSL.ndarray registered ~label:[ Printf.sprintf "lit_root_%d" n ] ~output_dims:[ n ] ()
      in
      let _, copy = copy_from_unrun ~tag:(Printf.sprintf "lit_copy_%d" n) lit in
      Option.iter
        (LC.accepted (Printf.sprintf "%d: a literal's copy from an unrun source" n) copy)
        ~f:(fun dst ->
          LC.exact
            (Printf.sprintf "%d: the copied literal reads its registered values" n)
            (values dst lit) registered));
  let num = TDSL.number ~label:[ "num_root" ] 3. in
  let dst, copy = copy_from_unrun ~tag:"num_copy" num in
  let msg = LC.refusal "a number's copy from an unrun source" copy in
  p "a number's copy from an unrun source: the refusal names gh-ocannl-2022"
    (LC.has msg "gh-ocannl-2022");
  LC.exact "the number reads as its fetch left it" (values dst num) [| 3. |]

let () =
  let n = 4 in
  let initial = count n ~from:1 in
  let uploaded = count n ~from:11 in
  let init ?mutable_ l =
    NTDSL.init ?mutable_ ~l ~prec:Ir.Ops.single ~o:[ n ] ~f:(fun i -> initial.(i.(0))) ()
  in
  let host =
    Ir.Ndarray.init_array ~debug:"init replacement" Ir.Ops.single ~dims:[| n |] ~padding:None
      ~f:(fun i -> uploaded.(i.(0)))
  in
  let writer x ctx =
    Context.compile
      ~name:("write_" ^ Ir.Tnode.label_name x.Tensor.value)
      ctx
      [%cd x =: x + !.100.]
      IDX.empty
  in
  let x = init "default_init" in
  let ctx = Context.auto () in
  let host_msg =
    LC.refusal "upload to a default init" (fun () -> Context.from_host ctx x.Tensor.value host)
  in
  let comp_msg = LC.refusal "comp writes a default init" (fun () -> writer x ctx) in
  let names_write name msg ~fn ~issue =
    List.iter [ fn; "default_init"; issue; "Operation.init ~mutable_:true" ] ~f:(fun text ->
        p (name ^ ": the refusal names " ^ text) (LC.has msg text))
  in
  names_write "upload to a default init" host_msg ~fn:"Context.from_host" ~issue:"gh-ocannl-2021";
  names_write "comp writes a default init" comp_msg ~fn:"Context.compile" ~issue:"gh-ocannl-2104";
  let y = double x in
  let ctx = Train.forward_once ctx y in
  LC.exact "refused writes leave the default init values intact" (values ctx y)
    (Array.map initial ~f:(( *. ) 2.));
  Context.release ctx;
  let x = init ~mutable_:true "mutable_init" in
  let ctx = Context.auto () in
  let compiled = LC.accepted "a comp can write an opted-out init" (fun () -> writer x ctx) in
  let ctx = Option.value_map compiled ~default:ctx ~f:fst in
  let uploaded_ctx =
    LC.accepted "an opted-out init accepts an upload" (fun () ->
        Context.from_host ctx x.Tensor.value host)
  in
  Option.iter uploaded_ctx ~f:(fun ctx ->
      LC.exact "the opted-out init holds the uploaded values" (values ctx x) uploaded;
      match compiled with
      | None -> Context.release ctx
      | Some (_, routine) ->
          let ctx = Context.run ctx routine in
          LC.exact "the comp reads and updates the opted-out init" (values ctx x)
            (Array.map uploaded ~f:(( +. ) 100.));
          Context.release ctx)

let () =
  let open Ir.Assignments in
  let n = 4 in
  let zeroes = Array.create ~len:n 0. in
  let reset_and_read ?mutable_ name =
    let x =
      NTDSL.init ?mutable_ ~l:name ~prec:Ir.Ops.single ~o:[ n ]
        ~f:(fun i -> Float.of_int (i.(0) + 1))
        ()
    in
    let array = x.Tensor.value in
    let reset = to_comp (Fetch { array; fetch_op = Constant 0.; dims = array.dims }) in
    let y = double x in
    let ctx, routine =
      Context.compile ~name (Context.auto ()) (sequence [ reset; Train.forward y ]) IDX.empty
    in
    let ctx = Context.run ctx routine in
    let got = values ctx y in
    Context.release ctx;
    got
  in
  let msg =
    LC.refusal "a fetch reset of a default init" (fun () ->
        LC.exact "the unexpectedly accepted reset cleared the reader's input"
          (reset_and_read "default_reset") zeroes)
  in
  List.iter
    [ "Context.compile"; "default_reset"; "gh-ocannl-2104"; "Operation.init ~mutable_:true" ]
    ~f:(fun text -> p ("the fetch-write refusal names " ^ text) (LC.has msg text));
  LC.exact "a fetch reset of an opted-out init clears the reader's input"
    (reset_and_read ~mutable_:true "mutable_reset")
    zeroes
