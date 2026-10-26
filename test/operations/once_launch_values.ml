(* gh-ocannl-1629: a one-off run ([Train.forward_once], over [Train.run_once]) refuses a RANGED
   launch parameter the caller gave no value, naming it, and runs at the values the caller passes
   through [~launch_values]; an unranged parameter it is given no value for still runs at 0.

   Before, [run_once] bound every launch parameter to 0 wholesale, so a one-off run of a tensor
   sliced by a batch index ran slice 0 only, silently -- the "forgot to assign" case gh-ocannl-940
   set out to refuse.

   The probe slices a three-row literal whose rows are distinct (10, 20, 30), plus 0.5 (a slice is a
   view, read through a computed tensor), so the value read back names the slice that ran. The
   refusal comes before the tensor's forward code is consumed, which the retry on the SAME tensor
   witnesses. A batch-slice index declared with no [static_range] is ranged too (its range is the
   sliced axis): refused alike. The unranged parameter is embedded as a value ([!@k]), so what runs
   at 0 is read back as 0, and a value given for it is read back. A symbolic extent is ranged as
   well, and its full range is a legal first value: the reduction over it sums that many ones. *)

open Base
open Stdio
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
open Verdict.Claims

let refusal f =
  match f () with
  | (_ : Context.t) -> None
  | exception Utils.User_error msg ->
      eprintf "refused: %s (not part of the golden)\n%!" msg;
      Some msg

let names msg (s : Ir.Indexing.static_symbol) =
  String.is_substring msg ~substring:(Ir.Indexing.symbol_ident s.Ir.Indexing.static_symbol ^ " ")

let names_slice msg s = names msg s && String.is_substring msg ~substring:"(batch-slice index,"

let rows () =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| 3; 1 |] in
  List.iteri [ 10.; 20.; 30. ] ~f:(fun i v -> Genarray.set ga [| i; 0 |] v);
  TDSL.rebatch ~l:"rows" (Ir.Ndarray.as_array Ir.Ops.Single ga) ()

let read ctx (t : Tensor.t) = (Context.get_values ctx t.Tensor.value).(0)

let () =
  let ctx = Context.auto () in
  (* A declared-range batch-slice index. *)
  let i, bindings = IDX.get_static_symbol ~static_range:3 IDX.empty in
  let x = rows () in
  let%op slice = (x @| i) + 0.5 in
  let unvalued = refusal (fun () -> Train.forward_once ~bindings ctx slice) in
  p "a one-off run with a ranged batch index given no value is refused" (Option.is_some unvalued);
  p "that refusal names the batch index, as one"
    (Option.exists unvalued ~f:(fun msg -> names_slice msg i));
  let ctx = Train.forward_once ~bindings ~launch_values:[ (i, 2) ] ctx slice in
  p "a retry on the same tensor runs at the value given: slice 2" Float.(read ctx slice = 30.5);
  (* A batch-slice index with no declared range. *)
  let j, bindings = IDX.get_static_symbol IDX.empty in
  let x = rows () in
  let%op slice = (x @| j) + 0.5 in
  let unvalued = refusal (fun () -> Train.forward_once ~bindings ctx slice) in
  p "a batch index with an inferred range, given no value, is refused too"
    (Option.exists unvalued ~f:(fun msg -> names_slice msg j));
  let ctx = Train.forward_once ~bindings ~launch_values:[ (j, 1) ] ctx slice in
  p "and runs at the value given: slice 1" Float.(read ctx slice = 20.5);
  (* An unranged parameter: 0 when given no value, the value when given one. *)
  let k, bindings = IDX.get_static_symbol IDX.empty in
  let%op embedded = !@k + 0.5 in
  let ctx = Train.forward_once ~bindings ctx embedded in
  p "an unranged parameter given no value runs at 0" Float.(read ctx embedded = 0.5);
  let%op embedded = !@k + 0.5 in
  let ctx = Train.forward_once ~bindings ~launch_values:[ (k, 5) ] ctx embedded in
  p "an unranged parameter runs at the value given" Float.(read ctx embedded = 5.5);
  (* A symbolic extent, first run at its full declared range: a legal extent value, which only the
     extent role -- still pending until lowering commits it -- tells from an out-of-range index. *)
  let extent_sum e =
    let%op x = { x = 0.5 } in
    let%op y = (2. *. x) ++ "s=>s" [ "s" ] in
    Shape.set_sym_dim s e;
    [%op y ++ "... => 0"]
  in
  let e, e_bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
  let total = extent_sum e in
  let unvalued = refusal (fun () -> Train.forward_once ~bindings:e_bindings ctx total) in
  p "a symbolic extent given no value is refused, naming it"
    (Option.exists unvalued ~f:(fun msg ->
         names msg e && String.is_substring msg ~substring:"(symbolic extent,"));
  let ctx = Train.forward_once ~bindings:e_bindings ~launch_values:[ (e, 4) ] ctx total in
  p "a symbolic extent runs at its full range when given it: the sum of 4 ones"
    Float.(read ctx total = 4.);
  let e2, e_bindings = IDX.get_static_symbol ~static_range:4 IDX.empty in
  let total = extent_sum e2 in
  let ctx = Train.forward_once ~bindings:e_bindings ~launch_values:[ (e2, 2) ] ctx total in
  p "and at a smaller value given: the sum of 2 ones" Float.(read ctx total = 2.);
  (* Values that cannot be meant. *)
  let%op embedded = !@k + 0.5 in
  let stray =
    refusal (fun () -> Train.forward_once ~bindings ~launch_values:[ (i, 0) ] ctx embedded)
  in
  p "a value for a symbol the bindings do not bind is refused, naming it"
    (Option.exists stray ~f:(fun msg -> names msg i));
  let twice =
    refusal (fun () -> Train.forward_once ~bindings ~launch_values:[ (k, 1); (k, 2) ] ctx embedded)
  in
  p "a parameter given a value twice is refused, naming it"
    (Option.exists twice ~f:(fun msg -> names msg k))
