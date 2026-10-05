(* gh-641: a scalar literal can choose host initialization without acquiring Reshape's one-element
   constraint. Opposing limits pin the representation as well as executed values; padded convolution
   pins the escape from the initializer that cannot migrate automatically. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments

let rec fetches tn = function
  | Asgns.Fetch { array; _ } -> Ir.Tnode.equal tn array
  | Seq (a, b) -> fetches tn a || fetches tn b
  | Block_comment (_, a) -> fetches tn a
  | Noop | Accum_op _ | Set_vec_unop _ -> false

let values ctx t expected label =
  Verdict.pass_fail_all2 label (Context.get_values ctx t.Tensor.value) expected ~f:Float.equal

let compile name t ~inspect =
  let comp = Train.forward t in
  let comp = { comp with asgns = Asgns.Block_comment (name, comp.asgns) } in
  Context.compile
    ~lowered_transform:(fun opt ->
      inspect opt;
      [ opt ])
    (Context.auto ()) comp Ir.Indexing.Empty

let scalar limit value =
  let tag = Printf.sprintf "scalar_%d_%d" limit (Int.of_float value) in
  let x = NTDSL.ndarray [| value |] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
  p "literal fetch follows the configured size limit"
    (Bool.equal (fetches x.value x.forward.asgns) (limit >= 1));
  let init = Option.value_exn (Ir.Host_inits.find x.value) in
  p "host initializer waits for shape inference" (not (Lazy.is_val init));
  let varying = NTDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
  let%op y = x + varying in
  let ctx, routine = compile tag y ~inspect:(fun _ -> ()) in
  let ctx = Context.run ctx routine in
  values ctx y
    (Array.init 6 ~f:(fun i -> value +. Float.of_int i))
    "broadcast scalar preserves explicit rows and executed values";
  let varying2 = NTDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
  let%op z = x + varying2 + 8. in
  let ctx2, routine2 = compile (tag ^ "_fresh") z ~inspect:(fun _ -> ()) in
  let ctx2 = Context.run ctx2 routine2 in
  values ctx2 z
    (Array.init 6 ~f:(fun i -> value +. Float.of_int (i + 8)))
    "fresh context uploads the scalar after its forward was consumed";
  Context.release ctx2;
  Context.release ctx

let scalar_root limit =
  let x = NTDSL.ndarray [| 7. |] ~label:[ "root_" ^ Int.to_string limit ] () in
  let ctx = Train.forward_once (Context.auto ()) x in
  p_all "rank-zero scalar root keeps its shape" [ x ] ~f:(fun t ->
      Array.is_empty (Lazy.force t.Tensor.value.Ir.Tnode.dims));
  values ctx x [| 7. |] "rank-zero scalar root initializes without a consumer";
  Context.release ctx

let inferred limit =
  let x =
    Tensor.term_init ~grad_spec:Tensor.Prohibit_grad [| 5. |] ~batch_dims:[] ~input_dims:[] ()
  in
  let%op y = x ++ "i => i" [ "i" ] in
  Shape.set_dim i 4;
  let ctx, routine = compile ("inferred_" ^ Int.to_string limit) y ~inspect:(fun _ -> ()) in
  let ctx = Context.run ctx routine in
  p "host scalar fill accepts dimensions inferred by an einsum"
    (Array.equal Int.equal (Lazy.force x.value.Ir.Tnode.dims) [| 4 |]);
  values ctx y [| 5.; 5.; 5.; 5. |] "inferred scalar initialization executes";
  Context.release ctx

let padded limit value =
  let tag = Printf.sprintf "padded_%d_%d" limit (Int.of_float value) in
  let x = NTDSL.ndarray [| value |] ~output_dims:[ 4 ] () in
  Train.set_materialized x.value;
  let kernel = NTDSL.ndarray [| 1.; 2.; 3. |] ~output_dims:[ 3 ] () in
  let%op conv = x +* "i=+k; k => i" kernel in
  let discriminant = NTDSL.ndarray [| 1.; 2.; 4.; 8. |] ~output_dims:[ 4 ] () in
  let%op y = conv + discriminant in
  let ctx, routine =
    compile tag y ~inspect:(fun opt ->
        p "padded materialized scalar init survives only with the in-kernel limit"
          (Bool.equal (Ll_test.count_set opt x.value > 0) (limit >= 1)))
  in
  p "convolution commits scalar halo padding" (Option.is_some (Ir.Tnode.get_padding x.value));
  let nd = Lazy.force (Option.value_exn (Ir.Host_inits.find x.value)) in
  let pads, neutral = Option.value_exn (Ir.Tnode.get_padding x.value) in
  let left = pads.(0).Ir.Ops.left and right = pads.(0).Ir.Ops.right in
  p "scalar host buffer fills the interior and neutral margins"
    (Float.equal neutral 0.
    && Array.equal Float.equal
         (Ir.Ndarray.retrieve_flat_values nd)
         (Array.init
            (left + 4 + right)
            ~f:(fun i -> if i < left || i >= left + 4 then 0. else value)));
  let ctx = Context.run ctx routine in
  let expected =
    [| 1. +. (5. *. value); 2. +. (6. *. value); 4. +. (6. *. value); 8. +. (3. *. value) |]
  in
  values ctx y expected "padded scalar convolution matches the independent reference";
  let ctx = Context.run ctx routine in
  values ctx y expected "padded scalar values remain correct on a second run";
  Context.release ctx

(* gh-ocannl-1218: the padding-aware fill writes exactly the interior. Uneven margins on all three
   axes discriminate every level of the offset arithmetic (in 2-D a dropped scaling of the outer
   offset goes unnoticed); the second case leaves the innermost axis unpadded, so its runs fold it
   in. The margins keep the neutral value the buffer was created with. *)
let padded_fill_offsets label dims pads =
  let nd =
    Ir.Ndarray.create_array ~debug:"fill_offsets" Ir.Ops.single ~dims ~padding:(Some (pads, -1.))
  in
  Ir.Ndarray.fill_from_float ~padding:pads nd 3.;
  let numel = Array.fold dims ~init:1 ~f:( * ) in
  let inside k =
    (* Decompose the row-major linear index [k], innermost axis first. *)
    let _, inside =
      Array.fold_right (Array.zip_exn dims pads) ~init:(k, true)
        ~f:(fun (d, Ir.Ops.{ left; right }) (rest, inside) ->
          let i = rest % d in
          (rest / d, inside && i >= left && i < d - right))
    in
    inside
  in
  p label
    (Array.equal Float.equal
       (Ir.Ndarray.retrieve_flat_values nd)
       (Array.init numel ~f:(fun k -> if inside k then 3. else -1.)))

(* gh-ocannl-1218: forcing a padded broadcast scalar's host initializer allocates a bounded number
   of OCaml heap words, independent of its cell count: a full-size float temporary alone is [numel]
   words, and a view per contiguous run grows with the run count. 512 is about five times what the
   buffer's own creation costs. Measured while lowering, after shape inference committed the padding
   and before linking forces it. *)
let allocation_witness ~name ~numel (x : Tensor.t) y ~check_layout =
  let measured = ref None in
  let ctx, _routine =
    compile name y ~inspect:(fun _ ->
        if Option.is_none !measured then
          let init = Option.value_exn (Ir.Host_inits.find x.value) in
          let forced_before = Lazy.is_val init in
          let before = Stdlib.Gc.allocated_bytes () in
          let _ : Ir.Ndarray.t = Lazy.force init in
          let words =
            (Stdlib.Gc.allocated_bytes () -. before) /. Float.of_int (Stdlib.Sys.word_size / 8)
          in
          measured := Some (forced_before, Ir.Tnode.get_padding x.value, words))
  in
  let forced_before, padding, words = Option.value_exn !measured in
  Stdio.eprintf "allocation witness %s: %.0f words for %d cells (not part of the golden)\n" name
    words numel;
  p "witness forces the host initializer itself" (not forced_before);
  p "witness buffer carries committed padding" (check_layout padding);
  p "padded scalar host fill allocates under 512 heap words, whatever its cell count"
    Float.(words < 512.);
  Context.release ctx

(* A rank-1 halo: the run is the whole interior. *)
let allocation_witness_rank1 () =
  let numel = 1 lsl 16 in
  let x = NTDSL.ndarray [| 3. |] ~output_dims:[ numel ] () in
  Train.set_materialized x.value;
  let kernel = NTDSL.ndarray [| 1.; 2.; 3. |] ~output_dims:[ 3 ] () in
  let%op y = x +* "i=+k; k => i" kernel in
  allocation_witness ~name:"alloc_witness" ~numel x y ~check_layout:Option.is_some

(* A 2-D halo over an innermost unpadded channel axis of extent 1: [h] contiguous runs, or [h * w]
   for a fill that issued one per innermost row, as many as there are cells. Over [h = 512] runs,
   even a 3-word pair allocated per run exceeds the bound. *)
let allocation_witness_rank3 () =
  let h = 512 and w = 128 in
  let x = NTDSL.ndarray [| 3. |] ~output_dims:[ h; w; 1 ] () in
  Train.set_materialized x.value;
  let kernel = NTDSL.ndarray (Array.init 9 ~f:Float.of_int) ~output_dims:[ 3; 3 ] () in
  let%op y = x +* "oh=+kh, ow=+kw, c; kh, kw => oh, ow, c" kernel in
  allocation_witness ~name:"alloc_witness_hwc" ~numel:(h * w) x y ~check_layout:(function
    | Some (pads, _) ->
        let dims = Lazy.force x.value.Ir.Tnode.dims in
        Array.length dims = 3
        && dims.(2) = 1
        && Ir.Ops.equal_axis_padding pads.(2) { left = 0; right = 0 }
        && pads.(0).left > 0 && pads.(1).left > 0
    | None -> false)

let parameter_reinit limit =
  let p = TDSL.param ~value:3. ("reinit_scalar_" ^ Int.to_string limit) ~output_dims:[ 3 ] () in
  let q =
    NTDSL.param ~values:[| 2.; 5.; 7. |]
      ("reinit_array_" ^ Int.to_string limit)
      ~output_dims:[ 3 ] ()
  in
  let%op y = p + q in
  let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty y in
  values ctx p [| 3.; 3.; 3. |] "scalar parameter initializes across its inferred shape";
  values ctx q [| 2.; 5.; 7. |] "host-backed array parameter initializes";
  let edited_p = [| 11.; 12.; 13. |] and edited_q = [| -1.; -2.; -3. |] in
  let ctx = Context.set_values ctx p.value edited_p in
  let ctx = Context.set_values ctx q.value edited_q in
  let ctx = Train.init_params ctx Ir.Indexing.Empty y in
  values ctx p edited_p "ordinary initialization preserves an edited scalar parameter";
  values ctx q edited_q "ordinary initialization preserves an edited array parameter";
  let ctx = Train.init_params ~reinit_all:true ctx Ir.Indexing.Empty y in
  values ctx p [| 3.; 3.; 3. |] "reinit_all restores the configured scalar parameter value";
  values ctx q [| 2.; 5.; 7. |] "reinit_all restores the configured array parameter values";
  let ctx = Train.forward_once ~skip_init:true ctx y in
  values ctx y [| 5.; 8.; 10. |] "forward reads the restored parameter buffers";
  Context.release ctx

let dependent_parameter limit =
  let p = TDSL.param ~value:2. ("reinit_source_" ^ Int.to_string limit) ~output_dims:[ 3 ] () in
  let derived =
    TDSL.param
      ~param_init:(NTDSL.add p (Tensor.number 1.))
      ("reinit_dependent_" ^ Int.to_string limit)
      ~output_dims:[ 3 ] ()
  in
  let varying = NTDSL.ndarray [| 1.; 3.; 5. |] ~output_dims:[ 3 ] () in
  let%op y = derived + varying in
  let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty y in
  values ctx derived [| 3.; 3.; 3. |]
    "computed parameter reads its host-backed initializer dependency";
  let ctx = Context.set_values ctx p.value [| 11.; 12.; 13. |] in
  let edited = [| 21.; 22.; 23. |] in
  let ctx = Context.set_values ctx derived.value edited in
  let ctx = Train.init_params ctx Ir.Indexing.Empty y in
  values ctx derived edited "ordinary initialization skips an edited computed parameter";
  let ctx = Train.init_params ~reinit_all:true ctx Ir.Indexing.Empty y in
  values ctx p [| 2.; 2.; 2. |] "reinit_all restores a nested host-backed parameter";
  values ctx derived [| 3.; 3.; 3. |] "computed initialization reads the restored dependency";
  let ctx = Train.forward_once ~skip_init:true ctx y in
  values ctx y [| 4.; 6.; 8. |] "forward reads the recomputed parameter after reinitialization";
  Context.release ctx

let independent_parameters limit =
  let p =
    TDSL.param ~value:2. ("independent_source_" ^ Int.to_string limit) ~output_dims:[ 3 ] ()
  in
  let state =
    NTDSL.param ~value:5. ("independent_state_" ^ Int.to_string limit) ~output_dims:[ 3 ] ()
  in
  let derived =
    TDSL.param ~param_init:(NTDSL.add p state)
      ("independent_derived_" ^ Int.to_string limit)
      ~output_dims:[ 3 ] ()
  in
  let varying = NTDSL.ndarray [| 1.; 3.; 5. |] ~output_dims:[ 3 ] () in
  let%op y = derived + varying in
  let ctx1 = Train.init_params (Context.auto ()) Ir.Indexing.Empty y in
  let ctx1 = Context.set_values ctx1 p.value [| 11.; 12.; 13. |] in
  let ctx1 = Context.set_values ctx1 state.value [| 21.; 22.; 23. |] in
  let ctx2 = Train.init_params (Context.auto ()) Ir.Indexing.Empty y in
  values ctx1 p [| 11.; 12.; 13. |] "a fresh context preserves another context's parameter edits";
  values ctx1 state [| 21.; 22.; 23. |]
    "a fresh context preserves another context's non-differentiable state";
  values ctx2 p [| 2.; 2.; 2. |] "a fresh context initializes its own nested parameter";
  values ctx2 state [| 5.; 5.; 5. |] "a fresh context initializes its own nested state";
  values ctx2 derived [| 7.; 7.; 7. |] "a fresh initializer reads context-owned dependencies";
  let ctx2 = Context.set_values ctx2 p.value [| 31.; 32.; 33. |] in
  let ctx2 = Context.set_values ctx2 state.value [| 41.; 42.; 43. |] in
  values ctx1 p [| 11.; 12.; 13. |] "parameter updates remain independent across contexts";
  values ctx1 state [| 21.; 22.; 23. |] "state updates remain independent across contexts";
  let ctx1 = Train.init_params ~reinit_all:true ctx1 Ir.Indexing.Empty y in
  values ctx1 p [| 2.; 2.; 2. |] "reinitialization restores the selected context's parameter";
  values ctx2 p [| 31.; 32.; 33. |] "reinitialization preserves another context's parameter";
  values ctx2 state [| 41.; 42.; 43. |] "reinitialization preserves another context's state";
  Context.release ctx1;
  let ctx2 = Train.forward_once ~skip_init:true ctx2 y in
  values ctx2 y [| 8.; 10.; 12. |] "releasing one context preserves the other context's forward";
  Context.release ctx2

let () =
  List.iter [ 1; 0 ] ~f:(fun limit ->
      Hashtbl.set Utils.config_file_args ~key:"limit_constant_fill_size" ~data:(Int.to_string limit);
      p "configured limit is effective"
        (String.equal
           (Utils.get_global_arg ~default:"16" ~arg_name:"limit_constant_fill_size")
           (Int.to_string limit));
      List.iter [ 0.; 3. ] ~f:(scalar limit);
      scalar_root limit;
      inferred limit;
      parameter_reinit limit;
      dependent_parameter limit;
      independent_parameters limit;
      List.iter [ 0.; 3. ] ~f:(padded limit));
  Ir.Ops.(
    padded_fill_offsets "padded fill keeps uneven margins on all three axes" [| 4; 5; 6 |]
      [| { left = 1; right = 0 }; { left = 0; right = 2 }; { left = 2; right = 1 } |];
    padded_fill_offsets "padded fill folds an unpadded innermost axis into its runs" [| 4; 5; 3 |]
      [| { left = 1; right = 0 }; { left = 2; right = 1 }; { left = 0; right = 0 } |]);
  allocation_witness_rank1 ();
  allocation_witness_rank3 ()
