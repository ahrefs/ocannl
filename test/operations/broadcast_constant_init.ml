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
      List.iter [ 0.; 3. ] ~f:(padded limit))
