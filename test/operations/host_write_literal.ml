(* gh-ocannl-2021: a hoisted literal must not silently ignore a host write. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let run ~n ~hoisted =
  let source = Array.init n ~f:(fun i -> Float.of_int (i + 1)) in
  let changed = Array.map source ~f:(fun x -> x +. 100.) in
  let x = TDSL.ndarray source ~label:[ "literal" ] ~output_dims:[ n ] () in
  Train.set_materialized x.Tensor.value;
  let early = Context.auto () in
  let early_refused =
    match Context.set_values early x.Tensor.value changed with
    | _ -> false
    | exception Utils.User_error message -> String.is_substring message ~substring:"gh-ocannl-2021"
  in
  p (Printf.sprintf "%d %b: literal write before compilation is refused" n hoisted) early_refused;
  Context.release early;
  let%op y = x *. 2 in
  let transform opt =
    if not hoisted then opt
    else
      let axis =
        List.hd_exn
          (List.find_exn (Ll_test.nest_paths opt.Ir.Low_level.llc) ~f:(fun p -> List.length p = 1))
      in
      Ir.Schedule.apply
        [
          Ir.Schedule.Stage
            {
              source = x.Tensor.value;
              tile_loops = [ axis ];
              shared = false;
              cooperative = None;
              hoisted = true;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            };
        ]
        opt
  in
  let ctx, routine =
    Context.compile
      ~name:(Printf.sprintf "literal_%d_%s" n (if hoisted then "hoisted" else "materialized"))
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ()) (Train.forward y) Ir.Indexing.Empty
  in
  let refused fn action =
    let message = match action () with _ -> "" | exception Utils.User_error message -> message in
    p
      (Printf.sprintf "%d %b: %s refuses the literal with a remedy" n hoisted fn)
      (String.is_substring message ~substring:("Context." ^ fn)
      && String.is_substring message ~substring:"literal"
      && String.is_substring message ~substring:"Operation.init")
  in
  let host = Context.to_host ctx x.Tensor.value in
  refused "set_values" (fun () -> Context.set_values ctx x.Tensor.value changed);
  refused "set_value" (fun () -> Context.set_value ctx x.Tensor.value [| 0 |] 999.);
  Ir.Ndarray.set_flat_values host changed;
  refused "from_host" (fun () -> Context.from_host ctx x.Tensor.value host);
  p_all2
    (Printf.sprintf "%d %b: refused writes leave the literal intact" n hoisted)
    (Context.get_values ctx x.Tensor.value)
    source ~f:Float.equal;
  let ctx = Context.run ctx routine in
  Context.get_values ctx y.Tensor.value

let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      let materialized = run ~n ~hoisted:false in
      let hoisted = run ~n ~hoisted:true in
      p_all2
        (Printf.sprintf "%d: materialized reference retains every literal value" n)
        materialized
        (Array.init n ~f:(fun i -> Float.of_int (2 * (i + 1))))
        ~f:Float.equal;
      p_all2
        (Printf.sprintf "%d: hoisted result matches the materialized reference" n)
        hoisted materialized ~f:Float.equal)

(* Parameterization changes the initializer's value node from a literal into mutable data. *)
let () =
  List.iter [ 2; 32 ] ~f:(fun n ->
      let initial = Array.init n ~f:(fun i -> Float.of_int (i + 1)) in
      let changed = Array.map initial ~f:(fun v -> v +. 100.) in
      let x =
        TDSL.param ~param_init:(TDSL.ndarray initial) "ndarray_parameter" ~output_dims:[ n ] ()
      in
      let%op y = x *. 2 in
      p
        (Printf.sprintf "%d: ndarray-initialized parameter cannot be hoisted" n)
        (not (Ir.Schedule.hoistable_constant x.Tensor.value));
      let ctx, routine =
        Context.compile
          ~name:(Printf.sprintf "ndarray_parameter_%d" n)
          (Context.auto ()) (Train.forward y) Ir.Indexing.Empty
      in
      let ctx = Context.set_values ctx x.Tensor.value changed in
      let ctx = Context.set_value ctx x.Tensor.value [| 0 |] 999. in
      let host = Context.to_host ctx x.Tensor.value in
      p
        (Printf.sprintf "%d: parameter accepts a single-cell host write" n)
        (Float.equal (Ir.Ndarray.retrieve_flat_values host).(0) 999.);
      Ir.Ndarray.set_flat_values host changed;
      let ctx = Context.from_host ctx x.Tensor.value host in
      let ctx = Context.run ctx routine in
      p_all2
        (Printf.sprintf "%d: ndarray-initialized parameter reads the host writes" n)
        (Context.get_values ctx y.Tensor.value)
        (Array.map changed ~f:(fun v -> 2. *. v))
        ~f:Float.equal)
