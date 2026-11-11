(* Shape-only regressions for gh-ocannl-2131. No backend is initialized. The symbolic case records
   the non-reproduction: Shape creates symbolic axes only at the default basis. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let leaf () = NTDSL.term ~label:[ "open" ] ~fetch_op:(Constant 2.) ()
let tagged basis = NTDSL.ndarray [| 1.; 2.; 3. |] ~output_axes:[ (basis, 3) ] ()
let capture n x = NTDSL.einsum1 ~capture_dims:[ n ] "n => n" x ()
let dims x = Array.to_list (Lazy.force x.Tensor.value.dims)
let rgb x = Array.mem (Shape.to_bases x.Tensor.shape) "rgb" ~equal:String.equal

let case label f =
  Verdict.case label (fun () ->
      Tensor.unsafe_reinitialize ();
      try f ()
      with Row.Shape_error _ as e ->
        Tensor.rewind_to_last_closure ~reason:"gh-2131 case" ();
        raise e)

let refused label f =
  let rejected =
    match f () with
    | () -> false
    | exception Row.Shape_error _ ->
        Tensor.rewind_to_last_closure ~reason:"gh-2131 expected refusal" ();
        true
  in
  p label rejected

let () =
  case "set_dim before a named capture" (fun () ->
      let n = Shape.get_variable_ref "n" in
      Shape.set_dim n 3;
      let y = capture n (tagged "rgb") in
      p "set_dim supplies the size without replacing rgb"
        (List.equal Int.equal (dims y) [ 3 ] && rgb y);
      let m = Shape.get_variable_ref "n" in
      Shape.set_equal n m;
      let z = capture m (leaf ()) in
      p "a sized capture publishes rgb for a later equality" (rgb z));
  case "set_dim after an open capture, then a named capture" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let y = capture n (leaf ()) in
      Shape.set_dim n 3;
      let z = capture n (tagged "rgb") in
      p "a pending size allows a later rgb constraint" (List.equal Int.equal (dims y) [ 3 ] && rgb z));
  List.iter
    [ (true, false); (false, true); (true, true) ]
    ~f:(fun (size_left, size_right) ->
      List.iter [ false; true ] ~f:(fun swap ->
          let name =
            Printf.sprintf "live equality, sizes %b/%b, order %s" size_left size_right
              (if swap then "reversed" else "forward")
          in
          let pair () =
            let n = Shape.get_variable_ref "n" and m = Shape.get_variable_ref "n" in
            let _x = capture n (leaf ()) and y = capture m (leaf ()) in
            if size_left then Shape.set_dim n 3;
            if size_right then Shape.set_dim m 3;
            if swap then Shape.set_equal m n else Shape.set_equal n m;
            ignore (capture n (tagged "rgb") : Tensor.t);
            (m, y)
          in
          case (name ^ ": propagation") (fun () ->
              let _, y = pair () in
              p
                (name ^ ": a later rgb basis propagates across the equality")
                (List.equal Int.equal (dims y) [ 3 ] && rgb y));
          case (name ^ ": conflict") (fun () ->
              let m, _ = pair () in
              refused (name ^ ": later different bases conflict") (fun () ->
                  ignore (dims (capture m (tagged "xyz")))))));
  List.iter [ false; true ] ~f:(fun closed ->
      List.iter [ false; true ] ~f:(fun bound ->
          List.iter [ false; true ] ~f:(fun swap ->
              let name =
                Printf.sprintf "source %s, target %s, argument order %s"
                  (if closed then "closed" else "open")
                  (if bound then "bound" else "unbound")
                  (if swap then "reversed" else "forward")
              in
              case name (fun () ->
                  let n = Shape.get_variable_ref "n" in
                  let y = capture n (tagged "rgb") in
                  if closed then ignore (dims y : int list);
                  let m = Shape.get_variable_ref "n" in
                  let z = if bound then Some (capture m (leaf ())) else None in
                  p
                    (name ^ ": the target is unsolved before set_equal")
                    (Option.is_none m.var_ref.solved_dim);
                  if swap then Shape.set_equal m n else Shape.set_equal n m;
                  let z = Option.value_or_thunk z ~default:(fun () -> capture m (leaf ())) in
                  p
                    (name ^ ": set_equal carries the solved rgb dimension")
                    (List.equal Int.equal (dims z) [ 3 ] && rgb z);
                  refused (name ^ ": set_equal still refuses another named basis") (fun () ->
                      ignore (dims (capture m (tagged "xyz"))))))));
  case "set_dim size conflict" (fun () ->
      let n = Shape.get_variable_ref "n" in
      Shape.set_dim n 4;
      refused "set_dim still refuses another size" (fun () ->
          ignore (dims (capture n (tagged "rgb")))));
  case "set_equal with an explicit size and a solved tag" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let y = capture n (tagged "rgb") in
      ignore (dims y : int list);
      let m = Shape.get_variable_ref "n" in
      Shape.set_dim m 3;
      Shape.set_equal n m;
      let z = capture m (leaf ()) in
      p "a size-only solution acquires the equal dimension's tag" (rgb z));
  case "two solved dimension bases conflict" (fun () ->
      let n = Shape.get_variable_ref "n" and m = Shape.get_variable_ref "n" in
      ignore (dims (capture n (tagged "rgb")));
      ignore (dims (capture m (tagged "xyz")));
      let message =
        match Shape.set_equal n m with
        | () -> None
        | exception Row.Shape_error (msg, _) ->
            Tensor.rewind_to_last_closure ~reason:"gh-2131 named basis refusal" ();
            Some msg
      in
      p "set_equal rejects two solved same-sized axes with different bases" (Option.is_some message);
      p "the basis refusal names the reference"
        (Option.exists message ~f:(fun msg -> String.is_substring msg ~substring:"label n")));
  case "symbolic recapture and proxy" (fun () ->
      let sym, _ = Ir.Indexing.get_static_symbol ~static_range:6 Ir.Indexing.Empty in
      let n = Shape.get_variable_ref "n" in
      Shape.set_sym_dim n sym;
      let x = leaf () in
      let y = capture n x in
      p "the public symbolic setter creates a default axis"
        (List.equal Int.equal (dims y) [ 6 ]
        && Array.mem (Shape.to_bases y.Tensor.shape) Row.default_basis ~equal:String.equal);
      let z = capture n y in
      p "a solved symbolic reference survives recapture"
        (List.equal Int.equal (dims z) [ 6 ]
        && Array.equal String.equal (Shape.to_bases z.Tensor.shape) [| Row.default_basis |]);
      let proxy =
        Shape.product_space_shape ~shape:z.Tensor.shape
          ~logic:(Transpose (Permute ("n => n", [ n ]), y.Tensor.shape))
      in
      Shape.propagate_shapes
        (Shape.make_update_step ~shape:proxy ~neutral_elem:None
           ~logic:(Terminal { is_param = false; logic = Fetch (Constant 2.) }));
      p "a solved symbolic reference survives the proxy"
        (Array.equal Int.equal (Shape.to_dims proxy) [| 6 |]
        && Array.equal String.equal (Shape.to_bases proxy) [| Row.default_basis |]));
  case "a numeric size meets a named use-site bound" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let x = leaf () in
      let y = capture n x in
      Shape.set_dim n 3;
      let z = NTDSL.add y (tagged "rgb") () in
      p "the numeric size takes the matching use-site basis"
        (List.equal Int.equal (dims z) [ 3 ] && rgb y));
  case "a dimension-row equality keeps only the product" (fun () ->
      let n = Shape.get_variable_ref "n" in
      ignore (dims (capture n (tagged "rgb")));
      let row = Shape.get_variable_ref "r" in
      let y = NTDSL.einsum1 ~capture_dims:[ row ] "..r.. => ..r.." (leaf ()) () in
      Shape.set_equal n row;
      ignore (dims y);
      let z = NTDSL.einsum1 ~capture_dims:[ row ] "r => r" (tagged "xyz") () in
      p "the row product does not acquire rgb"
        (List.equal Int.equal (dims z) [ 3 ]
        && Array.mem (Shape.to_bases z.Tensor.shape) "xyz" ~equal:String.equal));
  case "a held output size precedes window-pair settling" (fun () ->
      let build ~eager =
        Tensor.unsafe_reinitialize ();
        let x = leaf () and o = Shape.get_variable_ref "o" in
        let first =
          NTDSL.einsum ~capture_dims:[ o ] "2*o<+k; k => o" x (tagged "h")
            ?output_dims:(if eager then Some [ 1 ] else None)
            ()
        in
        let second = NTDSL.einsum "2*p<+k; k => p" x (tagged Row.default_basis) () in
        if not eager then Shape.set_dim o 1;
        let first_dims = dims first and second_dims = dims second in
        ( first_dims,
          Array.to_list (Shape.to_bases first.Tensor.shape),
          second_dims,
          Array.to_list (Shape.to_bases second.Tensor.shape) )
      in
      let eager = build ~eager:true in
      p "the eager window pair has distinct output tags"
        (Poly.equal eager ([ 1 ], [ Row.default_basis ], [ 1 ], [ "h" ]));
      let held = build ~eager:false in
      p "set_dim closes both windows with the eager sizes and tags" (Poly.equal eager held));
  case "a size claim preserves an arithmetically inferred broadcast top" (fun () ->
      let n = Row.get_var () and m = Row.get_var () in
      let row : Row.t =
        {
          beg_dims = [];
          dims = [ Var n; Var m ];
          bcast = Broadcastable;
          prov = Row.empty_provenance;
        }
      in
      let constraints : Row.constraint_ list =
        [
          Dim_size { dim = Var n; size = 1; origin = [] };
          Rows_constr
            {
              r = [ row ];
              constr = Total_elems { numerator = Num_elems 1; divided_by = []; keep_axis = false };
              origin = [];
            };
        ]
      in
      let pending, env = Row.solve_inequalities ~stage:Stage1 constraints Row.empty_env in
      let pending, env = Row.solve_inequalities ~stage:Stage2 pending env in
      let top v =
        match Row.get_dim_from_env env v with
        | Some (Dim { d = 1; basis; _ }) -> String.equal basis Row.bcast_if_1
        | _ -> false
      in
      p_empty "the numeric product and size leave no deferred constraint" ~over:constraints pending;
      p_all ~min:2 "a size-only 1 keeps the broadcast tops implied by a two-axis product of 1"
        [ n; m ] ~f:top);
  case "set_equal carries a closed free dimension's broadcast top" (fun () ->
      let n = Shape.get_variable_ref "n" in
      ignore (dims (capture n (leaf ())));
      let m = Shape.get_variable_ref "n" in
      let plain =
        NTDSL.init ~l:"plain_one" ~prec:Ir.Ops.single ~b:[] ~o:[ 1 ] ~f:(fun _ -> 2.) ()
      in
      ignore (dims (capture m plain));
      refused "set_equal refuses a closed broadcast top against a plain 1" (fun () ->
          Shape.set_equal n m))
