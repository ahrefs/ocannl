(* gh-ocannl-2191: a concatenation equation with tagged components gets one verdict in every order.

   The search is t2129's (gh-ocannl-2129, [test_trial_rollback]'s pool of windows, bounds, shared
   kernels and window pairs) with the concatenations added that it found order-dependent: [[v; u]]
   against a tagged size, and [[u; 2v]] against [[k; 2_t]], for each tag. Every store of two
   constraints, and of three where the third is open, is solved in every order, each constraint
   registered at its own Stage 1 pass as an operation would be, the variables closed by a shape row
   as a tensor's would be. On master [3ac86187c] 96 of the 16,730 stores depended on order, 32 of
   them refused in some orders; and in 1,492 runs the solver refused its own answer when told it
   first. Three things decided a basis on partial information: [bcast_if_1] above size 1 was
   claim-free inside concatenations alone, the [Concat]/[Dim] residual took a solved claim against a
   [default] size the fully solved arm refuses, and the [Concat]/[Concat] arms dropped one side's
   claim (subtracting a solved side, cancelling to a zero residual, cancelling claimed equal
   components).

   The claims: no store's verdict or solution depends on order; an answer is accepted when stated up
   front; registering a store's constraints in one batch agrees with registering them one by one;
   positive controls stay accepted, so refusing everything fails. Nonidentity stride families
   (gh-ocannl-2245) include plain and padded strides, solved inputs, identities and pairs. Strides
   enter every two-constraint store; three-constraint stores use the window/concat pool, including
   gh-ocannl-2254's fixed [u = 3 T] family. A solved/pinned matrix checks both equation orientations
   and registration orders. Before adding the stride families, the fixed family gave 21,124 stores,
   14 order-dependent before its fix: 12 from composite window kernels and two from
   gh-ocannl-2095's residual basis choice. Those two stores remain named, counted exceptions; every
   other store remains subject to all three claims. Counts go to stderr. The search runs whole. *)

open! Base
open Ocannl
open Verdict.Claims

let origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

let tagged d basis = Row.get_dim ~d ~basis ()
let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin }
let ineq res opnd : Row.constraint_ = Row.Dim_ineq { res; opnd; from_ = Sexp.Atom "test"; origin }
let closing = [ Row.Stage5; Stage6a; Stage6b; Stage7 ]

(* What closes [vs]: a shape row holding them, as a tensor's would. *)
let close vs : Row.constraint_ =
  Row.Shape_row
    ( {
        beg_dims = [];
        dims = List.map vs ~f:(fun v -> Row.Var v);
        bcast = Broadcastable;
        prov = Row.provenance ~sh_id:2191 ~kind:`Output;
      },
      origin )

let run batches =
  List.fold batches
    ~init:(Ok ([], Row.empty_env))
    ~f:(fun acc (stage, cs) ->
      Result.bind acc ~f:(fun (deferred, env) ->
          match Row.solve_inequalities ~stage (deferred @ cs) env with
          | r -> Ok r
          | exception Row.Shape_error (msg, _) -> Error msg))

let window ?(stride = 1) ~kernel over =
  Row.Affine
    { stride; over; conv = Some { dilation = 1; kernel; use_padding = false }; stride_offset = 0 }

let strided ?(padded = false) stride over =
  Row.Affine
    {
      stride;
      over;
      conv =
        (if padded then Some { dilation = 1; kernel = tagged 3 "h"; use_padding = true } else None);
      stride_offset = 0;
    }

let tags = [ Row.default_basis; Row.bcast_if_1; "a"; "h" ]

let rec permutations = function
  | [] -> [ [] ]
  | xs ->
      List.concat_mapi xs ~f:(fun i x ->
          let rest = List.filteri xs ~f:(fun j _ -> j <> i) in
          List.map (permutations rest) ~f:(fun perm -> x :: perm))

type item = { name : string; mk : Row.dim_var -> Row.dim_var -> Row.dim_var -> Row.constraint_ }

let pool =
  let wins =
    List.concat_map tags ~f:(fun kt ->
        List.concat_map tags ~f:(fun it ->
            List.map
              [ (1, 3, 3); (1, 1, 1); (2, 3, 5) ]
              ~f:(fun (stride, ks, is) ->
                {
                  name = Printf.sprintf "w(s%d,k%d %s)(v)=%d %s" stride ks kt is it;
                  mk =
                    (fun v _ _ -> eq (window ~stride ~kernel:(tagged ks kt) (Var v)) (tagged is it));
                })))
  and bounds =
    List.concat_map tags ~f:(fun t ->
        List.map [ 1; 3 ] ~f:(fun d ->
            { name = Printf.sprintf "v>=%d %s" d t; mk = (fun v _ _ -> ineq (tagged d t) (Var v)) }))
  and kernels =
    List.concat_map tags ~f:(fun et ->
        List.map tags ~f:(fun it ->
            {
              name = Printf.sprintf "w(k)(1 %s)=3 %s" et it;
              mk = (fun _ _ k -> eq (window ~kernel:(Var k) (tagged 1 et)) (tagged 3 it));
            }))
  and pairs =
    List.map tags ~f:(fun kt ->
        {
          name = Printf.sprintf "w(k3 %s)(v)=w(k3 %s)(u)" kt kt;
          mk =
            (fun v u _ ->
              eq (window ~kernel:(tagged 3 kt) (Var v)) (window ~kernel:(tagged 3 kt) (Var u)));
        })
  and concats =
    List.concat_map tags ~f:(fun t ->
        [
          {
            name = Printf.sprintf "[u; 2v]=[k; 2 %s]" t;
            mk =
              (fun v u k ->
                eq
                  (Row.Concat
                     [ Var u; Affine { stride = 2; over = Var v; conv = None; stride_offset = 0 } ])
                  (Row.Concat [ Var k; tagged 2 t ]));
          };
          {
            name = Printf.sprintf "[v; u]=4 %s" t;
            mk = (fun v u _ -> eq (Row.Concat [ Var v; Var u ]) (tagged 4 t));
          };
        ])
  and fixed =
    List.map tags ~f:(fun t ->
        { name = Printf.sprintf "u=3 %s" t; mk = (fun _ u _ -> eq (Var u) (tagged 3 t)) })
  in
  let strides =
    List.concat_map [ false; true ] ~f:(fun padded ->
        List.concat_map [ 1; 2 ] ~f:(fun stride ->
            List.concat_map [ 1; 2 ] ~f:(fun d ->
                List.map tags ~f:(fun t ->
                    {
                      name = Printf.sprintf "stride(p%b,s%d)(v)=%d %s" padded stride d t;
                      mk = (fun v _ _ -> eq (strided ~padded stride (Var v)) (tagged d t));
                    }))))
  in
  let stride_pairs =
    List.map [ 1; 2 ] ~f:(fun stride ->
        {
          name = Printf.sprintf "stride(pair,s%d)" stride;
          mk = (fun v u _ -> eq (strided 2 (Var v)) (strided stride (Var u)));
        })
  in
  wins @ bounds @ kernels @ pairs @ concats @ fixed @ strides @ stride_pairs

let item name = List.find_exn pool ~f:(fun i -> String.equal i.name name)

(* A solution: each variable's size and basis, or [None] where it stays unsized. *)
type solution = (string * (int * string) option) list

let solution_to_string (sol : solution) =
  String.concat ~sep:" "
    (List.map sol ~f:(fun (n, x) ->
         n ^ "=" ^ match x with Some (d, b) -> Printf.sprintf "%d of %s" d b | None -> "unsized"))

let has sol name d basis =
  Option.equal
    (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
    (List.Assoc.find_exn sol ~equal:String.equal name)
    (Some (d, basis))

type outcome = Refused | Accepted of solution

let outcome_to_string = function Refused -> "refused" | Accepted sol -> solution_to_string sol
let equal_outcome o1 o2 = String.equal (outcome_to_string o1) (outcome_to_string o2)

(* Solve [items] in [order]: one constraint per Stage 1 pass ([batched] puts them in one), after
   [pins], each fixing a variable to a value first. *)
let solve ?(batched = false) ?(pins = []) items order =
  let v = Row.get_var () and u = Row.get_var () and k = Row.get_var () in
  let vars = [ ("v", v); ("u", u); ("k", k) ] in
  let pins =
    List.filter_map pins ~f:(fun (n, x) ->
        Option.map x ~f:(fun (d, b) ->
            eq (Var (List.Assoc.find_exn vars ~equal:String.equal n)) (tagged d b)))
  in
  let cs = pins @ List.map order ~f:(fun i -> (List.nth_exn items i).mk v u k) in
  let registered =
    if batched then [ (Row.Stage1, cs) ] else List.map cs ~f:(fun c -> (Row.Stage1, [ c ]))
  in
  match
    run
      (registered
      @ List.map [ Row.Stage2; Stage3 ] ~f:(fun s -> (s, []))
      @ [ (Row.Stage4, [ close [ v; u; k ] ]) ]
      @ List.map closing ~f:(fun s -> (s, [])))
  with
  | Error _ -> Refused
  | Ok (_, env) ->
      Accepted
        (List.map vars ~f:(fun (n, var) ->
             ( n,
               match Row.get_dim_from_env env var with
               | Some (Dim { d; basis; _ }) -> Some (d, basis)
               | _ -> None )))

let store_name items = String.concat ~sep:" ; " (List.map items ~f:(fun i -> i.name))
let orders items = permutations (List.init (List.length items) ~f:Fn.id)
let outcomes items = List.map (orders items) ~f:(solve items)

type store = { items : item list; outs : outcome list }

let order_dependent { items; outs } =
  let dependent = not (List.for_all outs ~f:(equal_outcome (List.hd_exn outs))) in
  if dependent then
    Stdio.eprintf "order-dependent: %s => %s (not part of the golden)\n" (store_name items)
      (String.concat ~sep:" | " (List.map outs ~f:outcome_to_string));
  dependent

(* An answer the solver gives, stated first, is one it accepts. *)
let refuses_own_answer (items, order, sol) =
  match solve ~pins:sol items order with
  | Accepted _ -> false
  | Refused ->
      Stdio.eprintf "refuses its own answer: %s => %s (not part of the golden)\n" (store_name items)
        (solution_to_string sol);
      true

let batch_differs { items; outs } =
  let batched = solve ~batched:true items (List.init (List.length items) ~f:Fn.id) in
  let differs = not (equal_outcome batched (List.hd_exn outs)) in
  if differs then
    Stdio.eprintf "batched differs: %s => %s against %s (not part of the golden)\n"
      (store_name items) (outcome_to_string batched)
      (outcome_to_string (List.hd_exn outs));
  differs

let rec choose k = function
  | _ when k = 0 -> [ [] ]
  | [] -> []
  | x :: xs -> List.map (choose (k - 1) xs) ~f:(fun c -> x :: c) @ choose k xs

let stride_matrix () =
  let cases =
    List.cartesian_product [ false; true ] [ 1; 2 ]
    |> List.concat_map ~f:(fun (padded, stride) ->
        List.cartesian_product [ 1; 2 ] tags
        |> List.concat_map ~f:(fun (d, basis) ->
            List.cartesian_product tags [ false; true ]
            |> List.map ~f:(fun (input_basis, reverse) ->
                (padded, stride, d, basis, input_basis, reverse))))
    |> List.concat_map ~f:(fun c -> List.map [ [ 0; 1 ]; [ 1; 0 ] ] ~f:(fun order -> (c, order)))
  in
  p_all "stride basis rule agrees with pinned extents in both orientations and orders" cases
    ~f:(fun ((padded, stride, d, basis, input_basis, reverse), order) ->
      let expected_basis =
        if stride = 2 && d = 1 && String.equal basis Row.bcast_if_1 then Row.default_basis
        else basis
      in
      let items =
        [
          { name = "pinned extent"; mk = (fun v _ _ -> eq (Var v) (tagged d basis)) };
          {
            name = "stride";
            mk =
              (fun v _ _ ->
                let affine = strided ~padded stride (Var v)
                and input = tagged (stride * d) input_basis in
                if reverse then eq input affine else eq affine input);
          };
        ]
      in
      match solve items order with
      | Accepted _ -> String.equal expected_basis input_basis
      | Refused -> not (String.equal expected_basis input_basis))

let stride_pair_controls () =
  let pairs =
    List.cartesian_product [ false; true ] [ false; true ]
    |> List.concat_map ~f:(fun (left, right) ->
        List.map [ false; true ] ~f:(fun reverse ->
            {
              name = "stride pair";
              mk =
                (fun v u _ ->
                  let lhs = strided ~padded:left 2 (Var v)
                  and rhs = strided ~padded:right 2 (Var u) in
                  if reverse then eq rhs lhs else eq lhs rhs);
            }))
  in
  let pinned =
    List.cartesian_product pairs [ (2, "a"); (4, Row.default_basis); (1, Row.bcast_if_1) ]
    |> List.concat_map ~f:(fun (pair, (d, basis)) ->
        let pin = { name = "pin"; mk = (fun v _ _ -> eq (Var v) (tagged d basis)) } in
        List.map [ [ 0; 1 ]; [ 1; 0 ] ] ~f:(fun order -> (pair, pin, d, basis, order)))
  in
  p_all ~min:48 "pinned stride pairs size the other extent in both orders" pinned
    ~f:(fun (pair, pin, d, basis, order) ->
      match solve [ pair; pin ] order with
      | Refused -> false
      | Accepted sol ->
          let expected =
            if d = 1 && String.equal basis Row.bcast_if_1 then Row.default_basis else basis
          in
          Option.equal
            (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
            (List.Assoc.find_exn sol ~equal:String.equal "u")
            (Some (d, expected)));
  p_all ~min:8 "open stride pairs close to equal sized extents" pairs ~f:(fun pair ->
      match solve [ pair ] [ 0 ] with
      | Refused -> false
      | Accepted sol -> (
          match
            ( List.Assoc.find_exn sol ~equal:String.equal "v",
              List.Assoc.find_exn sol ~equal:String.equal "u" )
          with
          | Some (d1, b1), Some (d2, b2) -> d1 > 0 && d1 = d2 && String.equal b1 b2
          | _ -> false))

let () =
  (* As in [test_trial_rollback]: a third constraint is an open window, a shared kernel, a pair or a
     concatenation. *)
  let held =
    List.filter pool ~f:(fun i ->
        not
          (String.is_prefix i.name ~prefix:"stride("
          || String.is_prefix i.name ~prefix:"v>="
          || String.is_prefix i.name ~prefix:"w(s1,k1"
          || String.is_prefix i.name ~prefix:"w(s2"))
  in
  let stores =
    List.map (choose 2 pool @ choose 3 held) ~f:(fun items -> { items; outs = outcomes items })
  in
  let original =
    List.filter stores ~f:(fun { items; _ } ->
        List.for_all items ~f:(fun i -> not (String.is_prefix i.name ~prefix:"stride(")))
  in
  Stdio.eprintf "%d original stores, %d accepted in some order (not part of the golden)\n"
    (List.length original)
    (List.count original ~f:(fun { outs; _ } ->
         List.exists outs ~f:(function Accepted _ -> true | Refused -> false)));
  let accepted_runs =
    List.concat_map stores ~f:(fun { items; outs } ->
        List.filter_map
          (List.zip_exn (orders items) outs)
          ~f:(fun (order, out) ->
            match out with Refused -> None | Accepted sol -> Some (items, order, sol)))
  in
  let accepted_stores =
    List.count stores ~f:(fun { outs; _ } ->
        List.exists outs ~f:(function Accepted _ -> true | Refused -> false))
  in
  Stdio.eprintf "%d stores, %d accepted in some order, %d accepted runs (not part of the golden)\n"
    (List.length stores) accepted_stores (List.length accepted_runs);
  let stride_top_stores =
    let concat = item "[u; 2v]=[k; 2 default]" in
    let pairs = List.filter pool ~f:(fun i -> String.is_prefix i.name ~prefix:"w(k3 ") in
    let top_sources =
      List.filter pool ~f:(fun i ->
          String.is_prefix i.name ~prefix:"v>="
          || String.is_prefix i.name ~prefix:"w(k3 "
          || String.equal i.name "w(s1,k1 default)(v)=1 bcast_if_1"
          || String.equal i.name "w(s1,k1 bcast_if_1)(v)=1 bcast_if_1")
    in
    List.map top_sources ~f:(fun i -> [ i; concat ])
    @ List.map (choose 2 pairs) ~f:(fun is -> is @ [ concat ])
  in
  Stdio.eprintf "%d stride-top stores (not part of the golden)\n" (List.length stride_top_stores);
  let known_2095 { items; _ } =
    List.exists [ "a"; "h" ] ~f:(fun tag ->
        String.equal (store_name items)
          (Printf.sprintf "w(s1,k3 default)(v)=3 %s ; [v; u]=4 %s ; u=3 default" tag tag))
  in
  let dependent = List.filter stores ~f:order_dependent in
  Stdio.eprintf "%d order-dependent stores, %d named #2095 exceptions (not part of the golden)\n"
    (List.length dependent) (List.count stores ~f:known_2095);
  let verdicts names = outcomes (List.map names ~f:item) in
  Verdict.case "stride basis derivation" stride_matrix;
  Verdict.case "stride pair solutions" stride_pair_controls;
  Verdict.case "concatenations with tagged components settle alike in every order" (fun () ->
      p "exactly two stores are named as #2095 exceptions" (List.count stores ~f:known_2095 = 2);
      p_none ~min:16000 "stores whose verdict or solution depends on the order, except #2095"
        (List.filter stores ~f:(Fn.non known_2095))
        ~f:(fun store -> List.mem dependent store ~equal:phys_equal);
      p_none ~min:7000 "accepted answers refused when stated up front" accepted_runs
        ~f:refuses_own_answer;
      p_none ~min:16000 "stores registered in one batch settling otherwise" stores ~f:batch_differs;
      p "over 1,500 stores are accepted, so refusing everything fails" (accepted_stores > 1500);
      List.iter [ "a"; "h"; Row.bcast_if_1 ] ~f:(fun tag ->
          p_all
            ("shared claim with a zero residual settles to u = 1, k = 3_" ^ tag)
            (let names = [ "w(s2,k3 default)(v)=5 " ^ tag; "[u; 2v]=[k; 2 " ^ tag ^ "]" ] in
             let outs = verdicts names in
             Stdio.eprintf "shared-claim control: %s => %s (not part of the golden)\n"
               (String.concat ~sep:" ; " names)
               (String.concat ~sep:" | " (List.map outs ~f:outcome_to_string));
             outs)
            ~f:(function
              | Accepted sol ->
                  has sol "v" 2 tag && has sol "u" 1 Row.bcast_if_1 && has sol "k" 3 tag
              | Refused -> false));
      List.iter [ "a"; "h"; Row.bcast_if_1 ] ~f:(fun tag ->
          let items =
            List.map [ "w(s2,k3 default)(v)=5 " ^ tag; "[u; 2v]=[k; 2 " ^ tag ^ "]" ] ~f:item
            @ [
                {
                  name = "k = 3 default";
                  mk = (fun _ _ k -> eq (Var k) (tagged 3 Row.default_basis));
                };
              ]
          in
          p_all ("a later plain k keeps both concatenations' shared claim " ^ tag) (outcomes items)
            ~f:(function
            | Accepted sol -> has sol "u" 1 tag
            | Refused -> false));
      List.iter [ "a"; "h"; Row.bcast_if_1 ] ~f:(fun tag ->
          let items =
            List.map [ "w(s2,k3 default)(v)=5 " ^ tag; "[u; 2v]=[k; 2 " ^ tag ^ "]" ] ~f:item
            @ [
                {
                  name = "k >= 5 default";
                  mk = (fun _ _ k -> ineq (tagged 5 Row.default_basis) (Var k));
                };
              ]
          in
          let outs = outcomes items in
          Stdio.eprintf "waiting-side bound: %s => %s (not part of the golden)\n" (store_name items)
            (String.concat ~sep:" | " (List.map outs ~f:outcome_to_string));
          (* Diagnostic only: this bounded store is already refused on the base. Propagating its
             bound through the sum remains outside this closing-order correction. *)
          ());
      (* Twenty stores yield 64 orderings: the floor guards the name-based selection. *)
      p_all ~min:64 "a stride of the broadcast top settles in every order"
        (List.concat_map stride_top_stores ~f:outcomes) ~f:(function
        | Accepted _ -> true
        | Refused -> false);
      p_all "a filler stride cannot also supply a size-2 broadcast-atom claim"
        (verdicts [ "[u; 2v]=[k; 2 default]"; "[u; 2v]=[k; 2 bcast_if_1]" ])
        ~f:(equal_outcome Refused);
      let kernel_controls =
        List.concat_map [ Row.default_basis; Row.bcast_if_1 ] ~f:(fun et ->
            List.concat_map tags ~f:(fun it ->
                let fixed_tags =
                  if String.equal it "a" || String.equal it "h" then [ Row.default_basis; it ]
                  else [ Row.default_basis ]
                in
                List.concat_map fixed_tags ~f:(fun xt ->
                    verdicts
                      [
                        Printf.sprintf "w(k)(1 %s)=3 %s" et it;
                        "[u; 2v]=[k; 2 default]";
                        Printf.sprintf "u=3 %s" xt;
                      ])))
      in
      p_all ~min:72 "all twelve composite-kernel stores are accepted in all six orders"
        kernel_controls ~f:(function
        | Accepted _ -> true
        | Refused -> false);
      p_all "the issue's example is refused in every order"
        (verdicts [ "w(s1,k3 default)(v)=3 default"; "[v; u]=4 default"; "[v; u]=4 bcast_if_1" ])
        ~f:(equal_outcome Refused);
      p_all "and so is #1935's held-set vetting case, a named claim against a plain size"
        (verdicts [ "w(s1,k3 a)(v)=3 a"; "[v; u]=4 default"; "[v; u]=4 a" ])
        ~f:(equal_outcome Refused);
      p_all "control: [v; u] = 4 a with v = 1 a takes u = 3 a in every order"
        (verdicts [ "w(s1,k3 a)(v)=3 a"; "[v; u]=4 a" ])
        ~f:(function
          | Accepted sol ->
              Option.equal
                (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
                (List.Assoc.find_exn sol ~equal:String.equal "u")
                (Some (3, "a"))
          | Refused -> false);
      p_all "control: [u; 2v] = [k; 2 a] with k = 3 default takes u = 3 a in every order"
        (verdicts [ "w(s1,k3 default)(v)=3 a"; "w(k)(1 default)=3 default"; "[u; 2v]=[k; 2 a]" ])
        ~f:(function
          | Accepted sol ->
              Option.equal
                (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
                (List.Assoc.find_exn sol ~equal:String.equal "u")
                (Some (3, "a"))
          | Refused -> false))

(* Composite kernels meet the same arithmetic and closing rules as a bare variable. These probes
   also exercise a kernel equation that cannot solve eagerly, so recursing on the unchanged window
   would livelock. *)
let () =
  let dim d = tagged d Row.default_basis in
  let affine v = Row.Affine { stride = 2; over = Row.Var v; conv = None; stride_offset = 0 } in
  let settled batches =
    match run batches with Ok ([], env) -> Some env | Ok (_ :: _, _) | Error _ -> None
  in
  let sized env v d =
    match Row.get_dim_from_env env v with
    | Some (Dim { d = actual; basis; _ }) -> actual = d && String.equal basis Row.default_basis
    | _ -> false
  in
  Verdict.case "composite window kernel arithmetic and deferral" (fun () ->
      let v = Row.get_var () in
      p "an affine kernel is solved through its expression"
        (Option.value_map
           (settled [ (Row.Stage1, [ eq (window ~kernel:(affine v) (dim 1)) (dim 4) ]) ])
           ~default:false
           ~f:(fun env -> sized env v 2));
      let v = Row.get_var () and u = Row.get_var () in
      let kernel = Row.Concat [ Var v; Var u ] in
      let held = eq (window ~kernel (dim 1)) (dim 3) in
      p "a multi-variable kernel equation defers, then drains after a component is sized"
        (match run [ (Row.Stage1, [ held ]) ] with
        | Ok (_ :: _, _) ->
            Option.value_map
              (settled [ (Row.Stage1, [ held ]); (Row.Stage1, [ eq (Var u) (dim 1) ]) ])
              ~default:false
              ~f:(fun env -> sized env v 2 && sized env u 1)
        | _ -> false);
      let v = Row.get_var () and u = Row.get_var () in
      let kernel = Row.Concat [ affine v; dim 1 ] in
      Row.add_resolve_at_use_dim u;
      p "composite kernel variables wait for the iterator before guessing"
        (Option.value_map
           (settled
              ([ (Row.Stage1, [ eq (window ~kernel (Var u)) (dim 7); ineq (dim 3) (Var u) ]) ]
              @ List.map [ Row.Stage2; Stage3 ] ~f:(fun stage -> (stage, []))
              @ [ (Row.Stage4, [ close [ v; u ] ]) ]
              @ List.map closing ~f:(fun stage -> (stage, []))))
           ~default:false
           ~f:(fun env -> sized env v 2 && sized env u 3));
      let v = Row.get_var () and u = Row.get_var () in
      let kernel = Row.Concat [ affine v; dim 1 ] in
      p "an unbounded iterator closes before the composite kernel variables"
        (Option.value_map
           (settled
              ([ (Row.Stage1, [ eq (window ~kernel (Var u)) (dim 7) ]) ]
              @ List.map [ Row.Stage2; Stage3 ] ~f:(fun stage -> (stage, []))
              @ [ (Row.Stage4, [ close [ v; u ] ]) ]
              @ List.map closing ~f:(fun stage -> (stage, []))))
           ~default:false
           ~f:(fun env ->
             sized env v 3
             &&
             match Row.get_dim_from_env env u with
             | Some (Dim { d = 1; basis; _ }) -> String.equal basis Row.bcast_if_1
             | _ -> false));
      let sym, _ = Ir.Indexing.get_static_symbol ~static_range:3 Ir.Indexing.Empty in
      p "a rigid symbolic kernel cannot be equated with a concrete kernel size"
        (match
           run [ (Row.Stage1, [ eq (window ~kernel:(Row.get_sym_dim sym) (dim 1)) (dim 3) ]) ]
         with
        | Error _ -> true
        | Ok _ -> false))
