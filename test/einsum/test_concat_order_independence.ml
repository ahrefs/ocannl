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
   positive controls stay accepted, so refusing everything fails. Counts go to stderr. The search
   runs whole: it takes well under a second. *)

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
  in
  wins @ bounds @ kernels @ pairs @ concats

let item name = List.find_exn pool ~f:(fun i -> String.equal i.name name)

(* A solution: each variable's size and basis, or [None] where it stays unsized. *)
type solution = (string * (int * string) option) list

let solution_to_string (sol : solution) =
  String.concat ~sep:" "
    (List.map sol ~f:(fun (n, x) ->
         n ^ "=" ^ match x with Some (d, b) -> Printf.sprintf "%d of %s" d b | None -> "unsized"))

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

let () =
  (* As in [test_trial_rollback]: a third constraint is an open window, a shared kernel, a pair or a
     concatenation. *)
  let held =
    List.filter pool ~f:(fun i ->
        not
          (String.is_prefix i.name ~prefix:"v>="
          || String.is_prefix i.name ~prefix:"w(s1,k1"
          || String.is_prefix i.name ~prefix:"w(s2"))
  in
  let stores =
    List.map (choose 2 pool @ choose 3 held) ~f:(fun items -> { items; outs = outcomes items })
  in
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
  let verdicts names = outcomes (List.map names ~f:item) in
  Verdict.case "concatenations with tagged components settle alike in every order" (fun () ->
      p_none ~min:16000 "stores whose verdict or solution depends on the order" stores
        ~f:order_dependent;
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
                  let has n d basis =
                    Option.equal
                      (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
                      (List.Assoc.find_exn sol ~equal:String.equal n)
                      (Some (d, basis))
                  in
                  has "v" 2 tag && has "u" 1 Row.bcast_if_1 && has "k" 3 tag
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
            | Accepted sol ->
                Option.equal
                  (fun (d1, b1) (d2, b2) -> d1 = d2 && String.equal b1 b2)
                  (List.Assoc.find_exn sol ~equal:String.equal "u")
                  (Some (1, tag))
            | Refused -> false));
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
