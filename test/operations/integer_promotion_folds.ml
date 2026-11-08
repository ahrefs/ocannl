(* An interval fold reads an operator where the renderer leaves it (gh-ocannl-1957).

   A byte [a + b] is 300 to whatever compares it: C computes byte and uint16 operators in [int], and
   only a store or a scope converts the result back. [Interval.at_prec] models that conversion, so
   applied to an operator's result it decided [Cmple(a + b, 255)] over a sum of 300, where the
   kernel computes 0. The same goes for a float-valued operator at ANY integer precision, computed
   in float ([op_table_compile], gh-ocannl-1785): [Cmplt(Sin(a), 0)] folded to 0 at uint32.

   Each shape runs twice with a value worked out by hand: with its literals as written, which the
   interval analysis reads, and with each literal in a materialized node, which it cannot decide, so
   nothing but the renderer answers. Every shape discriminates: its cells take both outcomes where
   it claims the comparison survives. Where it claims the fold, the claim is what tells this model
   from one that only widens, since a [top] operator decides nothing.

   The reach of the defect is IR holding an operator under a comparison. A nested [%cd] expression
   is not one: it gets a node, so its comparison reads a kept scope, which holds the wrapped value
   ([integer_scope_collapse]). The first leg pins that.

   The constant arms answer the same question (gh-ocannl-2018). A fold replaces an operator by a
   leaf, and a leaf converts from double to its slot's precision, which is undefined where the value
   does not fit. So [Add(200, 100)] at byte is an operator, whose [int] 300 a store converts to 44
   and [Min] reads as 300. No wrapped constant gives both, so such an operator stays. That reaches
   [%op]: [0. - a] over an unsigned node needed no hand-built IR. A compiler's reading of an
   undefined conversion is no oracle, so a structural claim, on [simplify_llc] directly, says that
   no constant is left outside its slot's range, and fails whatever the compiler did with one. *)
open Base
module Train = Ocannl.Train
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode
open Verdict.Claims

let n = 4
let byte = Ops.byte
let uint16 = Ops.uint16
let uint32 = Ops.uint32
let single = Ops.single
let next_first_id = ref 26000

let node ?(dims = [| n |]) prec label =
  next_first_id := !next_first_id + 1;
  Ll_test.node_factory ~prec ~first_id:!next_first_id ~dims () label

type env = {
  arg : int -> LL.scalar_arg;
  lit : Ops.prec -> float -> LL.scalar_arg;
  p : Ops.prec;  (** the slot's, hence what the operators evaluate at *)
}

type expectation =
  | Survives  (** a comparison is left *)
  | Folds  (** no comparison is left *)
  | Keeps_operator  (** [simplify_llc] leaves an operator *)
  | Folds_to_leaf  (** [simplify_llc] leaves none *)
  | Value_only  (** nothing to claim of the form it takes *)

type case = {
  label : string;
  slot : Ops.prec;
  inputs : (Ops.prec * float array) list;
  bounded : bool;  (** the inputs carry their exact ranges as bounds candidates *)
  guard : bool;  (** [value] guards a store of 7 over a 9, instead of being stored *)
  value : env -> LL.scalar_t;
  expect : expectation;
  want : float array;
}

type outcome = {
  values : float array list;
  comparisons : int;
  operators : int;  (** in the simplified literal form *)
  strays : int;  (** constants left outside an integer slot's range *)
  stable : bool;  (** simplifying again changes nothing *)
}

let structural = function Keeps_operator | Folds_to_leaf | Value_only -> true | _ -> false

let run case ~literal =
  let slug = String.map case.label ~f:(fun c -> if Char.is_alphanum c then c else '_') in
  let name = "ipf_" ^ slug ^ if literal then "_lit" else "_node" in
  let inputs =
    List.mapi case.inputs ~f:(fun k (prec, values) ->
        (node prec (Printf.sprintf "%s_in%d" name k), prec, values))
  in
  let out = node case.slot (name ^ "_out") in
  let held = ref [] in
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let lit prec v =
    if literal then (LL.Constant v, prec)
    else
      let tn = node prec (Printf.sprintf "%s_lit%d" name (List.length !held)) in
      held := (tn, Array.create ~len:n v) :: !held;
      (Ll_test.get tn at, prec)
  in
  let env =
    {
      arg =
        (fun k ->
          let tn, prec, _ = List.nth_exn inputs k in
          (Ll_test.get tn at, prec));
      lit;
      p = case.slot;
    }
  in
  let body =
    if case.guard then
      Ll_test.seq
        (Ll_test.set out at (LL.Constant 9.))
        (Ll_test.if_at case.slot (case.value env) (Ll_test.set out at (LL.Constant 7.)))
    else Ll_test.set out at (case.value env)
  in
  let seed = List.map inputs ~f:(fun (tn, _, values) -> (tn, values)) @ !held in
  if case.bounded then
    List.iter inputs ~f:(fun (tn, _, values) ->
        let bound f = Float.to_int (Array.reduce_exn values ~f) in
        Tn.propose_bounds ~what:"test" tn
          (Ir.Interval.of_int_range (bound Float.min) (bound Float.max)));
  let materialized = out :: List.map seed ~f:fst in
  List.iter materialized ~f:Ll_test.materialize;
  let llc = Ll_test.loop_n s n body in
  let operators, strays, stable =
    if not (structural case.expect) then (0, 0, true)
    else
      let once = LL.simplify_llc [] llc in
      let twice = LL.simplify_llc [] once in
      let outside c =
        match Ops.integer_range case.slot with
        | Some (lo, hi) -> Float.(c < lo || c > hi)
        | None -> false
      in
      ( Ll_test.count_scalar once ~f:(function LL.Binop _ | LL.Unop _ -> true | _ -> false),
        Ll_test.count_scalar once ~f:(function LL.Constant c -> outside c | _ -> false),
        Sexp.equal (LL.sexp_of_t once) (LL.sexp_of_t twice) )
  in
  let o = Ll_test.optimize ~materialized ~name llc in
  let values = Ll_test.execute ~name o ~seed ~read:[ out ] in
  let comparisons =
    Ll_test.count_scalar o.LL.llc ~f:(function
      | LL.Binop ((Ops.Cmplt | Ops.Cmple | Ops.Cmpeq | Ops.Cmpne), _, _) -> true
      | _ -> false)
  in
  { values; comparisons; operators; strays; stable }

let check case =
  let literal = run case ~literal:true in
  let held = run case ~literal:false in
  let want = [ case.want ] in
  p
    (case.label ^ ": with its literals as written it gives the worked value")
    (Ll_test.same literal.values want);
  p
    (case.label ^ ": with each literal in a node it gives the worked value")
    (Ll_test.same held.values want);
  if structural case.expect then (
    p (case.label ^ ": no constant is left outside the slot's range") (literal.strays = 0);
    p (case.label ^ ": simplifying again changes nothing") literal.stable);
  match case.expect with
  | Survives -> p (case.label ^ ": the comparison survives") (literal.comparisons > 0)
  | Folds -> p (case.label ^ ": the interval analysis decides it") (literal.comparisons = 0)
  | Keeps_operator -> p (case.label ^ ": the operator stays") (literal.operators > 0)
  | Folds_to_leaf -> p (case.label ^ ": it folds to a leaf") (literal.operators = 0)
  | Value_only -> ()

let bin op a b ~p = (LL.Binop (op, a, b), p)
let un op a ~p = (LL.Unop (op, a), p)
let flat c = Array.create ~len:n c
let le e a = LL.Binop (Ops.Cmple, a, e.lit e.p 255.)
let lt0 e a = LL.Binop (Ops.Cmplt, a, e.lit e.p 0.)
let sum e = bin Ops.Add (e.arg 0) (e.arg 1) ~p:e.p
let diff e = bin Ops.Sub (e.arg 0) (e.arg 1) ~p:e.p

(* Sums 200, 300, 300, 255; differences 0, 100, 0, -155. *)
let a = [| 100.; 200.; 150.; 50. |]
let b = [| 100.; 100.; 150.; 205. |]

let case ?(slot = byte) ?(inputs = [ (byte, a); (byte, b) ]) ?(bounded = false) ?(guard = false)
    ?(expect = Survives) label value want =
  { label; slot; inputs; bounded; guard; value; expect; want }

(* An operator's result past the byte range, through each rule. *)
let over_byte = [| 1.; 0.; 0.; 1. |]

let () =
  let init l values = NTDSL.init ~l ~prec:byte ~o:[ n ] ~f:(fun idx -> values.(idx.(0))) () in
  let a = init "a" a and b = init "b" b in
  let%cd comp = { compared } =: (a + b <= 255.) in
  Tn.update_prec compared.value byte;
  let ctx, routine = Train.to_routine (Context.auto ()) Train.IDX.empty comp in
  Train.run ctx routine;
  (* The sum's node holds 200, 44, 44, 255. *)
  p "a nested %cd comparison reads its operand's node, which holds the wrapped sum"
    (Ll_test.same [ Context.get_values ctx compared.value ] [ flat 1. ])

let () =
  List.iter ~f:check
    [
      case "byte sum against 255" (fun e -> le e (sum e)) over_byte;
      case "byte sum over 255 from bounds" ~bounded:true
        ~inputs:[ (byte, [| 200.; 201.; 202.; 203. |]); (byte, [| 100.; 101.; 102.; 103. |]) ]
        ~expect:Folds
        (fun e -> le e (sum e))
        (flat 0.);
      case "byte sum not below zero" ~expect:Folds
        (fun e -> LL.Binop (Ops.Cmple, e.lit byte 0., sum e))
        (flat 1.);
      case "uint16 sum against 65535" ~slot:uint16
        ~inputs:
          [
            (uint16, [| 1000.; 40000.; 30000.; 65535. |]); (uint16, [| 1000.; 30000.; 30000.; 1. |]);
          ]
        (fun e -> LL.Binop (Ops.Cmple, sum e, e.lit uint16 65535.))
        [| 1.; 0.; 1.; 0. |];
      case "uint16 product against 65535" ~slot:uint16
        ~inputs:[ (uint16, [| 2.; 300.; 3.; 256. |]); (uint16, [| 3.; 300.; 21845.; 256. |]) ]
        (fun e ->
          LL.Binop (Ops.Cmple, bin Ops.Mul (e.arg 0) (e.arg 1) ~p:uint16, e.lit uint16 65535.))
        [| 1.; 0.; 1.; 0. |];
      case "byte difference below zero" (fun e -> lt0 e (diff e)) [| 0.; 0.; 0.; 1. |];
      case "byte difference not below zero, the other way round"
        (fun e -> LL.Binop (Ops.Cmple, e.lit byte 0., diff e))
        [| 1.; 1.; 1.; 0. |];
      case "byte negation below zero"
        ~inputs:[ (byte, [| 0.; 1.; 2.; 3. |]) ]
        (fun e -> lt0 e (un Ops.Neg (e.arg 0) ~p:byte))
        [| 0.; 1.; 1.; 1. |];
      case "byte negation never above zero" ~expect:Folds
        ~inputs:[ (byte, [| 0.; 1.; 2.; 3. |]) ]
        (fun e -> LL.Binop (Ops.Cmple, un Ops.Neg (e.arg 0) ~p:byte, e.lit byte 0.))
        (flat 1.);
      case "byte product against 255"
        ~inputs:[ (byte, [| 2.; 100.; 3.; 200. |]); (byte, [| 3.; 100.; 85.; 2. |]) ]
        (fun e -> le e (bin Ops.Mul (e.arg 0) (e.arg 1) ~p:byte))
        [| 1.; 0.; 1.; 0. |];
      case "byte triple product against 255"
        ~inputs:
          [ (byte, [| 2.; 100.; 3.; 200. |]); (byte, [| 3.; 100.; 5.; 2. |]); (byte, flat 2.) ]
        (fun e -> le e (LL.Ternop (Ops.Mul3, e.arg 0, e.arg 1, e.arg 2), byte))
        [| 1.; 0.; 1.; 0. |];
      case "byte Max of a sum against 255"
        ~inputs:[ (byte, a); (byte, b); (byte, flat 3.) ]
        (fun e -> le e (bin Ops.Max (sum e) (e.arg 2) ~p:byte))
        over_byte;
      case "byte Relu of a sum against 255" (fun e -> le e (un Ops.Relu (sum e) ~p:byte)) over_byte;
      case "byte Trunc of a sum against 255"
        (fun e -> le e (un Ops.Trunc (sum e) ~p:byte))
        over_byte;
      case "byte open gate over a sum against 255"
        ~inputs:[ (byte, a); (byte, b); (byte, flat 1.) ]
        (fun e -> le e (bin Ops.Relu_gate (e.arg 2) (sum e) ~p:byte))
        over_byte;
      case "byte select arm against 255"
        ~inputs:[ (byte, a); (byte, b); (byte, flat 1.); (byte, flat 2.) ]
        (fun e -> le e (LL.Ternop (Ops.Where, e.arg 2, sum e, e.arg 3), byte))
        over_byte;
      (* -155 % 7 is -1 in C, past a remainder rule that reads the operand as wrapped. *)
      case "byte remainder of a negative difference"
        (fun e -> lt0 e (bin Ops.Mod (diff e) (e.lit byte 7.) ~p:byte))
        [| 0.; 0.; 0.; 1. |];
      case "a select condition over a byte sum"
        ~inputs:[ (byte, a); (byte, b); (byte, flat 7.); (byte, flat 9.) ]
        (fun e -> LL.Ternop (Ops.Where, (le e (sum e), byte), e.arg 2, e.arg 3))
        [| 7.; 9.; 9.; 7. |];
      case "a statement guard over a byte sum" ~guard:true
        (fun e -> le e (sum e))
        [| 7.; 9.; 9.; 7. |];
      (* Float-valued: computed in float, at any integer precision. *)
      case "byte Exp against 255"
        ~inputs:[ (byte, [| 0.; 1.; 5.; 10. |]) ]
        (fun e -> le e (un Ops.Exp (e.arg 0) ~p:byte))
        [| 1.; 1.; 1.; 0. |];
      case "byte Sin below zero"
        ~inputs:[ (byte, [| 0.; 1.; 4.; 5. |]) ]
        (fun e -> lt0 e (un Ops.Sin (e.arg 0) ~p:byte))
        [| 0.; 0.; 1.; 1. |];
      case "uint32 Sin below zero" ~slot:uint32
        ~inputs:[ (uint32, [| 0.; 1.; 4.; 5. |]) ]
        (fun e -> lt0 e (un Ops.Sin (e.arg 0) ~p:uint32))
        [| 0.; 0.; 1.; 1. |];
      case "byte sum following a Sin below zero"
        ~inputs:[ (byte, [| 0.; 1.; 4.; 5. |]); (byte, flat 0.) ]
        (fun e -> lt0 e (bin Ops.Add (un Ops.Sin (e.arg 0) ~p:byte) (e.arg 1) ~p:byte))
        [| 0.; 0.; 1.; 1. |];
      case "byte Log of a Recip below zero"
        ~inputs:[ (byte, [| 1.; 2.; 3.; 4. |]) ]
        (fun e -> lt0 e (un Ops.Log (un Ops.Recip (e.arg 0) ~p:byte) ~p:byte))
        [| 0.; 1.; 1.; 1. |];
      (* The slot's precision is what evaluates, not the comparison's annotation: a single sum of
         byte reads compares 300, and a single sum of uint32 reads does not wrap. *)
      case "a byte comparison in a single slot" ~slot:single
        ~inputs:[ (byte, a); (byte, b); (single, flat 0.5) ]
        (fun e ->
          LL.Binop
            ( Ops.Add,
              bin Ops.Cmple (bin Ops.Add (e.arg 0) (e.arg 1) ~p:byte) (e.lit single 255.) ~p:byte,
              e.arg 2 ))
        [| 1.5; 0.5; 0.5; 1.5 |];
      case "a uint32 comparison in a single slot" ~slot:single
        ~inputs:
          [
            (uint32, [| 1.; 4e9; 2e9; 3e9 |]); (uint32, [| 1.; 1e9; 1.; 2e9 |]); (single, flat 0.5);
          ]
        (fun e ->
          LL.Binop
            ( Ops.Add,
              bin Ops.Cmple
                (bin Ops.Add (e.arg 0) (e.arg 1) ~p:uint32)
                (e.lit single 4294967295.) ~p:uint32,
              e.arg 2 ))
        [| 1.5; 0.5; 1.5; 0.5 |];
    ]

(* A one-hot sum over a float-valued index at a byte annotation: [k == 3 * Sin((byte)a)] holds at no
   [k] for a non-integral index, and the read converts to byte first, so 256 is index 0. No guard
   reads that as written, so it is no gather. *)
let () =
  let classes = 4 in
  let name = "ipf_one_hot" in
  let table = node ~dims:[| classes |] single (name ^ "_table") in
  let input = node uint16 (name ^ "_in") in
  let out = node single (name ^ "_out") in
  let s = Ll_test.sym () and k = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let index =
    ( LL.Binop
        (Ops.Mul, (LL.Constant 3., byte), (LL.Unop (Ops.Sin, (Ll_test.get input at, byte)), byte)),
      byte )
  in
  let term =
    LL.Ternop
      ( Ops.Where,
        (LL.Binop (Ops.Cmpeq, (Ll_test.embed k, byte), index), byte),
        (Ll_test.get table [| Ll_test.iter k |], single),
        (LL.Constant 0., single) )
  in
  let body =
    Ll_test.seq
      (Ll_test.set out at (LL.Constant 0.))
      (Ll_test.loop_n k classes
         (Ll_test.set out at (LL.Binop (Ops.Add, (Ll_test.get out at, single), (term, single)))))
  in
  let materialized = [ out; input; table ] in
  List.iter materialized ~f:Ll_test.materialize;
  let o = Ll_test.optimize ~materialized ~name (Ll_test.loop_n s n body) in
  let values =
    Ll_test.execute ~name o
      ~seed:[ (input, [| 256.; 1.; 2.; 3. |]); (table, [| 10.; 20.; 30.; 40. |]) ]
      ~read:[ out ]
  in
  p "a one-hot sum over a float-valued index stays a sum"
    (Ll_test.count_scalar o.LL.llc ~f:(function LL.Get_dynamic _ -> true | _ -> false) = 0);
  (* Index 0, then 2.52, 2.72, 0.42: only the first is a class. *)
  p "a float-valued one-hot index selects by its value at the byte reads"
    (Ll_test.same values [ [| 10.; 0.; 0.; 0. |] ])

(* What an expression's value becomes in a node of precision [prec]: a store wraps an unsigned one.
   No case here leaves a signed range: signed overflow is C's undefined behavior, operator or
   not. *)
let stored prec v =
  match Ops.integer_range prec with
  | Some (0., hi) ->
      let modulus = hi +. 1. in
      let r = Float.mod_float v modulus in
      if Float.(r < 0.) then r +. modulus else r
  | _ -> v

let is_unsigned prec = match Ops.integer_range prec with Some (0., _) -> true | _ -> false

(* The operator stays exactly where its result is not a value of the precision. *)
let negative_stays prec = if is_unsigned prec then Keeps_operator else Folds_to_leaf

(* What compares below zero: byte and uint16 compute in [int], and only uint32 and uint64 wrap. *)
let negative_compares prec = if is_unsigned prec && not (Ops.promoted_to_int prec) then 0. else 1.
let lits op a b e = LL.Binop (op, e.lit e.p a, e.lit e.p b)
let over x e c = bin Ops.Add (e.arg 0) (e.lit e.p x) ~p:c

let every_precision prec =
  let name s = Ops.prec_string prec ^ " " ^ s in
  let const ?expect label value want =
    case (name label) ~slot:prec ~inputs:[] ?expect value (flat want)
  in
  let with_x ?expect x label value want =
    case (name label) ~slot:prec ~inputs:[ (prec, flat x) ] ?expect value (flat want)
  in
  [
    const "3 - 5" ~expect:(negative_stays prec) (lits Ops.Sub 3. 5.) (stored prec (-2.));
    const "Neg 3" ~expect:(negative_stays prec)
      (fun e -> LL.Unop (Ops.Neg, e.lit e.p 3.))
      (stored prec (-3.));
    with_x 3. "0 - x" ~expect:Keeps_operator
      (fun e -> LL.Binop (Ops.Sub, e.lit e.p 0., e.arg 0))
      (stored prec (-3.));
    with_x 1. "(x + 3) - 5" ~expect:Keeps_operator
      (fun e -> LL.Binop (Ops.Sub, over 3. e e.p, e.lit e.p 5.))
      (stored prec (-1.));
    const "Neg 3 below zero" ~expect:Value_only
      (fun e -> lt0 e (un Ops.Neg (e.lit e.p 3.) ~p:e.p))
      (negative_compares prec);
    const "3 - 5 below zero" ~expect:Value_only
      (fun e -> lt0 e (bin Ops.Sub (e.lit e.p 3.) (e.lit e.p 5.) ~p:e.p))
      (negative_compares prec);
    (* The fold is still there wherever a leaf holds the result. *)
    const "3 + 4" ~expect:Folds_to_leaf (lits Ops.Add 3. 4.) 7.;
    (* An integer [Div] truncates. *)
    const "7 / 2" ~expect:Keeps_operator (lits Ops.Div 7. 2.) 3.;
  ]

(* In-range folds, read off [simplify_llc]: executing each at each precision would add nothing to [3
   + 4] above but kernels to compile. *)
let () =
  List.iter Ops.integer_precs ~f:(fun prec ->
      List.iter
        [
          ("6 * 7", Ops.Mul, 6., 7., 42.);
          ("6 / 3", Ops.Div, 6., 3., 2.);
          ("7 % 3", Ops.Mod, 7., 3., 1.);
          ("Max 3 4", Ops.Max, 3., 4., 4.);
          ("3 < 4", Ops.Cmplt, 3., 4., 1.);
        ]
        ~f:(fun (label, op, a, b, want) ->
          let out = node prec "ipf_fold_out" in
          let s = Ll_test.sym () in
          let value = LL.Binop (op, (LL.Constant a, prec), (LL.Constant b, prec)) in
          let llc = Ll_test.loop_n s n (Ll_test.set out [| Ll_test.iter s |] value) in
          let simplified = LL.simplify_llc [] llc in
          let wanted = function LL.Constant c -> Float.(c = want) | _ -> false in
          p
            (Ops.prec_string prec ^ " " ^ label ^ " folds to its value")
            (Ll_test.count_scalar simplified ~f:wanted = 1
            && Ll_test.count_scalar simplified ~f:(function LL.Binop _ -> true | _ -> false) = 0)))

(* A true quotient of 2251799813685248.8 rounds to an integer in double, where C truncates. *)
let () =
  List.iter [ Ops.int64; Ops.uint64 ] ~f:(fun prec ->
      let out = node prec "ipf_div_out" in
      let s = Ll_test.sym () in
      let value =
        LL.Binop (Ops.Div, (LL.Constant 11258999068426244., prec), (LL.Constant 5., prec))
      in
      let llc = Ll_test.loop_n s n (Ll_test.set out [| Ll_test.iter s |] value) in
      p
        (Ops.prec_string prec ^ " a quotient a double rounds onto an integer stays a division")
        (Ll_test.count_scalar (LL.simplify_llc [] llc) ~f:(function
           | LL.Binop (Ops.Div, _, _) -> true
           | _ -> false)
        = 1))

let two53 = 9007199254740992.

let () =
  List.iter ~f:check (List.concat_map Ops.integer_precs ~f:every_precision);
  List.iter ~f:check
    [
      case "byte 200 + 100" ~inputs:[] ~expect:Keeps_operator (lits Ops.Add 200. 100.) (flat 44.);
      case "uint16 40000 + 30000" ~slot:uint16 ~inputs:[] ~expect:Keeps_operator
        (lits Ops.Add 40000. 30000.) (flat 4464.);
      case "uint16 256 * 256" ~slot:uint16 ~inputs:[] ~expect:Keeps_operator
        (lits Ops.Mul 256. 256.) (flat 0.);
      case "uint32 4e9 + 5e8" ~slot:uint32 ~inputs:[] ~expect:Keeps_operator (lits Ops.Add 4e9 5e8)
        (flat 205032704.);
      case "uint32 65536 * 65536" ~slot:uint32 ~inputs:[] ~expect:Keeps_operator
        (lits Ops.Mul 65536. 65536.) (flat 0.);
      (* No wrapped constant gives these: each reads the sum as computed. *)
      case "byte Min of a sum over the range"
        ~inputs:[ (byte, flat 250.) ]
        ~expect:Keeps_operator
        (fun e ->
          LL.Binop (Ops.Min, bin Ops.Add (e.lit byte 200.) (e.lit byte 100.) ~p:byte, e.arg 0))
        (flat 250.);
      case "uint16 Min of a sum over the range" ~slot:uint16
        ~inputs:[ (uint16, flat 65000.) ]
        ~expect:Keeps_operator
        (fun e ->
          LL.Binop
            (Ops.Min, bin Ops.Add (e.lit uint16 40000.) (e.lit uint16 30000.) ~p:uint16, e.arg 0))
        (flat 65000.);
      case "a select on a byte sum of 256" ~inputs:[] ~expect:Value_only
        (fun e ->
          LL.Ternop
            ( Ops.Where,
              bin Ops.Add (e.lit byte 128.) (e.lit byte 128.) ~p:byte,
              e.lit byte 7.,
              e.lit byte 9. ))
        (flat 7.);
      case "a guard on a byte sum of 256" ~inputs:[] ~guard:true ~expect:Value_only
        (lits Ops.Add 128. 128.) (flat 7.);
      (* The canonicalizer mints constants as well. *)
      case "byte (x + 200) + 100"
        ~inputs:[ (byte, flat 1.) ]
        ~expect:Keeps_operator
        (fun e -> LL.Binop (Ops.Add, over 200. e byte, e.lit byte 100.))
        (flat 45.);
      case "byte (x * 100) * 3"
        ~inputs:[ (byte, flat 1.) ]
        ~expect:Keeps_operator
        (fun e ->
          LL.Binop (Ops.Mul, bin Ops.Mul (e.arg 0) (e.lit byte 100.) ~p:byte, e.lit byte 3.))
        (flat 44.);
      (* A fixed index converts from an integer, which wraps. *)
      case "byte fixed index 300" ~inputs:[] ~expect:Value_only
        (fun _ -> LL.Embed_index (Ir.Indexing.Fixed_idx 300))
        (flat 44.);
      case "uint16 fixed index 300" ~slot:uint16 ~inputs:[] ~expect:Folds_to_leaf
        (fun _ -> LL.Embed_index (Ir.Indexing.Fixed_idx 300))
        (flat 300.);
      (* Metal must not parse these integer leaves through f32 (gh-ocannl-2084). *)
      case "int32 2^24 + 1" ~slot:Ops.int32 ~inputs:[] ~expect:Folds_to_leaf
        (lits Ops.Add 16777216. 1.) (flat 16777217.);
      case "int64 70000 * 70000" ~slot:Ops.int64 ~inputs:[] ~expect:Folds_to_leaf
        (lits Ops.Mul 70000. 70000.) (flat 4900000000.);
      case "int32 literal 16777217" ~slot:Ops.int32 ~inputs:[] ~expect:Folds_to_leaf
        (fun e -> fst (e.lit e.p 16777217.))
        (flat 16777217.);
      case "int32 literal -16777217" ~slot:Ops.int32 ~inputs:[] ~expect:Folds_to_leaf
        (fun e -> fst (e.lit e.p (-16777217.)))
        (flat (-16777217.));
      case "int64 literal 2^53 + 2" ~slot:Ops.int64 ~inputs:[] ~expect:Value_only
        (fun e -> fst (e.lit e.p 9007199254740994.))
        (flat 9007199254740994.);
      case "int64 literal minimum" ~slot:Ops.int64 ~inputs:[] ~expect:Value_only
        (fun e -> fst (e.lit e.p (-9223372036854775808.)))
        (flat (-9223372036854775808.));
      case "uint64 literal 2^63 + 2048" ~slot:Ops.uint64 ~inputs:[] ~expect:Value_only
        (fun e -> fst (e.lit e.p 9223372036854777856.))
        (flat 9223372036854777856.);
      (* Past 2^53 a double cannot compute it, in range or not. *)
      case "int64 2^53 + 1 against 2^53" ~slot:Ops.int64 ~inputs:[] ~expect:Value_only
        (fun e ->
          LL.Binop
            ( Ops.Cmpeq,
              bin Ops.Add (e.lit Ops.int64 two53) (e.lit Ops.int64 1.) ~p:Ops.int64,
              e.lit Ops.int64 two53 ))
        (flat 0.);
      case "uint64 2^53 + 1 against 2^53" ~slot:Ops.uint64 ~inputs:[] ~expect:Value_only
        (fun e ->
          LL.Binop
            ( Ops.Cmpeq,
              bin Ops.Add (e.lit Ops.uint64 two53) (e.lit Ops.uint64 1.) ~p:Ops.uint64,
              e.lit Ops.uint64 two53 ))
        (flat 0.);
      (* A condition evaluates at its own annotation, whatever the store: a single one computes 300,
         and a byte one in a single store computes 256 in [int]. *)
      case "a single condition in a byte store" ~inputs:[] ~expect:Folds_to_leaf
        (fun e ->
          LL.Ternop
            ( Ops.Where,
              bin Ops.Add (e.lit single 200.) (e.lit single 100.) ~p:single,
              e.lit byte 7.,
              e.lit byte 9. ))
        (flat 7.);
      case "a byte condition in a single store" ~slot:single ~inputs:[] ~expect:Value_only
        (fun e ->
          LL.Ternop
            ( Ops.Where,
              bin Ops.Add (e.lit byte 128.) (e.lit byte 128.) ~p:byte,
              e.lit single 7.,
              e.lit single 9. ))
        (flat 7.);
      (* A non-integral operand converts first, and truncates. *)
      case "byte 2.5 + 3.5" ~inputs:[] ~expect:Keeps_operator (lits Ops.Add 2.5 3.5) (flat 5.);
    ]

(* The routes from [%op]. [0. - a] needs no update of a precision: the literal follows [a]'s. *)
let () =
  let inputs prec =
    NTDSL.init ~l:"a" ~prec ~o:[ 2 ] ~f:(fun idx -> Float.of_int (idx.(0) + 3)) ()
  in
  let values t = Context.get_values (Train.forward_once (Context.auto ()) t) t.Tensor.value in
  List.iter Ops.integer_precs ~f:(fun prec ->
      let claim label t want =
        p (Ops.prec_string prec ^ " %op " ^ label) (Ll_test.same [ values t ] [ want ])
      in
      let a = inputs prec in
      let%op negated = 0. - a in
      claim "0. - a is a negation" negated [| stored prec (-3.); stored prec (-4.) |];
      let a = inputs prec in
      let%op shifted = a + 3. - 5. in
      claim "a + 3. - 5. is a shift" shifted [| stored prec 1.; stored prec 2. |];
      let%op difference = 3. - 5. in
      Tn.update_prec difference.value prec;
      claim "3. - 5. at the node's precision" difference [| stored prec (-2.) |]);
  let%op sum = 200. + 100. in
  Tn.update_prec sum.value byte;
  p "byte %op 200. + 100. stores the wrapped sum" (Ll_test.same [ values sum ] [ [| 44. |] ])
