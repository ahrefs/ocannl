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
   ([integer_scope_collapse]). The first leg pins that. *)
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

type expectation = Survives | Folds

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
  let o = Ll_test.optimize ~materialized ~name (Ll_test.loop_n s n body) in
  let values = Ll_test.execute ~name o ~seed ~read:[ out ] in
  let comparisons =
    Ll_test.count_scalar o.LL.llc ~f:(function
      | LL.Binop ((Ops.Cmplt | Ops.Cmple | Ops.Cmpeq | Ops.Cmpne), _, _) -> true
      | _ -> false)
  in
  (values, comparisons)

let check case =
  let literal, comparisons = run case ~literal:true in
  let held, _ = run case ~literal:false in
  let want = [ case.want ] in
  p
    (case.label ^ ": with its literals as written it gives the worked value")
    (Ll_test.same literal want);
  p
    (case.label ^ ": with each literal in a node it gives the worked value")
    (Ll_test.same held want);
  match case.expect with
  | Survives -> p (case.label ^ ": the comparison survives") (comparisons > 0)
  | Folds -> p (case.label ^ ": the interval analysis decides it") (comparisons = 0)

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
