(* An integer scope collapses only where its value provably stays the same (gh-ocannl-1884).

   A [Local_scope] holds its node's value: the materialized store's conversion to the scope's
   precision and evaluation at it. Collapsing drops both, so a byte [a + b] that C's integer
   promotion computes as 300 is read as 300 where the node holds 44, and a byte [a / b] read in a
   float product divides in float. The materialized run is the oracle.

   Each shape runs in four forms: the producer materialized; the scoped form through the
   [?prelowered] seam as written, the uncollapsed oracle; the producer virtualized and inlined by
   [optimize], whose own simplify decides the collapse; and the scoped form through a schedule,
   whose trailing simplify decides it. Each compares with a value worked out by hand, and the last
   two claim whether the scope survived, both ways: a rule keeping every scope is as wrong as one
   collapsing them all, since the gh-343 gather rewrite recognizes a one-hot comparison only after
   its scope collapsed.

   A setter's own nested arithmetic has no node and no oracle, and stays C-promoted:
   [op_table_compile] pins [Min(a + b, c) = 250].

   No shape rests on signed overflow or an out-of-range float conversion, and every producer varies
   with its cell. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let n = 4
let byte = Ops.byte
let uint16 = Ops.uint16
let int32 = Ops.int32
let uint32 = Ops.uint32
let uint64 = Ops.uint64
let f32 = Ops.single
let bf = Ops.bfloat16
let next_first_id = ref 25000

let node prec label =
  next_first_id := !next_first_id + 1;
  Ll_test.node_factory ~prec ~first_id:!next_first_id ~dims:[| n |] () label

(* What a producer or a reader is built from, in one loop. *)
type env = { arg : int -> LL.scalar_arg; at : Idx.symbol }
type expectation = Kept | Collapsed

type shape = {
  label : string;
  scope_prec : Ops.prec;
  slot : Ops.prec;  (** the destination's, hence the reader's evaluation precision *)
  inputs : (Ops.prec * float array) list;
  value : env -> LL.scalar_t;  (** the node's setter, the scope's value *)
  reader : env -> LL.scalar_arg -> LL.scalar_t;  (** reads the node into the destination *)
  expect : expectation;
  want : float array;
}

let run shape ~form =
  let tag =
    match form with
    | `Materialized -> "mat"
    | `Oracle -> "pre"
    | `Inlined -> "inl"
    | `Scheduled -> "sch"
  in
  let slug = String.map shape.label ~f:(fun c -> if Char.is_alphanum c then c else '_') in
  let name = "isc_" ^ slug ^ "_" ^ tag in
  let inputs =
    List.mapi shape.inputs ~f:(fun k (prec, values) ->
        (node prec (Printf.sprintf "%s_in%d" name k), prec, values))
  in
  let mid = node shape.scope_prec (name ^ "_mid") and out = node shape.slot (name ^ "_out") in
  List.iter inputs ~f:(fun (tn, _, _) -> Ll_test.materialize tn);
  Ll_test.materialize out;
  let env s =
    let at = [| Ll_test.iter s |] in
    {
      arg =
        (fun k ->
          let tn, prec, _ = List.nth_exn inputs k in
          (Ll_test.get tn at, prec));
      at = s;
    }
  in
  let materialized = out :: List.map inputs ~f:(fun (tn, _, _) -> tn) in
  let store s operand = Ll_test.loop_n s n (Ll_test.set out [| Ll_test.iter s |] operand) in
  let two_loops () =
    let s1 = Ll_test.sym () and s2 = Ll_test.sym () in
    let at1 = [| Ll_test.iter s1 |] and at2 = [| Ll_test.iter s2 |] in
    LL.unflat_lines
      [
        Ll_test.loop_n s1 n (Ll_test.set mid at1 (shape.value (env s1)));
        store s2 (shape.reader (env s2) (Ll_test.get mid at2, shape.scope_prec));
      ]
  in
  let scoped () =
    let s = Ll_test.sym () in
    let at = [| Ll_test.iter s |] in
    Ll_test.materialize mid;
    Ll_test.optimize_scoped ~materialized:(mid :: materialized) ~name ~raw:(two_loops ())
      (store s
         (shape.reader (env s)
            (Ll_test.scope mid ~orig_indices:at (shape.value (env s)), shape.scope_prec)))
  in
  let o =
    match form with
    | `Materialized ->
        Ll_test.materialize mid;
        Ll_test.optimize ~materialized:(mid :: materialized) ~name (two_loops ())
    | `Inlined ->
        Ll_test.virtualize mid;
        Ll_test.optimize ~materialized ~name (two_loops ())
    | `Oracle -> scoped ()
    | `Scheduled -> (
        let o = scoped () in
        match o.LL.llc with
        | LL.For_loop { index; _ } ->
            Ir.Schedule.apply [ Ir.Schedule.Retype { axis = index; ty = LL.Serial } ] o
        | _ -> failwith "integer_scope_collapse: the scoped form is one loop")
  in
  let seed = List.map inputs ~f:(fun (tn, _, values) -> (tn, values)) in
  let values = Ll_test.execute ~name o ~seed ~read:[ out ] in
  (values, Ll_test.count_scopes o.LL.llc)

let check shape =
  let materialized, _ = run shape ~form:`Materialized in
  let oracle, _ = run shape ~form:`Oracle in
  let inlined, inlined_scopes = run shape ~form:`Inlined in
  let scheduled, scheduled_scopes = run shape ~form:`Scheduled in
  let want = [ shape.want ] in
  let scopes = match shape.expect with Kept -> 1 | Collapsed -> 0 in
  let verb = match shape.expect with Kept -> "keeps" | Collapsed -> "collapses" in
  p (shape.label ^ ": the materialized run gives the worked value") (Ll_test.same materialized want);
  p (shape.label ^ ": the uncollapsed scope gives the worked value") (Ll_test.same oracle want);
  p (shape.label ^ ": inlined, it gives the worked value") (Ll_test.same inlined want);
  p (shape.label ^ ": scheduled, it gives the worked value") (Ll_test.same scheduled want);
  p (Printf.sprintf "%s: optimize %s the scope" shape.label verb) (inlined_scopes = scopes);
  p
    (Printf.sprintf "%s: a schedule's simplify %s the scope" shape.label verb)
    (scheduled_scopes = scopes)

let bin ?(at = byte) op a b = (LL.Binop (op, a, b), at)
let un ?(at = byte) op a = (LL.Unop (op, a), at)
let const ?(at = byte) c = (LL.Constant c, at)
let plus e = LL.Binop (Ops.Add, e.arg 0, e.arg 1)
let index ?(at = byte) ?(offset = 0) e = (LL.Embed_index (Ll_test.aff [ (1, e.at) ] offset), at)
let ramp from = Array.init n ~f:(fun i -> from +. Float.of_int i)
let flat c = Array.create ~len:n c

(* The second operand of a reader is the last input. *)
let reader op k e operand = LL.Binop (op, operand, e.arg k)

let () =
  List.iter ~f:check
    [
      (* [200 + 100 = 300] holds 44, and so on: narrowing first compares 44 with 250. *)
      {
        label = "byte sum under byte Min";
        scope_prec = byte;
        slot = byte;
        inputs = [ (byte, ramp 200.); (byte, ramp 100.); (byte, flat 250.) ];
        value = plus;
        reader = reader Ops.Min 2;
        expect = Kept;
        want = [| 44.; 46.; 48.; 50. |];
      };
      {
        label = "byte sum under byte Mod";
        scope_prec = byte;
        slot = byte;
        inputs = [ (byte, ramp 200.); (byte, ramp 100.); (byte, flat 7.) ];
        value = plus;
        reader = reader Ops.Mod 2;
        expect = Kept;
        want = [| 2.; 4.; 6.; 1. |];
      };
      (* A byte difference below zero: unsigned at the store, [int] in the expression. *)
      {
        label = "negative byte difference under byte Max";
        scope_prec = byte;
        slot = byte;
        inputs = [ (byte, [| 100.; 110.; 120.; 130. |]); (byte, flat 200.); (byte, flat 1.) ];
        value = (fun e -> LL.Binop (Ops.Sub, e.arg 0, e.arg 1));
        reader = reader Ops.Max 2;
        expect = Kept;
        want = [| 156.; 166.; 176.; 186. |];
      };
      {
        label = "uint16 sum under uint16 Min";
        scope_prec = uint16;
        slot = uint16;
        inputs = [ (uint16, ramp 40000.); (uint16, ramp 30000.); (uint16, flat 60000.) ];
        value = plus;
        reader = reader Ops.Min 2;
        expect = Kept;
        want = [| 4464.; 4466.; 4468.; 4470. |];
      };
      (* Not a promotion: the slot is wider than the scope. *)
      {
        label = "uint32 sum in a uint64 slot";
        scope_prec = uint32;
        slot = uint64;
        inputs = [ (uint32, ramp 4e9); (uint32, ramp 1e9); (uint64, flat 1.) ];
        value = plus;
        reader = reader Ops.Add 2;
        expect = Kept;
        want = Array.init n ~f:(fun i -> 705032705. +. Float.of_int (2 * i));
      };
      {
        label = "int32 load narrowed to byte, read at int32";
        scope_prec = byte;
        slot = int32;
        inputs = [ (int32, ramp 300.); (int32, flat 1.) ];
        value = (fun e -> fst (e.arg 0));
        reader = reader Ops.Add 1;
        expect = Kept;
        want = [| 45.; 46.; 47.; 48. |];
      };
      {
        label = "byte sum in a float product";
        scope_prec = byte;
        slot = f32;
        inputs = [ (byte, ramp 200.); (byte, ramp 100.); (f32, flat 0.5) ];
        value = plus;
        reader = reader Ops.Mul 2;
        expect = Kept;
        want = [| 22.; 23.; 24.; 25. |];
      };
      (* No wrap at all: collapsed, the division would run in float. *)
      {
        label = "byte quotient in a float product";
        scope_prec = byte;
        slot = f32;
        inputs = [ (byte, [| 7.; 9.; 11.; 13. |]); (byte, flat 2.); (f32, flat 2.) ];
        value = (fun e -> LL.Binop (Ops.Div, e.arg 0, e.arg 1));
        reader = reader Ops.Mul 2;
        expect = Kept;
        want = [| 6.; 8.; 10.; 12. |];
      };
      (* Compared at byte [0 < 1, 1 < 1, 2 < 2, 3 < 4]; in float the third holds. *)
      {
        label = "byte comparison of float loads in a float product";
        scope_prec = byte;
        slot = f32;
        inputs =
          [
            (f32, [| 0.25; 1.5; 2.25; 3.75 |]); (f32, [| 1.75; 1.25; 2.75; 4.25 |]); (f32, flat 3.);
          ];
        value = (fun e -> LL.Binop (Ops.Cmplt, e.arg 0, e.arg 1));
        reader = reader Ops.Mul 2;
        expect = Kept;
        want = [| 3.; 0.; 0.; 3. |];
      };
      (* Neither side is exact in single: [2^24 + 1] and [2^24 + 3] round onto their neighbours. *)
      {
        label = "uint32 equality above the float limit in a float product";
        scope_prec = uint32;
        slot = f32;
        inputs =
          [
            (uint32, [| 16777217.; 16777219.; 16777219.; 16777217. |]);
            (uint32, [| 16777216.; 16777220.; 16777219.; 16777217. |]);
            (f32, flat 3.);
          ];
        value = (fun e -> LL.Binop (Ops.Cmpeq, e.arg 0, e.arg 1));
        reader = reader Ops.Mul 2;
        expect = Kept;
        want = [| 0.; 0.; 3.; 3. |];
      };
      (* The final interval is exact and fits byte; rendered at byte, [256 + i] is [i] first. *)
      {
        label = "int32 Min read at byte, an in-range result over an out-of-range leaf";
        scope_prec = int32;
        slot = byte;
        inputs = [ (byte, flat 1.) ];
        value = (fun e -> LL.Binop (Ops.Min, index ~at:int32 ~offset:256 e, const ~at:int32 5.));
        reader = reader Ops.Add 0;
        expect = Kept;
        want = flat 6.;
      };
      (* At double the comparison is always false; at byte the index converts and it holds. *)
      {
        label = "byte negated truth over an index past byte";
        scope_prec = byte;
        slot = byte;
        inputs = [ (byte, [| 0.; 1.; 2.; 3. |]); (byte, [| 1.; 2.; 3.; 4. |]) ];
        value = (fun e -> LL.Unop (Ops.Neg, bin Ops.Cmpeq (e.arg 0) (index ~offset:256 e)));
        reader = reader Ops.Min 1;
        expect = Kept;
        want = [| 1.; 2.; 3.; 4. |];
      };
      (* A truth is C [int] at any precision, so the same-type path must not take it: kept the local
         is unsigned; collapsed, [-1 < 1] compares signed. *)
      {
        label = "uint32 negated truth compared with a truth";
        scope_prec = uint32;
        slot = uint32;
        inputs =
          [ (uint32, [| 1.; 2.; 3.; 0. |]); (uint32, [| 2.; 3.; 1.; 0. |]); (uint32, flat 0.) ];
        value = (fun e -> LL.Unop (Ops.Neg, bin ~at:uint32 Ops.Cmplt (e.arg 0) (e.arg 1)));
        reader = (fun e operand -> LL.Binop (Ops.Cmplt, operand, un ~at:uint32 Ops.Not (e.arg 2)));
        expect = Kept;
        want = [| 0.; 0.; 1.; 1. |];
      };
      (* And what still collapses. Loop extents prove the sum in range. *)
      {
        label = "byte sum in range by its loop extent";
        scope_prec = byte;
        slot = byte;
        inputs = [ (byte, flat 2.) ];
        value = (fun e -> LL.Binop (Ops.Add, index e, const 3.));
        reader = reader Ops.Mul 0;
        expect = Collapsed;
        want = [| 6.; 8.; 10.; 12. |];
      };
      {
        label = "int32 sum in an int32 slot";
        scope_prec = int32;
        slot = int32;
        inputs = [ (int32, ramp 200.); (int32, ramp 100.); (int32, flat 303.) ];
        value = plus;
        reader = reader Ops.Min 2;
        expect = Collapsed;
        want = [| 300.; 302.; 303.; 303. |];
      };
      (* It wraps, but the expression has the scope's type, so it wraps the same. *)
      {
        label = "wrapping uint32 sum in a uint32 slot";
        scope_prec = uint32;
        slot = uint32;
        inputs = [ (uint32, ramp 4e9); (uint32, ramp 1e9); (uint32, flat 705032706.) ];
        value = plus;
        reader = reader Ops.Min 2;
        expect = Collapsed;
        want = [| 705032704.; 705032706.; 705032706.; 705032706. |];
      };
      {
        label = "byte load in an int32 scope in a float product";
        scope_prec = int32;
        slot = f32;
        inputs = [ (byte, ramp 200.); (f32, flat 0.5) ];
        value = (fun e -> fst (e.arg 0));
        reader = reader Ops.Mul 1;
        expect = Collapsed;
        want = [| 100.; 100.5; 101.; 101.5 |];
      };
      (* The integer-to-float arm of [widens_exactly]: every byte is a bfloat16. *)
      {
        label = "byte load in a bfloat16 scope in a single product";
        scope_prec = bf;
        slot = f32;
        inputs = [ (byte, [| 255.; 254.; 253.; 3. |]); (f32, flat 0.5) ];
        value = (fun e -> fst (e.arg 0));
        reader = reader Ops.Mul 1;
        expect = Collapsed;
        want = [| 127.5; 127.; 126.5; 1.5 |];
      };
      {
        label = "byte truth over byte loads in a float product";
        scope_prec = byte;
        slot = f32;
        inputs = [ (byte, [| 1.; 2.; 3.; 4. |]); (byte, [| 1.; 0.; 3.; 0. |]); (f32, flat 3.) ];
        value = (fun e -> LL.Binop (Ops.Cmpeq, e.arg 0, e.arg 1));
        reader = reader Ops.Mul 2;
        expect = Collapsed;
        want = [| 3.; 0.; 3.; 0. |];
      };
    ]

(* The operators the rule admits, each in range by its loop extent, at byte in a float product: the
   body renders at another precision and must compute the same. [x = i + 2], [y = i]. *)
let () =
  let x e = bin Ops.Add (index e) (const 2.) and y e = index e in
  let lt e c = bin Ops.Cmplt (y e) (const c) in
  let t b = if b then 1. else 0. in
  let row label value worked =
    let want = Array.init n ~f:(fun i -> 2. *. worked (Float.of_int i)) in
    {
      label = "admitted " ^ label;
      scope_prec = byte;
      slot = f32;
      inputs = [ (f32, flat 2.) ];
      value;
      reader = reader Ops.Mul 0;
      expect = Collapsed;
      want;
    }
  in
  let binary op e = LL.Binop (op, x e, y e) in
  List.iter ~f:check
    [
      row "Add" (binary Ops.Add) (fun i -> (2. *. i) +. 2.);
      row "Sub" (fun e -> LL.Binop (Ops.Sub, bin Ops.Add (index e) (const 3.), y e)) (fun _ -> 3.);
      row "Mul" (binary Ops.Mul) (fun i -> (i +. 2.) *. i);
      row "Max" (binary Ops.Max) (fun i -> i +. 2.);
      row "Min" (binary Ops.Min) Fn.id;
      row "Mod" (fun e -> LL.Binop (Ops.Mod, x e, const 3.)) (fun i -> Float.mod_float (i +. 2.) 3.);
      row "Neg" (fun e -> LL.Binop (Ops.Add, un Ops.Neg (y e), const 3.)) (fun i -> 3. -. i);
      row "Identity" (fun e -> LL.Unop (Ops.Identity, x e)) (fun i -> i +. 2.);
      row "Trunc" (fun e -> LL.Unop (Ops.Trunc, x e)) (fun i -> i +. 2.);
      row "Relu" (fun e -> LL.Unop (Ops.Relu, x e)) (fun i -> i +. 2.);
      row "Satur01" (fun e -> LL.Unop (Ops.Satur01, y e)) (fun i -> Float.min i 1.);
      row "Cmplt" (fun e -> LL.Binop (Ops.Cmplt, y e, const 2.)) (fun i -> t Float.(i < 2.));
      row "Cmple" (fun e -> LL.Binop (Ops.Cmple, y e, const 2.)) (fun i -> t Float.(i <= 2.));
      row "Cmpeq" (fun e -> LL.Binop (Ops.Cmpeq, y e, const 2.)) (fun i -> t Float.(i = 2.));
      row "Cmpne" (fun e -> LL.Binop (Ops.Cmpne, y e, const 2.)) (fun i -> t Float.(i <> 2.));
      row "And"
        (fun e -> LL.Binop (Ops.And, lt e 3., bin Ops.Cmplt (const 0.) (y e)))
        (fun i -> t Float.(i < 3. && i > 0.));
      row "Or"
        (fun e -> LL.Binop (Ops.Or, lt e 1., bin Ops.Cmplt (const 2.) (y e)))
        (fun i -> t Float.(i < 1. || i > 2.));
      row "Not" (fun e -> LL.Unop (Ops.Not, lt e 2.)) (fun i -> t Float.(i >= 2.));
      row "Where"
        (fun e -> LL.Ternop (Ops.Where, lt e 2., x e, y e))
        (fun i -> if Float.(i < 2.) then i +. 2. else i);
    ]

(* A scope assigned twice, [x = a + b; x = x mod 7]: substituted, the expression is in range and the
   first assignment's wrap would be gone, [300 mod 7] where the local gives [44 mod 7]. *)
let () =
  let name = "isc_two_assignments" in
  let a = node byte (name ^ "_a") and b = node byte (name ^ "_b") in
  let mid = node byte (name ^ "_mid") and out = node byte (name ^ "_out") in
  List.iter [ a; b; out ] ~f:Ll_test.materialize;
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let sum = LL.Binop (Ops.Add, (Ll_test.get a at, byte), (Ll_test.get b at, byte)) in
  let store v = Ll_test.loop_n s n (Ll_test.set out at v) in
  let id = LL.get_scope mid in
  let scoped =
    Ll_test.scope_of ~orig_indices:at id
      (LL.Seq
         ( LL.Set_local (id, sum),
           LL.Set_local (id, LL.Binop (Ops.Mod, (LL.Get_local id, byte), const 7.)) ))
  in
  let o =
    Ll_test.optimize_scoped ~materialized:[ a; b; out ] ~name
      ~raw:(store (LL.Binop (Ops.Mod, (sum, byte), const 7.)))
      (store scoped)
  in
  let sched = Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o in
  let got = Ll_test.execute ~name sched ~seed:[ (a, ramp 200.); (b, ramp 100.) ] ~read:[ out ] in
  p "two assignments: the first assignment's wrap stays" (Ll_test.same got [ [| 2.; 4.; 6.; 1. |] ]);
  p "two assignments: a schedule's simplify keeps the scope" (Ll_test.count_scopes sched.LL.llc = 1)

(* An operator's range says nothing of its domain: [FMA] has an interval and no integer spelling, so
   a byte scope over it stays a scope, refused by the renderer as the materialized node is, rather
   than collapsing into a float slot and compiling. Nothing is executed. *)
let () =
  let out = node f32 "isc_fma_out" in
  let mid = node byte "isc_fma_mid" in
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let small = (LL.Embed_index (Ll_test.iter s), byte) in
  let fma = LL.Ternop (Ops.FMA, small, small, (LL.Constant 1., byte)) in
  let llc =
    Ll_test.loop_n s n
      (Ll_test.set out at
         (LL.Binop (Ops.Mul, (Ll_test.scope mid fma, byte), (LL.Constant 2., f32))))
  in
  p "a byte scope over FMA stays a scope" (Ll_test.count_scopes (LL.simplify_llc [] llc) = 1)

(* [Ops.widens_exactly]. Both sides of the relationship read one table, which is the point, so the
   independent witnesses are what can catch a wrong endpoint. *)
let () =
  let check_pair (from, to_, expected) =
    p
      (Printf.sprintf "%s %s %s exactly" (Ops.prec_string from)
         (if expected then "widens into" else "does not widen into")
         (Ops.prec_string to_))
      (Bool.equal (Ops.widens_exactly ~from ~to_) expected)
  in
  List.iter ~f:check_pair
    [
      (byte, bf, true);
      (byte, Ops.half, true);
      (byte, Ops.fp8, false);
      (uint16, Ops.half, false);
      (uint16, bf, false);
      (uint16, f32, true);
      (uint32, f32, false);
      (int32, f32, false);
      (int32, Ops.double, true);
      (uint32, Ops.double, true);
      (Ops.int64, Ops.double, false);
      (uint64, Ops.double, false);
      (byte, int32, true);
      (uint16, uint32, true);
      (int32, uint32, false);
      (int32, uint64, false);
      (uint32, Ops.int64, true);
      (uint32, int32, false);
      (uint64, Ops.int64, false);
      (Ops.int64, uint64, false);
      (Ops.single, int32, false);
      (Ops.single, Ops.double, true);
      (Ops.uint4x32, Ops.double, false);
    ];
  let magnitude prec =
    let r = Ir.Interval.dtype_range prec in
    Float.(max (abs r.lo) r.hi)
  in
  p_all "an integer precision widens into a float exactly when its range lies within the limit"
    (List.cartesian_product Ops.integer_precs Ops.float_precs) ~f:(fun (from, to_) ->
      let limit = Option.value_exn (Ops.float_exact_int_limit to_) in
      Bool.equal (Ops.widens_exactly ~from ~to_) Float.(magnitude from <= limit))

(* The gh-343 gather still collapses. [classes = ids] is a uint32 comparison, both of its scopes
   take [ids]' precision top-down, read in a single product: the rewrite recognizes the comparison
   only once they are gone, so a scope left means no gather. *)
let () =
  let vocab = 6 and embed = 3 in
  let id_ints = [ 1; 3; 0; 5 ] in
  let table_values = Array.init (embed * vocab) ~f:Float.of_int in
  let ids = Nn_blocks.class_ids_of_int_list id_ints in
  let table =
    TDSL.ndarray table_values ~label:[ "isc_table" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let embedded =
    let open TDSL.O in
    table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids
  in
  let ctx = Train.forward_once (Context.cpu ()) embedded in
  let got = Context.get_values ctx embedded.Tensor.value in
  let expected =
    Array.concat_map (Array.of_list id_ints) ~f:(fun idx ->
        Array.init embed ~f:(fun o -> Float.of_int ((o * vocab) + idx)))
  in
  let optim_ctx = LL.empty_optimize_ctx () in
  let lowered =
    Ir.Assignments.lower optim_ctx ~unoptim_ll_source:None ~ll_source:None ~cd_source:None
      ~name:"isc_gather_probe" [] embedded.Tensor.forward.Ir.Assignments.asgns
  in
  let gathers = ref 0 in
  Ll_test.walk lowered.LL.llc ~on_scalar:(function LL.Get_dynamic _ -> Int.incr gathers | _ -> ());
  p "the one-hot gather collapses its scopes" (Ll_test.count_scopes lowered.LL.llc = 0);
  p "the one-hot gather is rewritten" (!gathers > 0);
  p "the gather computes the table's rows" (Ll_test.close got expected)
