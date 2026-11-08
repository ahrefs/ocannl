(* gh-ocannl-2052: scope-minting operand walks follow program order, left to right. Relative ids pin
   minting order, and definition/read positions pin the CSE survivor. The optimization rows also
   check executed values against a discriminating closed form. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:1_974_000 ~dims:[| 3 |] ()
let n = 3
let weights = [| 2.; 3.; 5. |]

(* [d[i] = w[i] * k + k], in two accumulations over a zeroed node: a scope body too long to collapse
   into its use. A different [k] makes two producers' scopes differ; one [k], alpha-equal. *)
let producer d ~w ~k =
  let s = sym () and t = sym () in
  LL.unflat_lines
    [
      zero d;
      loop_n s n (set_at d (iter s) (add (get d [| iter s |]) (mul (get w [| iter s |]) (c k))));
      loop_n t n (set_at d (iter t) (add (get d [| iter t |]) (c k)));
    ]

let value ~k i = (weights.(i) *. k) +. k

(* The scopes and scope reads among the operands of [out]'s statement, in operand order, outside any
   scope body: [`Def] for a definition, [`Read] for a [Get_local]. *)
let operands ~guards (o : LL.optimized) out =
  let found = ref [] in
  let collect llsc =
    walk_scalar llsc ~in_scopes:false ~on_scalar:(function
      | LL.Local_scope { id; _ } -> found := (`Def, id) :: !found
      | LL.Get_local id -> found := (`Read, id) :: !found
      | _ -> ())
  in
  walk o.LL.llc ~in_scopes:false ~on_stmt:(function
    | LL.If { cond = c, _; _ } when guards -> collect c
    | LL.Set { tn; llsc; _ } when Ir.Tnode.equal tn out -> collect llsc
    | _ -> ());
  List.rev !found

let label (id : LL.scope_id) = String.concat id.tn.Ir.Tnode.label

let show ops =
  String.concat ~sep:" "
    (List.map ops ~f:(fun (kind, id) ->
         (match kind with `Def -> "def:" | `Read -> "read:") ^ label id))

(* The producers' labels by ascending scope id: the order the scopes were minted in. *)
let minted ops =
  List.filter_map ops ~f:(function `Def, id -> Some id | `Read, _ -> None)
  |> List.sort ~compare:(fun (a : LL.scope_id) b -> Int.compare a.scope_id b.scope_id)
  |> List.map ~f:label |> String.concat ~sep:" "

(* Optimizes [stmt] over [producers], prints what the walk left, and runs it against [closed]. *)
let row ?(guards = false) name ~producers ~out ~w ~closed stmt =
  List.iter [ w; out ] ~f:materialize;
  let llc = LL.unflat_lines (List.map producers ~f:(fun (d, k) -> producer d ~w ~k) @ [ stmt ]) in
  let o = optimize ~name llc in
  p_all ~min:1 (name ^ ": every producer virtualizes") producers ~f:(fun (d, _) ->
      known_virtual o d);
  let ops = operands ~guards o out in
  Stdio.printf "%s: operands %s; minted %s\n" name (show ops) (minted ops);
  let got = execute ~name o ~seed:[ (w, weights); (out, blank n) ] ~read:[ out ] in
  p (name ^ ": the run gives the closed form") (same got [ Array.init n ~f:closed ]);
  ops

let ids ops = List.map ops ~f:(fun (_, (id : LL.scope_id)) -> id.scope_id)
let ascending l = List.is_sorted_strictly l ~compare:Int.compare

(* [out = a + b]: [a]'s scope is minted first. *)
let () =
  let name = "owo_binop" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let a = mk "a" and b = mk "b" in
  let t = sym () in
  let ops =
    row name
      ~producers:[ (a, 2.); (b, 3.) ]
      ~out ~w
      ~closed:(fun i -> value ~k:2. i +. value ~k:3. i)
      (loop_n t n (set_at out (iter t) (add (get a [| iter t |]) (get b [| iter t |]))))
  in
  p
    (name ^ ": two scopes, the first operand's minted first")
    (List.length ops = 2 && ascending (ids ops))

(* [out = fma(a, b, c)]: minted [a], [b], [c]. *)
let () =
  let name = "owo_ternop" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let a = mk "a" and b = mk "b" and c_ = mk "c" in
  let t = sym () in
  let at d = (get d [| iter t |], single) in
  let ops =
    row name
      ~producers:[ (a, 2.); (b, 3.); (c_, 7.) ]
      ~out ~w
      ~closed:(fun i -> (value ~k:2. i *. value ~k:3. i) +. value ~k:7. i)
      (loop_n t n (set_at out (iter t) (LL.Ternop (Ir.Ops.FMA, at a, at b, at c_))))
  in
  p (name ^ ": three scopes, minted left to right") (List.length ops = 3 && ascending (ids ops))

(* An undecided data condition keeps both [Where] arms, minted after the condition. *)
let () =
  let name = "owo_where" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let cond = mk "cond" and a = mk "a" and b = mk "b" in
  let t = sym () in
  let at d = (get d [| iter t |], single) in
  let ops =
    row name
      ~producers:[ (cond, 7.); (a, 2.); (b, 3.) ]
      ~out ~w ~closed:(value ~k:2.)
      (loop_n t n (set_at out (iter t) (LL.Ternop (Ir.Ops.Where, at cond, at a, at b))))
  in
  p
    (name ^ ": three scopes, condition then arms left to right")
    (List.length ops = 3 && ascending (ids ops))

(* The guard's scope precedes the scope in its guarded store. *)
let () =
  let name = "owo_if" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let a = mk "a" and b = mk "b" in
  let t = sym () in
  let ops =
    row ~guards:true name
      ~producers:[ (a, 2.); (b, 3.) ]
      ~out ~w ~closed:(value ~k:3.)
      (loop_n t n
         (LL.If
            { cond = (get a [| iter t |], single); body = set_at out (iter t) (get b [| iter t |]) }))
  in
  p (name ^ ": two scopes, condition before body") (List.length ops = 2 && ascending (ids ops))

(* [out = a * a]: the CSE keeps the first operand's copy, and the second reads it. *)
let () =
  let name = "owo_cse" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let a = mk "a" in
  let t = sym () in
  let ops =
    row name
      ~producers:[ (a, 2.) ]
      ~out ~w
      ~closed:(fun i -> value ~k:2. i *. value ~k:2. i)
      (loop_n t n (set_at out (iter t) (mul (get a [| iter t |]) (get a [| iter t |]))))
  in
  p
    (name ^ ": the second operand reads the scope the first defines")
    (match ops with [ (`Def, d); (`Read, r) ] -> LL.equal_scope_id r d | _ -> false)

(* [out = (a + b) + a]: of [a]'s two copies the first one survives, and [b], walked between them,
   sits between their ids. *)
let () =
  let name = "owo_cse_between" in
  let w = mk (name ^ "_w") and out = mk (name ^ "_out") in
  let a = mk "a" and b = mk "b" in
  let t = sym () in
  let at d = get d [| iter t |] in
  let ops =
    row name
      ~producers:[ (a, 2.); (b, 3.) ]
      ~out ~w
      ~closed:(fun i -> value ~k:2. i +. value ~k:3. i +. value ~k:2. i)
      (loop_n t n (set_at out (iter t) (add (add (at a) (at b)) (at a))))
  in
  p
    (name ^ ": the first copy survives, read by the last; the middle operand is minted after it")
    (match ops with
    | [ (`Def, a_id); (`Def, b_id); (`Read, r) ] ->
        LL.equal_scope_id r a_id && a_id.scope_id < b_id.scope_id
    | _ -> false)

(* Instantiation freshens loops inside nested operand scopes. Their original symbol ids may be in
   any order. The public synthetic-read seam uses decreasing negative symbols, so replayed operand
   bindings must decrease in tree order. *)
let freshening_row name ~arity ~rhs =
  let self = mk name in
  let cell = [| fixed 0 |] in
  let placements = Ir.Tnode.Placements.create () in
  let args =
    List.init arity ~f:(fun k ->
        let tn = mk (name ^ Int.to_string k) in
        Ir.Tnode.Placements.update placements tn Virtual Scope_local;
        let id = LL.get_scope tn in
        let i = sym () in
        scope_of ~orig_indices:cell id
          (loop_n i (k + 2) (LL.Set_local (id, LL.Embed_index (iter i)))))
  in
  let body =
    LL.instantiate_at_synthetic_read ~placements ~static_indices:[] self
      [ (Some cell, set self cell (rhs args)) ]
    |> Result.ok_or_failwith
  in
  let bindings = ref [] in
  walk body ~on_stmt:(function
    | LL.For_loop { index; _ } -> bindings := index :: !bindings
    | _ -> ());
  let bindings = List.rev !bindings in
  p
    (name ^ ": nested loop bindings are freshened left to right")
    (List.length bindings = arity
    && List.is_sorted_strictly bindings ~compare:(fun a b -> Ir.Indexing.Symbol.compare b a))

let () =
  freshening_row "owo_instantiate_binop" ~arity:2 ~rhs:(function
    | [ a; b ] -> add a b
    | _ -> assert false);
  freshening_row "owo_instantiate_ternop" ~arity:3 ~rhs:(function
    | [ a; b; c ] -> LL.Ternop (Ir.Ops.FMA, (a, single), (b, single), (c, single))
    | _ -> assert false)
