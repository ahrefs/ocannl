(* gh-ocannl-1974: the passes that mint or dedup scopes walk a statement's operands in a spelled
   order, last operand first.

   [Low_level.virtual_llc] mints a scope at each inlined read and the scalar CSE keeps one of a
   statement's alpha-equal scopes, so the order either walks the operands in numbers the scopes and
   picks the surviving copy -- and [C_syntax.pp_local_defs] emits definitions by ascending id, so it
   is the order of the emitted text. Both used to construct [Binop (op, (loop a, _), (loop b, _))],
   whose evaluation order OCaml leaves unspecified; in practice it ran right to left, and that is
   the order now written out with [let] bindings, so no kernel moved.

   Every row reads scope ids relative to each other, never by their value: the counter is
   process-wide. Reversing one of the spelled orders turns the matching row red. *)

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
let operands (o : LL.optimized) out =
  let found = ref [] in
  walk o.LL.llc ~in_scopes:false ~on_stmt:(function
    | LL.Set { tn; llsc; _ } when Ir.Tnode.equal tn out ->
        walk_scalar llsc ~in_scopes:false ~on_scalar:(function
          | LL.Local_scope { id; _ } -> found := (`Def, id) :: !found
          | LL.Get_local id -> found := (`Read, id) :: !found
          | _ -> ())
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
let row name ~producers ~out ~w ~closed stmt =
  List.iter [ w; out ] ~f:materialize;
  let llc = LL.unflat_lines (List.map producers ~f:(fun (d, k) -> producer d ~w ~k) @ [ stmt ]) in
  let o = optimize ~name llc in
  p_all ~min:1 (name ^ ": every producer virtualizes") producers ~f:(fun (d, _) ->
      known_virtual o d);
  let ops = operands o out in
  Stdio.printf "%s: operands %s; minted %s\n" name (show ops) (minted ops);
  let got = execute ~name o ~seed:[ (w, weights); (out, blank n) ] ~read:[ out ] in
  p (name ^ ": the run gives the closed form") (same got [ Array.init n ~f:closed ]);
  ops

let ids ops = List.map ops ~f:(fun (_, (id : LL.scope_id)) -> id.scope_id)
let descending l = List.is_sorted_strictly l ~compare:(fun a b -> Int.compare b a)

(* [out = a + b]: [b]'s scope is minted first. *)
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
    (name ^ ": two scopes, the second operand's minted first")
    (List.length ops = 2 && descending (ids ops))

(* [out = fma(a, b, c)]: minted [c], [b], [a]. *)
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
  p
    (name ^ ": three scopes, minted last operand first")
    (List.length ops = 3 && descending (ids ops))

(* [out = a * a]: the CSE keeps the second operand's copy, and the first reads it. *)
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
    (name ^ ": the first operand reads the scope the second defines")
    (match ops with [ (`Read, r); (`Def, d) ] -> LL.equal_scope_id r d | _ -> false)

(* [out = (a + b) + a]: of [a]'s two copies the last one survives, and [b], walked between them,
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
    (name ^ ": the last copy survives, read by the first; the middle operand is minted after it")
    (match ops with
    | [ (`Read, r); (`Def, b_id); (`Def, a_id) ] ->
        LL.equal_scope_id r a_id && a_id.scope_id < b_id.scope_id
    | _ -> false)
