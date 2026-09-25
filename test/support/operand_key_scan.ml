(** The reader behind [test/operations/operand_key_ratchet] (gh-ocannl-1018): operand fixtures that
    mint cell values from a hand-rolled modulus of a multi-axis key, bypassing {!Ll_test.cycle}'s
    blind-axis guard.

    The guard reaches only its callers. A site that spells the key itself —
    [~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5))] — computes the same
    arithmetic with nobody checking it, and that one was constant along axis 0 (20 mod 5 = 0), the
    fifth sighting of the trap gh-ocannl-640 named. A sweep did not end the series because the two
    spellings of the key do not read as the same thing to a person scanning a file; this reads both
    through one detector.

    {1 What it reads}

    The parse tree (ppxlib's, like every scan here), never the text: comments and string literals
    are not code, and a closure spread over several lines is one expression. A {e site} is an
    application of an integer-remainder operator — any identifier whose last component is [%], [mod]
    or [rem], so [Int.( % )], [Int.rem] and [Stdlib.( mod )] too — inside the body of a function,
    whose LEFT operand reads that function's parameter in one of two spellings:

    - {!Multi_index}: the parameter is an index array and the left operand reads two or more
      distinct axes of it ([v.(0)], [Array.get v 1]), or one axis through a non-literal index. Names
      [let]-bound inside the body to an expression that reads axes are followed:
      [let flat = (idcs.(0) * n) + idcs.(1) in … flat % 7] is a site. Any function counts, not only
      an [~f] argument, so [let value idcs = …] is read too.
    - {!Flat}: the function is the [~f] of [Array.init] / [List.init] (or the positional function of
      [Stdlib.Array.init]) whose length is written as a product at the call,
      [Array.init (m * k) ~f:(fun i -> …)], and the left operand mentions the flat parameter itself.
      A right operand that is syntactically the product of a TRAILING run of the length's factors is
      NOT a site: [i % cols] beside [i / cols] over [Array.init (rows * cols)], or [idx % (n * m)]
      over [Array.init (b * n * m)], is unflattening the index, the correct idiom, not minting a
      cycle.

    {1 What it deliberately does not read}

    Each of these is a boundary, stated so a miss beyond it is a known deferral rather than a silent
    hole:

    - A modulus of ONE literal axis ([idcs.(0) % 3], [if idcs.(1) % bm = 0 then …]): the value omits
      the other axes in plain sight, so nothing arithmetic hides the blindness. The flat spelling's
      factor exclusion is the same boundary seen from the other side.
    - A key computed OUTSIDE the function: [Array.init (n * n) ~f:value] with [value] defined
      elsewhere, a helper called as [~f:(fun idcs -> value idcs.(0) idcs.(1))] whose modulus lives
      in [value], or a flat length bound to a name before the call
      ([let len = m * k in Array.init len …]).
    - Aliases other than a plain [let x = …] inside the body — tuple and array patterns, [match],
      references — and a parameter bound by a pattern rather than a name ([fun [| i; j |] -> …]).
    - Shadowing: a name that rebinds the parameter inside the body is still read as the parameter.
    - Any value mixer other than the remainder operators (a hash, [land], a multiply-shift).

    The detector does not decide whether a site is blind; that is {!Ll_test.cycle}'s job, and the
    whole point is to route the arithmetic through it. A site is either converted (onto [cycle],
    [cycle_flat], [weighted], or a helper that calls them) or named in the ratchet's exemption list
    with the reason it is not an operand the guard should check. *)

open Base
open Ppxlib

type spelling = Multi_index | Flat
type site = { line : int; spelling : spelling; key : string }

let spelling_name = function Multi_index -> "multi-index" | Flat -> "flat"

(* An axis read of the parameter: a literal axis, or an index the scan cannot evaluate. *)
type read = Axis of int | Dynamic_axis | Whole

let remainder_op (e : expression) =
  match e.pexp_desc with
  | Pexp_ident { txt; _ } -> (
      match Longident.last_exn txt with "%" | "mod" | "rem" -> true | _ -> false)
  | _ -> false

let is_ident name (e : expression) =
  match e.pexp_desc with Pexp_ident { txt = Lident n; _ } -> String.equal n name | _ -> false

let array_get (e : expression) =
  match e.pexp_desc with
  | Pexp_apply
      ( { pexp_desc = Pexp_ident { txt = Ldot (Lident "Array", ("get" | "unsafe_get")); _ }; _ },
        [ (Nolabel, arr); (Nolabel, idx) ] ) ->
      Some (arr, idx)
  | _ -> None

(* Every read of [param] in [e], following [env]'s let-bound names. Shadowing of [param] by an inner
   binder is not tracked: a site found through a shadowed name reads an array of the same spelling,
   and the ratchet's exemption list is the place to say otherwise. *)
let reads ~param ~env e =
  let acc = ref [] in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        match array_get e with
        | Some (arr, idx) when is_ident param arr ->
            (match idx.pexp_desc with
            | Pexp_constant (Pconst_integer (k, None)) -> acc := Axis (Int.of_string k) :: !acc
            | _ -> acc := Dynamic_axis :: !acc);
            super#expression idx
        | _ -> (
            match e.pexp_desc with
            | Pexp_ident { txt = Lident n; _ } when String.equal n param -> acc := Whole :: !acc
            | Pexp_ident { txt = Lident n; _ } -> (
                match Map.find env n with Some rs -> acc := rs @ !acc | None -> ())
            | _ -> super#expression e)
    end
  in
  walker#expression e;
  !acc

let multi_axis reads =
  List.exists reads ~f:(function Dynamic_axis -> true | _ -> false)
  || List.length
       (List.dedup_and_sort ~compare:Int.compare
          (List.filter_map reads ~f:(function Axis k -> Some k | _ -> None)))
     >= 2

let source_key source (e : expression) =
  let start = e.pexp_loc.loc_start.pos_cnum and stop = e.pexp_loc.loc_end.pos_cnum in
  String.sub source ~pos:start ~len:(stop - start)
  |> String.split_on_chars ~on:[ ' '; '\n'; '\t'; '\r' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

(* The structural identity of an expression, locations ignored: what a factor and a divisor are
   compared by. *)
let shape (e : expression) = Stdlib.Format.asprintf "%a" Pprintast.expression e

let rec product_factors (e : expression) =
  match e.pexp_desc with
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "*"; _ }; _ }, [ (_, a); (_, b) ]) ->
      product_factors a @ product_factors b
  | _ -> [ e ]

(* The single plain-variable parameter and body of a one-argument function, if [e] is one. *)
let unary_function (e : expression) =
  match e.pexp_desc with
  | Pexp_function
      ( [
          {
            pparam_desc =
              Pparam_val
                ( Nolabel,
                  None,
                  ( { ppat_desc = Ppat_var { txt; _ }; _ }
                  | { ppat_desc = Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _); _ }
                    ) );
            _;
          };
        ],
        _,
        Pfunction_body body ) ->
      Some (txt, body)
  | _ -> None

(* Every parameter bound to a plain variable, with the function's body, for any function. *)
let named_params (e : expression) =
  match e.pexp_desc with
  | Pexp_function (params, _, Pfunction_body body) ->
      List.filter_map params ~f:(fun p ->
          match p.pparam_desc with
          | Pparam_val (_, _, { ppat_desc = Ppat_var { txt; _ }; _ })
          | Pparam_val
              (_, _, { ppat_desc = Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _); _ })
            ->
              Some (txt, body)
          | _ -> None)
  | _ -> []

(* The remainder applications in [body] whose left operand reads [param], with the reads. [let]
   bindings extend the alias environment for their body. *)
let remainders ~param body =
  let found = ref [] in
  let rec go env (e : expression) =
    match e.pexp_desc with
    | Pexp_apply (op, [ (Nolabel, lhs); (Nolabel, rhs) ]) when remainder_op op ->
        let rs = reads ~param ~env lhs in
        if not (List.is_empty rs) then found := (e, rs, rhs) :: !found;
        go env lhs;
        go env rhs
    | Pexp_let (_, bindings, let_body) ->
        List.iter bindings ~f:(fun vb -> go env vb.pvb_expr);
        let env =
          List.fold bindings ~init:env ~f:(fun env vb ->
              match vb.pvb_pat.ppat_desc with
              | Ppat_var { txt; _ } -> (
                  match reads ~param ~env vb.pvb_expr with
                  | [] -> env
                  | rs -> Map.set env ~key:txt ~data:rs)
              | _ -> env)
        in
        go env let_body
    | _ ->
        let walker =
          object
            inherit Ast_traverse.iter
            method! expression e' = go env e'
          end
        in
        (* Descend one level through the generic traversal, re-entering [go] on each child. *)
        walker#expression_desc e.pexp_desc
  in
  go (Map.empty (module String)) body;
  List.rev !found

let init_function (e : expression) =
  let init_ident (f : expression) =
    match f.pexp_desc with
    | Pexp_ident { txt = Ldot ((Lident m | Ldot (_, m)), "init"); _ } -> (
        match m with "Array" | "List" -> true | _ -> false)
    | _ -> false
  in
  match e.pexp_desc with
  | Pexp_apply (f, args) when init_ident f -> (
      let positional = List.filter_map args ~f:(function Nolabel, a -> Some a | _ -> None) in
      let labelled_f = List.find_map args ~f:(function Labelled "f", a -> Some a | _ -> None) in
      match (positional, labelled_f) with
      | len :: _, Some fn | [ len; fn ], None -> Some (len, fn)
      | _ -> None)
  | _ -> None

let sites source =
  let found = Hashtbl.create (module Int) in
  let record spelling (e : expression) =
    let offset = e.pexp_loc.loc_start.pos_cnum in
    if not (Hashtbl.mem found offset) then
      Hashtbl.set found ~key:offset
        ~data:{ line = e.pexp_loc.loc_start.pos_lnum; spelling; key = source_key source e }
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        List.iter (named_params e) ~f:(fun (param, body) ->
            List.iter (remainders ~param body) ~f:(fun (site, rs, _) ->
                if multi_axis rs then record Multi_index site));
        (match init_function e with
        | Some (len, fn) -> (
            let factors = product_factors len in
            match unary_function fn with
            | Some (param, body) when List.length factors >= 2 ->
                let factor_shapes = List.map factors ~f:shape in
                (* [i % d] with [d] the product of a TRAILING run of the length's factors is the
                   unflattening idiom: the offset within the innermost axes. *)
                let unflattens rhs =
                  let divisor = List.map (product_factors rhs) ~f:shape in
                  let n = List.length factor_shapes and k = List.length divisor in
                  k < n && List.equal String.equal (List.drop factor_shapes (n - k)) divisor
                in
                List.iter (remainders ~param body) ~f:(fun (site, rs, rhs) ->
                    if
                      List.exists rs ~f:(function Whole -> true | _ -> false)
                      && not (unflattens rhs)
                    then record Flat site)
            | _ -> ())
        | None -> ());
        super#expression e
    end
  in
  walker#structure (Parse.implementation (Lexing.from_string source));
  Hashtbl.data found |> List.sort ~compare:(fun a b -> Int.compare a.line b.line)

type exemption = Site of string | File

(** [violations ~exemptions rows] matches each site against the exemptions. A [Site key] row absorbs
    ONE site of its file whose whitespace-collapsed source text is [key] (a multiset, so two
    identical sites need two rows); a [File] row absorbs every site of its file. A site no row
    absorbs is a refusal, and so is a row that absorbs nothing — stale, because the site was
    converted or rewritten and the row has to go with it. *)
let violations ~exemptions rows =
  let used = Array.create ~len:(List.length exemptions) false in
  let indexed = List.mapi exemptions ~f:(fun i row -> (i, row)) in
  let take path key =
    match
      List.find indexed ~f:(fun (i, (p, kind, _)) ->
          String.equal p path
          && match kind with File -> true | Site k -> (not used.(i)) && String.equal k key)
    with
    | Some (i, _) ->
        used.(i) <- true;
        true
    | None -> false
  in
  let missing =
    List.concat_map rows ~f:(fun (path, sites) ->
        List.filter_map sites ~f:(fun site ->
            if take path site.key then None
            else
              Some
                (Printf.sprintf
                   "%s:%d: %s operand key `%s` is hand-rolled; mint it through Ll_test.cycle, \
                    cycle_flat or weighted so the blind-axis guard checks it, or name it in the \
                    exemption list with the reason it is not an operand"
                   path site.line (spelling_name site.spelling) site.key)))
  in
  let stale =
    List.filter_map indexed ~f:(fun (i, (path, kind, _)) ->
        if used.(i) then None
        else
          Some
            (match kind with
            | File -> Printf.sprintf "%s: stale operand-key file exemption" path
            | Site key -> Printf.sprintf "%s: stale operand-key exemption `%s`" path key))
  in
  missing @ stale
