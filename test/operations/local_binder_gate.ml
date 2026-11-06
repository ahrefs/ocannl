(* gh-ocannl-1973: the compile gate that every scope local a routine reads or writes has a binder.

   A local nothing binds renders as an identifier nothing declares, and before the gate the first to
   say so was the backend's compiler, about a kernel the user never wrote. Each refused row here is
   such a routine, hand-built in the post-optimize form and compiled through the [?prelowered] seam,
   so the refusal is [C_syntax.compile_proc]'s own; beside it runs the same routine with its binder,
   executed, since the gate is only worth having if it leaves valid code alone.

   The gate asks whether a binder exists, never whether it is emitted ahead of the read: the rows of
   [local_scope_sibling_reads] are the routines an ordering rule would refuse, and they compile
   under it. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:1_973_000 ~dims:[| 1 |] ()
let cell = [| fixed 0 |]

(* [out = a + 1] over a virtual [a = 5]: the twin whose optimization supplies the traced store and
   the placements of every row's hand-scoped routine. *)
let program name =
  let a = mk (name ^ "_a") and b = mk (name ^ "_b") and out = mk (name ^ "_out") in
  List.iter [ a; b ] ~f:virtualize;
  materialize out;
  let raw =
    LL.unflat_lines
      [
        set a cell (c 5.);
        set b cell (add (get a cell) (c 1.));
        set out cell (add (get a cell) (get b cell));
      ]
  in
  (a, b, out, raw)

(* The gate's spelling of a local over the node labelled [label]. *)
let local (id : LL.scope_id) label = "v" ^ Int.to_string id.scope_id ^ "_" ^ label

(* Compiles [scoped] and claims the gate refused it, naming each of [naming]. The text goes to
   stderr: it spells scope ids, which are minted from a process-wide counter. *)
let refused name ~raw ~naming scoped =
  let o = optimize_scoped ~name ~raw scoped in
  let msg = match link ~name o with _ -> "compiled" | exception exn -> Exn.to_string exn in
  Stdio.eprintf "%s (not part of the golden): %s\n%!" name msg;
  p
    (name ^ ": the compile gate refuses it")
    (String.is_substring msg ~substring:"C_syntax.compile_proc: a ");
  p_all ~min:1 (name ^ ": the refusal names the access, the local and where it sits") naming
    ~f:(fun substring -> String.is_substring msg ~substring)

(* Compiles and runs [scoped], claiming the gate let it through and [out] holds [want]. *)
let accepted name ~raw ~out ~want scoped =
  let o = optimize_scoped ~name ~raw scoped in
  let got = execute ~name o ~seed:[ (out, blank 1) ] ~read:[ out ] in
  p (name ^ ": the gate leaves it alone, and it runs") (same got [ [| want |] ])

(* A read at statement level: the hole the cross-statement hoist left (gh-ocannl-1895), where a
   remap missed a reader of the scope it had lifted and renamed. *)
let row_statement () =
  let name = "lbg_statement" in
  let a, _, out, raw = program name in
  let ia = LL.get_scope a in
  let read = set out cell (add (LL.Get_local ia) (c 1.)) in
  refused name ~raw read ~naming:[ "Get_local of " ^ local ia (name ^ "_a"); "at statement level" ];
  accepted (name ^ "_bound") ~raw ~out ~want:6.
    (LL.unflat_lines [ declare_local ia; LL.Set_local (ia, c 5.); read ])

(* A read inside a scope body: the refusal names the body and the pass that minted its scope. *)
let row_body () =
  let name = "lbg_body" in
  let a, b, out, raw = program name in
  let ia = LL.get_scope a and ib = LL.get_scope b in
  let reader = scope ~orig_indices:cell ~id:ib b (add (LL.Get_local ia) (c 1.)) in
  refused name ~raw
    (set out cell (add (c 5.) reader))
    ~naming:
      [
        "Get_local of " ^ local ia (name ^ "_a");
        "inside the body of local scope " ^ local ib (name ^ "_b");
        "(Inlined_computation)";
      ];
  accepted (name ^ "_bound") ~raw ~out ~want:11.
    (set out cell (add (scope ~orig_indices:cell ~id:ia a (c 5.)) reader))

(* A binder that shares the read's integer but names another node declares a different C variable:
   codegen spells a local from both. *)
let row_other_node () =
  let name = "lbg_other_node" in
  let a, b, out, raw = program name in
  let ia = LL.get_scope a in
  let stray = { ia with LL.tn = b } in
  refused name ~raw
    (set out cell (add (scope ~orig_indices:cell ~id:ia a (c 5.)) (LL.Get_local stray)))
    ~naming:
      [
        "Get_local of " ^ local stray (name ^ "_b"); local ia (name ^ "_a") ^ " shares its integer";
      ]

(* A write strands the same way as a read. *)
let row_write () =
  let name = "lbg_write" in
  let a, _, out, raw = program name in
  let ia = LL.get_scope a in
  let body = [ LL.Set_local (ia, c 5.); set out cell (LL.Get_local ia) ] in
  refused name ~raw (LL.unflat_lines body)
    ~naming:[ "Set_local of " ^ local ia (name ^ "_a"); "at statement level" ];
  accepted (name ^ "_bound") ~raw ~out ~want:5. (LL.unflat_lines (declare_local ia :: body))

(* A projection's discarded operand is never rendered. A read there strands nothing, so the gate
   leaves it alone; a binder there declares nothing, so a read of it elsewhere is stranded. *)
let row_discarded () =
  let name = "lbg_discarded" in
  let a, _, out, raw = program name in
  let ia = LL.get_scope a in
  let arg1 x y = LL.Binop (Ops.Arg1, (x, single), (y, single)) in
  accepted (name ^ "_read") ~raw ~out ~want:7. (set out cell (arg1 (c 7.) (LL.Get_local ia)));
  refused name ~raw
    (set out cell (add (arg1 (c 7.) (scope ~orig_indices:cell ~id:ia a (c 5.))) (LL.Get_local ia)))
    ~naming:[ "Get_local of " ^ local ia (name ^ "_a"); "at statement level" ]

let () =
  case "statement" row_statement;
  case "body" row_body;
  case "other_node" row_other_node;
  case "write" row_write;
  case "discarded" row_discarded;
  Stdio.printf "%!"
