(* gh-ocannl-1973: the compile gate that every scope local a routine reads or writes has a binder.

   A local nothing binds renders as an identifier nothing declares, and before the gate the first to
   say so was the backend's compiler, about a kernel the user never wrote. Each refused row here is
   such a routine, hand-built in the post-optimize form and compiled through the [?prelowered] seam,
   so the refusal is [C_syntax.compile_proc]'s own; beside it runs the same routine with its binder,
   executed, since the gate is only worth having if it leaves valid code alone.

   The gate checks lexical visibility for explicit declarations and scan state, but does not order
   hoisted Local_scope definitions: the rows of [local_scope_sibling_reads] are the routines an
   ordering rule would refuse, and they compile under it. *)

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
  let msg = match link ~name o with _ -> "compiled" | exception Invalid_argument msg -> msg in
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

(* gh-ocannl-2050: explicit declarations and scan state cannot escape their blocks. *)
let row_lexical name block =
  let a, _, out, raw = program name in
  let ia = LL.get_scope a in
  let define = LL.unflat_lines [ declare_local ia; LL.Set_local (ia, c 5.) ] in
  let read = set out cell (LL.Get_local ia) in
  refused name ~raw
    (seq (block define) read)
    ~naming:[ "Get_local of " ^ local ia (name ^ "_a"); "outside its lexical scope" ];
  refused (name ^ "_write") ~raw
    (seq (block define) (LL.Set_local (ia, c 9.)))
    ~naming:[ "Set_local of " ^ local ia (name ^ "_a"); "outside its lexical scope" ];
  refused (name ^ "_sibling") ~raw
    (seq (block define) (block read))
    ~naming:[ "Get_local of " ^ local ia (name ^ "_a"); "outside its lexical scope" ];
  accepted (name ^ "_inside") ~raw ~out ~want:5. (block (seq define read));
  accepted (name ^ "_outer") ~raw ~out ~want:5. (seq define (block read))

let row_guard_condition () =
  let name = "lbg_guard_condition" in
  let a, _, out, raw = program name in
  let ia = LL.get_scope a in
  refused name ~raw
    (if_ (LL.Get_local ia)
       (LL.unflat_lines
          [ declare_local ia; LL.Set_local (ia, c 5.); set out cell (LL.Get_local ia) ]))
    ~naming:[ "Get_local of " ^ local ia (name ^ "_a"); "outside its lexical scope" ]

let row_scan_escape () =
  let name = "lbg_scan_escape" in
  let a, _, out, raw = program name in
  let cr = carry ~init:(c 5.) a in
  let body = set_next cr (add (prev cr) (c 1.)) in
  let block body = scan ~upto:1 (sym ()) ~carried:[ cr ] body in
  refused name ~raw
    (seq (block body) (set out cell (prev cr)))
    ~naming:[ "Get_local of " ^ local cr.prev (name ^ "_a"); "outside its lexical scope" ];
  accepted (name ^ "_inside") ~raw ~out ~want:7. (block (seq body (set out cell (next cr))))

let row_scan_init () =
  let name = "lbg_scan_init" in
  let a, b, _, raw = program name in
  let ib = LL.get_scope b in
  let cr = carry ~init:(LL.Get_local ib) a in
  let body = LL.unflat_lines [ declare_local ib; LL.Set_local (ib, c 5.); set_next cr (prev cr) ] in
  refused name ~raw
    (scan ~upto:1 (sym ()) ~carried:[ cr ] body)
    ~naming:[ "Get_local of " ^ local ib (name ^ "_b"); "outside its lexical scope" ]

let row_scan_init_local () =
  let name = "lbg_scan_init_local" in
  let a, b, out, raw = program name in
  let ib = LL.get_scope b and init_id = LL.get_scope b in
  let init =
    scope_of init_id
      (LL.unflat_lines
         [ declare_local ib; LL.Set_local (ib, c 5.); LL.Set_local (init_id, LL.Get_local ib) ])
  in
  let cr = carry ~init a in
  let block body = scan ~upto:1 (sym ()) ~carried:[ cr ] body in
  refused name ~raw
    (block (set_next cr (LL.Get_local ib)))
    ~naming:[ "Get_local of " ^ local ib (name ^ "_b"); "outside its lexical scope" ];
  accepted (name ^ "_inside") ~raw ~out ~want:7.
    (block (seq (set_next cr (add (prev cr) (c 1.))) (set out cell (next cr))))

let () =
  case "loop_escape" (fun () -> row_lexical "lbg_loop_escape" (loop ~upto:1 (sym ())));
  case "guard_escape" (fun () -> row_lexical "lbg_guard_escape" (if_ (c 1.)));
  case "guard_condition" row_guard_condition;
  case "scan_escape" row_scan_escape;
  case "scan_init" row_scan_init;
  case "scan_init_local" row_scan_init_local;
  case "statement" row_statement;
  case "body" row_body;
  case "other_node" row_other_node;
  case "write" row_write;
  case "discarded" row_discarded;
  Stdio.printf "%!"
