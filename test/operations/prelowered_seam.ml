(* gh-ocannl-562: the [?prelowered] test seam — compiling and LINKING a hand-built
   [Ir.Low_level.optimized], so that IR shapes the [Assignments] pipeline never emits can be pinned
   by executed values and not only by structural assertions. This is where the seam is worked out;
   the [Ll_test] support library (gh-ocannl-600) packages it for the tests that use it.

   Before the seam, backend [compile] always re-lowered from [Assignments]: [?lowered_transform]
   substitutes the codegen input only, while [code.lowered] — which drives I/O classification
   ([input_and_output_nodes]), liveness planning and the context-buffer delta — stayed the compile's
   own lowering, so hand-built code could reach [analyze_proc]/[specialize_proc] and no further.

   Phase 1 is the seam itself on ordinary code: a hand-written pointwise kernel is seeded with
   [set_values], run, and read back with [get_values].

   Phase 2 is the pattern the seam exists for (the gh-ocannl-561 review's sibling-[Local_scope]
   case, unreachable through [Assignments] because the optimizer never emits scopes reading a node
   another statement overwrites):

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ lb := X[i] * 3 }] then [X[i] := 5]

   The reads in the sibling scopes precede the later write, so the coverage query must NOT report
   X's reads as covered: X is read before written, hence a routine input whose incoming buffer
   contents are preserved. Under a wrong [`Covered] verdict X's placement stays undecided, the
   virtualizer inlines its trivial [5.0] setter into the scope reads, and the seeded values silently
   vanish — which shows up here as executed values, the leg structural pins cannot reach (verified
   by forcing the verdict during development: the value checks fail alongside the classification
   ones).

   Phases 3 and 4 pin the scope-purity contract (gh-ocannl-584) the phase-2 program is written to
   respect: the overwrite of X lives in a statement of its own, not inside [scopeB]'s body, because
   a scope body does not execute where it is written — it renders hoisted ahead of the enclosing
   statement ordered by [scope_id], and [hoist_cross_statement_cse] can lift it out of the statement
   entirely. A body write is therefore malformed IR rather than under-specified IR. Phase 3 pins
   both pipeline gates and the analysis probe boundary between them; phase 4 pins the other three
   ways a body can reach outside itself.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
open Stdio
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:5620 ~dims:[| 4 |] ()

(* Whether [f] is refused by the scope-purity gate specifically -- not by any other
   [Invalid_argument] the pipeline might raise. *)
let rejected f =
  try
    f ();
    false
  with Invalid_argument msg -> String.is_substring msg ~substring:"validate_scope_bodies"

(* A hand-written [Y[i] = X[i] * 2 + 1] over 4 cells: seeded, executed and read back. *)
let phase1 () =
  let x = mk "pls_x" and y = mk "pls_y" in
  materialize y;
  let i = sym () in
  let llc = loop ~upto:3 i (set y [| iter i |] (add (mul (get x [| iter i |]) (c 2.)) (c 1.))) in
  let xv = [| 1.5; 2.5; 3.5; 4.5 |] in
  let opt = optimize ~name:"pls_pointwise" llc in
  (* The I/O claims read [inputs]/[outputs] off the compiled routine itself, so they pin the
     classification the link actually consumed, not a re-derivation of it (gh-ocannl-590). *)
  let ((_, routine) as linked) = link ~name:"pls_pointwise" opt in
  p "phase1: the hand-built read-only node is a routine input" (Set.mem routine.Context.inputs x);
  p "phase1: the written node is a routine output" (Set.mem routine.Context.outputs y);
  let ctx = run_linked linked ~seed:[ (x, xv) ] in
  let yv = Context.get_values ctx y in
  p "phase1: executed values match the hand-built formula"
    (close yv (Array.map xv ~f:(fun v -> (v *. 2.) +. 1.)))

(* The two sibling [Local_scope] operands of the gh-ocannl-561 sketch, with the overwrite of X where
   the scope-purity contract puts it — a statement of its own, after the reading statement:

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ lb := X[i] * 3 }; X[i] := 5]

   X's placement is left to the pipeline, so the read-before-write classification is what keeps X a
   materialized input.

   [scopeA] is an inlined accumulation (the virtualizer's own shape for an inlined reduction) while
   [scopeB] is a single assignment, which [simplify_llc] collapses into the enclosing expression —
   so the program covers both fates a scope operand can meet, and under purity the two agree: no
   read can observe whether it was hoisted or collapsed. *)
let phase2 () =
  let x = mk "pls2_x" and y = mk "pls2_y" in
  let la = mk ~dims:[| 1 |] "pls2_la" and lb = mk ~dims:[| 1 |] "pls2_lb" in
  materialize y;
  (* The scope-local scalars stand for inlined nodes, exactly as the virtualizer's own scopes do. *)
  virtualize la;
  virtualize lb;
  let i = sym () and k = sym () and j = sym () in
  let id_a = LL.get_scope la in
  let id_b = LL.get_scope lb in
  let scope_a : LL.scalar_t =
    LL.Local_scope
      {
        id = id_a;
        body =
          seq
            (LL.Set_local (id_a, c 0.))
            (loop ~upto:1 k (LL.Set_local (id_a, add (LL.Get_local id_a) (get x [| iter i |]))));
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
      }
  in
  let scope_b : LL.scalar_t =
    LL.Local_scope
      {
        id = id_b;
        body = LL.Set_local (id_b, mul (get x [| iter i |]) (c 3.));
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
      }
  in
  let llc =
    seq
      (loop ~upto:3 i (set y [| iter i |] (add scope_a scope_b)))
      (loop ~upto:3 j (set x [| iter j |] (c 5.)))
  in
  let xv = [| 1.5; 2.5; 3.5; 4.5 |] in
  let opt = optimize ~name:"pls_sibling_scopes" llc in
  (match Hashtbl.find opt.LL.traced_store x with
  | None -> p "phase2: X traced" false
  | Some traced -> p "phase2: X is classified read-before-write" traced.LL.read_before_write);
  (* As in phase 1, the routine's own [inputs]/[outputs] carry the classification under test:
     read-before-write is what keeps X a materialized input (gh-ocannl-590). *)
  let ((_, routine) as linked) = link ~name:"pls_sibling_scopes" opt in
  p "phase2: X is a routine input" (Set.mem routine.Context.inputs x);
  p "phase2: X is also a routine output" (Set.mem routine.Context.outputs x);
  let ctx = run_linked linked ~seed:[ (x, xv) ] in
  let yv = Context.get_values ctx y and xv_out = Context.get_values ctx x in
  (* Both scopes' reads see X's INCOMING value; the later statement's write lands. *)
  p "phase2: Y = 5*X_in (both scope operands read the input)"
    (close yv (Array.map xv ~f:(fun v -> 5. *. v)));
  p "phase2: X was overwritten by the following statement" (close xv_out [| 5.; 5.; 5.; 5. |])

(* gh-ocannl-584: the same program with the overwrite moved INTO [scopeB]'s body,

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ X[i] := 5; lb := X[i] * 3 }]

   is out of contract, and codegen says so instead of silently picking an order. The write would
   render hoisted ahead of the whole statement, so [scopeA]'s reads — and, had [scopeA] been a
   collapsible single assignment, its in-expression reads too — would see 5 rather than the incoming
   values, regardless of the operands' syntactic positions. *)
let phase3 () =
  let x = mk "pls3_x" and y = mk "pls3_y" in
  let la = mk ~dims:[| 1 |] "pls3_la" and lb = mk ~dims:[| 1 |] "pls3_lb" in
  materialize y;
  virtualize la;
  virtualize lb;
  let i = sym () and k = sym () in
  let id_a = LL.get_scope la in
  let id_b = LL.get_scope lb in
  let scope_a : LL.scalar_t =
    LL.Local_scope
      {
        id = id_a;
        body =
          seq
            (LL.Set_local (id_a, c 0.))
            (loop ~upto:1 k (LL.Set_local (id_a, add (LL.Get_local id_a) (get x [| iter i |]))));
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
      }
  in
  let scope_b : LL.scalar_t =
    LL.Local_scope
      {
        id = id_b;
        body =
          seq (set x [| iter i |] (c 5.)) (LL.Set_local (id_b, mul (get x [| iter i |]) (c 3.)));
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
      }
  in
  let llc = loop ~upto:3 i (set y [| iter i |] (add scope_a scope_b)) in
  (* The pipeline's entry gate. It has to be the entry, not just codegen:
     [hoist_cross_statement_cse] lifts a body shared by sibling statements to a top-level
     [Declare_local] + body, which would carry an impure body's write out of every [Local_scope] --
     past a codegen-only check, and executing once instead of once per user statement. *)
  p "phase3: the pipeline rejects a tensor-node write in a scope body"
    (rejected (fun () -> ignore (optimize ~name:"pls_impure_scope" llc)));
  (* The exit gate, reached via the analysis entry points -- which deliberately do NOT validate, so
     that probes like affine_extraction.ml can ask what the coverage query does with IR it must
     never trust. Everything past them does. *)
  let admitted = ref None in
  p "phase3: the analysis entry points admit it (the documented probe boundary)"
    (not
       (rejected (fun () ->
            admitted :=
              Some (LL.specialize_proc (LL.empty_optimize_ctx ()) (LL.analyze_proc [] llc)))));
  let opt = Option.value_exn !admitted in
  p "phase3: compiling the out-of-contract routine is rejected"
    (rejected (fun () ->
         ignore
           (Context.compile ~name:"pls_impure_scope" ~prelowered:opt
              ~lowered_transform:(fun o -> [ o ])
              (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty)))

(* The other three ways a body can reach outside itself. Each would make the placement of a scope
   body observable, which is the one thing purity buys: a [Set_local] of a sibling's local mutates
   what that scope returns, and neither a barrier nor a staged callback is scope-local state at all
   (the callback's emissions cannot even be inspected). Checked on minimal terms rather than through
   a compile -- the rule is structural. *)
let phase4 () =
  let x = mk "pls4_x" and y = mk "pls4_y" in
  let la = mk ~dims:[| 1 |] "pls4_la" and lb = mk ~dims:[| 1 |] "pls4_lb" in
  let i = sym () in
  let id_a = LL.get_scope la and id_b = LL.get_scope lb in
  let scope_with body : LL.scalar_t =
    LL.Local_scope { id = id_a; body; orig_indices = [| iter i |]; mint = LL.Inlined_computation }
  in
  let stmt body = loop ~upto:3 i (set y [| iter i |] (scope_with body)) in
  (* Writing a SIBLING scope's local from this body. *)
  p "phase4: a Set_local of a local the body does not own is rejected"
    (rejected (fun () ->
         LL.validate_scope_bodies
           (stmt (seq (LL.Set_local (id_b, c 1.)) (LL.Set_local (id_a, get x [| iter i |]))))));
  (* Owning it makes the same shape legal: a body may declare and write its own sub-local. *)
  p "phase4: a Set_local of a local the body declares is accepted"
    (not
       (rejected (fun () ->
            LL.validate_scope_bodies
              (stmt
                 (seq
                    (LL.Declare_local { id = id_b; needs_init = false })
                    (seq
                       (LL.Set_local (id_b, get x [| iter i |]))
                       (LL.Set_local (id_a, LL.Get_local id_b))))))));
  p "phase4: a Workgroup_barrier in a scope body is rejected"
    (rejected (fun () ->
         LL.validate_scope_bodies
           (stmt (seq LL.Workgroup_barrier (LL.Set_local (id_a, get x [| iter i |]))))));
  p "phase4: a Staged_compilation in a scope body is rejected"
    (rejected (fun () ->
         LL.validate_scope_bodies
           (stmt
              (seq
                 (LL.Staged_compilation (fun () -> PPrint.empty))
                 (LL.Set_local (id_a, get x [| iter i |]))))));
  (* The same constructs at statement level are ordinary code, so the gate must not flag them. *)
  p "phase4: the same constructs outside a scope body are accepted"
    (not
       (rejected (fun () ->
            LL.validate_scope_bodies
              (seq LL.Workgroup_barrier (loop ~upto:3 i (set y [| iter i |] (get x [| iter i |])))))))

(* gh-ocannl-584 review round 2: purity alone does not make the hoist sound. A body's INPUTS include
   the scope locals it reads, and [hoist_cross_statement_cse] evaluates a body shared by sibling
   statements ONCE ahead of the first user — so a [Set_local] of a local the body reads, sitting
   between two users, leaves the later user reading a stale value. A hazard check over tensor nodes
   alone cannot see that dependency; the check matches the body's [Local_read] rows against the
   lifted-over statements' [Local_write] rows (gh-ocannl-1050). Bodies reading a local declared
   outside them are ordinary pipeline output, not a hand-built curiosity:
   [eliminate_common_subexpressions] and this very pass create them (a strict "a body may only read
   locals it owns" rule was tried and rejects layer_norm_divided_mean's own lowering).

   [ext := 1; Y[0] = S{2*ext}; ext := 2; Y[1] = S'{2*ext}] with [S] and [S'] alpha-equivalent.
   Correct is [Y = [2; 4]]; hoisting [S] above the mutation gives [Y = [2; 2]].

   The positive control is the same program WITHOUT the intervening mutation, which must still hoist
   — otherwise "declines to hoist" would pass by never hoisting anything. *)
let phase5 () =
  let y = mk ~dims:[| 2 |] "pls5_y" in
  let ls = mk ~dims:[| 1 |] "pls5_ls" and ext_tn = mk ~dims:[| 1 |] "pls5_ext" in
  materialize y;
  virtualize ls;
  virtualize ext_tn;
  let ext = LL.get_scope ext_tn in
  (* An accumulation, so [simplify_llc] cannot collapse the scope before the hoist pass sees it. *)
  let scope () : LL.scalar_t =
    let id = LL.get_scope ls in
    let k = sym () in
    LL.Local_scope
      {
        id;
        body =
          seq
            (LL.Set_local (id, c 0.))
            (loop ~upto:1 k (LL.Set_local (id, add (LL.Get_local id) (LL.Get_local ext))));
        orig_indices = [||];
        mint = LL.Inlined_computation;
      }
  in
  let set_at idx llsc : LL.t = set y [| fixed idx |] llsc in
  let program ~mutated =
    seq
      (seq (LL.Declare_local { id = ext; needs_init = false }) (LL.Set_local (ext, c 1.)))
      (seq
         (set_at 0 (scope ()))
         (seq (if mutated then LL.Set_local (ext, c 2.) else LL.Noop) (set_at 1 (scope ()))))
  in
  (* Structural: the shared body is hoisted out of both users when nothing writes its local in
     between, and left in place when something does. *)
  p "phase5: a shared body reading an untouched local is still hoisted (positive control)"
    (count_scopes (LL.hoist_cross_statement_cse (program ~mutated:false)) = 0);
  p "phase5: a shared body whose local is written between users is not hoisted"
    (count_scopes (LL.hoist_cross_statement_cse (program ~mutated:true)) = 2);
  (* Executed: the later user must see the mutation, which is what the hoist would have erased. *)
  let _opt, results =
    optimize_and_execute ~name:"pls_local_hazard" (program ~mutated:true) ~seed:[] ~read:[ y ]
  in
  match results with
  | [ yv ] ->
      p "phase5: Y = [2; 4] — the second user sees the intervening write" (close yv [| 2.; 4. |])
  | _ -> assert false

(* gh-ocannl-584 review round 3: the raw [analyze_proc] / [specialize_proc] pair is public and does
   NOT validate — deliberately, so probes can ask what the analysis makes of IR it must never trust
   (affine_extraction.ml). But [specialize_proc] still runs [hoist_cross_statement_cse], and that is
   the one pass that can move an effect OUT of a [Local_scope]: two alpha-equivalent impure bodies
   in sibling statements would be lifted to a top-level [Declare_local] + body, after which the
   codegen gate sees no impure scope and compiles a silently changed program (the write executing
   once ahead of the first user instead of once per user statement). Phase 3 takes this route with a
   single scope, so it cannot trigger the hoist.

   The fix guards the PASS rather than the door: it declines to hoist a body that is not pure. The
   impure body therefore stays inside its scope, and the exit gate refuses the routine — rejected,
   never rewritten. Phase 5's positive control is the other half of this: pure shared bodies must
   still hoist. *)
let phase6 () =
  let x = mk "pls6_x" and y = mk "pls6_y" in
  let lb = mk ~dims:[| 1 |] "pls6_lb" in
  materialize y;
  virtualize lb;
  let i = sym () in
  (* Two alpha-equivalent impure scopes — same free index, same body shape, distinct ids — in
     sibling statements, the shape [hoist_shared_locals] groups. *)
  let impure_scope () : LL.scalar_t =
    let id = LL.get_scope lb in
    LL.Local_scope
      {
        id;
        body = seq (set x [| iter i |] (c 5.)) (LL.Set_local (id, mul (get x [| iter i |]) (c 3.)));
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
      }
  in
  let llc =
    loop ~upto:3 i
      (seq
         (set y [| iter i |] (impure_scope ()))
         (set y [| iter i |] (add (get y [| iter i |]) (impure_scope ()))))
  in
  let admitted = ref None in
  p "phase6: the analysis entry points admit two impure siblings"
    (not
       (rejected (fun () ->
            admitted :=
              Some (LL.specialize_proc (LL.empty_optimize_ctx ()) (LL.analyze_proc [] llc)))));
  let opt = Option.value_exn !admitted in
  (* The laundering the guard prevents: had the hoist fired, the write would sit at statement level
     and no [Local_scope] would carry it. *)
  p "phase6: the hoist declined, so the impure body is still inside its scope"
    (Option.is_some (LL.scope_purity_violation opt.LL.llc));
  p "phase6: compiling the specialized routine is still rejected"
    (rejected (fun () ->
         ignore
           (Context.compile ~name:"pls_impure_siblings" ~prelowered:opt
              ~lowered_transform:(fun o -> [ o ])
              (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty)))

let () =
  phase1 ();
  phase2 ();
  phase3 ();
  phase4 ();
  phase5 ();
  phase6 ();
  printf "prelowered seam: PASS\n"
