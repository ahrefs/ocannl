(* gh-563: the shared canonical-rendering core behind both digest consumers
   ([Low_level.analysis_digest] for the analysis cache, [Schedule_cache.canonicalize] for schedule
   replay). Those two are tested through their consumers (analysis_cache.ml, autotune_smoke.ml);
   this test pins the walk itself, which they now share.

   Part 1 is a golden of the rendering of a construct-rich program, under a policy that names tensor
   nodes by label (so the golden is stable — neither consumer's identity choice) and lets the walk
   alpha-rename loop binders and local scopes. Any change to how a construct enters a digest shows
   up here, and because there is exactly one walk it shows up for both consumers at once.

   Part 2 checks the policy seam: what the walk delegates (tensor-node identity, free symbols,
   binder shadowing, opaque constructs, [Tile_mma]) versus what it owns (loop-binder tokens and
   local-scope alpha indices — which is why alpha-variant lowerings of one routine render
   identically for both consumers). *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module CR = Ir.Low_level.Canonical_render

let single = Ops.single

(* Nodes and symbols from [Ll_test] (gh-ocannl-608); the rendering fixtures below stay written
   against the [Low_level] constructors, because what each one pins is the exact construct that
   reaches the walk. *)
let mk = Ll_test.node_factory ~first_id:7000 ~dims:[| 4 |] ()
let sym = Ll_test.sym
let arg sc : LL.scalar_arg = (sc, single)

(* A recording policy: everything the walk delegates lands in [log], so a test can assert on the
   delegation separately from the rendering. *)
let render ?(mma = CR.Structural_mma) ?(initial_tokens = []) llc =
  let buf = Buffer.create 256 in
  let log = ref [] in
  let note s = log := s :: !log in
  let add = Buffer.add_string buf in
  let policy =
    {
      (* By label rather than [Tn.uid] (the analysis cache's choice) or a first-occurrence index
         (the schedule cache's): keeps this golden stable and independent of both. *)
      CR.emit_tn = (fun tn -> add ("<" ^ List.hd_exn tn.Tn.label ^ ">"));
      emit_free_sym =
        (fun s ->
          note ("free:" ^ Idx.symbol_ident s);
          add "?");
      on_bind_loop =
        (fun s ~id ~shadowed ->
          note
            (Printf.sprintf "bind:%s=b%d%s" (Idx.symbol_ident s) id
               (if shadowed then ":shadowed" else "")));
      mark_incomplete = (fun () -> note "incomplete");
      mma;
      initial_tokens;
    }
  in
  CR.emit ~buf policy llc;
  (Buffer.contents buf, List.rev !log)

open Verdict.Claims

(* One "lowering" of a program touching a representative of every statement, scalar and index
   family: fresh symbols and scope ids per call, tensor nodes fixed by the caller — the shape
   sibling lowerings of one routine take. *)
let build_rich (table, ids, out, acc) =
  let i = sym () and j = sym () and c1 = sym () and c2 = sym () in
  let scope = LL.get_scope acc in
  let guarded =
    LL.Seq
      ( LL.Declare_local { id = scope; needs_init = true },
        LL.Seq
          ( LL.Set_local
              ( scope,
                LL.Binop
                  ( Ops.Add,
                    arg (LL.Get_local scope),
                    arg
                      (LL.Unop
                         ( Ops.Neg,
                           arg
                             (LL.Ternop
                                ( Ops.Where,
                                  arg (LL.Constant 1.5),
                                  arg (LL.Constant_bits 7L),
                                  arg (LL.Embed_index (Idx.Iterator j)) )) )) ) ),
            LL.Set
              {
                tn = out;
                idcs = [| Idx.Affine { symbols = [ (2, i); (1, j) ]; offset = 5 } |];
                llsc =
                  LL.Local_scope
                    {
                      id = LL.get_scope acc;
                      body = LL.Noop;
                      orig_indices = [| Idx.Iterator i |];
                      mint = LL.Inlined_computation;
                    };
                debug = "";
              } ) )
  in
  LL.Seq
    ( LL.Comment "presentational: skipped",
      LL.Seq
        ( LL.Zero_out out,
          LL.Seq
            ( LL.For_loop
                {
                  index = i;
                  from_ = 0;
                  to_ = 2;
                  axis = LL.Grid;
                  body =
                    LL.For_loop
                      {
                        index = j;
                        from_ = 1;
                        to_ = 3;
                        axis = LL.Unrolled;
                        body =
                          LL.If
                            { cond = arg (LL.Get (table, [| Idx.Fixed_idx 0 |])); body = guarded };
                      };
                },
              LL.Seq
                ( LL.Workgroup_barrier,
                  LL.Seq
                    ( LL.Set_dynamic
                        {
                          tn = out;
                          idcs = [| Idx.Fixed_idx 0 |];
                          dyn_axis = 0;
                          dyn_value = arg (LL.Get (ids, [| Idx.Sub_axis |]));
                          llsc =
                            LL.Get_dynamic
                              {
                                tn = table;
                                idcs = [| Idx.Concat [ c1; c2 ] |];
                                dyn_axis = 0;
                                dyn_value = arg (LL.Get_merge_buffer (ids, [| Idx.Fixed_idx 1 |]));
                              };
                          debug = "";
                        },
                      LL.Set_from_vec
                        {
                          tn = out;
                          idcs = [| Idx.Fixed_idx 2 |];
                          length = 4;
                          vec_unop = Ops.Uint4x32_to_prec_uniform;
                          arg = arg (LL.Get (table, [| Idx.Fixed_idx 3 |]));
                          debug = "";
                        } ) ) ) ) )

let part1 () =
  let tns = (mk "cr_table", mk "cr_ids", mk "cr_out", mk "cr_acc") in
  let text, log = render (build_rich tns) in
  Stdio.printf "rendering:\n%s\n\n" text;
  (* [c1]/[c2] are never bound by a loop: the two [Concat] symbols reach the free-symbol hook. *)
  Stdio.printf "delegated: %s\n\n" (String.concat ~sep:" " log);
  let text2, _ = render (build_rich tns) in
  p "an alpha-variant lowering renders identically" (String.equal text text2);
  let other = (mk "cr_table", mk "cr_ids", mk "cr_out", mk "cr_acc") in
  let text3, _ = render (build_rich other) in
  (* Same labels, different nodes: this policy renders them the same on purpose — the identity
     choice is the policy's, not the walk's. *)
  p "tensor-node identity is entirely the policy's" (String.equal text text3)

let part2 () =
  let a = mk "cr_a" and b = mk "cr_b" and d = mk "cr_d" in
  (* Opaque constructs delegate to [mark_incomplete]; the placeholder keeps the rendering
     well-formed. *)
  let _, log = render (LL.Staged_compilation (fun () -> PPrint.empty)) in
  p "Staged_compilation is reported incomplete" (List.equal String.equal log [ "incomplete" ]);
  let s = sym () in
  let loop body = LL.For_loop { index = s; from_ = 0; to_ = 2; body; axis = LL.Serial } in
  let _, log = render (loop (loop LL.Noop)) in
  p "a duplicated binder is reported shadowed"
    (List.equal String.equal log
       [ "bind:" ^ Idx.symbol_ident s ^ "=b0"; "bind:" ^ Idx.symbol_ident s ^ "=b1:shadowed" ]);
  (* A symbol bound by neither a loop binder nor [initial_tokens]: the schedule cache calls it
     unresolvable, the analysis cache renders it by ident. *)
  let free = sym () in
  let use_free =
    LL.Set { tn = d; idcs = [| Idx.Iterator free |]; llsc = LL.Constant 0.; debug = "" }
  in
  let text, log = render use_free in
  p "an unbound symbol reaches the free-symbol hook"
    (List.equal String.equal log [ "free:" ^ Idx.symbol_ident free ]);
  let text', log' = render ~initial_tokens:[ (free, "s0") ] use_free in
  p "initial_tokens pre-bind it instead" (List.is_empty log');
  p "and it then renders positionally"
    (String.is_substring text ~substring:"?" && String.is_substring text' ~substring:"s0");
  (* gh-ocannl-687: which pass minted a [Local_scope] is part of a program's identity — it decides
     which loops the action menu may target — so the walk must distinguish the two mints. Only the
     newer, schedule-minted form takes a distinguishing token, which is why part1's golden (an
     inlined scope) is unchanged by the field's arrival. *)
  let scope_stmt mint =
    LL.Set
      {
        tn = d;
        idcs = [| Idx.Fixed_idx 0 |];
        llsc = LL.Local_scope { id = LL.get_scope a; body = LL.Noop; orig_indices = [||]; mint };
        debug = "";
      }
  in
  let inlined_text, _ = render (scope_stmt LL.Inlined_computation) in
  let minted_text, _ = render (scope_stmt LL.Schedule_minted) in
  p "a virtualization-inlined and a schedule-minted scope render differently"
    (not (String.equal inlined_text minted_text));
  let lane = sym () in
  let mma =
    LL.Tile_mma
      {
        d = (d, [| Idx.Fixed_idx 0 |]);
        a = (a, [| Idx.Fixed_idx 0 |]);
        b = (b, [| Idx.Fixed_idx 0 |]);
        ta = false;
        tb = true;
        m = 16;
        n = 8;
        k = 8;
        ldd = 16;
        lda = 8;
        ldb = 8;
        lane;
        tile = None;
        fallback = LL.Noop;
      }
  in
  let text, log = render ~mma:CR.Structural_mma mma in
  Stdio.printf "structural mma: %s\n" text;
  p "structural mma is complete" (List.is_empty (List.filter log ~f:(String.equal "incomplete")));
  let text, log = render ~mma:CR.Opaque_mma mma in
  p "opaque mma renders a placeholder and reports incomplete"
    (String.equal text "mma;" && List.mem log "incomplete" ~equal:String.equal)

let () =
  part1 ();
  part2 ()
