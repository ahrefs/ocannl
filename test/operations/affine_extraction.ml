(* gh-494 waypoint 1: [Ir.Low_level.affine_accesses] — extraction of a program's tensor-node
   accesses as explicit affine relations, the queryable artifact behind the affine legality queries.
   The dump covers the statement forms broadly: plain nests, whole-node [Zero_out],
   read-modify-write accumulations (the [rmw] reduction-dependence flag), guarded statements,
   dynamic gathers/scatters, and vectorized writes. The tail composes extraction with
   [Affine.pair_conflict]: which loops of the reduction nest admit parallelization — the
   determinism-as-constraint reading, where the accumulation confines conflicts over [i] to one
   thread while any parallelization over the reduced [k] is refuted. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Aff = Ir.Affine
module Tn = Ir.Tnode
module Ops = Ir.Ops
module B = Ll_builders

let fresh_tn =
  let c = ref 960_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ops.single

let for_over ?(extent = 4) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; axis = LL.Serial }

let get tn idcs = LL.Get (tn, idcs)
let it s = Idx.Iterator s

let show_idx = function
  | Idx.Fixed_idx k -> Int.to_string k
  | Idx.Iterator s -> Idx.symbol_ident s
  | Idx.Affine { symbols; offset } ->
      let terms =
        List.map symbols ~f:(fun (c, s) -> Printf.sprintf "%d*%s" c (Idx.symbol_ident s))
      in
      String.concat ~sep:"+" (terms @ if offset = 0 then [] else [ Int.to_string offset ])
  | Idx.Sub_axis -> "sub"
  | Idx.Concat _ -> "concat"

let show_loops loops =
  String.concat ~sep:","
    (List.map loops ~f:(fun (s, (lo, hi)) -> Printf.sprintf "%s:%d..%d" (Idx.symbol_ident s) lo hi))

let show_path path =
  String.concat ~sep:"."
    (List.map path ~f:(function
      | Aff.Stmt k -> Int.to_string k
      | Aff.Arg k -> "a" ^ Int.to_string k
      | Aff.Cond -> "c"
      | Aff.Body -> "b"
      | Aff.Rhs -> "r"
      | Aff.Write -> "w"))

let show (a : Tn.t Aff.access) =
  let flags =
    String.concat ~sep:""
      (List.filter_map
         [
           (a.a_dynamic, "dyn ");
           (a.a_whole, "whole ");
           (a.a_vec_last, "vec ");
           (a.a_guarded, "guarded ");
           (a.a_gated, "gated ");
           (a.a_rmw, "rmw ");
         ]
         ~f:(fun (b, s) -> Option.some_if b s))
  in
  Stdio.printf "%-2s %-3s %-14s loops=[%s] path=[%s] %s\n"
    (if a.a_write then "wr" else "rd")
    (Tn.debug_name a.a_tn)
    (Printf.sprintf "[%s]" (String.concat_array ~sep:";" (Array.map a.a_map ~f:show_idx)))
    (show_loops a.a_loops) (show_path a.a_path) flags

let () =
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 3 |] in
  let c = fresh_tn "C" [| 4; 3 |] in
  let s = fresh_tn "S" [| 4 |] in
  let d = fresh_tn "D" [| 1 |] in
  let g = fresh_tn "G" [| 1 |] in
  let e = fresh_tn "E" [| 4 |] in
  let ids = fresh_tn "I" [| 4 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  let i2 = Idx.get_symbol () and i3 = Idx.get_symbol () and i6 = Idx.get_symbol () in
  let pointwise =
    (* for i: for j: C[i][j] = A[i][j] + B[j] *)
    for_over i
      (for_over ~extent:3 j
         (LL.Set
            {
              tn = c;
              idcs = [| it i; it j |];
              llsc = LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp));
              debug = "";
            }))
  in
  let reduction =
    (* for i2: for k: S[i2] = S[i2] + A[i2][k] — an accumulation: rmw *)
    for_over i2
      (for_over ~extent:5 k
         (LL.Set
            {
              tn = s;
              idcs = [| it i2 |];
              llsc = LL.Binop (Ops.Add, (get s [| it i2 |], sp), (get a [| it i2; it k |], sp));
              debug = "";
            }))
  in
  let guarded =
    (* if G[0] then D[0] = 1 — a conditional (never-definite) write *)
    LL.If
      {
        cond = (get g [| Idx.Fixed_idx 0 |], sp);
        body = LL.Set { tn = d; idcs = [| Idx.Fixed_idx 0 |]; llsc = LL.Constant 1.; debug = "" };
      }
  in
  let gather =
    (* for i3: E[i3] = A[I[i3]][0] — dynamic gather *)
    for_over i3
      (LL.Set
         {
           tn = e;
           idcs = [| it i3 |];
           llsc =
             LL.Get_dynamic
               {
                 tn = a;
                 idcs = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |];
                 dyn_axis = 0;
                 dyn_value = (get ids [| it i3 |], sp);
               };
           debug = "";
         })
  in
  let guarded_rmw =
    (* for i6: if E[i6] < 1 then E[i6] = E[i6] + 1 — the gh-554/gh-561 trap shape: the condition
       reads the node the guarded body writes, at the same position. The intra-statement path
       components keep them apart (the condition's read at [.c] is not subordinate to the body's
       write at [.b.w]), where the bare statement position made them alias. *)
    for_over i6
      (LL.If
         {
           cond = (LL.Binop (Ops.Cmplt, (get e [| it i6 |], sp), (LL.Constant 1., sp)), sp);
           body =
             LL.Set
               {
                 tn = e;
                 idcs = [| it i6 |];
                 llsc = LL.Binop (Ops.Add, (get e [| it i6 |], sp), (LL.Constant 1., sp));
                 debug = "";
               };
         })
  in
  let program =
    LL.unflat_lines [ LL.Zero_out s; pointwise; reduction; guarded; gather; guarded_rmw ]
  in
  Stdio.printf "=== affine_accesses dump ===\n";
  let accesses = LL.affine_accesses program in
  List.iter accesses ~f:show;

  Stdio.printf "\n=== which loops of the reduction nest parallelize? ===\n";
  let nest_accs =
    List.filter accesses ~f:(fun ac ->
        List.exists ac.Aff.a_loops ~f:(fun (sym, _) -> Idx.equal_symbol sym i2))
  in
  let range sym =
    List.find_map nest_accs ~f:(fun ac ->
        List.Assoc.find ac.Aff.a_loops sym ~equal:Idx.equal_symbol)
  in
  let dup sym = Option.is_some (range sym) in
  let check_par name ~parallelizable sym =
    (* Every pair over a common node with at least one write must confine its conflicts to one
       thread of [sym]. *)
    let verdicts =
      List.concat_map nest_accs ~f:(fun x ->
          List.filter_map nest_accs ~f:(fun y ->
              if (x.Aff.a_write || y.Aff.a_write) && x.Aff.a_tn.Tn.uid = y.Aff.a_tn.Tn.uid then
                Some
                  (Aff.pair_conflict ~range ~dup_left:dup ~dup_right:dup
                     ~pairs:[ (sym, sym) ]
                     ~left:x.Aff.a_map ~right:y.Aff.a_map)
              else None))
    in
    let safe =
      (not (List.is_empty verdicts))
      && List.for_all verdicts ~f:(function Aff.Cross_thread _ -> false | _ -> true)
    in
    (* The row is the reading; this is the decision. The table shows both answers because both are
       facts about the nest -- a map axis parallelizes, a reduced axis must not -- so the row cannot
       be phrased so that `true` always passes, and the claim beside it is what carries the verdict
       (Codex P2, round 2). Without it a conflict analysis that stopped seeing the reduction's
       cross-thread dependence would flip `false` to `true` here, exit zero, and be promotable.
       Stated in the direction that holds, on the same bound boolean the row prints. *)
    Stdio.printf "%s %s parallelizable: %b\n" (Idx.symbol_ident sym) name safe;
    Verdict.claimf "%s %s has a non-empty conflict census" (Idx.symbol_ident sym) name
      (not (List.is_empty verdicts));
    let decision = (not (List.is_empty verdicts)) && if parallelizable then safe else not safe in
    if parallelizable then Verdict.claimf "%s %s parallelizes" (Idx.symbol_ident sym) name decision
    else Verdict.claimf "%s %s does not parallelize" (Idx.symbol_ident sym) name decision
  in
  check_par "(map axis)" ~parallelizable:true i2;
  check_par "(reduced axis)" ~parallelizable:false k;

  Stdio.printf "\n=== sibling scope operands (gh-561 Arg components) ===\n";
  (* for i7: Y[i7] = scopeA{ la := X[i7] } + scopeB{ X[i7] := 5; lb := 1 } — two [Local_scope]
     operands inlined into one statement's rhs. Each scope occurrence extends the path with its own
     [Arg] evaluation position, so scope A's read and scope B's write of the same node never
     interleave their interior components — and coverage claims nothing across sibling operands
     (evaluation order among them is not modeled), so X keeps its read-before-write (input)
     classification. Scope B's write is deliberately out of contract (gh-ocannl-584): the point is
     that the query stays conservative on IR it must never trust, so it is probed here rather than
     compiled. *)
  let x = fresh_tn "X" [| 4 |] in
  let y2 = fresh_tn "Y" [| 4 |] in
  let i7 = Idx.get_symbol () in
  (* Scope nodes DECLARED virtual, so the program below survives [specialize_proc] for the
     decision-level check further down. Freshness alone is not enough and used to be silently
     insufficient: a node with no setter is decided non-virtual, and [specialize_proc] then rewrote
     each scope into a bare [Get] of a buffer nothing writes -- dropping, among other things, scope
     B's write, the very thing this leg is about. That normalization is a rejection since
     gh-ocannl-681, which is what turned the omission into an error rather than a quiet collapse. *)
  let scope_node label =
    let tn = fresh_tn label [| 4 |] in
    Tn.update_memory_mode tn Tn.Virtual (Site "99:test-setup");
    LL.get_scope tn
  in
  let la = scope_node "LA" and lb = scope_node "LB" in
  let scope_a : LL.scalar_t =
    LL.Local_scope
      {
        id = la;
        body = LL.Set_local (la, get x [| it i7 |]);
        orig_indices = [| it i7 |];
        mint = LL.Inlined_computation;
      }
  in
  let scope_b : LL.scalar_t =
    LL.Local_scope
      {
        id = lb;
        body =
          LL.Seq
            ( LL.Set { tn = x; idcs = [| it i7 |]; llsc = LL.Constant 5.; debug = "" },
              LL.Set_local (lb, LL.Constant 1.) );
        orig_indices = [| it i7 |];
        mint = LL.Inlined_computation;
      }
  in
  let sibling =
    for_over i7
      (LL.Set
         {
           tn = y2;
           idcs = [| it i7 |];
           llsc = LL.Binop (Ops.Add, (scope_a, sp), (scope_b, sp));
           debug = "";
         })
  in
  let sib_accs = LL.affine_accesses sibling in
  List.iter sib_accs ~f:show;
  let x_read =
    List.find_exn sib_accs ~f:(fun a -> (not a.Aff.a_write) && a.Aff.a_tn.Tn.uid = x.Tn.uid)
  in
  let x_writes = List.filter sib_accs ~f:(fun a -> a.Aff.a_write && a.Aff.a_tn.Tn.uid = x.Tn.uid) in
  (match Aff.read_covered_before ~read:x_read ~writes:x_writes () with
  | `Covered ->
      Verdict.fail "scope A's read covered by scope B's write: containment crossed sibling operands"
  | `Unknown _ ->
      Stdio.printf "scope A's read not covered by the sibling operand's write: correct\n");
  (* The reverse arrangement — the writing scope evaluated first in traversal order — is declined
     too: sibling [Arg] positions are incomparable, so no cross-operand ordering is claimed even
     where left-to-right emission would justify it. *)
  let sibling_rev =
    for_over i7
      (LL.Set
         {
           tn = y2;
           idcs = [| it i7 |];
           llsc = LL.Binop (Ops.Add, (scope_b, sp), (scope_a, sp));
           debug = "";
         })
  in
  let rev_accs = LL.affine_accesses sibling_rev in
  let x_read =
    List.find_exn rev_accs ~f:(fun a -> (not a.Aff.a_write) && a.Aff.a_tn.Tn.uid = x.Tn.uid)
  in
  let x_writes = List.filter rev_accs ~f:(fun a -> a.Aff.a_write && a.Aff.a_tn.Tn.uid = x.Tn.uid) in
  (match Aff.read_covered_before ~read:x_read ~writes:x_writes () with
  | `Covered ->
      Stdio.printf "read covered across sibling operands (write-first): ordering claimed\n"
  | `Unknown _ ->
      Stdio.printf "no ordering claimed across sibling operands (write-first): correct\n");

  (* The decision level: the same verdict driving the real pipeline. [analyze_proc] +
     [specialize_proc] run [decide_placements] (hence [reads_covered_query]) on this code, and must
     classify X as read-before-write — a routine input ([input_and_output_nodes]) whose incoming
     buffer is preserved. If coverage wrongly crossed the sibling operands, both facts would flip
     and X's incoming values could be silently overwritten. The classification→execution leg is
     pinned by read_before_write_flip.ml on pipeline-produced code; this pattern itself cannot reach
     codegen — a write inside a scope body is out of contract (gh-ocannl-584) and
     [Low_level.validate_scope_bodies] rejects it there, which is why the probe stops at the
     analysis and decision levels the query actually has to survive. *)
  let materialize tn = Tn.update_memory_mode tn Tn.On_device (Site "99:test-setup") in
  materialize x;
  materialize y2;
  let opt = LL.specialize_proc (LL.empty_optimize_ctx ()) (LL.analyze_proc [] sibling) in
  (match Base.Hashtbl.find opt.LL.traced_store x with
  | None -> Verdict.fail "decide_placements: X not traced -- the fact this leg checks is missing"
  | Some traced ->
      Verdict.p "decide_placements classifies X as read-before-write" traced.LL.read_before_write);
  let (inputs, _outputs), _merge = LL.input_and_output_nodes opt in
  Verdict.p "X is a routine input (incoming buffer preserved)" (Base.Set.mem inputs x)

(* gh-ocannl-1016: the two things the access relations did not carry, which every "between two
   points, what else runs?" question used to re-walk the raw code for — scalar gatedness ([a_gated])
   and the effects that are no tensor-node access ([LL.statement_effects], from the same walk). One
   program exercises each effect kind the raw pipeline can meet (a [Tile_mma]'s row is pinned
   through [Online_softmax]'s opacity legs in online_softmax.ml), a gate of each shape, a scope body
   under a gate, and a dead loop. *)
let () =
  Stdio.printf "\n=== scalar gates and statement effects (gh-ocannl-1016) ===\n";
  let scope_node label =
    let tn = fresh_tn label [| 1 |] in
    B.virtualize tn;
    tn
  in
  let p_ = fresh_tn "P" [| 4 |] and q_ = fresh_tn "Q" [| 4 |] and h_ = fresh_tn "H" [| 4 |] in
  let r_ = fresh_tn "R" [| 4 |] and m_ = fresh_tn "M" [| 4 |] in
  let lc = LL.get_scope (scope_node "LC") and ls = LL.get_scope (scope_node "LS") in
  let cr = B.carry ~init:(B.get p_ [| B.fixed 0 |]) (scope_node "CR") in
  let i8 = B.sym () and i9 = B.sym () and i10 = B.sym () and i11 = B.sym () in
  let dead = B.sym () in
  let program =
    LL.unflat_lines
      [
        (* 0 *)
        LL.Declare_local { id = lc; needs_init = false };
        (* 1: a [Where]'s arms are gated, its condition is not *)
        B.loop_n i8 4
          (LL.Set_local
             ( lc,
               B.where_
                 (B.get h_ [| B.iter i8 |])
                 (B.get p_ [| B.iter i8 |])
                 (B.get q_ [| B.iter i8 |]) ));
        (* 2: a scope body in a gated arm runs hoisted, ungated; a merge read in the other arm is
           gated *)
        B.loop_n i9 4
          (B.set r_
             [| B.iter i9 |]
             (B.where_
                (B.get h_ [| B.iter i9 |])
                (LL.Local_scope
                   {
                     id = ls;
                     body = LL.Set_local (ls, B.get p_ [| B.iter i9 |]);
                     orig_indices = [| B.iter i9 |];
                     mint = LL.Inlined_computation;
                   })
                (LL.Get_merge_buffer (m_, [| B.iter i9 |]))));
        (* 3: a gated second operand *)
        B.loop_n i10 4
          (B.set r_
             [| B.iter i10 |]
             (B.conj (B.get p_ [| B.iter i10 |]) (B.get q_ [| B.iter i10 |])));
        (* 4 *)
        LL.Workgroup_barrier;
        (* 5 *)
        LL.Staged_compilation (fun () -> PPrint.empty);
        (* 6: a scan's carried state is written at its init, by the body, and per rotation *)
        B.scan ~upto:3 i11 ~carried:[ cr ]
          (B.set_next cr (B.add (B.prev cr) (B.get q_ [| B.iter i11 |])));
        (* 7: a dead loop's rows carry the dead bound *)
        B.loop ~upto:(-1) dead (LL.Set_local (lc, B.c 0.));
      ]
  in
  let accs, effs = LL.affine_relations program in
  List.iter accs ~f:show;
  let show_kind : Tn.t Aff.effect_kind -> string = function
    | Aff.Local_write -> "local-write"
    | Aff.Local_declare -> "local-declare"
    | Aff.Scope_body -> "scope-body"
    | Aff.Barrier -> "barrier"
    | Aff.Staged -> "staged"
    | Aff.Mma -> "mma"
    | Aff.Merge_read tn -> "merge-read " ^ Tn.debug_name tn
  in
  List.iter effs ~f:(fun (e : Tn.t Aff.statement_effect) ->
      Stdio.printf "fx %-16s loops=[%s] path=[%s] %s%s%s\n" (show_kind e.e_kind)
        (show_loops e.e_loops) (show_path e.e_path)
        (if e.e_guarded then "guarded " else "")
        (if e.e_gated then "gated " else "")
        (if Aff.loops_live e.e_loops then "" else "dead"));
  let open Verdict.Claims in
  let reads_in stmt tn =
    List.filter accs ~f:(fun a ->
        (not a.Aff.a_write) && Aff.stmt_head a.a_path = stmt && Tn.equal a.a_tn tn)
  in
  let gated_as name cases =
    p_all name
      (List.concat_map cases ~f:(fun (stmt, tn, expected) ->
           List.map (reads_in stmt tn) ~f:(fun a -> (a, expected))))
      ~f:(fun ((a : Tn.t Aff.access), expected) -> Bool.equal a.a_gated expected)
  in
  gated_as "the Where's condition read is not gated" [ (1, h_, false) ];
  gated_as "both Where arms' reads are gated" [ (1, p_, true); (1, q_, true) ];
  gated_as "a scope body inside a gated arm runs ungated" [ (2, p_, false) ];
  gated_as "And's first operand is ungated and its second gated" [ (3, p_, false); (3, q_, true) ];
  let kinds_at stmt =
    List.filter_map effs ~f:(fun (e : Tn.t Aff.statement_effect) ->
        Option.some_if (Aff.stmt_head e.e_path = stmt) (show_kind e.e_kind))
  in
  p_exists "the merge read in the other arm is a gated effect row" effs
    ~f:(fun (e : Tn.t Aff.statement_effect) ->
      (match e.e_kind with Aff.Merge_read tn -> Tn.equal tn m_ | _ -> false) && e.e_gated);
  p "every statement's effect kinds are the expected ones"
    (List.equal (List.equal String.equal)
       (List.map (List.range 0 8) ~f:kinds_at)
       [
         [ "local-declare" ];
         [ "local-write" ];
         [ "scope-body"; "local-write"; "merge-read M" ];
         [];
         [ "barrier" ];
         [ "staged" ];
         [ "local-write"; "local-write"; "local-write" ];
         [ "local-write" ];
       ]);
  p_all "only the dead loop's row is dead" effs ~f:(fun (e : Tn.t Aff.statement_effect) ->
      Bool.equal (Aff.loops_live e.e_loops) (Aff.stmt_head e.e_path <> 7));
  (* The relationship, not a restatement: the view's gatedness IS [Access_fold]'s gated context —
     the convention every remaining [Access_fold] consumer reads — read for read, in order. *)
  let fold_reads =
    let open LL.Access_fold in
    let policy =
      {
        discarded_operands = Skip;
        gated_operands = Visit;
        dead_loops = Visit;
        local_scopes = Visit;
        guards = Ignore;
        scan_implicit = Skip;
      }
    in
    let hooks =
      {
        (hooks ()) with
        scalar =
          (fun ctx acc sc ->
            match sc with
            | LL.Get (tn, _) | LL.Get_dynamic { tn; _ } -> Continue ((tn.Tn.uid, ctx.gated) :: acc)
            | _ -> Continue acc);
      }
    in
    List.rev (fold ~policy ~hooks ~init:[] program)
  in
  let view_reads =
    List.filter_map accs ~f:(fun a ->
        Option.some_if (not a.Aff.a_write) (a.Aff.a_tn.Tn.uid, a.Aff.a_gated))
  in
  p "a_gated agrees with Access_fold's gated context, read for read"
    ((not (List.is_empty view_reads))
    && List.equal (fun (u, g) (u2, g2) -> u = u2 && Bool.equal g g2) fold_reads view_reads)
