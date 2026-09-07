(* gh-494 waypoint 3: the op-legality oracle over hand-built programs.

   A schedule op's obligation is decided by the affine queries: pairing the op's loop symbols as
   thread identity must leave every write-involving access pair Disjoint or Same_thread, with the
   reduction-reassociation license for Vectorized retypes and Swaps of accumulations. Verdicts are
   three-valued; both proven directions are sound (Op_unknown means "compile and see"). *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Sched = Ir.Schedule

let fresh_tn =
  let c = ref 970_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ir.Ops.single

let for_over ?(extent = 8) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; axis = LL.Serial }

let hand_built ~stmts ~tns_on_device ~tns_local =
  let optimize_ctx = LL.empty_optimize_ctx () in
  let plc = optimize_ctx.LL.placements in
  List.iter tns_on_device ~f:(fun tn -> Tn.Placements.update plc tn Tn.On_device 49);
  List.iter tns_local ~f:(fun tn -> Tn.Placements.update plc tn Tn.Local 49);
  let traced_store = Hashtbl.create (module Tn) in
  let llc = LL.unflat_lines stmts in
  List.iter (tns_on_device @ tns_local) ~f:(fun tn ->
      ignore (LL.get_node traced_store tn : LL.traced_array));
  {
    LL.traced_store;
    optimize_ctx;
    llc;
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Base.Set.empty (module Tn);
  }

let show = function
  | Sched.Op_legal -> "Legal"
  | Sched.Op_illegal _ -> "Illegal"
  | Sched.Op_unknown _ -> "Unknown"

let p name v = Stdio.printf "%-44s %s\n" name (show v)

let () =
  (* Matmul: zero-init nest plus an accumulation nest. *)
  let a = fresh_tn "a" [| 8; 8 |] and b = fresh_tn "b" [| 8; 8 |] and c = fresh_tn "c" [| 8; 8 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let i2 = Idx.get_symbol () and j2 = Idx.get_symbol () and k = Idx.get_symbol () in
  let get tn idcs = LL.Get (tn, idcs) in
  let zero_nest =
    for_over i
      (for_over j
         (LL.Set
            {
              tn = c;
              idcs = [| Idx.Iterator i; Idx.Iterator j |];
              llsc = LL.Constant 0.;
              debug = "";
            }))
  in
  let acc_nest =
    for_over i2
      (for_over j2
         (for_over k
            (LL.Set
               {
                 tn = c;
                 idcs = [| Idx.Iterator i2; Idx.Iterator j2 |];
                 llsc =
                   LL.Binop
                     ( Ir.Ops.Add,
                       (get c [| Idx.Iterator i2; Idx.Iterator j2 |], sp),
                       ( LL.Binop
                           ( Ir.Ops.Mul,
                             (get a [| Idx.Iterator i2; Idx.Iterator k |], sp),
                             (get b [| Idx.Iterator k; Idx.Iterator j2 |], sp) ),
                         sp ) );
                 debug = "";
               })))
  in
  (* Self-contained kernel: the accumulation nest alone (its rmw serves as the init-carrying form);
     every node is accessed only under this statement, so the intra-nest proof is the whole
     obligation for hardware retypes too. *)
  let opt = hand_built ~stmts:[ acc_nest ] ~tns_on_device:[ a; b; c ] ~tns_local:[] in
  (* Two sibling statements sharing [c]: hardware annotations interleave their threads with no
     grid-wide synchronization, so hardware retypes must downgrade to Unknown (Codex P1) while
     within-nest reorderings (Vectorized, Swap) keep their intra-nest verdicts. *)
  let opt2 = hand_built ~stmts:[ zero_nest; acc_nest ] ~tns_on_device:[ a; b; c ] ~tns_local:[] in
  let check name op = p name (Sched.op_legality opt op) in
  let check2 name op = p name (Sched.op_legality opt2 op) in
  check "retype outer parallel loop to Grid" (Sched.Retype { axis = i2; ty = LL.Grid });
  check "retype inner parallel loop to Workgroup" (Sched.Retype { axis = j2; ty = LL.Workgroup });
  check "retype reduction loop to Grid" (Sched.Retype { axis = k; ty = LL.Grid });
  check "retype reduction loop to Vectorized" (Sched.Retype { axis = k; ty = LL.Vectorized });
  check2 "cross-statement retype to Grid" (Sched.Retype { axis = i2; ty = LL.Grid });
  check2 "cross-statement zero-init loop to Grid" (Sched.Retype { axis = i; ty = LL.Grid });
  check2 "cross-statement Vectorized keeps scope" (Sched.Retype { axis = k; ty = LL.Vectorized });
  check2 "cross-statement swap keeps scope" (Sched.Swap { outer = i2; inner = j2 });
  check "swap parallel pair" (Sched.Swap { outer = i2; inner = j2 });
  check "swap parallel with reduction" (Sched.Swap { outer = j2; inner = k });
  check "swap non-nested loops" (Sched.Swap { outer = i2; inner = k });
  (let op, _, _ = Sched.split ~axis:k ~factor:4 ~outer:LL.Serial ~inner:LL.Workgroup in
   check "split reduction with Workgroup inner" op);
  (let op, _, _ = Sched.split ~axis:j2 ~factor:4 ~outer:LL.Serial ~inner:LL.Workgroup in
   check "split parallel with Workgroup inner" op);
  (let op, _, _ = Sched.split ~axis:k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
   check "split reduction serial/serial" op);
  check "unroll reduction" (Sched.Unroll { axis = k; materialize = false });

  (* Stage: apply's precondition surface is probed hermetically (a raising apply is a proven
     Illegal), then the containment query [Affine.read_covered_before] proves the staged tile covers
     the remapped reads — a covered serial packing stage is Legal; shared staging stays Unknown
     (barriers and launch geometry are validated downstream). *)
  let stage ?(source = a) ?(tile_loops = [ k ]) ?(shared = false) ?(cooperative = None)
      ?(hoisted = false) ?(swizzle = None) ?(pad_stride = None) ?(pipeline_depth = 1) () =
    Sched.Stage
      {
        source;
        tile_loops;
        shared;
        cooperative;
        hoisted;
        swizzle;
        pad_stride;
        pipeline_depth;
        tile_prec = None;
      }
  in
  check "stage packing proves coverage" (stage ());
  check "stage packing two tile loops" (stage ~tile_loops:[ i2; k ] ());
  check "stage of a written source" (stage ~source:c ~tile_loops:[ i2; j2 ] ());
  check "stage tile loop absent from the read" (stage ~tile_loops:[ j2 ] ());
  check "stage shared stays unknown" (stage ~shared:true ());
  check "stage hoisted without host-init data" (stage ~hoisted:true ());
  check "stage cooperative requires shared" (stage ~cooperative:(Some 32) ());

  (* Tensorize: role-assignment validity is decided by probing apply's micro-kernel recognition —
     the k role must be the (only) rmw-carrying loop and i/j the output dims, so of the 6 role
     permutations of the [i2 x j2 x k] accumulation only the identity assignment survives; the
     affine queries then prove i/j iteration independence (reduction reassociation licensed). *)
  let tz ~i ~j ~k name ck = ck name (fst (Sched.tensorize ~i ~j ~k ~simd_width:32 ())) in
  tz ~i:i2 ~j:j2 ~k "tensorize valid roles" check;
  tz ~i:j2 ~j:i2 ~k "tensorize i/j roles swapped" check;
  tz ~i:i2 ~j:k ~k:j2 "tensorize j role on the reduction" check;
  tz ~i:k ~j:j2 ~k:i2 "tensorize i role on the reduction" check;
  tz ~i:j2 ~j:k ~k:i2 "tensorize rotated roles (j k i)" check;
  tz ~i:k ~j:i2 ~k:j2 "tensorize rotated roles (k i j)" check;
  tz ~i:i2 ~j:j2 ~k "tensorize with cross-statement sharing" check2;

  (* A genuinely order-sensitive serial loop: lagged self-reference. *)
  let x = fresh_tn "x" [| 9 |] in
  let t = Idx.get_symbol () in
  let lag_nest =
    for_over t
      (LL.Set
         {
           tn = x;
           idcs = [| Idx.Affine { symbols = [ (1, t) ]; offset = 1 } |];
           llsc =
             LL.Binop
               ( Ir.Ops.Add,
                 (get x [| Idx.Iterator t |], sp),
                 (get x [| Idx.Affine { symbols = [ (1, t) ]; offset = 1 } |], sp) );
           debug = "";
         })
  in
  let opt_lag = hand_built ~stmts:[ lag_nest ] ~tns_on_device:[ x ] ~tns_local:[] in
  p "retype lagged recurrence to Grid"
    (Sched.op_legality opt_lag (Sched.Retype { axis = t; ty = LL.Grid }));

  (* The autotuner's menu contract on a REAL lowered matmul (not hand-built): the menu proposes all
     role permutations of a serial accumulation triple, and the oracle gates them. Pruning must
     remove exactly the proposals whose [Schedule.apply] raises (the candidate compile would fail)
     and never a viable candidate — assert the equivalence per permutation. *)
  let open Ocannl.Operation.DSL_modules in
  let m = 8 in
  let mav =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:5 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "olma" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "olmb" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let%op olmc = ma * mb in
  let mm_comp = Ocannl.Train.forward olmc in
  let captured = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx mm_comp Idx.Empty
  in
  let mm_opt = Option.value_exn !captured in
  (* The serial accumulation triple, as the menu's [collect_serial_triples] sees it. *)
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec find_triple (llc : LL.t) : (Idx.symbol * Idx.symbol * Idx.symbol) option =
    match llc with
    | LL.Seq (a, b) -> ( match find_triple a with Some _ as r -> r | None -> find_triple b)
    | LL.If { body; _ } -> find_triple body
    | LL.For_loop { index = ti; axis = LL.Serial; from_ = 0; body; _ } -> (
        match strip (LL.flat_lines [ body ]) with
        | [ LL.For_loop { index = tj; axis = LL.Serial; from_ = 0; body = jb; _ } ] -> (
            match strip (LL.flat_lines [ jb ]) with
            | [ LL.For_loop { index = tk; axis = LL.Serial; from_ = 0; body = kb; _ } ] -> (
                match strip (LL.flat_lines [ kb ]) with
                | [ LL.Set _ ] -> Some (ti, tj, tk)
                | _ -> None)
            | _ -> None)
        | _ -> None)
    | LL.For_loop { body; _ } -> find_triple body
    | _ -> None
  in
  let ti, tj, tk = Option.value_exn (find_triple mm_opt.LL.llc) in
  let perms =
    [
      ("i j k", ti, tj, tk);
      ("i k j", ti, tk, tj);
      ("j i k", tj, ti, tk);
      ("j k i", tj, tk, ti);
      ("k i j", tk, ti, tj);
      ("k j i", tk, tj, ti);
    ]
  in
  let viable = ref 0 in
  List.iter perms ~f:(fun (label, i, j, k) ->
      let op = fst (Sched.tensorize ~i ~j ~k ~simd_width:1 ()) in
      let verdict = Sched.op_legality mm_opt op in
      let hermetic =
        {
          mm_opt with
          LL.traced_store = Hashtbl.copy mm_opt.LL.traced_store;
          optimize_ctx = LL.copy_optimize_ctx mm_opt.LL.optimize_ctx;
        }
      in
      let applies =
        match Sched.apply [ op ] hermetic with
        | (_ : LL.optimized) -> true
        | exception Invalid_argument _ -> false
      in
      if applies then Int.incr viable;
      let pruned = match verdict with Sched.Op_illegal _ -> true | _ -> false in
      let sound = Bool.( = ) pruned (not applies) in
      Stdio.printf "menu tensorize roles (%s): %-8s apply %s  pruning %s\n" label (show verdict)
        (if applies then "succeeds" else "raises")
        (if sound then "sound" else "UNSOUND");
      Verdict.claim ("menu tensorize roles (" ^ label ^ "): pruning sound") sound);
  Stdio.printf "viable tensorize permutations survive the gate: %s\n"
    (if !viable > 0 then "yes" else "NO");
  Verdict.claim "viable tensorize permutations survive the gate" (!viable > 0);

  (* Sequential checking: verdicts against progressively applied code, stopping at illegal. *)
  let sched =
    [
      Sched.Retype { axis = i2; ty = LL.Grid };
      Sched.Retype { axis = k; ty = LL.Workgroup };
      Sched.Retype { axis = j2; ty = LL.Workgroup };
    ]
  in
  Stdio.printf "schedule_legality stops at the illegal op:\n";
  List.iter (Sched.schedule_legality opt sched) ~f:(fun (op, v) ->
      let label =
        match op with
        | Sched.Retype { axis; ty } ->
            Printf.sprintf "  Retype %s %s" (Idx.symbol_ident axis)
              (Sexp.to_string (LL.sexp_of_axis_type ty))
        | _ -> "  <op>"
      in
      p label v)
