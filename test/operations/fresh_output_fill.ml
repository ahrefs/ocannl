(* gh-ocannl-1873: discriminating by default, for every executed test.

   [Ll_test.run_linked] starts the nodes a hand-built routine writes first from a prefill
   (gh-ocannl-1774), but a test that drives [Context.run] itself, as every tensor-level test does,
   still starts them from allocation's zero, and a routine that wrongly drops a zero computes the
   same values from that zero. The [debug_fill_fresh_outputs] key, on in [test/config], makes
   [Context.run] start them from [Ir.Ops.debug_fill_value] instead.

   Part 1 pins the contract on hand-built IR, on any backend: what the key reaches (a buffer nothing
   wrote yet, of a node written first, or embedded and read first with nothing declaring it starts
   from zero: gh-ocannl-1966), what it leaves alone (a seeded node, a node a run wrote, declared
   state, a padding margin), the pattern at every storage precision, and that the key is what does
   it: a child process with the key off reads allocation's zero from the same program.

   Part 2 shows a correct routine does not notice: a default-pipeline matmul and a reduction, with
   no seed for their outputs, against closed forms. Correct routines in the undeclared class, which
   is why it is filled and not refused, are [resnet_block]'s and [test_padding_reset]'s. The
   reduction is the mutants' target: an init dropped before the analysis (A01 of gh-ocannl-1873)
   reads its target first, so with nothing here seeding it only the fill shows the drop. Their
   tables are in the PR bodies.

   Part 3 fills liveness-aliased buffers: a dropped zero on the first occupant of a shared range
   reaches an observable output as NaN, where it used to read the arena's zero. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Asgns = Ir.Assignments

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let child_flag = "--fresh-output-fill-child"
let node = L.node_factory ~first_id:1873000 ~dims:[| 4 |] ()
let x = node "fof_x"
let y = node "fof_y"
let () = List.iter [ x; y ] ~f:L.materialize
let xv = [| 10.; 20.; 30.; 40. |]

(* [y[i] = x[i] + 1] over the first two cells only: [y] is written first and never read, and its
   cells 2 and 3 are covered by no writer, so what they read back is what the buffer started
   from. *)
let partial_write () =
  let i = L.sym () in
  L.optimize ~name:"fof_partial"
    (L.loop ~upto:1 i (L.set_at y (L.iter i) (L.add (L.get x [| L.iter i |]) (L.c 1.))))

let covered = [| 11.; 21. |]
let uncovered vs = Array.sub vs ~pos:2 ~len:2
let written vs = Array.sub vs ~pos:0 ~len:2

(* Each leg links in a root of its own: a root's device buffers outlive its context values. *)
let linked o = L.link ~ctx:(Context.auto ()) ~name:"fof_partial" o

let run_seeded (ctx, routine) seed =
  Context.run
    (List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs))
    routine

(* Embedded state incremented cell by cell: read first, in the routine's own computation. *)
let bump_routine state ~name =
  L.materialize state;
  let i = L.sym () in
  let bump =
    L.optimize ~name
      (L.loop ~upto:3 i (L.set_at state (L.iter i) (L.add (L.get state [| L.iter i |]) (L.c 1.))))
  in
  let comp = { Asgns.empty_comp with embedded_nodes = Set.singleton (module Tn) state } in
  Context.compile ~name ~prelowered:bump
    ~lowered_transform:(fun o -> [ o ])
    (Context.auto ()) comp Ir.Indexing.Empty

(* Hex, so the parent's claim is on the bits the child read rather than on a rounded rendering. *)
let child () =
  let got = Context.get_values (run_seeded (linked (partial_write ())) [ (x, xv) ]) y in
  let state = node "fof_child_state" in
  let ctx, routine = bump_routine state ~name:"fof_child_bump" in
  let bumped = Context.get_values (Context.run ctx routine) state in
  Array.iter
    (Array.concat [ written got; uncovered got; bumped ])
    ~f:(fun v -> Stdio.printf "cell\t%h\n" v)

let part1 () =
  let key = Utils.get_global_flag ~default:false ~arg_name:"debug_fill_fresh_outputs" in
  p "test/config turns the key on, so that every executed test runs under it" key;
  let o = partial_write () in
  let ((_, routine) as pair) = linked o in
  p "a node the routine writes before reading is written first" (Set.mem routine.written_first y);
  p "a node the routine reads is not" (not (Set.mem routine.written_first x));
  let got = Context.get_values (run_seeded pair [ (x, xv) ]) y in
  p_all2 "the routine's own writes land over the fill" (written got) covered ~f:Float.equal;
  p_all "cells no writer covers read back the fill" (Array.to_list (uncovered got)) ~f:Float.is_nan;
  (* The buffer, not the node: a seed, a run and an earlier routine each keep it. *)
  let got = Context.get_values (run_seeded (linked o) [ (x, xv); (y, L.blank 4) ]) y in
  p_all2 "a seeded node keeps its seed" (uncovered got) [| L.sentinel; L.sentinel |] ~f:Float.equal;
  let ((_, routine) as pair) = linked o in
  let ctx1 = run_seeded pair [ (x, xv) ] in
  let ctx2 = Context.set_values ctx1 y (L.blank 4) in
  let got = Context.get_values (Context.run ctx2 routine) y in
  p_all2 "a second run does not fill over the values the buffer holds" (uncovered got)
    [| L.sentinel; L.sentinel |] ~f:Float.equal;
  let i = L.sym () in
  let whole = L.optimize ~name:"fof_whole" (L.loop ~upto:3 i (L.set_at y (L.iter i) (L.embed i))) in
  let ctx, first = L.link ~ctx:(Context.auto ()) ~name:"fof_whole" whole in
  let ctx, second = L.link ~ctx ~name:"fof_partial" o in
  let ctx = Context.set_values ctx x xv in
  let got = Context.get_values (Context.run (Context.run ctx first) second) y in
  p_all2 "a node an earlier routine wrote keeps what it wrote" (uncovered got) [| 2.; 3. |]
    ~f:Float.equal;
  (* Host-initialized: the node holds a value the routine did not put there. *)
  let hosted = node "fof_hosted" in
  L.materialize hosted;
  let hosted_values = [| 1.; 2.; 3.; 4. |] in
  let nd = Ir.Ndarray.create_array ~debug:"fof_hosted" Ops.single ~dims:[| 4 |] ~padding:None in
  Ir.Ndarray.set_flat_values nd hosted_values;
  Ir.Host_inits.register hosted (lazy nd);
  let i = L.sym () in
  let over =
    L.optimize ~name:"fof_hosted" (L.loop ~upto:1 i (L.set_at hosted (L.iter i) (L.c 9.)))
  in
  let ctx, routine = L.link ~ctx:(Context.auto ()) ~name:"fof_hosted" over in
  p "a host-initialized node is not written first" (not (Set.mem routine.written_first hosted));
  let got = Context.get_values (Context.run ctx routine) hosted in
  p_all2 "a host-initialized node keeps its host values" (uncovered got) (uncovered hosted_values)
    ~f:Float.equal;
  (* Read first: state the routine starts from allocation's zero by declaration (an embedded node
     marked zero-valid, as [Train.sgd_one]'s momentum), which the required-input set leaves out. *)
  let state = node "fof_state" in
  Tn.mark_zero_valid state;
  let ctx, routine = bump_routine state ~name:"fof_bump" in
  p "embedded state read first is no required input" (not (Set.mem routine.inputs state));
  p "embedded state read first is not written first" (not (Set.mem routine.written_first state));
  p "declared state is no undeclared read" (not (Set.mem routine.read_first_undeclared state));
  let ctx = Context.run ctx routine in
  p_all "declared state read first starts from zero"
    (Array.to_list (Context.get_values ctx state))
    ~f:(Float.equal 1.);
  p_all "declared state carries its value to the next run"
    (Array.to_list (Context.get_values (Context.run ctx routine) state))
    ~f:(Float.equal 2.);
  let state = node "fof_seeded_state" in
  Tn.mark_zero_valid state;
  let ctx, routine = bump_routine state ~name:"fof_seeded_bump" in
  p_all "declared state keeps a seed"
    (Array.to_list
       (Context.get_values
          (Context.run (Context.set_values ctx state (Array.create ~len:4 5.)) routine)
          state))
    ~f:(Float.equal 6.);
  (* The same routine, undeclared: what an accumulation that lost its init looks like. *)
  let lost = node "fof_lost" in
  let ctx, routine = bump_routine lost ~name:"fof_lost_bump" in
  p "undeclared embedded state read first is no required input" (not (Set.mem routine.inputs lost));
  p "nor is it written first" (not (Set.mem routine.written_first lost));
  p "it is an undeclared read" (Set.mem routine.read_first_undeclared lost);
  p_all "undeclared state read first reads the fill"
    (Array.to_list (Context.get_values (Context.run ctx routine) lost))
    ~f:Float.is_nan;
  let lost = node "fof_lost_seeded" in
  let ctx, routine = bump_routine lost ~name:"fof_lost_seeded_bump" in
  p_all "undeclared state keeps a seed"
    (Array.to_list
       (Context.get_values
          (Context.run (Context.set_values ctx lost (Array.create ~len:4 5.)) routine)
          lost))
    ~f:(Float.equal 6.)

(* The declaration through [%cd]: the field goes to the node, not to [term]. *)
let declared_inline () =
  let run_ctr ~name comp =
    let ctx, routine = Context.compile ~name (Context.auto ()) comp Ir.Indexing.Empty in
    let ctr = Set.choose_exn routine.outputs in
    (routine, ctr, Context.get_values (Context.run ctx routine) ctr)
  in
  let routine, ctr, got =
    run_ctr ~name:"fof_ctr"
      [%cd { fof_ctr; o = [ 2 ]; zero_valid = true } =: (!.1.0 *. fof_ctr) + 1]
  in
  p "{ x; zero_valid = true } marks the node" (Tn.is_zero_valid ctr);
  p "so it is no undeclared read" (not (Set.mem routine.read_first_undeclared ctr));
  p_all "and starts from zero" (Array.to_list got) ~f:(Float.equal 1.);
  let routine, ctr, got =
    run_ctr ~name:"fof_plain" [%cd { fof_plain; o = [ 2 ] } =: (!.1.0 *. fof_plain) + 1]
  in
  p "{ x } alone declares nothing" (Set.mem routine.read_first_undeclared ctr);
  p_all "so it reads the fill" (Array.to_list got) ~f:Float.is_nan

(* A padded node's interior starts from the fill, and its margins from the padding value: what a
   reader of a margin must read, written or not. Cells [1] and [2] are written, of the padded
   [0..5]; a second routine reads the left margin and an unwritten interior cell. *)
let padded () =
  let pad =
    Tn.create (Tn.Specified Ops.single) ~id:1873050 ~label:[ "fof_padded" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy (Some ([| Ops.{ left = 1; right = 1 } |], 7.)))
      ()
  in
  let margin = node "fof_margin" in
  List.iter [ pad; margin ] ~f:L.materialize;
  let i = L.sym () in
  let write =
    L.optimize ~name:"fof_pad_write" (L.loop ~upto:1 i (L.set_at pad (L.aff [ (1, i) ] 1) (L.c 5.)))
  in
  let read =
    L.optimize ~name:"fof_pad_read"
      (L.seq
         (L.set_at margin (L.fixed 0) (L.get pad [| L.fixed 0 |]))
         (L.set_at margin (L.fixed 1) (L.get pad [| L.fixed 3 |])))
  in
  let ctx, first = L.link ~ctx:(Context.auto ()) ~name:"fof_pad_write" write in
  let ctx, second = L.link ~ctx ~name:"fof_pad_read" read in
  let got = Context.get_values (Context.run (Context.run ctx first) second) margin in
  p "a padded node's margin keeps the padding value" (Float.equal got.(0) 7.);
  p "a padded node's unwritten interior cell reads the fill" (Float.is_nan got.(1))

(* The pattern at one storage precision: cell 0 written, cell 1 the fill as the host reads it. *)
let precision_leg ~first_id prec =
  let name = "fof_prec_" ^ Ops.prec_string prec in
  let tn = (L.node_factory ~prec ~first_id ~dims:[| 2 |] ()) name in
  L.materialize tn;
  let o = L.optimize ~name (L.set_at tn (L.fixed 0) (L.c 1.)) in
  let ctx, routine = L.link ~ctx:(Context.auto ()) ~name o in
  let got = Context.get_values (Context.run ctx routine) tn in
  (got.(0), got.(1), Ops.debug_fill_value prec)

let precisions () =
  let leg = Verdict.leg ~when_:(not (String.equal backend_name "metal")) ~on:"metal" () in
  let check ~first_id prec label =
    let cell0, fill, pattern = precision_leg ~first_id prec in
    p (label ^ ": the routine's write lands") (Float.equal cell0 1.);
    match pattern with
    | Some v when Float.is_nan v -> p (label ^ ": an unwritten cell reads NaN") (Float.is_nan fill)
    | Some v ->
        p (label ^ ": an unwritten cell reads the integer pattern") (Float.equal fill v);
        p (label ^ ": the pattern is clear of zero") Float.(v <> 0.)
    | None -> failwith (label ^ ": a scalar storage has a fill value")
  in
  List.iteri
    [
      (Ops.single, "single");
      (Ops.half, "half");
      (Ops.bfloat16, "bfloat16");
      (Ops.fp8, "fp8");
      (Ops.byte, "byte");
      (Ops.uint16, "uint16");
      (Ops.int32, "int32");
      (Ops.uint32, "uint32");
      (Ops.int64, "int64");
      (Ops.uint64, "uint64");
    ]
    ~f:(fun i (prec, label) -> check ~first_id:(1873100 + (10 * i)) prec label);
  Verdict.in_leg ~leg "double: an unwritten cell reads NaN" (fun label ->
      let _, fill, _ = precision_leg ~first_id:1873300 Ops.double in
      p label (Float.is_nan fill));
  p "packed RNG state holds no scalar to fill" (Option.is_none (Ops.debug_fill_value Ops.uint4x32))

(* {1 The key is what does it} *)
let key_off () =
  let ((status, out, _) as run) =
    Fresh_process.run [ child_flag; "--ocannl_debug_fill_fresh_outputs=false" ]
  in
  (match status with
  | Unix.WEXITED 0 -> ()
  | _ -> Fresh_process.report ~label:"the key-off child" run);
  let cells =
    List.filter_map (String.split_lines out) ~f:(fun line ->
        match String.split line ~on:'\t' with
        | [ "cell"; v ] -> Some (Float.of_string v)
        | _ -> None)
  in
  p_all2 "with the key off, the same program leaves allocation's zero in the unwritten cells"
    (Array.sub (Array.of_list cells) ~pos:0 ~len:4)
    (Array.append covered [| 0.; 0. |])
    ~f:Float.equal;
  p_all2 "with the key off, undeclared state starts from allocation's zero"
    (Array.sub (Array.of_list cells) ~pos:4 ~len:4)
    (Array.create ~len:4 1.) ~f:Float.equal

(* {1 Part 2: a correct routine does not notice} *)

(* As [default_prefill]'s Part 2, driven through [Context.run] with no seed for the output, past the
   GPU pipeline's [min_parallel] so a GPU backend maps its output loops onto the grid. *)
let m, k, n = (8, 5, 12)
let a_init idcs = L.cycle ~dims:[| m; k |] ~modulus:7 ~offset:1. ~stride:1. idcs
let b_init idcs = L.cycle ~dims:[| k; n |] ~modulus:11 ~offset:(-4.) ~stride:1. idcs
let products = L.matmul_products ~m ~n ~k a_init b_init
let closed_form = Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))

let part2 () =
  p_all "the operands' products sum exactly in f32" products
    ~f:(L.exact_partials ~any_sign:true ~sig_bits:(Ops.significand_bits Ops.single));
  let ma = NTDSL.init ~l:"fof_a" ~prec:Ops.single ~o:[ m ] ~i:[ k ] ~f:a_init () in
  let mb = NTDSL.init ~l:"fof_b" ~prec:Ops.single ~o:[ k ] ~i:[ n ] ~f:b_init () in
  let%op mc = ma * mb in
  let ctx, routine = Context.compile (Context.auto ()) (Train.forward mc) Ir.Indexing.Empty in
  p "the matmul's output is written first" (Set.mem routine.written_first mc.Tensor.value);
  p_all2 "the default pipeline's matmul from the fill is the closed form"
    (Context.get_values (Context.run ctx routine) mc.Tensor.value)
    closed_form ~f:Float.equal;
  (* Non-injective, so its init is owed for the accumulation and for nothing else. *)
  let ra = NTDSL.init ~l:"fof_ra" ~prec:Ops.single ~o:[ m; k ] ~f:a_init () in
  let%op r = ra ++ "ab => a" in
  Train.set_materialized r.Tensor.value;
  let ctx, routine = Context.compile (Context.auto ()) (Train.forward r) Ir.Indexing.Empty in
  p "the reduction's output is written first" (Set.mem routine.written_first r.Tensor.value);
  p_all2 "an unseeded reduction from the fill is the closed form"
    (Context.get_values (Context.run ctx routine) r.Tensor.value)
    (Array.init m ~f:(fun i ->
         List.sum (module Float) (List.range 0 k) ~f:(fun j -> a_init [| i; j |])))
    ~f:Float.equal

(* {1 Part 3: liveness-aliased buffers} *)

(* A chain of matrix-vector products: each output is written from its [Zero_out] on, so the outer
   two are liveness-disjoint, and [h1], being first, is the range's first occupant. [h3] is shorter,
   so the two ranges differ in size. A hand-cut segment per product, for a backend whose liveness is
   per segment. *)
let segments (o : LL.optimized) =
  let flush seg acc = if List.is_empty seg then acc else List.rev seg :: acc in
  let rec cut seg acc = function
    | [] -> List.rev (flush seg acc)
    | (LL.For_loop _ as loop) :: rest -> cut [] (flush (loop :: seg) acc) rest
    | stmt :: rest -> cut (stmt :: seg) acc rest
  in
  List.map
    (cut [] [] (LL.flat_lines [ o.LL.llc ]))
    ~f:(fun lines -> { o with LL.llc = LL.unflat_lines lines })

let drop_zero tn found (o : LL.optimized) =
  (LL.get_node o.LL.traced_store tn).LL.zero_initialized_by_code <- false;
  let keep = function
    | LL.Zero_out t when Tn.equal t tn ->
        found := true;
        false
    | _ -> true
  in
  { o with LL.llc = LL.unflat_lines (List.filter (LL.flat_lines [ o.LL.llc ]) ~f:keep) }

let chain_dims = (8, 16, 12)
let chain_entry idcs = L.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-1.) ~stride:1. idcs

let chain ~label =
  let n, m1, m3 = chain_dims in
  let init ~l ~o ~i =
    NTDSL.init ~l:(label ^ l) ~prec:Ops.single ~o:[ o ] ~i:[ i ] ~f:chain_entry ()
  in
  let v = NTDSL.init ~l:(label ^ "x") ~prec:Ops.single ~o:[ n ] ~f:(fun _ -> 1.) () in
  let a1 = init ~l:"a1" ~o:m1 ~i:n and a2 = init ~l:"a2" ~o:n ~i:m1 in
  let a3 = init ~l:"a3" ~o:m3 ~i:n in
  let%op h1 = a1 * v in
  let%op h2 = a2 * h1 in
  let%op h3 = a3 * h2 in
  let%op out = h3 ++ "i=>0" in
  List.iter [ h1; h2; h3 ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  Tn.set_observable out.Tensor.value;
  (h1, h3, out)

let chain_closed_form () =
  let n, m1, m3 = chain_dims in
  let apply ~rows ~cols x =
    Array.init rows ~f:(fun r ->
        List.sum (module Float) (List.range 0 cols) ~f:(fun c -> chain_entry [| r; c |] *. x.(c)))
  in
  let h1 = apply ~rows:m1 ~cols:n (Array.create ~len:n 1.) in
  let h2 = apply ~rows:n ~cols:m1 h1 in
  Array.fold (apply ~rows:m3 ~cols:n h2) ~init:0. ~f:( +. )

let aliased () =
  Unix.putenv "OCANNL_BUFFER_ALIASING" "true";
  let aliased ctx (t : Tensor.t) =
    match Context.observability ctx t.Tensor.value with
    | Unobservable_aliased [ v ] -> Tn.equal v t.Tensor.value
    | _ -> false
  in
  let keep = NTDSL.term ~label:[ "fof_keep" ] ~output_dims:[ 2 ] ?fetch_op:None () in
  let h1, h3, out = chain ~label:"fof_good_" in
  let comp = Asgns.sequence [ Train.forward out; [%cd keep =+ 1] ] in
  let ctx, routine =
    Context.compile ~lowered_transform:segments (Context.auto ()) comp Ir.Indexing.Empty
  in
  p "the first occupant of a shared range is aliased" (aliased ctx h1);
  p "and so is the shorter node over it" (aliased ctx h3);
  p "the aliased node is written first" (Set.mem routine.written_first h1.Tensor.value);
  p "the host's own write to an aliased node stays refused"
    (try
       ignore (Context.set_values ctx h1.Tensor.value (Array.create ~len:16 0.) : Context.t);
       false
     with Utils.User_error msg -> String.is_substring msg ~substring:"buffer-aliased");
  let ctx = Context.set_values ctx keep.Tensor.value [| 5.; 5. |] in
  let expected = chain_closed_form () in
  let ctx = Context.run ctx routine in
  p "a correct chain from the fill is the closed form"
    (Float.equal (Context.get_values ctx out.Tensor.value).(0) expected);
  p_all "a seeded node beside the arena keeps its value"
    (Array.to_list (Context.get_values ctx keep.Tensor.value))
    ~f:(Float.equal 6.);
  p "a second run, which is not filled, is the closed form too"
    (Float.equal (Context.get_values (Context.run ctx routine) out.Tensor.value).(0) expected);
  let h1, _, out = chain ~label:"fof_lost_" in
  let found = ref false in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> segments (drop_zero h1.Tensor.value found o))
      (Context.auto ()) (Train.forward out) Ir.Indexing.Empty
  in
  p "the transform dropped the first occupant's zero" !found;
  p "that node is aliased" (aliased ctx h1);
  p "its dropped zero reaches an observable output as the fill"
    (Float.is_nan (Context.get_values (Context.run ctx routine) out.Tensor.value).(0));
  Unix.putenv "OCANNL_BUFFER_ALIASING" "false"

let () =
  if Array.mem Stdlib.Sys.argv child_flag ~equal:String.equal then child ()
  else (
    part1 ();
    declared_inline ();
    padded ();
    precisions ();
    key_off ();
    part2 ();
    aliased ();
    Stdio.eprintf "fresh_output_fill: backend %s (not part of the golden)\n%!" backend_name)
