(* gh-ocannl-1653: a stored template's components that a later store supersedes are replaced, not
   kept. Each case runs a LINEAGE of routines (one optimize context, the routines virtualized in
   turn) that writes a virtual [x] and reads it into [out], every routine built afresh -- fresh loop
   symbols, as a re-lowering mints them. Before, every store prepended, so recompiling the same code
   added a whole copy of [x]'s components, which every later read replayed.

   - Recompiling: a reset-led accumulation ([zero x; x += v]), a plain overwrite ([x := v]) and a
   two-slice block keep their component counts and their routine sizes, and the plain overwrite
   gains no range guards. - What must survive: an accumulation on top of an inherited value, and a
   second identical one, both contribute; different code is no twin; a plain re-store whose twin
   lies behind an accumulation keeps the twin (dropping it would replay the accumulation ahead of
   the local's first write); a consumer-only routine stores nothing and still replays the producer.

   Every case is executed and held to the MATERIALIZED reading: the same routines run in sequence on
   one context with [x] declared materialized. Producer values [a[i] * (1 + i)] vary with the
   iteration and stay off the zero-init; outputs are seeded with the sentinel. *)
open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level

let dim = 4
let mk = node_factory ~first_id:16530 ~dims:[| dim |] ()
let a_vals = Array.init dim ~f:(fun i -> 10. +. Float.of_int i)

(* [v[i] = a[i] * (1 + i)], the value every producer below writes or adds. *)
let v_vals = Array.mapi a_vals ~f:(fun i a -> a *. Float.of_int (1 + i))

let components (ctx : LL.optimize_ctx) tn =
  Option.value_map (Hashtbl.find ctx.LL.computations tn) ~default:0 ~f:List.length

let size (o : LL.optimized) = count_stmt ~in_scopes:true ~f:(fun _ -> true) o.LL.llc

type nodes = { x : Ir.Tnode.t; a : Ir.Tnode.t; out : Ir.Tnode.t }

let value ~a s = mul (get a [| iter s |]) (tick s)

let read_x { x; out; _ } () =
  let s = sym () in
  loop_n s dim (set out [| iter s |] (get x [| iter s |]))

let reset_acc ({ x; a; _ } as n) () =
  let s = sym () in
  List.reduce_exn ~f:seq
    [
      zero x; loop_n s dim (set x [| iter s |] (add (get x [| iter s |]) (value ~a s))); read_x n ();
    ]

let acc_only ({ x; a; _ } as n) () =
  let s = sym () in
  seq (loop_n s dim (set x [| iter s |] (add (get x [| iter s |]) (value ~a s)))) (read_x n ())

let plain ({ x; a; _ } as n) () =
  let s = sym () in
  seq (loop_n s dim (set x [| iter s |] (value ~a s))) (read_x n ())

let plain_other ({ x; a; _ } as n) () =
  let s = sym () in
  seq (loop_n s dim (set x [| iter s |] (add (value ~a s) (c 1.)))) (read_x n ())

(* Two slices, [x[0..1]] and [x[2..3]] (the second offset by 100), each from its own loop. *)
let block ({ x; a; _ } as n) () =
  let s1 = sym () and s2 = sym () in
  let off = aff [ (1, s2) ] 2 in
  List.reduce_exn ~f:seq
    [
      loop_n s1 2 (set x [| iter s1 |] (value ~a s1));
      loop_n s2 2 (set x [| off |] (add (mul (get a [| off |]) (add (c 3.) (embed s2))) (c 100.)));
      read_x n ();
    ]

let run_case ~label ~tag ~(routines : (nodes -> unit -> LL.t) list) ~counts ~outs
    ?(same_size = false) () =
  let n = { x = mk (tag ^ "x"); a = mk (tag ^ "a"); out = mk (tag ^ "out") } in
  List.iter [ n.a; n.out ] ~f:materialize;
  let seed = [ (n.a, a_vals); (n.out, blank (Array.length a_vals)) ] in
  let ctx = LL.empty_optimize_ctx () in
  let deferred =
    List.mapi routines ~f:(fun k r ->
        let o = optimize_in ctx ~name:(Printf.sprintf "lts_%s_%d" tag k) (r n ()) in
        (o, components ctx n.x))
  in
  p (label ^ ": x is virtual") (known_virtual (fst (List.hd_exn deferred)) n.x);
  p
    (label ^ ": x's template holds "
    ^ String.concat ~sep:", " (List.map counts ~f:Int.to_string)
    ^ " components after each routine")
    (List.equal Int.equal (List.map deferred ~f:snd) counts);
  (if same_size then
     let first = fst (List.hd_exn deferred) in
     p_all (label ^ ": every recompile emits a routine as large as the first, with as many guards")
       (List.tl_exn deferred) ~f:(fun (o, _) ->
         size o = size first && count_where o = count_where first));
  let got =
    List.mapi deferred ~f:(fun k (o, _) ->
        List.hd_exn (execute ~name:(Printf.sprintf "lts_%s_%d" tag k) o ~seed ~read:[ n.out ]))
  in
  p_all2
    (label ^ ": out executes to its expected values")
    (Array.of_list got) (Array.of_list outs) ~f:(Array.equal Float.equal);
  (* The materialized reading: the same routines in sequence on one context, [x] materialized. *)
  materialize n.x;
  let mctx = LL.empty_optimize_ctx () in
  let _, expected =
    List.fold_mapi routines ~init:None ~f:(fun k cctx r ->
        let name = Printf.sprintf "lts_%s_m%d" tag k in
        let o = optimize_in mctx ~name (r n ()) in
        let cctx = run ?ctx:cctx ~name o ~seed in
        (Some cctx, Context.get_values cctx n.out))
  in
  p_all2
    (label ^ ": the reading agrees with the materialized one")
    (Array.of_list got) (Array.of_list expected) ~f:(Array.equal Float.equal)

let () =
  let v = v_vals and times k = Array.map v_vals ~f:(fun e -> Float.of_int k *. e) in
  run_case ~label:"reset-led accumulation, recompiled" ~tag:"rst"
    ~routines:[ reset_acc; reset_acc; reset_acc ]
    ~counts:[ 2; 2; 2 ] ~outs:[ v; v; v ] ~same_size:true ();
  run_case ~label:"plain overwrite, recompiled" ~tag:"pln" ~routines:[ plain; plain; plain ]
    ~counts:[ 1; 1; 1 ] ~outs:[ v; v; v ] ~same_size:true ();
  let block_vals = Array.mapi v ~f:(fun i e -> if i < 2 then e else e +. 100.) in
  run_case ~label:"two-slice block, recompiled" ~tag:"blk" ~routines:[ block; block; block ]
    ~counts:[ 2; 2; 2 ]
    ~outs:[ block_vals; block_vals; block_vals ]
    ~same_size:true ();
  run_case ~label:"accumulations on an inherited value" ~tag:"acc"
    ~routines:[ reset_acc; acc_only; acc_only ] ~counts:[ 2; 3; 4 ]
    ~outs:[ v; times 2; times 3 ]
    ();
  run_case ~label:"different code" ~tag:"dif" ~routines:[ plain; plain_other ] ~counts:[ 1; 2 ]
    ~outs:[ v; Array.map v ~f:(fun e -> e +. 1.) ]
    ();
  run_case ~label:"a plain re-store behind an accumulation" ~tag:"bhd"
    ~routines:[ plain; acc_only; plain ] ~counts:[ 1; 2; 3 ]
    ~outs:[ v; times 2; v ]
    ();
  run_case ~label:"a consumer-only routine" ~tag:"con" ~routines:[ reset_acc; read_x; read_x ]
    ~counts:[ 2; 2; 2 ] ~outs:[ v; v; v ] ()
