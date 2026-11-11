(* A [Where] arm's scope definitions run under the select's condition (gh-ocannl-2145).

   [C_syntax.pp_local_defs] used to emit every scope definition of a statement ahead of it, those of
   a discarded [Where] arm included: a scope in an arm read whatever its addresses were where the
   condition failed, outside the [?:] that short-circuits the arm's own expression. Only the
   inliner's guarded replays were protected (gh-ocannl-2077), by a guard carried into the body;
   hand-built and prelowered IR was not. Now a short-circuiting select's arm definitions render
   under [if (cond)] / [else], and a definition runs under the conditions every occurrence and every
   read of its local share.

   Each row is post-optimize IR reaching the backend through [Ll_test.optimize_scoped]. A row whose
   arm reads outside its node does so where the guard fails by 2^26 cells, so a backend that ran it
   would read hundreds of megabytes off the buffer (on base, a fault on cc at [-O0]); the values,
   checked on every backend against a closed form, can only come out right without that read. The
   emitted-text claims hold the containment itself: every line reading such a node sits inside an
   [if] or [else] block of its statement. A row whose definition must run unconditionally -- a read
   of its local outside the arm, the same scope in both arms, an unconditional reader -- reads in
   range, and its claim is the opposite one. The cross-statement hoist rows run
   [Low_level.hoist_cross_statement_cse] and [eliminate_common_subexpressions] themselves. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Generated.init ~backend_name
let facts = Context.backend_facts_of_name backend_name
let mk = node_factory ~first_id:2_145_000 ~dims:[| 8 |] ()
let mk_int = node_factory ~prec:Ops.int32 ~first_id:2_145_500 ~dims:[| 8 |] ()

(* The iterations of every row, and the cells of each output. *)
let n = 4

(* How far an arm's read lands outside its node where the guard fails. *)
let far = 1 lsl 26

(* Discriminating inputs: distinct per cell, off zero and off the sentinel. *)
let xs = Array.init 8 ~f:(fun k -> Float.of_int ((3 * k) + 1))

(* A virtual node for scope locals. *)
let local_node ?(prec = single) () =
  let v = if Ops.is_float prec then mk "v" else mk_int "v" in
  virtualize v;
  v

(* A scope over a local of [v] (by default a fresh node) summing [x] at [at k] for [k] in [0, 2).
   Two scopes over one node and one input are alpha-equal, which the CSE passes merge. *)
let sum_scope ?(prec = single) ?(v = local_node ~prec ()) x ~at =
  let id = LL.get_scope v in
  let k = sym () in
  let step =
    LL.Set_local (id, LL.Binop (Ops.Add, (LL.Get_local id, prec), (LL.Get (x, [| at k |]), prec)))
  in
  (id, scope_of id (seq (LL.Set_local (id, LL.Constant 0.)) (loop_n k 2 step)))

let in_range i = fun k -> aff [ (1, i); (1, k) ] 0

(* [x] at [far * (i - pass) + k]: in range at [i = pass] only, where it reads [x[k]]. *)
let off_at ~pass i = fun k -> aff [ (far, i); (1, k) ] (-far * pass)
let sum2 base = xs.(base) +. xs.(base + 1)

(* The raw twin [optimize_scoped] needs: reads every input, writes every output. *)
let raw ~inputs ~outputs =
  let i = sym () in
  loop_n i n
    (LL.unflat_lines
       (List.concat_map outputs ~f:(fun out ->
            List.map inputs ~f:(fun x -> set_at out (iter i) (get x [| iter i |])))))

let run_row ~name ~inputs ~outputs llc =
  List.iter (inputs @ outputs) ~f:materialize;
  let o = optimize_scoped ~name ~raw:(raw ~inputs ~outputs) llc in
  let seed = List.map inputs ~f:(fun x -> (x, xs)) @ List.map outputs ~f:(fun o -> (o, blank n)) in
  let got = execute ~name o ~seed ~read:outputs in
  (got, Generated.main_logic (Generated.read name))

(* The kernel lines reading [x], and whether each sits inside an [if] or [else] block. *)
let reads_of source x =
  let lines = Array.of_list (String.split_lines source) in
  (* A read indexes the node's identifier, directly or through a cast (Metal's volatile reads). *)
  let ident = Generated.node_ident ~facts (List.hd_exn x.Tn.label) in
  let reads l =
    String.is_substring l ~substring:(ident ^ "[") || String.is_substring l ~substring:(ident ^ ")[")
  in
  let conditional_blocks =
    Generated.brace_intervals lines
    |> List.filter ~f:(fun (opening, _) ->
        (* The line an [if] test wrapped onto: back to its statement's first line. *)
        let ends_statement j =
          let l = String.rstrip lines.(j) in
          List.exists [ ";"; "{"; "}" ] ~f:(fun suffix -> String.is_suffix l ~suffix)
        in
        let rec first j = if j > 0 && not (ends_statement (j - 1)) then first (j - 1) else j in
        let l = String.strip lines.(first opening) in
        String.is_prefix l ~prefix:"if (" || String.is_prefix l ~prefix:"} else {")
  in
  List.filter_mapi (Array.to_list lines) ~f:(fun j l ->
      if reads l then Some (List.exists conditional_blocks ~f:(fun (o, c) -> o < j && j < c))
      else None)

(* [guarded], except for the statement storing [out], whose expression short-circuits its reads. *)
let guarded_but_store name source x ~out =
  let store = Generated.node_ident ~facts (List.hd_exn out.Tn.label) ^ "[" in
  (* Drop the store statement, from its first line to the one ending it. *)
  let _, lines =
    List.fold (String.split_lines source) ~init:(false, []) ~f:(fun (in_store, acc) l ->
        let in_store = in_store || String.is_substring l ~substring:store in
        if in_store then (not (String.is_suffix (String.rstrip l) ~suffix:";"), acc)
        else (false, l :: acc))
  in
  p_all
    (name ^ ": every read of " ^ Tn.debug_name x ^ " ahead of the store sits under its conditions")
    (reads_of (String.concat ~sep:"\n" (List.rev lines)) x)
    ~f:Fn.id

let guarded name source x =
  p_all
    (name ^ ": every read of " ^ Tn.debug_name x ^ " sits under its condition")
    (reads_of source x) ~f:Fn.id

let unguarded name source x =
  p_none
    (name ^ ": the reads of " ^ Tn.debug_name x ^ " run unconditionally")
    (reads_of source x) ~f:Fn.id

let values name got expected = p (name ^ ": the values are the closed form's") (same got expected)

(* The row's loop over [i], its body built from the symbol. *)
let loop body =
  let i = sym () in
  loop_n i n (body i)

(* The then-arm's scope reads in range at [i = 0] only. *)
let row_then () =
  let name = "was_then" in
  let x = mk "was_then_x" and out = mk ~dims:[| n |] "was_then_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
        set_at out (iter i) (where_ (lt (embed i) (ic 1)) s (add (c 100.) (embed i))))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 101.; 102.; 103. |] ];
  guarded name source x

(* The else-arm's scope reads in range at [i = 3] only. *)
let row_else () =
  let name = "was_else" in
  let x = mk "was_else_x" and out = mk ~dims:[| n |] "was_else_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:3 i) in
        set_at out (iter i) (where_ (lt (embed i) (ic 3)) (add (c 100.) (embed i)) s))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| 100.; 101.; 102.; sum2 0 |] ];
  guarded name source x

(* [i < 3 ? (i < 2 ? 50 + i : s) : 100 + i]: [s] reads in range at [i = 2] only, under the outer
   condition holding and the inner one failing. *)
let row_nested () =
  let name = "was_nested" in
  let x = mk "was_nested_x" and out = mk ~dims:[| n |] "was_nested_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:2 i) in
        set_at out (iter i)
          (where_
             (lt (embed i) (ic 3))
             (where_ (lt (embed i) (ic 2)) (add (c 50.) (embed i)) s)
             (add (c 100.) (embed i))))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| 50.; 51.; sum2 0; 103. |] ];
  guarded name source x

(* An integer compute precision: a select there always short-circuits. *)
let row_integer () =
  let name = "was_integer" in
  let x = mk_int "was_integer_x" and out = mk_int ~dims:[| n |] "was_integer_out" in
  let int32 = Ops.int32 in
  let llc =
    loop (fun i ->
        let _, s = sum_scope ~prec:int32 x ~at:(off_at ~pass:0 i) in
        set_at out (iter i)
          (LL.Ternop
             (Ops.Where, (lt (embed i) (ic 1), iprec ()), (s, int32), (LL.Constant 5., int32))))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 5.; 5.; 5. |] ];
  guarded name source x

(* Two guarded scopes, the one minted first reading the other's local: the body that reads runs
   after the one it reads, inside the same block, though its scope id is the lower one. *)
let row_sibling () =
  let name = "was_sibling" in
  let x = mk "was_sibling_x" and out = mk ~dims:[| n |] "was_sibling_out" in
  let llc =
    loop (fun i ->
        let va = mk "va" in
        virtualize va;
        let a = LL.get_scope va in
        let b, sb = sum_scope x ~at:(off_at ~pass:0 i) in
        (* Two assignments, so the scope cannot collapse into the arm's expression. *)
        let sa =
          scope_of a
            (seq
               (LL.Set_local (a, mul (c 2.) (LL.Get_local b)))
               (LL.Set_local (a, add (LL.Get_local a) (get x [| aff [ (far, i) ] 4 |]))))
        in
        set_at out (iter i) (where_ (lt (embed i) (ic 1)) (add sb sa) (c 9.)))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| (3. *. sum2 0) +. xs.(4); 9.; 9.; 9. |] ];
  guarded name source x

(* [(i < 1 ? s : 7) + s'], [s'] alpha-equal to [s]: the scalar CSE keeps [s] and reads its local for
   [s'], outside the arm, so [s] has to run unconditionally (its reads are in range). Guarding it
   would leave the read with the zero its declaration starts from. *)
let row_cse_use () =
  let name = "was_cse_use" in
  let x = mk "was_cse_use_x" and out = mk ~dims:[| n |] "was_cse_use_out" in
  let llc =
    loop (fun i ->
        let v = local_node () in
        let _, s = sum_scope ~v x ~at:(in_range i) and _, s' = sum_scope ~v x ~at:(in_range i) in
        set_at out (iter i) (add (where_ (lt (embed i) (ic 1)) s (c 7.)) s'))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  p
    (name ^ ": the CSE reads the arm's local outside the arm")
    (count_scalar llc ~f:(function LL.Get_local _ -> true | _ -> false) >= 1);
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ Array.init n ~f:(fun i -> (if i < 1 then sum2 i else 7.) +. sum2 i) ];
  unguarded name source x

(* [i < 1 ? s : s' + 1] with [s'] alpha-equal to [s]: both arms need the one definition, which runs
   whichever arm is taken. *)
let row_both_arms () =
  let name = "was_both_arms" in
  let x = mk "was_both_arms_x" and out = mk ~dims:[| n |] "was_both_arms_out" in
  let llc =
    loop (fun i ->
        let v = local_node () in
        let _, s = sum_scope ~v x ~at:(in_range i) and _, s' = sum_scope ~v x ~at:(in_range i) in
        set_at out (iter i) (where_ (lt (embed i) (ic 1)) s (add s' (c 1.))))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ Array.init n ~f:(fun i -> sum2 i +. if i < 1 then 0. else 1.) ];
  unguarded name source x

(* An unconditional scope whose body reads the local of a scope in an arm: the arm's definition runs
   unconditionally too (its reads are in range). *)
let row_unconditional_reader () =
  let name = "was_reader" in
  let x = mk "was_reader_x" and out = mk ~dims:[| n |] "was_reader_out" in
  let llc =
    loop (fun i ->
        let b, sb = sum_scope x ~at:(in_range i) in
        let va = mk "va" in
        virtualize va;
        let a = LL.get_scope va in
        let sa = scope_of a (LL.Set_local (a, mul (c 3.) (LL.Get_local b))) in
        set_at out (iter i) (add (where_ (lt (embed i) (ic 1)) sb (c 7.)) sa))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ Array.init n ~f:(fun i -> (if i < 1 then sum2 i else 7.) +. (3. *. sum2 i)) ];
  unguarded name source x

(* Two statements sharing an alpha-equal scope only in their arms: the cross-statement hoist would
   lift it ahead of both, unconditionally, so it keeps each statement's copy. *)
let row_hoist_arms () =
  let name = "was_hoist_arms" in
  let x = mk "was_hoist_arms_x" in
  let o1 = mk ~dims:[| n |] "was_hoist_arms_o1" and o2 = mk ~dims:[| n |] "was_hoist_arms_o2" in
  let llc =
    loop (fun i ->
        let v = local_node () in
        let _, s = sum_scope ~v x ~at:(off_at ~pass:0 i)
        and _, s' = sum_scope ~v x ~at:(off_at ~pass:0 i) in
        seq
          (set_at o1 (iter i) (where_ (lt (embed i) (ic 1)) s (c 7.)))
          (set_at o2 (iter i) (where_ (lt (embed i) (ic 1)) s' (c 8.))))
  in
  let llc = LL.hoist_cross_statement_cse llc in
  p
    (name ^ ": no shared definition is hoisted")
    (count_stmt llc ~f:(function LL.Declare_local _ -> true | _ -> false) = 0);
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ o1; o2 ] llc in
  values name got [ [| sum2 0; 7.; 7.; 7. |]; [| sum2 0; 8.; 8.; 8. |] ];
  guarded name source x

(* The same sharing with one member outside every arm: its reads run anyway, so it still hoists. *)
let row_hoist_unconditional () =
  let name = "was_hoist_uncond" in
  let x = mk "was_hoist_uncond_x" in
  let o1 = mk ~dims:[| n |] "was_hoist_uncond_o1" and o2 = mk ~dims:[| n |] "was_hoist_uncond_o2" in
  let llc =
    loop (fun i ->
        let v = local_node () in
        let _, s = sum_scope ~v x ~at:(in_range i) and _, s' = sum_scope ~v x ~at:(in_range i) in
        seq
          (set_at o1 (iter i) (where_ (lt (embed i) (ic 1)) s (c 7.)))
          (set_at o2 (iter i) (add s' (c 1.))))
  in
  let llc = LL.hoist_cross_statement_cse llc in
  p
    (name ^ ": the shared definition is hoisted")
    (count_stmt llc ~f:(function LL.Declare_local _ -> true | _ -> false) = 1);
  let got, _source = run_row ~name ~inputs:[ x ] ~outputs:[ o1; o2 ] llc in
  values name got
    [
      Array.init n ~f:(fun i -> if i < 1 then sum2 i else 7.);
      Array.init n ~f:(fun i -> sum2 i +. 1.);
    ]

(* A value select a blending backend evaluates both arms of: no branch is added, so its arm
   definitions run unconditionally where it blends (cc, HIP), and under its condition where it
   short-circuits (CUDA, Metal). *)
let row_blend () =
  let name = "was_blend" in
  let x = mk "was_blend_x" and m = mk ~dims:[| n |] "was_blend_m" in
  let out = mk ~dims:[| n |] "was_blend_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(in_range i) in
        set_at out (iter i)
          (LL.Ternop
             ( Ops.Where,
               (cmp_at single Ops.Cmplt (get m [| iter i |]) (c 0.5), single),
               (s, single),
               (c 7., single) )))
  in
  List.iter [ x; m; out ] ~f:materialize;
  let o = optimize_scoped ~name ~raw:(raw ~inputs:[ x; m ] ~outputs:[ out ]) llc in
  let mask = [| 0.; 1.; 0.; 1. |] in
  let got = execute ~name o ~seed:[ (x, xs); (m, mask); (out, blank n) ] ~read:[ out ] in
  values name got [ Array.init n ~f:(fun i -> if Float.(mask.(i) < 0.5) then sum2 i else 7.) ];
  let source = Generated.main_logic (Generated.read name) in
  let blends = Ir.Backend_intf.renders_on_host_pool facts || String.equal backend_name "hip" in
  (* One claim name on every backend: the golden is backend-uniform. *)
  let claim =
    name ^ ": the arm's reads follow the select's rendering, blended or short-circuited"
  in
  if blends then p_none claim (reads_of source x) ~f:Fn.id
  else p_all claim (reads_of source x) ~f:Fn.id

(* [relu_gate(1 - i, w)], [w] a [Where] whose condition reads [x] at [far * i]: the gate shuts at [i
   >= 1], where the condition must not be tested at all, so the [Where]'s block sits inside the
   gate's own test (Codex review of the gh-ocannl-2145 PR). The index in the condition keeps it from
   blending. *)
let row_gate () =
  let name = "was_gate" in
  let x = mk "was_gate_x" and out = mk ~dims:[| n |] "was_gate_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
        let cond =
          cmp_at single Ops.Cmplt (c 0.5)
            (add (get x [| aff [ (far, i) ] 0 |]) (mul (c 0.) (embed i)))
        in
        set_at out (iter i)
          (binop Ops.Relu_gate
             (binop Ops.Sub (c 1.) (embed i))
             (LL.Ternop (Ops.Where, (cond, single), (s, single), (c 3., single)))))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 0.; 0.; 0. |] ];
  guarded_but_store name source x ~out

(* [(i < 2 ? (i < 1 ? s : 0) : 0) + (i < 3 ? (i < 1 ? s' : 0) : 0)], [s'] alpha-equal to [s]: the
   CSE shares one definition between the two paths, which runs under their disjunction -- the common
   prefix of the two would be no condition at all, dropping the [i < 1] both share (Codex review of
   the gh-ocannl-2145 PR). *)
let row_disjunction () =
  let name = "was_disjunction" in
  let x = mk "was_disjunction_x" and out = mk ~dims:[| n |] "was_disjunction_out" in
  let llc =
    loop (fun i ->
        let v = local_node () in
        let _, s = sum_scope ~v x ~at:(off_at ~pass:0 i)
        and _, s' = sum_scope ~v x ~at:(off_at ~pass:0 i) in
        let below k = lt (embed i) (ic k) in
        set_at out (iter i)
          (add
             (where_ (below 2) (where_ (below 1) s (c 0.)) (c 0.))
             (where_ (below 3) (where_ (below 1) s' (c 0.)) (c 0.))))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  p
    (name ^ ": the CSE shares the definition between the paths")
    (count_scalar llc ~f:(function LL.Get_local _ -> true | _ -> false) >= 1);
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| 2. *. sum2 0; 0.; 0.; 0. |] ];
  guarded name source x

let () =
  row_then ();
  row_else ();
  row_nested ();
  row_integer ();
  row_sibling ();
  row_cse_use ();
  row_both_arms ();
  row_unconditional_reader ();
  row_hoist_arms ();
  row_hoist_unconditional ();
  row_blend ();
  row_gate ();
  row_disjunction ()
