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
   [Low_level.hoist_cross_statement_cse] and [eliminate_common_subexpressions] themselves.

   The gh-ocannl-2258 rows are the shapes the renderer cannot protect -- a far read whose only
   protection is a gate (which guards no definition), a blendable value select, or an index guard
   not evaluable anywhere under a condition reading a local (gh-ocannl-2311), or a definition also
   read outside its arm -- refused by [Low_level.validate_guarded_reads] at compile and never
   launched, each beside a twin that runs: the same program reading in range, and the protections
   the check credits (an index select inside or around the gate, an [If]). *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Ir.C_syntax.scope_order_oracles_enabled () := true
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

(* gh-ocannl-2213: the scope bodies as [Access_fold] walks them, each paired with the kernel line
   holding its last assignment, and whether they are in text order: {!Scope_emission}'s reader,
   shared with [where_scope_generator]. *)
let body_ends = Scope_emission.body_ends
let in_text_order = Scope_emission.in_text_order
let recorded name = List.Assoc.find !(Ir.C_syntax.scope_order_oracles ()) name ~equal:String.equal

(* The fold's [Emitted] mode, under the oracle the routine compiled with and over the tree it
   rendered, walks every scope body the kernel holds, in the order its text holds them; the golden
   pins how many. [row_emission_order] is the row where the operand order does not. *)
let emission_order_claims ~name source =
  match recorded name with
  | None -> p (name ^ ": the compile recorded its scope emission oracle") false
  | Some (oracle, llc) ->
      let ends = body_ends (Emitted (LL.Scope_order.emission oracle)) llc source in
      (* A row whose definitions all left the tree (hoisted, collapsed) claims nothing here. *)
      if not (List.is_empty ends) then
        p
          (Printf.sprintf
             "%s: Access_fold's Emitted mode walks the %d scope bodies in the kernel's text order"
             name (List.length ends))
          (in_text_order ends)

let run_row ~name ~inputs ~outputs llc =
  List.iter (inputs @ outputs) ~f:materialize;
  let o = optimize_scoped ~name ~raw:(raw ~inputs ~outputs) llc in
  let seed = List.map inputs ~f:(fun x -> (x, xs)) @ List.map outputs ~f:(fun o -> (o, blank n)) in
  let got = execute ~name o ~seed ~read:outputs in
  let source = Generated.main_logic (Generated.read name) in
  emission_order_claims ~name source;
  (got, source)

(* The kernel lines reading [x], and whether each sits inside an [if] or [else] block. *)
let reads_of source x =
  let lines = Array.of_list (String.split_lines source) in
  (* A read indexes the node's identifier, directly or through a cast (Metal's volatile reads). *)
  let ident = Generated.node_ident ~facts (List.hd_exn x.Tn.label) in
  let reads l =
    String.is_substring l ~substring:(ident ^ "[") || String.is_substring l ~substring:(ident ^ ")[")
  in
  (* A statement's first line, back from line [j] over the lines it wrapped from. *)
  let ends_statement j =
    let l = String.rstrip lines.(j) in
    List.exists [ ";"; "{"; "}" ] ~f:(fun suffix -> String.is_suffix l ~suffix)
  in
  let rec first j = if j > 0 && not (ends_statement (j - 1)) then first (j - 1) else j in
  (* A read in an [if] test after a [&&]: the test evaluates it only where the conditions ahead of
     it hold. *)
  let behind_conjunction j =
    let f = first j in
    let text = String.concat ~sep:" " (List.init (j - f + 1) ~f:(fun k -> lines.(f + k))) in
    String.is_prefix (String.strip lines.(f)) ~prefix:"if ("
    &&
    match (String.substr_index text ~pattern:" && ", String.substr_index text ~pattern:ident) with
    | Some conj, Some read -> conj < read
    | _ -> false
  in
  let conditional_blocks =
    Generated.brace_intervals lines
    |> List.filter ~f:(fun (opening, _) ->
        let l = String.strip lines.(first opening) in
        String.is_prefix l ~prefix:"if (" || String.is_prefix l ~prefix:"} else {")
  in
  List.filter_mapi (Array.to_list lines) ~f:(fun j l ->
      if reads l then
        Some
          (List.exists conditional_blocks ~f:(fun (o, c) -> o < j && j < c) || behind_conjunction j)
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

(* The definition reads a buffer. It runs unconditionally with eager blending and under the
   condition on backends that keep loaded arms conditional. *)
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
  let ctx = Context.auto () in
  let blends =
    Ir.Backend_intf.blends_serial_loop_selects (Context.backend_facts ctx)
    && Option.is_some ((Context.codegen_capabilities ctx).Ir.Backend_intf.select_blend single)
  in
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

(* [i < 2 ? r + (i < 1 ? s : 0) + (i <= 0 ? s' : 0) : 0], [s'] alpha-equal to [s], both reading
   [r]'s local: [s] runs under [(i < 2 && i < 1) || (i < 2 && i <= 0)], [r] under [i < 2], which
   absorbs both, and [s] must still run after [r] (Codex review of the gh-ocannl-2145 PR). *)
let row_cross_order () =
  let name = "was_cross_order" in
  let x = mk "was_cross_order_x" and out = mk ~dims:[| n |] "was_cross_order_out" in
  let llc =
    loop (fun i ->
        let r, sr = sum_scope x ~at:(in_range i) in
        let vs = local_node () in
        let s_scope () =
          let id = LL.get_scope vs in
          scope_of id
            (seq
               (LL.Set_local (id, LL.Get_local r))
               (LL.Set_local (id, add (LL.Get_local id) (get x [| aff [ (far, i) ] 0 |]))))
        in
        let s = s_scope () and s' = s_scope () in
        set_at out (iter i)
          (where_
             (lt (embed i) (ic 2))
             (add sr
                (add
                   (where_ (lt (embed i) (ic 1)) s (c 0.))
                   (where_ (le (embed i) (ic 0)) s' (c 0.))))
             (c 0.)))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0 +. (2. *. (sum2 0 +. xs.(0))); sum2 1; 0.; 0. |] ];
  guarded name source x

(* [i < 1 ? (r ? s : 0) : 0], [s] minted before [r]: a test of [r]'s value guards nothing, so [s]
   runs under [i < 1] alone, never testing [r] ahead of [r]'s own definition (Codex review of the
   gh-ocannl-2145 PR, round 2). *)
let row_guard_reads () =
  let name = "was_guard_reads" in
  let x = mk "was_guard_reads_x" and out = mk ~dims:[| n |] "was_guard_reads_out" in
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
        let _, r = sum_scope x ~at:(in_range i) in
        set_at out (iter i)
          (where_
             (lt (embed i) (ic 1))
             (LL.Ternop
                (Ops.Where, (add r (mul (c 0.) (embed i)), single), (s, single), (c 0., single)))
             (c 0.)))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 0.; 0.; 0. |] ];
  guarded name source x

(* [(i < 1 ? (r ? s : 0) : 0) + (3 <= i ? (s' ? r' : 0) : 0)], the primed scopes alpha-equal to the
   others: after the CSE, [r]'s conditions would test [s] and [s]'s test [r], a cycle; tests of
   scope values guard nothing, so none arises (Codex review of the gh-ocannl-2145 PR, round 4). Both
   read in range; the indices in the conditions keep them from blending. *)
let row_guard_cycle () =
  let name = "was_guard_cycle" in
  let x = mk "was_guard_cycle_x" and out = mk ~dims:[| n |] "was_guard_cycle_out" in
  let llc =
    loop (fun i ->
        let vr = local_node () and vs = local_node () in
        let _, r = sum_scope ~v:vr x ~at:(in_range i)
        and _, r' = sum_scope ~v:vr x ~at:(in_range i) in
        let at_s k = aff [ (1, i); (1, k) ] 2 in
        let _, sc = sum_scope ~v:vs x ~at:at_s and _, sc' = sum_scope ~v:vs x ~at:at_s in
        let truth v = add v (mul (c 0.) (embed i)) in
        let sel cond a = LL.Ternop (Ops.Where, (truth cond, single), (a, single), (c 0., single)) in
        set_at out (iter i)
          (add
             (where_ (lt (embed i) (ic 1)) (sel r sc) (c 0.))
             (where_ (le (ic 3) (embed i)) (sel sc' r') (c 0.))))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  let got, _source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| xs.(2) +. xs.(3); 0.; 0.; xs.(3) +. xs.(4) |] ]

(* [(i < 1 ? s : 0) ? r : 0], [r] minted first, [s] copying [r]'s local: [r]'s conditions would test
   [s], whose body reads [r], a cycle closed through a body; the test of a value reading [s] guards
   nothing (Codex review of the gh-ocannl-2145 PR, round 5). *)
let row_mixed_cycle () =
  let name = "was_mixed_cycle" in
  let x = mk "was_mixed_cycle_x" and out = mk ~dims:[| n |] "was_mixed_cycle_out" in
  let llc =
    loop (fun i ->
        let r, sr = sum_scope x ~at:(in_range i) in
        let vs = local_node () in
        let s_id = LL.get_scope vs in
        let ss =
          scope_of s_id
            (seq
               (LL.Set_local (s_id, LL.Get_local r))
               (LL.Set_local (s_id, mul (LL.Get_local s_id) (c 1.))))
        in
        set_at out (iter i) (where_ (where_ (lt (embed i) (ic 1)) ss (c 0.)) sr (c 0.)))
  in
  let got, _source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 0.; 0.; 0. |] ]

(* [(i < 2 ? (r ? (i < 1 ? s : 0) : 0) : 0) + (i <= 0 ? (s' ? r' : 0) : 0)], the primed scopes
   alpha-equal to the others, [s] reading in range at [i = 0] only: the tests of the scopes' values
   guard nothing, but the index guard after one of them still does, so [s] runs under [(i < 2 && i <
   1) || i <= 0] (Codex review of the gh-ocannl-2145 PR). *)
let row_value_test_range () =
  let name = "was_value_test_range" in
  let x = mk "was_value_test_range_x" and out = mk ~dims:[| n |] "was_value_test_range_out" in
  let llc =
    loop (fun i ->
        let vr = local_node () and vs = local_node () in
        let _, r = sum_scope ~v:vr x ~at:(in_range i)
        and _, r' = sum_scope ~v:vr x ~at:(in_range i) in
        let _, sc = sum_scope ~v:vs x ~at:(off_at ~pass:0 i)
        and _, sc' = sum_scope ~v:vs x ~at:(off_at ~pass:0 i) in
        let truth v = add v (mul (c 0.) (embed i)) in
        let sel cond a = LL.Ternop (Ops.Where, (truth cond, single), (a, single), (c 0., single)) in
        set_at out (iter i)
          (add
             (where_ (lt (embed i) (ic 2)) (sel r (where_ (lt (embed i) (ic 1)) sc (c 0.))) (c 0.))
             (where_ (le (embed i) (ic 0)) (sel sc' r') (c 0.))))
  in
  let llc = LL.eliminate_common_subexpressions llc in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0 +. sum2 0; 0.; 0.; 0. |] ];
  guarded name source x

(* [r ? (x[far * i] > 0.5 ? s : 0) : 0], [r] true at [i = 0] only: the test of [r]'s value guards
   nothing, and the nested condition, which reads [x] where only [r] protects it, stays out of every
   test ahead of the statement (Codex review of the gh-ocannl-2145 PR, round 7). *)
let row_value_test_nested () =
  let name = "was_value_test_nested" in
  let x = mk "was_value_test_nested_x" and out = mk ~dims:[| n |] "was_value_test_nested_out" in
  let llc =
    loop (fun i ->
        let vr = local_node () in
        let r = LL.get_scope vr in
        let sr =
          scope_of r
            (seq
               (LL.Set_local (r, c 0.))
               (LL.Set_local (r, add (LL.Get_local r) (cmp_at single Ops.Cmplt (embed i) (c 1.)))))
        in
        let _, sc = sum_scope x ~at:(in_range i) in
        let truth v = add v (mul (c 0.) (embed i)) in
        let inner = cmp_at single Ops.Cmplt (c 0.5) (truth (get x [| aff [ (far, i) ] 0 |])) in
        set_at out (iter i)
          (LL.Ternop
             ( Ops.Where,
               (truth sr, single),
               (LL.Ternop (Ops.Where, (inner, single), (sc, single), (c 0., single)), single),
               (c 0., single) )))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| sum2 0; 0.; 0.; 0. |] ];
  (* The far read: only in the statement's own short-circuiting expression. *)
  let store = Generated.node_ident ~facts (List.hd_exn out.Tn.label) ^ "[" in
  let far_text = Int.to_string far in
  let _, ahead =
    List.fold (String.split_lines source) ~init:(false, []) ~f:(fun (in_store, acc) l ->
        let in_store = in_store || String.is_substring l ~substring:store in
        if in_store then (not (String.is_suffix (String.rstrip l) ~suffix:";"), acc)
        else (false, l :: acc))
  in
  p_none (name ^ ": no test ahead of the statement reads x where only the value test protects it")
    ahead ~f:(fun l -> String.is_substring l ~substring:far_text)

(* gh-ocannl-2213: [(i < 1 ? a + c + d : 0) + b], the scopes minted in the order [a], [c], [d], [b],
   [c]'s body reading [d]'s local. The renderer emits the unconditional [b] first, then the
   conditional ones by ascending id except that [c] follows [d], the definition it reads: [b], [a],
   [d], [c], where the operand order is [a], [c], [d], [b]. *)
let row_emission_order () =
  let name = "was_emission_order" in
  let x = mk "was_emission_order_x" and out = mk ~dims:[| n |] "was_emission_order_out" in
  let llc =
    loop (fun i ->
        let _, a = sum_scope x ~at:(off_at ~pass:0 i) in
        let vc = local_node () in
        let c_id = LL.get_scope vc in
        let d_id, d = sum_scope x ~at:(off_at ~pass:0 i) in
        let cs =
          scope_of c_id
            (seq
               (LL.Set_local (c_id, LL.Get_local d_id))
               (LL.Set_local (c_id, add (LL.Get_local c_id) (get x [| aff [ (far, i) ] 0 |]))))
        in
        let _, b = sum_scope x ~at:(in_range i) in
        set_at out (iter i) (add (where_ (lt (embed i) (ic 1)) (add a (add cs d)) (c 0.)) b))
  in
  let got, source = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| (4. *. sum2 0) +. xs.(0); sum2 1; sum2 2; sum2 3 |] ];
  (* The floor, and the negative control: the [Emitted] walk ends all four bodies, and the
     operand-order walk visits the same four out of the kernel's order. *)
  match recorded name with
  | None -> p (name ^ ": the compile recorded its scope emission oracle") false
  | Some (oracle, rendered) ->
      let ids ends = List.sort (List.map ends ~f:fst) ~compare:Int.compare in
      let emitted = body_ends (Emitted (LL.Scope_order.emission oracle)) rendered source in
      let operand = body_ends Visit rendered source in
      p
        (name ^ ": the Emitted walk ends the same four bodies the operand-order walk visits")
        (List.length emitted = 4 && List.equal Int.equal (ids emitted) (ids operand));
      p
        (name ^ ": the operand-order walk visits them out of the kernel's order")
        (List.length operand = 4 && not (in_text_order operand))

(* gh-ocannl-2241: hand-built conditions differing only in discarded projection operands or
   heterogeneous annotations that compute at the same precision. Compile each beside its
   spelled-alike twin, retaining the symbols, scopes and routine name so the entire main logic must
   match. The mixed statement has two distinct scope definitions: grouping its conditions
   differently is observable to check_scope_order, even though both expressions compute alike. *)
let row_condition_key ~name ~precision ~negative condition =
  let x = mk (name ^ "_x") and out = mk ~dims:[| n |] (name ^ "_out") in
  let i = sym () in
  let _, a = sum_scope x ~at:(in_range i) in
  let _, b = sum_scope x ~at:(in_range i) in
  let first = condition i false and second = condition i true in
  let program second =
    loop_n i n (set_at out (iter i) (add (where_ first a (c 0.)) (where_ second b (c 0.))))
  in
  let _, twin = run_row ~name ~inputs:[ x ] ~outputs:[ out ] (program first) in
  let expect_same =
    if not precision then true
    else
      match recorded name with
      | None -> false
      | Some (oracle, _) -> Ops.equal_prec (oracle.comp_prec Ops.fp8) (oracle.comp_prec single)
  in
  Generated.arm name;
  let got, mixed = run_row ~name ~inputs:[ x ] ~outputs:[ out ] (program second) in
  values name got [ Array.init n ~f:(fun k -> if k < 2 then 2. *. sum2 k else 0.) ];
  p
    (name ^ ": condition text agrees exactly when the backend normalizes annotations alike")
    (Bool.equal expect_same (String.equal twin mixed));
  let distinct = negative i in
  Generated.arm name;
  let _, changed = run_row ~name ~inputs:[ x ] ~outputs:[ out ] (program distinct) in
  p
    (name ^ ": changing a rendered operand or precision changes the main logic")
    (not (String.equal twin changed));
  match recorded name with
  | None -> p (name ^ ": condition-key oracle was recorded") false
  | Some (oracle, _) ->
      let key cond =
        LL.Scope_order.Select { cond; prec = Ops.index_prec (); unconverted = false }
      in
      let equal = LL.Scope_order.equal_key ~comp_prec:oracle.comp_prec in
      p
        (name ^ ": IR key equality follows backend precision normalization")
        (Bool.equal expect_same (equal (key first) (key second)));
      p
        (name ^ ": a rendered difference keeps IR keys apart")
        (not (equal (key first) (key distinct)));
      p (name ^ ": signed zero remains distinct") (not (equal (key (c 0.)) (key (c (-0.)))));
      p
        (name ^ ": explicit rounding retains its exact target")
        (not
           (equal
              (key (LL.Unop (Ops.Round_prec, (embed i, Ops.fp8))))
              (key (LL.Unop (Ops.Round_prec, (embed i, single))))))

let projection_condition ?(limit = 2) op i different =
  let kept = lt (embed i) (ic limit) in
  let dead = (ic (if different then 2 else 1), Ops.index_prec ()) in
  match Ops.binop_conditionality op with
  | Only_first -> LL.Binop (op, (kept, Ops.index_prec ()), dead)
  | Only_second -> LL.Binop (op, dead, (kept, Ops.index_prec ()))
  | Both_operands | Gated_second -> invalid_arg "projection fixture requires a projection"

let precision_condition ?(prec = Ops.fp8) i different =
  LL.Ternop
    ( Ops.Where,
      ( LL.Binop (Ops.Cmplt, (embed i, Ops.index_prec ()), (c 2., single)),
        if different then prec else single ),
      (c 1., single),
      (c 0., single) )

let row_condition_keys () =
  List.iter
    [
      ( "was_key_arg1",
        false,
        projection_condition Ops.Arg1,
        fun i -> projection_condition ~limit:3 Ops.Arg1 i false );
      ( "was_key_arg2",
        false,
        projection_condition Ops.Arg2,
        fun i -> projection_condition ~limit:3 Ops.Arg2 i false );
      ( "was_key_precision",
        true,
        (fun i different -> precision_condition i different),
        fun i -> precision_condition ~prec:Ops.int32 i true );
    ]
    ~f:(fun (name, precision, condition, negative) ->
      try row_condition_key ~name ~precision ~negative condition
      with Invalid_argument message ->
        Stdio.eprintf "%s: %s\n%!" name message;
        p (name ^ ": scope-order check accepts spelled-alike conditions") false)

(* === gh-ocannl-2258: a read no index guard protects where it renders is refused === *)

(* Claims the IR check and the compile gate refuse [llc], naming each of [naming]; never launched,
   so a missing refusal fails a claim rather than faulting. The text goes to stderr: it spells scope
   ids, minted from a process-wide counter. *)
let refused_row ~name ~inputs ~outputs ~naming llc =
  List.iter (inputs @ outputs) ~f:materialize;
  p
    (name ^ ": the IR check names the unprotected read")
    (Option.exists (LL.unguarded_read_violation llc) ~f:(fun msg ->
         String.is_substring msg ~substring:"(gh-ocannl-2258)"));
  let msg =
    match link ~name (optimize_scoped ~name ~raw:(raw ~inputs ~outputs) llc) with
    | _ -> "compiled"
    | exception Invalid_argument msg -> msg
  in
  Stdio.eprintf "%s (not part of the golden): %s\n%!" name msg;
  p
    (name ^ ": the compile gate refuses it")
    (String.is_prefix msg ~prefix:"C_syntax.compile_proc: a read of ");
  p_all ~min:1 (name ^ ": the refusal names the read and what fails to protect it") naming
    ~f:(fun substring -> String.is_substring msg ~substring)

(* A data condition, [m[i] < 1], which mentions no index: blendable at [single]. *)
let data_cond m i = cmp_at single Ops.Cmplt (get m [| iter i |]) (c 1.)
let data_select m i a b = LL.Ternop (Ops.Where, (data_cond m i, single), (a, single), (b, single))
let gate_idx i e = binop Ops.Relu_gate (binop Ops.Sub (c 1.) (embed i)) e

(* The issue's program, [relu_gate(1 - i, (m < 1 ? 3 : s))] with [s] in range at [i = 0] only, the
   one iteration the gate admits: the blended select holds [s]'s definition at no condition, so it
   renders ahead of the statement, outside the gate's test (a SIGBUS on cc). Its twin, [s] in range
   everywhere, runs. *)
let row_gate_blend () =
  let name = "was_gate_blend" in
  let x = mk "was_gate_blend_x" and m = mk "was_gate_blend_m" in
  let out = mk ~dims:[| n |] "was_gate_blend_out" in
  let program at =
    loop (fun i ->
        let _, s = sum_scope x ~at:(at i) in
        set_at out (iter i) (gate_idx i (data_select m i (c 3.) s)))
  in
  refused_row ~name ~inputs:[ x; m ] ~outputs:[ out ]
    (program (off_at ~pass:0))
    ~naming:
      [
        "a read of was_gate_blend_x can be out of range";
        "a backend may evaluate both arms (a blend)";
        "Relu_gate";
        "a gate guards no definition";
      ];
  let name = "was_gate_blend_in_range" in
  let got, _ = run_row ~name ~inputs:[ x; m ] ~outputs:[ out ] (program in_range) in
  (* [m] holds [xs], never below 1: the select takes the scope. *)
  values name got [ [| sum2 0; 0.; 0.; 0. |] ]

(* [relu_gate(1 - i, s)]: the gate is the scope's only protection. *)
let row_gate_direct () =
  let name = "was_gate_direct" in
  let x = mk "was_gate_direct_x" and out = mk ~dims:[| n |] "was_gate_direct_out" in
  refused_row ~name ~inputs:[ x ] ~outputs:[ out ]
    (loop (fun i ->
         let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
         set_at out (iter i) (gate_idx i s)))
    ~naming:[ "a read of was_gate_direct_x"; "Relu_gate"; "a gate guards no definition" ]

(* [(m < 1 ? 3 : s)] with no gate at all: a blend runs both arms. *)
let row_blend_off () =
  let name = "was_blend_off" in
  let x = mk "was_blend_off_x" and m = mk "was_blend_off_m" in
  let out = mk ~dims:[| n |] "was_blend_off_out" in
  refused_row ~name ~inputs:[ x; m ] ~outputs:[ out ]
    (loop (fun i ->
         let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
         set_at out (iter i) (data_select m i (c 3.) s)))
    ~naming:[ "a read of was_blend_off_x"; "a backend may evaluate both arms (a blend)" ]

(* [(i < 1 ? s : 0) + s'], [s'] a read of [s]'s local outside the arm, as the CSE leaves a second
   alpha-equal copy: the definition runs wherever its use does, unprotected. *)
let row_unprotected_use () =
  let name = "was_unprotected_use" in
  let x = mk "was_unprotected_use_x" and out = mk ~dims:[| n |] "was_unprotected_use_out" in
  refused_row ~name ~inputs:[ x ] ~outputs:[ out ]
    (loop (fun i ->
         let id, s = sum_scope x ~at:(off_at ~pass:0 i) in
         set_at out (iter i) (add (where_ (lt (embed i) (ic 1)) s (c 0.)) (LL.Get_local id))))
    ~naming:[ "a read of was_unprotected_use_x"; "in the body of local scope" ]

(* The protections the check credits, each run: [relu_gate(m, (i < 1 ? s : 3))], [(i < 1 ?
   relu_gate(m, s) : 3)], and [if (i < 1) out = s]. *)
let row_protected () =
  let x = mk "was_protected_x" and m = mk "was_protected_m" in
  let gate_m i e = binop Ops.Relu_gate (get m [| iter i |]) e in
  List.iter
    [
      ( "was_protected_in_gate",
        fun out i s -> set_at out (iter i) (gate_m i (where_ (lt (embed i) (ic 1)) s (c 3.))) );
      ( "was_protected_around_gate",
        fun out i s -> set_at out (iter i) (where_ (lt (embed i) (ic 1)) (gate_m i s) (c 3.)) );
      ("was_protected_if", fun out i s -> if_idx (lt (embed i) (ic 1)) (set_at out (iter i) s));
    ]
    ~f:(fun (name, store) ->
      let out = mk ~dims:[| n |] (name ^ "_out") in
      let llc =
        loop (fun i ->
            let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
            store out i s)
      in
      p (name ^ ": the IR check accepts it") (Option.is_none (LL.unguarded_read_violation llc));
      let got, _ = run_row ~name ~inputs:[ x; m ] ~outputs:[ out ] llc in
      (* [m] holds [xs], positive: the gate passes. *)
      let rest = if String.is_suffix name ~suffix:"_if" then sentinel else 3. in
      values name got [ [| sum2 0; rest; rest; rest |] ])

(* [(i < 1 ? s : 0) + r], [r]'s body reading [s]'s local only under [if (i < 1)]: the renderer runs
   a definition wherever a body reading it runs, whatever guards sit inside that body, so [s] runs
   unconditionally. *)
let row_body_reader_guard () =
  let name = "was_body_reader_guard" in
  let x = mk "was_body_reader_guard_x" and out = mk ~dims:[| n |] "was_body_reader_guard_out" in
  refused_row ~name ~inputs:[ x ] ~outputs:[ out ]
    (loop (fun i ->
         let sid, s = sum_scope x ~at:(off_at ~pass:0 i) in
         let r = LL.get_scope (local_node ()) in
         let reader =
           scope_of r
             (seq
                (LL.Set_local (r, c 0.))
                (if_idx (lt (embed i) (ic 1)) (LL.Set_local (r, LL.Get_local sid))))
         in
         set_at out (iter i) (add (where_ (lt (embed i) (ic 1)) s (c 0.)) reader)))
    ~naming:[ "a read of was_body_reader_guard_x"; "is read" ]

(* [relu_gate(1 - i, s)], [s] a single assignment reading [x] in range at [i = 0] only: the collapse
   renders it inline, where the gate's test short-circuits it, so the check on the rendered tree
   accepts it (with routine logging off), and it runs. *)
let row_gate_collapsed () =
  let name = "was_gate_collapsed" in
  let x = mk "was_gate_collapsed_x" and out = mk ~dims:[| n |] "was_gate_collapsed_out" in
  let llc =
    loop (fun i ->
        let id = LL.get_scope (local_node ()) in
        let s = scope_of id (LL.Set_local (id, get x [| aff [ (far, i) ] 0 |])) in
        set_at out (iter i) (gate_idx i s))
  in
  let got, _ = run_row ~name ~inputs:[ x ] ~outputs:[ out ] llc in
  values name got [ [| xs.(0); 0.; 0.; 0. |] ]

(* The entry gate ([Low_level.optimize_proc]), on hand-built trees that go through [optimize]
   themselves, where the rows above swap theirs in after it: [relu_gate(1 - i, s)] with a two-step
   [s] is refused there too, and the single-assignment [s] of [row_gate_collapsed], which the
   collapse renders inline behind the gate's test, is accepted there as at compile, and runs. *)
let row_entry_gate () =
  let name = "was_entry_direct" in
  let x = mk "was_entry_direct_x" and out = mk ~dims:[| n |] "was_entry_direct_out" in
  List.iter [ x; out ] ~f:materialize;
  let llc =
    loop (fun i ->
        let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
        set_at out (iter i) (gate_idx i s))
  in
  let msg = match optimize ~name llc with _ -> "optimized" | exception Invalid_argument m -> m in
  Stdio.eprintf "%s (not part of the golden): %s\n%!" name msg;
  p
    (name ^ ": the entry gate refuses it, naming the read")
    (String.is_prefix msg ~prefix:"Low_level.optimize_proc: a read of was_entry_direct_x ");
  let name = "was_entry_collapsed" in
  let x = mk "was_entry_collapsed_x" and out = mk ~dims:[| n |] "was_entry_collapsed_out" in
  List.iter [ x; out ] ~f:materialize;
  let llc =
    loop (fun i ->
        let id = LL.get_scope (local_node ()) in
        let s = scope_of id (LL.Set_local (id, get x [| aff [ (far, i) ] 0 |])) in
        set_at out (iter i) (gate_idx i s))
  in
  p
    (name ^ ": the entry view credits the single assignment as inline")
    (Option.is_none (LL.unguarded_read_violation ~inline_single_assignments:true llc));
  let _, got = optimize_and_execute ~name llc ~seed:[ (x, xs); (out, blank n) ] ~read:[ out ] in
  values name got [ [| xs.(0); 0.; 0.; 0. |] ]

(* gh-ocannl-2311: [r ? (g ? s : 0) : 0], [r] a value test reading a scope local (true at [i = 0]
   only) and [s] in range at [i = 0] only. Under a condition reading a local, [Scope_order.arms]
   keeps on [s]'s definition only the guards evaluable anywhere, so with [g] spelled [i / 2 < 0.5]
   (a division) [s] would render ahead of [g]'s test; both gates refuse it, naming the dropped
   guard. Lowering builds no such guard as a read's only protection: its range guards are affine
   index comparisons at the read ([value_select_division_guard]). The twin, [g] spelled [i < 1], is
   kept and runs. *)
let row_value_test_dropped () =
  let program ~x ~out inner =
    loop (fun i ->
        let r = LL.get_scope (local_node ()) in
        let sr =
          scope_of r
            (seq
               (LL.Set_local (r, c 0.))
               (LL.Set_local (r, add (LL.Get_local r) (cmp_at single Ops.Cmplt (embed i) (c 1.)))))
        in
        let _, s = sum_scope x ~at:(off_at ~pass:0 i) in
        let truth v = add v (mul (c 0.) (embed i)) in
        set_at out (iter i)
          (LL.Ternop
             ( Ops.Where,
               (truth sr, single),
               (LL.Ternop (Ops.Where, (inner i, single), (s, single), (c 0., single)), single),
               (c 0., single) )))
  in
  let division i = cmp_at single Ops.Cmplt (binop Ops.Div (embed i) (c 2.)) (c 0.5) in
  let x = mk "was_value_test_dropped_x" and out = mk ~dims:[| n |] "was_value_test_dropped_out" in
  refused_row ~name:"was_value_test_dropped" ~inputs:[ x ] ~outputs:[ out ]
    (program ~x ~out division)
    ~naming:
      [
        "a read of was_value_test_dropped_x can be out of range";
        "under an index guard not evaluable anywhere";
        "a select whose condition reads a scope local";
      ];
  (* The entry gate, through [optimize] itself: [refused_row]'s [optimize_scoped] optimizes the raw
     twin and holds only the compile gate to the tree. *)
  let name = "was_value_test_dropped_entry" in
  let xe = mk (name ^ "_x") and oute = mk ~dims:[| n |] (name ^ "_out") in
  List.iter [ xe; oute ] ~f:materialize;
  let msg =
    match optimize ~name (program ~x:xe ~out:oute division) with
    | _ -> "optimized"
    | exception Invalid_argument m -> m
  in
  Stdio.eprintf "%s (not part of the golden): %s\n%!" name msg;
  p
    (name ^ ": the entry gate refuses it, naming the read and the dropped guard")
    (String.is_prefix msg ~prefix:("Low_level.optimize_proc: a read of " ^ name ^ "_x ")
    && String.is_substring msg ~substring:"under an index guard not evaluable anywhere");
  let name = "was_value_test_kept" in
  let got, source =
    run_row ~name ~inputs:[ x ] ~outputs:[ out ]
      (program ~x ~out (fun i -> cmp_at single Ops.Cmplt (embed i) (c 1.)))
  in
  values name got [ [| sum2 0; 0.; 0.; 0. |] ];
  guarded name source x

let () =
  row_condition_keys ();
  row_emission_order ();
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
  row_disjunction ();
  row_cross_order ();
  row_guard_reads ();
  row_guard_cycle ();
  row_mixed_cycle ();
  row_value_test_range ();
  row_value_test_nested ();
  row_value_test_dropped ();
  row_gate_blend ();
  row_gate_direct ();
  row_blend_off ();
  row_unprotected_use ();
  row_protected ();
  row_body_reader_guard ();
  row_gate_collapsed ();
  row_entry_gate ()
