(* Precision-neutral accumulator localization (gh-ocannl-693).

   A recognized serial reduction nest holds its accumulator in a scope LOCAL across the whole nest
   and stores once, after the nest — at EVERY precision, not only where the numerics policy asks for
   a widening (gh-ocannl-639). Before this, [C_syntax.try_localize_serial_reduce] declined whenever
   [acc_prec] was the identity on the storage precision, so an f32 reduction that no schedule op
   reached did one global read-modify-write per inner iteration: a scalar loss reduction, a gradient
   accumulated over a batch loop, an unmatched contraction. On Metal that residency was not merely
   likely but guaranteed — [volatile_serial_accumulation] shadows exactly that statement shape with
   a volatile-qualified alias, to stop the shader compiler touching it.

   The claims are therefore both structural and executed:

   - the emitted f32 kernel forwards the lowered zero-init directly into a local, updates the local
   inside the reduction loop, and writes the node once — the dead device zeroing store and the
   opening node read are both absent, with no [acc[k] = ... acc[k] ...] inside the loop; - the
   workaround decision agrees with what the compile REPORTED (gh-ocannl-782): the routine's
   volatility census names the local, says whether its accumulating device reads use the workaround,
   and says whether this backend asked for the workaround at all — so the expectation is read off
   the compile instead of off the backend's name, and the emitted text is checked against it. On a
   backend that requests the workaround the accumulator stays plain and register-resident while the
   accumulating loop's device reads use expression-level [volatile] pointer casts (gh-ocannl-820);
   on one that does not, both stay plain. Either way the kernel carries no RMW shadow declaration:
   localization lifted that device-memory RMW, and the census's RMW-read count says so independently
   of the text; - the values match a host reference computed in the same summation order. The
   producers discriminate: every element is [1 + 10*i + j], so it varies with BOTH loop symbols and
   is clear of the zero the accumulator is initialized to — a constant producer would survive a
   dropped or replayed iteration, and a value omitting a symbol would survive a wrong substitution.

   Two nest shapes, because they exercise different placements of the localized store: a scalar
   reduction over both axes (the loss shape — the store lands above every loop, at function scope)
   and a row reduction (the batch-gradient / contraction shape — the store lands inside the
   surviving output loop, whose symbol the cell DOES mention, so the volatile predicate is false
   there too). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Test_utils.Generated.init ~backend_name
let facts = Context.backend_facts_of_name backend_name
let rows = 6
let cols = 7

(* Uploads per-cell values off every reduction's result (they are negative, the results positive)
   into an accumulator between its compile and its run: a fresh device buffer is already zero, so
   without it a dropped or wrongly forwarded zero-init would match the host reference anyway
   (gh-ocannl-1625). *)
let prefill ctx tn =
  Context.set_values ctx tn (Array.init (Tn.num_elems tn) ~f:(fun k -> -.(100. +. Float.of_int k)))

(* Discriminating producer: varies with both symbols, never zero. *)
let elem i j = 1.0 +. (10.0 *. Float.of_int i) +. Float.of_int j
let data = Array.init (rows * cols) ~f:(fun n -> elem (n / cols) (n % cols))

(* A device read rendered through Metal's expression-level volatile pointer cast has [ident)[idx]]
   rather than [ident[idx]]. Count both as one semantic node access. *)
let count_node_accesses source ident =
  List.sum
    (module Int)
    [ ident ^ "["; ident ^ ")[" ]
    ~f:(fun pattern -> List.length (String.substr_index_all source ~may_overlap:false ~pattern))

let x =
  TDSL.ndarray data ~label:[ "resx" ] ~batch_dims:[] ~input_dims:[] ~output_dims:[ rows; cols ] ()

(* Leg 1: scalar reduction over both axes — the loss shape. *)
let%op total = x ++ "ij => 0"

(* Leg 2: reduction over the row axis only — the batch-gradient / contraction shape. *)
let%op per_col = x ++ "ij => j"

(* Leg 3: an index-only scalar reduction. Its localized update has no materialized device read, so
   even Metal selects no expression to cast. This is the negative control for the census: requested
   capability is not the same as an emitted workaround site (Codex P2, review round 1 on #553). *)
let indices = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ rows; cols ] ()
let%op index_total = indices ++ "ij => 0"

let () =
  Train.set_materialized total.Tensor.value;
  Train.set_materialized per_col.Tensor.value;
  Train.set_materialized index_total.Tensor.value;
  let ctx = Context.auto () in
  let ctx, r_total =
    Context.compile ~name:"res_total" ctx (Train.forward total) Ir.Indexing.Empty
  in
  let ctx = Context.run (prefill ctx total.Tensor.value) r_total in
  let ctx, r_per_col =
    Context.compile ~name:"res_per_col" ctx (Train.forward per_col) Ir.Indexing.Empty
  in
  let ctx = Context.run (prefill ctx per_col.Tensor.value) r_per_col in
  let ctx, r_index_total =
    Context.compile ~name:"res_index_total" ctx (Train.forward index_total) Ir.Indexing.Empty
  in
  let ctx = Context.run (prefill ctx index_total.Tensor.value) r_index_total in
  let got_total = (Context.get_values ctx total.Tensor.value).(0) in
  let got_per_col = Context.get_values ctx per_col.Tensor.value in
  let got_index_total = (Context.get_values ctx index_total.Tensor.value).(0) in

  (* Host reference, same summation order as the emitted nest. *)
  let ref_total =
    let acc = ref 0.0 in
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        acc := !acc +. elem i j
      done
    done;
    !acc
  in
  let ref_per_col =
    Array.init cols ~f:(fun j ->
        let acc = ref 0.0 in
        for i = 0 to rows - 1 do
          acc := !acc +. elem i j
        done;
        !acc)
  in
  Verdict.p "scalar reduction matches host reference" Float.(abs (got_total - ref_total) < 1e-3);
  Verdict.p_close "row reduction matches host reference" got_per_col ref_per_col ~tol:(fun _ ->
      1e-3);
  let ref_index_total = Float.of_int (rows * cols * ((rows * cols) - 1) / 2) in
  Verdict.p "index-only reduction matches host reference"
    Float.(abs (got_index_total - ref_index_total) < 1e-3);

  (* The reference must be able to tell a dropped-iteration kernel apart from a correct one: with
     [rows * cols] terms all distinct and nonzero, a last-iteration-only result is far away. *)
  let last_only = elem (rows - 1) (cols - 1) in
  Verdict.p "reference discriminates a last-iteration-only result"
    Float.(abs (ref_total - last_only) > 1.0);

  (* Structural: the localized form. The scope local is [v<scope_id>_<ident>], where [<ident>] is
     the code name codegen derived for the node -- not predictable from the tensor's label (it goes
     through a blacklist and a dot-stripping pass), so it is READ OFF the zero-seeded local rather
     than guessed. If no statement of the shape [v<n>_<ident> = ...0.0...] exists, the accumulator
     was not localized with the zero forwarded into it, which is the failure this test exists to
     catch.

     Split on [;] rather than on newlines: the pretty-printer breaks a long value expression across
     lines, so a line-based read of "does this statement touch the node twice" would answer no for
     exactly the wide read-modify-write it is meant to catch. *)
  let normalize st = String.concat ~sep:" " (String.split_on_chars st ~on:[ '\n'; '\t' ]) in
  let zero_scope_init st =
    match String.substr_index st ~pattern:" = " with
    | None -> None
    | Some at ->
        (* Splitting on [;] carries a [for] header's tail into the next statement, so the assigned
           name is the LAST token before the [=], not the whole prefix. *)
        let lhs =
          match List.rev (String.split_on_chars (String.prefix st at) ~on:[ ' '; '\n'; '\t' ]) with
          | last :: _ -> last
          | [] -> ""
        in
        let rhs = String.strip (String.drop_prefix st (at + 3)) in
        Option.bind (String.index lhs '_') ~f:(fun us ->
            let ident = String.drop_prefix lhs (us + 1) in
            let is_scope_local =
              String.is_prefix lhs ~prefix:"v" && us > 1
              && String.for_all (String.sub lhs ~pos:1 ~len:(us - 1)) ~f:Char.is_digit
              && not (String.is_empty ident)
            in
            if
              is_scope_local
              && String.is_substring rhs ~substring:"0.0"
              && not (String.is_substring rhs ~substring:"[")
            then Some (lhs, ident, st)
            else None)
  in
  (* The census names the local by the identifier the declaration carries ([scope_local_ident] is
     the one definition of that convention), so "is this one volatile" can be answered off the
     emitted text without guessing where the qualifier would sit. *)
  let declared_volatile source local =
    match String.substr_index source ~pattern:(" " ^ local) with
    | None -> None
    | Some at ->
        let before = String.prefix source at in
        let line = List.last_exn (String.split_lines before) in
        Some (String.is_substring line ~substring:"volatile")
  in
  let check_localized (compiled : Context.routine) routine label =
    let source = Test_utils.Generated.read routine in
    let statements = List.map (String.split source ~on:';') ~f:normalize in
    let fail_all () =
      List.iter
        [
          "accumulator opens from zero in a scope local";
          "the reduction updates the local, not the node";
          "no statement both reads and writes the node";
          "the dead zeroing store and opening read are absent";
          "the node is stored exactly once, from the local";
          "the census names the accumulator the kernel declares";
          "the accumulator declaration stays plain";
          "the accumulating loop's volatile-read form is the one the census reports";
          "the forwarded zero seed performs no device read";
          "no rmw shadow declaration, by the census and by the text";
        ] ~f:(fun c -> Verdict.p (label ^ ": " ^ c) false)
    in
    match List.find_map statements ~f:zero_scope_init with
    | None -> fail_all ()
    | Some (local, ident, zero_seed) ->
        let count st pattern =
          List.length (String.substr_index_all st ~may_overlap:false ~pattern)
        in
        (* Both spellings, as everywhere else in this file: a device read Metal renders through its
           expression-level volatile cast is still a read of the node, and counting only the plain
           one would let the read half of a read-modify-write go missing — which is the direction
           that PASSES the "no statement both reads and writes the node" claim below over a kernel
           that does exactly that. *)
        let node_accesses st = count_node_accesses st ident in
        let accumulation_updates =
          List.filter statements ~f:(fun st ->
              node_accesses st = 0
              && count st local >= 2
              && String.is_substring st ~substring:(local ^ " = "))
        in
        Verdict.p (label ^ ": accumulator opens from zero in a scope local") true;
        (* The accumulation: assigns the local from itself, and never touches the node. *)
        Verdict.p
          (label ^ ": the reduction updates the local, not the node")
          (not (List.is_empty accumulation_updates));
        (* The read-modify-write shape is one statement reading and writing the node. Its absence is
           what localization buys; the access-count claim below separately pins the zero DSE. *)
        Verdict.p_all (label ^ ": no statement both reads and writes the node") statements
          ~f:(fun st -> node_accesses st <= 1);
        (* Store-to-load forwarding removes both accesses that existed only to carry zero into the
           local. The sole surviving node subscript is the closing store. *)
        Verdict.p
          (label ^ ": the dead zeroing store and opening read are absent")
          (1 = List.sum (module Int) statements ~f:node_accesses);
        Verdict.p
          (label ^ ": the node is stored exactly once, from the local")
          (1
          = List.count statements ~f:(fun st ->
              node_accesses st = 1 && String.is_substring st ~substring:("] = " ^ local)));
        (* What the compile REPORTED about this accumulator (gh-ocannl-782), and the emitted text
           checked against it. The expectation is the census's [requested] flag — the capability as
           the backend that ran this compile stated it — not a second reading of the backend name.
           These claims are therefore backend-uniform: they say the same thing on Metal, where the
           workaround is requested, and on the C backends, where it is not. *)
        let v = compiled.Context.volatility in
        let accumulations =
          List.filter_map v.Ir.C_syntax.entries ~f:(fun (_, site) ->
              match site with
              | Ir.C_syntax.Volatile_accumulation_reads name -> Some (name, true)
              | Ir.C_syntax.Plain_accumulator name -> Some (name, false)
              | Ir.C_syntax.Volatile_rmw_reads _ -> None)
        in
        Verdict.p
          (label ^ ": the census names the accumulator the kernel declares")
          (List.mem (List.map accumulations ~f:fst) local ~equal:String.equal);
        Verdict.p_all (label ^ ": the accumulator declaration stays plain") accumulations
          ~f:(fun (name, _) ->
            match declared_volatile source name with Some declared -> not declared | None -> false);
        Verdict.p_all
          (label ^ ": the accumulating loop's volatile-read form is the one the census reports")
          accumulations ~f:(fun (_, volatile_reads) ->
            let update_reads_device =
              List.exists accumulation_updates ~f:(fun st -> String.is_substring st ~substring:"[")
            in
            Bool.equal volatile_reads (v.Ir.C_syntax.requested && update_reads_device)
            && (not (List.is_empty accumulation_updates))
            && List.for_all accumulation_updates ~f:(fun st ->
                let reads_device = String.is_substring st ~substring:"[" in
                Bool.equal
                  (String.is_substring st ~substring:"device volatile float*")
                  (v.Ir.C_syntax.requested && reads_device)));
        Verdict.p
          (label ^ ": the forwarded zero seed performs no device read")
          ((not (String.is_substring zero_seed ~substring:"["))
          && not (String.is_substring zero_seed ~substring:"volatile"));
        (* Localization lifted the device-memory read-modify-write the pointer shadow pins, so this
           routine has none — asserted twice over, from the census and from the emitted text, which
           is what makes either one a check rather than a restatement. *)
        Verdict.p
          (label ^ ": no rmw shadow declaration, by the census and by the text")
          (v.Ir.C_syntax.volatile_rmw_reads = 0
          && not (String.is_substring source ~substring:"__rmw_"))
  in
  (* Hand-built post-schedule form: keep Expand_zero's per-cell nest visible to codegen. The
     reference runs init and reduction in separate kernels, forcing the opening output read. *)
  let module LL = Ir.Low_level in
  let open Ll_test in
  let node = node_factory ~first_id:9890 () in
  let src = node ~dims:[| 2; 4 |] "res_expanded_src" in
  let out = node ~dims:[| 2 |] "res_expanded_out" in
  List.iter [ src; out ] ~f:materialize;
  let z = sym () and j = sym () and i = sym () in
  let init = loop_n z 2 (set_at out (iter z) (c 0.)) in
  let cell = [| iter j |] in
  let reduction =
    loop_n j 2 (loop_n i 4 (set out cell (add (get out cell) (get src [| iter j; iter i |]))))
  in
  let program = seq init reduction in
  let opt = optimize_scoped ~materialized:[ src; out ] ~name:"res_expanded" ~raw:program program in
  let ref_ctx, ref_r =
    Context.compile ~name:"res_expanded_reference" ~prelowered:opt
      ~lowered_transform:(fun o -> [ { o with LL.llc = init }; { o with LL.llc = reduction } ])
      (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let seed = [ (src, Array.init 8 ~f:(fun n -> elem (n / 4) (n % 4))); (out, [| -7.; -9. |]) ] in
  let ref_ctx = run_linked (ref_ctx, ref_r) ~seed in
  let want = Context.get_values ref_ctx out in
  let ctx, r_expanded = link ~name:"res_expanded" opt in
  let ctx = run_linked (ctx, r_expanded) ~seed in
  Verdict.p_all2 "expanded-zero reduction matches a separately materialized initialization"
    (Context.get_values ctx out) want ~f:Float.equal;
  check_localized r_expanded "res_expanded" "expanded-zero reduction";
  let partial = seq (loop_n z 1 (set_at out (iter z) (c 0.))) reduction in
  let partial_opt =
    optimize_scoped ~materialized:[ src; out ] ~name:"res_expanded_partial" ~raw:partial partial
  in
  let ctx, r_partial = link ~name:"res_expanded_partial" partial_opt in
  let ctx = run_linked (ctx, r_partial) ~seed in
  Verdict.p "partial per-cell zero preserves the untouched cell's opening value"
    (Array.equal Float.equal (Context.get_values ctx out) [| 10.; 41. |]);
  let partial_source = Test_utils.Generated.read "res_expanded_partial" in
  Verdict.p "partial per-cell zero retains zero store, opening read and closing store"
    (count_node_accesses partial_source "res_expanded_out" = 3);
  check_localized r_total "res_total" "scalar reduction";
  check_localized r_per_col "res_per_col" "row reduction";
  check_localized r_index_total "res_index_total" "index-only reduction";
  Verdict.p "index-only accumulation is censused plain because it emits no device read"
    (r_index_total.Context.volatility.Ir.C_syntax.volatile_accumulations = 0
    && r_index_total.Context.volatility.Ir.C_syntax.plain_accumulations = 1)

(* The covering-write half of the DSE condition: a localized reduction of only [out[0]] does not
   overwrite [out[1]], so the whole-node [Zero_out] is live and must stay. The producer varies with
   its loop symbol and never equals the zero/sentinels, while the untouched second cell makes a
   dropped zero fail even if the accumulated first cell happens to be right. *)
let () =
  let module LL = Ir.Low_level in
  let node = Ll_test.node_factory ~first_id:9810 () in
  let src = node ~dims:[| 8 |] "res_partial_src" and out = node ~dims:[| 2 |] "res_partial_out" in
  List.iter [ src; out ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let cell = [| Ll_test.fixed 0 |] in
  let program =
    LL.Seq
      ( LL.Zero_out out,
        Ll_test.loop_n i 8
          (Ll_test.set out cell
             (Ll_test.add (Ll_test.get out cell) (Ll_test.get src [| Ll_test.iter i |]))) )
  in
  let optimized = Ll_test.optimize ~materialized:[ src; out ] ~name:"res_partial" program in
  let ctx, routine = Ll_test.link ~name:"res_partial" optimized in
  let src_values = Array.init 8 ~f:(fun k -> Float.of_int (k + 1)) in
  let ctx =
    Ll_test.run_linked (ctx, routine) ~seed:[ (src, src_values); (out, [| -7.0; -9.0 |]) ]
  in
  let got = Context.get_values ctx out in
  Verdict.p "partial-cell localized reduction preserves the live whole-node zero"
    (Array.equal Float.equal got [| 36.0; 0.0 |]);
  let source = Test_utils.Generated.read "res_partial" in
  let accesses = count_node_accesses source "res_partial_out" in
  Verdict.p "partial-cell reduction retains zero store, opening read, and closing store"
    (accesses = 3)

(* Covering the node is not enough when an enclosing loop repeats the same cells outside the scope
   the localizer can form. Here [k] repeats every [out[j]] cell, while only the inner [i] reduction
   can localize. Forwarding zero into that inner scope would reset each cell once per [k] instead of
   accumulating both slices. *)
let () =
  let module LL = Ir.Low_level in
  let node = Ll_test.node_factory ~first_id:9860 () in
  let src = node ~dims:[| 2; 2; 2 |] "res_repeat_src" in
  let out = node ~dims:[| 2 |] "res_repeat_out" in
  List.iter [ src; out ] ~f:Ll_test.materialize;
  let k = Ll_test.sym () and j = Ll_test.sym () and i = Ll_test.sym () in
  let out_cell = [| Ll_test.iter j |] in
  let update =
    Ll_test.set out out_cell
      (Ll_test.add (Ll_test.get out out_cell)
         (Ll_test.get src [| Ll_test.iter k; Ll_test.iter j; Ll_test.iter i |]))
  in
  let program =
    LL.Seq (LL.Zero_out out, Ll_test.loop_n k 2 (Ll_test.loop_n j 2 (Ll_test.loop_n i 2 update)))
  in
  let optimized = Ll_test.optimize ~materialized:[ src; out ] ~name:"res_repeat" program in
  let ctx, routine = Ll_test.link ~name:"res_repeat" optimized in
  let values = Array.init 8 ~f:(fun n -> Float.of_int (n + 1)) in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed:[ (src, values); (out, [| -7.0; -9.0 |]) ] in
  Verdict.p "repeated covering reduction preserves accumulation across the outer loop"
    (Array.equal Float.equal (Context.get_values ctx out) [| 14.0; 22.0 |]);
  let source = Test_utils.Generated.read "res_repeat" in
  let accesses = count_node_accesses source "res_repeat_out" in
  Verdict.p "repeated covering reduction retains zero store, opening read, and closing store"
    (accesses = 3)

(* A dead enclosing loop executes no closing store at all. The ordinary optimizer drops its body, so
   inject this shape through the same post-optimize seam schedule-minted IR uses: codegen still
   renders the body, and an inner localizer must not consume the whole-node seed merely because the
   dead loop is absent from the write map. The nonzero host seed makes dropping [Zero_out]
   immediately visible. *)
let () =
  let module LL = Ir.Low_level in
  let node = Ll_test.node_factory ~first_id:9870 () in
  let out = node ~dims:[| 2 |] "res_dead_out" in
  Ll_test.materialize out;
  let k = Ll_test.sym () and j = Ll_test.sym () and i = Ll_test.sym () in
  let out_cell = [| Ll_test.iter j |] in
  let update =
    Ll_test.set out out_cell (Ll_test.add (Ll_test.get out out_cell) (LL.Constant 1.0))
  in
  let program =
    LL.Seq (LL.Zero_out out, Ll_test.loop_n k 0 (Ll_test.loop_n j 2 (Ll_test.loop_n i 2 update)))
  in
  let optimized =
    Ll_test.optimize_scoped ~materialized:[ out ] ~name:"res_dead" ~raw:program program
  in
  let ctx, routine = Ll_test.link ~name:"res_dead" optimized in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed:[ (out, [| -7.0; -9.0 |]) ] in
  Verdict.p "dead enclosing loop preserves the live whole-node zero"
    (Array.equal Float.equal (Context.get_values ctx out) [| 0.0; 0.0 |]);
  let source = Test_utils.Generated.read "res_dead" in
  Verdict.p "dead enclosing loop retains zero store, opening read, and closing store"
    (count_node_accesses source "res_dead_out" = 3)

(* A symbolic extent guard can make a statically covering output loop execute only a prefix. The
   serial renderer fuses this exact [j < extent] shape into the loop header, so the DSE predicate
   itself must reject the guarded affine write; relying on the ordinary [If] rendering boundary is
   insufficient. The untouched row starts nonzero to make a dropped whole-node zero observable. *)
let () =
  let module LL = Ir.Low_level in
  let module Idx = Ir.Indexing in
  let node = Ll_test.node_factory ~first_id:9880 () in
  let src = node ~dims:[| 2; 2 |] "res_extent_src" in
  let out = node ~dims:[| 2 |] "res_extent_out" in
  List.iter [ src; out ] ~f:Ll_test.materialize;
  let j = Ll_test.sym () and i = Ll_test.sym () in
  let extent = Idx.make_static_symbol ~static_range:2 ~used_as:`Extent (Idx.get_symbol ()) in
  let bindings : Idx.unit_bindings = Idx.Bind (extent, Idx.Empty) in
  let out_cell = [| Ll_test.iter j |] in
  let update =
    Ll_test.set out out_cell
      (Ll_test.add (Ll_test.get out out_cell)
         (Ll_test.get src [| Ll_test.iter j; Ll_test.iter i |]))
  in
  let iprec = Ir.Ops.index_prec () in
  let guard =
    LL.Binop
      ( Ir.Ops.Cmplt,
        (LL.Embed_index (Ll_test.iter j), iprec),
        (LL.Embed_index (Idx.Iterator extent.static_symbol), iprec) )
  in
  let program =
    LL.Seq
      ( LL.Zero_out out,
        Ll_test.loop_n j 2 (LL.If { cond = (guard, iprec); body = Ll_test.loop_n i 2 update }) )
  in
  let optimized =
    Ll_test.optimize ~materialized:[ src; out ] ~static_indices:(Idx.bound_symbols bindings)
      ~name:"res_extent" program
  in
  let ctx, routine =
    Context.compile ~name:"res_extent" ~prelowered:optimized
      ~lowered_transform:(fun x -> [ x ])
      (Context.auto ()) Ir.Assignments.empty_comp bindings
  in
  Idx.find_exn routine.Context.bindings extent := 1;
  let ctx =
    Ll_test.run_linked (ctx, routine)
      ~seed:[ (src, [| 1.0; 2.0; 4.0; 8.0 |]); (out, [| -7.0; -9.0 |]) ]
  in
  Verdict.p "symbolically guarded coverage preserves the whole-node zero"
    (Array.equal Float.equal (Context.get_values ctx out) [| 3.0; 0.0 |]);
  let source = Test_utils.Generated.read "res_extent" in
  let accesses = count_node_accesses source "res_extent_out" in
  Verdict.p "symbolically guarded coverage retains zero store, opening read, and closing store"
    (accesses = 3)

(* Cross-statement CSE lifts a shared scope out of both users as [Declare_local; body]. That form
   renders its declaration before its accumulating [Set_local], so the census must retain the site
   until rendering observes whether the update emitted a volatile read (Codex P2, review round 2 on
   #553). This hand-built leg makes the optimizer produce that exact form and executes both
   users. *)
let () =
  let module LL = Ir.Low_level in
  let node = Ll_test.node_factory ~first_id:9820 ~dims:[| 8 |] () in
  let src = node "res_lift_src"
  and out_a = node ~dims:[| 1 |] "res_lift_a"
  and out_b = node ~dims:[| 1 |] "res_lift_b"
  and tmp = node ~dims:[| 1 |] "res_lift_tmp" in
  List.iter [ src; out_a; out_b ] ~f:Ll_test.materialize;
  Ll_test.virtualize tmp;
  (* One fresh loop symbol per user: the two scopes are alpha-equivalent, which is what the CSE
     lifts, while [optimize]'s loop-binder lint refuses one symbol bound twice (gh-ocannl-1402). *)
  let id = LL.get_scope tmp in
  let scoped () =
    let i = Ll_test.sym () in
    let body =
      LL.Seq
        ( LL.Set_local (id, Ll_test.c 0.0),
          Ll_test.loop_n i 8
            (LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.get src [| Ll_test.iter i |])))
        )
    in
    LL.Local_scope { id; body; orig_indices = [| Ll_test.fixed 0 |]; mint = LL.Inlined_computation }
  in
  let program =
    LL.Seq
      ( Ll_test.set out_a [| Ll_test.fixed 0 |] (scoped ()),
        Ll_test.set out_b [| Ll_test.fixed 0 |] (scoped ()) )
  in
  let optimized = Ll_test.optimize ~materialized:[ src; out_a; out_b ] ~name:"res_lifted" program in
  let rec count_declarations = function
    | LL.Declare_local _ -> 1
    | LL.Seq (a, b) -> count_declarations a + count_declarations b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> count_declarations body
    | LL.Tile_mma { fallback; _ } -> count_declarations fallback
    | _ -> 0
  in
  Verdict.p "cross-statement CSE produces one lifted accumulator declaration"
    (count_declarations optimized.LL.llc = 1 && Ll_test.count_scopes optimized.LL.llc = 0);
  let ctx, routine = Ll_test.link ~name:"res_lifted" optimized in
  let values = Array.init 8 ~f:(fun k -> Float.of_int (k + 1)) in
  let ctx =
    Ll_test.run_linked (ctx, routine)
      ~seed:[ (src, values); (out_a, [| -1.0 |]); (out_b, [| -2.0 |]) ]
  in
  Verdict.p "lifted accumulator executes once for both users"
    Float.(
      equal (Context.get_values ctx out_a).(0) 36.0 && equal (Context.get_values ctx out_b).(0) 36.0);
  let volatility = routine.Context.volatility in
  Verdict.p "lifted accumulator contributes exactly one census site"
    (volatility.Ir.C_syntax.volatile_accumulations + volatility.plain_accumulations = 1);
  Verdict.p "lifted accumulator census follows its emitted volatile read"
    (Bool.equal (volatility.volatile_accumulations = 1) volatility.requested
    && Bool.equal (volatility.plain_accumulations = 1) (not volatility.requested));
  let source = Test_utils.Generated.read "res_lifted" in
  Verdict.p "lifted accumulating read uses the backend-requested form"
    (Bool.equal
       (String.is_substring source ~substring:"device volatile float*")
       volatility.requested)

(* A data-dependent guard can be the accumulating loop's only device read: [if mask[i] then local +=
   1]. The scope remains reduction-shaped because the guard does not observe the local. Metal must
   cast that controlling read even though the update expression itself dereferences no node pointer
   (Codex P1, review round 3 on #553). *)
let () =
  let module LL = Ir.Low_level in
  let node = Ll_test.node_factory ~first_id:9840 ~dims:[| 8 |] () in
  let mask = node "res_guard_mask"
  and out = node ~dims:[| 1 |] "res_guard_out"
  and tmp = node ~dims:[| 1 |] "res_guard_tmp" in
  List.iter [ mask; out ] ~f:Ll_test.materialize;
  Ll_test.virtualize tmp;
  let i = Ll_test.sym () in
  let id = LL.get_scope tmp in
  let body =
    LL.Seq
      ( LL.Set_local (id, Ll_test.c 0.0),
        Ll_test.loop_n i 8
          (LL.If
             {
               cond = (Ll_test.get mask [| Ll_test.iter i |], Ir.Ops.single);
               body = LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.c 1.0));
             }) )
  in
  let program =
    Ll_test.set out
      [| Ll_test.fixed 0 |]
      (LL.Local_scope
         { id; body; orig_indices = [| Ll_test.fixed 0 |]; mint = LL.Inlined_computation })
  in
  let optimized = Ll_test.optimize ~materialized:[ mask; out ] ~name:"res_guarded" program in
  Verdict.p "conditional reduction retains its scope-local accumulator"
    (Ll_test.count_scopes optimized.LL.llc = 1);
  let ctx, routine = Ll_test.link ~name:"res_guarded" optimized in
  let ctx =
    Ll_test.run_linked (ctx, routine)
      ~seed:[ (mask, [| 1.0; 0.0; 1.0; 1.0; 0.0; 0.0; 1.0; 0.0 |]); (out, [| -1.0 |]) ]
  in
  Verdict.p "conditional reduction matches the selected-term reference"
    Float.(equal (Context.get_values ctx out).(0) 4.0);
  let volatility = routine.Context.volatility in
  Verdict.p "conditional reduction contributes exactly one census site"
    (volatility.Ir.C_syntax.volatile_accumulations + volatility.plain_accumulations = 1);
  Verdict.p "conditional reduction census follows the controlling read"
    (Bool.equal (volatility.volatile_accumulations = 1) volatility.requested
    && Bool.equal (volatility.plain_accumulations = 1) (not volatility.requested));
  let source = Test_utils.Generated.read "res_guarded" in
  Verdict.p "conditional reduction casts its controlling device read"
    (Bool.equal
       (String.is_substring source
          ~substring:("volatile float*)" ^ Test_utils.Generated.node_ident ~facts "res_guard_mask"))
       volatility.requested)

(* {1 Serial device-memory RMWs: Set_dynamic scatters and guarded RMWs (gh-ocannl-2474)}

   Metal's [volatile_serial_accumulation] workaround also covers [Set_dynamic] device-memory RMWs
   whose static axes are invariant across an enclosing serial loop, plus any [If] guard inside that
   serial loop controlling a device-memory RMW ([Set] or [Set_dynamic]). Each leg executes against a
   host reference with nonzero prefill and discriminating inputs, checks the volatility census, and
   pins the emitted text via [Test_utils.Generated]. *)
let () =
  let module Cs = Ir.C_syntax in
  let module Ops = Ir.Ops in
  let prefill_ctx = prefill in
  let open Ll_test in
  let normalize st = String.concat ~sep:" " (String.split_on_chars st ~on:[ '\n'; '\t' ]) in
  let read_routine (routine : Context.routine) name =
    let artifact = if List.length routine.Context.segments > 1 then name ^ "__seg" else name in
    Test_utils.Generated.read artifact
  in
  let has_volatile_float source label =
    String.is_substring source
      ~substring:("volatile float*)" ^ Test_utils.Generated.node_ident ~facts label)
  in
  let has_volatile_uint source label =
    String.is_substring source
      ~substring:("volatile uint*)" ^ Test_utils.Generated.node_ident ~facts label)
  in
  let rmw_idents (v : Cs.volatility_summary) =
    List.filter_map v.Cs.entries ~f:(fun (_, site) ->
        match site with Cs.Volatile_rmw_reads name -> Some name | _ -> None)
  in
  let if_lines source =
    List.filter_map (String.split_lines source) ~f:(fun line ->
        let trimmed = String.strip line in
        if String.is_prefix trimmed ~prefix:"if (" then Some trimmed else None)
  in
  (* Leg 1: hand-built guarded [Set_dynamic] scatter RMW over a serial position loop. Repeated token
     2 accumulates 3 hits, token 0 accumulates 1 hit, token 4 is OOB and skipped, and untouched rows
     keep their nonzero prefill. *)
  let () =
    let vocab = 4 and d_model = 2 and positions = 5 in
    let ids =
      node_factory ~prec:Ops.uint32 ~first_id:9910 ~dims:[| positions |] () "res_scat_ids"
    in
    let src = node_factory ~first_id:9911 ~dims:[| positions; d_model |] () "res_scat_src" in
    let acc = node_factory ~first_id:9912 ~dims:[| d_model; vocab |] () "res_scat_acc" in
    List.iter [ ids; src; acc ] ~f:materialize;
    let pos = sym () and d = sym () in
    let id_read = get ids [| iter pos |] in
    let guard = cmp_at Ops.uint32 Ops.Cmplt id_read (ic vocab) in
    let acc_idcs = [| iter d; fixed 0 |] in
    let scatter =
      scatter_add ~tn:acc ~idcs:acc_idcs ~dyn_axis:1 ~dyn_value:(id_read, Ops.uint32)
        (get src [| iter pos; iter d |], Ops.single)
    in
    let program = loop_n pos positions (if_at Ops.uint32 guard (loop_n d d_model scatter)) in
    (* [optimize] refuses [Set_dynamic]; supply a plain raw twin with the same nodes. *)
    let raw =
      loop_n pos positions
        (loop_n d d_model
           (set acc
              [| iter d; fixed 0 |]
              (add
                 (get acc [| iter d; fixed 0 |])
                 (add (get src [| iter pos; iter d |]) (get ids [| iter pos |])))))
    in
    let opt = optimize_scoped ~materialized:[ ids; src; acc ] ~name:"res_scat_rmw" ~raw program in
    let ctx, routine = link ~name:"res_scat_rmw" opt in
    let ids_vals = [| 2.; 0.; 2.; 4.; 2. |] in
    let src_vals =
      Array.init (positions * d_model) ~f:(fun k ->
          let p = k / d_model and ch = k % d_model in
          1.0 +. (4.0 *. Float.of_int p) +. (0.5 *. Float.of_int ch))
    in
    let acc_init = Array.init (d_model * vocab) ~f:(fun k -> 10.0 +. (0.25 *. Float.of_int k)) in
    let ctx =
      run_linked (ctx, routine) ~seed:[ (ids, ids_vals); (src, src_vals); (acc, acc_init) ]
    in
    let expected = Array.copy acc_init in
    for p = 0 to positions - 1 do
      let v = Int.of_float ids_vals.(p) in
      if v >= 0 && v < vocab then
        for ch = 0 to d_model - 1 do
          let idx = (ch * vocab) + v in
          expected.(idx) <- expected.(idx) +. src_vals.((p * d_model) + ch)
        done
    done;
    Verdict.p_all2 "guarded Set_dynamic RMW matches host reference" (Context.get_values ctx acc)
      expected ~f:Float.equal;
    let v = routine.Context.volatility in
    let acc_ident = Test_utils.Generated.node_ident ~facts "res_scat_acc" in
    Verdict.p "guarded Set_dynamic RMW reports one volatile RMW site iff requested"
      ((v.Cs.volatile_rmw_reads = if v.Cs.requested then 1 else 0)
      && v.Cs.volatile_accumulations = 0 && v.Cs.plain_accumulations = 0
      && List.equal String.equal (rmw_idents v) (if v.Cs.requested then [ acc_ident ] else []));
    let source = read_routine routine "res_scat_rmw" in
    let guard_lines = if_lines source in
    Verdict.p_exists "guarded Set_dynamic RMW emits a bounds-guard if statement" guard_lines
      ~f:(fun line ->
        String.is_substring line ~substring:(Test_utils.Generated.node_ident ~facts "res_scat_ids"));
    Verdict.p_all "guarded Set_dynamic RMW casts its If-guard device read iff requested" guard_lines
      ~f:(fun line -> Bool.equal (has_volatile_uint line "res_scat_ids") v.Cs.requested);
    let statements = List.map (String.split source ~on:';') ~f:normalize in
    let scatter_stmts =
      List.filter statements ~f:(fun st ->
          match String.substr_index st ~pattern:" = " with
          | None -> false
          | Some at ->
              let lhs = String.strip (String.prefix st at) in
              String.is_substring lhs ~substring:(acc_ident ^ "["))
    in
    Verdict.p "guarded Set_dynamic RMW emits one scatter assignment statement"
      (List.length scatter_stmts = 1);
    Verdict.p_all
      "guarded Set_dynamic RMW casts its dynamic index and RHS device reads iff requested"
      scatter_stmts ~f:(fun st ->
        match String.substr_index st ~pattern:" = " with
        | None -> false
        | Some at ->
            let lhs = String.strip (String.prefix st at) in
            let rhs = String.drop_prefix st (at + 3) in
            (not (has_volatile_float lhs "res_scat_acc"))
            && Bool.equal (has_volatile_uint lhs "res_scat_ids") v.Cs.requested
            && Bool.equal (has_volatile_float rhs "res_scat_acc") v.Cs.requested
            && Bool.equal (has_volatile_uint rhs "res_scat_ids") v.Cs.requested
            && Bool.equal (has_volatile_float rhs "res_scat_src") v.Cs.requested)
  in

  (* Leg 2: end-to-end lowered embedding backward scatter through [rewrite_one_hot_reductions]. *)
  let () =
    Tensor.unsafe_reinitialize ();
    let vocab = 6 and embed = 4 and positions = 5 in
    let id_ints = [ 2; 0; 2; vocab; 2 ] in
    let ids = Nn_blocks.class_ids_of_int_list ~label:"res_emb_ids" id_ints in
    let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
    let table =
      TDSL.param
        ~values:(Array.init (embed * vocab) ~f:(fun k -> 0.5 *. Float.of_int (k + 1)))
        "res_emb_tbl" ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
    in
    let coeff_vals =
      Array.init (positions * embed) ~f:(fun k ->
          let p = k / embed and d = k % embed in
          1.0 +. (2.0 *. Float.of_int p) +. (0.25 *. Float.of_int d))
    in
    let coeff =
      TDSL.ndarray coeff_vals ~label:[ "res_emb_coeff" ] ~batch_dims:[ positions ]
        ~output_dims:[ embed ] ()
    in
    let%op embedded = table * one_hot in
    let%op loss = (embedded *. coeff) ++ "...|... => |->0" in
    let update = Train.grad_update loss in
    let ctx = Context.auto () in
    let ctx = Train.init_params ctx Train.IDX.empty loss in
    let ctx, routine = Context.compile ~name:"res_emb_bwd" ctx update Ir.Indexing.Empty in
    let table_grad = (Option.value_exn ~here:[%here] table.Tensor.diff).Tensor.grad in
    let ctx = Context.run (prefill_ctx ctx table_grad) routine in
    let expected = Array.create ~len:(embed * vocab) 0.0 in
    List.iteri id_ints ~f:(fun p v ->
        if v >= 0 && v < vocab then
          for d = 0 to embed - 1 do
            let idx = (d * vocab) + v in
            expected.(idx) <- expected.(idx) +. coeff_vals.((p * embed) + d)
          done);
    Verdict.p_all2 "lowered embedding backward scatter matches host oracle"
      (Context.get_values ctx table_grad)
      expected ~f:Float.equal;
    let v = routine.Context.volatility in
    let grad_ident = Test_utils.Generated.node_ident ~facts "res_emb_tbl_grad" in
    Verdict.p "lowered embedding backward scatter reports one volatile RMW site iff requested"
      ((v.Cs.volatile_rmw_reads = if v.Cs.requested then 1 else 0)
      && List.equal String.equal (rmw_idents v) (if v.Cs.requested then [ grad_ident ] else []));
    let source = read_routine routine "res_emb_bwd" in
    let guard_lines =
      List.filter (if_lines source) ~f:(fun line ->
          String.is_substring line ~substring:(Test_utils.Generated.node_ident ~facts "res_emb_ids"))
    in
    Verdict.p_exists "lowered embedding backward scatter emits a bounds guard on ids" guard_lines
      ~f:(Fn.const true);
    Verdict.p_all "lowered embedding backward scatter casts its If-guard ids read iff requested"
      guard_lines ~f:(fun line -> Bool.equal (has_volatile_uint line "res_emb_ids") v.Cs.requested);
    Verdict.p "lowered embedding backward scatter casts its table-grad RMW read iff requested"
      (Bool.equal (has_volatile_float source "res_emb_tbl_grad") v.Cs.requested)
  in

  (* Leg 3: guarded unlocalized [Set] device-memory RMW inside a serial loop. The sibling store
     [sink[i] = out[0]] prevents [try_localize_serial_reduce] from lifting [out[0]], leaving a
     genuine guarded device-memory [Set] RMW whose [If] condition must also cast its device read
     without double-counting the RMW in the volatility census. *)
  let () =
    let n = 6 in
    let mask = node_factory ~first_id:9920 ~dims:[| n |] () "res_gset_mask" in
    let src = node_factory ~first_id:9921 ~dims:[| n |] () "res_gset_src" in
    let out = node_factory ~first_id:9922 ~dims:[| 1 |] () "res_gset_out" in
    let sink = node_factory ~first_id:9923 ~dims:[| n |] () "res_gset_sink" in
    List.iter [ mask; src; out; sink ] ~f:materialize;
    let i = sym () in
    let program =
      loop_n i n
        (if_
           (get mask [| iter i |])
           (seq
              (set out [| fixed 0 |] (add (get out [| fixed 0 |]) (get src [| iter i |])))
              (set sink [| iter i |] (get out [| fixed 0 |]))))
    in
    let opt = optimize ~materialized:[ mask; src; out; sink ] ~name:"res_gset_rmw" program in
    let ctx, routine = link ~name:"res_gset_rmw" opt in
    let mask_vals = [| 1.; 0.; 1.; 1.; 0.; 1. |] in
    let src_vals = Array.init n ~f:(fun k -> 1.0 +. (2.0 *. Float.of_int k)) in
    let sink_init = Array.init n ~f:(fun k -> -.(10.0 +. Float.of_int k)) in
    let ctx =
      run_linked (ctx, routine)
        ~seed:[ (mask, mask_vals); (src, src_vals); (out, [| 5.0 |]); (sink, sink_init) ]
    in
    let expected_sink = Array.copy sink_init in
    let acc = ref 5.0 in
    for k = 0 to n - 1 do
      if Float.(mask_vals.(k) <> 0.) then begin
        acc := !acc +. src_vals.(k);
        expected_sink.(k) <- !acc
      end
    done;
    Verdict.p "guarded Set RMW matches host reference"
      (Float.equal (Context.get_values ctx out).(0) !acc
      && Array.equal Float.equal (Context.get_values ctx sink) expected_sink);
    let v = routine.Context.volatility in
    let out_ident = Test_utils.Generated.node_ident ~facts "res_gset_out" in
    Verdict.p "guarded Set RMW reports one volatile RMW site iff requested"
      ((v.Cs.volatile_rmw_reads = if v.Cs.requested then 1 else 0)
      && List.equal String.equal (rmw_idents v) (if v.Cs.requested then [ out_ident ] else []));
    let source = read_routine routine "res_gset_rmw" in
    let guard_lines = if_lines source in
    Verdict.p_exists "guarded Set RMW emits its controlling if statement" guard_lines
      ~f:(fun line ->
        String.is_substring line ~substring:(Test_utils.Generated.node_ident ~facts "res_gset_mask"));
    Verdict.p_all "guarded Set RMW casts its If-guard device read iff requested" guard_lines
      ~f:(fun line -> Bool.equal (has_volatile_float line "res_gset_mask") v.Cs.requested);
    Verdict.p "guarded Set RMW casts its RMW reads iff requested"
      (Bool.equal (has_volatile_float source "res_gset_out") v.Cs.requested
      && Bool.equal (has_volatile_float source "res_gset_src") v.Cs.requested)
  in

  (* Negative controls: none of these is a serial device-memory RMW across the enclosing guard's
     serial loop, so none may acquire spurious volatile casts even on Metal. *)
  (* Negative control 1: non-RMW guarded [Set_dynamic] overwrite (does not read its target). *)
  let () =
    let vocab = 4 and positions = 4 in
    let ids = node_factory ~prec:Ops.uint32 ~first_id:9930 ~dims:[| positions |] () "res_ow_ids" in
    let src = node_factory ~first_id:9931 ~dims:[| positions |] () "res_ow_src" in
    let out = node_factory ~first_id:9932 ~dims:[| vocab |] () "res_ow_out" in
    List.iter [ ids; src; out ] ~f:materialize;
    let pos = sym () in
    let id_read = get ids [| iter pos |] in
    let guard = cmp_at Ops.uint32 Ops.Cmplt id_read (ic vocab) in
    let program =
      loop_n pos positions
        (if_at Ops.uint32 guard
           (scatter ~tn:out
              ~idcs:[| fixed 0 |]
              ~dyn_axis:0 ~dyn_value:(id_read, Ops.uint32)
              (get src [| iter pos |])))
    in
    let raw =
      loop_n pos positions
        (set out [| iter pos |] (add (get src [| iter pos |]) (get ids [| iter pos |])))
    in
    let opt = optimize_scoped ~materialized:[ ids; src; out ] ~name:"res_scat_ow" ~raw program in
    let ctx, routine = link ~name:"res_scat_ow" opt in
    let ids_vals = [| 3.; 1.; 4.; 1. |] in
    let src_vals = [| 2.0; 4.0; 8.0; 16.0 |] in
    let out_init = [| -1.; -2.; -3.; -4. |] in
    let ctx =
      run_linked (ctx, routine) ~seed:[ (ids, ids_vals); (src, src_vals); (out, out_init) ]
    in
    Verdict.p_all2 "non-RMW Set_dynamic overwrite matches host reference"
      (Context.get_values ctx out) [| -1.; 16.0; -3.; 2.0 |] ~f:Float.equal;
    let source = read_routine routine "res_scat_ow" in
    Verdict.p "non-RMW Set_dynamic overwrite stays unqualified and unrecorded"
      (routine.Context.volatility.Cs.volatile_rmw_reads = 0
      && not (String.is_substring source ~substring:"volatile"))
  in

  (* Negative control 2: static-axis-varying [Set_dynamic] RMW — the sole serial loop [d] indexes
     the static axis [0], so distinct iterations write disjoint rows. *)
  let () =
    let vocab = 4 and d_model = 3 in
    let ids = node_factory ~prec:Ops.uint32 ~first_id:9940 ~dims:[| 1 |] () "res_st_ids" in
    let src = node_factory ~first_id:9941 ~dims:[| d_model |] () "res_st_src" in
    let acc = node_factory ~first_id:9942 ~dims:[| d_model; vocab |] () "res_st_acc" in
    List.iter [ ids; src; acc ] ~f:materialize;
    let d = sym () in
    let id_read = get ids [| fixed 0 |] in
    let guard = cmp_at Ops.uint32 Ops.Cmplt id_read (ic vocab) in
    let program =
      loop_n d d_model
        (if_at Ops.uint32 guard
           (scatter_add ~tn:acc
              ~idcs:[| iter d; fixed 0 |]
              ~dyn_axis:1 ~dyn_value:(id_read, Ops.uint32)
              (get src [| iter d |], Ops.single)))
    in
    let raw =
      loop_n d d_model
        (set acc
           [| iter d; fixed 0 |]
           (add
              (get acc [| iter d; fixed 0 |])
              (add (get src [| iter d |]) (get ids [| fixed 0 |]))))
    in
    let opt =
      optimize_scoped ~materialized:[ ids; src; acc ] ~name:"res_scat_static" ~raw program
    in
    let ctx, routine = link ~name:"res_scat_static" opt in
    let src_vals = [| 1.5; 2.5; 3.5 |] in
    let acc_init = Array.init (d_model * vocab) ~f:(fun k -> Float.of_int (k + 1)) in
    let ctx =
      run_linked (ctx, routine) ~seed:[ (ids, [| 2. |]); (src, src_vals); (acc, acc_init) ]
    in
    let expected = Array.copy acc_init in
    for ch = 0 to d_model - 1 do
      expected.((ch * vocab) + 2) <- expected.((ch * vocab) + 2) +. src_vals.(ch)
    done;
    Verdict.p_all2 "static-axis-varying Set_dynamic RMW matches host reference"
      (Context.get_values ctx acc) expected ~f:Float.equal;
    let source = read_routine routine "res_scat_static" in
    Verdict.p "static-axis-varying Set_dynamic RMW stays unqualified and unrecorded"
      (routine.Context.volatility.Cs.volatile_rmw_reads = 0
      && not (String.is_substring source ~substring:"volatile"))
  in

  (* Negative control 3: an outer [If] above the repeating serial loop [pos] (varying only along
     static axis [d]) stays plain while the inner [Set_dynamic] inside [pos] still qualifies. *)
  let () =
    let vocab = 4 and d_model = 2 and positions = 3 in
    let flag = node_factory ~first_id:9950 ~dims:[| d_model |] () "res_og_flag" in
    let ids = node_factory ~prec:Ops.uint32 ~first_id:9951 ~dims:[| positions |] () "res_og_ids" in
    let src = node_factory ~first_id:9952 ~dims:[| positions; d_model |] () "res_og_src" in
    let acc = node_factory ~first_id:9953 ~dims:[| d_model; vocab |] () "res_og_acc" in
    List.iter [ flag; ids; src; acc ] ~f:materialize;
    let d = sym () and pos = sym () in
    let id_read = get ids [| iter pos |] in
    let program =
      loop_n d d_model
        (if_
           (get flag [| iter d |])
           (loop_n pos positions
              (scatter_add ~tn:acc
                 ~idcs:[| iter d; fixed 0 |]
                 ~dyn_axis:1 ~dyn_value:(id_read, Ops.uint32)
                 (get src [| iter pos; iter d |], Ops.single))))
    in
    let raw =
      loop_n d d_model
        (loop_n pos positions
           (set acc
              [| iter d; fixed 0 |]
              (add
                 (get acc [| iter d; fixed 0 |])
                 (add
                    (get flag [| iter d |])
                    (add (get src [| iter pos; iter d |]) (get ids [| iter pos |]))))))
    in
    let opt =
      optimize_scoped ~materialized:[ flag; ids; src; acc ] ~name:"res_scat_outer" ~raw program
    in
    let ctx, routine = link ~name:"res_scat_outer" opt in
    let flag_vals = [| 1.; 0. |] in
    let ids_vals = [| 1.; 1.; 3. |] in
    let src_vals = [| 1.; 2.; 4.; 8.; 16.; 32. |] in
    let acc_init = Array.init (d_model * vocab) ~f:(fun k -> 0.5 *. Float.of_int k) in
    let ctx =
      run_linked (ctx, routine)
        ~seed:[ (flag, flag_vals); (ids, ids_vals); (src, src_vals); (acc, acc_init) ]
    in
    let expected = Array.copy acc_init in
    for ch = 0 to d_model - 1 do
      if Float.(flag_vals.(ch) <> 0.) then
        for p = 0 to positions - 1 do
          let v = Int.of_float ids_vals.(p) in
          expected.((ch * vocab) + v) <-
            expected.((ch * vocab) + v) +. src_vals.((p * d_model) + ch)
        done
    done;
    Verdict.p_all2 "outer-guarded Set_dynamic RMW matches host reference"
      (Context.get_values ctx acc) expected ~f:Float.equal;
    let v = routine.Context.volatility in
    let source = read_routine routine "res_scat_outer" in
    Verdict.p "outer guard above repeating serial loop stays unqualified"
      (not (has_volatile_float source "res_og_flag"));
    Verdict.p "inner Set_dynamic RMW still qualifies and records iff requested"
      ((v.Cs.volatile_rmw_reads = if v.Cs.requested then 1 else 0)
      && Bool.equal (has_volatile_float source "res_og_acc") v.Cs.requested)
  in

  (* Negative control 4: routine-local ([Local]) [Set_dynamic] RMW is not a device-memory pointer
     and stays unqualified even inside a serial loop. *)
  let () =
    let vocab = 4 and positions = 4 in
    let ids = node_factory ~prec:Ops.uint32 ~first_id:9960 ~dims:[| positions |] () "res_loc_ids" in
    let src = node_factory ~first_id:9961 ~dims:[| positions |] () "res_loc_src" in
    let tmp = node_factory ~first_id:9962 ~dims:[| vocab |] () "res_loc_tmp" in
    let out = node_factory ~first_id:9963 ~dims:[| vocab |] () "res_loc_out" in
    List.iter [ ids; src; out ] ~f:materialize;
    Tn.update_memory_mode tmp Tn.Local (Site "99:test-setup");
    let pos = sym () and v = sym () in
    let id_read = get ids [| iter pos |] in
    let guard = cmp_at Ops.uint32 Ops.Cmplt id_read (ic vocab) in
    let program =
      seq (zero tmp)
        (seq
           (loop_n pos positions
              (if_at Ops.uint32 guard
                 (scatter_add ~tn:tmp
                    ~idcs:[| fixed 0 |]
                    ~dyn_axis:0 ~dyn_value:(id_read, Ops.uint32)
                    (get src [| iter pos |], Ops.single))))
           (loop_n v vocab (set out [| iter v |] (get tmp [| iter v |]))))
    in
    let raw =
      seq (zero tmp)
        (seq
           (loop_n pos positions
              (set tmp
                 [| fixed 0 |]
                 (add
                    (get tmp [| fixed 0 |])
                    (add (get src [| iter pos |]) (get ids [| iter pos |])))))
           (loop_n v vocab (set out [| iter v |] (get tmp [| iter v |]))))
    in
    let opt = optimize_scoped ~materialized:[ ids; src; out ] ~name:"res_scat_local" ~raw program in
    let ctx, routine = link ~name:"res_scat_local" opt in
    let ids_vals = [| 1.; 3.; 1.; 4. |] in
    let src_vals = [| 2.0; 5.0; 7.0; 11.0 |] in
    let ctx = run_linked (ctx, routine) ~seed:[ (ids, ids_vals); (src, src_vals) ] in
    Verdict.p_all2 "routine-local Set_dynamic RMW matches host reference"
      (Context.get_values ctx out) [| 0.; 9.0; 0.; 5.0 |] ~f:Float.equal;
    let source = read_routine routine "res_scat_local" in
    Verdict.p "routine-local Set_dynamic RMW stays unqualified and unrecorded"
      (routine.Context.volatility.Cs.volatile_rmw_reads = 0
      && not (String.is_substring source ~substring:"volatile"))
  in
  ()

(* {1 The volatility census's own bracket}

   [with_volatility_census] is what makes the census a property of the compiled routine rather than
   of whichever caller remembered to collect it (gh-ocannl-782), so its nesting and restoration are
   pinned directly — with hand-written entries, so these claims do not depend on what any backend
   renders. [requested] is part of the state it restores: a nested compile on a backend that asks
   for the workaround must not leave an enclosing collection claiming the flag of the inner one. *)
let () =
  let module Cs = Ir.C_syntax in
  let entries_equal =
    List.equal (fun (n1, s1) (n2, s2) -> String.equal n1 n2 && Cs.equal_volatility_site s1 s2)
  in
  let outer = Cs.Volatile_accumulation_reads "v1_outer"
  and inner = Cs.Plain_accumulator "v2_inner" in
  let inner_summary, outer_summary =
    Cs.with_volatility_census (fun () ->
        Cs.volatility_census () := ("outer_kernel", outer) :: !(Cs.volatility_census ());
        Cs.volatility_requested () := true;
        let (), inner_summary =
          Cs.with_volatility_census (fun () ->
              Cs.volatility_census () := ("inner_kernel", inner) :: !(Cs.volatility_census ()))
        in
        inner_summary)
  in
  Verdict.p "an inner bracket summarizes only its own sites"
    (entries_equal inner_summary.Cs.entries [ ("inner_kernel", inner) ]);
  (* Additively, not shadowing: an enclosing collection still sees what an inner compile censused,
     or wrapping the compile path in the bracket would silently empty every outer one. *)
  Verdict.p "the enclosing bracket sees both, in emission order"
    (entries_equal outer_summary.Cs.entries [ ("outer_kernel", outer); ("inner_kernel", inner) ]);
  Verdict.p "an inner bracket reports the capability of the compile it bracketed"
    (not inner_summary.Cs.requested);
  Verdict.p "the enclosing bracket keeps its own capability across the nested one"
    outer_summary.Cs.requested;
  Verdict.p "the counts classify what was collected"
    (outer_summary.Cs.volatile_accumulations = 1
    && outer_summary.Cs.plain_accumulations = 1
    && outer_summary.Cs.volatile_rmw_reads = 0);
  (* Over what the outer bracket collected: the global is empty AFTER a bracket that recorded
     something, not one that never had anything to leave behind. *)
  Verdict.p_empty "a completed bracket leaves the census global as it found it"
    ~over:outer_summary.Cs.entries
    !(Cs.volatility_census ());
  Verdict.p "collection is off outside every bracket" (not !(Cs.volatility_census_enabled ()))
