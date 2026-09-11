(* gh-ocannl-164: exact C rendering of the CPU-improvements bundle, on hand-built low-level IR
   through the actual backend codegen path ([C_syntax.compile_proc]).

   - A [Vectorized] loop renders as the backend's vectorization pragmas followed by a plain serial
   [for] under [Pure_C_config] (whose [vector_bytes = 0] disables explicit SIMD); with
   [vectorize_pragma = []] (the GPU configs' choice) it renders as the plain serial loop — the legal
   fallback. - With [vector_bytes > 0] (the cc backend's default), an eligible [Vectorized] body
   renders via GCC/Clang vector extensions: typedef + vector loads/arithmetic/stores in lanes-sized
   chunks, a splat for lane-uniform stores, and a serial remainder loop; ineligible bodies (e.g. a
   non-contiguous access) keep the pragma rendering. - Materialized kernel parameters carry the
   [restrict] qualifier; local stack arrays carry the SIMD alignment attribute. - A slice-alias
   tnode reaching the parameter list is rejected loudly: with [restrict] an aliased parent+view
   parameter pair would be a miscompile, not just a redundant pointer. Assignments lowering never
   produces one, but hand-built IR (schedule layer, tests) could. *)

open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level
module B = Ll_builders

let make_optimized llc tns : LL.optimized =
  let traced_store = Hashtbl.create (module Tn) in
  List.iter tns ~f:(fun tn -> ignore (LL.get_node traced_store tn : LL.traced_array));
  {
    traced_store;
    optimize_ctx = Ir.Low_level.empty_optimize_ctx ();
    llc;
    merge_node = None;
    workgroup_shared = Base.Set.empty (module Tn);
    simdgroup_fragments = Base.Set.empty (module Tn);
    swizzled = Base.Map.empty (module Tn);
    pipelined = Base.Map.empty (module Tn);
    zero_fringe = Base.Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Base.Set.empty (module Tn);
  }

let make_on_device id label =
  let tn =
    Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
      ~unpadded_dims:(lazy [| 8 |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode tn Tn.On_device 998;
  tn

let vec_loop ~axis tn =
  let i = Idx.get_symbol () in
  B.loop ~upto:7 ~axis i (B.set tn [| Idx.Iterator i |] (LL.Constant 1.0))

let compile_with_pure_config ~name optimized =
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    let procs = [| optimized.LL.llc |]
    let full_printf_support = true
  end))
  in
  let _kparams, doc, _launch = Syntax.compile_proc ~name [] optimized in
  doc

let () =
  (* --- [Vectorized] under the C config: pragmas + serial loop; restrict on the parameter. --- *)
  let out = make_on_device 1 "out" in
  let doc =
    compile_with_pure_config ~name:"vec_kernel"
      (make_optimized (vec_loop ~axis:LL.Vectorized out) [ out ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc;
  Stdio.printf "\n";

  (* --- [Vectorized] under a config with no pragmas (the GPU configs' serial fallback). --- *)
  let out2 = make_on_device 2 "out2" in
  let optimized2 = make_optimized (vec_loop ~axis:LL.Vectorized out2) [ out2 ] in
  let module Fallback_syntax = Ir.C_syntax.C_syntax (struct
    include Ir.C_syntax.Pure_C_config (struct
      let procs = [| optimized2.LL.llc |]
      let full_printf_support = true
    end)

    let vectorize_pragma = []
  end) in
  let _kparams, doc2, _launch =
    Fallback_syntax.compile_proc ~name:"vec_fallback_kernel" [] optimized2
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc2;
  Stdio.printf "\n";

  (* --- A local (stack-array) node picks up the SIMD alignment attribute. --- *)
  let local =
    let tn =
      Tn.create (Tn.Default Ops.single) ~id:3 ~label:[ "scratch" ]
        ~unpadded_dims:(lazy [| 8 |])
        ~padding:(lazy None)
        ()
    in
    Tn.update_memory_mode tn Tn.Local 997;
    tn
  in
  let out3 = make_on_device 4 "out3" in
  let i = Idx.get_symbol () in
  let llc3 =
    B.loop ~upto:7 i
      (LL.Seq
         ( B.set local [| Idx.Iterator i |] (LL.Constant 2.0),
           B.set out3 [| Idx.Iterator i |] (LL.Get (local, [| Idx.Iterator i |])) ))
  in
  let doc3 =
    compile_with_pure_config ~name:"aligned_local_kernel" (make_optimized llc3 [ local; out3 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc3;
  Stdio.printf "\n";

  (* --- Explicit SIMD emission with [vector_bytes = 32]: an eligible elementwise body renders as
     vector-extension code (8 float lanes); the lane-uniform constant store splats. --- *)
  let compile_with_vector_config ~name optimized =
    let module Syntax = Ir.C_syntax.C_syntax (struct
      include Ir.C_syntax.Pure_C_config (struct
        let procs = [| optimized.LL.llc |]
        let full_printf_support = true
      end)

      let vector_bytes = 32
    end) in
    let _kparams, doc, _launch = Syntax.compile_proc ~name [] optimized in
    doc
  in
  let inp = make_on_device 7 "inp" in
  let out4 = make_on_device 8 "out4" in
  let i = Idx.get_symbol () in
  let elementwise =
    B.loop ~upto:7 ~axis:LL.Vectorized i
      (LL.Seq
         ( B.set out4 [| Idx.Iterator i |]
             (LL.Binop
                ( Ops.Add,
                  (LL.Get (inp, [| Idx.Iterator i |]), Ops.single),
                  (LL.Constant 2.0, Ops.single) )),
           B.set inp [| Idx.Iterator i |] (LL.Constant 1.0) ))
  in
  let doc4 =
    compile_with_vector_config ~name:"vec_simd_kernel" (make_optimized elementwise [ inp; out4 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc4;
  Stdio.printf "\n";

  (* --- Ineligible for explicit SIMD (non-contiguous: coefficient 2 on the loop index): the pragma
     rendering remains. --- *)
  let inp2 = make_on_device 9 "inp2" in
  let out5 = make_on_device 10 "out5" in
  let i = Idx.get_symbol () in
  let strided =
    B.loop ~upto:3 ~axis:LL.Vectorized i
      (B.set out5 [| Idx.Iterator i |]
         (LL.Get (inp2, [| Idx.Affine { symbols = [ (2, i) ]; offset = 0 } |])))
  in
  let doc5 =
    compile_with_vector_config ~name:"vec_strided_kernel" (make_optimized strided [ inp2; out5 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc5;
  Stdio.printf "\n";

  (* --- [Ops.FMA] renders fused (the simplifier synthesizes it from mul-add trees): clang's
     [__builtin_elementwise_fma] where available, else a per-lane fmaf loop -- never the
     maybe-contracted [a * b + c], which could double-round against the fused scalar path. --- *)
  let g1 = make_on_device 11 "g1" in
  let g2 = make_on_device 12 "g2" in
  let out6 = make_on_device 13 "out6" in
  let i = Idx.get_symbol () in
  let fma_body =
    B.loop ~upto:7 ~axis:LL.Vectorized i
      (B.set out6 [| Idx.Iterator i |]
         (LL.Ternop
            ( Ops.FMA,
              (LL.Get (g1, [| Idx.Iterator i |]), Ops.single),
              (LL.Get (g2, [| Idx.Iterator i |]), Ops.single),
              (LL.Get (out6, [| Idx.Iterator i |]), Ops.single) )))
  in
  let doc6 =
    compile_with_vector_config ~name:"vec_fma_kernel" (make_optimized fma_body [ g1; g2; out6 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc6;
  Stdio.printf "\n";

  (* --- SIMD reduction rendering (gh-ocannl-468): an FMA-form dot-product accumulation renders as 4
     independent accumulator chains (ggml's ggml_vec_dot_f32 pattern) initialized from the first 4
     blocks, a fused main loop advancing by 32, a register + lane fold into the accumulator, and a
     serial tail. --- *)
  let make_sized id label dims =
    let tn =
      Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
        ~unpadded_dims:(lazy dims)
        ~padding:(lazy None)
        ()
    in
    Tn.update_memory_mode tn Tn.On_device 996;
    tn
  in
  let da = make_sized 14 "da" [| 72 |] in
  let db = make_sized 15 "db" [| 72 |] in
  let dacc = make_sized 16 "dacc" [| 1 |] in
  let i = Idx.get_symbol () in
  let dot_red =
    B.loop ~upto:71 ~axis:LL.Vectorized i
      (B.set dacc [| Idx.Fixed_idx 0 |]
         (LL.Ternop
            ( Ops.FMA,
              (LL.Get (da, [| Idx.Iterator i |]), Ops.single),
              (LL.Get (db, [| Idx.Iterator i |]), Ops.single),
              (LL.Get (dacc, [| Idx.Fixed_idx 0 |]), Ops.single) )))
  in
  let doc7 =
    compile_with_vector_config ~name:"vec_dot_reduce_kernel"
      (make_optimized dot_red [ da; db; dacc ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc7;
  Stdio.printf "\n";

  (* --- A max-reduce over extent 16 clamps to 2 chains; the combines and folds go through the
     per-lane fmaxf loops (no vector infix for Max), keeping the scalar path's NaN semantics. --- *)
  let ma = make_sized 17 "ma" [| 16 |] in
  let macc = make_sized 18 "macc" [| 1 |] in
  let i = Idx.get_symbol () in
  let max_red =
    B.loop ~upto:15 ~axis:LL.Vectorized i
      (B.set macc [| Idx.Fixed_idx 0 |]
         (LL.Binop
            ( Ops.Max,
              (LL.Get (macc, [| Idx.Fixed_idx 0 |]), Ops.single),
              (LL.Get (ma, [| Idx.Iterator i |]), Ops.single) )))
  in
  let doc8 =
    compile_with_vector_config ~name:"vec_max_reduce_kernel" (make_optimized max_red [ ma; macc ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc8;
  Stdio.printf "\n";

  (* --- An accumulating body the explicit renderings decline (strided, non-contiguous contrib) must
     fall back to a serial loop with NO vectorization pragma: the pragma would assert iteration
     independence that the loop-carried accumulation does not satisfy. Under Pure_C_config the
     pragmas are otherwise emitted (see the first kernel above). The fallback is the LOCALIZED
     serial form (gh-ocannl-693) -- the accumulator is held in a scope local across the nest at
     every precision -- which is orthogonal to the pragma question this kernel pins. --- *)
  let sa = make_sized 19 "sa" [| 16 |] in
  let sacc = make_sized 20 "sacc" [| 1 |] in
  let i = Idx.get_symbol () in
  let strided_red =
    B.loop ~upto:7 ~axis:LL.Vectorized i
      (B.set sacc [| Idx.Fixed_idx 0 |]
         (LL.Binop
            ( Ops.Add,
              (LL.Get (sacc, [| Idx.Fixed_idx 0 |]), Ops.single),
              (LL.Get (sa, [| Idx.Affine { symbols = [ (2, i) ]; offset = 0 } |]), Ops.single) )))
  in
  let doc9 =
    compile_with_pure_config ~name:"vec_strided_reduce_kernel"
      (make_optimized strided_red [ sa; sacc ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc9;
  Stdio.printf "\n";

  (* --- Register-tiled Tile_mma rendering (gh-ocannl-469, tinyBLAS's mnpack): a hand-built Tile_mma
     with an FMA-form fallback over awkward extents (6x29x5) renders the 4x3 C-tile of 8-lane
     vectors (AVX2-class register budget at vector_bytes = 32) held across the k-loop, with the row
     and column edges peeled into scalar fmaf loops — all under the same lane-0 guard as the
     fallback. --- *)
  let tile_operands () =
    let td = make_sized 21 "td" [| 6; 29 |] in
    let ta = make_sized 22 "ta" [| 6; 5 |] in
    let tb = make_sized 23 "tb" [| 5; 29 |] in
    (td, ta, tb)
  in
  let f00 = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |] in
  let tile_mma_loop ~body_form (td, ta, tb) =
    let lane = Idx.get_symbol () in
    let fi = Idx.get_symbol () and fj = Idx.get_symbol () and fl = Idx.get_symbol () in
    let dg = LL.Get (td, [| Idx.Iterator fi; Idx.Iterator fj |]) in
    let ag = LL.Get (ta, [| Idx.Iterator fi; Idx.Iterator fl |]) in
    let bg = LL.Get (tb, [| Idx.Iterator fl; Idx.Iterator fj |]) in
    let llsc =
      match body_form with
      | `Fma -> LL.Ternop (Ops.FMA, (ag, Ops.single), (bg, Ops.single), (dg, Ops.single))
      | `Plain ->
          LL.Binop
            ( Ops.Add,
              (dg, Ops.single),
              (LL.Binop (Ops.Mul, (ag, Ops.single), (bg, Ops.single)), Ops.single) )
    in
    let nest =
      let set = B.set td [| Idx.Iterator fi; Idx.Iterator fj |] llsc in
      let mk index to_ body = B.loop ~upto:to_ index body in
      mk fi 5 (mk fj 28 (mk fl 4 set))
    in
    B.loop ~upto:0 ~axis:LL.Workgroup lane
      (B.tile_mma ~d:(td, f00) ~a:(ta, f00) ~b:(tb, f00) ~m:6 ~n:29 ~k:5 ~lane nest)
  in
  let td, ta, tb = tile_operands () in
  let doc10 =
    compile_with_vector_config ~name:"tile_mma_reg_kernel"
      (make_optimized (tile_mma_loop ~body_form:`Fma (td, ta, tb)) [ td; ta; tb ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc10;
  Stdio.printf "\n";

  (* --- The plain-add (non-FMA) fallback form is declined: its maybe-contracted [a * b + c] could
     not promise bitwise equality with a fused vector twin, so the scalar fallback renders under the
     lane-0 guard instead. --- *)
  let td2 = make_sized 24 "td2" [| 6; 29 |] in
  let ta2 = make_sized 25 "ta2" [| 6; 5 |] in
  let tb2 = make_sized 26 "tb2" [| 5; 29 |] in
  let doc11 =
    compile_with_vector_config ~name:"tile_mma_plain_kernel"
      (make_optimized (tile_mma_loop ~body_form:`Plain (td2, ta2, tb2)) [ td2; ta2; tb2 ])
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc11;
  Stdio.printf "\n";

  (* --- The whole-vector FMA arms, at every (compute precision, lane count) the emission can reach
     (gh-ocannl-614, gh-ocannl-621). A kernel pins only the width its own [cc_vector_bytes] selects
     — 32 bytes above, and the setting is read once per process — so the table is printed directly
     instead. What the golden holds fixed: which widths have a whole-vector arm at all (everything
     else keeps the per-lane loop that gcc spills or scalarizes), the guard each arm sits behind,
     the operand order [a * b + dst], and the extra mask/rounding arguments of the AVX-512 forms. A
     guard that stopped being mutually exclusive with its siblings, or an arm that silently
     disappeared at one width, shows up here as a diff.

     The 16-lane f32, 8-lane f64, native-fp16 and aarch64 rows could not be executed where they were
     written; what was checked, per row, is that the arm compiles under the target its guard names
     and renders exactly one fused instruction at [-ffp-contract=off]. --- *)
  let module Arms = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    let procs = [||]
    let full_printf_support = true
  end))
  in
  Stdio.printf "\n/* --- whole-vector FMA arms by (compute precision, lanes) --- */\n";
  List.iter
    [ (Ops.half, "f16"); (Ops.single, "f32"); (Ops.double, "f64") ]
    ~f:(fun (prec, label) ->
      List.iter [ 2; 4; 8; 16; 32 ] ~f:(fun lanes ->
          Stdio.printf "\n/* %s x %d lanes */\n" label lanes;
          PPrint.ToChannel.pretty 0.9 100 Stdio.stdout
            (Arms.vec_acc_fma ~prec ~lanes ~dst:"acc__" ~a:"lhs__" ~b:"rhs__");
          Stdio.printf "\n"));

  (* --- An alias view as a would-be kernel parameter must be rejected loudly. --- *)
  let parent = make_on_device 5 "parent" in
  let view = make_on_device 6 "view" in
  Tn.set_alias_of view ~parent
    ~batch_idx:
      {
        Idx.static_symbol = Idx.get_symbol ();
        static_range = Some 1;
        used_as_extent = false;
        used_as_slice = false;
      };
  (match
     try
       ignore
         (compile_with_pure_config ~name:"alias_kernel"
            (make_optimized (vec_loop ~axis:LL.Serial view) [ view ])
           : PPrint.document);
       None
     with Invalid_argument msg -> Some msg
   with
  | Some msg -> Verdict.p "alias parameter rejected" (String.is_substring msg ~substring:"restrict")
  | None -> Stdio.printf "alias parameter rejected: false\n");
  Stdio.printf "%!"
