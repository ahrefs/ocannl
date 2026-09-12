(* Pure assembly dialect and source-anchor controls (gh-ocannl-937). Run without emitting a kernel
   or invoking any compiler. *)
open Base
module Census = Test_utils.Asm_census

let routine = "census_kernel"

(* {2 Assembler dialects this host cannot produce}

   Finding after finding here has been about an assembly shape a Linux/gcc box never emits, on a
   platform CI actually builds. Three silenced the census WHOLESALE rather than skewing a number --
   Apple's dot-less [LBB0_9] labels, which made every branch target unrecognizable; clang's [.file 0
   "dir" "name" md5 0x...], whose checksum reads as the path and leaves no file number matched; and
   the [; =>This Inner Loop Header] annotation, which stops a label line from ending in a colon. All
   three fail the same way, every row reporting "no loop", which is loud.

   The fourth is quieter and is the one to learn from: Apple's arm64 assembler writes a NEON
   instruction's arrangement on the MNEMONIC ([fmla.4s v0, v1, v2]), so the loops were found, their
   spans and instruction totals were right, and only the vector/scalar SPLIT read zero. A claim
   asking for more vector than scalar work then reads [0 > 0] on a perfectly rendered tile, while
   its neighbours pass on whatever few instructions the operand rules still caught. So a dialect
   fixture owes a claim about the COUNTS and not only about the loop being found.

   None of the four is reachable from a fixture compiled here, so they are pinned against SYNTHETIC
   assembly instead. That is not a weaker check for these properties -- what is being tested is the
   parser's handling of a documented syntax, and the syntax is what is written down. *)

let dialect_probes () =
  (* Line 3 carries the anchor; the [.loc] directives below point at it. *)
  let source = "int f(void) {\n  /* prologue */\n  ACC = MAXOF(ACC, SRC);\n  return 0;\n}\n" in
  let anchor = Census.anchor_lines ~source ~patterns:[ "MAXOF" ] in
  let elf =
    String.concat ~sep:"\n"
      [
        "\t.file\t\"census_kernel.c\"";
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:";
        ".L2:";
        "\t.loc 1 3 12";
        "\taddps\t%xmm1, %xmm0";
        "\tjne\t.L2";
        "\tret";
      ]
  in
  (* Every Mach-O detail that has silenced this census, in ONE fixture, because they were found one
     round at a time: dot-less [LBB] labels, a checksummed [.file], the [;] loop-header annotation
     Apple clang writes after a label, and the underscore Mach-O puts on every C symbol. The first
     version of this probe used a BARE [LBB0_1:] and so did not exercise the annotation -- which is
     how the [;] gap survived the round that added the probe. A fixture for a dialect should carry
     the dialect's noise, not just its identifiers. *)
  let darwin_clang =
    String.concat ~sep:"\n"
      [
        "\t.file\t0 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "_f:                                     ; @f";
        "LBB0_1:                                 ; =>This Inner Loop Header: Depth=1";
        "\t.loc\t1 3 12";
        "\tfmax.4s\tv0, v0, v1";
        "\tb.ne\tLBB0_1";
        "\tret";
      ]
  in
  (* The same loop with a libm call in it, which the census must SEE. Mach-O spells it [_fmaxf];
     matching [libm_names] without stripping that underscore reports zero calls and passes the
     central gh-ocannl-649 claim over a loop that is exactly the regression. *)
  let darwin_with_libm =
    String.concat ~sep:"\n"
      [
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "_f:";
        "LBB0_1:                                 ; =>This Inner Loop Header: Depth=1";
        "\t.loc\t1 3 12";
        "\tcallq\t_fmaxf";
        "\tjne\tLBB0_1";
        "\tretq";
      ]
  in
  (* {b The same aarch64 loop in the two dialects that assemble it.} GAS puts a NEON instruction's
     arrangement on the REGISTERS ([fmla v2.4s, v3.4s, v4.4s]) and Apple's assembler puts it on the
     MNEMONIC ([fmla.4s v2, v3, v4]), and the counts have to read the same either way: an
     instruction's class is a fact about the instruction, not about how a listing spells it.

     This is the fourth Mach-O detail to have silenced the census (gh-ocannl-752), and the first to
     silence only the COUNTS: the dot-less labels, the checksummed [.file] and the [;] annotation
     each made every loop go missing, which every "found" claim reports at once. Here the loops were
     found, the spans and instruction totals were right, and only the vector/scalar split read zero
     -- so the tile rows on CI's macos-latest leg were being judged on the handful of instructions
     the operand rules happened to catch ([ldr q0], [fcvtl v2.4s, v0.4h], which have no Apple alias)
     out of the 28 to 78 real ones. That is why these two fixtures carry the counting claim below
     and not merely the "is it found" one the probe started with: the found-ness was never in
     question on this dialect. *)
  let arm64_body spelling =
    String.concat ~sep:"\n"
      ([ "\t.file 1 \"/build/census_kernel.c\""; "f:"; ".L2:"; "\t.loc 1 3 12" ]
      @ spelling
      @ [ "\tfmax\ts5, s5, s6"; "\tb.ne\t.L2"; "\tret" ])
  in
  let arm64_gnu = arm64_body [ "\tfmax\tv0.4s, v0.4s, v1.4s"; "\tfmla\tv2.4s, v3.4s, v4.4s" ] in
  let arm64_apple = arm64_body [ "\tfmax.4s\tv0, v0, v1"; "\tfmla.4s\tv2, v3, v4" ] in
  (* Clang may attribute an inlined expression to another line of the generated construct. This
     synthetic ELF/clang listing points at line 8, while the unique [START] anchor is line 6. The
     brace-delimited range models that toolchain without admitting a neighbouring construct. *)
  let clang_source =
    String.concat ~sep:"\n"
      [
        "int f(void) {";
        "  /* Main logic. */";
        "  { /* one generated reduction */";
        "    /* A } in prose is not the end of the construct. */";
        "    const char *brace = \"{\";";
        "    START = SOURCE;";
        "    for (;;) {";
        "      ACC = MAXOF(ACC, SOURCE);";
        "    }";
        "  }";
        "}";
      ]
  in
  let clang_exact = Census.anchor_lines ~source:clang_source ~patterns:[ "START =" ] in
  let clang_range =
    Census.anchor_block_lines ~source:clang_source ~after_pattern:"/* Main logic. */"
      ~patterns:[ "START =" ]
  in
  let clang_x86 =
    String.concat ~sep:"\n"
      [
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "f:";
        ".LBB0_1:";
        "\t.loc\t1 8 12";
        "\tmaxps\t%xmm1, %xmm0";
        "\tjne\t.LBB0_1";
        "\tretq";
      ]
  in
  (* The fp8 codec loop and accumulator loop share source attribution. The ordinary selector below
     sees the smaller codec loop; [Smallest_outer_anchor_carrier] must see the immediate combine and
     therefore the deliberately injected libm call after the codec loop. This is the negative
     control for the production no-libm claims: the same predicate must reject this wrong
     combine. *)
  let nested =
    String.concat ~sep:"\n"
      [
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:";
        ".Louter:";
        "\t.loc 1 8 12";
        "\taddps\t%xmm1, %xmm0";
        ".Lcodec:";
        "\t.loc 1 8 12";
        "\taddps\t%xmm2, %xmm0";
        "\tjne\t.Lcodec";
        "\tcallq\tfmaxf";
        "\tjne\t.Louter";
        "\tretq";
      ]
  in
  let nested_parsed = Census.parse ~asm:nested ~source_basename:(routine ^ ".c") in
  let innermost = Census.census_in nested_parsed Census.Max_min ~anchor:clang_range in
  let combine =
    Census.census_in ~selection:Census.Smallest_outer_anchor_carrier nested_parsed Census.Max_min
      ~anchor:clang_range
  in
  let has_libm = function Some c -> c.Census.counts.Census.libm_calls > 0 | None -> false in
  let unknown =
    String.concat ~sep:"\n"
      [
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:";
        ".L2:";
        "\t.loc 1 3 12";
        "\tocannl_future_dialect\tfoo, bar";
        "\tjne\t.L2";
        "\tret";
      ]
  in
  let censused asm = Census.census Census.Max_min ~asm ~source_basename:(routine ^ ".c") ~anchor in
  Verdict.p_all "the census models GCC/GAS and Clang assembly on x86-64 and aarch64"
    [
      ("gnu/elf", elf);
      ("apple-clang", darwin_clang);
      ("apple-clang+libm", darwin_with_libm);
      ("arm64/gas", arm64_gnu);
      ("arm64/apple", arm64_apple);
      ("x86/elf-clang", clang_x86);
    ]
    ~f:(fun (name, asm) ->
      Census.loop_edges ~asm > 0
      && Option.is_some
           (if String.equal name "x86/elf-clang" then
              Census.census Census.Max_min ~asm ~source_basename:(routine ^ ".c")
                ~anchor:clang_range
            else censused asm));
  (* Exact counts rather than "the two agree": agreement alone is what the blind reading also
     satisfied, both dialects having answered [vector=0 scalar_fp=0] before the arrangement
     mnemonics were read. Two packed instructions and one scalar one, in a fixture of three, so the
     claim fails if either dialect's packed forms stop counting AND if a packed form starts counting
     as scalar work (or the [fmax s5] scalar one as packed). *)
  Verdict.p_all "both arm64 dialects' instructions are classified, not only their loops found"
    [ ("arm64/gas", arm64_gnu); ("arm64/apple", arm64_apple) ]
    ~f:(fun (_, asm) ->
      match censused asm with
      | Some c -> c.Census.counts.Census.vector_ops = 2 && c.Census.counts.Census.scalar_fp_ops = 1
      | None -> false);
  Verdict.p "a libm call spelled the Mach-O way is still counted as one"
    (match censused darwin_with_libm with
    | Some c -> c.Census.counts.Census.libm_calls > 0
    | None -> false);
  Verdict.p "clang line attribution is found through the construct range, not the exact anchor line"
    (Option.is_none
       (Census.census Census.Max_min ~asm:clang_x86 ~source_basename:(routine ^ ".c")
          ~anchor:clang_exact)
    && Option.is_some
         (Census.census_source_in
            (Census.parse ~asm:clang_x86 ~source_basename:(routine ^ ".c"))
            Census.Max_min ~source:clang_source ~patterns:[ "START =" ]
            ~after_pattern:"/* Main logic. */"));
  Verdict.p "the fp8 combine selector rejects a deliberate libm regression outside its codec loop"
    ((not (has_libm innermost))
    && has_libm combine
    && Option.value_map innermost ~default:false ~f:(fun inner ->
        Option.value_map combine ~default:false ~f:(fun outer -> inner.span < outer.span)));
  Verdict.p "an instruction in an unknown dialect is reported as residual"
    (match censused unknown with
    | Some c ->
        c.Census.counts.Census.instructions = 2
        && c.Census.counts.Census.residual = 2
        && c.Census.counts.Census.vector_ops = 0
        && c.Census.counts.Census.scalar_fp_ops = 0
    | None -> false)

(* {2 That the exact anchor is tried before the range}

   The range fallback is a WIDENING, and a widening that fires unconditionally changes the answer on
   the toolchain that never needed it. gcc attributes the established rows precisely AND emits loops
   of its own inside the same construct -- a staging copy, a vectorizer peel -- which the
   construct's brace-delimited range names just as readily as the accumulator loop the widening was
   for. [Innermost] then reports the compiler's little loop: a smaller span, a handful of
   instructions, and every inequality about "the accumulator loop" evaluated over something that is
   not one. That is what the first CI cycle of the fallback showed on the gcc columns
   (gh-ocannl-844), and it is a regression no fixture here could have caught: the clang probe above
   pins the case where the exact anchor answers NOTHING, so the two readings never disagree in it
   and the ordering between them is unobservable. Hence a fixture whose readings DISAGREE, and a
   claim on each of the three answers -- what the range alone says, what the exact lines alone say,
   and which of the two {!Census.census_source_in} returns. Pinning only the third would pass just
   as well over a fixture that had stopped discriminating.

   The source is a reduction whose tensor-derived name appears on its accumulator declaration, its
   update and its store, so the smallest brace-delimited block containing all three is the
   construct's -- which also contains the staging loop's line, one nesting level in. The exact lines
   name the update alone, which only the accumulator loop carries. *)
let anchor_precedence_probe () =
  (* Line 4 declares the accumulator, line 10 updates it and line 13 stores it, all three carrying
     the row's tensor-derived name; line 7 is the staging copy's, which no anchor pattern names. *)
  let source =
    String.concat ~sep:"\n"
      [
        "int f(void) {";
        "  /* Main logic. */";
        "  { /* one generated reduction */";
        "    float acc_dot_bf16[K];";
        "    for (int i = 0; i < N; ++i) {";
        "      for (int lane = 0; lane < K; ++lane) {";
        "        stage[lane] = bfloat16_to_single(raw[i][lane]);";
        "      }";
        "      for (int k = 0; k < K; ++k) {";
        "        acc_dot_bf16[k] = ocannl_fma(acc_dot_bf16[k], stage[k], wgt[i][k]);";
        "      }";
        "    }";
        "    out_dot_bf16 = ocannl_hsum(acc_dot_bf16);";
        "  }";
        "  return 0;";
        "}";
      ]
  in
  (* [.Li] is the row's own outer loop, [.Lk] its accumulator loop, and [.Lstage] the staging copy
     gcc emitted for line 7 -- inside [.Li], nested beside [.Lk] and shorter than it, which is the
     whole point: [Innermost] prefers whichever candidate is shortest, so admitting [.Lstage] as a
     candidate is admitting it as the answer. Only [.Lk] carries a [.loc] for an exact anchor line;
     [.Li] carries one only by containing [.Lk]. *)
  let asm =
    String.concat ~sep:"\n"
      [
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:";
        ".Li:";
        "\t.loc 1 5 3";
        "\taddq\t$1, %rax";
        ".Lstage:";
        "\t.loc 1 7 9";
        "\tmovdqu\t(%rsi), %xmm3";
        "\tcvtph2ps\t%xmm3, %xmm3";
        "\tjne\t.Lstage";
        ".Lk:";
        "\t.loc 1 10 9";
        "\tvmovups\t(%rdi), %ymm1";
        "\tvfmadd231ps\t(%rdx), %ymm1, %ymm0";
        "\tvmovups\t%ymm0, (%rdi)";
        "\taddq\t$32, %rdi";
        "\taddq\t$32, %rdx";
        "\tcmpq\t%rcx, %rdi";
        "\tjne\t.Lk";
        "\tjne\t.Li";
        "\tret";
      ]
  in
  let patterns = [ "dot_bf16" ] in
  let parsed = Census.parse ~asm ~source_basename:(routine ^ ".c") in
  let read anchor = Census.census_in parsed Census.Fma ~anchor in
  let exact = read (Census.anchor_lines ~source ~patterns) in
  let range =
    read (Census.anchor_block_lines ~source ~after_pattern:"/* Main logic. */" ~patterns)
  in
  (* Label AND span, so a reading that found the right loop under the wrong edge, or the wrong loop
     under a label rename, is not a pass. *)
  let is c ~label ~span =
    match c with
    | Some (c : Census.t) -> String.equal c.Census.loop_label label && c.Census.span = span
    | None -> false
  in
  Verdict.p "the brace range alone reads the staging loop gcc nested beside the accumulator"
    (is range ~label:".Lstage" ~span:4);
  Verdict.p "the exact source anchor reads the accumulator loop" (is exact ~label:".Lk" ~span:8);
  Verdict.p "a census row takes its exact source anchor ahead of the brace-range fallback"
    (is
       (Census.census_source_in parsed Census.Fma ~source ~patterns
          ~after_pattern:"/* Main logic. */")
       ~label:".Lk" ~span:8)

let () =
  dialect_probes ();
  anchor_precedence_probe ()
