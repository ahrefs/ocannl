(** What the codegen-text scan calls a member, on input built to break it rather than on whatever
    the repository happens to hold today (gh-ocannl-712).

    The live census in [codegen_text_inventory] exercises only the shapes some golden or some test
    currently spells, and both halves of the classifier are wrong in ways that are silent. A marker
    that stops matching shrinks the inventory rather than failing it, so a golden quietly leaves the
    list a codegen author reads. A marker that matches too much adds a file nobody has to re-run,
    which is the cheaper error but trains the reader to skim. And on the source side an unrecognised
    idiom loses the fragment, not the file, so the inventory keeps looking complete.

    So each rule is pinned here on input of its own, and -- for every marker whose needle a claim
    label could plausibly contain -- beside the nearest thing that must NOT be a member: a
    {!Verdict} line quoting the kernel's vocabulary, a memory-mode table naming [On_device], prose
    about how a constant is spelled. Those are the negative controls of the RULE. *)

(* This file's own fixtures spell emitted-kernel syntax, and this file is deliberately NOT a member
   of either population: the fixtures are inputs to the classifier, not assertions about a kernel
   this repository emits, so a codegen change owes them nothing. What decides that is the rule
   itself rather than an exemption -- the source rule asks whether a file READS generated source,
   and nothing here does. *)

open Base
module Scan = Test_utils.Codegen_text_scan

let printf = Test_utils.Refusal_control_manifest.printf
let fail fmt = Printf.ksprintf Verdict.fail fmt

(** How a classified golden reads, for comparison: the families, then where the evidence came from.
    [none] is a file the scan does not call a member. *)
let render_golden = function
  | None -> "none"
  | Some (g : Scan.golden) ->
      Printf.sprintf "[%s]%s%s"
        (String.concat ~sep:" " g.Scan.families)
        (match g.Scan.by_extension with Some ext -> " extension " ^ ext | None -> "")
        (match g.Scan.tags with [] -> "" | tags -> " markers " ^ String.concat ~sep:" " tags)

let c_kernel =
  {|
 void zero_out_codegen(
    float *restrict out) {

  /* Local declarations and initialization. */
  float acc_a[2] __attribute__((aligned(32))) = {0};

  /* Main logic. */
  for (int32_t i2 = 0; i2 <= 1; ++i2) {
    acc_a[i2] = (float)(0.0);
  }
}
|}

let golden_cases =
  [
    (* The marker families, each on the text it was written for. *)
    ( "a C kernel is a member by its markers, under any name",
      "some_test.expected",
      c_kernel,
      "[c] markers c-align-attr c-decl-banner c-for c-logic-banner c-prec-cast c-restrict" );
    ( "a CUDA kernel names the launch vocabulary",
      "gpu_test.expected",
      "__global__ void k(float *restrict o) { o[threadIdx.x] = (float)(1.0); }",
      "[c cuda] markers c-prec-cast c-restrict cuda-kernel" );
    ( "a Metal kernel names its address spaces",
      "msl_test.expected",
      "kernel void k(device float *o [[buffer(0)]]) { threadgroup float frag[8]; }",
      "[metal] markers metal-kernel" );
    ( "the compact IR serialization is an assignment without the spaces",
      "canonical_render.expected",
      "rendering:\n\
      \       c;zero <cr_out>;for b0=0..2@Grid{set \
       <cr_out>[(2*b0+1*b1+5),]:=scope(<cr_acc>.1)[b0,]{nop;};}\n\
      \       an alpha-variant lowering renders identically: true\n",
      "[ll] markers ll-assign" );
    ( "a low-level IR dump is a member by its loop headers and assignments",
      "dump_test.expected",
      "c_fwd (): /* c fwd */\n  for i10 = 0 to 2 {\n    a[i10] := 5*i10;\n  }\n  /* end */",
      "[ll] markers ll-assign ll-end ll-loop" );
    ( "a routine log carries both the IR line and the C rendering",
      "run-cc-0-0.log.expected",
      "COMMENT: init params for g\n# a[0] := -4.0;\na[0]{=MAYBE UNINITIALIZED} = (float)(-4.0)",
      "[c ll routine-log] markers c-prec-cast ll-assign routine-log" );
    (* Declared by extension: a member whatever it holds, because the name says which artifact it
       snapshots. A machine without the toolchain records a notice here and the file is still that
       backend's snapshot, to be re-recorded when the hardware next runs. *)
    ( "a snapshot is a member by its extension whatever it holds",
      "top_down_prec.cu.expected",
      "CUDA is not available on this machine.\n",
      "[cuda] extension .cu.expected" );
    ( "a declaring extension names the substrate; markers stay as evidence",
      "top_down_prec.metal.expected",
      "kernel void k() { }",
      "[metal] extension .metal.expected markers metal-kernel" );
    ( "HIP spells CUDA's launch vocabulary, and is still HIP",
      "zero_out_local_decl.hip.expected",
      "__global__ void k(float *restrict o) { o[threadIdx.x] = (float)(1.0); }",
      "[hip] extension .hip.expected markers c-prec-cast c-restrict cuda-kernel" );
    ( "a .hip snapshot is its own family, not CUDA's",
      "x.hip.expected",
      "nothing here\n",
      "[hip] extension .hip.expected" );
    ( "an .ll snapshot is declared even when the dump is empty",
      "x.ll.expected",
      "",
      "[ll] extension .ll.expected" );
    (* The negative controls: the nearest legitimate goldens the classifier must leave alone. Every
       one of these is a real shape from this repository's test output. *)
    ( "a verdict quoting the kernel's vocabulary is prose, not Metal",
      "schedule_pad.expected",
      "padded GPU intrinsics fire against the threadgroup fragment: true\n\
       padded 33x65x70 runs the register-tiled micro-kernel: true\n",
      "none" );
    ( "a claim about how a constant is spelled is not a constant",
      "prose.expected",
      "every emitted float constant carries a radix point, as in (float)(0.0): true\n",
      "none" );
    ( "a PASS column is a verdict however it reads",
      "columns.expected",
      "the kernel declares threadgroup float fragments: PASS\n",
      "none" );
    ( "a memory-mode table is not device code",
      "test_metal_storage_mode.expected",
      "On_device              -> Shared\nOn_host                -> Managed\n",
      "none" );
    ( "a numeric table of shapes is not a loop nest",
      "shapes.expected",
      "batch 4 x input 3 -> output 2\ntotal elements: 24\n",
      "none" );
    ( "a failure line is a verdict too",
      "failed.expected",
      "FAIL: the kernel contains (float)(0.0) where it should not\n",
      "none" );
  ]

(** The emitters these cases are written against.

    A FIXTURE, not the frontier. The live census derives its set from the compiler libraries'
    compiled interfaces ({!Emitter_frontier}, gh-ocannl-748) and [emitter_frontier_cases] controls
    that derivation on interfaces built to break it; what is pinned HERE is what the scan does with
    such a set once it has one -- which call sites it recognises, which it refuses, and where the
    text a buffer-writing emitter deposits travels. *)
let emitters =
  let emitter ?(destinations = []) name origin =
    { Scan.emitter_name = name; Scan.origins = [ origin ]; Scan.destinations }
  in
  [
    emitter "compile_proc" "Ir.C_syntax.C_syntax.compile_proc";
    emitter "compile_main" "Ir.C_syntax.C_syntax.compile_main";
    emitter "to_doc" "Ir.Low_level.to_doc";
    emitter "to_doc_cstyle" "Ir.Low_level.to_doc_cstyle";
    emitter "emit" "Ir.Low_level.Canonical_render.emit" ~destinations:[ Scan.At_label "buf" ];
    (* An emitter whose buffer carries no label, so a call site addresses it by position. Nothing in
       the libraries has that shape today; the rule has to have it either way, since a position is
       what an unlabelled destination is. *)
    emitter "render_into" "Ir.Low_level.render_into" ~destinations:[ Scan.At_position 0 ];
  ]

let render_site = function
  | None -> "none"
  | Some (s : Scan.site) ->
      Printf.sprintf "%s%s%s%s"
        (String.concat ~sep:" " s.Scan.pins)
        (if s.Scan.partial then " +partial" else "")
        (if s.Scan.direct then " +direct" else "")
        (if s.Scan.rendered then " +rendered" else "")

let source_cases =
  [
    ( "assert_emits pins its contains argument",
      {ocaml|let () = Generated.assert_emits ~routine:"r" ~contains:"__shared__" "shared"|ocaml},
      {|"__shared__"|} );
    ( "assert_omits pins the fragment that must be absent",
      {ocaml|let () = Test_utils.Generated.assert_omits ~routine:r ~contains:"volatile" "no rmw"|ocaml},
      {|"volatile"|} );
    ( "the has idiom: a predicate closing over the source",
      {ocaml|let () =
  let src = Generated.read "pad_packed" in
  let has s = String.is_substring src ~substring:s in
  p "tiled" (has "Tile_mma register tiling" && not (has "tmma_"))|ocaml},
      {|"Tile_mma register tiling" "tmma_"|} );
    ( "a predicate taking the source as a parameter pins at a tainted call site",
      {ocaml|let src_has src s = String.is_substring src ~substring:s
let () =
  let vec = Generated.read "nsc_vec_bf16" in
  p "widened" (src_has vec "OCANNL_VEC_WIDEN_BFLOAT16")|ocaml},
      {|"OCANNL_VEC_WIDEN_BFLOAT16"|} );
    ( "the same predicate over text that is not generated source pins nothing",
      {ocaml|let src_has src s = String.is_substring src ~substring:s
let () =
  let _ = Generated.read "r" in
  p "backend" (src_has backend_name "metal")|ocaml},
      "" );
    ( "a substring test straight against the read is a pin",
      {ocaml|let () = p "cast" (String.is_substring (Generated.read "r") ~substring:"(float)(0.0)")|ocaml},
      {|"(float)(0.0)"|} );
    ( "a sprintf format is a pinned fragment with a hole in it",
      {ocaml|let () =
  let src = Generated.read "pad_packed" in
  let has s = String.is_substring src ~substring:s in
  p "bound" (has (Printf.sprintf "< (int)(%d.0))) {" m_ext))|ocaml},
      {|sprintf "< (int)(%d.0))) {"|} );
    ( "text the scan cannot name marks the itemisation partial, without losing the file",
      {ocaml|let () =
  let src = Generated.read "r" in
  let has s = String.is_substring src ~substring:s in
  p "shared" (has shared_keyword)|ocaml},
      "+partial" );
    ( "taint reaches through a helper that returns the source",
      {ocaml|let read_on_cpu routine = if on_cpu then Generated.read routine else ""
let () =
  let src = read_on_cpu "nsc_half_fma" in
  let has t = String.is_substring src ~substring:t in
  p "fma" (has "OCANNL_HALF_FMA")|ocaml},
      {|"OCANNL_HALF_FMA"|} );
    ( "a routine name is not a pinned fragment",
      {ocaml|let () = Generated.assert_emits ~routine:"aw_bf16_naive" ~contains:"fmaf(" "fma"|ocaml},
      {|"fmaf("|} );
    ( "a test opening build_files/ itself is a member, and is told apart",
      {ocaml|let sources =
  Stdlib.Sys.readdir (Utils.build_files_dir ())
  |> Array.to_list
  |> List.map ~f:(fun f -> Stdio.In_channel.read_all f)
let () =
  let has substring = List.exists sources ~f:(String.is_substring ~substring) in
  p "guard" (has ": (float)(0.0))")|ocaml},
      {|": (float)(0.0))" +direct|} );
    ( "counting occurrences with ~pattern pins the fragment counted",
      {ocaml|let () =
  let count_sub src sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
  let src2 = Generated.read "pipe_mm_d2" in
  p "rotated" (count_sub src2 "% 2" >= 4)|ocaml},
      {|"% 2"|} );
    ( "a concatenation with a literal part is a fragment with a hole in it",
      {ocaml|let () =
  let src = Generated.read "flit_f32" in
  let has s = String.is_substring src ~substring:s in
  p "spelled" (has ("(float)(" ^ spelling ^ ")"))|ocaml},
      {|"(float)(" ^ ... ^ ")"|} );
    ( "the haystack may reach the test through a local binding",
      {ocaml|let has sub s =
  let body = match String.substr_index s ~pattern:"Main logic" with
    | Some i -> String.subo s ~pos:i
    | None -> s
  in
  String.is_substring body ~substring:sub
let () =
  let src = Generated.read "uvl_fwd" in
  p "lane" (has "_uniform_lane(" src)|ocaml},
      {|"Main logic" "_uniform_lane("|} );
    ( "a predicate over the backend's NAME pins nothing, however it is spelled",
      {ocaml|let () =
  let _ = Generated.read "r" in
  let on s = String.is_substring backend_name ~substring:s in
  p "gpu" (on "cuda" || on "metal")|ocaml},
      "" );
    ( "an unannotated compiler-plan classifier remains visible as an unattributed pin",
      {ocaml|let plan_has_mutable_input plan =
  List.exists ["-fplugin"; "-fprofile-use"]
    ~f:(fun option -> String.is_substring plan ~substring:option)
let () = ignore (Generated.read "r")|ocaml},
      "+partial" );
    ( "the compiler-plan annotation is binding-scoped and cannot hide a real generated-text pin",
      {ocaml|let plan_has_mutable_input plan =
  List.exists ["-fplugin"; "-fprofile-use"]
    ~f:(fun option -> String.is_substring plan ~substring:option)
[@@ocannl.codegen_text.compiler_plan]
let () =
  p "real pin"
    (String.is_substring (Generated.read "r") ~substring:"emitted_kernel_token")|ocaml},
      {|"emitted_kernel_token"|} );
    ( "a source reached through a tuple pattern is still a source",
      {ocaml|let run () = (values, Generated.read "uvl_fwd")
let () =
  let _vals, src = run () in
  p "vec" (String.is_substring src ~substring:"_uniform_vec(")|ocaml},
      {|"_uniform_vec("|} );
    ( "a test binding its own build_file is not reading the artifact directory",
      {ocaml|let build_file path ~extra_pad entries = write path entries ~extra_pad
let () =
  let _ = build_file "aligned.safetensors" ~extra_pad:0 entries in
  p "aligned" true|ocaml},
      "none" );
    ( "the qualified Utils.build_file is a read of the artifact directory",
      {ocaml|let () = p "wrote" (Stdlib.Sys.file_exists (Utils.build_file "k.c"))|ocaml},
      "+direct" );
    (* Module aliases. A conventional short alias is ordinary OCaml, and a scan matching the literal
       component would not merely mis-attribute such a file -- it would drop it from the inventory
       entirely, which is the silent direction (Codex P2, round 1). *)
    ( "a module alias of the reader is the reader",
      {ocaml|module G = Test_utils.Generated
let () = G.assert_emits ~routine:"r" ~contains:"__syncthreads()" "synced"|ocaml},
      {|"__syncthreads()"|} );
    ( "an alias of an alias is an alias",
      {ocaml|module G = Test_utils.Generated
module H = G
let () =
  let src = H.read "r" in
  p "shared" (String.is_substring src ~substring:"__shared__")|ocaml},
      {|"__shared__"|} );
    ( "an alias bound in expression position counts too",
      {ocaml|let go () =
  let module G = Test_utils.Generated in
  let src = G.read "r" in
  String.is_substring src ~substring:"threadgroup float"|ocaml},
      {|"threadgroup float"|} );
    ( "an alias of Utils is a direct artifact read",
      {ocaml|module U = Utils
let () = p "wrote" (Stdlib.Sys.file_exists (U.build_file "k.c"))|ocaml},
      "+direct" );
    (* Lexical scope (gh-ocannl-1079): an alias is the module it names only where its binding is in
       scope. Collected file-wide, an alias of the reader made a same-named module elsewhere the
       reader too, and its fixture text a pin. *)
    ( "a same-named alias in another scope is not the reader",
      {ocaml|let fixture () =
  let module G = struct
    let read name = Stdio.In_channel.read_all name
  end in
  p "fixture" (String.is_substring (G.read "fixture.txt") ~substring:"(float)(0.0)")
let kernel () =
  let module G = Test_utils.Generated in
  G.assert_emits ~routine:"r" ~contains:"__shared__" "shared"|ocaml},
      {|"__shared__"|} );
    ( "an alias shadowed by a later binding is no longer the reader",
      {ocaml|module G = Test_utils.Generated
let () = G.assert_emits ~routine:"r" ~contains:"__shared__" "shared"
let fixture () =
  let module G = Fixture_reader in
  p "fixture" (String.is_substring (G.read "fixture.txt") ~substring:"(float)(0.0)")|ocaml},
      {|"__shared__"|} );
    ( "a functor parameter shadows an alias of the reader in the functor's body",
      {ocaml|module G = Test_utils.Generated
let () = G.assert_emits ~routine:"r" ~contains:"__shared__" "shared"
module Check (G : READER) = struct
  let () = p "fixture" (String.is_substring (G.read "fixture.txt") ~substring:"(float)(0.0)")
end|ocaml},
      {|"__shared__"|} );
    ( "an unqualified read is not the reader, which is what the qualifier is for",
      {ocaml|let read routine = Stdio.In_channel.read_all routine
let () = p "loaded" (String.is_substring (read "fixture.txt") ~substring:"(float)(0.0)")|ocaml},
      "none" );
    (* The third route to generated text: rendering it in memory, with no artifact in between. A
       rule naming only the other two left five sources and a whole scan root invisible (Codex P2,
       round 2). *)
    ( "rendering the emitter's document is reaching generated text",
      {ocaml|let compile optimized =
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    let procs = [| optimized |]
  end)) in
  let _kparams, doc, _launch = Syntax.compile_proc ~name [] optimized in
  doc_to_string doc
let () =
  let c = compile opt in
  p "guard" (String.is_substring c ~substring:"? producer[")|ocaml},
      {|"? producer[" +rendered|} );
    ( "a dump printer is an emitter too, under whatever alias",
      {ocaml|module LL = Ir.Low_level
let () =
  let src = render (LL.to_doc_cstyle () stmt) in
  p "radix" (String.is_substring src ~substring:"-0.0")|ocaml},
      {|"-0.0" +rendered|} );
    ( "a test emitting to a golden pins nothing and is still a member",
      {ocaml|module LL = Ir.Low_level
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc () llc)|ocaml},
      "+rendered" );
    ( "an unqualified to_doc is the test's own, not an emitter",
      {ocaml|let to_doc row = PPrint.string (render_row row)
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (to_doc header)|ocaml},
      "none" );
    (* Round 3's genre: the membership rules learned the third route and the PIN rules had not, so a
       fragment could be dropped while the file stayed listed -- nothing looked wrong, and a grep of
       the inventory missed the assertion. *)
    ( "an inline emitter render in the haystack still pins its fragment",
      {ocaml|module LL = Ir.Low_level
let () = p "radix" (String.is_substring (render (LL.to_doc () llc)) ~substring:"-0.0")|ocaml},
      {|"-0.0" +rendered|} );
    ( "an inline build_files read in the haystack still pins its fragment",
      {ocaml|let () =
  p "slots"
    (String.is_substring (Stdio.In_channel.read_all (Utils.build_file "k.metal"))
       ~substring:"uint* __pool_slots")|ocaml},
      {|"uint* __pool_slots" +direct|} );
    ( "a helper that hard-codes the fragment pins it, taking the source as its parameter",
      {ocaml|let has_barrier src = String.is_substring src ~substring:"__syncthreads()"
let () = p "barrier" (has_barrier (Generated.read "r"))|ocaml},
      {|"__syncthreads()"|} );
    ( "a helper that slices on a banner and tests its own parameter pins both",
      {ocaml|let has sub s =
  let body = match String.substr_index s ~pattern:"Main logic" with
    | Some i -> String.subo s ~pos:i
    | None -> s
  in
  String.is_substring body ~substring:sub
let () = p "lane" (has "_uniform_lane(" (Generated.read "r"))|ocaml},
      {|"Main logic" "_uniform_lane("|} );
    ( "a fragment named through a binding is still that fragment",
      {ocaml|let () =
  let arrow = " := " in
  let statement = render (Ir.Low_level.to_doc () stmt) in
  p "arrow" (Option.is_some (String.substr_index statement ~pattern:arrow))|ocaml},
      {|" := " +rendered|} );
    (* A fragment named through a binding is the binding the name reaches WHERE IT IS SPELLED
       (gh-ocannl-1079). Keyed by name file-wide, two legs binding one name -- the shape of
       schedule_mma_matmul's per-backend [body_begin] markers -- lost both pins even when the values
       agreed (staging#855), and a parameter or a shadowing binding of the name read as some literal
       elsewhere in the file (gh-ocannl-1063 recorded another leg's text that way). *)
    ( "same-named literal lets in different scopes each pin their own text",
      {ocaml|let metal_leg () =
  let src = Generated.read "r_metal" in
  let body_begin = "/* simdgroup fragment reduction body begins */" in
  p "metal" (Option.is_some (String.substr_index src ~pattern:body_begin))
let cuda_leg () =
  let src = Generated.read "r_cuda" in
  let body_begin = "/* mma.sync fragment reduction body begins */" in
  p "cuda" (Option.is_some (String.substr_index src ~pattern:body_begin))|ocaml},
      {|"/* mma.sync fragment reduction body begins */" "/* simdgroup fragment reduction body begins */"|}
    );
    ( "same-named literal lets pin their text even when the values agree",
      {ocaml|let register_body_begin = "/* mma.sync fragment reduction body begins */"
let bf16_leg () =
  let src = Generated.read "r_bf16" in
  p "bf16" (Option.is_some (String.substr_index src ~pattern:register_body_begin))
let f16_leg () =
  let src = Generated.read "r_f16" in
  let register_body_begin = "/* mma.sync fragment reduction body begins */" in
  p "f16" (Option.is_some (String.substr_index src ~pattern:register_body_begin))|ocaml},
      {|"/* mma.sync fragment reduction body begins */"|} );
    ( "a parameter is not a same-named literal let elsewhere in the file",
      {ocaml|let resident src ~body_begin = Option.is_some (String.substr_index src ~pattern:body_begin)
let () = p "resident" (resident (Generated.read "r") ~body_begin:"/* wmma body begins */")
let metal_marker () =
  let body_begin = "/* simdgroup fragment reduction body begins */" in
  describe body_begin|ocaml},
      {|"/* wmma body begins */"|} );
    ( "labelled source and several text parameters match reordered call arguments",
      {ocaml|let resident ~body_begin ~src ~body_end =
  Option.is_some (String.substr_index src ~pattern:body_begin)
  && Option.is_some (String.substr_index src ~pattern:body_end)
let () = p "resident" (resident ~body_end:"end marker" ~src:(Generated.read "r") ~body_begin:"begin marker")|ocaml},
      {|"begin marker" "end marker"|} );
    ( "labelled parameters do not shift positional source or text arguments",
      {ocaml|let has ?(enabled = true) ~unused src sub = String.is_substring src ~substring:sub
let () = p "marker" (has ~unused:"not a pin" (Generated.read "r") "real marker")|ocaml},
      {|"real marker"|} );
    ( "a helper reading generated source internally pins its labelled markers",
      {ocaml|let check ~build ~marker =
  let src = Generated.read (build ()) in
  String.is_substring src ~substring:marker
let () = p "marker" (check ~build:compile ~marker:"internal marker")|ocaml},
      {|"internal marker"|} );
    ( "labelled predicates over ordinary text pin no generated fragments",
      {ocaml|let has ~src ~sub = String.is_substring src ~substring:sub
let () =
  let generated = Generated.read "r" in
  p "ordinary" (has ~src:backend_name ~sub:"not a pin")|ocaml},
      "" );
    ( "a binding shadowing a literal let is not that literal",
      {ocaml|let marker = "/* stale marker */"
let () =
  let src = Generated.read "r" in
  let marker = current_marker () in
  p "marker" (String.is_substring src ~substring:marker)|ocaml},
      "+partial" );
    ( "a buffer-writing serializer is an emitter too",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let render llc =
  let buf = Buffer.create 256 in
  CR.emit ~buf policy llc;
  Buffer.contents buf
let () = p "free" (String.is_substring (render llc) ~substring:"s0")|ocaml},
      {|"s0" +rendered|} );
    (* Round 5's genre: the write and the read of a buffer-writing emitter can sit in different
       bindings, and then neither carries taint -- the first binds no name, the second calls no
       emitter. The DESTINATION is what the text lands in, so it is seeded from the call. *)
    ( "the buffer a serializer writes into carries the text, across bindings",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let buf = Buffer.create 256
let () = CR.emit ~buf policy llc
let source = Buffer.contents buf
let () = p "free" (String.is_substring source ~substring:"s0")|ocaml},
      {|"s0" +rendered|} );
    ( "an emitter bound to a local name is still the emitter",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let write = CR.emit
let () =
  let buf = Buffer.create 256 in
  write ~buf policy llc;
  p "free" (String.is_substring (Buffer.contents buf) ~substring:"s0")|ocaml},
      {|"s0" +rendered|} );
    ( "an alias of that alias is the emitter too",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let write = CR.emit
let write_again = write
let () =
  let buf = Buffer.create 256 in
  write_again ~buf policy llc;
  p "free" (String.is_substring (Buffer.contents buf) ~substring:"s0")|ocaml},
      {|"s0" +rendered|} );
    (* Round 2's genre, and the last shape of it a scan of one file can follow: the emitter behind a
       WRAPPER, whose own parameter is what the caller's buffer arrives through. *)
    ( "a wrapper around an emitter carries its caller's buffer",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let write ~buf policy llc = CR.emit ~buf policy llc
let () =
  let output = Buffer.create 256 in
  write ~buf:output policy llc;
  p "free" (String.is_substring (Buffer.contents output) ~substring:"s0")|ocaml},
      {|"s0" +rendered|} );
    ( "a wrapper whose buffer parameter carries no label is addressed by position",
      {ocaml|module LL = Ir.Low_level
let write buf llc = LL.render_into buf llc
let () =
  let output = Buffer.create 256 in
  write output llc;
  p "radix" (String.is_substring (Buffer.contents output) ~substring:"-0.0")|ocaml},
      {|"-0.0" +rendered|} );
    (* And the backstop for the shapes it cannot: PPrint's own buffer renderer is not an emitter of
       ours, so nothing taints [buf] -- the file is a member through [LL.to_doc] all the same, and
       what must not happen is the fragment vanishing with no sign. *)
    ( "a buffer this scan did not see filled marks the itemisation partial",
      {ocaml|module LL = Ir.Low_level
let () =
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  p "radix" (String.is_substring (Buffer.contents buf) ~substring:"-0.0")|ocaml},
      "+partial +rendered" );
    ( "a buffer read through an alias of Buffer marks it partial just the same",
      {ocaml|module LL = Ir.Low_level
module B = Buffer
let () =
  let buf = B.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  p "radix" (String.is_substring (B.contents buf) ~substring:"-0.0")|ocaml},
      "+partial +rendered" );
    ( "an unattributed buffer read travels along bindings like taint does",
      {ocaml|module LL = Ir.Low_level
let () =
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  let source = Buffer.contents buf in
  p "radix" (String.is_substring source ~substring:"-0.0")|ocaml},
      "+partial +rendered" );
    ( "an unattributed buffer reaching a helper marks it partial too",
      {ocaml|module LL = Ir.Low_level
let has src sub = String.is_substring src ~substring:sub
let () =
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  let source = Buffer.contents buf in
  p "radix" (has source "-0.0")|ocaml},
      "+partial +rendered" );
    ( "a buffer aliased under a signature constraint is still a buffer",
      {ocaml|module LL = Ir.Low_level
module B = (Buffer : module type of Buffer)
let () =
  let buf = B.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  p "radix" (String.is_substring (B.contents buf) ~substring:"-0.0")|ocaml},
      "+partial +rendered" );
    ( "a local name bound to something else is not an emitter",
      {ocaml|let write = Buffer.add_string
let () =
  let buf = Buffer.create 256 in
  write buf (describe shape);
  p "shapes agree" (String.is_substring (Buffer.contents buf) ~substring:"3x5")|ocaml},
      "none" );
    ( "a buffer nobody wrote generated text into carries nothing",
      {ocaml|let buf = Buffer.create 256
let () = Buffer.add_string buf (describe shape)
let source = Buffer.contents buf
let () = p "shapes agree" (String.is_substring source ~substring:"3x5")|ocaml},
      "none" );
    ( "a test that reads no generated source is not a member",
      {ocaml|let () = p "shapes agree" (String.is_substring rendered ~substring:"3x5")|ocaml},
      "none" );
    ( "naming the reader in a comment is not reading it",
      {ocaml|(* Generated.read would answer this, but the check is on values. *)
let () = p_all2 "values" got want ~f:Float.equal|ocaml},
      "none" );
  ]

(** Spellings the scan refuses rather than approximates: {!Scan.rejections}. Each case is a source
    and how many refusals it earns.

    Every route is attributed by the qualifier at the call site, and an [open] takes the qualifier
    away -- after which the call reads exactly like a local function of the same name and the file
    drops out of the census silently. Refusing is what keeps that convention from being adopted
    without anyone noticing; the negative controls below are the shapes that must stay legal, since
    a refusal that fires on an innocent file is a broken build (gh-ocannl-748). *)
let rejection_cases =
  [
    ( "opening the reader hides its calls from the qualifier",
      {ocaml|open Test_utils.Generated
let () = p "shared" (String.is_substring (read "r") ~substring:"__shared__")|ocaml},
      1 );
    ( "opening an ALIAS of the reader hides them just the same",
      {ocaml|module G = Test_utils.Generated
open G
let () = assert_emits ~routine:"r" ~contains:"__syncthreads()" "synced"|ocaml},
      1 );
    ( "opening the emitter's module hides the render",
      {ocaml|open Ir.Low_level.Canonical_render
let () =
  emit ~buf policy llc;
  p "free" (String.is_substring (Buffer.contents buf) ~substring:"s0")|ocaml},
      1 );
    ( "an open in expression position is an open",
      {ocaml|let go () =
  let open Test_utils.Generated in
  String.is_substring (read "r") ~substring:"threadgroup float"|ocaml},
      1 );
    ( "opening an ALIAS of the emitter's module hides it under a name no origin spells",
      {ocaml|module CR = Ir.Low_level.Canonical_render
open CR
let () =
  emit ~buf policy llc;
  p "free" (String.is_substring (Buffer.contents buf) ~substring:"s0")|ocaml},
      1 );
    (* The controls. Opening a module is ordinary OCaml; what is refused is opening one whose names
       this scan attributes, and then using one of THOSE names. *)
    ( "opening a module a FUNCTOR produced hides the emitter too",
      {ocaml|let compile optimized =
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    let procs = [| optimized |]
  end)) in
  let open Syntax in
  let _kparams, doc, _launch = compile_proc ~name [] optimized in
  doc_to_string doc|ocaml},
      1 );
    ( "opening a functor application directly hides the emitter as surely",
      {ocaml|let compile optimized =
  let open Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    let procs = [| optimized |]
  end)) in
  let _kparams, doc, _launch = compile_proc ~name [] optimized in
  doc_to_string doc|ocaml},
      1 );
    ( "an include hides the emitter as an open does",
      {ocaml|include Ir.Low_level.Canonical_render

let () =
  emit ~buf policy llc;
  p "free" (String.is_substring (Buffer.contents buf) ~substring:"s0")|ocaml},
      1 );
    ( "a name the file binds for itself is not refused where that binding is in scope",
      {ocaml|open Ir.Low_level

let to_doc x = local_render x
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (to_doc value)|ocaml},
      0 );
    (* And the other half of scope (gh-ocannl-1079): a binding of the name somewhere ELSE in the
       file does not take the opened one out of reach, and an alias rebound in a later scope does
       not change what an earlier open of it opened. Both were refusals not made -- the silent
       direction. *)
    ( "a name bound only in another scope does not shadow the opened one",
      {ocaml|open Ir.Low_level

let helper () =
  let to_doc x = local_render x in
  to_doc value
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (to_doc llc)|ocaml},
      1 );
    ( "an alias rebound in a later scope does not change what an earlier open opened",
      {ocaml|module CR = Ir.Low_level.Canonical_render
open CR
let () = emit ~buf policy llc
let other () =
  let module CR = Fixture in
  CR.describe ()|ocaml},
      1 );
    ( "an open governs its own scope, not the whole file",
      {ocaml|let render_row row =
  let open Ir.Low_level in
  describe row
let to_doc row = PPrint.string (render_row row)
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (to_doc header)|ocaml},
      0 );
    ( "a structure-level open governs the items after it",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let () = p "before" (emit_count = 3)
open CR
let () = emit ~buf policy llc|ocaml},
      1 );
    ( "an open inside a nested module dies with it",
      {ocaml|module Inner = struct
  open Ir.Low_level
  let () = p "inner" (describe llc <> "")
end

let to_doc row = PPrint.string (render_row row)
let () = PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (to_doc header)|ocaml},
      0 );
    ( "opening Utils without reading the artifact directory is fine",
      {ocaml|open Utils
let () = p "tree" (Tree_map.is_empty (Tree_map.empty ()))|ocaml},
      0 );
    ( "a name an emitter shares with an unopened module is not hidden",
      {ocaml|open Base
let () = p "count" (to_doc rows = 3)|ocaml},
      0 );
    ( "the qualified spelling is what everything already uses",
      {ocaml|module G = Test_utils.Generated
let () = G.assert_emits ~routine:"r" ~contains:"__shared__" "shared"|ocaml},
      0 );
  ]

(** The goldens a test's own output makes members, or does not: {!Scan.classify_associated}. Each
    case is the golden's contents, and what the rule answers for a golden sitting beside a source
    member. *)
let association_cases =
  [
    ( "a table of dumped constants is text derived from generated code",
      "exact value                %cd dump                   C-style dump\n\
      \       0x1.999999999999ap-4       0.1                        0.1\n\
      \       every dumped constant parses back to the double it names: true\n",
      "[derived] beside t.ml" );
    ( "a census of the decisions a kernel was built from moves with them",
      "seeds: standard, both hoistable: total=18 whole=2 packed=12\n\
      \       seeded packed pad-composition matches the serial twin bitwise: true\n",
      "[derived] beside t.ml" );
    (* The negative control that decides the rule: a schedule test's golden is a column of booleans,
       and a boolean does not move when codegen does -- the claim goes on reading true. Pulling
       those in would add a line per schedule test and train the reader to skim. *)
    ( "a golden of nothing but claims is the test's verdict, not its output",
      "padded packed matmul matches the serial twin bitwise: true\n\
      \       pad guard over an unstaged operand is rejected: true\n",
      "none" );
    ( "blank lines do not make a verdict golden into output",
      "first claim holds: true\n\n   \nsecond claim holds: PASS\n",
      "none" );
    ("an empty golden is not output either", "", "none");
  ]

let render_association = function
  | None -> "none"
  | Some (g : Scan.golden) ->
      Printf.sprintf "[%s]%s"
        (String.concat ~sep:" " g.Scan.families)
        (match g.Scan.beside with Some source -> " beside " ^ source | None -> "")

let () =
  List.iter association_cases ~f:(fun (name, contents, expected) ->
      let found =
        render_association (Scan.classify_associated ~path:"t.expected" ~contents ~source:"t.ml")
      in
      if String.equal found expected then printf "ok: association -- %s\n" name
      else fail "association -- %s: expected [%s], found [%s]" name expected found);
  List.iter
    [
      ("a plain source", "d/x.ml", "d/x");
      ("a select real", "d/x.real.ml", "d/x");
      ("a select stub", "d/x.missing.ml", "d/x");
    ]
    ~f:(fun (name, path, expected) ->
      let found = Scan.source_stem path in
      if String.equal found expected then printf "ok: stem -- %s\n" name
      else fail "stem -- %s: expected [%s], found [%s]" name expected found);
  List.iter golden_cases ~f:(fun (name, path, contents, expected) ->
      let found = render_golden (Scan.classify_golden ~path ~contents) in
      if String.equal found expected then printf "ok: golden -- %s\n" name
      else fail "golden -- %s: expected [%s], found [%s]" name expected found);
  List.iter source_cases ~f:(fun (name, source, expected) ->
      let found =
        try render_site (Scan.classify_source ~emitters ~path:"case.ml" ~contents:source)
        with _ ->
          fail "source -- %s: the snippet does not parse" name;
          "<unparsed>"
      in
      if String.equal (String.strip found) (String.strip expected) then
        printf "ok: source -- %s\n" name
      else fail "source -- %s: expected [%s], found [%s]" name expected found);
  List.iter rejection_cases ~f:(fun (name, source, expected) ->
      let found =
        try List.length (Scan.rejections ~emitters ~path:"case.ml" ~contents:source)
        with _ ->
          fail "rejection -- %s: the snippet does not parse" name;
          -1
      in
      if expected = found then printf "ok: rejection -- %s\n" name
      else fail "rejection -- %s: expected %d refusals, found %d" name expected found);
  Test_utils.Refusal_control_manifest.print "codegen_text_inventory.ml"
