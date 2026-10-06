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
module Attempt = Test_utils.Scan_attempt

let printf = Stdio.printf
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
        (match s.Scan.partial with
        | [] -> ""
        | boundaries ->
            " +partial(" ^ String.concat ~sep:", " (List.map boundaries ~f:Scan.boundary_tag) ^ ")")
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
      "+partial(unvalidated)" );
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
      "+partial(computed)" );
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
      "+partial(callback)" );
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
      {|"_uniform_vec(" +partial(opaque)|} );
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
    ( "each test in a helper checks its own haystack",
      {ocaml|let src = Generated.read "r"
let check ~marker ~backend_marker =
  String.is_substring src ~substring:marker
  && String.is_substring backend_name ~substring:backend_marker
let () = p "marker" (check ~marker:"kernel marker" ~backend_marker:"cuda")|ocaml},
      {|"kernel marker"|} );
    ( "a helper reading generated source inline pins its labelled marker",
      {ocaml|let check ~routine ~marker =
  String.is_substring (Generated.read routine) ~substring:marker
let () = p "marker" (check ~routine:"r" ~marker:"inline marker")|ocaml},
      {|"inline marker"|} );
    ( "omitted optional markers use their defaults and explicit markers override them",
      {ocaml|let has src ?(marker = "default marker") () = String.is_substring src ~substring:marker
let explicit src ?(marker = "unused default") () = String.is_substring src ~substring:marker
let () =
  p "default" (has (Generated.read "r") ());
  p "explicit" (explicit (Generated.read "r") ~marker:"explicit marker" ())|ocaml},
      {|"default marker" "explicit marker"|} );
    ( "callbacks retain hard-coded markers whose source the helper validates",
      {ocaml|let check routine =
  String.is_substring (Generated.read routine) ~substring:"callback marker"
let () = List.iter ["r"] ~f:check|ocaml},
      {|"callback marker" +partial(callback)|} );
    ( "constrained labelled source and marker parameters retain their names",
      {ocaml|let has ~(src : string) ~(marker : string) =
  String.is_substring src ~substring:marker
let () = p "marker" (has ~src:(Generated.read "r") ~marker:"typed marker")|ocaml},
      {|"typed marker"|} );
    ( "a shadowed source parameter does not validate the caller marker",
      {ocaml|let has ~src ~marker =
  describe src;
  let src = backend_name in
  String.is_substring src ~substring:marker
let () = p "ordinary" (has ~src:(Generated.read "r") ~marker:"not a pin")|ocaml},
      "+partial(rebound)" );
    ( "forwarding wrappers with generated defaults stay visibly partial",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let check ?(src = Generated.read "r") ~marker () = has src ~marker
let () = p "marker" (check ~marker:"default source marker" ())|ocaml},
      "+partial(forwarded)" );
    ( "anonymous forwarding callbacks stay visibly partial",
      {ocaml|let barrier src = String.is_substring src ~substring:"eta marker"
let () = List.iter [Generated.read "r"] ~f:(fun src -> barrier src)|ocaml},
      {|"eta marker" +partial(callback)|} );
    ( "generated source stored through mutation stays visibly partial",
      {ocaml|let has src = String.is_substring src ~substring:"mutated marker"
let source = ref ""
let () = source := Generated.read "r"; p "marker" (has !source)|ocaml},
      "+partial(mutation)" );
    ( "a prefix ! the file binds itself is not a dereference",
      {ocaml|let ( ! ) s = String.lowercase s
let has src = String.is_substring src ~substring:"bang marker"
let () = ignore (Generated.read "r"); p "m" (has !backend_name)|ocaml},
      "+partial(unvalidated)" );
    ( "an infix := the file binds itself is not a write",
      {ocaml|let ( := ) _ text = text
let has src = String.is_substring src ~substring:"assign marker"
let probe () = ignore (dummy := Generated.read "r"); backend_name
let () = p "m" (has (probe ()))|ocaml},
      "+partial(unvalidated)" );
    ( "a partial application keeps the callee's own buffer boundary beside its unsupplied one",
      {ocaml|let render flag src =
  let b = Buffer.create 8 in
  Buffer.add_string b src;
  if flag then Buffer.contents b else src
let partial = render true
let () = p "m" (String.is_substring (partial (Generated.read "r")) ~substring:"buffered partial marker")|ocaml},
      {|"buffered partial marker" +partial(unsupplied, mutation)|} );
    ( "a forwarding wrapper used as a callback stays visibly partial",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let check src = has src "callback wrapper marker"
let () = List.iter [Generated.read "r"] ~f:check|ocaml},
      {|"callback wrapper marker" +partial(callback)|} );
    ( "a forwarding wrapper keeps unresolved caller markers visibly partial",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let check src marker = has src marker
let () = p "marker" (check (Generated.read "r") "forwarded marker")|ocaml},
      "+partial(forwarded)" );
    ( "a self-sourcing wrapper has no false partial",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let check name = p "marker" (has (Generated.read name) ~marker:"own marker")
let () = check "r"|ocaml},
      {|"own marker"|} );
    ( "an unused self-sourcing wrapper has no false partial",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let check name = p "marker" (has (Generated.read name) ~marker:"unused own marker")
let unused_check = check|ocaml},
      {|"unused own marker"|} );
    ( "a wrapper sourcing through local aliases has no false partial",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let check name =
  let src = Generated.read name in
  let src = strip_volatile_casts src in
  p "marker" (has src ~marker:"normalized own marker")
let () = check "r"|ocaml},
      {|"normalized own marker"|} );
    ( "same-parameter source normalization preserves real pins despite a callback",
      {ocaml|let has src ~marker =
  let src = strip_volatile_casts src in
  String.is_substring src ~substring:marker
let () =
  p "marker" (has (Generated.read "r") ~marker:"real marker");
  List.iter [Generated.read "other"] ~f:(has ~marker:"callback marker")|ocaml},
      {|"real marker" +partial(unsupplied)|} );
    ( "normalizing an unrelated shadow preserves uncertainty without caller pins",
      {ocaml|let has src ~marker =
  let src = strip_volatile_casts backend_name in
  String.is_substring src ~substring:marker
let () = p "ordinary" (has (Generated.read "r") ~marker:"not a pin")|ocaml},
      "+partial(rebound)" );
    ( "a generated collection callback keeps its known fragment and explicit uncertainty",
      {ocaml|let src = Generated.read "r"
let () = List.iter (String.split_lines src) ~f:(fun line ->
  String.is_substring line ~substring:"known callback marker")|ocaml},
      {|"known callback marker" +partial(callback)|} );
    ( "pattern-bound source fragments inherit the scrutinee provenance",
      {ocaml|let src = Generated.read "r"
let () = match String.split_lines src with
  | line :: _ -> p "marker" (String.is_substring line ~substring:"first line marker")
  | [] -> ()|ocaml},
      {|"first line marker"|} );
    ( "ordinary arguments forwarded by a known wrapper contribute no fragments",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let check src = has src "ordinary wrapper marker"
let () = ignore (Generated.read "r"); p "ordinary" (check backend_name)|ocaml},
      "+partial(forwarded)" );
    ( "exception messages do not inherit generated source from the protected computation",
      {ocaml|let () = try
  ignore (Generated.read "r"); failwith "failure"
with Failure msg -> p "exception" (String.is_substring msg ~substring:"ordinary failure marker")|ocaml},
      "" );
    ( "a completed predicate result is an ordinary value in either conditional branch",
      {ocaml|let has src = String.is_substring src ~substring:"result marker"
let () =
  let src = Generated.read "r" in
  let ok = if enabled then has src else not (has src) in
  p "marker" ok|ocaml},
      {|"result marker"|} );
    ( "a normalizer never borrows generated provenance from a different call",
      {ocaml|let normalize src = String.lowercase src
let has src ~marker =
  let src = normalize src in
  String.is_substring src ~substring:marker
let () =
  p "kernel" (has (Generated.read "r") ~marker:"kernel marker");
  p "ordinary" (has backend_name ~marker:"not a kernel marker")|ocaml},
      {|"kernel marker" +partial(unvalidated)|} );
    ( "a source-returning helper validates an internal read through its routine parameter",
      {ocaml|let read routine = Generated.read routine
let check ~routine ~marker =
  let src = read routine in
  String.is_substring src ~substring:marker
let () = p "marker" (check ~routine:"r" ~marker:"returned source marker")|ocaml},
      {|"returned source marker"|} );
    ( "a source-returning buffer helper preserves the unresolved-buffer backstop",
      {ocaml|let read buf = Buffer.contents buf
let () =
  ignore (Generated.read "r");
  p "marker" (String.is_substring (read buf) ~substring:"buffer marker")|ocaml},
      "+partial(mutation)" );
    ( "a buffer-returning helper preserves inputs used by earlier writes with explicit uncertainty",
      {ocaml|let render doc =
  let buf = Buffer.create 100 in
  PPrint.ToBuffer.pretty 0.7 100 buf doc;
  Buffer.contents buf
let src = render (LL.to_doc value)
let () = p "marker" (String.is_substring src ~substring:"buffered document marker")|ocaml},
      {|"buffered document marker" +partial(mutation) +rendered|} );
    ( "a helper result depends only on its returned formal argument",
      {ocaml|let first x ignored = x
let src = first backend_name (Generated.read "r")
let () = p "ordinary" (String.is_substring src ~substring:"not a returned pin")|ocaml},
      "" );
    ( "the returned formal still carries generated provenance past an ignored ordinary argument",
      {ocaml|let first x ignored = x
let src = first (Generated.read "r") backend_name
let () = p "marker" (String.is_substring src ~substring:"returned formal marker")|ocaml},
      {|"returned formal marker"|} );
    ( "a constant-returning helper does not borrow generated argument provenance",
      {ocaml|let ordinary src = backend_name
let src = ordinary (Generated.read "r")
let () = p "ordinary" (String.is_substring src ~substring:"constant is not a pin")|ocaml},
      "" );
    ( "exception match patterns are independent of the successful generated result",
      {ocaml|let () = match Generated.read "r" with
  | src -> p "marker" (String.is_substring src ~substring:"successful result marker")
  | exception Failure msg -> p "exception" (String.is_substring msg ~substring:"ordinary exception marker")|ocaml},
      {|"successful result marker"|} );
    ( "an ordinary tuple-match sibling does not borrow generated provenance",
      {ocaml|let () = match (Generated.read "r", backend_name) with
  | src, name -> p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      "" );
    ( "a generated tuple-match component retains its own marker",
      {ocaml|let () = match (Generated.read "r", backend_name) with
  | src, name -> p "marker" (String.is_substring src ~substring:"tuple marker")|ocaml},
      {|"tuple marker"|} );
    ( "an ordinary record-match sibling does not borrow generated provenance",
      {ocaml|let () = match { src = Generated.read "r"; name = backend_name } with
  | { src; name } -> p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      "" );
    ( "a generated record-match component retains its own marker",
      {ocaml|let () = match { src = Generated.read "r"; name = backend_name } with
  | { src; name } -> p "marker" (String.is_substring src ~substring:"record marker")|ocaml},
      {|"record marker"|} );
    ( "tuple-let components keep generated and ordinary sources separate",
      {ocaml|let src, name = (Generated.read "r", backend_name)
let () =
  p "marker" (String.is_substring src ~substring:"tuple-let marker");
  p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      {|"tuple-let marker"|} );
    ( "record-let components keep generated and ordinary sources separate",
      {ocaml|let { src; name } = { src = Generated.read "r"; name = backend_name }
let () =
  p "marker" (String.is_substring src ~substring:"record-let marker");
  p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      {|"record-let marker"|} );
    ( "a match over a try result retains the successful generated payload",
      {ocaml|let () = match (try Some (Generated.read "r") with _ -> None) with
  | Some src -> p "marker" (String.is_substring src ~substring:"try result marker")
  | None -> ()|ocaml},
      {|"try result marker"|} );
    ( "a match over an ordinary try result ignores a preceding generated read",
      {ocaml|let () = match (try ignore (Generated.read "r"); Some backend_name with _ -> None) with
  | Some src -> p "ordinary" (String.is_substring src ~substring:"cuda")
  | None -> ()|ocaml},
      "" );
    ( "a selected optional default leaves no callable identity on the completed boolean",
      {ocaml|let has ?(marker = "selected default marker") src () =
  String.is_substring src ~substring:marker
let () = let ok = has (Generated.read "r") () in p "marker" ok|ocaml},
      {|"selected default marker"|} );
    ( "an ordinary replacement retains uncertainty without contributing a body literal",
      {ocaml|let has src = let src = backend_name in String.is_substring src ~substring:"cuda"
let () = p "ordinary" (has (Generated.read "r"))|ocaml},
      "+partial(rebound)" );
    ( "an aliased ordinary replacement retains uncertainty without contributing a body literal",
      {ocaml|let has src =
  let src = String.lowercase backend_name in
  let alias = src in String.is_substring alias ~substring:"cuda"
let () = p "ordinary" (has (Generated.read "r"))|ocaml},
      "+partial(rebound)" );
    ( "a followed partial call retains its unresolved source dependency and known fragment",
      {ocaml|let second ignored src = src
let partial = second backend_name
let src = partial (Generated.read "r")
let () = p "marker" (String.is_substring src ~substring:"partial source marker")|ocaml},
      {|"partial source marker" +partial(unsupplied)|} );
    ( "a partial call returning an ordinary supplied argument ignores later generated text",
      {ocaml|let first src ignored = src
let partial = first backend_name
let src = partial (Generated.read "r")
let () = p "ordinary" (String.is_substring src ~substring:"not a partial pin")|ocaml},
      "" );
    ( "dynamic optional source forwarding retains generated evidence with uncertainty",
      {ocaml|let id ?src () = Option.value_exn src
let forwarded = Some (Generated.read "r")
let src = id ?src:forwarded ()
let () = p "marker" (String.is_substring src ~substring:"forwarded source marker")|ocaml},
      {|"forwarded source marker" +partial(unsupplied)|} );
    ( "dynamic optional source forwarding preserves its possible generated default",
      {ocaml|let id ?(src = Generated.read "r") () = src
let forwarded = None
let src = id ?src:forwarded ()
let () = p "marker" (String.is_substring src ~substring:"possible default marker")|ocaml},
      {|"possible default marker" +partial(unsupplied)|} );
    ( "dynamic ordinary forwarding with an ordinary default contributes no fragment",
      {ocaml|let id ?(src = backend_name) () = src
let forwarded = Some backend_name
let src = id ?src:forwarded ()
let () = ignore (Generated.read "r"); p "ordinary" (String.is_substring src ~substring:"not a forwarded pin")|ocaml},
      "" );
    ( "a function-case result inherits its actual positional input",
      {ocaml|let id = function src -> src
let src = id (Generated.read "r")
let () = p "marker" (String.is_substring src ~substring:"case source marker")|ocaml},
      {|"case source marker"|} );
    ( "a function-case constant result does not borrow its generated input",
      {ocaml|let id = function _ -> backend_name
let src = id (Generated.read "r")
let () = p "ordinary" (String.is_substring src ~substring:"not a case pin")|ocaml},
      "" );
    ( "a function-case predicate validates its source and caller marker",
      {ocaml|let has ~marker = function src -> String.is_substring src ~substring:marker
let () = p "marker" (has ~marker:"case predicate marker" (Generated.read "r"))|ocaml},
      {|"case predicate marker"|} );
    ( "a function-case predicate on ordinary input contributes no caller marker",
      {ocaml|let has ~marker = function src -> String.is_substring src ~substring:marker
let () = ignore (Generated.read "r"); p "ordinary" (has ~marker:"not a case predicate pin" backend_name)|ocaml},
      "+partial(unvalidated)" );
    ( "an untraced diagnostic callback contributes uncertainty but no generated fragment",
      {ocaml|let () = ignore (Generated.read "r")
let () = List.iter nodes ~f:(fun node ->
  String.is_substring (Ir.Tnode.debug_name node) ~substring:"bwd_rowdot")|ocaml},
      "+partial(callback)" );
    ( "an untraced refutation callback contributes no emitted-text marker",
      {ocaml|let () = ignore (Generated.read "r")
let () = p_all "refutation" (Sspace.refutations tree) ~f:(fun wit ->
  String.is_substring wit ~substring:"does not divide innermost contraction extent k=12")|ocaml},
      "+partial(callback)" );
    ( "a known helper preserves the predicate callable it returns",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let id f = f
let check = id has
let () = p "marker" (check (Generated.read "r") "returned callable marker")|ocaml},
      {|"returned callable marker" +partial(callback)|} );
    ( "a returned predicate callable on ordinary input contributes no marker",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let id f = f
let check = id has
let () = ignore (Generated.read "r"); p "ordinary" (check backend_name "cuda")|ocaml},
      "+partial(callback, unvalidated)" );
    ( "an opaque function-case ordinary tuple component stays partial without a pin",
      {ocaml|let second = function src, name -> name
let name = second (Generated.read "r", backend_name)
let () = p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      "+partial(opaque)" );
    ( "an opaque function-case generated tuple component remains explicitly uncertain",
      {ocaml|let first = function src, name -> src
let src = first (Generated.read "r", backend_name)
let () = p "marker" (String.is_substring src ~substring:"opaque tuple marker")|ocaml},
      "+partial(opaque)" );
    ( "an opaque function-case ordinary record component stays partial without a pin",
      {ocaml|let second = function { src; name } -> name
let name = second { src = Generated.read "r"; name = backend_name }
let () = p "ordinary" (String.is_substring name ~substring:"cuda")|ocaml},
      "+partial(opaque)" );
    ( "a function-case internal read remains independent of opaque components",
      {ocaml|let read = function src, name -> Generated.read "internal"
let src = read (backend_name, backend_name)
let () = p "marker" (String.is_substring src ~substring:"independent case marker")|ocaml},
      {|"independent case marker"|} );
    ( "a source returned through ref mutation remains visibly uncorrelated",
      {ocaml|let store x = let r = ref "" in r := x; !r
let src = store (Generated.read "r")
let () = p "marker" (String.is_substring src ~substring:"mutation marker")|ocaml},
      "+partial(mutation)" );
    ( "ordinary input through ref mutation cannot validate a source marker",
      {ocaml|let store x = let r = ref "" in r := x; !r
let src = store backend_name
let () = ignore (Generated.read "r"); p "ordinary" (String.is_substring src ~substring:"cuda")|ocaml},
      "+partial(mutation)" );
    ( "a discarded record write keeps the source dependency explicitly uncertain",
      {ocaml|let store x = let r = { value = "" } in r.value <- x; r.value
let src = store (Generated.read "r")
let () = p "marker" (String.is_substring src ~substring:"record mutation marker")|ocaml},
      "+partial(mutation)" );
    ( "a let-bound write keeps its source dependency explicitly uncertain",
      {ocaml|let store x = let r = ref "" in let () = r := x in !r
let src = store (Generated.read "r")
let () = p "marker" (String.is_substring src ~substring:"let mutation marker")|ocaml},
      "+partial(mutation)" );
    ( "a generated ref initializer cannot validate text after an ordinary overwrite",
      {ocaml|let store x = let r = ref (Generated.read "initial") in r := x; !r
let src = store backend_name
let () = p "ordinary" (String.is_substring src ~substring:"cuda")|ocaml},
      "+partial(mutation)" );
    ( "a generated record initializer cannot validate an overwritten field",
      {ocaml|let store x = let r = { value = Generated.read "initial" } in r.value <- x; r.value
let src = store backend_name
let () = p "ordinary" (String.is_substring src ~substring:"cuda")|ocaml},
      "+partial(opaque, mutation)" );
    ( "an independent generated result survives a preceding write boundary",
      {ocaml|let store x = let r = ref "" in r := x; Generated.read "internal"
let src = store backend_name
let () = p "marker" (String.is_substring src ~substring:"independent mutation marker")|ocaml},
      {|"independent mutation marker" +partial(mutation)|} );
    ( "a helper-local generated read propagates through normalization aliases",
      {ocaml|let check ~routine ~marker =
  let src = Generated.read routine in
  let alias = String.lowercase src in
  let second = String.strip alias in
  String.is_substring second ~substring:marker
let () = p "marker" (check ~routine:"r" ~marker:"aliased marker")|ocaml},
      {|"aliased marker"|} );
    ( "same-named local predicates pin only their in-scope composite context",
      {ocaml|let kernel () =
  let has ~src ~marker = String.is_substring src ~substring:("kernel:" ^ marker) in
  has ~src:(Generated.read "r") ~marker:"actual marker"
let ordinary () =
  let has ~src ~marker = String.is_substring src ~substring:("backend:" ^ marker) in
  has ~src:backend_name ~marker:"not a pin"
let () = p "kernel" (kernel ()); p "ordinary" (ordinary ())|ocaml},
      {|"actual marker" "kernel:" ^ ... +partial(unvalidated)|} );
    ( "a nested predicate validates an enclosing source parameter at its call",
      {ocaml|let outer src =
  let inner ~marker = String.is_substring src ~substring:marker in
  inner ~marker:"captured marker"
let () = p "marker" (outer (Generated.read "r"))|ocaml},
      {|"captured marker"|} );
    ( "a nested predicate validates its captured source through an alias",
      {ocaml|let outer src =
  let inner ~marker =
    let alias = String.lowercase src in
    String.is_substring alias ~substring:marker
  in
  inner ~marker:"captured marker"
let () = p "marker" (outer (Generated.read "r"))|ocaml},
      {|"captured marker"|} );
    ( "a nested backend-name predicate does not capture an unrelated source parameter",
      {ocaml|let outer src =
  let inner ~marker = String.is_substring backend_name ~substring:marker in
  describe src;
  inner ~marker:"cuda"
let () = p "ordinary" (outer (Generated.read "r"))|ocaml},
      "" );
    ( "nested helpers own their tests independently of enclosing predicates",
      {ocaml|let outer ~routine =
  let src = Generated.read routine in
  let inner ~inner_marker = String.is_substring src ~substring:inner_marker in
  inner ~inner_marker:"inner marker"
let () = p "marker" (outer ~routine:"r")|ocaml},
      {|"inner marker"|} );
    ( "validated source aliases propagate into nested predicate calls",
      {ocaml|let inner code = String.is_substring code ~substring:"inner marker"
let outer input =
  let alias = input in
  String.is_substring input ~substring:"outer marker" && inner alias
let () = p "markers" (outer (Generated.read "r"))|ocaml},
      {|"inner marker" "outer marker"|} );
    ( "an ordinary direct call does not hide an unresolved generated-source callback",
      {ocaml|let barrier src = String.is_substring src ~substring:"barrier marker"
let () =
  p "ordinary" (barrier backend_name);
  List.iter [Generated.read "r"] ~f:barrier|ocaml},
      "+partial(callback, unvalidated)" );
    ( "unfollowed callbacks keep hard-coded predicate text visibly partial",
      {ocaml|let barrier src = String.is_substring src ~substring:"barrier marker"
let () = List.iter [Generated.read "r"] ~f:barrier|ocaml},
      "+partial(callback)" );
    ( "validated source parameters propagate through nested predicate calls",
      {ocaml|let inner code = String.is_substring code ~substring:"inner marker"
let outer input = String.is_substring input ~substring:"outer marker" && inner input
let () = p "markers" (outer (Generated.read "r"))|ocaml},
      {|"inner marker" "outer marker"|} );
    ( "marker-first partial applications remain visibly partial",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let check = has ~marker:"actual marker"
let () = p "marker" (check (Generated.read "r"))|ocaml},
      "+partial(unsupplied)" );
    ( "explicit optional absence selects a generated-source default",
      {ocaml|let has ?(src = Generated.read "r") ~marker () = String.is_substring src ~substring:marker
let () = p "marker" (has ?src:None ~marker:"actual marker" ())|ocaml},
      {|"actual marker"|} );
    ( "unresolved optional source forwarding remains visibly partial",
      {ocaml|let absent = None
let has ?(src = Generated.read "r") ~marker () = String.is_substring src ~substring:marker
let () = p "marker" (has ?src:absent ~marker:"actual marker" ())|ocaml},
      {|"actual marker" +partial(unsupplied)|} );
    ( "explicit optional presence unwraps the supplied generated source",
      {ocaml|let has ?(src = backend_name) ~marker () = String.is_substring src ~substring:marker
let () = p "marker" (has ?src:(Some (Generated.read "r")) ~marker:"actual marker" ())|ocaml},
      {|"actual marker"|} );
    ( "hard-coded predicate fragments require their own call-site source",
      {ocaml|let check src1 src2 ~marker =
  String.is_substring src1 ~substring:marker
  && String.is_substring src2 ~substring:"backend-only literal"
let () = p "marker" (check (Generated.read "r") backend_name ~marker:"kernel marker")|ocaml},
      {|"kernel marker" +partial(unvalidated)|} );
    ( "partial applications do not select optional defaults prematurely",
      {ocaml|let has src ?(marker = "unused default") () = String.is_substring src ~substring:marker
let check = has (Generated.read "r")
let () = p "marker" (check ~marker:"actual marker" ())|ocaml},
      "+partial(unsupplied, unvalidated)" );
    ( "a composite marker retains every caller-supplied parameter",
      {ocaml|let has src ~prefix ~suffix = String.is_substring src ~substring:(prefix ^ ":" ^ suffix)
let () = p "marker" (has (Generated.read "r") ~prefix:"first" ~suffix:"second")|ocaml},
      {|"first" "second" ... ^ ":" ^ ...|} );
    ( "a later generated binding does not taint an earlier source parameter",
      {ocaml|let has src ~marker =
  let earlier = String.is_substring src ~substring:marker in
  let src = Generated.read "r" in
  describe src;
  earlier
let () = p "ordinary" (has backend_name ~marker:"cuda")|ocaml},
      "+partial(unvalidated)" );
    ( "a labelled marker inside an expression keeps its literal context",
      {ocaml|let symbol ~emitted src = String.is_substring src ~substring:("void " ^ emitted ^ "(")
let () = p "symbol" (symbol ~emitted:"asm__" (Generated.read "r"))|ocaml},
      {|"asm__" "void " ^ ... ^ "("|} );
    ( "a labelled text parameter shadowed by a lambda pattern stays partial",
      {ocaml|let check ~marker src =
  Option.iter (current_marker ()) ~f:(fun (marker, other) ->
    String.is_substring src ~substring:marker)
let () = p "marker" (check ~marker:"caller marker" (Generated.read "r"))|ocaml},
      "+partial(callback)" );
    ( "a local source alias is not generated because another scope uses its name",
      {ocaml|let other () = let src = Generated.read "r" in describe src
let has ~s ~sub =
  let src = String.lowercase s in
  String.is_substring src ~substring:sub
let () = p "ordinary" (has ~s:backend_name ~sub:"cuda")|ocaml},
      "+partial(unvalidated)" );
    ( "labelled predicates over ordinary text pin no generated fragments",
      {ocaml|let has ~src ~sub = String.is_substring src ~substring:sub
let () =
  let generated = Generated.read "r" in
  p "ordinary" (has ~src:backend_name ~sub:"not a pin")|ocaml},
      "+partial(unvalidated)" );
    ( "a binding shadowing a literal let is not that literal",
      {ocaml|let marker = "/* stale marker */"
let () =
  let src = Generated.read "r" in
  let marker = current_marker () in
  p "marker" (String.is_substring src ~substring:marker)|ocaml},
      "+partial(computed)" );
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
    ( "a compiler-plan annotation does not hide an exported emitter wrapper",
      {ocaml|module CR = Ir.Low_level.Canonical_render
let write ~buf policy llc = CR.emit ~buf policy llc
[@@ocannl.codegen_text.compiler_plan]
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
      "+partial(mutation) +rendered" );
    ( "a buffer read through an alias of Buffer marks it partial just the same",
      {ocaml|module LL = Ir.Low_level
module B = Buffer
let () =
  let buf = B.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  p "radix" (String.is_substring (B.contents buf) ~substring:"-0.0")|ocaml},
      "+partial(mutation) +rendered" );
    ( "an unattributed buffer read travels along bindings like taint does",
      {ocaml|module LL = Ir.Low_level
let () =
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  let source = Buffer.contents buf in
  p "radix" (String.is_substring source ~substring:"-0.0")|ocaml},
      "+partial(mutation) +rendered" );
    ( "an unattributed buffer reaching a helper marks it partial too",
      {ocaml|module LL = Ir.Low_level
let has src sub = String.is_substring src ~substring:sub
let () =
  let buf = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  let source = Buffer.contents buf in
  p "radix" (has source "-0.0")|ocaml},
      "+partial(mutation) +rendered" );
    ( "a buffer aliased under a signature constraint is still a buffer",
      {ocaml|module LL = Ir.Low_level
module B = (Buffer : module type of Buffer)
let () =
  let buf = B.create 256 in
  PPrint.ToBuffer.pretty 0.9 100 buf (LL.to_doc () llc);
  p "radix" (String.is_substring (B.contents buf) ~substring:"-0.0")|ocaml},
      "+partial(mutation) +rendered" );
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

(** One source per partial-itemisation boundary (gh-ocannl-1210), each built to meet that boundary
    alone: the comparison is exact, so a case reporting its own category beside another one fails as
    surely as one reporting the wrong category. Above, the controls inherited from the provenance
    model pin the category each existing shape earns; these state the categories themselves, and
    every constructor of {!Scan.boundary} must have one -- quantified over its derived enumeration,
    so a constructor added without a case fails the claim. Each case is the boundary, a name, the
    source, and the fragments it still names. *)
let boundary_cases =
  [
    ( Scan.Computed_fragment,
      "a fragment computed from ordinary values has no spelling to name",
      {ocaml|let () =
  let src = Generated.read "r" in
  p "m" (String.is_substring src ~substring:(marker_for backend))|ocaml},
      "" );
    ( Scan.Untraced_callback,
      "a source-taking helper handed to a combinator is an untraced callback",
      {ocaml|let check src = String.is_substring src ~substring:"callback-only marker"
let () = ignore (Generated.read "r"); p_all "m" sources ~f:check|ocaml},
      "" );
    ( Scan.Forwarded_parameter,
      "a fragment arriving as a wrapper's parameter is forwarded",
      {ocaml|let has src marker = String.is_substring src ~substring:marker
let expect marker = p marker (has (Generated.read "r") marker)
let () = expect "wrapper marker"|ocaml},
      "" );
    ( Scan.Opaque_component,
      "generated text read back out of a record field is an opaque component",
      {ocaml|let compile () = { source = Generated.read "r"; ok = true }
let () =
  let out = compile () in
  p "m" (String.is_substring out.source ~substring:"field marker")|ocaml},
      "" );
    ( Scan.Unsupplied_application,
      "a partial application leaves the source it depends on unsupplied",
      {ocaml|let choose flag src = if flag then src else ""
let pick = choose true
let () = p "m" (String.is_substring (pick (Generated.read "r")) ~substring:"chosen marker")|ocaml},
      {|"chosen marker"|} );
    ( Scan.Mutation,
      "a haystack read back out of an array cell is a mutation",
      {ocaml|let has src = String.is_substring src ~substring:"cell marker"
let cache = Array.create ~len:1 ""
let () = cache.(0) <- Generated.read "r"; p "m" (has cache.(0))|ocaml},
      "" );
    ( Scan.Replaced_binding,
      "a source parameter rebound to other text",
      {ocaml|let has src ~marker =
  let src = default_source () in
  String.is_substring src ~substring:marker
let () = p "m" (has (Generated.read "r") ~marker:"replaced marker")|ocaml},
      "" );
    ( Scan.Unvalidated_haystack,
      "a source-taking helper handed ordinary text is unvalidated",
      {ocaml|let has src ~marker = String.is_substring src ~substring:marker
let () = ignore (Generated.read "r"); p "m" (has device_name ~marker:"plain marker")|ocaml},
      "" );
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

(** How {!Attempt.run} tells the input's fault from the scanner's: the parser's syntax and lexical
    errors are the file's, anything else -- here {!Scan.boundaries_of}'s precondition, broken by a
    provenance no source can produce -- is reported as itself. *)
let attempt_cases =
  let parse source () = ignore (Scan.rejections ~emitters ~path:"case.ml" ~contents:source) in
  [
    ("a syntax error is the file's", parse "let =\n", "unparsed");
    ("a lexical error is the file's", parse "let s = \"unterminated\n", "unparsed");
    ("a source that parses is scanned", parse "let x = 1\n", "scanned");
    ( "a scanner exception is the scanner's, with its text",
      (fun () ->
        ignore (Scan.boundaries_of { Scan.no_provenance with Scan.uncertainty = Scan.Unresolved })),
      "raised Invalid_argument(\"Codegen_text_scan: an uncertain provenance names no boundary\")" );
  ]

let render_attempt = function
  | Attempt.Scanned () -> "scanned"
  | Attempt.Unparsed -> "unparsed"
  | Attempt.Raised exn -> "raised " ^ exn

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
        Attempt.attempted ~what:"source" ~name ~default:"<unscanned>" (fun () ->
            render_site (Scan.classify_source ~emitters ~path:"case.ml" ~contents:source))
      in
      if String.equal (String.strip found) (String.strip expected) then
        printf "ok: source -- %s\n" name
      else fail "source -- %s: expected [%s], found [%s]" name expected found);
  List.iter boundary_cases ~f:(fun (boundary, name, source, pins) ->
      let expected = String.strip (pins ^ " +partial(" ^ Scan.boundary_tag boundary ^ ")") in
      let found =
        Attempt.attempted ~what:"boundary" ~name ~default:"<unscanned>" (fun () ->
            render_site (Scan.classify_source ~emitters ~path:"case.ml" ~contents:source))
      in
      if String.equal (String.strip found) expected then printf "ok: boundary -- %s\n" name
      else fail "boundary -- %s: expected [%s], found [%s]" name expected found);
  Verdict.p_all "every partial-itemisation boundary has a case meeting it alone"
    Scan.all_of_boundary ~f:(fun boundary ->
      List.exists boundary_cases ~f:(fun (b, _, _, _) -> Poly.equal b boundary));
  List.iter attempt_cases ~f:(fun (name, f, expected) ->
      let found = render_attempt (Attempt.run f) in
      if String.equal found expected then printf "ok: attempt -- %s\n" name
      else fail "attempt -- %s: expected [%s], found [%s]" name expected found);
  List.iter rejection_cases ~f:(fun (name, source, expected) ->
      let found =
        Attempt.attempted ~what:"rejection" ~name ~default:(-1) (fun () ->
            List.length (Scan.rejections ~emitters ~path:"case.ml" ~contents:source))
      in
      if expected = found then printf "ok: rejection -- %s\n" name
      else fail "rejection -- %s: expected %d refusals, found %d" name expected found);
  Test_utils.Refusal_control_manifest.print "codegen_text_inventory.ml"
