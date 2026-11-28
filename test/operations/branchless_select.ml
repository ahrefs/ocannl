(* Value selects render branchless on cc, guards never do (gh-ocannl-1622).

   A [Where] whose arm holds a load, spelled as C's short-circuiting [?:], compiles to a compare and
   a conditional branch: C forbids evaluating the unselected arm, and the compiler cannot prove the
   load dereferenceable. Inlined into the nests that consume them, the ReLU and max-pool backward
   selects made cifar_resnet's step time follow its activation sparsity (gh-ocannl-1557). The cc
   backend now renders a value select it may speculate ([Low_level.speculable_select]) as a GNU
   statement expression that evaluates both arms into [__auto_type] locals and then selects, so the
   compiler if-converts; the locals keep each arm's C type, so the blend has the [?:] spelling's
   exact type too. The same [Where] is also how lowering guards a read that may be out of range, and
   those must keep short-circuiting.

   Pinned here:

   - Values, on every backend, bitwise against an OCaml reference of the short-circuiting semantics:
   an unselected NaN or infinity does not reach the result (a mask multiply would leak it), a
   selected negative zero stays negative, a NaN condition selects the then-arm of a [Where] and
   gates a [Relu_gate] shut, and [Satur01_gate] passes its operand only for inputs in (0, 1). Legs
   at f32, f64, a half-precision condition beside an f32 result, and bf16 storage computing at f32;
   and, on cc, a gate and a [Where] composed into further arithmetic, whose rounding follows the
   [?:] spelling's C type (a gate's [0.0f] literal and a [Where] of floats both stay float). -
   Emission, on cc: those legs call the blend and keep no [?:] spelling of their select. Count the
   condition-buffer reads in the generated statements for the two [Satur01_gate] fixtures. The count
   covers the direct and volatile-cast subscript spellings of the condition buffer. Guards, on cc:
   an index-guarded out-of-range read and a value-guarded [Get_dynamic] gather (its index runs far
   out of range on the inactive lanes) keep [?:]; a value select over an index guard blends outside
   and keeps the inner guard. Executed with the hazardous reads inactive. - Assembly, on cc's own
   compiler and kernel flags: no conditional branch is attributed to a select's source lines, and
   those lines carry instructions, the realistic leg's included (gh-ocannl-1660); on CUDA
   (gh-ocannl-1695), the same claim of the code object the run compiled (SASS through [ptxas] and
   [nvdisasm]). HIP keeps loaded arms conditional and blends the register-only gate
   (gh-ocannl-2427), whose AMDGPU code object is read through [llvm-objdump]. On Metal
   (gh-ocannl-1828, gh-ocannl-2400), the AIR the Metal Toolchain compiles the run's [.metal]
   artifact to under the run's own compile options, which keeps the short-circuit's branch on
   parallel loaded-arm legs (pinned as counted on the toolchain version counted, the zero-branch
   claim skipping there), compiles the parallel register-only gate branch-free (0 conditional
   branches and a [select] instruction), and blends single-threaded serial-loop selects branch-free
   under [Blend_serial_loop]. - The realistic composition: a ReLU feeding a 2x2 max pool, backward
   through the default placements (the selects inlined into their consumers) — the gradient emits
   the blend and equals, bitwise, both an OCaml reference and the same program with inlining off. *)

(* cc-fp-contract: exact-operands -- selects, adds and a pool gradient multiplying dyadic weights by
   one *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Generated = Test_utils.Generated
module Asm = Test_utils.Asm_census
module Cc_backend = Context.Cc_backend

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* The C backends are the ones whose syntax config is cc's. *)
let on_cc = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)
let skipped = Verdict.skipped ~backend:backend_name

(* The GPU code dialects this census can read (gh-ocannl-1695): the code object a CUDA or HIP run
   compiled, and on Metal (gh-ocannl-1828) the AIR the Metal Toolchain compiles the run's [.metal]
   artifact to -- a re-compile, as [Asm.disassemble_gpu] states, skipped where the toolchain is
   missing. CUDA's target boundary is in [census_claim]; select-emission policy belongs to
   [Cuda_like_config.Cuda]. *)
let gpu_dialect =
  match backend_name with
  | "cuda" -> Some Asm.Sass
  | "hip" -> Some Asm.Amdgpu
  | "metal" -> Some Asm.Air
  | _ -> None

(* The backend whose census reads AIR, a re-compile of the run's source (gh-ocannl-1828). *)
let on_air = String.equal backend_name "metal"

(* The Metal backend's own host probes ([Metal_backend.host_capabilities]), which the host record
   every test depends on carries, so installing or updating the toolchain reruns this test. *)
let metal_host_capabilities = lazy (Context.host_capabilities ())

let metal_host_capability key =
  List.Assoc.find (Lazy.force metal_host_capabilities) ~equal:String.equal key

(* The compile options the Metal run applied: the math API the runtime accepts, as the backend
   reports it, through the builder the backend's own compiles call ([Compiler_options.metal_of_run],
   which owns the other input, routine logging). [None] where the report names no API this build
   knows, which the census reads as an unreadable artifact, a failure, never a guess. *)
let metal_options =
  lazy
    (Option.map
       (Option.bind
          (metal_host_capability "metal_math_api")
          ~f:Ir.Compiler_options.metal_math_api_of_name)
       ~f:(fun math_api -> Ir.Compiler_options.metal_of_run ~math_api))

let () = Generated.init ~backend_name

(* Derived from the backend's [C_syntax_config.select_blend_syntax] via
   {!Ir.Backend_intf.codegen_capabilities.select_blend} (gh-ocannl-2386). *)
let caps = lazy (Context.codegen_capabilities (Context.auto ()))

let select_blend prec =
  let c = Lazy.force caps in
  c.Ir.Backend_intf.select_blend (c.compute_prec prec)

let blends_at prec = Option.is_some (select_blend prec)
let blends = blends_at single
let facts = Context.backend_facts_of_name backend_name
let blends_loaded_arms = Ir.Backend_intf.charges_select_arm_union facts
let on_hip = String.equal backend_name "hip"
let hip_leg = Verdict.leg ~when_:on_hip ~on:backend_name ()

(* The loop the census legs' select runs under. On a GPU dialect, a [Grid] loop: one select per
   thread, the shape a parallel backward kernel has (a serial loop's unrolled copies are the other
   shape, which the realistic pool leg keeps). Serial elsewhere. *)
let census_axis = if Option.is_some gpu_dialect then LL.Grid else LL.Serial

(* The kernel-text claims each leg makes where the backend blends, skipped where [select_blend]
   answers [None] at the leg's precision (and [on_cc] restricts [cpu_only] legs, whose expected
   values are C-typed): written once (gh-ocannl-1703, gh-ocannl-2386). *)
let blend_leg ~cpu_only ~prec =
  Verdict.leg
    ~when_:(((not cpu_only) || on_cc) && blends_loaded_arms && blends_at prec)
    ~on:backend_name ()

let single_blend_leg = Verdict.leg ~when_:blends ~on:backend_name ()

let grid_register_blend_leg =
  Verdict.leg ~when_:(blends && (blends_loaded_arms || on_hip)) ~on:backend_name ()

let single_loaded_blend_leg = blend_leg ~cpu_only:false ~prec:single

let single_serial_blend_leg =
  Verdict.leg
    ~when_:(blends && Ir.Backend_intf.blends_serial_loop_selects facts)
    ~on:backend_name ()

let over_locals_leg =
  Verdict.leg
    ~when_:(match select_blend single with Some `Blend_over_locals -> true | _ -> false)
    ~on:backend_name ()

(* gh-ocannl-1778: the [outlined] mode is this test run with every kernel forced outlined
   ([cc_kernel_part_weight=1], [cc_compile_unit_min_bytes=0] on its rule), under the plain run's
   golden. It claims the pool kernel -- the one large enough to outline -- was rendered outlined, so
   a threshold change leaving it whole cannot turn the variant into a second copy of the plain
   run. *)
let outlined =
  match
    List.filter
      (List.tl_exn (Array.to_list Stdlib.Sys.argv))
      ~f:(fun arg -> not (String.is_prefix arg ~prefix:"--"))
  with
  | [] -> false
  | [ "outlined" ] -> true
  | _ -> failwith "usage: branchless_select.exe [outlined] [--ocannl_<key>=<value>...]"

(* NaN against NaN by kind (the payload is the hardware's), everything else on the bits: a signed
   zero is half of what is pinned. *)
let same_bits a b =
  (Float.is_nan a && Float.is_nan b) || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let nan = Float.nan
let inf = Float.infinity
let n = 8

(* Per precision, a factory of [n]-cell nodes with an id range of its own. *)
let mk_single = node_factory ~first_id:16220 ~dims:[| n |] ()
let mk_double = node_factory ~prec:Ops.double ~first_id:16260 ~dims:[| n |] ()
let mk_half = node_factory ~prec:Ops.half ~first_id:16300 ~dims:[| n |] ()
let mk_bf16 = node_factory ~prec:Ops.bfloat16 ~first_id:16340 ~dims:[| n |] ()
let mk_int32 = node_factory ~prec:Ops.int32 ~first_id:16380 ~dims:[| n |] ()

(* The kernel's statements ({!Generated.main_logic}): not the builtins prelude, which spells [?:]
   itself (the integer-power helpers, the select helpers), and, where the kernel was rendered
   outlined, its parts' bodies spliced in at their calls. The text after the kernel's name read only
   the kernel's own body, which an outlined kernel spends calling its parts: its positive claims
   went red and its negative ones held over nothing (gh-ocannl-1778). *)
let kernel_text name = Generated.main_logic (Generated.read name)
let has text s = String.is_substring text ~substring:s

(* Every short-circuiting spelling a value select has on cc ([Ops.ternop_c_syntax],
   [Ops.binop_c_syntax], float arms). *)
let short_circuit_forms = [ ") != 0.0 ? ("; " > 0.0 ?"; "; }) ?" ]

(* The blend's own spelling, every backend's ([C_syntax.blend_local_a]): cc's [__auto_type] and the
   GPU dialects' [auto] local of the first arm. *)
let blend_marker = Ir.C_syntax.blend_local_a ^ " ="

(* The claim that [text] keeps no short-circuiting spelling of a value select. Under [`Blend_truth]
   (cc), none of [short_circuit_forms]. Under [`Blend_over_locals] (GPU dialects), the blend keeps
   the dialect's own [?:] over its locals ([C_syntax.Blend_over_locals]), so what must not remain is
   a [?] selecting anything but a local: every [?] -- of which a blended kernel has at least its
   blend's own -- is followed by [C_syntax.blend_local_a]. *)
let no_short_circuit_claim ~prec label text =
  match select_blend prec with
  | Some `Blend_truth -> p_none label short_circuit_forms ~f:(has text)
  | Some `Blend_over_locals ->
      let after_each_question_mark =
        List.filter_mapi (String.to_list text) ~f:(fun i ch ->
            Option.some_if (Char.equal ch '?') (String.lstrip (String.drop_prefix text (i + 1))))
      in
      p_all label after_each_question_mark ~f:(String.is_prefix ~prefix:Ir.C_syntax.blend_local_a)
  | None -> p label false

(* The guards' short-circuiting spelling: cc's [Ops.ternop_c_syntax] opens the arm with a
   parenthesis, the GPU dialects' does not. *)
let guard_spelling = if on_cc then "? (" else " ? "

(* Executes [llc] writing [out], every node materialized and the inputs seeded. *)
let execute_leg ~name ~seed ~out llc =
  List.iter (out :: List.map seed ~f:fst) ~f:materialize;
  let _o, got = optimize_and_execute ~name llc ~seed ~read:[ out ] in
  List.hd_exn got

(* The kernel statement lines a census of [source]'s selects reads: each line matching [patterns],
   and every line of each blend's statement expression, from its [({] to its [})] -- a truth
   expression too long for the selection's line wraps onto the line above it, where the max-pool
   gate's comparison branched unseen by a census of the matching lines alone (gh-ocannl-1660).
   Statement lines only ({!Generated.main_logic_lines}): the prelude's [OCANNL_SELECT] definition
   spells the names too. *)
let select_lines ~patterns source =
  let lines = Array.of_list (String.split_lines source) in
  let blocks =
    Generated.brace_intervals lines
    |> List.filter ~f:(fun (opening, _) -> has lines.(opening) blend_marker)
    |> List.concat_map ~f:(fun (opening, closing) -> List.range (opening + 1) (closing + 2))
  in
  Set.inter
    (Set.of_list (module Int) (Generated.main_logic_lines source))
    (Set.union (Asm.anchor_lines ~source ~patterns) (Set.of_list (module Int) blocks))

(* Whether [name]'s select compiled branch-free: compiled by the cc backend's own compiler with its
   own kernel flags (at [-O3], its default level), no conditional branch is attributed to a line
   [select_lines] reads, and those lines carry instructions (an anchor naming nothing, or a select
   folded away, fails). A loop's control that DWARF puts on such a line, an integer counter compared
   with a constant bound ([Asm.is_counter_test]), is set apart: gcc 13 attributes a peeled outer
   loop's test to the max-pool gate's line, where gcc 15 attributes it elsewhere. Each anchored
   line's own count goes to stderr, so a red run names the line that branched, and a red run also
   prints each branch with the instruction before it ([Asm.explain_branch_claim]). *)
let compiled_branch_free ~patterns name =
  let src_path = Generated.path name in
  let source = Generated.read name in
  let asm_path = Stdlib.Filename.temp_file ("bs_" ^ name ^ "_") ".s" in
  let host =
    {
      Asm.label = "host";
      command = Cc_backend.compiler_command () ^ " " ^ Cc_backend.kernel_compiler_flags ();
      march = "";
      note = "the cc backend's compiler";
    }
  in
  match Asm.compile host ~opt_level:3 ~src_path ~asm_path with
  | Error out ->
      Stdio.eprintf "branchless_select: %s did not compile for the census: %s\n" name out;
      false
  | Ok () ->
      let asm = Stdio.In_channel.read_all asm_path in
      (try Stdlib.Sys.remove asm_path with _ -> ());
      let parsed = Asm.parse ~asm ~source_basename:(Stdlib.Filename.basename src_path) in
      let anchor = select_lines ~patterns source in
      let lines = Array.of_list (String.split_lines source) in
      Set.iter anchor ~f:(fun l ->
          let b, c, i = Asm.conditional_branches_in parsed ~anchor:(Set.singleton (module Int) l) in
          Stdio.eprintf
            "branchless_select: %s line %d: %d branch(es), %d loop-counter test(s), in %d \
             instruction(s): %s\n"
            name l b c i
            (String.strip lines.(l - 1)));
      let branches, counters, instructions = Asm.conditional_branches_in parsed ~anchor in
      Stdio.eprintf
        "branchless_select: %s: %d conditional branch(es), %d loop-counter test(s) set apart, in \
         %d instruction(s) on %d select line(s) (not part of the golden)\n"
        name branches counters instructions (Set.length anchor);
      (* On a red, each branch with the instruction that set its flags, so the run itself tells a
         select's branch from a loop control DWARF put on its line. *)
      let sites, _ = Asm.conditional_branch_sites parsed ~anchor in
      Asm.explain_branch_claim ~source ~prefix:("branchless_select: " ^ name)
        ~holds:(branches = 0 && instructions > 0)
        sites

(* The executing CUDA device's architecture, [sm_<major><minor>], from the backend's own device
   query ({!Context.static_properties}: ordinal 0, the device [Context.auto] runs on, after the
   driver's [CUDA_VISIBLE_DEVICES] remapping), so the census assembles what the driver's JIT does
   for that device and not another GPU's. *)
let cuda_arch =
  lazy
    (if not (String.equal backend_name "cuda") then None
     else Option.map (List.hd (Ll_test.cuda_compute_capabilities ())) ~f:(Printf.sprintf "sm_%d"))

(* The Metal Toolchain an AIR census compiles with, its version line (or why there is none) on
   stderr once: the reading's provenance, and the version an exact count is pinned to. *)
let metal_toolchain =
  lazy
    (let t =
       match metal_host_capability "metal_toolchain" with
       | Some line when Option.is_some (Asm.metal_toolchain_version line) -> Ok line
       | reported ->
           Error
             (Printf.sprintf "no Metal Toolchain (%s): the backend reports %s" Asm.metal_compiler
                (Option.value reported ~default:"no metal_toolchain capability"))
     in
     (match (on_air, t) with
     | true, Ok version ->
         Stdio.eprintf "branchless_select: AIR census toolchain: %s (not part of the golden)\n"
           version
     | true, Error why ->
         Stdio.eprintf "branchless_select: no AIR census: %s (not part of the golden)\n" why
     | false, _ -> ());
     t)

(* What a GPU census read of one artifact: whether it could be read at all -- the tool ran, its line
   tables name the artifact's source, the anchor is nonempty and its lines carry instructions -- and
   how many conditional branches it attributed to the anchored lines. *)
type gpu_reading = { readable : bool; branches : int }

let branch_free r = r.readable && r.branches = 0

(* The GPU counterpart of [compiled_branch_free] (gh-ocannl-1695): the code object the run itself
   compiled for [name] -- [<name>.ptx] on CUDA, [<name>.hsaco] on HIP, or each [<name>.part<k>] a
   split compile wrote -- disassembled with its line tables ([Asm.disassemble_gpu]; on Metal, the
   run's [<name>.metal] re-compiled to AIR under the run's options, gh-ocannl-1828), and the
   conditional branches attributed to a line [select_lines] reads in the source that artifact was
   compiled from. One reading per artifact, a branching one printing its sites; no artifact at all
   is an empty list, which a claim's [p_all] fails, and a missing tool is an unreadable reading: on
   CUDA and HIP the census is the claim, so neither skips. *)
(* Shared GPU artifact enumeration and disassembly plumbing for both line-anchored [gpu_census] and
   whole-function [air_function_branch_free_select]. *)
let for_each_gpu_artifact dialect name ~on_error ~on_parsed =
  let artifact_ext, source_ext =
    match dialect with
    | Asm.Sass -> (".ptx", ".cu")
    | Asm.Amdgpu -> (".hsaco", ".hip")
    | Asm.Air -> (".metal", ".metal")
  in
  (* The whole program's artifact, or each part a split compile wrote ([<name>.part<k>], from 0). *)
  let exists suffix = Stdlib.Sys.file_exists (Generated.path ~ext:(suffix ^ artifact_ext) name) in
  let rec parts k =
    let suffix = Printf.sprintf ".part%d" k in
    if exists suffix then suffix :: parts (k + 1) else []
  in
  let suffixes = (if exists "" then [ "" ] else []) @ parts 0 in
  let say_line m = Stdio.eprintf "branchless_select: %s: %s (not part of the golden)\n" name m in
  if List.is_empty suffixes then say_line ("no " ^ artifact_ext ^ " artifact");
  List.map suffixes ~f:(fun suffix ->
      let stem = name ^ suffix in
      let artifact = Generated.path ~ext:(suffix ^ artifact_ext) name in
      let ext = suffix ^ source_ext in
      let source_basename = Stdlib.Filename.basename (Generated.path ~ext name) in
      let source = Generated.read ~ext name in
      match
        Asm.disassemble_gpu ?cuda_arch:(Lazy.force cuda_arch)
          ?metal_options:(Lazy.force metal_options)
          ?metal_toolchain:(Result.ok (Lazy.force metal_toolchain))
          dialect ~artifact
      with
      | Error why -> on_error ~say_line ~stem ~why
      | Ok (dump, toolchain) ->
          let parsed = Asm.parse_gpu dialect ~dump ~source_basename in
          on_parsed ~say_line ~stem ~source_basename ~source ~toolchain parsed)

let gpu_census dialect ~patterns name =
  for_each_gpu_artifact dialect name
    ~on_error:(fun ~say_line ~stem ~why ->
      say_line (Printf.sprintf "%s: the census cannot read it: %s" stem why);
      { readable = false; branches = 0 })
    ~on_parsed:(fun ~say_line ~stem ~source_basename ~source ~toolchain parsed ->
      let say fmt = Printf.ksprintf say_line fmt in
      let anchor = select_lines ~patterns source in
      let lines = Array.of_list (String.split_lines source) in
      Set.iter anchor ~f:(fun l ->
          let sites, i = Asm.gpu_branch_sites parsed ~anchor:(Set.singleton (module Int) l) in
          say "%s line %d: %d branch(es) in %d instruction(s): %s" stem l (List.length sites) i
            (String.strip lines.(l - 1)));
      let sites, instructions = Asm.gpu_branch_sites parsed ~anchor in
      say
        "%s: %d conditional branch(es) in %d instruction(s) on %d select line(s), %d %s line \
         marker(s) for %s, by %s"
        stem (List.length sites) instructions (Set.length anchor) parsed.g_markers
        (Asm.gpu_dialect_name dialect) source_basename toolchain;
      let reading =
        {
          readable = instructions > 0 && parsed.g_markers > 0 && not (Set.is_empty anchor);
          branches = List.length sites;
        }
      in
      (* Every branch with the instruction before it, so the run names what branched. *)
      ignore
        (Asm.explain_branch_claim ~source ~prefix:("branchless_select: " ^ stem)
           ~holds:(branch_free reading) sites
          : bool);
      reading)

(* Function-wide AIR check for a single-select kernel (gh-ocannl-2400): unlike [gpu_census], which
   scopes branches to lines matching [patterns] via [!dbg], this inspects every instruction in the
   compiled AIR function (with or without debug locations) and verifies both that zero conditional
   branches (excluding [!llvm.loop] back-edges when [allow_loop_back_edges] is true) appear anywhere
   in the function and that an AIR [select] instruction is emitted. *)
let air_function_branch_free_select ?(patterns = [ " ? " ]) ?(allow_loop_back_edges = false) name =
  for_each_gpu_artifact Asm.Air name
    ~on_error:(fun ~say_line ~stem ~why ->
      say_line (Printf.sprintf "%s: the AIR function census cannot read it: %s" stem why);
      false)
    ~on_parsed:(fun ~say_line ~stem ~source_basename ~source ~toolchain parsed ->
      let say fmt = Printf.ksprintf say_line fmt in
      let anchor = select_lines ~patterns source in
      let conds =
        List.filter parsed.g_lines ~f:(fun (_, g) ->
            g.g_cond
            && not (allow_loop_back_edges && String.is_substring g.g_text ~substring:"!llvm.loop"))
      in
      let selects =
        List.count parsed.g_lines ~f:(fun (_, g) -> String.equal g.g_mnemonic "select")
      in
      say
        "%s: whole-function AIR: %d non-loop conditional branch(es), %d select(s) in %d \
         instruction(s), %d line marker(s) for %s, by %s"
        stem (List.length conds) selects (List.length parsed.g_lines) parsed.g_markers
        source_basename toolchain;
      List.iter conds ~f:(fun (asm_line, g) ->
          say "%s: AIR conditional branch at line %d: %s" stem asm_line g.g_text);
      parsed.g_markers > 0 && (not (Set.is_empty anchor)) && List.is_empty conds && selects > 0)

(* Metal's boundary (gh-ocannl-1828, settled in gh-ocannl-2400): Metal uses [Blend_serial_loop],
   blending speculable selects inside single-threaded serial-loop kernels (where keeping [?:] blocks
   loop pipelining and costs even at 0% active) and keeping short-circuiting [?:] in parallel GPU
   kernels (where keeping [?:] outperforms eager blending on sparse, uniform-false and causal-mask
   loaded-arm inputs on M4 Max, while AGX machine-code lowering itself is unobserved without a
   disassembler). In AIR (Metal Toolchain 32023.922, mac-studio M4 Max), a parallel [Grid] select
   whose arm loads device memory keeps [1 br i1] on each of the four hand-built legs, while a
   parallel load-free arm ([bs_register_gate]) already compiles to an AIR function with [0]
   conditional branches and a [select] instruction, and a single-threaded serial-loop loaded arm
   ([bs_serial_loop_select]) blends to [0] conditional branches and a [select] instruction. The
   realistic pool kernel's [?:] lines, whose anchor (as on CUDA) also holds the window loops' index
   guards, carry one branch on the guards' line.

   So the zero-branch claim over parallel loaded-arm selects skips on Metal citing [air_boundary],
   the loaded-arm AIR counts stay pinned on the Metal Toolchain versions they were counted on (an
   environment skip on any other), and [bs_register_gate], [bs_serial_loop_select], and the
   [bs_where_zero] negative control check the whole AIR function (including instructions without
   debug locations) whenever the Metal Toolchain is present. *)
let air_boundary = "gh-ocannl-2400"
let air_branches = 1
let air_pinned_toolchains = [ "32023.922" ]
let air_census_runs () = on_air && Result.is_ok (Lazy.force metal_toolchain)
let air_leg = Verdict.leg ~when_:on_air ~on:backend_name ()

let air_toolchain_leg =
  Verdict.sub_leg ~aggregation:`Environment air_leg
    ~when_:(lazy (air_census_runs ()))
    ~on:"a host without the Metal Toolchain (xcrun -sdk macosx metal)"

let air_pinned_leg =
  Verdict.sub_leg ~aggregation:`Environment air_toolchain_leg
    ~when_:
      (lazy
        (match Lazy.force metal_toolchain with
        | Ok line ->
            Option.value_map
              (Asm.metal_toolchain_version line)
              ~default:false
              ~f:(List.mem air_pinned_toolchains ~equal:String.equal)
        | Error _ -> false))
    ~on:"a Metal Toolchain the AIR counts were not pinned on"

(* Read emitted select source lines in the backend compiler's assembly. CUDA's claim covers the
   floor PTX target policy only, including the pool kernel (gh-ocannl-1822). On Metal, the AIR
   census's pin follows the zero-branch claim's row. *)
let census_claim ~patterns ~gpu_patterns ?(air_expected = Some air_branches) ~air_claim ~air_skip
    claim name =
  let claim = claim ^ " (CUDA: floor PTX target only)" in
  let native_cuda =
    String.equal backend_name "cuda"
    && Ir.Cuda_mma.equal_ptx_target (Ir.Cuda_mma.current_ptx_target ()) Native
  in
  if native_cuda then
    Stdio.eprintf
      "branchless_select: %s: native PTX is outside the floor-target claim (gh-ocannl-1822: seen \
       with ptxas 13.4 at sm_120); value claims remain active\n"
      name;
  let air_readings = lazy (gpu_census Asm.Air ~patterns:gpu_patterns name) in
  (* Metal's census runs wherever the toolchain does, its counts on stderr, pinned or not. *)
  if air_census_runs () then ignore (Lazy.force air_readings : gpu_reading list);
  Verdict.gate ~aggregation:`Environment ~when_:(not native_cuda) ~on:"CUDA native PTX target" claim
    (fun () ->
      if on_cc then p claim (compiled_branch_free ~patterns name)
      else
        match gpu_dialect with
        | Some Asm.Air -> (
            match air_expected with
            | None ->
                Verdict.in_leg ~leg:air_toolchain_leg claim (fun label ->
                    p_all label (Lazy.force air_readings) ~f:branch_free)
            | Some _ ->
                Verdict.skipped claim
                  ~backend:(Printf.sprintf "metal (%s, %s)" air_skip air_boundary))
        | Some Asm.Amdgpu ->
            Verdict.in_leg ~leg:single_loaded_blend_leg claim (fun label ->
                p_all label (gpu_census Asm.Amdgpu ~patterns:gpu_patterns name) ~f:branch_free)
        | Some dialect ->
            p_all claim (gpu_census dialect ~patterns:gpu_patterns name) ~f:branch_free
        | None ->
            Stdio.eprintf
              "branchless_select: %s: no census on %s: no tool here reads its compiler's output \
               (not part of the golden)\n"
              name backend_name;
            skipped claim);
  Option.iter air_expected ~f:(fun expected ->
      in_legf ~leg:air_pinned_leg "%s (pinned on Metal Toolchain %s; %s)" (air_claim expected)
        (String.concat ~sep:", " air_pinned_toolchains) air_boundary (fun label ->
          p_all label (Lazy.force air_readings) ~f:(fun r -> r.readable && r.branches = expected)))

(* === The value legs. === *)

type leg = {
  name : string;
  prec : Ops.prec;
  cpu_only : bool;  (** whose expected values are the C backends' (C typing, or no f64 elsewhere) *)
  seed : (Tn.t * float array) list;
  out : Tn.t;
  llc : LL.t;
  want : float array;
}

(* [?else_zero]: the else-arm is the constant [0.0] — the backward gates' shape, [where cond g 0].
   It is also the shape a compiler cannot if-convert by selecting between two addresses instead,
   which it does when both arms are loads of one index. *)
let where_leg ?(else_zero = false) ?(cpu_only = false) ?axis ~name ~mk ~cmk ~cprec ~prec () =
  let cv = [| 1.; 0.; nan; -0.; 2.; 0.; -.inf; 0.5 |] in
  let gv = [| -0.; nan; 3.5; nan; inf; inf; -2.25; -0. |] in
  let ev =
    if else_zero then Array.create ~len:n 0. else [| 5.; -0.; 7.; -0.; nan; -.inf; 1.; 9. |]
  in
  let want = Array.init n ~f:(fun i -> if Float.(cv.(i) <> 0.) then gv.(i) else ev.(i)) in
  let cond = cmk (name ^ "_c") and g = mk (name ^ "_g") and e = mk (name ^ "_e") in
  let out = mk (name ^ "_out") in
  let i = sym () in
  let else_ = if else_zero then c 0. else get e [| iter i |] in
  let sel =
    LL.Ternop (Ops.Where, (get cond [| iter i |], cprec), (get g [| iter i |], prec), (else_, prec))
  in
  {
    name;
    prec;
    cpu_only;
    seed = ([ (cond, cv); (g, gv) ] @ if else_zero then [] else [ (e, ev) ]);
    out;
    llc = loop_n ?axis i n (set out [| iter i |] sel);
    want;
  }

let gate_leg ?axis ?(mk = mk_single) ?(prec = single) ?(offset = 0.) ~name ~op ~semantics ~xv ~gv ()
    =
  let x = mk (name ^ "_x") and g = mk (name ^ "_g") in
  let out = mk (name ^ "_out") in
  let i = sym () in
  let at tn = (get tn [| iter i |], prec) in
  let cond =
    if Float.equal offset 0. then at x else (LL.Binop (Ops.Add, at x, (c offset, prec)), prec)
  in
  let sel = LL.Binop (op, cond, at g) in
  {
    name;
    prec;
    cpu_only = false;
    seed = [ (x, xv); (g, gv) ];
    out;
    llc = loop_n ?axis i n (set out [| iter i |] sel);
    want =
      Array.init n ~f:(fun i ->
          let x = xv.(i) in
          semantics (if Float.equal offset 0. then x else x +. offset) gv.(i));
  }

(* A select inlined into further arithmetic, [out = (sel + h) - k] at f32. Both a gate's [0.0f] arm
   and a [Where] of float arms keep the result at f32, so each arithmetic step rounds. The gate used
   to inherit double from [0.0]; gh-ocannl-2045 makes its spelling agree with the requested
   precision. The inputs distinguish those two evaluation types. *)
let composed_leg ~name ~select =
  let x = mk_single (name ^ "_x") and g = mk_single (name ^ "_g") and e = mk_single (name ^ "_e") in
  let h = mk_single (name ^ "_h") and k = mk_single (name ^ "_k") in
  let out = mk_single (name ^ "_out") in
  let xv = [| 1.; 1.; 1.; -1.; 1.; 0.; 1.; 2. |] in
  let gv = [| 1.; 3.; -1.; 5.; 0.5; 1.; 0.25; 1. |] in
  let ev = [| 7.; 7.; 7.; 9.; 7.; 2.; 7.; 7. |] in
  let big = Array.create ~len:n 16777216. in
  let i = sym () in
  let at tn = (get tn [| iter i |], single) in
  let sel, selected =
    match select with
    | `Relu_gate ->
        ( LL.Binop (Ops.Relu_gate, at x, at g),
          Array.init n ~f:(fun i -> if Float.(xv.(i) > 0.) then gv.(i) else 0.) )
    | `Where ->
        ( LL.Ternop (Ops.Where, at x, at g, at e),
          Array.init n ~f:(fun i -> if Float.(xv.(i) <> 0.) then gv.(i) else ev.(i)) )
  in
  let to_f32 v = Int32.float_of_bits (Int32.bits_of_float v) in
  let want = Array.mapi selected ~f:(fun i s -> to_f32 (to_f32 (s +. big.(i)) -. big.(i))) in
  let sum = LL.Binop (Ops.Add, (sel, single), at h) in
  let llsc = LL.Binop (Ops.Sub, (sum, single), at k) in
  {
    name;
    prec = single;
    cpu_only = true;
    seed =
      ([ (x, xv); (g, gv); (h, big); (k, big) ]
      @ match select with `Where -> [ (e, ev) ] | `Relu_gate -> []);
    out;
    llc = loop_n i n (set out [| iter i |] llsc);
    want;
  }

(* [where (x == m) g 0] at f32: the max-pool backward gate's own shape, a float EQUALITY test over a
   loaded gradient (gh-ocannl-1695: gcc branched on exactly this test, gh-ocannl-1660), over equal,
   unequal and unordered pairs, signed zeros comparing equal. Where the test fails the gradient is
   NaN or an infinity, which must not reach the result. *)
let eq_gate_leg ?axis ~name () =
  let x = mk_single (name ^ "_x") and m = mk_single (name ^ "_m") and g = mk_single (name ^ "_g") in
  let out = mk_single (name ^ "_out") in
  let xv = [| 1.; 2.; nan; 0.; -0.; inf; 3.; 5. |] in
  let mv = [| 1.; 3.; 1.; -0.; 0.; inf; 3.; nan |] in
  let gv = [| -0.; nan; inf; 4.; -2.5; 8.; 0.25; -.inf |] in
  let i = sym () in
  let at tn = (get tn [| iter i |], single) in
  let cond = LL.Binop (Ops.Cmpeq, at x, at m) in
  let sel = LL.Ternop (Ops.Where, (cond, single), at g, (c 0., single)) in
  {
    name;
    prec = single;
    cpu_only = false;
    seed = [ (x, xv); (m, mv); (g, gv) ];
    out;
    llc = loop_n ?axis i n (set out [| iter i |] sel);
    want = Array.init n ~f:(fun i -> if Float.(xv.(i) = mv.(i)) then gv.(i) else 0.);
  }

(* [if (i < 6) out[i] = where (c[i]) g[i] 0] under the census loop: the select inside a kernel's
   bounds guard, the shape a grid wider than its tensor has -- where a hand-written CUDA probe
   compiled by nvcc (13.4, sm_120) kept [BSSY; @P0 BRA] around the load, though OCANNL's NVRTC path
   predicates it. [out] is seeded, so the two cells the guard skips read back what was seeded. *)
let masked_leg ?axis ~name () =
  let cv = [| 1.; 0.; 1.; -0.; 2.; 0.; 1.; 1. |] in
  let gv = [| -0.; nan; 3.5; nan; inf; inf; 5.; 6. |] in
  let ov = [| 9.; 9.; 9.; 9.; 9.; 9.; 7.; 8. |] in
  let cond = mk_single (name ^ "_c") and g = mk_single (name ^ "_g") in
  let out = mk_single (name ^ "_out") in
  let i = sym () in
  let sel =
    LL.Ternop
      (Ops.Where, (get cond [| iter i |], single), (get g [| iter i |], single), (c 0., single))
  in
  {
    name;
    prec = single;
    cpu_only = false;
    seed = [ (cond, cv); (g, gv); (out, ov) ];
    out;
    llc = loop_n ?axis i n (if_idx (lt (embed i) (ic 6)) (set out [| iter i |] sel));
    want =
      Array.init n ~f:(fun i ->
          if i >= 6 then ov.(i) else if Float.(cv.(i) <> 0.) then gv.(i) else 0.);
  }

let legs =
  [
    where_leg ~name:"bs_where_f32" ~mk:mk_single ~cmk:mk_single ~cprec:single ~prec:single ();
    where_leg ~else_zero:true ~axis:census_axis ~name:"bs_where_zero" ~mk:mk_single ~cmk:mk_single
      ~cprec:single ~prec:single ();
    where_leg ~cpu_only:true ~name:"bs_where_f64" ~mk:mk_double ~cmk:mk_double ~cprec:Ops.double
      ~prec:Ops.double ();
    where_leg ~name:"bs_where_hcond" ~mk:mk_single ~cmk:mk_half ~cprec:Ops.half ~prec:single ();
    where_leg ~name:"bs_where_bf16" ~mk:mk_bf16 ~cmk:mk_bf16 ~cprec:Ops.bfloat16 ~prec:Ops.bfloat16
      ();
    (* [x > 0.0 ? g : 0.0]: a NaN or a zero of either sign gates shut, to POSITIVE zero. *)
    gate_leg ~axis:census_axis ~name:"bs_relu_gate" ~op:Ops.Relu_gate
      ~semantics:(fun x g -> if Float.(x > 0.) then g else 0.)
      ~xv:[| nan; -0.; 0.; 1.; -1.; inf; 2.; -.inf |]
      ~gv:[| 1.; 2.; nan; -0.; nan; 4.; inf; 8. |]
      ();
    gate_leg ~name:"bs_sat01_gate" ~op:Ops.Satur01_gate
      ~semantics:(Ops.interpret_binop Ops.Satur01_gate)
      ~xv:[| 0.5; 1.5; -0.5; nan; 0.; 0.75; -0.; 1. |]
      ~gv:[| -0.; nan; nan; 7.; inf; -2.5; 1.; nan |]
      ();
    gate_leg ~name:"bs_sat01_gate_int32" ~op:Ops.Satur01_gate ~mk:mk_int32 ~prec:Ops.int32
      ~offset:1.
      ~semantics:(Ops.interpret_binop Ops.Satur01_gate)
      ~xv:[| -1.; 0.; 1.; 2.; -30.; 40.; -3.; 5. |]
      ~gv:[| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
      ();
    (* Composed into [(sel + h) - k] with [h = k = 2^24]: the sum keeps a small [sel] exactly at
       double but rounds it away at f32, so the result shows the C type the select had. *)
    composed_leg ~name:"bs_relu_composed" ~select:`Relu_gate;
    composed_leg ~name:"bs_where_composed" ~select:`Where;
    eq_gate_leg ~axis:census_axis ~name:"bs_where_eq" ();
    masked_leg ~axis:census_axis ~name:"bs_where_masked" ();
  ]

let () =
  List.iter legs ~f:(fun leg ->
      (* A skipped claim prints its label as a passing one does, so one golden holds on every
         backend: the cpu-only legs skip under the same labels cc evaluates. *)
      let claim = leg.name ^ ": values match the select's semantics, bitwise" in
      (if leg.cpu_only && not on_cc then skipped claim
       else
         let got = execute_leg ~name:leg.name ~seed:leg.seed ~out:leg.out leg.llc in
         p_all2 claim got leg.want ~f:same_bits);
      let text = lazy (kernel_text leg.name) in
      let leg' = blend_leg ~cpu_only:leg.cpu_only ~prec:leg.prec in
      in_legf ~leg:leg' "%s: the select calls the blend helper" leg.name (fun label ->
          p label (has (Lazy.force text) blend_marker));
      in_legf ~leg:leg' "%s: no short-circuiting spelling of the select remains" leg.name
        (fun label -> no_short_circuit_claim ~prec:leg.prec label (Lazy.force text)))

(* The causal fill selects a scaled score when admitted. Loading the score on discarded lanes is the
   unified-memory HIP regression in gh-ocannl-2427. *)
let () =
  let name = "bs_mask_fill" in
  let mask = mk_single "mask" and score = mk_single "score" and out = mk_single "out" in
  let i = sym () in
  let idx = [| iter i |] in
  let at tn = (get tn idx, single) in
  let scaled = LL.Binop (Ops.Mul, at score, (c 0.5, single)) in
  let fill = LL.Ternop (Ops.Where, at mask, (scaled, single), (c (-.inf), single)) in
  let scores = Array.init n ~f:(fun i -> Float.of_int ((2 * i) + 1)) in
  let masks = Array.init n ~f:(fun i -> if i % 2 = 0 then 1. else 0.) in
  let want =
    Array.map2_exn masks scores ~f:(fun m s -> if Float.(m > 0.) then 0.5 *. s else -.inf)
  in
  let llc = loop_n ~axis:census_axis i n (set out idx fill) in
  let got = execute_leg ~name ~seed:[ (mask, masks); (score, scores) ] ~out llc in
  p_all2 "causal fill: the mask selects scaled scores and negative infinities" got want ~f:same_bits;
  in_leg ~leg:hip_leg "causal fill: HIP keeps the score load in the selected arm" (fun claim ->
      let text = kernel_text name in
      let ident = Option.value_exn score.Tn.code_name in
      let score_access = facts.node_ident_prefix ^ ident ^ "[" in
      let after_question_mark = String.lsplit2 text ~on:'?' in
      p claim
        ((not (has text blend_marker))
        && Option.exists after_question_mark ~f:(fun (before, after) ->
            (not (has before score_access)) && has after score_access)))

(* The gradient was loaded by an earlier statement, so the gate only selects a register value. *)
let () =
  let name = "bs_register_gate" in
  let mask = mk_single "mask" and grad = mk_single "grad" and out = mk_single "out" in
  let local = mk_single ~dims:[||] "register_grad" in
  virtualize local;
  let id = LL.get_scope local in
  let i = sym () in
  let idx = [| iter i |] in
  let masks = Array.init n ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let grads = Array.init n ~f:(fun i -> Float.of_int (i + 1)) in
  let gate payload = LL.Binop (Ops.Relu_gate, (get mask idx, single), (payload, single)) in
  let in_loop body = loop_n ~axis:census_axis i n body in
  let raw = in_loop (set out idx (gate (get grad idx))) in
  let scoped =
    in_loop
      (seq
         (LL.Declare_local { id; needs_init = false })
         (seq (LL.Set_local (id, get grad idx)) (set out idx (gate (LL.Get_local id)))))
  in
  List.iter [ mask; grad; out ] ~f:materialize;
  let optimized = optimize_scoped ~name ~raw scoped in
  let got =
    List.hd_exn (execute ~name optimized ~seed:[ (mask, masks); (grad, grads) ] ~read:[ out ])
  in
  let want = Array.map2_exn masks grads ~f:(fun m g -> if Float.(m > 0.) then g else 0.) in
  p_all2 "register-only gradient gate: the selected gradient reaches the output" got want
    ~f:same_bits;
  in_leg ~leg:grid_register_blend_leg "register-only gradient gate: the emitter blends"
    (fun claim -> p claim (has (kernel_text name) blend_marker));
  in_leg ~leg:hip_leg "register-only gradient gate: HIP emits branch-free selection" (fun claim ->
      p_all claim (gpu_census Asm.Amdgpu ~patterns:[ "ocannl_sel_" ] name) ~f:branch_free);
  in_leg ~leg:air_toolchain_leg
    "register-only gradient gate: Metal's AIR function has 0 conditional branches and emits select"
    (fun claim -> p_all claim (air_function_branch_free_select name) ~f:Fn.id)

(* gh-ocannl-2400: a loaded-arm select inside a single-threaded [Serial] loop (no [Grid] or
   [Workgroup] loop). Metal's [Blend_serial_loop] blends it over [auto] locals via a GNU statement
   expression and compiles the whole AIR function with zero conditional branches and a [select]. *)
let () =
  let name = "bs_serial_loop_select" in
  let mask = mk_single (name ^ "_m")
  and grad = mk_single (name ^ "_g")
  and out = mk_single (name ^ "_out") in
  let i = sym () in
  let idx = [| iter i |] in
  let masks = Array.init n ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let grads = Array.init n ~f:(fun i -> Float.of_int (i + 1)) in
  let gate = LL.Binop (Ops.Relu_gate, (get mask idx, single), (get grad idx, single)) in
  let llc = loop_n ~axis:LL.Serial i n (set out idx gate) in
  let got = execute_leg ~name ~seed:[ (mask, masks); (grad, grads) ] ~out llc in
  let want = Array.map2_exn masks grads ~f:(fun m g -> if Float.(m > 0.) then g else 0.) in
  p_all2 "serial-loop loaded gate: values match the select's semantics, bitwise" got want
    ~f:same_bits;
  in_leg ~leg:air_leg "serial-loop loaded gate: Metal emits the blend inside a serial-only kernel"
    (fun claim -> p claim (has (kernel_text name) blend_marker));
  in_leg ~leg:air_toolchain_leg
    "serial-loop loaded gate: Metal's AIR function has 0 non-loop conditional branches and emits \
     select" (fun claim ->
      p_all claim
        (air_function_branch_free_select ~patterns:[ "ocannl_sel_" ] ~allow_loop_back_edges:true
           name)
        ~f:Fn.id)

let () =
  List.iter [ "bs_sat01_gate"; "bs_sat01_gate_int32" ] ~f:(fun name ->
      let leg = List.find_exn legs ~f:(fun leg -> String.equal leg.name name) in
      let condition, _ = List.hd_exn leg.seed in
      let text = kernel_text name in
      pf "%s: reads the condition once" name
        (Option.value_map condition.Tn.code_name ~default:false ~f:(fun ident ->
             let ident = facts.node_ident_prefix ^ ident in
             let reads suffix =
               List.length
                 (String.substr_index_all text ~may_overlap:false ~pattern:(ident ^ suffix))
             in
             reads "[" + reads ")[" = 1)))

(* The assembly claim, on the shapes of the backward pass: [where cond g 0], a [Relu_gate] and the
   max-pool gate's float equality, each gating a load from a buffer argument (which the compiler
   cannot prove dereferenceable). *)
let () =
  List.iter [ "bs_where_zero"; "bs_relu_gate"; "bs_where_eq"; "bs_where_masked" ] ~f:(fun name ->
      let claim = name ^ ": no conditional branch on the select's lines (the backend's compiler)" in
      (* Anchored on either spelling, so the same claim reads base's [?:] lines too. A hand-built
         leg's kernel holds this one select and nothing else conditional. *)
      census_claim
        ~patterns:("ocannl_sel_" :: short_circuit_forms)
        ~gpu_patterns:[ "ocannl_sel_"; " ? " ]
        ~air_claim:
          (Printf.sprintf
             "%s: Metal's AIR keeps the select's short-circuit, %d conditional branch(es) on its \
              line"
             name)
        ~air_skip:"its AIR keeps the select's short-circuit" claim name);
  in_leg ~leg:air_toolchain_leg
    "bs_where_zero: whole-function AIR check rejects a loaded-arm [?:] in a Grid loop (negative \
     control)" (fun claim ->
      let readings = air_function_branch_free_select "bs_where_zero" in
      p claim ((not (List.is_empty readings)) && List.exists readings ~f:not))

(* === The guards that must keep short-circuiting. === *)

let a_vals = [| 1.5; 2.5; 3.5; 4.5 |]

(* [out[i] := i < 4 ? a[i] : 0] over [i < 6]: [a] has 4 cells, so the then-arm is out of range
   exactly where the guard fails. *)
let () =
  let name = "bs_index_guard" in
  let a = mk_single ~dims:[| 4 |] (name ^ "_a") and out = mk_single ~dims:[| 6 |] (name ^ "_out") in
  let i = sym () in
  let llc =
    loop_n i 6 (set out [| iter i |] (where_ (lt (embed i) (ic 4)) (get a [| iter i |]) (c 0.)))
  in
  let got = execute_leg ~name ~seed:[ (a, a_vals) ] ~out llc in
  p_all2 (name ^ ": values") got [| 1.5; 2.5; 3.5; 4.5; 0.; 0. |] ~f:same_bits;
  in_leg ~leg:single_blend_leg (name ^ ": an index guard keeps its [?:] and calls no blend")
    (fun claim ->
      let text = kernel_text name in
      p claim (has text guard_spelling && not (has text blend_marker)))

(* [out[i] := 0 <= idx[i] && idx[i] < 4 ? table[idx[i]] : 0]: the guard reads only a data value — no
   index at all — and the inactive lanes' rows are far out of range, so only the [Get_dynamic] in
   its arm keeps it short-circuiting. *)
let () =
  let name = "bs_gather_guard" in
  let table = mk_single ~dims:[| 4 |] (name ^ "_t") and idx = mk_single (name ^ "_idx") in
  let out = mk_single (name ^ "_out") in
  let iv = [| 0.; 3.; 1e6; -5e5; 2.; 1.; 4.; -1. |] in
  let i = sym () in
  let row = get idx [| iter i |] in
  let guard = conj (le (ic 0) row) (lt row (ic 4)) in
  let read = gather ~tn:table ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:(row, single) in
  let llc = loop_n i n (set out [| iter i |] (where_ guard read (c 0.))) in
  let got = execute_leg ~name ~seed:[ (table, a_vals); (idx, iv) ] ~out llc in
  let want =
    Array.map iv ~f:(fun r -> if Float.(r >= 0. && r < 4.) then a_vals.(Float.to_int r) else 0.)
  in
  p_all2 (name ^ ": values") got want ~f:same_bits;
  in_leg ~leg:single_blend_leg (name ^ ": a value-guarded gather keeps its [?:] and calls no blend")
    (fun claim ->
      let text = kernel_text name in
      p claim (has text guard_spelling && not (has text blend_marker)))

(* The index-derived score keeps both arms load-free, so the NaN literal alone keeps the GPU blend
   short-circuiting (gh-ocannl-1695). *)
let () =
  let name = "bs_nan_fill" in
  let m = mk_single (name ^ "_m") in
  let out = mk_single (name ^ "_out") in
  let mv = [| 1.; 0.; 1.; 1.; 0.; 0.; 1.; 0. |] in
  let i = sym () in
  let score = LL.Binop (Ops.Add, (LL.Embed_index (iter i), single), (c 1., single)) in
  let sel =
    LL.Ternop
      ( Ops.Where,
        (get m [| iter i |], single),
        (LL.Binop (Ops.Mul, (score, single), (c 0.5, single)), single),
        (c nan, single) )
  in
  let llc = loop_n i n (set out [| iter i |] sel) in
  let got = execute_leg ~name ~seed:[ (m, mv) ] ~out llc in
  let want =
    Array.init n ~f:(fun i -> if Float.(mv.(i) <> 0.) then Float.of_int (i + 1) *. 0.5 else nan)
  in
  p_all2 (name ^ ": values, the NaN fill reaching the masked cells") got want ~f:same_bits;
  in_leg ~leg:over_locals_leg
    (name ^ ": a NaN-literal arm keeps its [?:] where the blend binds locals") (fun claim ->
      let text = kernel_text name in
      p claim (has text guard_spelling && not (has text blend_marker)))

(* [out[i] := v[i] != 0 ? (i < 4 ? a[i] : 0) : -1] over [i < 6]. The inner index guard protects the
   read. HIP keeps the outer test too because a nested arm contains a load. *)
let () =
  let name = "bs_nested_guard" in
  let a = mk_single ~dims:[| 4 |] (name ^ "_a") and v = mk_single ~dims:[| 6 |] (name ^ "_v") in
  let out = mk_single ~dims:[| 6 |] (name ^ "_out") in
  let vv = [| 1.; 0.; 1.; 0.; nan; 1. |] in
  let i = sym () in
  let inner = where_ (lt (embed i) (ic 4)) (get a [| iter i |]) (c 0.) in
  let outer =
    LL.Ternop (Ops.Where, (get v [| iter i |], single), (inner, single), (c (-1.), single))
  in
  let llc = loop_n i 6 (set out [| iter i |] outer) in
  let got = execute_leg ~name ~seed:[ (a, a_vals); (v, vv) ] ~out llc in
  p_all2 (name ^ ": values") got [| 1.5; -1.; 3.5; -1.; 0.; 0. |] ~f:same_bits;
  in_leg ~leg:single_loaded_blend_leg
    (name ^ ": the value select blends around the inner guard's [?:]") (fun claim ->
      let text = kernel_text name in
      p claim (has text blend_marker && has text guard_spelling));
  in_leg ~leg:hip_leg (name ^ ": HIP keeps both conditionals around the nested load") (fun claim ->
      let text = kernel_text name in
      let store = facts.node_ident_prefix ^ Option.value_exn out.Tn.code_name ^ "[" in
      let statement =
        Option.bind (String.substr_index text ~pattern:store) ~f:(fun pos ->
            String.lsplit2 (String.drop_prefix text pos) ~on:';')
      in
      p claim
        (Option.exists statement ~f:(fun (statement, _) ->
             (not (has statement blend_marker)) && String.count statement ~f:(Char.equal '?') = 2)))

(* [out[i] := v[i] != 0 ? ((int)a[i] / (int)b[i] == 0 ? g[i] : 0) : -1], and the same with one more
   select between: [v[i] != 0 ? (u[i] != 0 ? (... division select ...) : 0) : -1]. The division's
   condition is an int32 comparison, so it renders as C INTEGER division even though the division's
   own annotation says f32 (a homogeneous operator renders its operands at its own precision, review
   round 1), and [b] is zero exactly where the outer select is off. Blending the outer select would
   execute it: SIGFPE on x86, a silently discarded value on AArch64 -- so the claim is the
   structural one, the outer select keeping its [?:]. The deeper form is review round 2's: the
   middle select keeps short-circuiting, but its condition [u] can hold where [v] does not, so it
   protects nothing once the outer arm evaluates eagerly. *)
let int_div_leg ~name ~middle_select =
  let mk4 = node_factory ~first_id:(if middle_select then 16400 else 16380) ~dims:[| 4 |] () in
  let v = mk4 (name ^ "_v") and u = mk4 (name ^ "_u") in
  let a = mk4 (name ^ "_a") and b = mk4 (name ^ "_b") in
  let g = mk4 (name ^ "_g") and out = mk4 (name ^ "_out") in
  let vv = [| 1.; 0.; 1.; 0. |] and uv = [| 1.; 1.; 0.; 1. |] in
  let av = [| 1.; 5.; 6.; 7. |] and bv = [| 2.; 0.; 3.; 0. |] in
  let gv = [| 0.5; 1.5; 2.5; 3.5 |] in
  let i = sym () in
  let at tn = (get tn [| iter i |], single) in
  let quotient = LL.Binop (Ops.Div, at a, at b) in
  let div_cond = LL.Binop (Ops.Cmpeq, (quotient, single), (c 0., single)) in
  let division_select = LL.Ternop (Ops.Where, (div_cond, Ops.int32), at g, (c 0., single)) in
  let arm =
    if middle_select then LL.Ternop (Ops.Where, at u, (division_select, single), (c 0., single))
    else division_select
  in
  let outer = LL.Ternop (Ops.Where, at v, (arm, single), (c (-1.), single)) in
  let llc = loop_n i 4 (set out [| iter i |] outer) in
  let seed = [ (v, vv); (a, av); (b, bv); (g, gv) ] @ if middle_select then [ (u, uv) ] else [] in
  let got = execute_leg ~name ~seed ~out llc in
  (* Integer quotients 1/2 = 0 and 6/3 = 2; the middle select is off at lane 2. *)
  let want = [| 0.5; -1.; 0.; -1. |] in
  p_all2 (name ^ ": values") got want ~f:same_bits;
  (* The outer select, over [v], keeps its [?:]; an inner one may blend, since its condition is
     evaluated inside the outer arm either way. *)
  let claim = name ^ ": a select whose arm divides integers keeps its [?:]" in
  if on_cc then
    let text = kernel_text name in
    p claim
      (List.exists (String.split_lines text) ~f:(fun l ->
           has l (name ^ "_v[") && has l ") != 0.0 ? ("))
  else skipped claim

let () =
  int_div_leg ~name:"bs_int_div_guard" ~middle_select:false;
  int_div_leg ~name:"bs_int_div_nested" ~middle_select:true

(* === The realistic composition: ReLU into a 2x2 max pool, backward. === *)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let ph = 4
let pw = 4

(* Ties within windows (the quantized pattern repeats values), negatives, and exact zeros. *)
let x_f idcs =
  match idcs with [| r; c |] -> Float.of_int (((r * 3) + (c * 5)) % 7) -. 3. | _ -> assert false

let w_f idcs =
  match idcs with [| r; c |] -> 0.25 *. Float.of_int (1 + r + (2 * c)) | _ -> assert false

let run_pool ~name =
  Tensor.unsafe_reinitialize ();
  let x =
    Operation.init ~l:"bs_px" ~prec:Ops.single ~b:[] ~o:[ ph; pw ] ~f:x_f
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    Operation.init ~l:"bs_pw" ~prec:Ops.single ~b:[]
      ~o:[ ph / 2; pw / 2 ]
      ~f:w_f ~grad_spec:Tensor.Prohibit_grad ()
  in
  let%op y = Nn_blocks.max_pool2d () (relu x) in
  let%op loss = (y *. w) ++ "... | ... => 0" in
  Train.set_materialized (grad_of x);
  (* The pool's gradient a buffer argument, as in a network too big to keep it routine-local: the
     max-pool gate's arm is then a load the compiler cannot speculate, so the short-circuiting
     spelling would branch on it. *)
  Train.set_materialized (grad_of y);
  let ctx, routine =
    Context.compile ~name (Context.auto ()) (Train.grad_update loss) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx (grad_of x)

(* The gradient the short-circuiting semantics define: a cell gets its window's weight where its
   ReLU value achieves the window's maximum (every tie) and its input is positive. *)
let pool_reference =
  Array.init (ph * pw) ~f:(fun k ->
      let r = k / pw and c = k % pw in
      let h r c = Float.max (x_f [| r; c |]) 0. in
      let wr = r / 2 and wc = c / 2 in
      let m =
        List.fold
          [ (0, 0); (0, 1); (1, 0); (1, 1) ]
          ~init:Float.neg_infinity
          ~f:(fun acc (dr, dc) -> Float.max acc (h ((2 * wr) + dr) ((2 * wc) + dc)))
      in
      if Float.(h r c = m && x_f [| r; c |] > 0.) then w_f [| wr; wc |] else 0.)

let () =
  let inlined = run_pool ~name:"bs_pool_inlined" in
  let settings = LL.virtualize_settings in
  let simple = settings.inline_simple_computations
  and complex = settings.inline_complex_computations in
  settings.inline_simple_computations <- false;
  settings.inline_complex_computations <- false;
  let materialized = run_pool ~name:"bs_pool_materialized" in
  settings.inline_simple_computations <- simple;
  settings.inline_complex_computations <- complex;
  p "the reference gradient is discriminating (zero and nonzero cells)"
    (Array.exists pool_reference ~f:(fun v -> Float.(v = 0.))
    && Array.exists pool_reference ~f:(fun v -> Float.(v > 0.)));
  p_all2 "pool backward: the inlined gradient equals the reference, bitwise" inlined pool_reference
    ~f:same_bits;
  p_all2 "pool backward: the inlined gradient equals the inlining-off run, bitwise" inlined
    materialized ~f:same_bits;
  if outlined then
    Verdict.claim "the forced run rendered the pool kernel outlined"
      (not (List.is_empty (Generated.outlined_parts (Generated.read "bs_pool_inlined"))));
  in_leg ~leg:single_serial_blend_leg "pool backward: the inlined gradient's selects call the blend"
    (fun claim ->
      let text = kernel_text "bs_pool_inlined" in
      p claim (has text blend_marker));
  (* The realistic kernel's branch census (gh-ocannl-1660). gcc at x86 is where it bites: over the
     blend's two values it still branched on both selects, the max-pool gate's float equality test
     ([vucomiss], [jp], [je]: two flags defeat its if-conversion) and the ReLU gate, whose float arm
     is converted to the double result inside a branch of its own -- 16 and 8 conditional branches
     on gcc 15.2, until [OCANNL_SELECT] selected the bit patterns under gcc. Every one of them was
     the select's: the enclosing index guards' [?:], which share the select's closing line, fold
     away once the window loops unroll, and no branch was attributed to their lines. On clang the
     claim does not discriminate at this size (the short-circuiting spelling already if-converts
     these few unrolled loads); it holds there as it does on gcc. *)
  let claim =
    "pool backward: no conditional branch on the inlined selects' lines (the backend's compiler)"
  in
  (* On a GPU dialect that keeps the short-circuiting spelling (CUDA), every [?:] line: the selects
     share their lines with the window loops' index guards, so this anchors a superset, and holds
     only if neither branches (gh-ocannl-1695). *)
  census_claim ~patterns:[ "ocannl_sel_" ]
    ~gpu_patterns:(if blends then [ "ocannl_sel_" ] else [ " ? " ])
    ~air_expected:None
    ~air_claim:(fun _ -> "")
    ~air_skip:"" claim "bs_pool_inlined"
