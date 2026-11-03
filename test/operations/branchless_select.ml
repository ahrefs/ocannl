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
   gates a [Relu_gate] shut, and [Satur01_gate] yields its operand where its test FAILS. Legs at
   f32, f64, a half-precision condition beside an f32 result, and bf16 storage computing at f32;
   and, on cc, a gate and a [Where] composed into further arithmetic, whose rounding follows the
   [?:] spelling's C type (a gate's [0.0] literal makes it double, a [Where] of floats stays float).
   - Emission, on cc: those legs call the blend and keep no [?:] spelling of their select. - Guards,
   on cc: an index-guarded out-of-range read and a value-guarded [Get_dynamic] gather (its index
   runs far out of range on the inactive lanes) keep [?:]; a value select over an index guard blends
   outside and keeps the inner guard. Executed with the hazardous reads inactive. - Assembly, on
   cc's own compiler and kernel flags: no conditional branch is attributed to a select's source
   lines, and those lines carry instructions, the realistic leg's included (gh-ocannl-1660); on CUDA
   and HIP (gh-ocannl-1695; HIP blends, CUDA's ptxas predicates the short-circuiting spelling
   itself), the same claim of the code object the run compiled (SASS through [ptxas] and [nvdisasm],
   AMDGPU through [llvm-objdump]), Metal's skipped by name. - The realistic composition: a ReLU
   feeding a 2x2 max pool, backward through the default placements (the selects inlined into their
   consumers) — the gradient emits the blend and equals, bitwise, both an OCaml reference and the
   same program with inlining off. *)

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

(* The GPU backends whose code object a census can read (gh-ocannl-1695), by its dialect. Of them
   HIP blends, since AMDGPU branched on every census leg; CUDA keeps the short-circuiting spelling,
   which ptxas if-converts on every one (a predicated load), so there the census claim holds of that
   spelling. Metal is neither: no tool here reads what its compiler emits (the offline [metal]
   compiler ships with full Xcode only, and Apple's GPU ISA has no public disassembler), so it keeps
   the short-circuiting spelling and its census claims are skipped by name. *)
let gpu_dialect =
  match backend_name with "cuda" -> Some Asm.Sass | "hip" -> Some Asm.Amdgpu | _ -> None

let blends = on_cc || String.equal backend_name "hip"

(* The loop the census legs' select runs under. On a GPU dialect, a [Grid] loop: one select per
   thread, the shape a parallel backward kernel has (a serial loop's unrolled copies are the other
   shape, which the realistic pool leg keeps). Serial elsewhere. *)
let census_axis = if Option.is_some gpu_dialect then LL.Grid else LL.Serial

(* The kernel-text claims each leg makes where the backend blends, skipped by name elsewhere:
   written once (gh-ocannl-1703). *)
let blend_leg ~cpu_only = Verdict.leg ~when_:(on_cc || (blends && not cpu_only)) ~on:backend_name ()
let () = Generated.init ~backend_name

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

(* The kernel's statements ({!Generated.main_logic}): not the builtins prelude, which spells [?:]
   itself (the integer-power helpers, the select helpers), and, where the kernel was rendered
   outlined, its parts' bodies spliced in at their calls. The text after the kernel's name read only
   the kernel's own body, which an outlined kernel spends calling its parts: its positive claims
   went red and its negative ones held over nothing (gh-ocannl-1778). *)
let kernel_text name = Generated.main_logic (Generated.read name)
let has text s = String.is_substring text ~substring:s

(* Every short-circuiting spelling a value select has on cc ([Ops.ternop_c_syntax],
   [Ops.binop_c_syntax], float arms). *)
let short_circuit_forms = [ ") != 0.0 ? ("; " > 0.0 ?"; ")) > 0.0 ? 0.0 :" ]

(* The blend's own spelling, every backend's ([C_syntax.blend_local_a]): cc's [__auto_type] and the
   GPU dialects' [auto] local of the first arm. *)
let blend_marker = Ir.C_syntax.blend_local_a ^ " ="

(* The claim that [text] keeps no short-circuiting spelling of a value select. On cc, none of
   [short_circuit_forms]. On a GPU dialect the blend keeps the dialect's own [?:] over its locals
   ([C_syntax.Blend_over_locals]), so what must not remain is a [?] selecting anything but a local:
   every [?] -- of which a blended kernel has at least its blend's own -- is followed by
   [C_syntax.blend_local_a]. *)
let no_short_circuit_claim label text =
  if on_cc then p_none label short_circuit_forms ~f:(has text)
  else
    let after_each_question_mark =
      List.filter_mapi (String.to_list text) ~f:(fun i ch ->
          Option.some_if (Char.equal ch '?') (String.lstrip (String.drop_prefix text (i + 1))))
    in
    p_all label after_each_question_mark ~f:(String.is_prefix ~prefix:Ir.C_syntax.blend_local_a)

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
     else
       match
         Ir.Backend_intf.parse_static_properties (Context.static_properties (Context.auto ()))
       with
       | Some { devices = first :: _; _ } -> (
           let field key =
             Option.bind (List.Assoc.find first key ~equal:String.equal) ~f:(function
               | Sexp.Atom a -> Int.of_string_opt a
               | Sexp.List _ -> None)
           in
           match (field "compute_capability_major", field "compute_capability_minor") with
           | Some major, Some minor -> Some (Printf.sprintf "sm_%d%d" major minor)
           | _ -> None)
       | _ -> None)

(* The GPU counterpart of [compiled_branch_free] (gh-ocannl-1695): the code object the run itself
   compiled for [name] -- [<name>.ptx] on CUDA, [<name>.hsaco] on HIP, or each [<name>.part<k>] a
   split compile wrote -- disassembled with its line tables ([Asm.disassemble_gpu]), and no
   conditional branch attributed to a line [select_lines] reads in the source that artifact was
   compiled from. One verdict per artifact: each holds only when its line tables name its source and
   the anchored lines carry instructions, and no artifact at all is an empty list, which the claim's
   [p_all] fails; a missing tool fails too, never skips: on this backend the census is the claim. *)
let gpu_branch_free dialect ~patterns name =
  let artifact_ext, source_ext =
    match dialect with Asm.Sass -> (".ptx", ".cu") | Asm.Amdgpu -> (".hsaco", ".hip")
  in
  (* The whole program's artifact, or each part a split compile wrote ([<name>.part<k>], from 0). *)
  let exists stem = Stdlib.Sys.file_exists (Generated.path ~ext:artifact_ext stem) in
  let rec parts k =
    let stem = Printf.sprintf "%s.part%d" name k in
    if exists stem then stem :: parts (k + 1) else []
  in
  let stems = (if exists name then [ name ] else []) @ parts 0 in
  let say fmt =
    Printf.ksprintf
      (fun m -> Stdio.eprintf "branchless_select: %s: %s (not part of the golden)\n" name m)
      fmt
  in
  if List.is_empty stems then say "no %s artifact" artifact_ext;
  (* Every artifact censused, a red one printing its sites, before the verdict. *)
  List.map stems ~f:(fun stem ->
      let artifact = Generated.path ~ext:artifact_ext stem in
      let source_basename = stem ^ source_ext in
      let source = Generated.read ~ext:source_ext stem in
      match Asm.disassemble_gpu ?cuda_arch:(Lazy.force cuda_arch) dialect ~artifact with
      | Error why ->
          say "%s: the census cannot read it: %s" stem why;
          false
      | Ok (dump, toolchain) ->
          let parsed = Asm.parse_gpu dialect ~dump ~source_basename in
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
          Asm.explain_branch_claim ~source ~prefix:("branchless_select: " ^ stem)
            ~holds:
              (List.is_empty sites && instructions > 0 && parsed.g_markers > 0
              && not (Set.is_empty anchor))
            sites)

(* The census claim on whichever compiler the backend has: cc's host compiler, a GPU's code object,
   or -- on Metal -- a skip naming why. *)
let census_claim ~patterns ~gpu_patterns claim name =
  if on_cc then p claim (compiled_branch_free ~patterns name)
  else
    match gpu_dialect with
    | Some dialect -> p_all claim (gpu_branch_free dialect ~patterns:gpu_patterns name) ~f:Fn.id
    | None ->
        Stdio.eprintf
          "branchless_select: %s: no census on %s: no tool here reads its compiler's output (not \
           part of the golden)\n"
          name backend_name;
        skipped claim

(* === The value legs. === *)

type leg = {
  name : string;
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
    cpu_only;
    seed = ([ (cond, cv); (g, gv) ] @ if else_zero then [] else [ (e, ev) ]);
    out;
    llc = loop_n ?axis i n (set out [| iter i |] sel);
    want;
  }

let gate_leg ?axis ~name ~op ~semantics ~xv ~gv () =
  let x = mk_single (name ^ "_x") and g = mk_single (name ^ "_g") in
  let out = mk_single (name ^ "_out") in
  let i = sym () in
  let sel = LL.Binop (op, (get x [| iter i |], single), (get g [| iter i |], single)) in
  {
    name;
    cpu_only = false;
    seed = [ (x, xv); (g, gv) ];
    out;
    llc = loop_n ?axis i n (set out [| iter i |] sel);
    want = Array.init n ~f:(fun i -> semantics xv.(i) gv.(i));
  }

(* A select inlined into further arithmetic, [out = (sel + h) - k] at f32. The expected values are
   the C semantics of the short-circuiting spelling, which the blend must keep: [x > 0.0 ? g : 0.0]
   is a DOUBLE (its [0.0] literal), so the gate's sum and difference are exact; a [Where] of float
   arms is a float, so both round at f32. An f32-typed blend would flip the first (review round 1).
   cc only: another dialect types the literal its own way. *)
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
  let sel, want =
    match select with
    | `Relu_gate ->
        ( LL.Binop (Ops.Relu_gate, at x, at g),
          Array.init n ~f:(fun i -> if Float.(xv.(i) > 0.) then gv.(i) else 0.) )
    | `Where ->
        let to_f32 v = Int32.float_of_bits (Int32.bits_of_float v) in
        ( LL.Ternop (Ops.Where, at x, at g, at e),
          Array.init n ~f:(fun i ->
              let s = if Float.(xv.(i) <> 0.) then gv.(i) else ev.(i) in
              to_f32 (to_f32 (s +. big.(i)) -. big.(i))) )
  in
  let sum = LL.Binop (Ops.Add, (sel, single), at h) in
  let llsc = LL.Binop (Ops.Sub, (sum, single), at k) in
  {
    name;
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
    (* [fabsf(floorf(x)) > 0.0 ? 0.0 : g]: the operand passes where the test FAILS — [x] in [0, 1),
       and NaN. *)
    gate_leg ~name:"bs_sat01_gate" ~op:Ops.Satur01_gate
      ~semantics:(fun x g -> if Float.(abs (round_down x) > 0.) then 0. else g)
      ~xv:[| 0.5; 1.5; -0.5; nan; 0.; 0.75; -1.; 2. |]
      ~gv:[| -0.; nan; nan; 7.; inf; -2.5; 1.; nan |]
      ();
    (* Composed into [(sel + h) - k] with [h = k = 2^24]: the sum keeps a small [sel] exactly at
       double but rounds it away at f32, so the result shows the C type the select had. *)
    composed_leg ~name:"bs_relu_composed" ~select:`Relu_gate;
    composed_leg ~name:"bs_where_composed" ~select:`Where;
    eq_gate_leg ~axis:census_axis ~name:"bs_where_eq" ();
    masked_leg ~axis:census_axis ~name:"bs_where_masked" ();
  ]

(* The value claims a GPU backend fails on special values, a pre-existing divergence this test
   exposed (ahrefs/ocannl#1647), each paired with a backend observed failing it (CUDA RTX 5070 Ti
   and HIP gfx1151 at 1d62ce19, Metal M-series at 387de34a). [Satur01_gate] on all three, cells 3
   and 4: a NaN and a zero operand gate shut, since the GPU renderings spell [0 < x && x < 1 ? g :
   0] ([Ops.interpret_binop]'s semantics) where the reference follows cc's [fabs(floor(x)) > 0 ? 0 :
   g]. On HIP, besides, every select whose condition is NaN at a cell: the [Where] legs take the
   else-arm, [bs_nested_guard] included. Blended (gh-ocannl-1695, gfx1151) that reaches the
   zero-else [bs_where_zero] too, which the short-circuiting spelling had kept exact, while
   [Relu_gate], whose NaN operand the short-circuiting spelling passed, now gates shut exactly. The
   pairs are the defect's record, not a capability: no backend predicate names them. Each such claim
   is skipped on its backend only, with every disagreeing cell printed to stderr (and a note once
   none disagrees); every other backend, and every other claim, stays exact.

   HIP's NaN-condition entries are fast math's: with [hip_fast_math] off, as [profile=reproducible]
   sets it (gh-ocannl-1779; the [-no_hip_fast_math] rule), hiprtc honors NaN and every one of them
   matched on gfx1151, so they are recorded under fast math only and the claims hold exactly there.
   [bs_sat01_gate] is the semantics gap above and stays under either policy. *)
let hip_fast_math = Ir.Cuda_like_config.hip_fast_math ()

let known_divergence_1647 =
  [
    ("bs_sat01_gate", "cuda", true);
    ("bs_sat01_gate", "metal", true);
    ("bs_sat01_gate", "hip", true);
    ("bs_where_f32", "hip", hip_fast_math);
    ("bs_where_hcond", "hip", hip_fast_math);
    ("bs_where_bf16", "hip", hip_fast_math);
    ("bs_where_zero", "hip", hip_fast_math);
    ("bs_nested_guard", "hip", hip_fast_math);
  ]

(* [name]'s bitwise value claim, or its gated skip where ahrefs/ocannl#1647 records it failing. *)
let values_claim ~name claim got want =
  let diverges =
    List.exists known_divergence_1647 ~f:(fun (leg, backend, recorded) ->
        recorded && String.equal leg name && String.is_substring backend_name ~substring:backend)
  in
  if diverges then (
    Array.iteri got ~f:(fun i v ->
        let w = want.(i) in
        if not (same_bits v w) then
          Stdio.eprintf
            "branchless_select: %s cell %d: got %h, the reference %h (known divergence, \
             ahrefs/ocannl#1647; not part of the golden)\n"
            name i v w);
    Stdio.eprintf "branchless_select: %s: value claim gated on %s by ahrefs/ocannl#1647%s\n" name
      backend_name
      (if Array.for_all2_exn got want ~f:same_bits then " (every cell now matches: lift the gate)"
       else "");
    skipped claim)
  else p_all2 claim got want ~f:same_bits

let () =
  List.iter legs ~f:(fun leg ->
      (* A skipped claim prints its label as a passing one does, so one golden holds on every
         backend: the cpu-only legs skip under the same labels cc evaluates. *)
      let claim = leg.name ^ ": values match the select's semantics, bitwise" in
      (if leg.cpu_only && not on_cc then skipped claim
       else
         let got = execute_leg ~name:leg.name ~seed:leg.seed ~out:leg.out leg.llc in
         values_claim ~name:leg.name claim got leg.want);
      let text = lazy (kernel_text leg.name) in
      let leg' = blend_leg ~cpu_only:leg.cpu_only in
      in_legf ~leg:leg' "%s: the select calls the blend helper" leg.name (fun label ->
          p label (has (Lazy.force text) blend_marker));
      in_legf ~leg:leg' "%s: no short-circuiting spelling of the select remains" leg.name
        (fun label -> no_short_circuit_claim label (Lazy.force text)))

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
        ~gpu_patterns:[ "ocannl_sel_"; " ? " ] claim name)

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
  let claim = name ^ ": an index guard keeps its [?:] and calls no blend" in
  if blends then
    let text = kernel_text name in
    p claim (has text guard_spelling && not (has text blend_marker))
  else skipped claim

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
  let claim = name ^ ": a value-guarded gather keeps its [?:] and calls no blend" in
  if blends then
    let text = kernel_text name in
    p claim (has text guard_spelling && not (has text blend_marker))
  else skipped claim

(* [out[i] := m[i] != 0 ? s[i] * 0.5 : NAN]: a masked attention's fill, a NaN LITERAL arm. Under
   HIP's no-NaNs fast math a NaN bound to a local is the optimizer's to drop, and a blended [where
   mask s NAN] came back finite (gh-ocannl-1695), so the GPU blend leaves such a select
   short-circuiting; cc's blend selects bit patterns and keeps it. The NaN reaches the masked cells
   on every backend. *)
let () =
  let name = "bs_nan_fill" in
  let m = mk_single (name ^ "_m") and sv = mk_single (name ^ "_s") in
  let out = mk_single (name ^ "_out") in
  let mv = [| 1.; 0.; 1.; 1.; 0.; 0.; 1.; 0. |] in
  let svv = [| 2.; 4.; -6.; 0.5; 1.; 3.; -0.; 8. |] in
  let i = sym () in
  let sel =
    LL.Ternop
      ( Ops.Where,
        (get m [| iter i |], single),
        (LL.Binop (Ops.Mul, (get sv [| iter i |], single), (c 0.5, single)), single),
        (c nan, single) )
  in
  let llc = loop_n i n (set out [| iter i |] sel) in
  let got = execute_leg ~name ~seed:[ (m, mv); (sv, svv) ] ~out llc in
  let want = Array.init n ~f:(fun i -> if Float.(mv.(i) <> 0.) then svv.(i) *. 0.5 else nan) in
  p_all2 (name ^ ": values, the NaN fill reaching the masked cells") got want ~f:same_bits;
  let claim = name ^ ": a NaN-literal arm keeps its [?:] where the blend binds locals" in
  if String.equal backend_name "hip" then
    let text = kernel_text name in
    p claim (has text guard_spelling && not (has text blend_marker))
  else skipped claim

(* [out[i] := v[i] != 0 ? (i < 4 ? a[i] : 0) : -1] over [i < 6]: the outer, value select blends —
   its arm's one read sits behind a guard of its own — and the inner index guard stays [?:]. *)
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
  values_claim ~name (name ^ ": values") got [| 1.5; -1.; 3.5; -1.; 0.; 0. |];
  let claim = name ^ ": the value select blends around the inner guard's [?:]" in
  if blends then
    let text = kernel_text name in
    p claim (has text blend_marker && has text guard_spelling)
  else skipped claim

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
  let claim = "pool backward: the inlined gradient's selects call the blend" in
  if blends then
    let text = kernel_text "bs_pool_inlined" in
    p claim (has text blend_marker)
  else skipped claim;
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
    claim "bs_pool_inlined"
