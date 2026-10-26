(* Value selects render branchless on cc, guards never do (gh-ocannl-1622).

   A [Where] whose arm holds a load, spelled as C's short-circuiting [?:], compiles to a compare and
   a conditional branch: C forbids evaluating the unselected arm, and the compiler cannot prove the
   load dereferenceable. Inlined into the nests that consume them, the ReLU and max-pool backward
   selects made cifar_resnet's step time follow its activation sparsity (gh-ocannl-1557). The cc
   backend now renders a value select it may speculate ([Low_level.speculable_select]) as a call to
   [ocannl_select_f32] / [ocannl_select_f64], whose arguments always evaluate, so the compiler
   if-converts. The same [Where] is also how lowering guards a read that may be out of range, and
   those must keep short-circuiting.

   Pinned here:

   - Values, on every backend, bitwise against an OCaml reference of the short-circuiting semantics:
   an unselected NaN or infinity does not reach the result (a mask multiply would leak it), a
   selected negative zero stays negative, a NaN condition selects the then-arm of a [Where] and
   gates a [Relu_gate] shut, and [Satur01_gate] yields its operand where its test FAILS. Legs at
   f32, f64, a half-precision condition beside an f32 result, and bf16 storage computing at f32. -
   Emission, on cc: those legs call the blend and keep no [?:] spelling of their select. - Guards,
   on cc: an index-guarded out-of-range read and a value-guarded [Get_dynamic] gather (its index
   runs far out of range on the inactive lanes) keep [?:]; a value select over an index guard blends
   outside and keeps the inner guard. Executed with the hazardous reads inactive. - Assembly, on
   cc's own compiler and kernel flags: no conditional branch is attributed to a select's source
   lines, and those lines carry instructions. - The realistic composition: a ReLU feeding a 2x2 max
   pool, backward through the default placements (the selects inlined into their consumers) — the
   gradient emits the blend and equals, bitwise, both an OCaml reference and the same program with
   inlining off. *)

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
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The C backends are the ones whose syntax config is cc's, the only one with a blend. *)
let on_cc = Ir.Schedule.backend_is_cpu backend_name
let skipped = Verdict.skipped ~backend:backend_name
let () = Generated.init ~backend_name

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

(* The kernel's own text: the generated source past its builtins prelude, which spells [?:] itself
   (the integer-power helpers, the select helpers). *)
let kernel_text name =
  let src = Generated.read name in
  match String.substr_index src ~pattern:(name ^ "(") with
  | Some i -> String.drop_prefix src i
  | None -> ""

let has text s = String.is_substring text ~substring:s

(* Every short-circuiting spelling a value select has on cc ([Ops.ternop_c_syntax],
   [Ops.binop_c_syntax], float arms). *)
let short_circuit_forms = [ ") != 0.0 ? ("; " > 0.0 ?"; ")) > 0.0 ? 0.0 :" ]

(* Executes [llc] writing [out], every node materialized and the inputs seeded. *)
let execute_leg ~name ~seed ~out llc =
  List.iter (out :: List.map seed ~f:fst) ~f:materialize;
  let _o, got = optimize_and_execute ~name llc ~seed ~read:[ out ] in
  List.hd_exn got

(* Whether [name]'s select compiled branch-free: compiled by the cc backend's own compiler with its
   own kernel flags (at [-O3], its default level), no conditional branch is attributed to a source
   line spelling a select, and those lines carry instructions (an anchor naming nothing, or a select
   folded away, fails). *)
let compiled_branch_free name =
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
      let anchor = Asm.anchor_lines ~source ~patterns:("ocannl_select_f" :: short_circuit_forms) in
      let branches, instructions = Asm.conditional_branches_in parsed ~anchor in
      Stdio.eprintf
        "branchless_select: %s: %d conditional branch(es) in %d instruction(s) on %d select \
         line(s) (not part of the golden)\n"
        name branches instructions (Set.length anchor);
      branches = 0 && instructions > 0

(* === The value legs. === *)

type leg = {
  name : string;
  blend : string;  (** the helper the leg's select must call on cc *)
  seed : (Tn.t * float array) list;
  out : Tn.t;
  llc : LL.t;
  want : float array;
}

(* [?else_zero]: the else-arm is the constant [0.0] — the backward gates' shape, [where cond g 0].
   It is also the shape a compiler cannot if-convert by selecting between two addresses instead,
   which it does when both arms are loads of one index. *)
let where_leg ?(else_zero = false) ~name ~blend ~mk ~cmk ~cprec ~prec () =
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
    blend;
    seed = ([ (cond, cv); (g, gv) ] @ if else_zero then [] else [ (e, ev) ]);
    out;
    llc = loop_n i n (set out [| iter i |] sel);
    want;
  }

let gate_leg ~name ~op ~semantics ~xv ~gv =
  let x = mk_single (name ^ "_x") and g = mk_single (name ^ "_g") in
  let out = mk_single (name ^ "_out") in
  let i = sym () in
  let sel = LL.Binop (op, (get x [| iter i |], single), (get g [| iter i |], single)) in
  {
    name;
    blend = "ocannl_select_f32(";
    seed = [ (x, xv); (g, gv) ];
    out;
    llc = loop_n i n (set out [| iter i |] sel);
    want = Array.init n ~f:(fun i -> semantics xv.(i) gv.(i));
  }

let legs =
  [
    where_leg ~name:"bs_where_f32" ~blend:"ocannl_select_f32(" ~mk:mk_single ~cmk:mk_single
      ~cprec:single ~prec:single ();
    where_leg ~else_zero:true ~name:"bs_where_zero" ~blend:"ocannl_select_f32(" ~mk:mk_single
      ~cmk:mk_single ~cprec:single ~prec:single ();
    where_leg ~name:"bs_where_f64" ~blend:"ocannl_select_f64(" ~mk:mk_double ~cmk:mk_double
      ~cprec:Ops.double ~prec:Ops.double ();
    where_leg ~name:"bs_where_hcond" ~blend:"ocannl_select_f32(" ~mk:mk_single ~cmk:mk_half
      ~cprec:Ops.half ~prec:single ();
    where_leg ~name:"bs_where_bf16" ~blend:"ocannl_select_f32(" ~mk:mk_bf16 ~cmk:mk_bf16
      ~cprec:Ops.bfloat16 ~prec:Ops.bfloat16 ();
    (* [x > 0.0 ? g : 0.0]: a NaN or a zero of either sign gates shut, to POSITIVE zero. *)
    gate_leg ~name:"bs_relu_gate" ~op:Ops.Relu_gate
      ~semantics:(fun x g -> if Float.(x > 0.) then g else 0.)
      ~xv:[| nan; -0.; 0.; 1.; -1.; inf; 2.; -.inf |]
      ~gv:[| 1.; 2.; nan; -0.; nan; 4.; inf; 8. |];
    (* [fabsf(floorf(x)) > 0.0 ? 0.0 : g]: the operand passes where the test FAILS — [x] in [0, 1),
       and NaN. *)
    gate_leg ~name:"bs_sat01_gate" ~op:Ops.Satur01_gate
      ~semantics:(fun x g -> if Float.(abs (round_down x) > 0.) then 0. else g)
      ~xv:[| 0.5; 1.5; -0.5; nan; 0.; 0.75; -1.; 2. |]
      ~gv:[| -0.; nan; nan; 7.; inf; -2.5; 1.; nan |];
  ]

let () =
  List.iter legs ~f:(fun leg ->
      let is_f64 = has leg.blend "f64" in
      if is_f64 && not on_cc then skipped (leg.name ^ ": values match the select's semantics")
      else
        let got = execute_leg ~name:leg.name ~seed:leg.seed ~out:leg.out leg.llc in
        p_all2
          (leg.name ^ ": values match the select's semantics, bitwise")
          got leg.want ~f:same_bits;
        if on_cc then (
          let text = kernel_text leg.name in
          p (leg.name ^ ": the select calls the blend helper") (has text leg.blend);
          p_none
            (leg.name ^ ": no short-circuiting spelling of the select remains")
            short_circuit_forms ~f:(has text))
        else (
          skipped (leg.name ^ ": the select calls the blend helper");
          skipped (leg.name ^ ": no short-circuiting spelling of the select remains")))

(* The assembly claim, on the two shapes of the backward pass: [where cond g 0] and a [Relu_gate],
   each gating a load from a buffer argument (which the compiler cannot prove dereferenceable). *)
let () =
  List.iter [ "bs_where_zero"; "bs_relu_gate" ] ~f:(fun name ->
      let claim = name ^ ": no conditional branch on the select's lines (host compiler)" in
      if on_cc then p claim (compiled_branch_free name) else skipped claim)

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
  if on_cc then
    let text = kernel_text name in
    p claim (has text "? (" && not (has text "ocannl_select_f"))
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
  if on_cc then
    let text = kernel_text name in
    p claim (has text "? (" && not (has text "ocannl_select_f"))
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
  p_all2 (name ^ ": values") got [| 1.5; -1.; 3.5; -1.; 0.; 0. |] ~f:same_bits;
  let claim = name ^ ": the value select blends around the inner guard's [?:]" in
  if on_cc then
    let text = kernel_text name in
    p claim (has text "ocannl_select_f32(" && has text "? (")
  else skipped claim

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
  let claim = "pool backward: the inlined gradient's selects call the blend" in
  if on_cc then
    let text = kernel_text "bs_pool_inlined" in
    p claim (has text "ocannl_select_f32(")
  else skipped claim;
  (* Not discriminating against the short-circuiting spelling at this size — the base compiler
     already if-converts these few unrolled loads — so the two hand-built legs above carry the
     assembly claim; this one keeps the realistic composition branch-free as it grows. *)
  let claim = "pool backward: no conditional branch on the select's lines (host compiler)" in
  if on_cc then p claim (compiled_branch_free "bs_pool_inlined") else skipped claim
