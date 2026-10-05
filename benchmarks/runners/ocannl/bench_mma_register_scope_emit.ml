(* gh-ocannl-1073: emit and validate a deep-k staged fp8 kernel before timing its precompiled
   baseline and register-resident versions. gh-ocannl-1190: [--layout=swizzled] stages both operand
   tiles as [Swizzle_b128], the autotuner's swizzled twin of the same geometry, so the two staged
   layouts can be timed against each other. The output path is the first non-flag argument. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module S = Ir.Schedule
module Generated = Test_utils.Generated

type layout = Plain | Swizzled

let layout_name = function Plain -> "plain" | Swizzled -> "swizzled"

let () =
  let args = Bench_args.create "bench_mma_register_scope_emit" in
  let path =
    match Bench_args.positional args with
    | [ path ] -> path
    | _ ->
        Bench_args.bad args
          "usage: OUTPUT.cu [--layout=plain|swizzled] ([--] before an OUTPUT.cu starting with -)"
  in
  let layout =
    match Bench_args.flag_value args ~flag:"layout" with
    | None | Some "plain" -> Plain
    | Some "swizzled" -> Swizzled
    | Some other -> Bench_args.bad args "unknown --layout=%s (plain|swizzled)" other
  in
  (* The load path each layout must render through: the plain twin's per-lane gathers, the swizzled
     twin's ldmatrix A (its row-major B uses swizzle-aware byte gathers, gh-ocannl-1073). A timing
     whose census disagrees measured some other kernel. *)
  let expected_rendering =
    match layout with
    | Plain -> Ir.C_syntax.Mma_intrinsics
    | Swizzled -> Ir.C_syntax.Mma_intrinsics_ldmatrix
  in
  Utils.settings.output_debug_files_in_build_directory <- true;
  Generated.init ~backend_name:"cuda";
  let m = 8192 and n = 32 and k = 4096 in
  let fa idx = Float.of_int (1 + (((idx.(0) * 17) + (idx.(1) * 13)) % 7 / 3 % 2)) in
  (* B varies along K (gh-ocannl-1190 review): with a K coefficient divisible by 11 every K row of B
     was equal, so a mis-addressed A fragment that permutes K within a row still summed exactly. *)
  let fb idx = Float.of_int (1 + (((idx.(0) * 13) + (idx.(1) * 19)) % 11 / 5 % 2)) in
  let a = NTDSL.init ~l:"probe_a" ~prec:Ir.Ops.fp8 ~i:[ k ] ~o:[ m ] ~f:fa () in
  let b = NTDSL.init ~l:"probe_b" ~prec:Ir.Ops.fp8 ~i:[ n ] ~o:[ k ] ~f:fb () in
  let%op t = a * b in
  Ir.Tnode.update_prec t.Tensor.value Ir.Ops.single;
  let schedule (opt : LL.optimized) =
    let i, j, k =
      match List.find_exn (Ll_test.nest_paths opt.llc) ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = S.expand_zero ~tn:t.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = S.split ~axis:zi ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, ii = S.split ~axis:i ~factor:16 ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, ko, ki = S.split ~axis:k ~factor:32 ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _ = S.tensorize ~i:ii ~j ~k:ki ~simd_width:32 () in
    let stage source tile_loops =
      S.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = Some 32;
          hoisted = false;
          swizzle = (match layout with Plain -> None | Swizzled -> Some LL.Swizzle_b128);
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    S.apply
      [
        ez;
        sp_zi;
        S.Retype { axis = zj; ty = LL.Workgroup };
        sp_i;
        sp_k;
        S.Swap { outer = j; inner = ko };
        S.Swap { outer = ii; inner = ko };
        stage a.Tensor.value [ ii; ki ];
        stage b.Tensor.value [ ki; j ];
        tz;
      ]
      opt
  in
  let name = "mma_register_scope_probe" in
  let comp = Train.forward t in
  let comp = { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment (name, comp.asgns) } in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ schedule opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let mma = routine.Context.mma in
  Stdio.eprintf "backend=cuda layout=%s shape=%dx%dx%d %s\n%!" (layout_name layout) m n k
    (Ir.C_syntax.mma_summary_string mma);
  if
    List.is_empty mma.Ir.C_syntax.renderings
    || not
         (List.for_all mma.renderings ~f:(fun (_, r) ->
              Ir.C_syntax.equal_mma_rendering r expected_rendering))
  then
    failwith
      (Printf.sprintf "layout=%s rendered %s, expected only %s" (layout_name layout)
         (Ir.C_syntax.mma_summary_string mma)
         (Sexp.to_string (Ir.C_syntax.sexp_of_mma_rendering expected_rendering)));
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx t.Tensor.value in
  (* Every cell's exact reference is indexed by the operands' small independent periods; compute
     each residue pair once, then check the entire output. *)
  let reference =
    Array.init (7 * 11) ~f:(fun cell ->
        List.sum
          (module Float)
          (List.range 0 k)
          ~f:(fun l -> fa [| cell / 11; l |] *. fb [| l; cell % 11 |]))
  in
  Array.iteri got ~f:(fun cell v ->
      let want = reference.((cell / n % 7 * 11) + (cell % n % 11)) in
      if not (Float.equal v want) then failwith (Printf.sprintf "probe mismatch at cell %d" cell));
  Stdio.Out_channel.write_all path ~data:(Generated.read name);
  Stdio.printf "generated %s kernel (%s) validated and exported to %s\n%!" (layout_name layout)
    (Ir.C_syntax.mma_summary_string mma)
    path
