(* SIMD reduction rendering of [Vectorized] accumulation loops (gh-ocannl-468, ggml's
   ggml_vec_dot_f32 idiom): executed parity of chain-rendered reductions against serial twins /
   OCaml-computed references, plus structural checks on the generated source.

   A [Vectorized] retype of a loop whose body is a single accumulation statement [acc = op(acc,
   contrib(i))] renders on the CPU backends as independent vector accumulator chains updated in a
   fused main loop, folded register- and lane-wise at exit, plus a serial tail — the strict-FP
   reassociation the retype licenses, so values are compared with a tolerance. On GPU backends the
   vector style is packed loads, which has no reduction form: the accumulation renders as a plain
   serial loop, and every printed boolean holds on every backend.

   Covered: - [Sched.Retype ~ty:Vectorized] of the innermost loop of a real lowered sum (extent 517:
   not a block multiple, so the serial tail executes). - A hand-built FMA-form dot-product
   accumulation loop (the recognizer's Ternop form). - A hand-built max-reduce: a non-Add combine
   with no identity constant (chains initialize from the first blocks of contributions). - A
   hand-built strided (non-contiguous) accumulation: ineligible for the chain rendering, it must
   fall back to a plain serial loop with NO vectorization pragma — the pragma would assert iteration
   independence that the loop-carried accumulation does not satisfy. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let single = Ir.Ops.single

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The innermost loop of the first top-level nest. *)
let rec innermost_loop (llc : LL.t) : Idx.symbol option =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  match llc with
  | LL.Seq (a, b) -> ( match innermost_loop a with Some r -> Some r | None -> innermost_loop b)
  | LL.For_loop { index; body; _ } -> (
      match strip (LL.flat_lines [ body ]) with
      | [ single ] -> ( match innermost_loop single with Some r -> Some r | None -> Some index)
      | _ -> Some index)
  | LL.If { body; _ } -> innermost_loop body
  | _ -> None

(* Replace the lowered serial reduction with a single [Vectorized] accumulation loop over a fresh
   index; the renderer owns the chains. The transform drops the lowered [Zero_out] of the
   accumulator, so un-mark [zero_initialized_by_code]: allocation then zeroes the buffer, giving the
   accumulation the same all-zeros starting point as the serial lowering. *)
let reduce_transform ~n ~body_of (s : Tn.t) (opt : LL.optimized) : LL.optimized =
  (LL.get_node opt.traced_store s).LL.zero_initialized_by_code <- false;
  let i = Idx.get_symbol () in
  {
    opt with
    llc = LL.For_loop { index = i; from_ = 0; to_ = n - 1; body = body_of i; axis = Vectorized };
  }

let run ~name ~transform t =
  let comp = named name (Train.forward t) in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx t.Tensor.value).(0)

let it i = Idx.Iterator i
let f0 = Idx.Fixed_idx 0

let check_generated ~expect_chains name =
  let src = Generated.read name in
  let has sub = String.is_substring src ~substring:sub in
  if on_cpu then
    Bool.equal expect_chains (has "Vectorized reduction rendering") && not (has "#pragma")
  else (not (has "Vectorized reduction rendering")) && not (has "vector_size")

let () =
  (* --- Retype the innermost loop of a real lowered sum (extent 517: serial tail executes). --- *)
  let n = 517 in
  let vv = Array.init n ~f:(fun k -> (Float.of_int (k % 21) *. 0.25) -. 2.) in
  let expected_sum = Array.fold vv ~init:0. ~f:( +. ) in
  let v = TDSL.ndarray vv ~label:[ "v" ] ~output_dims:[ n ] () in
  let%op s0 = v ++ "i=>0" in
  let got_serial = run ~name:"red_sum_serial" ~transform:(fun opt -> opt) s0 in
  p "serial sum correct" (approx got_serial expected_sum);
  let%op s1 = v ++ "i=>0" in
  let got =
    run ~name:"red_sum_simd"
      ~transform:(fun opt ->
        let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
        Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt)
      s1
  in
  p "vectorized sum matches the serial twin" (approx got expected_sum);
  p "sum renders as accumulator chains (CPU) or serially (GPU)"
    (check_generated ~expect_chains:true "red_sum_simd");

  (* --- FMA-form dot product (the recognizer's Ternop form), extent 100. --- *)
  let m = 100 in
  let av = Array.init m ~f:(fun k -> (Float.of_int (k % 7) *. 0.5) -. 1.) in
  let bv = Array.init m ~f:(fun k -> Float.of_int (k % 5) -. 2.) in
  let expected_dot = Array.fold2_exn av bv ~init:0. ~f:(fun acc a b -> acc +. (a *. b)) in
  let va = TDSL.ndarray av ~label:[ "va" ] ~output_dims:[ m ] () in
  let vb = TDSL.ndarray bv ~label:[ "vb" ] ~output_dims:[ m ] () in
  let%op d1 = va +* "i;i=>0" vb in
  let got_dot =
    run ~name:"red_dot_simd"
      ~transform:
        (reduce_transform ~n:m d1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = d1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Ternop
                     ( Ir.Ops.FMA,
                       (Get (va.Tensor.value, [| it i |]), single),
                       (Get (vb.Tensor.value, [| it i |]), single),
                       (Get (d1.Tensor.value, [| f0 |]), single) );
                 debug = "";
               }))
      d1
  in
  p "vectorized fma dot parity" (approx got_dot expected_dot);
  p "dot renders as accumulator chains (CPU) or serially (GPU)"
    (check_generated ~expect_chains:true "red_dot_simd");

  (* --- Max-reduce, extent 40: non-Add combine, chains initialize from the first blocks. The max is
     positive, so the allocation-zeroed accumulator start does not affect the result. --- *)
  let q = 40 in
  let wv = Array.init q ~f:(fun k -> Float.of_int (k * 13 % 29) -. 5.) in
  let expected_max = Array.fold wv ~init:Float.neg_infinity ~f:Float.max in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ q ] () in
  let%op x1 = w @^^ "i=>0" in
  let got_max =
    run ~name:"red_max_simd"
      ~transform:
        (reduce_transform ~n:q x1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = x1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Max,
                       (Get (x1.Tensor.value, [| f0 |]), single),
                       (Get (w.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      x1
  in
  p "vectorized max parity" (approx got_max expected_max);
  p "max renders as accumulator chains (CPU) or serially (GPU)"
    (check_generated ~expect_chains:true "red_max_simd");

  (* --- Max/Min over data that is almost all NaN, extent 40 (gh-ocannl-649). The combine is no
     longer a per-lane [fmaxf] call but a whole-vector compare-and-select (on aarch64, one
     [FMAXNM]), so its NaN behaviour is this code's responsibility rather than libm's -- and no
     structural check on the emitted loop can see it, which is why this leg executes.

     A NaN cannot SURVIVE the rendering whatever the combine does: the lane fold and the serial tail
     both go through the scalar [fmaxf], which drops it. What a wrong combine does is DESTROY the
     extremum on its way past. [fmax(x, NaN)] is [x], but the obvious [x >= NaN ? x : NaN] is NaN,
     so a chain or lane holding the maximum is overwritten the first time a NaN arrives as the
     second operand, and the scalar fold then discards that NaN and answers with whatever was left.

     Hence the data: every element NaN except [sv.(0)], so the extremum is destroyed under a wrong
     combine no matter how the width ladder picks [lanes] and [chains] on this host -- index 0 is
     lane 0 of chain 0 under every decomposition, and it is never in the serial tail. The
     accumulator starts at allocation-zero, so a destroyed extremum reads as 0 and a preserved one
     as the planted value; both are exact, so [Float.equal], not [approx]. --- *)
  let q2 = 40 in
  let nan_reduce ~name ~op ~planted =
    let sv = Array.create ~len:q2 Float.nan in
    sv.(0) <- planted;
    let sn = TDSL.ndarray sv ~label:[ "sn_" ^ name ] ~output_dims:[ q2 ] () in
    let%op acc = sn @^^ "i=>0" in
    run ~name
      ~transform:
        (reduce_transform ~n:q2 acc.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = acc.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( op,
                       (Get (acc.Tensor.value, [| f0 |]), single),
                       (Get (sn.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      acc
  in
  p "max keeps the extremum past a NaN operand, as fmax does"
    (Float.equal (nan_reduce ~name:"red_nanmax_simd" ~op:Ir.Ops.Max ~planted:7.5) 7.5);
  p "min keeps the extremum past a NaN operand, as fmin does"
    (Float.equal (nan_reduce ~name:"red_nanmin_simd" ~op:Ir.Ops.Min ~planted:(-9.25)) (-9.25));

  (* --- Strided (non-contiguous) accumulation: ineligible for chains, must run as a plain serial
     loop without a vectorization pragma. Sums the even-indexed elements of [u]. --- *)
  let r = 24 in
  let uv = Array.init (2 * r) ~f:(fun k -> (Float.of_int (k % 11) *. 0.125) -. 0.5) in
  let expected_strided =
    Array.foldi uv ~init:0. ~f:(fun k acc x -> if k % 2 = 0 then acc +. x else acc)
  in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ 2 * r ] () in
  let%op y1 = u ++ "i=>0" in
  let got_strided =
    run ~name:"red_strided_simd"
      ~transform:
        (reduce_transform ~n:r y1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = y1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Add,
                       (Get (y1.Tensor.value, [| f0 |]), single),
                       (Get (u.Tensor.value, [| Idx.affine ~symbols:[ (2, i) ] ~offset:0 |]), single)
                     );
                 debug = "";
               }))
      y1
  in
  p "strided accumulation parity (serial fallback)" (approx got_strided expected_strided);
  p "strided accumulation declines chains and emits no pragma"
    (check_generated ~expect_chains:false "red_strided_simd")
