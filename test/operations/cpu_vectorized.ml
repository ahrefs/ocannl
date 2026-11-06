(* cc-fp-contract: no-contraction -- each cast leg holds a sum or a product and never both, so no
   multiply-add forms, and the mul-add legs compare within a tolerance *)

(* Explicit SIMD rendering of [Vectorized] loops (gh-ocannl-164 follow-up,
   archive/docs/proposals/watch-ocannl-README-md-347818d3.md): executed parity of vector-rendered
   kernels against their serial twins, plus structural checks on the generated source.

   Covered here: - A plain [Retype ~ty:Vectorized] of the innermost loop of an elementwise mul-add
   kernel, with an extent that is not a lane multiple (517), so the serial remainder loop executes.
   - [Split] then [Retype] of the inner loop: the store/load index vectors carry an [Affine] last
   component (factor * outer + inner), exercising the coefficient-1 contiguity rule.

   On GPU backends (gh-ocannl-463) eligible loops render as 128-bit packed loads/stores
   ([reinterpret_cast] of the aligned pack structs, llm.c's Packed128) instead of vector extensions.
   The 517-column case is ineligible there — a 517-wide last dimension breaks the lane-alignment
   guarantee — and renders serially, while the split 512-column case is eligible (the [Affine]
   coefficient and last dimension 512 are both lane multiples). Every printed boolean holds on every
   backend and at every vector width. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx_tol _ = 1e-4
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The vector width is a per-machine fact (16 NEON, 32 AVX2, 64 AVX-512), so the split factor of the
   second leg is derived from it rather than spelled: a factor below one vector leaves the retyped
   inner loop unable to fill a single vector, and the leg would then assert the absence of the
   rendering it exists to exercise. GPU backends report 0 here and keep the factor at 8 — one packed
   128-bit load is 4 f32 lanes, and 8 is what their eligibility rule was written against. *)
let simd_vector_bytes =
  (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.simd_vector.bytes

let split_factor = max 8 (simd_vector_bytes / Ir.Ops.prec_in_bytes Ir.Ops.single)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The innermost loop of the first top-level nest. *)
let rec innermost_loop (llc : LL.t) : Ir.Indexing.symbol option =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  match llc with
  | LL.Seq (a, b) -> ( match innermost_loop a with Some r -> Some r | None -> innermost_loop b)
  | LL.For_loop { index; body; _ } -> (
      match strip (LL.flat_lines [ body ]) with
      | [ single ] -> ( match innermost_loop single with Some r -> Some r | None -> Some index)
      | _ -> Some index)
  | LL.If { body; _ } -> innermost_loop body
  | _ -> None

let () =
  let rows = 6 and cols = 517 in
  let n = rows * cols in
  let av =
    Array.init n ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus:19 ~offset:0. ~stride:0.5)
  in
  let bv =
    Array.init n
      ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus:23 ~offset:(-11.) ~stride:1.)
  in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ rows; cols ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ rows; cols ] () in
  let run ~name ~transform t =
    let comp = named name (Train.forward t) in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx t.Tensor.value
  in

  (* --- Retype the innermost loop (extent 517: 517 = lanes * k + remainder). --- *)
  let%op c0 = (a *. b) + a in
  let twin = run ~name:"simd_twin" ~transform:(fun opt -> opt) c0 in
  let%op c1 = (a *. b) + a in
  let got =
    run ~name:"simd_retype"
      ~transform:(fun opt ->
        let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
        Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt)
      c1
  in
  p_close "vectorized mul-add values match the serial twin" got twin ~tol:approx_tol;
  (let src = Generated.read "simd_retype" in
   let has s = String.is_substring src ~substring:s in
   let ok =
     if on_cpu then has "vector_size" && has "__builtin_memcpy" && not (has "#pragma")
     else
       (* 517 is not a lane multiple: rows other than the first start at unaligned offsets, so the
          packed rendering must decline and fall back to a plain serial loop. *)
       (not (has "vector_size")) && not (has "reinterpret_cast")
   in
   p "vectorized rendering with serial remainder" ok);

  (* --- A virtual cast twin in lanes (gh-ocannl-1975): the rounding is an expression, so the loop
     vectorizes, and every lane rounds as the serial remainder and a materialized twin do. Values at
     f16's rounding boundaries, ties and signed zeros among them, across the whole 517-lane row and
     its tail. --- *)
  let boundary =
    [|
      1. +. (2. **. -11.);
      1. +. (3. *. (2. **. -11.));
      -1. -. (2. **. -11.);
      -1. -. (3. *. (2. **. -11.));
      1. +. (2. **. -12.);
      -0.;
      2. **. -26.;
      -.(2. **. -26.);
      70000.;
      -70000.;
      0.1;
      -0.3;
      Ll_test.cycle_flat ~dims:[| 1; 17 |] ~modulus:13 ~offset:0. ~stride:0.25 5;
    |]
  in
  let cycled len = Array.init len ~f:(fun cell -> boundary.(cell % Array.length boundary)) in
  let twin_of_master ~materialize ~dims label =
    let values = cycled (Array.fold dims ~init:1 ~f:( * )) in
    let m = TDSL.ndarray values ~label:[ label ] ~output_dims:(Array.to_list dims) () in
    let t = Operation.cast ~grad_spec:Tensor.Prohibit_grad m () in
    Ir.Tnode.update_prec t.Tensor.value Ir.Ops.half;
    (if materialize then Train.set_materialized else Train.set_virtual) t.Tensor.value;
    t
  in
  let same_bits got want =
    Array.equal (fun x y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)) got want
  in
  let retyped opt =
    let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
    Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt
  in
  let in_lanes label ~name ?(dims = [| rows; cols |]) build =
    let result t =
      let c : Tensor.t = build t in
      Ir.Tnode.update_prec c.Tensor.value Ir.Ops.single;
      c
    in
    let want =
      run ~name:(name ^ "_twin") ~transform:Fn.id
        (result (twin_of_master ~materialize:true ~dims "tm"))
    in
    let got =
      run ~name ~transform:retyped (result (twin_of_master ~materialize:false ~dims "tv"))
    in
    p (label ^ ": every lane matches a materialized twin bitwise") (same_bits got want);
    gated ~when_:on_cpu ~on:backend_name
      (label ^ ": the rounding runs in lanes")
      (String.is_substring (Generated.read name) ~substring:"vector_size")
  in
  in_lanes "cast read in an f32 sum" ~name:"simd_cast" (fun t ->
      let%op c = t + a in
      c);
  in_lanes "cast read in an f32 reduction" ~name:"simd_cast_reduce" (fun t ->
      let%op c = t ++ "ab => a" in
      c);
  (* A read the lanes do not vary: it rounds in the scalar render, before the splat. *)
  in_lanes "cast read uniform across the lanes" ~name:"simd_cast_uniform" ~dims:[| rows |] (fun t ->
      let%op c = t +* "a; ab => ab" a in
      c);

  (* --- Split by 8 then retype the inner loop: Affine last components (8*j_o + j_i). --- *)
  let cols2 = 512 in
  let n2 = rows * cols2 in
  let ev =
    Array.init n2
      ~f:(Ll_test.cycle_flat ~dims:[| rows; cols2 |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let fv =
    Array.init n2 ~f:(Ll_test.cycle_flat ~dims:[| rows; cols2 |] ~modulus:11 ~offset:1. ~stride:1.)
  in
  let e = TDSL.ndarray ev ~label:[ "e" ] ~output_dims:[ rows; cols2 ] () in
  let f = TDSL.ndarray fv ~label:[ "f" ] ~output_dims:[ rows; cols2 ] () in
  let%op d0 = e /. f in
  let twin2 = run ~name:"simd_split_twin" ~transform:(fun opt -> opt) d0 in
  let%op d1 = e /. f in
  let got2 =
    run ~name:"simd_split"
      ~transform:(fun opt ->
        let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
        let op, _, inner =
          Sched.split ~axis:j ~factor:split_factor ~outer:LL.Serial ~inner:LL.Serial
        in
        Sched.apply [ op; Sched.Retype { axis = inner; ty = LL.Vectorized } ] opt)
      d1
  in
  p_close "split+vectorized values match the serial twin" got2 twin2 ~tol:approx_tol;
  let src = Generated.read "simd_split" in
  let has s = String.is_substring src ~substring:s in
  p "affine-indexed vector accesses rendered"
    (if on_cpu then has "vector_size" && not (has "#pragma")
     else
       (* Eligible on GPU: dims and the Affine coefficient are lane multiples, so the packed
          rendering emits reinterpret_cast float4_t loads/stores (gh-ocannl-463). *)
       has "reinterpret_cast" && has "float4_t" && not (has "vector_size"))
