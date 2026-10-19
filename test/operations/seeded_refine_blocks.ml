(* gh-ocannl-1413: the seeded untuned default's site refinement compares its default segmentation
   with the finer [arity_cuts] one block by block -- a block being a maximal run of units whose two
   ends are boundaries in both -- and takes a block's fine segments when more of them seed.

   Part 1 drives [Autotune.refine_segments] on synthetic segmentations, so each rule is pinned
   independently of what a backend's fission happens to produce: the per-segment refinement of
   gh-ocannl-1324 is the one-coarse-segment block, a coarse segment that seeds is refined when its
   pieces seed more, and a block whose fine cuts MOVE a boundary is refined as a whole.

   Part 2 is that last shape on a real routine: attention's score product [s = q k^T] followed by
   its row maximum. The default GPU segmentation keeps the score product with the maximum's [-inf]
   initialization, a companion that cannot follow the product's arity, so no seed of that segment
   survives; the finer segmentation puts the initialization with the maximum instead, so the score
   product's own segment partitions no default segment. Before the block rule the seeded default
   left it untiled (gpt2_mini_train's four layers' score products). On CUDA/HIP it must now be
   seeded without a fallback, and both outputs must stay bitwise the default pipeline's on
   non-dyadic inputs; elsewhere nothing is seeded. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* Part 1. A segment is [(first unit, kind, units, name)]; a name seeds when [seeding] lists it. *)
let refine ~seeding coarse fine =
  let decided = ref [] in
  let segs, refined =
    Autotune.refine_segments ~coarse
      ~fine:(lazy fine)
      ~seed:(fun () name ->
        decided := name :: !decided;
        Option.some_if (List.mem seeding name ~equal:String.equal) name)
  in
  (List.map segs ~f:(fun (at, (), n, s, name) -> (at, n, Option.is_some s, name)), refined, !decided)

let names segs = List.map segs ~f:(fun (_, _, _, name) -> name)
let seg at n name = (at, (), n, name)

let () =
  (* gh-ocannl-1324's case: an unseeded weight gradient plus bias gradient, one piece seeding. *)
  let segs, refined, _ =
    refine ~seeding:[ "w" ] [ seg 0 2 "w+b" ] (Some [ seg 0 1 "w"; seg 1 1 "b" ])
  in
  p "an unseeded segment is replaced by its pieces where one of them seeds"
    (List.equal String.equal (names segs) [ "w"; "b" ] && refined = 1);
  (* A projection's weight gradient seeds with the data gradients as companions; cut apart, every
     product seeds. *)
  let segs, refined, _ =
    refine
      ~seeding:[ "wk+x+wq+x"; "wk"; "x1"; "wq"; "x2" ]
      [ seg 0 4 "wk+x+wq+x" ]
      (Some [ seg 0 1 "wk"; seg 1 1 "x1"; seg 2 1 "wq"; seg 3 1 "x2" ])
  in
  p "a seeded segment is replaced by its pieces where more of them seed"
    (List.equal String.equal (names segs) [ "wk"; "x1"; "wq"; "x2" ] && refined = 1);
  p_all "each of its pieces carries its seed" segs ~f:(fun (_, _, s, _) -> s);
  let segs, refined, _ =
    refine ~seeding:[ "w+b"; "w" ] [ seg 0 2 "w+b" ] (Some [ seg 0 1 "w"; seg 1 1 "b" ])
  in
  p "a seeded segment whose pieces seed no more keeps its launch"
    (List.equal String.equal (names segs) [ "w+b" ] && refined = 0);
  (* The moved boundary: [(qk, init) (max)] against [(qk) (init, max)], with a coarse segment on
     either side that both segmentations share. *)
  let segs, refined, _ =
    refine ~seeding:[ "qk" ]
      [ seg 0 1 "pre"; seg 1 2 "qk+init"; seg 3 1 "max"; seg 4 1 "post" ]
      (Some [ seg 0 1 "pre"; seg 1 1 "qk"; seg 2 2 "init+max"; seg 4 1 "post" ])
  in
  p "a block whose fine cuts move a boundary takes its fine segments"
    (List.equal String.equal (names segs) [ "pre"; "qk"; "init+max"; "post" ] && refined = 2);
  p "the refined segmentation still covers every unit once, in order"
    (List.equal
       (fun (a1, n1) (a2, n2) -> a1 = a2 && n1 = n2)
       (List.map segs ~f:(fun (at, n, _, _) -> (at, n)))
       [ (0, 1); (1, 1); (2, 2); (4, 1) ]);
  let segs, refined, _ =
    refine ~seeding:[]
      [ seg 0 2 "qk+init"; seg 2 1 "max" ]
      (Some [ seg 0 1 "qk"; seg 1 2 "init+max" ])
  in
  p "a block whose fine segments seed no more keeps its coarse segments"
    (List.equal String.equal (names segs) [ "qk+init"; "max" ] && refined = 0);
  let segs, refined, decided =
    refine ~seeding:[ "a" ] [ seg 0 1 "a"; seg 1 1 "b" ] (Some [ seg 0 1 "a'"; seg 1 1 "b'" ])
  in
  p "a block both segmentations cut alike is the coarse segment, decided once"
    (List.equal String.equal (names segs) [ "a"; "b" ]
    && refined = 0
    && List.equal String.equal (List.sort decided ~compare:String.compare) [ "a"; "b" ]);
  let segs, refined, _ = refine ~seeding:[ "x" ] [ seg 0 2 "ab" ] (Some [ seg 0 1 "x" ]) in
  p "a finer segmentation covering other units is ignored"
    (List.equal String.equal (names segs) [ "ab" ] && refined = 0);
  let segs, refined, _ = refine ~seeding:[] [ seg 0 2 "ab" ] None in
  p "no finer segmentation leaves the coarse one"
    (List.equal String.equal (names segs) [ "ab" ] && refined = 0)

(* Part 2. *)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let seeds_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let nb = 4
and ns = 64
and nh = 2
and nd = 32

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let q_values = values ~n:(nb * ns * nh * nd) ~stride:389
let k_values = values ~n:(nb * ns * nh * nd) ~stride:577
let q = NTDSL.ndarray q_values ~label:[ "rb_q" ] ~output_dims:[ nb; ns; nh; nd ] ()
let k = NTDSL.ndarray k_values ~label:[ "rb_k" ] ~output_dims:[ nb; ns; nh; nd ] ()

(* The fixture's order sensitivity, the control that gives the bitwise claims below their teeth: the
   scores reduced on the host in f32, over [d] in REVERSE order. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let scores_reversed =
  let out = Array.create ~len:(nb * ns * nh * ns) 0. and cell = ref 0 in
  for b = 0 to nb - 1 do
    for s = 0 to ns - 1 do
      for h = 0 to nh - 1 do
        for t = 0 to ns - 1 do
          let row s = ((((b * ns) + s) * nh) + h) * nd in
          let acc = ref 0. in
          for d = nd - 1 downto 0 do
            acc := f32 (!acc +. f32 (f32 q_values.(row s + d) *. f32 k_values.(row t + d)))
          done;
          out.(!cell) <- !acc;
          Int.incr cell
        done
      done
    done
  done;
  out

let%op scores = q +* "bshd;bthd=>bsht" k
let%op row_max = scores @^^ "bsht=>bsh"

let () =
  Train.set_materialized scores.Tensor.value;
  Train.set_materialized row_max.Tensor.value

let comp = Train.forward row_max
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx scores.Tensor.value, Context.get_values ctx row_max.Tensor.value)

let () =
  let s_default, m_default = run Context.compile in
  let report = ref None in
  let s_seeded, m_seeded =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_reduction:nd
          ~report:(fun r -> report := Some r)
          ctx comp bindings)
  in
  let r = Option.value_exn ~here:[%here] !report in
  List.iter r.Autotune.sd_seeded ~f:(fun (at, p) ->
      Stdio.eprintf "seeded: unit %d %s (not part of the golden)\n" at
        (Autotune.sketch_params_label p));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the score product is seeded, its segment moved off the maximum's initialization"
    (List.length r.sd_seeded = 1 && r.sd_refined >= 1);
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded routine compiles without falling back to the default"
    (not r.sd_fell_back);
  Verdict.gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the seeded default seeds nothing"
    (List.is_empty r.sd_seeded && r.sd_refined = 0);
  p_exists "the scores are not trivially zero" (Array.to_list s_default) ~f:(fun v ->
      Float.(abs v > 1e-3));
  p_exists "the fixture is order-sensitive: a reversed f32 reduction differs from it in some cell"
    (List.zip_exn (Array.to_list scores_reversed) (Array.to_list s_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  p_all2 "the seeded scores are bitwise the default pipeline's" s_seeded s_default ~f:bitwise;
  p_all2 "the seeded row maxima are bitwise the default pipeline's" m_seeded m_default ~f:bitwise
