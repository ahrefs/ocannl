(* The contract {!Ll_test.cycle} and {!Ll_test.drift} document, pinned (gh-ocannl-639, Codex round 1
   on the extraction PR): the recipe's two halves are conditions, not rules of thumb, and a future
   caller reusing the family gets both checked rather than assumed.

   Blindness is mechanical, so [cycle] raises on it: a modulus dividing an axis's row-major stride
   makes the value constant along that axis, and coprimality with the reduction EXTENT does not rule
   that out — [|2; 4; 3|] with modulus 3 has strides [12; 3; 1], so despite 3 and 4 being coprime
   only the innermost index moves the value.

   Blindness and the value set are separate knobs (ahrefs/ocannl#1024): [~radix] lifts a blind axis
   without touching [~modulus], so a site whose five values are load-bearing keeps them at a size
   the plain row-major key cannot serve, while leaving [~radix] out keeps the arithmetic of the
   hand-written idiom every converted site replaced.

   Exactness is numeric, so it is exhibited here rather than argued: the cells are bf16-exact, some
   partial sum of the reductions accum_width actually builds is NOT (which is what makes per-step
   narrowing visible — a nonzero mean over too few terms would be the zero-mean trap wearing a
   different hat), and every partial sum is f32-exact (which is what lets the f64 host-side
   reference reproduce the widened kernel bitwise). *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let bf16_exact x = Float.equal x (Ir.Ops.bfloat16_to_single (Ir.Ops.single_to_bfloat16 x))
let f32_exact x = Float.equal x (Stdlib.Int32.float_of_bits (Stdlib.Int32.bits_of_float x))

(* The shapes accum_width's drift operands take: the two-axis reduction, its vectorized twin, and
   the 16-wide row the Workgroup_reduce legs sum. *)
let shapes = [ [| 4; 6; 6 |]; [| 4; 4; 32 |]; [| 4; 16 |] ]
let row_axis = [| 4; 16 |]

(* Every multi-index of [dims], in row-major order. *)
let all_indices dims =
  Array.fold dims ~init:[ [||] ] ~f:(fun acc extent ->
      List.concat_map acc ~f:(fun idcs -> List.init extent ~f:(fun j -> Array.append idcs [| j |])))

(* The reduction over [dims]'s trailing axes at outer index 0, as running partial sums. *)
let partials ~dims =
  List.folding_map
    (all_indices (Array.subo dims ~pos:1))
    ~init:0.0
    ~f:(fun acc rest ->
      let sum = acc +. Ll_test.drift ~dims (Array.append [| 0 |] rest) in
      (sum, sum))

(* Whether stepping any one index of [idcs] by one, where [dims] leaves room, moves [f]'s value. *)
let moves_along_every_axis ~dims f idcs =
  Array.for_alli dims ~f:(fun ax extent ->
      idcs.(ax) + 1 >= extent
      ||
      let next = Array.copy idcs in
      next.(ax) <- idcs.(ax) + 1;
      not (Float.equal (f idcs) (f next)))

let refuses f =
  try
    ignore (f ());
    false
  with Invalid_argument _ -> true

(* sketch_family_tree's [wb]: the five integers in [-2, 2] on a 20x20 operand, where 5 divides the
   row stride. *)
let square = [| 20; 20 |]
let wb ?radix idcs = Ll_test.cycle ?radix ~dims:square ~modulus:5 ~offset:(-2.) ~stride:1. idcs

let () =
  p "cycle rejects a modulus blind to an axis, which coprimality with the extent does not rule out"
    (try
       ignore (Ll_test.cycle ~dims:[| 2; 4; 3 |] ~modulus:3 ~offset:1. ~stride:0.5 [| 0; 0; 0 |]);
       false
     with Invalid_argument _ -> true);
  p_all "cycle accepts the shapes the accumulator-width legs use" shapes ~f:(fun dims ->
      Option.is_none (Ll_test.blind_axis ~dims ~modulus:13 ()));
  (* What lets a conversion move no golden: the default key is the row-major offset, so [cycle] is
     the idiom [wa] was written in before sketch_family_tree converted it. *)
  p_all "without a radix, cycle computes the hand-written flat idiom it replaces"
    (all_indices square) ~f:(fun idcs ->
      Float.equal
        (Ll_test.cycle ~dims:square ~modulus:7 ~offset:0. ~stride:0.5 idcs)
        (Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 7) *. 0.5));
  p "without a radix, cycle refuses modulus 5 on the 20x20 operand"
    (refuses (fun () -> wb [| 0; 0 |]));
  p_all "a radix coprime to the modulus lifts the blind axis: every index moves the value"
    (all_indices square)
    ~f:(moves_along_every_axis ~dims:square (wb ~radix:7));
  p "under that radix the modulus still chooses the value set: exactly the five integers in [-2, 2]"
    (let values = List.map (all_indices square) ~f:(wb ~radix:7) in
     List.equal Float.equal
       (List.dedup_and_sort values ~compare:Float.compare)
       [ -2.; -1.; 0.; 1.; 2. ]);
  p_exists "a radix off 1 (mod modulus) keeps the square operand distinct from its transpose"
    (all_indices square) ~f:(fun idcs ->
      not (Float.equal (wb ~radix:7 idcs) (wb ~radix:7 [| idcs.(1); idcs.(0) |])));
  p "a radix sharing a factor with the modulus is refused, however it relates to the dims"
    (refuses (fun () -> wb ~radix:5 [| 0; 0 |]) && refuses (fun () -> wb ~radix:10 [| 0; 0 |]));
  (* Place values are reduced as they are built: 1_000_003^4 overflows an int, and the wrapped power
     must not read as a multiple of the modulus (PR review round 1). *)
  let rank5 = [| 2; 2; 2; 2; 2 |] in
  p_all "a large radix on a high-rank shape is judged by its residue, not an overflowed power"
    (all_indices rank5)
    ~f:
      (moves_along_every_axis ~dims:rank5
         (Ll_test.cycle ~radix:1_000_003 ~dims:rank5 ~modulus:5 ~offset:0. ~stride:1.));
  (* Residue products stay below 2^60 only while the modulus is at most 2^30, so a larger one is
     refused outright rather than risk a wrapped product (PR review round 2). Radix 7 is coprime to
     both moduli and its residues are too small to wrap, so the bound is the only thing that can
     refuse the first and nothing refuses the second. *)
  let at_modulus modulus () =
    Ll_test.cycle ~radix:7 ~dims:[| 2; 2; 2 |] ~modulus ~offset:0. ~stride:1. [| 1; 1; 1 |]
  in
  p "a modulus past Ll_test.max_modulus is refused, and one at it is accepted"
    (refuses (at_modulus (Ll_test.max_modulus + 1))
    && not (refuses (at_modulus Ll_test.max_modulus)));
  (* The two forms a site is written in — [NTDSL.init]'s multi-index and [Array.init]'s flat offset,
     in row-major order — have to mint the same operand once a radix makes the key more than the
     offset itself. *)
  let dims = [| 3; 4; 5 |] in
  p_alli "under a radix, cycle_flat over the row-major offsets is cycle over the multi-indices"
    (all_indices dims) ~f:(fun i idcs ->
      Float.equal
        (Ll_test.cycle ~radix:7 ~dims ~modulus:5 ~offset:0. ~stride:1. idcs)
        (Ll_test.cycle_flat ~radix:7 ~dims ~modulus:5 ~offset:0. ~stride:1. i));
  p_all "drift varies with every index of every shape used" shapes ~f:(fun dims ->
      let base = Array.map dims ~f:(fun _ -> 0) in
      Array.for_alli dims ~f:(fun ax _ ->
          let bumped = Array.copy base in
          bumped.(ax) <- 1;
          not (Float.equal (Ll_test.drift ~dims base) (Ll_test.drift ~dims bumped))));
  p_all "every drift cell is exact in bf16" shapes ~f:(fun dims ->
      List.for_all (all_indices dims) ~f:(fun idcs -> bf16_exact (Ll_test.drift ~dims idcs)));
  (* The fact the accumulator-width legs rest on: the sums leave bf16 exactness within the extents
     they reduce, so an accumulator narrowing per step diverges from one narrowing at the store. *)
  p_all "a drift reduction leaves bf16 exactness within the extents these tests reduce" shapes
    ~f:(fun dims -> List.exists (partials ~dims) ~f:(fun s -> not (bf16_exact s)));
  p_all "every drift partial sum stays exact in f32" shapes ~f:(fun dims ->
      List.for_all (partials ~dims) ~f:f32_exact);
  (* The crossing drift's doc names: the eleventh term of the 16-wide row reaches 275/64, odd and
     above bf16's guaranteed-exact bound of 2^8 units of 1/64. *)
  p "the 16-wide row's first bf16-inexact partial sum is the eleventh, at 275/64"
    (match List.findi (partials ~dims:row_axis) ~f:(fun _ s -> not (bf16_exact s)) with
    | Some (i, s) -> i = 10 && Float.equal s (275.0 /. 64.0)
    | None -> false)
