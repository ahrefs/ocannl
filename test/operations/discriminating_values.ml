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
     refused outright rather than risk a wrapped product (PR review round 2). Over [|1; 8|] radix 3
     is coprime to both moduli and reaches exactly the eight residues the row-major key does, so the
     bound is the only thing that can refuse the first and nothing refuses the second. *)
  let at_modulus modulus () =
    Ll_test.cycle ~radix:3 ~dims:[| 1; 8 |] ~modulus ~offset:0. ~stride:1. [| 0; 7 |]
  in
  p "a modulus past Ll_test.max_modulus is refused, and one at it is accepted"
    (refuses (at_modulus (Ll_test.max_modulus + 1))
    && not (refuses (at_modulus Ll_test.max_modulus)));
  (* Coprimality alone does not keep the value set: over [|2; 2; 3|] under modulus 6 the row-major
     key reaches all six residues, while radix 5 — coprime, not blind — reaches {0, 1, 2, 3, 5}.
     cycle refuses that rather than silently drop a value (PR review round 3). *)
  p "a radix that would drop a value the row-major key reaches is refused"
    (refuses (fun () ->
         Ll_test.cycle ~radix:5 ~dims:[| 2; 2; 3 |] ~modulus:6 ~offset:0. ~stride:1. [| 0; 0; 0 |]));
  (* Base's [%] is non-negative for a positive modulus, so a negative radix is its residue. *)
  let sq5 = [| 5; 5 |] in
  p_all "a negative radix mints the operand its residue does: radix -2 is radix 3 under modulus 5"
    (all_indices sq5) ~f:(fun idcs ->
      Float.equal
        (Ll_test.cycle ~radix:(-2) ~dims:sq5 ~modulus:5 ~offset:0. ~stride:1. idcs)
        (Ll_test.cycle ~radix:3 ~dims:sq5 ~modulus:5 ~offset:0. ~stride:1. idcs));
  (* The two forms a site is written in — [NTDSL.init]'s multi-index and [Array.init]'s flat offset,
     in row-major order — have to mint the same operand once a radix makes the key more than the
     offset itself. *)
  let dims = [| 3; 4; 5 |] in
  p_alli "under a radix, cycle_flat over the row-major offsets is cycle over the multi-indices"
    (all_indices dims) ~f:(fun i idcs ->
      Float.equal
        (Ll_test.cycle ~radix:7 ~dims ~modulus:5 ~offset:0. ~stride:1. idcs)
        (Ll_test.cycle_flat ~radix:7 ~dims ~modulus:5 ~offset:0. ~stride:1. i));
  (* [weighted] is what the conv-shaped fixtures converted onto (gh-ocannl-1018): same arithmetic as
     the idiom they were written in, so no golden moves, and the same guard [cycle] applies. *)
  let conv = [| 2; 3; 3; 4 |] in
  p_all "weighted computes the hand-written conv-fixture idiom it replaces" (all_indices conv)
    ~f:(fun idcs ->
      Float.equal
        (Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1. idcs)
        (Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7)));
  p "weighted refuses a weight that is a multiple of the modulus"
    (refuses (fun () ->
         Ll_test.weighted ~weights:[| 1; 14 |] ~modulus:7 ~offset:0. ~stride:1. [| 0; 0 |]));
  p "weighted refuses a weight vector of the wrong rank"
    (refuses (fun () ->
         Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:7 ~offset:0. ~stride:1. [| 0; 0; 0 |]));
  let cube = [| 3; 3; 3 |] in
  p_all "with no weight a multiple of the modulus, weighted varies with every index"
    (all_indices cube) ~f:(fun idcs ->
      moves_along_every_axis ~dims:cube
        (Ll_test.weighted ~weights:[| 1; -1; 2 |] ~modulus:5 ~offset:0. ~stride:1.)
        idcs);
  p_all "a negative weight still lands in the value set" (all_indices cube) ~f:(fun idcs ->
      let v = Ll_test.weighted ~weights:[| 1; -1; 2 |] ~modulus:5 ~offset:0. ~stride:1. idcs in
      Float.(v >= 0. && v <= 4.));
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

(* --- {!Ll_test.exact_partials}, the HIP bitwise legs' derived premise (gh-ocannl-1693). Its
   soundness is checked against an independent oracle rather than against its own arithmetic: every
   partial sum of any association is the sum of some subset of the products, so where it accepts a
   cell, every subset sum must be representable at the width it was asked about. *)

(* Whether [x] is representable with [bits] significant bits, exponent range aside. *)
let representable ~bits x =
  Float.(x = 0.)
  ||
  let m, _ = Float.frexp x in
  Float.is_integer (Float.ldexp m bits)

let subset_sums cell =
  List.fold cell ~init:[ 0. ] ~f:(fun sums x -> sums @ List.map sums ~f:(fun s -> s +. x))

let widths = [ 2; 3; 4; 5; 6 ]

(* Small cells of one sign at several grains (powers of two, and 3/8 and 3/2, whose grains are finer
   than their values), some holding zeros: enough of them straddle every width's bound. *)
let cells =
  List.concat_map (List.range 1 9) ~f:(fun len ->
      List.concat_map [ 0.125; 0.375; 0.5; 1.5; 3. ] ~f:(fun stride ->
          List.concat_map [ 3; 5; 7 ] ~f:(fun modulus ->
              List.concat_map [ 1.; -1. ] ~f:(fun sign ->
                  List.map [ 0; 1 ] ~f:(fun lift ->
                      List.init len ~f:(fun t ->
                          sign *. Float.of_int ((t * 2 % modulus) + lift) *. stride))))))

let () =
  let accepts ~bits cell = Ll_test.exact_partials ~sig_bits:bits cell in
  p_all
    "exact_partials is sound: where it accepts a cell, every partial sum of every association is \
     exact at that width" (List.cartesian_product widths cells) ~f:(fun (bits, cell) ->
      (not (accepts ~bits cell))
      || List.for_all (subset_sums cell) ~f:(fun s -> representable ~bits s));
  p_all "exact_partials accepts some of those cells, and refuses some, at every width" widths
    ~f:(fun bits ->
      List.exists cells ~f:(accepts ~bits) && List.exists cells ~f:(fun c -> not (accepts ~bits c)));
  p_all
    "exact_partials' bound is tight: it accepts 2^p grains and refuses 2^p + 1, whose sum is \
     inexact at p bits"
    [ 3; 5; 8; 11; 24 ] ~f:(fun bits ->
      let top = Float.ldexp 1. bits in
      accepts ~bits [ top -. 1.; 1. ]
      && (not (accepts ~bits [ top; 1. ]))
      && not (representable ~bits (top +. 1.)));
  p
    "exact_partials refuses mixed signs even where every partial sum is exact (gfx11's WMMA \
     premise)"
    ((not (accepts ~bits:24 [ 1.; -1. ])) && not (accepts ~bits:24 [ 0.5; 0.25; -0.125 ]));
  p "exact_partials reads zeros as signless, and an empty or all-zero cell as exact"
    (accepts ~bits:8 [ 1.; 0.; 2. ]
    && accepts ~bits:8 [ -1.; -0.; -2. ]
    && accepts ~bits:8 []
    && accepts ~bits:8 [ 0.; -0. ]);
  p "exact_partials refuses a non-finite product"
    ((not (accepts ~bits:24 [ 1.; Float.infinity ])) && not (accepts ~bits:24 [ Float.nan ]));
  p "exact_partials refuses a width outside 1 .. 52, and takes both ends"
    (refuses (fun () -> accepts ~bits:0 [ 1. ])
    && refuses (fun () -> accepts ~bits:53 [ 1. ])
    && accepts ~bits:1 [ 1.; 1. ]
    && accepts ~bits:52 [ 1. ]);
  p "grain_exponent is the weight of the lowest set bit, and none for zero"
    (List.equal (Option.equal Int.equal)
       (List.map [ 0.375; 6.; 1.; -0.75; 0x1p-20; 0.; Float.nan ] ~f:Ll_test.grain_exponent)
       [ Some (-3); Some 1; Some 0; Some (-2); Some (-20); None; None ]);
  (* The ledger's own operands: [schedule_mma_matmul]'s bf16 cycles pass at bf16, while the same B
     cycle recentred on zero (values -1 .. 1), the shape of the signed operands gh-ocannl-1432
     retired, is refused for its sign alone — it still passes the bound. *)
  let n = 32 in
  let a = Ll_test.cycle ~dims:[| n; n |] ~modulus:3 ~offset:0. ~stride:0.25 in
  let b ~offset = Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset ~stride:0.5 in
  let products ~offset = Ll_test.matmul_products ~m:n ~n ~k:n a (b ~offset) in
  p_all "schedule_mma_matmul's bf16 operands keep every cell exact in bf16" (products ~offset:0.)
    ~f:(accepts ~bits:8);
  p_exists "their cycle recentred on zero is refused at bf16" (products ~offset:(-2.)) ~f:(fun c ->
      not (accepts ~bits:8 c));
  p_all "and it is the sign that refuses it: the magnitudes still fit bf16" (products ~offset:(-2.))
    ~f:(fun c -> accepts ~bits:8 (List.map c ~f:Float.abs))
