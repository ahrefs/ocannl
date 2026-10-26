(* gh-ocannl-1553: every statement that declares a fixed-name C temporary renders as a block of its
   own, so two of them side by side in one C scope compile.

   gh-ocannl-1498 found the failure this pins: [Set_from_vec] declared its vector temporary as
   [vec_result] and opened a block around it only when its argument had local definitions, so two
   packed draws in one scope redeclared it and the routine did not compile. That arm is now braced
   on every path. An audit of [C_syntax] and the four backends' overrides found every other
   fixed-name temporary either braced by the renderer that declares it, on every path, or declared
   once per kernel (a kernel parameter, the pool or log prologue). Each statement family that
   declares one is built here twice, as siblings in one block, compiled on the selected backend and
   executed against values computed on the host.

   The routine-logging arms of [Set] and [Set_dynamic] ([new_set_v], also one of each beside each
   other), of [Set_local] ([new_set_local_v]) and of [Set_from_vec] ([vec_result]). The plain
   [Set_from_vec] arm ([vec_result], the original failure; [simplify_rng_operands] keeps its own leg
   for the gh-ocannl-1498 shape that exposed it). Two [Vectorized] loops of each kind, elementwise
   and reducing, whose temporaries are named by counters that restart per loop ([vset1__],
   [vred_*]). Two [Tile_mma] statements in one lane body: the register tile's [tmma_*] names on the
   C backends, a GPU intrinsic arm's [__mma_*] names.

   The executable runs twice. Without routine logging it covers the families that render only when
   logging is off (vectorized loops and the register tile decline under logging); the second run,
   [--ocannl_debug_log_from_routines=true --ocannl_log_level=2], covers the logging arms. Each leg
   first claims that the shape under test survived optimization (two of the statements are direct
   siblings in one block), so a pass that split or folded them cannot make a leg pass vacuously.

   GPU backends: the CPU vector rendering is the C backends'. There a vectorized loop renders packed
   or serial; the leg still compiles and executes, and its claims about the rendering are skipped.
   The tile leg reads which way each tile went from the routine's mma census on every backend. The
   logged leg executes on every backend, Metal included: it is the witness that Metal routine
   logging no longer crashes the host (gh-ocannl-1594). *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Sched = Ir.Schedule
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let logged = Utils.debug_log_from_routines ()
let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()
let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let get = Ll_test.get

(* The largest number of statements satisfying [f] that are direct siblings in one block of [llc]:
   the top level, a loop body, a guarded body. The renderer gives each of these one C scope, which
   is the scope two fixed-name declarations would collide in. *)
let rec siblings ~f (llc : LL.t) =
  let stmts = LL.flat_lines [ llc ] in
  let here = List.count stmts ~f in
  List.fold stmts ~init:here ~f:(fun acc stmt ->
      match stmt with
      | LL.For_loop { body; _ } | LL.If { body; _ } -> Int.max acc (siblings ~f body)
      | _ -> acc)

let is_set = function LL.Set _ -> true | _ -> false
let is_set_dynamic = function LL.Set_dynamic _ -> true | _ -> false
let is_set_local = function LL.Set_local _ -> true | _ -> false
let is_set_from_vec = function LL.Set_from_vec _ -> true | _ -> false
let is_tile = function LL.Tile_mma _ -> true | _ -> false
let is_vectorized = function LL.For_loop { axis = LL.Vectorized; _ } -> true | _ -> false

let two_siblings what ~f (o : LL.optimized) =
  p
    ("two " ^ what ^ " are siblings in one block of the compiled routine")
    (siblings ~f o.LL.llc >= 2)

(* Distinct small integers, so every expected value below is exact in single precision whatever
   order a backend sums in. *)
let input n = Array.init n ~f:(fun i -> Float.of_int ((i * 5 % 11) - 4))
let read2 = function [ x; y ] -> (x, y) | _ -> assert false
let read3 = function [ x; y; z ] -> (x, y, z) | _ -> assert false

(* [what]'s routine compiled and run, claimed on every backend; returns the values read back. *)
let compile_and_run what ~name o ~seed ~read =
  (* No capture window around the run (gh-ocannl-1636): CUDA and HIP routine logs are device-side
     [printf], and under the [debug_log_to_stream_files] this leg asks for, the backend routes them
     to the stream files itself, so stdout stays the golden's data channel. A line of them reaching
     stdout fails the golden diff. Metal's logs are drained at synchronization from a C-side buffer
     (gh-ocannl-1594). *)
  let ctx, _routine = Ll_test.run_routine ~name o ~seed in
  let values = List.map read ~f:(Context.get_values ctx) in
  p (what ^ ": the routine compiles with them in one scope") true;
  values

(* {1 Logging arms} *)

let n = 8

(* Two [Set]s in one loop body: under logging each declares [new_set_v]. *)
let check_logged_sets ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "sets_a" and x = mk "sets_x" and y = mk "sets_y" in
  List.iter [ a; x; y ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n
      (Ll_test.seq
         (Ll_test.set_at x at (Ll_test.add (get a [| at |]) (Ll_test.c 1.)))
         (Ll_test.set_at y at (Ll_test.mul (get a [| at |]) (Ll_test.c 2.))))
  in
  let name = "fnt_sets" in
  let o = Ll_test.optimize ~materialized:[ a; x; y ] ~name llc in
  two_siblings "logged Sets" ~f:is_set o;
  let av = input n in
  let values = "logged Sets: both stores write their values" in
  compile_and_run "logged Sets" ~name o
    ~seed:[ (a, av); (x, Ll_test.blank n); (y, Ll_test.blank n) ]
    ~read:[ x; y ]
  |> fun got ->
  let vx, vy = read2 got in
  p_all2 values (Array.append vx vy)
    (Array.append (Array.map av ~f:(fun v -> v +. 1.)) (Array.map av ~f:(fun v -> v *. 2.)))
    ~f:Float.equal

(* Two [Set_dynamic]s and a [Set] in one loop body: under logging all three declare [new_set_v]. The
   rows come from a reversing permutation stored in a node, so each scatter lands somewhere its loop
   index alone does not say. *)
let check_logged_dynamic ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "dyn_a" and rows = mk "dyn_rows" and x = mk "dyn_x" and y = mk "dyn_y" in
  let z = mk "dyn_z" in
  List.iter [ a; rows; x; y; z ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let row = (get rows [| at |], Ops.single) in
  let scatter tn v = Ll_test.scatter ~tn ~idcs:[| Ll_test.fixed 0 |] ~dyn_axis:0 ~dyn_value:row v in
  let body ~store =
    Ll_test.loop_n i n
      (Ll_test.seq
         (store x (Ll_test.add (get a [| at |]) (Ll_test.c 1.)))
         (Ll_test.seq
            (store y (Ll_test.mul (get a [| at |]) (Ll_test.c 2.)))
            (Ll_test.set_at z at (Ll_test.add (get a [| at |]) (Ll_test.c 3.)))))
  in
  (* The analyses refuse a [Set_dynamic], so the scatters reach the backend through the scoped seam,
     over a twin that writes the same nodes at the loop index and reads the rows. *)
  let raw = body ~store:(fun tn v -> Ll_test.set_at tn at (Ll_test.add v (get rows [| at |]))) in
  let name = "fnt_dynamic" in
  let o =
    Ll_test.optimize_scoped ~materialized:[ a; rows; x; y; z ] ~name ~raw (body ~store:scatter)
  in
  two_siblings "logged Set_dynamics" ~f:is_set_dynamic o;
  p "a logged Set is a sibling of the logged Set_dynamics"
    (siblings ~f:(fun s -> is_set s || is_set_dynamic s) o.LL.llc >= 3);
  let av = input n in
  let reversed = Array.init n ~f:(fun i -> Float.of_int (n - 1 - i)) in
  let at_row f = Array.init n ~f:(fun r -> f av.(n - 1 - r)) in
  let values = "logged Set_dynamics and a Set: all three write their values" in
  compile_and_run "logged Set_dynamics and a Set" ~name o
    ~seed:
      [
        (a, av); (rows, reversed); (x, Ll_test.blank n); (y, Ll_test.blank n); (z, Ll_test.blank n);
      ]
    ~read:[ x; y; z ]
  |> fun got ->
  let vx, vy, vz = read3 got in
  p_all2 values
    (Array.concat [ vx; vy; vz ])
    (Array.concat
       [ at_row (fun v -> v +. 1.); at_row (fun v -> v *. 2.); Array.map av ~f:(fun v -> v +. 3.) ])
    ~f:Float.equal

(* Two [Set_local]s in one loop body: under logging each declares [new_set_local_v]. Optimization
   would fold single-assignment locals away, so the scoped form replaces the optimized twin through
   [Ll_test.optimize_scoped] and reaches the backend as written. *)
let check_logged_locals ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "loc_a" and x = mk "loc_x" in
  let one = mk "loc_one" and two = mk "loc_two" in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  List.iter [ one; two ] ~f:Ll_test.virtualize;
  let id1 = LL.get_scope one and id2 = LL.get_scope two in
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let v1 = Ll_test.add (get a [| at |]) (Ll_test.c 1.) in
  let v2 = Ll_test.mul (get a [| at |]) (Ll_test.c 2.) in
  let raw = Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add v1 v2)) in
  let scoped =
    Ll_test.loop_n i n
      (LL.unflat_lines
         (Ll_test.define_local id1 v1 @ Ll_test.define_local id2 v2
         @ [ Ll_test.set_at x at (Ll_test.add (LL.Get_local id1) (LL.Get_local id2)) ]))
  in
  let name = "fnt_locals" in
  let o = Ll_test.optimize_scoped ~materialized:[ a; x ] ~name ~raw scoped in
  two_siblings "logged Set_locals" ~f:is_set_local o;
  let av = input n in
  let values = "logged Set_locals: both carry their values" in
  compile_and_run "logged Set_locals" ~name o ~seed:[ (a, av); (x, Ll_test.blank n) ] ~read:[ x ]
  |> function
  | [ vx ] -> p_all2 values vx (Array.map av ~f:(fun v -> (3. *. v) +. 1.)) ~f:Float.equal
  | _ -> assert false

(* {1 Set_from_vec, logged or not} *)

let threefry a b = LL.Binop (Ops.Threefry4x32_light, a, b)

(* Two packed draws straight-line in one scope, each declaring [vec_result]. One key and counter, so
   both draw the same four values. *)
let check_two_packed_draws ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| 4 |] () in
  let a = mk "packed_a" and b = mk "packed_b" in
  List.iter [ a; b ] ~f:Ll_test.materialize;
  let block () =
    ( threefry
        (threefry (LL.Constant_bits 7L, u4) (LL.Constant_bits 42L, u4), u4)
        (LL.Embed_index (Idx.Fixed_idx 0), iprec),
      u4 )
  in
  let draw tn =
    Ll_test.set_from_vec ~length:4 ~vec_unop:Ops.Uint4x32_to_prec_uniform tn [| Idx.Fixed_idx 0 |]
      (block ())
  in
  let name = if logged then "fnt_packed_logged" else "fnt_packed" in
  let o = Ll_test.optimize ~materialized:[ a; b ] ~name (Ll_test.seq (draw a) (draw b)) in
  two_siblings "packed draws" ~f:is_set_from_vec o;
  let in_range = "packed draws: every draw lies in [0, 1)" in
  let equal = "packed draws: both draw bitwise-equal values" in
  compile_and_run "packed draws" ~name o ~seed:[] ~read:[ a; b ] |> fun got ->
  let va, vb = read2 got in
  p_all in_range (Array.to_list va) ~f:(fun x -> Float.(0. <= x && x < 1.));
  p_all2 equal va vb ~f:bits_equal

(* {1 Families that render only without logging} *)

let width = 64

let count_substring src ~substring =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern:substring)

(* Two vectorized elementwise loops and two vectorized reductions, all siblings at the routine's top
   level. On the C backends each renders as a block that names its temporaries by a counter
   restarting at zero. The extent is a multiple of every vector width, so no remainder loop is
   needed for a loop to vectorize. *)
let check_vectorized ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| width |] () in
  let mk1 = Ll_test.node_factory ~first_id:(first_id + 50) ~dims:[| 1 |] () in
  let a = mk "vec_a" and b = mk "vec_b" and x = mk "vec_x" and y = mk "vec_y" in
  let s1 = mk1 "vec_s1" and s2 = mk1 "vec_s2" in
  let nodes = [ a; b; x; y; s1; s2 ] in
  List.iter nodes ~f:Ll_test.materialize;
  let i = Ll_test.sym () and j = Ll_test.sym () and k = Ll_test.sym () and l = Ll_test.sym () in
  let it = Ll_test.iter in
  let zero = Ll_test.fixed 0 in
  let llc =
    LL.unflat_lines
      [
        Ll_test.loop_n i width
          (Ll_test.set_at x (it i)
             (Ll_test.add (Ll_test.mul (get a [| it i |]) (get b [| it i |])) (Ll_test.c 1.)));
        Ll_test.loop_n j width
          (Ll_test.set_at y (it j)
             (Ll_test.add (get a [| it j |]) (Ll_test.mul (Ll_test.c 3.) (get b [| it j |]))));
        Ll_test.loop_n k width
          (Ll_test.set_at s1 zero (Ll_test.add (get s1 [| zero |]) (get a [| it k |])));
        Ll_test.loop_n l width
          (Ll_test.set_at s2 zero (Ll_test.add (get s2 [| zero |]) (get b [| it l |])));
      ]
  in
  let name = "fnt_vectorized" in
  let o =
    Sched.apply
      (List.map [ i; j; k; l ] ~f:(fun axis -> Sched.Retype { axis; ty = LL.Vectorized }))
      (Ll_test.optimize ~materialized:nodes ~name llc)
  in
  p "four vectorized loops are siblings in one block of the compiled routine"
    (siblings ~f:is_vectorized o.LL.llc >= 4);
  let av = input width and bv = Array.init width ~f:(fun i -> Float.of_int ((i * 3 % 7) - 3)) in
  let vx, vy, vs1, vs2 =
    match
      Ll_test.execute ~name o
        ~seed:
          [
            (a, av);
            (b, bv);
            (x, Ll_test.blank width);
            (y, Ll_test.blank width);
            (s1, [| 0. |]);
            (s2, [| 0. |]);
          ]
        ~read:[ x; y; s1; s2 ]
    with
    | [ vx; vy; vs1; vs2 ] -> (vx, vy, vs1, vs2)
    | _ -> assert false
  in
  let sum v = Array.fold v ~init:0. ~f:( +. ) in
  p_all2 "vectorized loops: all four compile in one scope and compute their values"
    (Array.concat [ vx; vy; vs1; vs2 ])
    (Array.concat
       [
         Array.init width ~f:(fun c -> (av.(c) *. bv.(c)) +. 1.);
         Array.init width ~f:(fun c -> av.(c) +. (3. *. bv.(c)));
         [| sum av; sum bv |];
       ])
    ~f:Float.equal;
  let src = Generated.read name in
  gated ~when_:on_cpu ~on:backend_name
    "vectorized loops: both elementwise loops render the vector form"
    (count_substring src ~substring:"{ /* Vectorized rendering:" = 2);
  gated ~when_:on_cpu ~on:backend_name "vectorized loops: both reductions render the vector form"
    (count_substring src ~substring:"{ /* Vectorized reduction rendering:" = 2)

(* Two [Tile_mma] statements in one lane body, each with its own destination; on the C backends each
   renders the register tile, whose [tmma_*] names repeat from tile to tile. *)
let check_two_tiles ~first_id =
  let m = 8 in
  let mk = Ll_test.node_factory ~first_id ~dims:[| m; m |] () in
  let a = mk "tile_a" and b = mk "tile_b" and d1 = mk "tile_d1" and d2 = mk "tile_d2" in
  let nodes = [ a; b; d1; d2 ] in
  List.iter nodes ~f:Ll_test.materialize;
  let it = Ll_test.iter in
  let micro_kernel d =
    let i = Ll_test.sym () and j = Ll_test.sym () and l = Ll_test.sym () in
    Ll_test.loop_n i m @@ Ll_test.loop_n j m @@ Ll_test.loop_n l m
    @@ Ll_test.set d
         [| it i; it j |]
         (LL.Ternop
            ( Ops.FMA,
              (get a [| it i; it l |], Ops.single),
              (get b [| it l; it j |], Ops.single),
              (get d [| it i; it j |], Ops.single) ))
  in
  let k1 = micro_kernel d1 and k2 = micro_kernel d2 in
  (* The lane axis is one SIMD group wide, the width [Schedule.tensorize] is given: the GPU
     intrinsics execute across the whole group, and the C backends run the tile on lane 0. *)
  let simd_width = 32 in
  let lane = Ll_test.sym () in
  let origin = [| Ll_test.fixed 0; Ll_test.fixed 0 |] in
  let tile d fallback =
    Ll_test.tile_mma ~m ~n:m ~k:m ~lane ~d:(d, origin) ~a:(a, origin) ~b:(b, origin) fallback
  in
  let name = "fnt_tiles" in
  let o =
    Ll_test.optimize_scoped ~materialized:nodes ~name ~raw:(Ll_test.seq k1 k2)
      (LL.For_loop
         {
           index = lane;
           from_ = 0;
           to_ = simd_width - 1;
           axis = LL.Workgroup;
           body = Ll_test.seq (tile d1 k1) (tile d2 k2);
         })
  in
  two_siblings "Tile_mma statements" ~f:is_tile o;
  let cells = m * m in
  let av = input cells and bv = Array.init cells ~f:(fun c -> Float.of_int ((c * 7 % 5) - 2)) in
  let d2_init = Array.init cells ~f:(fun c -> Float.of_int (100 + c)) in
  let values, census =
    Ll_test.execute_with_census ~name o
      ~seed:[ (a, av); (b, bv); (d1, Array.create ~len:cells 0.); (d2, d2_init) ]
      ~read:[ d1; d2 ]
  in
  let vd1, vd2 = read2 values in
  let product c =
    let r = c / m and col = c % m in
    List.sum (module Float) (List.range 0 m) ~f:(fun l -> av.((r * m) + l) *. bv.((l * m) + col))
  in
  p_all2 "two tiles: both compile in one lane body and accumulate their products"
    (Array.append vd1 vd2)
    (Array.append (Array.init cells ~f:product)
       (Array.init cells ~f:(fun c -> d2_init.(c) +. product c)))
    ~f:Float.equal;
  let census = Ll_test.renderings census in
  (* Read against the backend's own capability judgment: where it advertises an f32 tile (the C
     backends' register tile, Metal's simdgroup matrices) both statements must render tensorized,
     elsewhere both the lane-0 fallback, so neither arm can pass for the other. *)
  p "two tiles: both render tensorized where the backend advertises the tile, else both decline"
    (List.length census = 2
    && Ll_test.census_agrees_with_capability ~backend_name ~what:"two tiles" ~a_prec:Ops.single
         ~b_prec:Ops.single ~d_prec:Ops.single census)

let () =
  if logged then (
    check_logged_sets ~first_id:1553000;
    check_logged_dynamic ~first_id:1553100;
    check_logged_locals ~first_id:1553200;
    check_two_packed_draws ~first_id:1553300)
  else (
    Utils.settings.output_debug_files_in_build_directory <- true;
    Generated.init ~backend_name;
    check_two_packed_draws ~first_id:1553300;
    check_vectorized ~first_id:1553400;
    check_two_tiles ~first_id:1553500)
