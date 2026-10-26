(* gh-ocannl-1600: the ranking score's memory-transaction term.

   Every score before it tied a layout whose lanes read contiguous cells with one whose lanes stride
   through memory -- same work, same launch, same footprints -- so [model_default] could not rank
   the coalesced layouts (coal-n, coal-k) against their uncoalesced twins. Per global access and
   lane group, {!Ir.Cost_model.memory_transactions} counts the 32-byte sectors touched beyond the
   compact count for the same distinct bytes.

   On hand-built kernels at a lane width of 32, every count compared is exact (sector counts are
   integers, the residue average a dyadic fraction): contiguous, stride-2, column (stride-32) and
   broadcast reads; a two-slot block whose lanes wrap rows; a partial lane group; idle lanes past a
   narrow sibling nest's extent; a lane's register tile counted as one footprint (a 4-wide tile
   reads contiguously, a column tile does not); grid blocks and serial iterations multiplying the
   count; the base residues a serial unit stride shifts through (a base-only count would read 0);
   f16 against f32; a flattened [Sub_axis] map counting as its unflattened twin; non-global nodes
   and [Tile_mma] statements left out; a dynamic gather counted one sector per lane and reported
   uncovered. Then the leg: without a rate, or without an excess, the compute leg is bitwise what it
   was; with them, it adds exactly the excess over the rate, each context's over its fill under a
   launch model. And the occupancy factor: theoretical occupancy's exact fractions, and the compute
   leg over it. *)

open Base
open Verdict.Claims
module LL = Ir.Low_level
module CM = Ir.Cost_model
module Idx = Ir.Indexing

let make32 = Ll_test.node_factory ~first_id:972_000_000 ~dims:[||] ()
let make16 = Ll_test.node_factory ~prec:Ir.Ops.half ~first_id:972_500_000 ~dims:[||] ()
let node ?(half = false) label dims = (if half then make16 else make32) ~dims label
let it = Ll_test.iter
let wg = LL.Workgroup

(* [d[di] = x[xi]]: one read and one write per execution. *)
let copy d di x xi = Ll_test.set d di (Ll_test.get x xi)
let all_global (_ : Ir.Tnode.t) = true

let excess ?(is_global = all_global) code =
  List.sum
    (module Float)
    (CM.memory_transactions ~lane_width:32 ~is_global code)
    ~f:(fun t -> t.CM.tx_excess)

let uncovered code =
  List.sum
    (module Int)
    (CM.memory_transactions ~lane_width:32 ~is_global:all_global code)
    ~f:(fun t -> t.CM.tx_uncovered)

(* A one-slot block of [n] lanes copying [x] at [xi w] into a contiguous [d]. *)
let lanes ?half ~tag ~n ~xdims xi =
  let d = node ?half (tag ^ "_d") [| n |] and x = node ?half (tag ^ "_x") xdims in
  let w = Ll_test.sym () in
  Ll_test.loop_n ~axis:wg w n (copy d [| it w |] x (xi w))

let () =
  p "a contiguous f32 read and write: 4 sectors for 128 bytes, no excess"
    (Float.equal (excess (lanes ~tag:"tx_c" ~n:32 ~xdims:[| 32 |] (fun w -> [| it w |]))) 0.);
  p "a stride-2 read touches 8 sectors for 4 sectors' bytes: excess 4"
    (Float.equal
       (excess (lanes ~tag:"tx_s2" ~n:32 ~xdims:[| 64 |] (fun w -> [| Ll_test.aff [ (2, w) ] 0 |])))
       4.);
  p "a column read (stride 32 cells) touches 32 sectors: excess 28"
    (Float.equal
       (excess
          (lanes ~tag:"tx_col" ~n:32 ~xdims:[| 32; 32 |] (fun w -> [| it w; Ll_test.fixed 0 |])))
       28.);
  p "a broadcast read (no lane coordinate) is one sector: no excess"
    (Float.equal
       (excess (lanes ~tag:"tx_bc" ~n:32 ~xdims:[| 4 |] (fun _ -> [| Ll_test.fixed 1 |])))
       0.);
  p "f16 lanes: contiguous is 2 sectors, no excess; stride 2 is 4 sectors: excess 2"
    (Float.equal
       (excess (lanes ~half:true ~tag:"tx_h" ~n:32 ~xdims:[| 32 |] (fun w -> [| it w |])))
       0.
    && Float.equal
         (excess
            (lanes ~half:true ~tag:"tx_h2" ~n:32 ~xdims:[| 64 |] (fun w ->
                 [| Ll_test.aff [ (2, w) ] 0 |])))
         2.);
  p "a partial lane group: 48 lanes at stride 2 are excess 4 + 2"
    (Float.equal
       (excess (lanes ~tag:"tx_p" ~n:48 ~xdims:[| 96 |] (fun w -> [| Ll_test.aff [ (2, w) ] 0 |])))
       6.)

let () =
  (* A two-slot block: [w0] binds .x (extent 8), [w1] .y (extent 4); a lane group is 4 rows of 8. *)
  let two ~tag ~xdims xi =
    let d = node (tag ^ "_d") [| 4; 8 |] and x = node (tag ^ "_x") xdims in
    let w0 = Ll_test.sym () and w1 = Ll_test.sym () in
    Ll_test.loop_n ~axis:wg w1 4
      (Ll_test.loop_n ~axis:wg w0 8 (copy d [| it w1; it w0 |] x (xi w0 w1)))
  in
  p "lanes reading row-major along .x are contiguous across the wrapped rows: no excess"
    (Float.equal (excess (two ~tag:"tx_rm" ~xdims:[| 4; 8 |] (fun w0 w1 -> [| it w1; it w0 |]))) 0.);
  p "lanes reading 8 rows of 4 cells (16 bytes each) touch 8 sectors: excess 4"
    (Float.equal (excess (two ~tag:"tx_t" ~xdims:[| 8; 64 |] (fun w0 w1 -> [| it w0; it w1 |]))) 4.)

let () =
  (* A wide nest sets the launch block to 64 lanes; a sibling nest of 16 lanes reads at stride 2. *)
  let wide = lanes ~tag:"tx_w" ~n:64 ~xdims:[| 64 |] (fun w -> [| it w |]) in
  let narrow = lanes ~tag:"tx_n" ~n:16 ~xdims:[| 32 |] (fun w -> [| Ll_test.aff [ (2, w) ] 0 |]) in
  p "a narrow sibling's idle lanes are not borrowed: 16 lanes at stride 2 are excess 2"
    (Float.equal (excess (Ll_test.seq wide narrow)) 2.);
  (* Four grid blocks, each repeating the stride-2 group; a serial loop of 3 around it. *)
  let x = node "tx_gx" [| 4; 3; 64 |] and d = node "tx_gd" [| 4; 3; 32 |] in
  let b = Ll_test.sym () and k = Ll_test.sym () and w = Ll_test.sym () in
  let grid =
    Ll_test.loop_n ~axis:LL.Grid b 4
      (Ll_test.loop_n ~axis:wg w 32
         (Ll_test.loop_n k 3
            (copy d [| it b; it k; it w |] x [| it b; it k; Ll_test.aff [ (2, w) ] 0 |])))
  in
  p "grid blocks and serial iterations multiply the excess: 4 * 3 * 4"
    (Float.equal (excess grid) 48.);
  p "and the context is the 4 blocks of 32 threads"
    (List.equal Poly.equal
       (List.map (CM.memory_transactions ~lane_width:32 ~is_global:all_global grid) ~f:(fun t ->
            (t.CM.tx_blocks, t.CM.tx_threads)))
       [ (4, 32) ])

let () =
  (* A 4-wide register tile: 8 lanes, each reading 4 consecutive cells. Per instruction the lanes
     stride by 16 bytes (4 sectors for 32 bytes' worth), but the tile's 4 instructions together read
     128 contiguous bytes: one footprint, no excess. *)
  let x = node "tx_tx" [| 32 |] and d = node "tx_td" [| 32 |] in
  let w = Ll_test.sym () and j = Ll_test.sym () in
  let tile =
    Ll_test.loop_n ~axis:wg w 8
      (Ll_test.loop_n j 4
         (copy d [| Ll_test.aff [ (4, w); (1, j) ] 0 |] x [| Ll_test.aff [ (4, w); (1, j) ] 0 |]))
  in
  p "a lane's 4-wide register tile is one footprint with its lane group's: no excess"
    (Float.equal (excess tile) 0.);
  (* The same tile reading a column per lane: 8 lanes x 4 rows of a 32-wide matrix are 32 sectors
     for 4 sectors' bytes, whichever instruction issues them. *)
  let y = node "tx_ty" [| 32; 32 |] and e = node "tx_te" [| 8; 4 |] in
  let w = Ll_test.sym () and j = Ll_test.sym () in
  let strided =
    Ll_test.loop_n ~axis:wg w 8
      (Ll_test.loop_n j 4
         (copy e [| it w; it j |] y [| Ll_test.aff [ (4, w); (1, j) ] 0; Ll_test.fixed 0 |]))
  in
  p "a register tile reading down a column still pays: excess 28" (Float.equal (excess strided) 28.)

let () =
  (* Theoretical occupancy: 1536 thread slots and 64 KiB per multiprocessor. *)
  let occ w t =
    CM.occupancy ~workgroup_bytes:w ~block_threads:t ~threads_per_mp:1536 ~bytes_per_mp:65536
  in
  p "blocks of 256 threads with 4 KiB each fill the multiprocessor's threads: occupancy 1"
    (Float.equal (occ 4096 256) 1.);
  p "16 KiB blocks of 256 threads: 4 resident, 1024 of 1536 slots"
    (Float.equal (occ 16384 256) (1024. /. 1536.));
  p "a 64 KiB block of 64 threads is alone: 64 of 1536 slots"
    (Float.equal (occ 65536 64) (64. /. 1536.));
  p "a block over the capacity still counts as one resident block"
    (Float.equal (occ 131072 64) (64. /. 1536.));
  p "a kernel without workgroup memory is bounded by its threads alone: 2 blocks of 640"
    (Float.equal (occ 0 640) (1280. /. 1536.))

let () =
  (* A serial unit stride shifts the base by 4 bytes: at 7 of the 8 residues a 128-byte run
     straddles 5 sectors. Per execution (4 + 7 * 5) / 8 - 4 = 0.875, over 8 iterations 7. *)
  let x = node "tx_rx" [| 40 |] and d = node "tx_rd" [| 8; 32 |] in
  let k = Ll_test.sym () and w = Ll_test.sym () in
  let code =
    Ll_test.loop_n k 8
      (Ll_test.loop_n ~axis:wg w 32
         (copy d [| it k; it w |] x [| Ll_test.aff [ (1, k); (1, w) ] 0 |]))
  in
  p "a serial unit stride is counted over its sector residues: excess 7, not a base-only 0"
    (Float.equal (excess code) 7.)

let () =
  (* A flattened map: [Sub_axis] over [2; 32] indexed by one stride-2 lane term. *)
  let flat = node "tx_fx" [| 2; 32 |] and plain = node "tx_px" [| 64 |] in
  let d1 = node "tx_fd" [| 32 |] and d2 = node "tx_pd" [| 32 |] in
  let w1 = Ll_test.sym () and w2 = Ll_test.sym () in
  let flattened =
    Ll_test.loop_n ~axis:wg w1 32
      (copy d1 [| it w1 |] flat [| Idx.Sub_axis; Ll_test.aff [ (2, w1) ] 0 |])
  in
  let unflattened =
    Ll_test.loop_n ~axis:wg w2 32 (copy d2 [| it w2 |] plain [| Ll_test.aff [ (2, w2) ] 0 |])
  in
  p "a flattened Sub_axis map counts as its unflattened twin: excess 4 each"
    (Float.equal (excess flattened) 4. && Float.equal (excess unflattened) 4.)

let () =
  let x = node "tx_ngx" [| 32; 32 |] and d = node "tx_ngd" [| 32 |] in
  let w = Ll_test.sym () in
  let col = Ll_test.loop_n ~axis:wg w 32 (copy d [| it w |] x [| it w; Ll_test.fixed 0 |]) in
  p "a non-global node is not counted: the same column read from a workgroup tile costs nothing"
    (Float.equal (excess col) 28.
    && Float.equal (excess ~is_global:(fun tn -> not (Ir.Tnode.equal tn x)) col) 0.);
  (* The same column read as a Tile_mma's fallback: left out. *)
  let a = node "tx_ma" [| 2; 2 |]
  and b = node "tx_mb" [| 2; 2 |]
  and dm = node "tx_md" [| 2; 2 |] in
  let tile =
    Ll_test.tile_mma
      ~d:(dm, [| Ll_test.fixed 0; Ll_test.fixed 0 |])
      ~a:(a, [| Ll_test.fixed 0; Ll_test.fixed 0 |])
      ~b:(b, [| Ll_test.fixed 0; Ll_test.fixed 0 |])
      col
  in
  p "a Tile_mma statement's fallback is left out of the count" (Float.equal (excess tile) 0.);
  (* A dynamic gather: its row is the data's, so the access is estimated and reported uncovered. *)
  let table = node "tx_dt" [| 8; 32 |]
  and row = node "tx_dr" [| 32 |]
  and out = node "tx_do" [| 32 |] in
  let w = Ll_test.sym () in
  let gather =
    Ll_test.loop_n ~axis:wg w 32
      (Ll_test.set out
         [| it w |]
         (Ll_test.gather ~tn:table
            ~idcs:[| Ll_test.fixed 0; it w |]
            ~dyn_axis:0
            ~dyn_value:(Ll_test.get row [| it w |], Ir.Ops.single)))
  in
  p "a dynamic gather is one uncovered access, estimated at one sector per lane"
    (uncovered gather = 1 && Float.equal (excess gather) 28.)

(* The leg. Power-of-two constants keep every quotient exact. *)
let () =
  let peak = 1024. and issue = 256. and rate = 8. in
  let code =
    Ll_test.loop_n ~axis:LL.Grid (Ll_test.sym ()) 2
      (lanes ~tag:"tx_l" ~n:32 ~xdims:[| 32; 32 |] (fun w -> [| it w; Ll_test.fixed 0 |]))
  in
  let s = CM.analyze code in
  let transactions = CM.memory_transactions ~lane_width:32 ~is_global:all_global code in
  let compute ?launch ?transactions ?transaction_rate ?occupancy () =
    fst
      (CM.ranking_legs_of ~peak_flops:peak ~peak_memory_bandwidth:peak ~issue_rate:issue ?launch
         ?transactions ?transaction_rate ?occupancy s None)
  in
  let eq = Option.equal Float.equal in
  p "the kernel's excess is 2 blocks of a 28-excess column read" (Float.equal (excess code) 56.);
  p "without a rate the compute leg is bitwise as before"
    (eq (compute ~transactions ()) (compute ()));
  p "a rate without an excess adds nothing"
    (eq
       (compute
          ~transactions:(CM.memory_transactions ~lane_width:32 ~is_global:(fun _ -> false) code)
          ~transaction_rate:rate ())
       (compute ()));
  p "with a rate the compute leg adds exactly the excess over it"
    (match (compute ~transactions ~transaction_rate:rate (), compute ()) with
    | Some t, Some c -> Float.equal t (c +. (56. /. rate))
    | _ -> false);
  (* A fill model of 4 multiprocessors saturating at 64 threads: 2 blocks of 32 fill a quarter. *)
  let fm = { CM.fm_multiprocessors = 4; fm_fill_threads = 64 } in
  let launch = (fm, CM.launch_contexts code) in
  p "under a launch model the excess is divided by its context's fill (a quarter)"
    (match (compute ~launch ~transactions ~transaction_rate:rate (), compute ~launch ()) with
    | Some t, Some c -> Float.equal t (c +. (56. /. 0.25 /. rate))
    | _ -> false);
  p "an occupancy of a half doubles the compute leg; an occupancy of 1 leaves it bitwise"
    (match
       ( compute ~transactions ~transaction_rate:rate ~occupancy:0.5 (),
         compute ~transactions ~transaction_rate:rate (),
         compute ~transactions ~transaction_rate:rate ~occupancy:1. () )
     with
    | Some h, Some t, Some o -> Float.equal h (2. *. t) && Float.equal o t
    | _ -> false);
  p "and the term never lowers the leg below the bound's operations at the ceiling"
    (Option.exists (compute ~transactions ~transaction_rate:rate ()) ~f:(fun c ->
         Float.(c >= of_int s.CM.flops /. peak)))
