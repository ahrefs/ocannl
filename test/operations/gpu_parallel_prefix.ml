(* gh-ocannl-1133: the default GPU schedule maps every loop of a nest's proved parallel chain -- the
   leading loops [Grid] (slots >= 2 folding onto .z), the innermost the [Workgroup] lane, split into
   [Grid] blocks past the block size -- under one common hardware topology per kernel, and falls
   back to the two-loop presets when the proof, the topology, the device caps or the useful work say
   so.

   Every case is EXECUTED against a host reference: producers write [1 + flat index], which varies
   with every coordinate, consumers add a per-case constant, and cells no writer covers would keep a
   sentinel. The launch dimensions are asserted beside the values, so a case cannot pass by falling
   back to a serial kernel. The expected geometries are the slot rule applied by hand to the chosen
   plan (innermost grid loop binds .x, the next .y, the rest fold their product onto .z; likewise
   workgroup loops onto .x/.y/.z). On a GPU backend the executed leg runs the hardware binding; on
   cc the annotated loops render serially and the values still pin the index maps. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule
module BI = Ir.Backend_intf

let node = L.node_factory ~first_id:1133000 ~dims:[| 1 |] ()
let numel dims = Array.fold dims ~init:1 ~f:( * )

let nest dims f =
  let syms = Array.map dims ~f:(fun _ -> L.sym ()) in
  let body = f (Array.map syms ~f:L.iter) in
  Array.fold_right (Array.zip_exn syms dims) ~init:body ~f:(fun (s, n) body -> L.loop_n s n body)

(* [1 + row-major flat index]: identifies every coordinate and stays clear of zero. *)
let value idcs dims =
  Array.foldi idcs ~init:(L.c 1.) ~f:(fun axis acc idx ->
      let stride =
        Array.fold
          (Array.sub dims ~pos:(axis + 1) ~len:(Array.length dims - axis - 1))
          ~init:1 ~f:( * )
      in
      L.add acc (L.mul (L.c (Float.of_int stride)) (LL.Embed_index idx)))

let ramp dims ~plus = Array.init (numel dims) ~f:(fun i -> Float.of_int (i + 1) +. plus)

let geometry_is name (d : LL.launch_dims) ~grid ~block =
  p
    (Printf.sprintf "%s: launch grid %s block %s" name
       (String.concat ~sep:";" (List.map (Array.to_list grid) ~f:Int.to_string))
       (String.concat ~sep:";" (List.map (Array.to_list block) ~f:Int.to_string)))
    (Array.equal Int.equal d.LL.grid grid && Array.equal Int.equal d.LL.block block)

let schedule ?(block_size = 256) ?(fill = 1) ?(limits = BI.no_hardware_limits) opt =
  S.apply (S.default_gpu ~block_size ~min_parallel:64 ~workgroup_fill:fill ~limits opt) opt

(* Producer [a := 1 + flat], consumer [b := a + 3] over the same traversal: a dependent pair whose
   chains align position by position, so both nests take the same plan. *)
let dependent ?block_size ?fill ?limits ~name ~dims ~grid ~block () =
  let a = node ~dims (name ^ "_a") and b = node ~dims (name ^ "_b") in
  List.iter [ a; b ] ~f:L.materialize;
  let producer = nest dims (fun idcs -> L.set a idcs (value idcs dims)) in
  let consumer = nest dims (fun idcs -> L.set b idcs (L.add (L.get a idcs) (L.c 3.))) in
  let opt = L.optimize ~materialized:[ a; b ] ~name (L.seq producer consumer) in
  let scheduled = schedule ?block_size ?fill ?limits opt in
  geometry_is name (LL.launch_dims scheduled.llc) ~grid ~block;
  let got = L.execute ~name scheduled ~seed:[] ~read:[ a; b ] in
  p_all2 (name ^ ": every producer cell") (List.nth_exn got 0) (ramp dims ~plus:0.) ~f:Float.equal;
  p_all2 (name ^ ": every consumer cell") (List.nth_exn got 1) (ramp dims ~plus:3.) ~f:Float.equal

(* Two independent producers of different ranks in one kernel. *)
let mixed_rank ?fill ~name ~dims_a ~dims_b ~grid ~block () =
  let a = node ~dims:dims_a (name ^ "_a") and b = node ~dims:dims_b (name ^ "_b") in
  List.iter [ a; b ] ~f:L.materialize;
  let na = nest dims_a (fun idcs -> L.set a idcs (value idcs dims_a)) in
  let nb = nest dims_b (fun idcs -> L.set b idcs (L.add (value idcs dims_b) (L.c 0.5))) in
  let opt = L.optimize ~materialized:[ a; b ] ~name (L.seq na nb) in
  let scheduled = schedule ?fill opt in
  geometry_is name (LL.launch_dims scheduled.llc) ~grid ~block;
  let got = L.execute ~name scheduled ~seed:[] ~read:[ a; b ] in
  p_all2
    (name ^ ": every cell of the first node")
    (List.nth_exn got 0) (ramp dims_a ~plus:0.) ~f:Float.equal;
  p_all2
    (name ^ ": every cell of the second node")
    (List.nth_exn got 1) (ramp dims_b ~plus:0.5) ~f:Float.equal

(* The consumer walks the last two axes transposed, reading the producer's cell at the swapped
   position: alignment keeps the common prefix only, and the plan maps no more than that proof. *)
let transposed ~name ~dims ~grid ~block =
  let a = node ~dims (name ^ "_a") in
  let tdims = Array.copy dims in
  let r = Array.length dims in
  tdims.(r - 1) <- dims.(r - 2);
  tdims.(r - 2) <- dims.(r - 1);
  let b = node ~dims:tdims (name ^ "_b") in
  List.iter [ a; b ] ~f:L.materialize;
  let producer = nest dims (fun idcs -> L.set a idcs (value idcs dims)) in
  let consumer =
    nest tdims (fun idcs ->
        let src = Array.copy idcs in
        src.(r - 1) <- idcs.(r - 2);
        src.(r - 2) <- idcs.(r - 1);
        L.set b idcs (L.add (L.get a src) (L.c 3.)))
  in
  let opt = L.optimize ~materialized:[ a; b ] ~name (L.seq producer consumer) in
  let scheduled = schedule opt in
  geometry_is name (LL.launch_dims scheduled.llc) ~grid ~block;
  let got = List.nth_exn (L.execute ~name scheduled ~seed:[] ~read:[ b ]) 0 in
  let expected =
    Array.init (numel tdims) ~f:(fun i ->
        let t = L.unflat ~dims:tdims i in
        let s = Array.copy t in
        s.(r - 1) <- t.(r - 2);
        s.(r - 2) <- t.(r - 1);
        Float.of_int (L.flat ~dims s + 1) +. 3.)
  in
  p_all2 (name ^ ": every transposed consumer cell") got expected ~f:Float.equal

let zeros ~name ~dims_list ~grid ~block =
  let tns = List.mapi dims_list ~f:(fun i dims -> node ~dims (Printf.sprintf "%s_%d" name i)) in
  List.iter tns ~f:L.materialize;
  let opt =
    L.optimize ~materialized:tns ~name
      (List.fold (List.tl_exn tns)
         ~init:(L.zero (List.hd_exn tns))
         ~f:(fun acc tn -> L.seq acc (L.zero tn)))
  in
  let scheduled =
    S.apply
      (S.zero_expansion ~block_size:256 ~min_parallel:64 ~workgroup_fill:1
         ~limits:BI.no_hardware_limits tns)
      opt
  in
  geometry_is name (LL.launch_dims scheduled.llc) ~grid ~block;
  let seed =
    List.map2_exn tns dims_list ~f:(fun tn dims -> (tn, Array.create ~len:(numel dims) 17.))
  in
  let got = L.execute ~name scheduled ~seed ~read:tns in
  p_all
    (name ^ ": every cell of every node is cleared")
    (List.concat_map got ~f:Array.to_list)
    ~f:(Float.equal 0.)

let () =
  Stdlib.Printf.eprintf "gpu_parallel_prefix backend: %s (not part of the golden)\n%!"
    (Context.backend_name (Context.auto ()));
  Stdio.printf "--- the slot rule's one owner ---\n";
  let geo nests = S.launch_geometry_of_nests nests in
  p "a fold whose product overflows is refused, not wrapped"
    (Option.is_none (geo [ ([ Int.max_value / 2; 4; 1; 1 ], [ 32 ]) ]));
  p "the same shape at a representable extent folds its two outer slots onto .z"
    (Option.equal Int.equal
       (Option.bind (geo [ ([ 1024; 4; 1; 1 ], [ 32 ]) ]) ~f:(fun g -> g.S.lg_grid_z))
       (Some 4096));
  (* Two nests peaking at different folded slots: the product of per-slot maxima (3 * 3), not the
     largest per-nest product (6). *)
  p "per-slot maxima are taken across nests before the fold"
    (Option.equal Int.equal
       (Option.bind
          (geo [ ([ 2; 3; 1; 1 ], [ 1 ]); ([ 3; 2; 1; 1 ], [ 1 ]) ])
          ~f:(fun g -> g.S.lg_grid_z))
       (Some 9));

  Stdio.printf "--- ranks 1 to 5, dependent nests over equal traversals ---\n";
  (* One loop: split into four blocks, the last a 232-lane tail (1000 = 3 * 256 + 232). *)
  dependent ~name:"pp_rank1_tail" ~dims:[| 1000 |] ~grid:[| 4; 1; 1 |] ~block:[| 256; 1; 1 |] ();
  dependent ~name:"pp_rank2" ~dims:[| 96; 40 |] ~grid:[| 96; 1; 1 |] ~block:[| 40; 1; 1 |] ();
  (* A small leading axis becomes one more grid slot instead of a serial loop. *)
  dependent ~name:"pp_rank3_small_leading" ~dims:[| 6; 20; 48 |] ~grid:[| 20; 6; 1 |]
    ~block:[| 48; 1; 1 |] ();
  dependent ~name:"pp_rank4" ~dims:[| 3; 4; 5; 64 |] ~grid:[| 5; 4; 3 |] ~block:[| 64; 1; 1 |] ();
  (* Grid slots 2 and 3 fold onto .z: 2 * 3. *)
  dependent ~name:"pp_rank5_fold" ~dims:[| 2; 3; 4; 5; 32 |] ~grid:[| 5; 4; 6 |]
    ~block:[| 32; 1; 1 |] ();
  (* A singleton loop of a hand-built nest is one more (extent-1) grid slot. *)
  dependent ~name:"pp_singleton" ~dims:[| 4; 1; 50; 32 |] ~grid:[| 50; 1; 4 |] ~block:[| 32; 1; 1 |]
    ();
  (* A lane past the block size: Grid blocks of it, never a serial outer part. *)
  dependent ~name:"pp_lane_split_tail" ~dims:[| 6; 300 |] ~grid:[| 2; 6; 1 |] ~block:[| 256; 1; 1 |]
    ();

  Stdio.printf "--- device caps: the lane plan at the cap, the two-loop plan past it ---\n";
  let yz cap = { BI.no_hardware_limits with BI.max_grid_yz = Some cap } in
  dependent ~limits:(yz 4) ~name:"pp_grid_y_at_cap" ~dims:[| 3; 4; 5; 64 |] ~grid:[| 5; 4; 3 |]
    ~block:[| 64; 1; 1 |] ();
  (* .y = 4 > 3: the presets' suffix pair (5, 64), the leading loops serial. *)
  dependent ~limits:(yz 3) ~name:"pp_grid_y_past_cap" ~dims:[| 3; 4; 5; 64 |] ~grid:[| 5; 1; 1 |]
    ~block:[| 64; 1; 1 |] ();
  dependent ~limits:(yz 6) ~name:"pp_grid_z_at_cap" ~dims:[| 2; 3; 4; 5; 32 |] ~grid:[| 5; 4; 6 |]
    ~block:[| 32; 1; 1 |] ();
  (* .z = 2 * 3 > 5, .y = 4 within it: again the presets' pair (5, 32). *)
  dependent ~limits:(yz 5) ~name:"pp_grid_z_past_cap" ~dims:[| 2; 3; 4; 5; 32 |] ~grid:[| 5; 1; 1 |]
    ~block:[| 32; 1; 1 |] ();

  Stdio.printf "--- workgroup fill ---\n";
  (* Fill 256: the workgroup takes the loop above the lane while 32 * 8 stays within the block. *)
  dependent ~fill:256 ~name:"pp_fill_2d" ~dims:[| 2; 16; 8; 32 |] ~grid:[| 16; 2; 1 |]
    ~block:[| 32; 8; 1 |] ();
  dependent ~fill:1 ~name:"pp_fill_1d" ~dims:[| 2; 16; 8; 32 |] ~grid:[| 8; 16; 2 |]
    ~block:[| 32; 1; 1 |] ();
  (* A .y workgroup cap below 8 keeps the one-dimensional workgroup; 8 admits the widening. *)
  let wg y = { BI.no_hardware_limits with BI.max_workgroup_dims = Some (256, y, 64) } in
  dependent ~fill:256 ~limits:(wg 4) ~name:"pp_fill_y_past_cap" ~dims:[| 2; 16; 8; 32 |]
    ~grid:[| 8; 16; 2 |] ~block:[| 32; 1; 1 |] ();
  dependent ~fill:256 ~limits:(wg 8) ~name:"pp_fill_y_at_cap" ~dims:[| 2; 16; 8; 32 |]
    ~grid:[| 16; 2; 1 |] ~block:[| 32; 8; 1 |] ();

  Stdio.printf "--- dependent nests over unequal traversals ---\n";
  (* Extents agree on (4, 32) only: both nests map exactly that proved prefix, which is the presets'
     own pair. *)
  transposed ~name:"pp_transposed" ~dims:[| 4; 32; 4; 32 |] ~grid:[| 4; 1; 1 |]
    ~block:[| 32; 1; 1 |];

  Stdio.printf "--- mixed ranks in one kernel ---\n";
  (* (4, 2 | 64) and (4 | 128): the rank-2 nest splits its lane into a one-block grid slot to meet
     the rank-3 nest's two; the union launch, 2 x 4 groups of 128, is twice each nest's own. *)
  mixed_rank ~name:"pp_mixed_split" ~dims_a:[| 4; 2; 64 |] ~dims_b:[| 4; 128 |] ~grid:[| 2; 4; 1 |]
    ~block:[| 128; 1; 1 |] ();
  (* (4, 8 | 32) and (16 | 64) unify the same way, but the union launch (8 x 16 groups of 64) is
     eight times either nest's own, so the presets' pairs stand: (8, 32) and (16, 64). *)
  mixed_rank ~name:"pp_mixed_wasteful" ~dims_a:[| 4; 8; 32 |] ~dims_b:[| 16; 64 |]
    ~grid:[| 16; 1; 1 |] ~block:[| 64; 1; 1 |] ();
  (* Fill 256 gives 8 x 32 and 4 x 64 workgroups, whose per-slot maxima would launch 64 x 8 = 512
     threads per group: past the block size, so both fall back to one-loop workgroups. *)
  mixed_rank ~fill:256 ~name:"pp_fill_union_product" ~dims_a:[| 64; 8; 32 |] ~dims_b:[| 64; 4; 64 |]
    ~grid:[| 8; 64; 1 |] ~block:[| 64; 1; 1 |] ();
  (* Three grid slots against at most two: no common topology, so both keep the presets' pairs ((8,
     32) after the suffix choice, and (16, 64)). *)
  mixed_rank ~name:"pp_mixed_fallback" ~dims_a:[| 2; 4; 8; 32 |] ~dims_b:[| 16; 64 |]
    ~grid:[| 16; 1; 1 |] ~block:[| 64; 1; 1 |] ();
  zeros ~name:"pp_zeros_mixed_split"
    ~dims_list:[ [| 4; 2; 64 |]; [| 4; 128 |] ]
    ~grid:[| 2; 4; 1 |] ~block:[| 128; 1; 1 |];
  zeros ~name:"pp_zeros_mixed_wasteful"
    ~dims_list:[ [| 4; 8; 32 |]; [| 16; 64 |] ]
    ~grid:[| 16; 1; 1 |] ~block:[| 64; 1; 1 |];
  (* A singleton axis stays a serial loop outside the zero plan. *)
  zeros ~name:"pp_zeros_singleton"
    ~dims_list:[ [| 1; 6; 20; 48 |] ]
    ~grid:[| 20; 6; 1 |] ~block:[| 48; 1; 1 |];
  zeros ~name:"pp_zeros_mixed_fallback"
    ~dims_list:[ [| 2; 4; 8; 32 |]; [| 16; 64 |] ]
    ~grid:[| 16; 1; 1 |] ~block:[| 64; 1; 1 |]
