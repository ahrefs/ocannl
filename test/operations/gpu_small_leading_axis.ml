(* gh-ocannl-995: choose GPU ownership after looking past a small leading batch axis. Every producer
   identifies every coordinate; dependent nests and an incompatible traversal exercise both the new
   choice and the conservative fallback. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule

let node = L.node_factory ~first_id:995000 ~dims:[| 2; 128; 32 |] ()

let nest dims f =
  let syms = Array.map dims ~f:(fun _ -> L.sym ()) in
  let body = f (Array.map syms ~f:L.iter) in
  Array.fold_right (Array.zip_exn syms dims) ~init:body ~f:(fun (s, n) body -> L.loop_n s n body)

let value idcs dims =
  Array.foldi idcs ~init:(L.c 1.) ~f:(fun axis acc idx ->
      let stride =
        Array.fold
          (Array.sub dims ~pos:(axis + 1) ~len:(Array.length dims - axis - 1))
          ~init:1 ~f:( * )
      in
      L.add acc (L.mul (L.c (Float.of_int stride)) (LL.Embed_index idx)))

let dims_equal d grid block =
  Array.equal Int.equal d.LL.grid [| grid; 1; 1 |]
  && Array.equal Int.equal d.LL.block [| block; 1; 1 |]

let run ~name ~dims ~transpose ~grid ~block =
  let a = node ~dims (name ^ "_a") and b = node ~dims (name ^ "_b") in
  List.iter [ a; b ] ~f:L.materialize;
  let producer = nest dims (fun idcs -> L.set a idcs (value idcs dims)) in
  let consumer_dims = Array.copy dims in
  if transpose then (
    consumer_dims.(1) <- dims.(2);
    consumer_dims.(2) <- dims.(1));
  let consumer =
    nest consumer_dims (fun idcs ->
        if transpose then (
          let t = idcs.(1) in
          idcs.(1) <- idcs.(2);
          idcs.(2) <- t);
        L.set b idcs (L.add (L.get a idcs) (L.c 3.)))
  in
  let opt = L.optimize ~materialized:[ a; b ] ~name (L.seq producer consumer) in
  let scheduled = S.apply (S.default_gpu ~block_size:256 ~min_parallel:64 opt) opt in
  p (name ^ ": selected launch geometry") (dims_equal (LL.launch_dims scheduled.llc) grid block);
  let got = List.hd_exn (L.execute ~name scheduled ~seed:[] ~read:[ b ]) in
  let expected = Array.init (Array.fold dims ~init:1 ~f:( * )) ~f:(fun i -> Float.of_int (i + 4)) in
  p_all2 (name ^ ": every coordinate survives dependent nests") got expected ~f:Float.equal

(* The consumer reads a complete producer row only after the producer nest has finished. Its extra
   coordinates need not match the producer's. This is safe under the original pair, but suffix
   alignment can either trim that pair or decline it entirely. *)
let mismatched_suffix ~name ~producer_dims ~consumer_dims ~grid ~block =
  let a = node ~dims:producer_dims (name ^ "_a") and b = node ~dims:consumer_dims (name ^ "_b") in
  List.iter [ a; b ] ~f:L.materialize;
  let producer = nest producer_dims (fun idcs -> L.set a idcs (value idcs producer_dims)) in
  let consumer =
    nest consumer_dims (fun idcs ->
        let source = Array.mapi idcs ~f:(fun i idx -> if i < 2 then idx else L.fixed 0) in
        L.set b idcs (L.add (L.get a source) (value idcs consumer_dims)))
  in
  let opt = L.optimize ~materialized:[ a; b ] ~name (L.seq producer consumer) in
  let scheduled = S.apply (S.default_gpu ~block_size:256 ~min_parallel:64 opt) opt in
  p
    (name ^ ": original geometry survives suffix disagreement")
    (dims_equal (LL.launch_dims scheduled.llc) grid block);
  let got = List.hd_exn (L.execute ~name scheduled ~seed:[] ~read:[ b ]) in
  let row_size dims =
    Array.fold (Array.sub dims ~pos:2 ~len:(Array.length dims - 2)) ~init:1 ~f:( * )
  in
  let expected =
    Array.init (Array.fold consumer_dims ~init:1 ~f:( * )) ~f:(fun i ->
        Float.of_int (2 + i + (i / row_size consumer_dims * row_size producer_dims)))
  in
  p_all2 (name ^ ": dependent row reads remain correct") got expected ~f:Float.equal

let () =
  Stdlib.Printf.eprintf "gpu_small_leading_axis backend: %s\n%!"
    (Context.backend_name (Context.auto ()));
  run ~name:"gsa_batch2" ~dims:[| 2; 128; 32 |] ~transpose:false ~grid:128 ~block:32;
  run ~name:"gsa_batch8" ~dims:[| 8; 128; 32 |] ~transpose:false ~grid:128 ~block:32;
  run ~name:"gsa_batch_head" ~dims:[| 2; 8; 128; 32 |] ~transpose:false ~grid:128 ~block:32;
  run ~name:"gsa_unequal_order" ~dims:[| 2; 128; 32 |] ~transpose:true ~grid:1 ~block:1;
  mismatched_suffix ~name:"gsa_trimmed_suffix" ~producer_dims:[| 2; 32; 64 |]
    ~consumer_dims:[| 2; 32; 128 |] ~grid:2 ~block:32;
  mismatched_suffix ~name:"gsa_failed_suffix" ~producer_dims:[| 8; 8; 32; 16 |]
    ~consumer_dims:[| 8; 8; 16; 16 |] ~grid:8 ~block:8;
  let dims = [| 2; 128; 32 |] in
  let a = node ~dims "gsa_zero" in
  L.materialize a;
  let opt = L.optimize ~materialized:[ a ] ~name:"gsa_zero" (L.zero a) in
  let scheduled =
    S.apply
      (S.zero_expansion ~block_size:256 ~min_parallel:64 ~limits:Ir.Backend_intf.no_hardware_limits
         [ a ])
      opt
  in
  p "expanded zeros use the same suffix geometry" (dims_equal (LL.launch_dims scheduled.llc) 128 32);
  let seed = Array.create ~len:(2 * 128 * 32) 17. in
  let got = List.hd_exn (L.execute ~name:"gsa_zero" scheduled ~seed:[ (a, seed) ] ~read:[ a ]) in
  p_all "expanded zeros clear every coordinate" (Array.to_list got) ~f:(Float.equal 0.)
