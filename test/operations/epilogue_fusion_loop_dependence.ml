(* gh-ocannl-1844: [Fuse_epilogue]'s dependence check over the store-back site's enclosing loops.

   The relocated tail reads the accumulator right after the site's store-back. A write of the
   accumulator that runs after that read leaves the tail holding a value the final accumulator no
   longer has. Textually later writes are refused by the written-again check. This test is about the
   other kind: a write textually BEFORE the site, inside an enclosing loop of more than one trip,
   which runs again in the next iteration -- after the previous iteration's store-back. Before this
   issue nothing checked it (a loop starting at 0 was descended under any enclosing loop), so every
   refusal claimed below fused on the base.

   The programs are hand-built (no lowering emits an accumulator rewritten across iterations) in the
   two shapes [fuse_at_fragment] recognizes: the fragment store-back [Tensorize]'s contraction
   leaves on the CPU pipelines (a width-1 [Workgroup] lane loop over a tile nest storing a
   registered fragment into the accumulator) and the whole-K [Tile_mma] block. An elementwise tail
   [out = acc + bias] follows. Values are small dyadics, so every comparison is exact, and they run
   through [Ll_test.interpret], the IR's reference semantics ([Tile_mma] through its scalar
   fallback, which is its footprint):

   - item 1, refused: an earlier write lands on cells another iteration's tail has read. The base's
   relocation of it, built by hand and interpreted, computes different values from the unfused
   program, so the refusal guards a real miscompile, not a hypothetical one; - a separated earlier
   write (indexed by the loop as the store-back is) fuses, values equal; - equal coefficients are
   not enough: an offset carrying the write into the previous iteration's cells is refused, and its
   relocation miscomputes; - nested loops: a write separated by the outer loop but not the inner one
   is refused naming the inner loop; a transposed write (iterations (0,1) and (1,0) meet) is
   refused; a write separated by both fuses, values equal; - the whole-K site, the check's second
   entry point: the overlap refused, a separated write fused with values equal; - a whole-node
   [Zero_out] inside the loop is refused. *)

open Base
open Ocannl.Operation.DSL_modules
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule
module Idx = Ir.Indexing

let node = node_factory ~first_id:1844000 ~dims:[| 2; 3 |] ()

type case = { llc : LL.t; acc : Tn.t; out : Tn.t; frag : Tn.t; inputs : (Tn.t * float array) list }

let seq_all = function [] -> LL.Noop | s :: tl -> List.fold tl ~init:s ~f:seq
let lane_loop ~lane body = loop ~axis:LL.Workgroup ~upto:0 lane body

(* Neither [optimize] nor the interpreter takes a [Tile_mma]; its scalar fallback computes the same
   cells. *)
let rec unmma (llc : LL.t) : LL.t =
  match llc with
  | LL.Tile_mma { fallback; _ } -> unmma fallback
  | LL.Seq (a, b) -> LL.Seq (unmma a, unmma b)
  | LL.For_loop fc -> LL.For_loop { fc with body = unmma fc.body }
  | LL.If i -> LL.If { i with body = unmma i.body }
  | other -> other

(* The record [Fuse_epilogue] reads: the program itself (the record built from its [Tile_mma]-free
   twin), with [frag] registered as the fragment the contraction would have minted ([is_fragment] is
   membership, gh-ocannl-1721). *)
let optimized ~name (c : case) : LL.optimized =
  List.iter (c.acc :: c.out :: c.frag :: List.map c.inputs ~f:fst) ~f:materialize;
  let o = optimize ~name (unmma c.llc) in
  { o with LL.llc = c.llc; simdgroup_fragments = Set.singleton (module Tn) c.frag }

let refusal ~name c = Sched.fuse_epilogue_witness ~target:c.acc (optimized ~name c)

let fused ~name c =
  (Sched.apply [ Sched.Fuse_epilogue { target = c.acc; shared = false } ] (optimized ~name c))
    .LL.llc

let interpreted c llc =
  let memory = interp_memory c.inputs in
  interpret memory (unmma llc);
  Array.copy (Hashtbl.find_exn memory c.out)

let refused_naming ~what ~loop = function
  | Some msg ->
      Stdio.eprintf "%s: %s (not part of the golden)\n%!" what msg;
      String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
      && String.is_substring msg ~substring:("enclosing loop " ^ Idx.symbol_ident loop)
  | None -> false

let tail2 ~acc ~out ~bias ~d0 ~d1 =
  let t0 = sym () and t1 = sym () in
  loop_n t0 d0
    (loop_n t1 d1
       (set out
          [| iter t0; iter t1 |]
          (add (get acc [| iter t0; iter t1 |]) (get bias [| iter t1 |]))))

(* A value varying with the iteration and the cell, clear of every store-back value. *)
let w_value a b = add (c 100.) (add (mul (c 10.) (embed a)) (embed b))
let ramp n ~scale = Array.init n ~f:(fun i -> scale *. Float.of_int (i + 1))

(* --- One enclosing loop [b] (two trips) over acc[b, j], [j < 3]. [earlier ~acc ~b ~j] is the write
   before the store-back, inside a loop over [j]. With [~relocated], the program is the one the
   base's relocation emitted: the tail beside the store-back at the site's indices, and no tail nest
   after the loop. --- *)
let one_loop ?(relocated = false) tag ~earlier =
  let acc = node ("acc" ^ tag) and out = node ("out" ^ tag) and x = node ("x" ^ tag) in
  let frag = node ~dims:[| 3 |] ("fragment" ^ tag) and bias = node ~dims:[| 3 |] ("bias" ^ tag) in
  let b = sym () and j = sym () and jf = sym () and js = sym () and lane = sym () in
  let tail_at bi ji = set out [| bi; ji |] (add (get acc [| bi; ji |]) (get bias [| ji |])) in
  let store =
    lane_loop ~lane (loop_n js 3 (set acc [| iter b; iter js |] (get frag [| iter js |])))
  in
  let site =
    if relocated then seq store (lane_loop ~lane (loop_n js 3 (tail_at (iter b) (iter js))))
    else store
  in
  let body =
    seq_all
      [
        loop_n j 3 (earlier ~acc ~b ~j);
        loop_n jf 3 (set frag [| iter jf |] (add (get x [| iter b; iter jf |]) (c 0.5)));
        site;
      ]
  in
  let llc =
    if relocated then loop_n b 2 body else seq (loop_n b 2 body) (tail2 ~acc ~out ~bias ~d0:2 ~d1:3)
  in
  ({ llc; acc; out; frag; inputs = [ (x, ramp 6 ~scale:1.); (bias, ramp 3 ~scale:0.25) ] }, b)

(* --- The offset case, one axis: acc[2b + j], [j < 2]. The earlier write, guarded to stay in
   bounds, is acc[2b - 1]: the same coefficient as the store-back, but in iteration 1 it rewrites
   acc[1], which iteration 0 stored and its tail read. --- *)
let offset_loop ?(relocated = false) tag =
  let acc = node ~dims:[| 4 |] ("acc" ^ tag) and out = node ~dims:[| 4 |] ("out" ^ tag) in
  let x = node ~dims:[| 4 |] ("x" ^ tag) and bias = node ~dims:[| 4 |] ("bias" ^ tag) in
  let frag = node ~dims:[| 2 |] ("fragment" ^ tag) in
  let b = sym () and jf = sym () and js = sym () and lane = sym () and t = sym () in
  let cell s = aff [ (2, b); (1, s) ] 0 in
  let tail_at idx = set out [| idx |] (add (get acc [| idx |]) (get bias [| idx |])) in
  let store = lane_loop ~lane (loop_n js 2 (set acc [| cell js |] (get frag [| iter js |]))) in
  let site =
    if relocated then seq store (lane_loop ~lane (loop_n js 2 (tail_at (cell js)))) else store
  in
  let body =
    seq_all
      [
        if_ (le (ic 1) (embed b)) (set acc [| aff [ (2, b) ] (-1) |] (w_value b b));
        loop_n jf 2 (set frag [| iter jf |] (add (get x [| cell jf |]) (c 0.5)));
        site;
      ]
  in
  let llc =
    if relocated then loop_n b 2 body else seq (loop_n b 2 body) (loop_n t 4 (tail_at (iter t)))
  in
  ({ llc; acc; out; frag; inputs = [ (x, ramp 4 ~scale:1.); (bias, ramp 4 ~scale:0.25) ] }, b)

(* --- Two nested loops b1, b2 (two trips each) over acc[b1, b2]; the earlier write sits inside
   both. --- *)
let nested tag ~earlier =
  let acc = node ~dims:[| 2; 2 |] ("acc" ^ tag) and out = node ~dims:[| 2; 2 |] ("out" ^ tag) in
  let x = node ~dims:[| 2; 2 |] ("x" ^ tag) and bias = node ~dims:[| 2 |] ("bias" ^ tag) in
  let frag = node ~dims:[| 1 |] ("fragment" ^ tag) in
  let b1 = sym () and b2 = sym () and lane = sym () in
  let body =
    seq_all
      [
        earlier ~acc ~b1 ~b2;
        set frag [| fixed 0 |] (add (get x [| iter b1; iter b2 |]) (c 0.5));
        lane_loop ~lane (set acc [| iter b1; iter b2 |] (get frag [| fixed 0 |]));
      ]
  in
  let llc = seq (loop_n b1 2 (loop_n b2 2 body)) (tail2 ~acc ~out ~bias ~d0:2 ~d1:2) in
  ({ llc; acc; out; frag; inputs = [ (x, ramp 4 ~scale:1.); (bias, ramp 2 ~scale:0.25) ] }, b1, b2)

(* --- The whole-K site: per [b] (two trips), a 2x2x2 [Tile_mma] block into acc[b, 0..1, 0..1],
   after an earlier write [earlier ~acc ~b ~i ~j] inside a nest over [i, j] (the block accumulates
   onto it, so it is the block's init). [frag] is unused: no fragment is involved. --- *)
let whole_k tag ~earlier =
  let acc = node ~dims:[| 2; 2; 2 |] ("acc" ^ tag)
  and out = node ~dims:[| 2; 2; 2 |] ("out" ^ tag) in
  let ma = node ~dims:[| 2; 2 |] ("ma" ^ tag) and mb = node ~dims:[| 2; 2 |] ("mb" ^ tag) in
  let bias = node ~dims:[| 2 |] ("bias" ^ tag) and frag = node ~dims:[| 1 |] ("unused" ^ tag) in
  let b = sym () and i = sym () and j = sym () and lane = sym () in
  let fi = sym () and fj = sym () and fk = sym () in
  let cell = [| iter b; iter fi; iter fj |] in
  let fallback =
    loop_n fi 2
      (loop_n fj 2
         (loop_n fk 2
            (set acc cell
               (add (get acc cell)
                  (mul (get ma [| iter fi; iter fk |]) (get mb [| iter fk; iter fj |]))))))
  in
  let block =
    tile_mma ~m:2 ~n:2 ~k:2 ~lane
      ~d:(acc, [| iter b; fixed 0; fixed 0 |])
      ~a:(ma, [| fixed 0; fixed 0 |])
      ~b:(mb, [| fixed 0; fixed 0 |])
      fallback
  in
  let body = seq (loop_n i 2 (loop_n j 2 (earlier ~acc ~b ~i ~j))) (lane_loop ~lane block) in
  let t0 = sym () and t1 = sym () and t2 = sym () in
  let tcell = [| iter t0; iter t1; iter t2 |] in
  let tail =
    loop_n t0 2
      (loop_n t1 2 (loop_n t2 2 (set out tcell (add (get acc tcell) (get bias [| iter t2 |])))))
  in
  ( {
      llc = seq (loop_n b 2 body) tail;
      acc;
      out;
      frag;
      inputs = [ (ma, ramp 4 ~scale:1.); (mb, ramp 4 ~scale:0.5); (bias, ramp 2 ~scale:0.25) ];
    },
    b )

let () =
  (* Item 1: acc[0, j] rewritten in every iteration -- in iteration 1, after iteration 0's tail
     (fused) read acc[0, j]. *)
  let item1 ~acc ~b ~j = set acc [| fixed 0; iter j |] (w_value b j) in
  let case, b = one_loop "1" ~earlier:item1 in
  p "item 1: an earlier write onto another iteration's cells is refused, naming the loop"
    (refused_naming ~what:"item 1" ~loop:b (refusal ~name:"ep1844_item1" case));
  let relocated, _ = one_loop "1r" ~relocated:true ~earlier:item1 in
  let want = interpreted case case.llc in
  p_exists "item 1: the relocation the check refuses computes different values (premise)"
    (List.zip_exn (Array.to_list (interpreted relocated relocated.llc)) (Array.to_list want))
    ~f:(fun (a, b) -> not (Float.equal a b));
  (* The separated control: the earlier write is indexed by [b] as the store-back is. *)
  let case, _ =
    one_loop "2" ~earlier:(fun ~acc ~b ~j -> set acc [| iter b; iter j |] (w_value b j))
  in
  p "separated: an earlier write indexed by the loop fuses"
    (Option.is_none (refusal ~name:"ep1844_sep" case));
  p_all2 "separated: fused values equal the unfused program"
    (interpreted case (fused ~name:"ep1844_sepf" case))
    (interpreted case case.llc) ~f:Float.equal;
  (* Equal coefficients, an offset across the iteration boundary. *)
  let case, b = offset_loop "3" in
  p "offset: an equal-coefficient write offset into the previous iteration's cells is refused"
    (refused_naming ~what:"offset" ~loop:b (refusal ~name:"ep1844_offset" case));
  let relocated, _ = offset_loop "3r" ~relocated:true in
  p_exists "offset: the relocation the check refuses computes different values (premise)"
    (List.zip_exn
       (Array.to_list (interpreted relocated relocated.llc))
       (Array.to_list (interpreted case case.llc)))
    ~f:(fun (a, b) -> not (Float.equal a b));
  (* Nested loops. *)
  let case, _, b2 =
    nested "4" ~earlier:(fun ~acc ~b1 ~b2 -> set acc [| iter b1; fixed 0 |] (w_value b1 b2))
  in
  p "nested: a write separated by the outer loop only is refused, naming the inner loop"
    (refused_naming ~what:"nested inner" ~loop:b2 (refusal ~name:"ep1844_inner" case));
  let case, b1, b2 =
    nested "5" ~earlier:(fun ~acc ~b1 ~b2 -> set acc [| iter b2; iter b1 |] (w_value b1 b2))
  in
  let msg = refusal ~name:"ep1844_transposed" case in
  p "nested: a transposed write is refused"
    (refused_naming ~what:"nested transposed" ~loop:b1 msg
    || refused_naming ~what:"nested transposed" ~loop:b2 msg);
  let case, _, _ =
    nested "6" ~earlier:(fun ~acc ~b1 ~b2 -> set acc [| iter b1; iter b2 |] (w_value b1 b2))
  in
  p "nested: a write separated by both loops fuses"
    (Option.is_none (refusal ~name:"ep1844_nsep" case));
  p_all2 "nested: fused values equal the unfused program"
    (interpreted case (fused ~name:"ep1844_nsepf" case))
    (interpreted case case.llc) ~f:Float.equal;
  (* The whole-K site. *)
  let case, b =
    whole_k "7" ~earlier:(fun ~acc ~b:_ ~i ~j -> set acc [| fixed 0; iter i; iter j |] (c 0.))
  in
  p "whole-K: an earlier write onto another iteration's block is refused, naming the loop"
    (refused_naming ~what:"whole-K" ~loop:b (refusal ~name:"ep1844_wk" case));
  let case, _ =
    whole_k "8" ~earlier:(fun ~acc ~b ~i ~j -> set acc [| iter b; iter i; iter j |] (c 0.))
  in
  p "whole-K: a separated earlier write fuses" (Option.is_none (refusal ~name:"ep1844_wksep" case));
  p_all2 "whole-K: fused values equal the unfused program"
    (interpreted case (fused ~name:"ep1844_wksepf" case))
    (interpreted case case.llc) ~f:Float.equal;
  (* A whole-node write. *)
  let case, b = one_loop "9" ~earlier:(fun ~acc ~b:_ ~j:_ -> zero acc) in
  p "whole node: a Zero_out of the accumulator inside the loop is refused"
    (refused_naming ~what:"whole node" ~loop:b (refusal ~name:"ep1844_zero" case));
  Stdio.printf "epilogue_fusion_loop_dependence: done\n"
