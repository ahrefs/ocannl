(* gh-ocannl-2136: byte draws must compare as unsigned before a storage round-trip. All-ones words
   deterministically exercise the high bit; zero words exercise the other comparison branch. The
   packed store is the materialized unsigned oracle. *)
open Base
open Ll_test
open Verdict.Claims

let () =
  p_all ~min:256 "host scalar draws cover every unsigned byte" (List.init 256 ~f:Fn.id)
    ~f:(fun byte -> Ops.uint4x32_to_byte_uniform (Ops.int64_to_uint4x32 (Int64.of_int byte)) = byte);
  Stdio.eprintf "byte_uniform_unsigned: backend %s (not part of the golden)\n"
    (Context.backend_name (Lazy.force base_ctx));
  let n = 16 in
  let mk = node_factory ~prec:Ops.byte ~first_id:2136000 ~dims:[| n |] () in
  let packed = mk "bu_packed" and direct = mk "bu_direct" and stored = mk "bu_stored" in
  let scalar = mk ~dims:[| 1 |] "bu_scalar" in
  List.iter [ packed; direct; stored; scalar ] ~f:materialize;
  let key = LL.Constant_bits (-1L) in
  let i = sym () in
  let at = [| iter i |] in
  let lane =
    LL.Binop (Ops.Uint4x32_to_prec_uniform_lane, (key, Ops.uint4x32), (embed i, iprec ()))
  in
  let below v = cmp_at Ops.byte Ops.Cmplt v (c 100.) in
  let name = "bu_unsigned" in
  let llc =
    LL.unflat_lines
      [
        set_from_vec ~length:n ~vec_unop:Ops.Uint4x32_to_prec_uniform packed
          [| fixed 0 |]
          (key, Ops.uint4x32);
        loop_n i n
          (LL.unflat_lines [ set direct at (below lane); set stored at (below (get packed at)) ]);
        set_at scalar (fixed 0)
          (below (LL.Unop (Ops.Uint4x32_to_prec_uniform1, (key, Ops.uint4x32))));
      ]
  in
  let o = optimize ~name ~materialized:[ packed; direct; stored; scalar ] llc in
  let values = execute ~name o ~seed:[] ~read:[ packed; direct; stored; scalar ] in
  let draws = List.nth_exn values 0 and direct = List.nth_exn values 1 in
  let stored = List.nth_exn values 2 and scalar = List.nth_exn values 3 in
  p_all2 "packed bytes cover high-bit and zero values" draws
    (Array.init n ~f:(fun j -> if j < 8 then 255. else 0.))
    ~f:Float.equal;
  p_all2 "stored unsigned comparisons take both branches" stored
    (Array.init n ~f:(fun j -> if j < 8 then 0. else 1.))
    ~f:Float.equal;
  p_all2 "lane comparisons match unsigned byte storage" direct stored ~f:Float.equal;
  p "scalar comparison treats 255 as unsigned" Float.(scalar.(0) = 0.)
