(* gh-ocannl-2473: a bf16 scratch cell must round at the store and hold the same bits in private
   memory as in a device buffer. Feed the gathers and scatters a runtime permutation so every index
   is exercised. The second read follows a re-zero and a packed overwrite of the first eight cells,
   so the packed write and the explicit zero loop both affect the result. *)

(* cc-fp-contract: no-contraction -- the kernel moves, converts and zeroes values, and the packed
   uniform conversion contains no multiply-add *)

open Base
open Ll_test
open Verdict.Claims
module Generated = Test_utils.Generated

let count = 192
let packed_lanes = Ops.vec_unop_lanes Ops.bfloat16
let packed_key = 0x1234_5678_9abc_def0L
let packed_counter = 0x1a2b_3c4d_5e6f_7081L
let ids = Array.init count ~f:(fun i -> Float.of_int (i * 5 % count))

let values =
  Array.init count ~f:(fun i ->
      let v = Float.of_int (256 + i) in
      if i % 2 = 0 then v else -.v)

let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let packed_expected =
  let words =
    Ops.threefry4x32_light (Ops.int64_to_uint4x32 packed_key) (Ops.int64_to_uint4x32 packed_counter)
  in
  Array.init packed_lanes ~f:(fun lane ->
      (* The packed draw takes the low then high 16 bits of each word. *)
      let word = words.(lane / 2) in
      let bits = if lane % 2 = 0 then (word land 0xFFFF) lsl 16 else word land 0xFFFF0000 in
      Ops.bfloat16_to_single (Ops.uint4x32_to_bfloat16_uniform [| bits; 0; 0; 0 |]))

let run ~name ~local ~first_id =
  let make ?(prec = Ops.single) offset dims label =
    node_factory ~prec ~first_id:(first_id + offset) ~dims () label
  in
  let src = make 0 [| count |] "scratch_src"
  and indices = make ~prec:Ops.int32 1 [| count |] "scratch_ids"
  and scratch = make ~prec:Ops.bfloat16 2 [| count |] "scratch"
  and dst = make 3 [| 2; count |] "scratch_dst" in
  let materialized = [ src; indices; dst ] in
  List.iter materialized ~f:materialize;
  let materialized =
    if local then (
      Tn.update_memory_mode scratch Tn.Local (Site "99:test-setup");
      materialized)
    else (
      materialize scratch;
      scratch :: materialized)
  in
  let i = sym () and j = sym () and k = sym () in
  let which s = (get indices [| iter s |], Ops.int32) in
  let value = get src [| iter i |] in
  let packed_block =
    LL.Binop
      ( Ops.Threefry4x32_light,
        (LL.Constant_bits packed_key, Ops.uint4x32),
        (LL.Constant_bits packed_counter, Ops.uint4x32) )
  in
  let body scatter =
    LL.unflat_lines
      [
        zero scratch;
        loop_n i count scatter;
        loop_n j count
          (set dst
             [| fixed 0; iter j |]
             (LL.Get_dynamic
                { tn = scratch; idcs = [| fixed 0 |]; dyn_axis = 0; dyn_value = which j }));
        zero scratch;
        set_from_vec ~length:packed_lanes ~vec_unop:Ops.Uint4x32_to_prec_uniform scratch
          [| fixed 0 |]
          (packed_block, Ops.uint4x32);
        loop_n k count (set dst [| fixed 1; iter k |] (get scratch [| iter k |]));
      ]
  in
  (* A dynamic scatter is minted after fact tracing. The static write registers the same nodes
     before the test substitutes the actual scatter. *)
  let raw = body (set_at scratch (iter i) value) in
  let scatter =
    LL.Set_dynamic
      {
        tn = scratch;
        idcs = [| fixed 0 |];
        dyn_axis = 0;
        dyn_value = which i;
        llsc = value;
        debug = "scatter into bf16 scratch";
      }
  in
  let optimized = optimize_scoped ~name ~materialized ~raw (body scatter) in
  let output =
    Ll_test.execute ~name optimized ~seed:[ (src, values); (indices, ids) ] ~read:[ dst ]
  in
  (List.hd_exn output, known_local optimized scratch)

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let metal = String.equal backend "metal" in
  if metal then (
    Utils.settings.output_debug_files_in_build_directory <- true;
    Generated.init ~backend_name:backend);
  let private_name = "bf16_scratch_private" in
  let private_values, is_local = run ~name:private_name ~local:true ~first_id:2473000 in
  let device_values, _ = run ~name:"bf16_scratch_device" ~local:false ~first_id:2473010 in
  let rounded = Array.map values ~f:(Ops.round_to_prec Ops.bfloat16) in
  p "bf16 scratch is placed locally" is_local;
  p_exists "the input distinguishes bf16 rounding from f32 storage"
    (List.zip_exn (Array.to_list values) (Array.to_list rounded))
    ~f:(fun (a, b) -> not (bitwise a b));
  p_all2 "dynamic accesses read the values rounded at the scratch store"
    (Array.sub private_values ~pos:0 ~len:count)
    rounded ~f:bitwise;
  let packed_draws = Array.to_list packed_expected in
  p_all "every packed draw is distinct and nonzero" packed_draws ~f:(fun draw ->
      Float.(draw > 0.) && List.count packed_draws ~f:(bitwise draw) = 1);
  p_all2 "every packed lane stores its own draw"
    (Array.sub private_values ~pos:count ~len:packed_lanes)
    packed_expected ~f:bitwise;
  p_all "re-zero clears the cells beyond the packed store"
    (Array.to_list
       (Array.sub private_values ~pos:(count + packed_lanes) ~len:(count - packed_lanes)))
    ~f:(bitwise 0.);
  p_all2 "private and device scratch agree bitwise" private_values device_values ~f:bitwise;
  in_leg ~leg:(Verdict.leg ~when_:metal ~on:backend ()) "Metal declares the two-byte scratch cells"
    (fun claim ->
      p_exists claim
        (Generated.declared_arrays (Generated.read private_name))
        ~f:(fun a -> String.equal a.typ "ocannl_bf16_cell" && a.elems = count))
