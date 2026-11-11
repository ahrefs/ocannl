(* Materialized integer inputs straddle bf16 midpoints by ONE integer, including past 2^53. Assemble
   high + delta on the device: host float writers cannot represent those odd int64s. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Ops = Ir.Ops

let next_id = ref 48000

let node prec n label =
  next_id := !next_id + 1;
  Ll_test.node_factory ~prec ~first_id:!next_id ~dims:[| n |] () label

let run prec target name delta bases =
  let n = Array.length bases in
  let input = node prec n (name ^ "_input") in
  let source = node prec n (name ^ "_source") in
  let output = node target n (name ^ "_output") in
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let op = if delta < 0 then Ops.Sub else Ops.Add in
  let value =
    LL.Binop (op, (Ll_test.get input at, prec), (LL.Constant (Float.of_int (Int.abs delta)), prec))
  in
  let t = Ll_test.sym () in
  let dst_at = [| Ll_test.iter t |] in
  let llc =
    Ll_test.seq
      (Ll_test.loop_n s n (Ll_test.set source at value))
      (Ll_test.loop_n t n
         (Ll_test.set output dst_at (LL.Unop (Ops.Round_prec, (Ll_test.get source dst_at, prec)))))
  in
  let o = Ll_test.optimize ~materialized:[ input; source; output ] ~name llc in
  List.hd_exn (Ll_test.execute ~name o ~seed:[ (input, bases) ] ~read:[ output ])

let check prec max_e signed =
  let label = Ops.prec_string prec in
  List.iter [ -1; 0; 1 ] ~f:(fun delta ->
      let pairs = ref [] in
      for e = 24 to max_e do
        (* Both tie parities, every significand, and the carry into the next exponent. *)
        for q = 128 to 255 do
          let step = Float.ldexp 1. (e - 7) in
          let midpoint = (Float.of_int q +. 0.5) *. step in
          let rounded = if delta < 0 || (delta = 0 && q land 1 = 0) then q else q + 1 in
          let want = Float.of_int rounded *. step in
          pairs := (midpoint, want) :: !pairs;
          if signed then
            let rounded = if delta > 0 || (delta = 0 && q land 1 = 0) then q else q + 1 in
            pairs := (Float.neg midpoint, Float.neg (Float.of_int rounded *. step)) :: !pairs
        done
      done;
      let bases = Array.of_list (List.map !pairs ~f:fst) in
      let want = Array.of_list (List.map !pairs ~f:snd) in
      let name = Printf.sprintf "ibf_%s_%d" label (delta + 1) in
      let got = run prec Ops.bfloat16 name delta bases in
      p_all2 ~min:(Array.length bases)
        (Printf.sprintf "%s: midpoint offset %d rounds once to bf16" label delta)
        got want ~f:Float.equal)

let edge prec label delta bases want =
  let got = run prec Ops.bfloat16 ("ibf_" ^ label) delta bases in
  p_all2 ~min:(Array.length bases)
    (label ^ ": integer edge values round once to bf16")
    got want ~f:Float.equal

let () =
  check Ops.int32 30 true;
  check Ops.uint32 31 false;
  check Ops.int64 62 true;
  check Ops.uint64 63 false;
  List.iter [ Ops.int64; Ops.uint64 ] ~f:(fun prec ->
      edge prec
        (Ops.prec_string prec ^ "_small")
        0
        [| 0.; 1.; 2.; 127.; 128.; 255.; 256.; 257.; 259.; 511.; 65535. |]
        [| 0.; 1.; 2.; 127.; 128.; 255.; 256.; 256.; 260.; 512.; 65536. |]);
  edge Ops.int64 "int64_min" 0 [| Float.ldexp (-1.) 63 |] [| Float.ldexp (-1.) 63 |];
  edge Ops.int64 "int64_max" 1023 [| Float.ldexp 1. 63 -. 1024. |] [| Float.ldexp 1. 63 |];
  edge Ops.uint64 "uint64_max" 2047 [| Float.ldexp 1. 64 -. 2048. |] [| Float.ldexp 1. 64 |];
  let witness = [| 16842753. |] in
  let half = run Ops.int32 Ops.half "ibf_half" 0 witness in
  p_all "half: the wide integer witness still overflows" (Array.to_list half)
    ~f:(Float.equal Float.infinity);
  let fp8 = run Ops.int32 Ops.fp8 "ibf_fp8" 0 witness in
  p_all "fp8: the wide integer witness still saturates" (Array.to_list fp8) ~f:(Float.equal 57344.)
