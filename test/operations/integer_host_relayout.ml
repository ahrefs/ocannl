(* Same-storage relayouts must preserve raw integer bits, including values a float cannot represent.
   Host shard/gather and Schedule packing run on every backend's host; the padded virtual read
   executes on the selected device. Float-valued Parallel reads are outside the bit-preservation
   contract. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Tn = Ir.Tnode
module A = Stdlib.Bigarray.Genarray

let patterns =
  [|
    0x20000000000001L;
    0x20000000000003L;
    Int64.max_value;
    0x4000000000000001L;
    -1L;
    -3L;
    -1023L;
    -2047L;
  |]

let dims = [| 4; 2 |]
let pads = [| Ops.{ left = 1; right = 2 }; Ops.{ left = 1; right = 1 } |]

let raw prec ~padded =
  let nd =
    Nd.create_array ~debug:"integer_relayout" prec
      ~dims:(if padded then [| 7; 4 |] else dims)
      ~padding:(if padded then Some (pads, 0.) else None)
  in
  (match nd with
  | Nd.Int64_nd a | Nd.Uint64_nd a ->
      Array.iteri patterns ~f:(fun k bits ->
          A.set a (if padded then [| (k / 2) + 1; (k % 2) + 1 |] else [| k / 2; k % 2 |]) bits)
  | _ -> assert false);
  nd

let bits ?padding nd =
  match nd with
  | Nd.Int64_nd a | Nd.Uint64_nd a ->
      Nd.fold_bigarray ?padding a ~init:[] ~f:(fun acc _ v -> v :: acc) |> List.rev |> Array.of_list
  | _ -> assert false

let check label nd = p_all2 label (bits nd) patterns ~f:Int64.equal
let host (t : Tensor.t) = Lazy.force (Option.value_exn (Ir.Host_inits.find t.value))

let packing prec id =
  let nd = raw prec ~padded:false in
  let t = TDSL.rebatch ~l:"pack_literal" nd () in
  let source = t.Tensor.value in
  Tn.set_host_constant source;
  let target = Ll_test.node_factory ~prec ~first_id:id ~dims () "pack_output" in
  let i = Ll_test.sym () and j = Ll_test.sym () in
  let at = [| Ll_test.iter i; Ll_test.iter j |] in
  let llc =
    Ll_test.loop_n i 4 (Ll_test.loop_n j 2 (Ll_test.set target at (Ll_test.get source at)))
  in
  let o =
    Ll_test.optimize ~materialized:[ source; target ]
      ~name:("relayout_pack_" ^ Ops.prec_string prec)
      llc
  in
  let stage tile_prec =
    Ir.Schedule.Stage
      {
        source;
        tile_loops = [ i; j ];
        shared = false;
        cooperative = None;
        hoisted = true;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec;
      }
  in
  let packed = Ir.Schedule.apply [ stage None ] o in
  let panels =
    Hashtbl.keys packed.Ir.Low_level.traced_store
    |> List.filter ~f:(fun tn -> (not (Tn.equal tn source)) && Ir.Host_inits.mem tn)
  in
  p "hoisted stage replaces the integer source with one packed panel" (List.length panels = 1);
  p_all "typed packing preserves every integer bit" panels ~f:(fun tn ->
      let panel = Lazy.force (Option.value_exn (Ir.Host_inits.find tn)) in
      Array.equal Int64.equal (bits panel) patterns);
  let other_precs = List.filter Ops.all_precs ~f:(fun p -> not (Ops.equal_prec p prec)) in
  p_all "tile precision gate refuses integer widening" other_precs ~f:(fun target_prec ->
      match Ir.Schedule.apply [ stage (Some target_prec) ] o with
      | _ -> false
      | exception Invalid_argument msg -> String.is_substring msg ~substring:"not an exact widening");
  p_all "hoisted rounding gate excludes integer sources" Ops.all_precs ~f:(fun round_prec ->
      Option.is_none (Ir.Schedule.hoisted_read_rounding ~source ~round_prec))

let device prec id =
  let node = Ll_test.node_factory ~prec ~first_id:id ~dims () in
  let source = node "relayout_source" in
  let v =
    Tn.create (Tn.Specified prec) ~id:(id + 3) ~label:[ "relayout_virtual" ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy (Some (pads, 5.)))
      ()
  in
  let output = node "relayout_output" in
  let i = Ll_test.sym () and j = Ll_test.sym () in
  let at = [| Ll_test.iter i; Ll_test.iter j |] in
  let padded = [| Ll_test.aff [ (1, i) ] 1; Ll_test.aff [ (1, j) ] 1 |] in
  let k = Ll_test.sym () and l = Ll_test.sym () in
  let output_at = [| Ll_test.iter k; Ll_test.iter l |] in
  let read_at = [| Ll_test.aff [ (1, k) ] 1; Ll_test.aff [ (1, l) ] 1 |] in
  let llc =
    Ll_test.seq
      (Ll_test.loop_n i 4 (Ll_test.loop_n j 2 (Ll_test.set v padded (Ll_test.get source at))))
      (Ll_test.loop_n k 4
         (Ll_test.loop_n l 2 (Ll_test.set output output_at (Ll_test.get v read_at))))
  in
  let name = "relayout_" ^ Ops.prec_string prec in
  let o = Ll_test.optimize ~materialized:[ source; output ] ~name llc in
  let ctx, routine = Ll_test.link ~name o in
  let ctx = Context.from_host ctx source (raw prec ~padded:false) in
  let ctx = Context.run ctx routine in
  p "padded intermediate is virtual"
    (match Tn.Placements.get (Context.placements ctx) v with
    | Some (Tn.Virtual, _) -> true
    | _ -> false);
  let nd = Context.to_host ctx v in
  p_all2 "padded recompute preserves every integer bit" (bits ~padding:pads nd) patterns
    ~f:Int64.equal;
  p "padded recompute restores the neutral fringe"
    (match nd with
    | Nd.Int64_nd a | Nd.Uint64_nd a -> Int64.equal (A.get a [| 0; 0 |]) 5L
    | _ -> false);
  Context.release ctx

let () =
  Stdio.eprintf "integer relayout backend: %s\n%!" (Context.backend_name (Context.auto ()));
  List.iteri [ Ops.int64; Ops.uint64 ] ~f:(fun n prec ->
      let name = Ops.prec_string prec in
      Verdict.case (name ^ " shard/gather") (fun () ->
          let source = raw prec ~padded:false in
          let t = TDSL.rebatch ~l:"relayout_literal" source () in
          let shards = Parallel.shard_along ~axis:0 ~n_shards:2 t in
          p_all2 "shards preserve every integer bit"
            (Array.concat (Array.to_list (Array.map shards ~f:(fun t -> bits (host t)))))
            patterns ~f:Int64.equal;
          (* Gather independent inputs too: shard and gather cannot undo matching mistakes. *)
          let independent =
            Array.init 2 ~f:(fun k ->
                let nd = raw prec ~padded:false in
                match nd with
                | Nd.Int64_nd a ->
                    TDSL.rebatch ~l:"independent" (Nd.Int64_nd (A.sub_left a (k * 2) 2)) ()
                | Nd.Uint64_nd a ->
                    TDSL.rebatch ~l:"independent" (Nd.Uint64_nd (A.sub_left a (k * 2) 2)) ()
                | _ -> assert false)
          in
          check "gather preserves every integer bit" (host (Parallel.gather ~axis:0 independent));
          let padded =
            TDSL.wrap_padded ~l:"padded_literal" ~b:[ 4 ] ~o:[ 2 ] ~padding:pads ~padded_value:0.
              (raw prec ~padded:true) ()
          in
          let shards = Parallel.shard_along ~axis:0 ~n_shards:2 padded in
          check "padded shard/gather preserves the logical cells"
            (host (Parallel.gather ~axis:0 shards)));
      Verdict.case (name ^ " packing") (fun () -> packing prec (54000 + (n * 10)));
      Verdict.case (name ^ " float negative control") (fun () ->
          let nd = raw prec ~padded:false in
          let dst = raw prec ~padded:false in
          Nd.set_from_float dst [| 0; 0 |] (Nd.get_as_float nd [| 0; 0 |]);
          p "float round trip loses odd low bits above 2^53"
            (not (Int64.equal (bits dst).(0) patterns.(0)));
          p "float round trip refuses the upper endpoint rounded out of range"
            (try
               let idx = if Ops.equal_prec prec Ops.uint64 then [| 2; 0 |] else [| 1; 0 |] in
               Nd.set_from_float dst idx (Nd.get_as_float nd idx);
               false
             with Nd.Unrepresentable_integer_value _ -> true));
      Verdict.case (name ^ " device recompute") (fun () -> device prec (55000 + (n * 10))))

(* Equal, small values make the old float-based gather succeed: these controls pin refusal of an
   incompatible layout/precision, independently of the edge-value conversion failures. *)
let literal prec rows =
  let nd =
    Nd.create_array ~debug:"relayout_refusal" prec ~dims:[| rows; 2 |] ~padding:(Some ([||], 1.))
  in
  TDSL.rebatch ~l:"relayout_refusal" nd ()

let () =
  Verdict.case "incompatible gather" (fun () ->
      p "gather refuses mixed storage precisions"
        (match Parallel.gather ~axis:0 [| literal Ops.int64 2; literal Ops.uint64 2 |] with
        | _ -> false
        | exception Invalid_argument msg -> String.is_substring msg ~substring:"precision mismatch");
      p "gather refuses unequal batch lengths"
        (match Parallel.gather ~axis:0 [| literal Ops.int64 2; literal Ops.int64 3 |] with
        | _ -> false
        | exception Invalid_argument msg -> String.is_substring msg ~substring:"incompatible shape"))
