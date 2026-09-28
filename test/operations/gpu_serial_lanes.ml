(* gh-ocannl-1003 stage 1: the default GPU annotator's lane geometry. The online-softmax rewrite
   hoists the probability read out of the value width, leaving the value pass as

   for b, s, h { for t { p := P[b, s, h, t]; for e { O[b, s, h, e] += p * V[b, t, h, e] } } }

   whose plain path stops at the preamble: the presets took two of (b, s, h) and ran the [e] loop
   serially inside every thread. The lane geometry is Grid (b, s, h) -> Serial t -> Workgroup e: a
   lane recomputes the uniform [p] in its own register and owns its [O] cells.

   Legs: 1. the hoisted shape gets the lane geometry -- every Grid slot and the lane width, the lane
   bound inside the serial loop -- and executes every cell exactly (integer-valued data, so the
   comparison is exact on every backend); 2. a lane nest sharing its kernel with a plain nest
   declines (the slots would not line up) and keeps the presets' geometry, values exact; 3. a lane
   body reading another lane's cell declines (a cross-thread conflict), values exact under the
   serial order; 4. a preamble holding an inlined reduction declines (every lane would recompute
   it); 5. the real pipeline: the rewritten attention's value pass is scheduled with lanes on a GPU
   backend and not on the CPU one, and the forward agrees with the composed model on the run's
   backend. The emitted kernel's lane binding sits inside the serial loop (a GPU backend's generated
   source; skipped on cc, which renders hardware loops serially). *)

open Base
open Stdio
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name

(* The register a lane binds on each GPU dialect; [None] on the C backends, which render hardware
   loops serially. *)
let lane_register =
  match backend_name with
  | "metal" -> Some "lid.x"
  | "cuda" | "hip" -> Some "threadIdx.x"
  | _ -> None

let b_n = 2
let s_n = 16
let h_n = 4
let t_n = 16
let e_n = 32
let node = L.node_factory ~first_id:1003000 ~dims:[| b_n; s_n; h_n; e_n |] ()

(* Integer-valued operands varying with every symbol of their iteration: every product and partial
   sum below stays an exact float, so the device's summation order is observable only through which
   cells it adds, never through rounding. *)
let p_value b s h t = Float.of_int (1 + ((b + (2 * s) + (3 * h) + (5 * t)) % 7))
let v_value b t h e = Float.of_int (1 + (((7 * b) + (3 * h) + (5 * t) + e) % 11))
let o_seed b s h e = Float.of_int ((((((b * s_n) + s) * h_n) + h) * e_n) + e)

let fill ~dims f =
  let n = Array.fold dims ~init:1 ~f:( * ) in
  Array.init n ~f:(fun i -> f (L.unflat ~dims i))

let p_dims = [| b_n; s_n; h_n; t_n |]
let v_dims = [| b_n; t_n; h_n; e_n |]
let o_dims = [| b_n; s_n; h_n; e_n |]

type lane_case = {
  name : string;
  opt : LL.optimized;
  seed : (Ir.Tnode.t * float array) list;
  read : Ir.Tnode.t list;
  expected : float array list;
}

(* The hoisted value pass over [P], [V] into [O], with the lane body [body] (given the local [p] and
   the symbols), optionally followed by a plain elementwise nest writing [Q]. *)
let hoisted ~name ~lane_body ~reference ?(with_plain = false) () =
  let p_tn = node ~dims:p_dims (name ^ "_p")
  and v_tn = node ~dims:v_dims (name ^ "_v")
  and o_tn = node ~dims:o_dims (name ^ "_o")
  and q_tn = node ~dims:o_dims (name ^ "_q")
  and p_local = node ~dims:[| 1 |] (name ^ "_prob") in
  L.virtualize p_local;
  List.iter [ p_tn; v_tn; o_tn; q_tn ] ~f:L.materialize;
  let p = LL.get_scope p_local in
  let b = L.sym () and s = L.sym () and h = L.sym () and t = L.sym () and e = L.sym () in
  let value_pass =
    L.loop_n b b_n @@ L.loop_n s s_n @@ L.loop_n h h_n @@ L.loop_n t t_n
    @@ LL.unflat_lines
         [
           LL.Declare_local { id = p; needs_init = false };
           LL.Set_local (p, L.get p_tn [| L.iter b; L.iter s; L.iter h; L.iter t |]);
           L.loop_n e e_n (lane_body ~o_tn ~v_tn ~p ~b ~s ~h ~t ~e);
         ]
  in
  let plain =
    let b = L.sym () and s = L.sym () and h = L.sym () and e = L.sym () in
    L.loop_n b b_n @@ L.loop_n s s_n @@ L.loop_n h h_n @@ L.loop_n e e_n
    @@ L.set q_tn
         [| L.iter b; L.iter s; L.iter h; L.iter e |]
         (L.add (L.mul (L.c 100.) (L.tag b s)) (L.tag h e))
  in
  let llc = if with_plain then L.seq value_pass plain else value_pass in
  let materialized = [ p_tn; v_tn; o_tn ] @ if with_plain then [ q_tn ] else [] in
  let opt = L.optimize ~materialized ~name llc in
  let seed =
    [
      (p_tn, fill ~dims:p_dims (fun i -> p_value i.(0) i.(1) i.(2) i.(3)));
      (v_tn, fill ~dims:v_dims (fun i -> v_value i.(0) i.(1) i.(2) i.(3)));
      (o_tn, fill ~dims:o_dims (fun i -> o_seed i.(0) i.(1) i.(2) i.(3)));
    ]
  in
  let expected_o = reference () in
  let expected_q =
    fill ~dims:o_dims (fun i ->
        let tag x y = Float.of_int (1 + (10 * x) + y) in
        (100. *. tag i.(0) i.(1)) +. tag i.(2) i.(3))
  in
  {
    name;
    opt;
    seed;
    read = o_tn :: (if with_plain then [ q_tn ] else []);
    expected = expected_o :: (if with_plain then [ expected_q ] else []);
  }

(* [O[b, s, h, e] += p * V[b, t, h, e]], in the serial order: t ascending per cell. *)
let accumulate ~o_tn ~v_tn ~p ~b ~s ~h ~t ~e =
  let o_idx = [| L.iter b; L.iter s; L.iter h; L.iter e |] in
  L.set o_tn o_idx
    (L.add (L.get o_tn o_idx)
       (L.mul (LL.Get_local p) (L.get v_tn [| L.iter b; L.iter t; L.iter h; L.iter e |])))

let accumulate_reference () =
  fill ~dims:o_dims (fun i ->
      let b, s, h, e = (i.(0), i.(1), i.(2), i.(3)) in
      List.fold (List.range 0 t_n) ~init:(o_seed b s h e) ~f:(fun acc t ->
          acc +. (p_value b s h t *. v_value b t h e)))

(* [O[b, s, h, e] := O[b, s, h, E-1-e] + p]: lane [e] reads the cell lane [E-1-e] writes. *)
let mirror ~o_tn ~v_tn:_ ~p ~b ~s ~h ~t:_ ~e =
  L.set o_tn
    [| L.iter b; L.iter s; L.iter h; L.iter e |]
    (L.add
       (L.get o_tn [| L.iter b; L.iter s; L.iter h; L.aff [ (-1, e) ] (e_n - 1) |])
       (LL.Get_local p))

let mirror_reference () =
  let o = fill ~dims:o_dims (fun i -> o_seed i.(0) i.(1) i.(2) i.(3)) in
  for b = 0 to b_n - 1 do
    for s = 0 to s_n - 1 do
      for h = 0 to h_n - 1 do
        for t = 0 to t_n - 1 do
          let p = p_value b s h t in
          for e = 0 to e_n - 1 do
            let at e = L.flat ~dims:o_dims [| b; s; h; e |] in
            o.(at e) <- o.(at (e_n - 1 - e)) +. p
          done
        done
      done
    done
  done;
  o

(* Whether [llc] holds the lane shape: a Serial loop whose body is scope-local work ahead of one
   loop of axis [inner] (default [Workgroup]: the lane geometry; [Serial]: the unannotated
   hoist). *)
let lane_under_serial ?(inner = LL.Workgroup) (llc : LL.t) =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec go (llc : LL.t) =
    match llc with
    | LL.For_loop { axis = LL.Serial; body; _ } -> (
        match List.rev (strip (LL.flat_lines [ body ])) with
        | LL.For_loop { axis; _ } :: preamble
          when LL.equal_axis_type axis inner
               && (not (List.is_empty preamble))
               && List.for_all preamble ~f:(function
                 | LL.Declare_local _ | LL.Set_local _ -> true
                 | _ -> false) ->
            true
        | _ -> go body)
    | LL.For_loop { body; _ } | LL.If { body; _ } | LL.Scan_loop { body; _ } -> go body
    | LL.Seq (a, b) -> go a || go b
    | _ -> false
  in
  go llc

let dims_are (d : LL.launch_dims) ~grid ~block =
  Array.equal Int.equal d.LL.grid grid && Array.equal Int.equal d.LL.block block

let execute case =
  let scheduled = S.apply (S.default_gpu ~block_size:256 ~min_parallel:64 case.opt) case.opt in
  let got = L.execute ~name:case.name scheduled ~seed:case.seed ~read:case.read in
  (scheduled, got)

let check_values case got =
  List.iter2_exn (List.zip_exn case.read got) case.expected ~f:(fun (tn, got) want ->
      p_all2
        (Printf.sprintf "%s: every cell of %s holds the serial order's exact value" case.name
           (Ir.Tnode.debug_name tn))
        got want ~f:Float.equal)

(* The rewritten attention, for leg 5 (online_softmax.ml's model at a size the presets alone leave
   under-parallel). One head of width 32 is above the recompute cap
   ([virtualize_max_inline_reduction] = 16), so the scores are stored and the value pass's preamble
   reads them; four heads of width 8 are under it, so the preamble recomputes [q . k] inline. *)
let batch = 2
let seq = 16
let d_model = 32

let attention ~heads =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let block =
    Nn_blocks.multi_head_attention ~label:[ "lanes" ] ~num_heads:heads ~d_k:(d_model / heads)
      ~d_v:(d_model / heads) ()
  in
  let%op y = x + block ~train_step:None ~mask x in
  y

let forward ~heads ~on =
  Tensor.unsafe_reinitialize ();
  Ir.Online_softmax.set_enabled (Some on);
  let t = attention ~heads in
  let ctx = Train.forward_once (Context.auto ()) t in
  let values = Context.get_values ctx t.Tensor.value in
  let optimized =
    Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
      ~cd_source:None ~name:"lanes_probe" [] t.Tensor.forward.Ir.Assignments.asgns
  in
  Ir.Online_softmax.set_enabled None;
  (values, optimized)

let () =
  eprintf "gpu_serial_lanes backend: %s (not part of the golden)\n%!" backend_name;
  printf "--- leg 1: the hoisted value pass takes the lane geometry ---\n";
  let case =
    hoisted ~name:"lanes_hoisted" ~lane_body:accumulate ~reference:accumulate_reference ()
  in
  p "lanes_hoisted: the optimized nest keeps the hoisted shape"
    (lane_under_serial ~inner:LL.Serial case.opt.llc);
  let scheduled, got = execute case in
  p "lanes_hoisted: Grid (h, s, b) on .x, .y, .z and a 32-wide lane"
    (dims_are (LL.launch_dims scheduled.llc) ~grid:[| h_n; s_n; b_n |] ~block:[| e_n; 1; 1 |]);
  p "lanes_hoisted: the Workgroup lane sits inside the serial key loop, past the preamble"
    (lane_under_serial scheduled.llc);
  check_values case got;
  let claim = "lanes_hoisted: the emitted kernel binds the lane inside the serial loop" in
  (match lane_register with
  | None -> skipped ~backend:backend_name claim
  | Some register ->
      let src = Generated.read case.name in
      let first_for = String.substr_index src ~pattern:"for (" in
      let lane = String.substr_index src ~pattern:register in
      p claim (match (first_for, lane) with Some f, Some l -> f < l | _ -> false));

  printf "--- leg 2: a lane nest sharing its kernel with a plain nest keeps the presets ---\n";
  let case =
    hoisted ~name:"lanes_mixed" ~lane_body:accumulate ~reference:accumulate_reference
      ~with_plain:true ()
  in
  let scheduled, got = execute case in
  p "lanes_mixed: no lane geometry" (not (lane_under_serial scheduled.llc));
  p "lanes_mixed: the presets' suffix pair, Grid s x Workgroup h"
    (dims_are (LL.launch_dims scheduled.llc) ~grid:[| s_n; 1; 1 |] ~block:[| h_n; 1; 1 |]);
  check_values case got;

  printf "--- leg 3: a lane reading another lane's cell declines ---\n";
  let case = hoisted ~name:"lanes_mirror" ~lane_body:mirror ~reference:mirror_reference () in
  let scheduled, got = execute case in
  p "lanes_mirror: no lane geometry" (not (lane_under_serial scheduled.llc));
  p "lanes_mirror: the presets' suffix pair, Grid s x Workgroup h"
    (dims_are (LL.launch_dims scheduled.llc) ~grid:[| s_n; 1; 1 |] ~block:[| h_n; 1; 1 |]);
  check_values case got;

  printf "--- leg 4: a preamble holding an inlined reduction keeps the presets ---\n";
  (* Every lane recomputes the preamble, so a lane geometry multiplies its work by the lane width:
     the recomputed score [q . k] of the flash-attention form cost 1.5x at seq 1024 on Metal that
     way (benchmarks/report-gh1003-stage1.md). The scope is spliced into the optimized record by
     hand -- the annotator reads only the code and the placements -- so the leg is structural. *)
  let case =
    hoisted ~name:"lanes_scoped" ~lane_body:accumulate ~reference:accumulate_reference ()
  in
  let k_tn = node ~dims:v_dims "lanes_scoped_k" and acc = node ~dims:[| 1 |] "lanes_scoped_acc" in
  L.materialize k_tn;
  L.virtualize acc;
  let acc = LL.get_scope acc in
  let rec splice (llc : LL.t) : LL.t =
    match llc with
    | LL.For_loop fc -> LL.For_loop { fc with body = splice fc.body }
    | LL.Seq (a, b) -> LL.Seq (splice a, splice b)
    | LL.Set_local (p, (LL.Get (_, idcs) as read)) ->
        let d = L.sym () in
        let reduce =
          LL.Local_scope
            {
              id = acc;
              body =
                LL.Seq
                  ( LL.Set_local (acc, read),
                    L.loop_n d 4
                      (LL.Set_local
                         ( acc,
                           L.add (LL.Get_local acc)
                             (L.get k_tn [| idcs.(0); idcs.(3); idcs.(2); L.iter d |]) )) );
              orig_indices = [| L.fixed 0 |];
              mint = LL.Schedule_minted;
            }
        in
        LL.Set_local (p, reduce)
    | other -> other
  in
  let scoped = { case.opt with llc = splice case.opt.llc } in
  p "lanes_scoped: the preamble holds the inlined loop" (not (LL.equal scoped.llc case.opt.llc));
  let scheduled = S.apply (S.default_gpu ~block_size:256 ~min_parallel:64 scoped) scoped in
  p "lanes_scoped: no lane geometry" (not (lane_under_serial scheduled.llc));
  p "lanes_scoped: the presets' suffix pair, Grid s x Workgroup h"
    (dims_are (LL.launch_dims scheduled.llc) ~grid:[| s_n; 1; 1 |] ~block:[| h_n; 1; 1 |]);

  printf "--- leg 5: the rewritten attention's value pass, through the real pipeline ---\n";
  List.iter
    [ ("stored scores", 1); ("recomputed scores", 4) ]
    ~f:(fun (form, heads) ->
      let composed, _ = forward ~heads ~on:false in
      let rewritten, optimized = forward ~heads ~on:true in
      let segments backend_name =
        S.maybe_default_schedules ~backend_name ~static_indices:[] optimized
        |> List.map ~f:(fun (o : LL.optimized) -> o.llc)
      in
      (* The CPU pipeline first: the GPU one promotes statement-crossing locals in this lineage. *)
      let cpu = segments "cc" in
      p_exists
        (Printf.sprintf "attention, %s: the lowering carries the hoisted value pass" form)
        [ optimized.llc ]
        ~f:(lane_under_serial ~inner:LL.Serial);
      p_none
        (Printf.sprintf "attention, %s: the CPU preset schedules no lanes" form)
        cpu ~f:lane_under_serial;
      let gpu = segments "metal" in
      if heads = 1 then
        p_exists
          (Printf.sprintf "attention, %s: a GPU segment schedules the value pass with lanes" form)
          gpu ~f:lane_under_serial
      else
        p_none
          (Printf.sprintf
             "attention, %s: no GPU segment takes lanes (every lane would recompute the dot \
              product)"
             form)
          gpu ~f:lane_under_serial;
      let close g w = Float.(abs (g -. w) <= 1e-4 *. max 1. (abs w)) in
      p_all2
        (Printf.sprintf "attention, %s: the rewritten forward agrees with the composed one" form)
        rewritten composed ~f:close)
