(* gh-ocannl-1630: the untuned CPU default gives each direct conv nest per-channel vector
   accumulators ([Schedule.channel_vectors], composed into [Schedule.cpu_preset] behind config
   [cpu_schedule_channel_vectors]). Lowering leaves a conv as the perfect nest [for w.., oc, r..:
   d[w.., oc] += x * k], which the serial rendering localizes into one scalar multiply-add chain per
   output. The preset sinks [oc] below the reduction loops (each output keeps its terms in their
   order), packs the kernel channel-minor at the routine root, privatizes the accumulators into a
   routine-local tile and renders the channel loop as explicit SIMD whose arithmetic keeps the
   scalar chain's expression shape -- so the claim is BITWISE parity with the scalar-chain rendering
   (the key off), on every contraction mode.

   Legs, on cc (every context is an explicit [Context.cpu ()]): 1. Executed parity, per shape
   (padded, valid, stride 2, batch 1, channel counts below, at, above and not a multiple of the
   vector width, a long reduction, conv -> relu -> conv): the default compile equals the key-off
   compile bit for bit, under [cc_backend_fp_contract=off] (pinned in-process) and again under the
   compiler's own contraction ([auto]); two shapes also with the pool forced on and off. The
   fixtures are non-dyadic and vary with every axis. 2. The transform is observed, not inferred: the
   default routine's source holds the explicit SIMD channel loop and the kernel pack, the key-off
   routine's holds neither. 3. Negative controls on the fixtures: a twin whose two outer reduction
   loops are interchanged (another summation order) differs from the reference in some cell, and so
   does the reference compiled under [cc_backend_fp_contract=fast] against [off] -- the fixtures can
   see a reordering and a contraction, so the parity claims have content. 4. Accumulation into a
   destination already holding values (its zero dropped, the cells prefilled by a discriminating
   nest): parity again, through the default pipeline applied by hand. 5. Declines keep the plain
   preset: an f16 conv, a depthwise conv (two operands read the output channel), a kernel written in
   the routine, a matmul (no sliding window) -- each against a positive twin that does take the
   transform; and a fault injected at the pack declines the site when raised typed, but escapes the
   probe when raised untyped (a transform bug). 6. Training: one SGD step of a two-conv network
   gives bitwise-equal kernels, key on vs off. *)

(* Contraction: the claims compare the channel-vector rendering with the scalar chain it replaces,
   whose per-cell expression is the same [acc + x * k]. The process pins
   [cc_backend_fp_contract=off] before the first compile; the [auto] legs and the [fast] control
   switch it around one compile pair each. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Sched = Ir.Schedule
module Generated = Test_utils.Generated

let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name:"cc"
let facts = Context.backend_facts_of_name "cc"

let set_key key v =
  match v with
  | Some data -> Hashtbl.set Utils.config_file_args ~key ~data
  | None -> Hashtbl.remove Utils.config_file_args key

let with_key key v f =
  let saved = Hashtbl.find Utils.config_file_args key in
  set_key key (Some v);
  Exn.protect ~f ~finally:(fun () -> set_key key saved)

let with_contraction mode f =
  Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" mode;
  Exn.protect ~f ~finally:(fun () -> Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off")

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

(* Non-dyadic values in about [-0.7, 0.7], varying with every axis. *)
let reals ~weights idcs =
  Ll_test.weighted ~weights ~modulus:11 ~offset:(-4.9) ~stride:(1. /. 7.) idcs

let image ?(prec = Ir.Ops.single) ~tag ~b ~h ~w ~ic () =
  let dims = Array.of_list (b @ [ h; w; ic ]) in
  let weights = Array.of_list (List.map b ~f:(fun _ -> 5) @ [ 3; 7; 2 ]) in
  let nd = Ir.Ndarray.init_array ~debug:(tag ^ "x") prec ~dims ~padding:None ~f:(reals ~weights) in
  NTDSL.reshape ~l:(tag ^ "x") ~b ~o:[ h; w; ic ] nd ()

let kernel ?(prec = Ir.Ops.single) ~tag ~ic ~oc () =
  let nd =
    Ir.Ndarray.init_array ~debug:(tag ^ "k") prec ~dims:[| oc; 3; 3; ic |] ~padding:None
      ~f:(reals ~weights:[| 4; 9; 5; 3 |])
  in
  NTDSL.reshape ~l:(tag ^ "k") ~i:[ 3; 3; ic ] ~o:[ oc ] nd ()

let spec ~padded ~stride =
  let m = if padded then "=+" else "<+" in
  Printf.sprintf
    "...| %d*oh%skh, %d*ow%skw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." stride m
    stride m

let conv ?(stride = 1) ~padded ~label x k =
  NTDSL.O.einsum ~label:[ label ] (spec ~padded ~stride) x k

type shape = {
  tag : string;
  b : int list;
  hw : int;
  ic : int;
  oc : int;
  padded : bool;
  stride : int;
  chained : bool;
}

(* A fresh graph per compile: both twins mint their own nodes over the same fixture values. *)
let build s =
  let x = image ~tag:s.tag ~b:s.b ~h:s.hw ~w:s.hw ~ic:s.ic () in
  let k = kernel ~tag:s.tag ~ic:s.ic ~oc:s.oc () in
  let y = conv ~stride:s.stride ~padded:s.padded ~label:(s.tag ^ "y") x k in
  if not s.chained then (y, k)
  else
    let k2 = kernel ~tag:(s.tag ^ "2") ~ic:s.oc ~oc:s.oc () in
    let r = NTDSL.O.relu y in
    (conv ~padded:true ~label:(s.tag ^ "y2") r k2, k)

let bits a = Array.map a ~f:Int64.bits_of_float
let same a b = Array.equal Int64.equal (bits a) (bits b)

(* Compile [t]'s forward under [name] through the untuned default (or [lowered_transform]), run it
   once, and return its values. *)
let run ?lowered_transform ~name t =
  let ctx = Context.cpu () in
  let comp = named name (Train.forward t) in
  let ctx, routine = Context.compile ?lowered_transform ~name ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

let source name =
  (* A fissioned routine writes its segments to one source file. *)
  let seg = name ^ "__seg" in
  if Stdlib.Sys.file_exists (Generated.path seg) then Generated.read seg else Generated.read name

let compile_pair ~name s =
  let on =
    with_key "cpu_schedule_channel_vectors" "true" (fun () ->
        run ~name:(name ^ "_on") (fst (build s)))
  in
  let off =
    with_key "cpu_schedule_channel_vectors" "false" (fun () ->
        run ~name:(name ^ "_off") (fst (build s)))
  in
  (on, off)

let simd_loop = "Vectorized rendering"
let pack_prefix = "tile_"

let shapes =
  [
    { tag = "pad"; b = [ 2 ]; hw = 6; ic = 5; oc = 12; padded = true; stride = 1; chained = false };
    { tag = "tail"; b = [ 2 ]; hw = 5; ic = 4; oc = 13; padded = true; stride = 1; chained = false };
    {
      tag = "narrow";
      b = [ 2 ];
      hw = 6;
      ic = 3;
      oc = 4;
      padded = false;
      stride = 1;
      chained = false;
    };
    { tag = "wide"; b = [ 2 ]; hw = 4; ic = 8; oc = 40; padded = true; stride = 1; chained = false };
    { tag = "s2"; b = [ 2 ]; hw = 8; ic = 4; oc = 16; padded = true; stride = 2; chained = false };
    { tag = "b1"; b = [ 1 ]; hw = 6; ic = 4; oc = 8; padded = true; stride = 1; chained = false };
    { tag = "chain"; b = [ 2 ]; hw = 6; ic = 3; oc = 8; padded = true; stride = 1; chained = true };
  ]

(* Leg 1 and 2. *)
let () =
  List.iter shapes ~f:(fun s ->
      let on, off = compile_pair ~name:("cv_" ^ s.tag) s in
      p
        (s.tag ^ ": the channel-vector conv equals the scalar chain bitwise (contraction off)")
        (same on off);
      p
        (s.tag
       ^ ": the default routine renders the channel loop as explicit SIMD over a packed kernel")
        (let src = source ("cv_" ^ s.tag ^ "_on") in
         String.is_substring src ~substring:simd_loop
         && String.is_substring src ~substring:(pack_prefix ^ s.tag ^ "k"));
      p
        (s.tag ^ ": the key-off routine renders neither")
        (let src = source ("cv_" ^ s.tag ^ "_off") in
         (not (String.is_substring src ~substring:simd_loop))
         && not (String.is_substring src ~substring:pack_prefix));
      let on, off = with_contraction "auto" (fun () -> compile_pair ~name:("cva_" ^ s.tag) s) in
      p
        (s.tag ^ ": the channel-vector conv equals the scalar chain bitwise (compiler contraction)")
        (same on off));
  (* The pool forced on (no work gate) and off (an unreachable gate). *)
  List.iter
    [ ("0", "pool"); ("1000000000000", "serial") ]
    ~f:(fun (mw, how) ->
      List.iter (List.take shapes 2) ~f:(fun s ->
          let on, off =
            with_key "cpu_schedule_min_work" mw (fun () ->
                compile_pair ~name:("cv" ^ how ^ "_" ^ s.tag) s)
          in
          p
            (s.tag ^ " (" ^ how ^ "): the channel-vector conv equals the scalar chain bitwise")
            (same on off)))

(* The optimized program a real compile of [t]'s forward hands the default pipeline. *)
let optimized_of ~name (t : Tensor.t) =
  let captured = ref None in
  ignore
    (run ~name
       ~lowered_transform:(fun o ->
         captured := Some o;
         Sched.maybe_default_schedules ~facts ~static_indices:[] o)
       t);
  Option.value_exn ~here:[%here] !captured

let swaps_of sched =
  List.filter_map sched ~f:(function Sched.Swap { inner; _ } -> Some inner | _ -> None)

(* Leg 3: the fixtures see a reordering and a contraction. *)
let () =
  let s = List.hd_exn shapes in
  let reference =
    with_key "cpu_schedule_channel_vectors" "false" (fun () -> run ~name:"cv_ref" (fst (build s)))
  in
  let reordered =
    run ~name:"cv_reorder"
      ~lowered_transform:(fun o ->
        match swaps_of (Sched.channel_vectors ~base:[] o) with
        | r1 :: r2 :: _ ->
            [ Sched.apply (Sched.default_cpu o @ [ Sched.Swap { outer = r1; inner = r2 } ]) o ]
        | _ -> failwith "cv_reorder: no conv site with two reduction loops")
      (fst (build s))
  in
  p "control: interchanging two reduction loops changes some cell (the fixture sees the order)"
    (not (same reference reordered));
  let fused =
    with_contraction "fast" (fun () ->
        with_key "cpu_schedule_channel_vectors" "false" (fun () ->
            run ~name:"cv_fused" (fst (build s))))
  in
  p "control: compiler contraction changes some cell (the fixture sees a fused multiply-add)"
    (not (same reference fused))

(* Leg 4: a destination already holding values. The conv's zero is dropped and a discriminating nest
   prefills its cells, then the default pipeline runs by hand -- key on against key off. *)
let () =
  let s = List.hd_exn shapes in
  let prefill (o : LL.optimized) =
    let d =
      List.find_map_exn (LL.flat_lines [ o.llc ]) ~f:(function
        | LL.Zero_out tn -> Some tn
        | _ -> None)
    in
    let dims = Lazy.force d.Tn.dims in
    let syms = Array.map dims ~f:(fun _ -> Ll_test.sym ()) in
    (* [0.3 + sum_i 0.013 * (i + 2) * idx_i]: non-dyadic, varying with every axis. *)
    let value =
      Array.foldi syms ~init:(Ll_test.c 0.3) ~f:(fun i acc s ->
          Ll_test.add acc
            (Ll_test.mul (Ll_test.embed s) (Ll_test.c (0.013 *. Float.of_int (i + 2)))))
    in
    let nest =
      Array.foldi syms
        ~init:(Ll_test.set d (Array.map syms ~f:Ll_test.iter) value)
        ~f:(fun i body _ ->
          let k = Array.length syms - 1 - i in
          Ll_test.loop_n syms.(k) dims.(k) body)
    in
    let llc =
      LL.unflat_lines
        (List.concat_map (LL.flat_lines [ o.llc ]) ~f:(function
          | LL.Zero_out tn when Tn.equal tn d -> [ nest ]
          | stmt -> [ stmt ]))
    in
    (LL.get_node o.traced_store d).LL.zero_initialized_by_code <- true;
    { o with llc }
  in
  let go key name =
    with_key "cpu_schedule_channel_vectors" key (fun () ->
        run ~name
          ~lowered_transform:(fun o ->
            Sched.maybe_default_schedules ~facts ~static_indices:[] (prefill o))
          (fst (build s)))
  in
  let on = go "true" "cv_acc_on" and off = go "false" "cv_acc_off" in
  p "accumulating into a prefilled destination: channel vectors equal the scalar chain bitwise"
    (same on off);
  p "the prefilled leg took the transform"
    (String.is_substring (source "cv_acc_on") ~substring:simd_loop)

(* Leg 5: declines, each against a positive twin. *)
let () =
  let sites ~name t = Sched.channel_vectors ~base:[] (optimized_of ~name t) in
  let s = List.hd_exn shapes in
  let y, k = build s in
  let pos = sites ~name:"cv_pos" y in
  let opt = optimized_of ~name:"cv_written" y in
  p_empty "a kernel written in the routine declines (the pack would read it mid-routine)" ~over:pos
    (Sched.channel_vectors ~base:[]
       { opt with llc = Ll_test.seq (Ll_test.zero k.Tensor.value) opt.llc });
  let hx = image ~prec:Ir.Ops.half ~tag:"h" ~b:[ 2 ] ~h:6 ~w:6 ~ic:5 () in
  let hk = kernel ~prec:Ir.Ops.half ~tag:"h" ~ic:5 ~oc:12 () in
  p_empty "an f16 conv declines (narrow storage keeps the scalar chain)" ~over:pos
    (sites ~name:"cv_half" (conv ~padded:true ~label:"hy" hx hk));
  let dx = image ~tag:"dw" ~b:[ 2 ] ~h:6 ~w:6 ~ic:8 () in
  let dk =
    NTDSL.reshape ~l:"dwk" ~i:[ 3; 3 ] ~o:[ 8 ]
      (Ir.Ndarray.init_array ~debug:"dwk" Ir.Ops.single ~dims:[| 8; 3; 3 |] ~padding:None
         ~f:(reals ~weights:[| 4; 9; 5 |]))
      ()
  in
  let dy =
    NTDSL.O.einsum ~label:[ "dwy" ] "...| 1*oh=+kh, 1*ow=+kw, c; |kh, kw -> c => ...| oh, ow, c" dx
      dk
  in
  p_empty "a depthwise conv declines (two operands read the output channel)" ~over:pos
    (sites ~name:"cv_dw" dy);
  let ma =
    NTDSL.reshape ~l:"ma" ~b:[ 16 ] ~o:[ 32 ]
      (Ir.Ndarray.init_array ~debug:"ma" Ir.Ops.single ~dims:[| 16; 32 |] ~padding:None
         ~f:(reals ~weights:[| 3; 7 |]))
      ()
  in
  let mb =
    NTDSL.reshape ~l:"mb" ~i:[ 32 ] ~o:[ 24 ]
      (Ir.Ndarray.init_array ~debug:"mb" Ir.Ops.single ~dims:[| 24; 32 |] ~padding:None
         ~f:(reals ~weights:[| 2; 5 |]))
      ()
  in
  p_empty "a matmul declines (no sliding window)" ~over:pos (sites ~name:"cv_mm" NTDSL.O.(mb * ma));
  (* Only an op's typed refusal declines a site (gh-ocannl-1323): the same fault raised typed drops
     the site, raised untyped -- what a transform bug raises -- escapes the probe. *)
  let opt = optimized_of ~name:"cv_fault" y in
  let at_pack fault = function Sched.Stage _ -> fault () | _ -> () in
  let typed () =
    raise
      (Ir.Schedule_outcome.Cause_at
         ( Ir.Schedule_outcome.Transform,
           Ir.Schedule_outcome.Illegal_schedule { check = "test"; detail = "injected" } ))
  in
  p_empty "an op's typed refusal declines the site" ~over:pos
    (Sched.with_transform_fault (at_pack typed) (fun () -> Sched.channel_vectors ~base:[] opt));
  p "an untyped raise inside a transform escapes the probe"
    (match
       Sched.with_transform_fault
         (at_pack (fun () -> invalid_arg "injected transform bug"))
         (fun () -> Sched.channel_vectors ~base:[] opt)
     with
    | _ -> false
    | exception Invalid_argument _ -> true)

(* Leg 6: one SGD step of a two-conv network, key on vs off. *)
let () =
  let step key =
    with_key "cpu_schedule_channel_vectors" key (fun () ->
        let x = image ~tag:("tr" ^ key) ~b:[ 2 ] ~h:6 ~w:6 ~ic:4 () in
        let param tag ic oc =
          Operation.init ~l:(tag ^ key) ~prec:Ir.Ops.single ~i:[ 3; 3; ic ] ~o:[ oc ]
            ~f:(reals ~weights:[| 4; 9; 5; 3 |])
            ~grad_spec:Tensor.Require_grad ()
        in
        let k1 = param "trk1" 4 8 and k2 = param "trk2" 8 8 in
        let tconv label x k = TDSL.O.einsum ~label:[ label ] (spec ~padded:true ~stride:1) x k in
        let y = tconv ("tr1" ^ key) x k1 in
        let y2 = tconv ("tr2" ^ key) (TDSL.O.relu y) k2 in
        let%op loss = y2 ++ "...|... => 0" in
        let update = Train.grad_update loss in
        let params = Set.of_list (module Tensor) [ k1; k2 ] in
        let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.01) ~params loss in
        let ctx = Context.cpu () in
        let ctx = Train.init_params ctx Idx.Empty loss in
        let comp = named ("cv_train_" ^ key) (Ir.Assignments.sequence [ update; sgd ]) in
        let ctx, routine = Context.compile ctx comp Idx.Empty in
        let ctx = Context.run ctx routine in
        (Context.get_values ctx k1.Tensor.value, Context.get_values ctx k2.Tensor.value))
  in
  let k1on, k2on = step "true" and k1off, k2off = step "false" in
  p "training step: both kernels after one SGD step are bitwise equal, key on vs off"
    (same k1on k1off && same k2on k2off);
  p "training step: the default step took the transform"
    (String.is_substring (source "cv_train_true") ~substring:simd_loop)
