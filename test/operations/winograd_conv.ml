(* gh-ocannl-505: the Winograd F(2x2, 3x3) convolution rewrite ([Ir.Winograd]), a member of the
   algebraic-rewrite tier over raw lowered code. A 3x3 stride-1 conv's accumulation nest becomes a
   kernel transform, an input-tile transform, 16 per-position channel contractions (ordinary
   products, for the matmul sketches) and an inverse transform accumulated into the output; an odd
   output extent keeps its last row and column in the direct form.

   Every executed leg compares the rewritten conv against the SAME conv in the direct form (the gate
   off), with the session reinitialized before each build so both mint the same nodes.

   Legs: 1. Exact parity on bounded integer fixtures, per shape: padded and valid convs, odd output
   extents (the remainder strips), a non-square image, a batch of one (a fixed index, which
   [Autotune.detect_conv] refuses and this rewrite carries through), and two chained conv sites. The
   inputs lie in [-4, 4] and the kernel in [-3, 3], varying with every axis (batch, both spatial
   axes, channel; each kernel tap and both channels), so a transposed kernel, a shifted tile or a
   wrong sign changes some cell. Exactness is argued, not assumed: every coefficient of the
   transforms is dyadic, the kernel transform scales by at most 1/4, and every partial sum of either
   form is a multiple of 1/4 below 2^20 in magnitude (the second chained conv's inputs are the
   first's outputs, at most 9 * 5 * 12 = 540), so both forms compute their sums exactly in f32 and
   must agree BITWISE. Structurally per shape: with the gate off, the direct nest covers every
   output cell and [Autotune.detect_conv] reports the 3x3 stride-1 site (except at batch one, its
   documented singleton refusal); rewritten, the direct form covers only the remainder strips, the
   three transformed buffers are written, no full conv site is left, and the contraction is a matmul
   site writing the product buffer. The pass is idempotent. 2. Tolerance parity on non-dyadic
   inputs: two-sided per cell, at an explicitly looser tier than the conv sketches' reassociation
   (the transformed-domain arithmetic rounds differently, not just in another order); the digits go
   to stderr. 3. The kernel transform is recomputed every call: one compiled routine, run, the
   kernel's values replaced through [Context.set_values], run again -- a transform computed once (or
   cached across calls) would serve the old kernel. 4. Training: a two-conv padded network's
   parameter gradients, gate on against gate off, within tolerance (the backward stays direct; the
   forward it reads is rewritten). 5. The tuned path: [Autotune.tune] over a rewritten conv seeds
   matmul sketches for the contractions, and the tuned routine matches the direct form within
   tolerance. 6. Declines: 5x5, stride 2, dilation 2, a 3x1 window and an f16 conv keep their direct
   nest and mint nothing; on hand-built raw lowerings, a conv whose batch axis also reads the window
   symbol (not separable) is left alone, and an accumulation into a destination holding nonzero
   values keeps them (the rewrite accumulates, it does not overwrite). *)

open Base
open Stdio
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Winograd = Ir.Winograd

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

(* {1 Fixtures} *)

(* Bounded integers varying with every axis: [(sum_i w_i * idx_i) mod m - off]. *)
let ints ~weights ~m ~off idcs =
  Ll_test.weighted ~weights ~modulus:m ~offset:(Float.of_int (-off)) ~stride:1. idcs

(* Non-dyadic values in about [-0.7, 0.7]: [((sum_i w_i * idx_i) mod 11 - 4.9) / 7]. *)
let reals ~weights idcs =
  Ll_test.weighted ~weights ~modulus:11 ~offset:(-4.9) ~stride:(1. /. 7.) idcs

let image ~tag ~b ~h ~w ~ic ~f =
  let dims = Array.of_list (b @ [ h; w; ic ]) in
  let nd = Ir.Ndarray.init_array ~debug:(tag ^ "x") Ir.Ops.single ~dims ~padding:None ~f in
  NTDSL.reshape ~l:(tag ^ "x") ~b ~o:[ h; w; ic ] nd ()

let kernel ?(kh = 3) ?(kw = 3) ?(prec = Ir.Ops.single) ~tag ~ic ~oc ~f () =
  let nd =
    Ir.Ndarray.init_array ~debug:(tag ^ "k") prec ~dims:[| oc; kh; kw; ic |] ~padding:None ~f
  in
  NTDSL.reshape ~l:(tag ^ "k") ~i:[ kh; kw; ic ] ~o:[ oc ] nd ()

let spec ~padded ~stride ~dilation =
  let m = if padded then "=+" else "<+" in
  let d = if dilation = 1 then "" else Int.to_string dilation ^ "*" in
  Printf.sprintf
    "...| %d*oh%s%skh, %d*ow%s%skw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." stride
    m d stride m d

let conv ?(stride = 1) ?(dilation = 1) ~padded ~label x k =
  NTDSL.O.einsum ~label:[ label ] (spec ~padded ~stride ~dilation) x k

(* {1 Structure} *)

(* The output cells the DIRECT form computes: over every perfect nest whose [Set] writes [d] and
   reads [kern], the product of its loop extents divided by the reduction extents (window and
   channel), i.e. the cells it covers. *)
let direct_cells (llc : LL.t) ~(d : Tn.t) ~(kern : Tn.t) ~reduction =
  let rec reads_kern (sc : LL.scalar_t) =
    match sc with
    | LL.Get (tn, _) -> Tn.equal tn kern
    | LL.Binop (_, (a, _), (b, _)) -> reads_kern a || reads_kern b
    | LL.Ternop (_, (a, _), (b, _), (c, _)) -> reads_kern a || reads_kern b || reads_kern c
    | LL.Unop (_, (a, _)) -> reads_kern a
    | _ -> false
  in
  let rec go extents (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) -> go extents a + go extents b
    | LL.For_loop { from_; to_; body; _ } -> go ((to_ - from_ + 1) :: extents) body
    | LL.Set { tn; llsc; _ } when Tn.equal tn d && reads_kern llsc ->
        List.fold extents ~init:1 ~f:( * ) / reduction
    | _ -> 0
  in
  go [] llc

let written (llc : LL.t) =
  LL.affine_accesses llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)

let roles_written llc =
  List.filter_map (written llc) ~f:Winograd.minted_role |> List.dedup_and_sort ~compare:Poly.compare

let all_roles = List.dedup_and_sort ~compare:Poly.compare [ `Kernel; `Input; `Product ]

(* A conv site [detect_conv] reports with a 3-wide window, stride 1 and dilation 1 on both axes:
   what the rewrite is meant to claim. *)
let eligible_conv llc =
  Option.filter (Autotune.detect_conv llc) ~f:(fun s ->
      List.length s.Autotune.c_axes = 2
      && List.for_all s.Autotune.c_axes ~f:(fun cx ->
          cx.Autotune.cx_nk = 3 && cx.cx_stride = 1 && cx.cx_dilation = 1))

let optimized_of (t : Tensor.t) =
  Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
    ~cd_source:None ~name:"wino_probe" [] t.Tensor.forward.Ir.Assignments.asgns

(* {1 Runs} *)

type run = {
  values : float array;
  raw : LL.t;  (** The raw lowering, ahead of the rewrite tier. *)
  opt : LL.optimized;
  d : Tn.t;
  kern : Tn.t;
}

(* Build [build ()] (returning the output, the conv node checked structurally, and that conv's
   kernel) and run its forward with the rewrite [on] or off. *)
let forward ~on build =
  Tensor.unsafe_reinitialize ();
  Winograd.set_enabled (Some on);
  let y, d, kern = build () in
  let ctx = Train.forward_once (Context.auto ()) y in
  let values = Context.get_values ctx y.Tensor.value in
  let raw = Ir.Assignments.to_low_level y.Tensor.forward.Ir.Assignments.asgns in
  let opt = optimized_of y in
  Winograd.set_enabled None;
  { values; raw; opt; d = d.Tensor.value; kern = kern.Tensor.value }

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (Array.to_list
          (Array.map
             (Array.sub got ~pos:0 ~len:(Int.min 8 (Array.length got)))
             ~f:(fun v -> Printf.sprintf "%.9g" v))))

(* {1 Leg 1: exact parity and structure, per shape} *)

type shape = {
  name : string;
  b : int list;
  h : int;  (** Input extent. *)
  w : int;
  ic : int;
  oc : int;
  padded : bool;
}

let out_extent s n = if s.padded then n else n - 2

let single_conv ~f_x ~f_k s () =
  let x = image ~tag:s.name ~b:s.b ~h:s.h ~w:s.w ~ic:s.ic ~f:f_x in
  let k = kernel ~tag:s.name ~ic:s.ic ~oc:s.oc ~f:f_k () in
  let y = conv ~padded:s.padded ~label:(s.name ^ "y") x k in
  (y, y, k)

let x_ints = ints ~weights:[| 7; 1; 3; 5 |] ~m:9 ~off:4
let k_ints = ints ~weights:[| 1; 3; 2; 5 |] ~m:7 ~off:3

let leg1 (s : shape) =
  printf "--- leg 1 %s: %s %dx%d, batch %s, %d -> %d channels ---\n" s.name
    (if s.padded then "padded" else "valid")
    s.h s.w
    (String.concat ~sep:"x" (List.map s.b ~f:Int.to_string))
    s.ic s.oc;
  let build = single_conv ~f_x:x_ints ~f_k:k_ints s in
  let direct = forward ~on:false build in
  let wino = forward ~on:true build in
  report (s.name ^ " direct") direct.values;
  report (s.name ^ " winograd") wino.values;
  let oh = out_extent s s.h and ow = out_extent s s.w in
  let batch = List.fold s.b ~init:1 ~f:( * ) in
  let cells = batch * oh * ow * s.oc in
  let even_cells = batch * (oh / 2 * 2) * (ow / 2 * 2) * s.oc in
  let reduction = 9 * s.ic in
  let rewritten = Winograd.rewrite direct.raw in
  p
    (s.name ^ ": the direct nest covers every output cell")
    (direct_cells direct.raw ~d:direct.d ~kern:direct.kern ~reduction = cells);
  p
    (s.name ^ ": rewritten, the direct form covers only the odd-extent remainder")
    (direct_cells rewritten ~d:direct.d ~kern:direct.kern ~reduction = cells - even_cells);
  p (s.name ^ ": the pass changes the raw lowering") (not (LL.equal rewritten direct.raw));
  p (s.name ^ ": the pass is idempotent") (LL.equal (Winograd.rewrite rewritten) rewritten);
  p_none (s.name ^ ": the gate off mints nothing") (written direct.opt.LL.llc) ~f:(fun tn ->
      Option.is_some (Winograd.minted_role tn));
  p
    (s.name ^ ": the gate on writes the kernel, input and product transforms")
    (List.equal Poly.equal (roles_written wino.opt.LL.llc) all_roles);
  if List.mem s.b 1 ~equal:Int.equal then
    p
      (s.name ^ ": Autotune.detect_conv refuses the singleton batch, which the rewrite carries")
      (Option.is_none (Autotune.detect_conv direct.opt.LL.llc))
  else
    p
      (s.name ^ ": Autotune.detect_conv reports the 3x3 stride-1 site the rewrite claims")
      (Option.is_some (eligible_conv direct.opt.LL.llc));
  if oh % 2 = 0 && ow % 2 = 0 then
    p
      (s.name ^ ": rewritten, no conv site is left")
      (Option.is_none (Autotune.detect_conv wino.opt.LL.llc));
  p
    (s.name ^ ": the contraction is a matmul site accumulating the product buffer")
    (match Autotune.detect_matmul wino.opt.LL.llc with
    | Some m -> Option.is_some (Winograd.minted_role m.Autotune.m_d)
    | None -> false);
  p_all2
    (s.name ^ ": Winograd matches the direct form bitwise")
    wino.values direct.values ~f:Float.equal

let shapes =
  [
    { name = "wpad"; b = [ 2 ]; h = 6; w = 6; ic = 3; oc = 4; padded = true };
    { name = "wval"; b = [ 2 ]; h = 8; w = 10; ic = 2; oc = 3; padded = false };
    { name = "wodd"; b = [ 2 ]; h = 7; w = 9; ic = 3; oc = 2; padded = false };
    { name = "wrect"; b = [ 3 ]; h = 5; w = 8; ic = 5; oc = 2; padded = true };
    { name = "wb1"; b = [ 1 ]; h = 6; w = 4; ic = 2; oc = 3; padded = true };
  ]

let () =
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  List.iter shapes ~f:leg1;
  (* Two chained sites: a padded conv feeding a valid one. *)
  printf "--- leg 1 wchain: two chained conv sites ---\n";
  let build () =
    let x = image ~tag:"wch" ~b:[ 2 ] ~h:6 ~w:6 ~ic:2 ~f:x_ints in
    let k1 = kernel ~tag:"wch1" ~ic:2 ~oc:3 ~f:k_ints () in
    let k2 = kernel ~tag:"wch2" ~ic:3 ~oc:2 ~f:(ints ~weights:[| 2; 1; 3; 1 |] ~m:5 ~off:2) () in
    let y1 = conv ~padded:true ~label:"wch1y" x k1 in
    let y2 = conv ~padded:false ~label:"wch2y" y1 k2 in
    (y2, y2, k2)
  in
  let direct = forward ~on:false build in
  let wino = forward ~on:true build in
  p "wchain: no conv site is left" (Option.is_none (Autotune.detect_conv wino.opt.LL.llc));
  p "wchain: two sites' worth of transforms are written (6 distinct minted buffers)"
    (List.length
       (List.dedup_and_sort ~compare:Tn.compare
          (List.filter (written wino.opt.LL.llc) ~f:(fun tn ->
               Option.is_some (Winograd.minted_role tn))))
    = 6);
  p_all2 "wchain: Winograd matches the direct form bitwise" wino.values direct.values ~f:Float.equal

(* {1 Leg 2: tolerance parity on non-dyadic values} *)

let close_tol ~tol w = Float.(tol *. max 1. (abs w))

let () =
  printf "--- leg 2: non-dyadic inputs, within 1e-4 relative ---\n";
  let s = { name = "wreal"; b = [ 2 ]; h = 9; w = 8; ic = 6; oc = 5; padded = true } in
  let build =
    single_conv ~f_x:(reals ~weights:[| 3; 1; 4; 7 |]) ~f_k:(reals ~weights:[| 2; 5; 1; 3 |]) s
  in
  let direct = forward ~on:false build in
  let wino = forward ~on:true build in
  report "wreal direct" direct.values;
  report "wreal winograd" wino.values;
  let worst =
    Array.fold2_exn wino.values direct.values ~init:0. ~f:(fun acc a b ->
        Float.max acc (Float.abs (a -. b) /. Float.max 1. (Float.abs b)))
  in
  eprintf "wreal: worst relative difference %.3g (not part of the golden)\n%!" worst;
  p "wreal: the rewrite fired" (List.equal Poly.equal (roles_written wino.opt.LL.llc) all_roles);
  p_close "wreal: Winograd matches the direct form within 1e-4 relative" wino.values direct.values
    ~tol:(close_tol ~tol:1e-4)

(* {1 Leg 3: the kernel transform is recomputed every call} *)

let () =
  printf "--- leg 3: a changed kernel is seen by the next call ---\n";
  let run_twice ~on =
    Tensor.unsafe_reinitialize ();
    Winograd.set_enabled (Some on);
    let x = image ~tag:"wst" ~b:[ 2 ] ~h:6 ~w:6 ~ic:2 ~f:x_ints in
    let k = kernel ~tag:"wst" ~ic:2 ~oc:3 ~f:k_ints () in
    let y = conv ~padded:true ~label:"wsty" x k in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx (named "wino_stale" (Train.forward y)) Idx.Empty in
    let ctx = Context.run ctx routine in
    let first = Context.get_values ctx y.Tensor.value in
    let dims = [| 3; 3; 3; 2 |] in
    let k2 =
      Array.init
        (3 * 3 * 3 * 2)
        ~f:(Ll_test.cycle_flat ~radix:3 ~dims ~modulus:7 ~offset:(-3.) ~stride:(-1.))
    in
    let ctx = Context.set_values ctx k.Tensor.value k2 in
    let ctx = Context.run ctx routine in
    let second = Context.get_values ctx y.Tensor.value in
    Winograd.set_enabled None;
    (first, second)
  in
  let d1, d2 = run_twice ~on:false in
  let w1, w2 = run_twice ~on:true in
  p "the replaced kernel changes the direct output" (not (Array.equal Float.equal d1 d2));
  p_all2 "first call: Winograd matches the direct form bitwise" w1 d1 ~f:Float.equal;
  p_all2 "second call, new kernel: Winograd matches the direct form bitwise" w2 d2 ~f:Float.equal

(* {1 Leg 4: training through the rewritten forward} *)

let () =
  printf "--- leg 4: training -- parameter gradients agree with the direct form ---\n";
  let train ~on =
    Tensor.unsafe_reinitialize ();
    Winograd.set_enabled (Some on);
    let x = image ~tag:"wtr" ~b:[ 2 ] ~h:6 ~w:6 ~ic:2 ~f:(reals ~weights:[| 3; 1; 4; 7 |]) in
    let%op h1 = relu (Nn_blocks.conv2d ~label:[ "wtr1" ] ~use_padding:true ~out_channels:3 () x) in
    let%op h2 = Nn_blocks.conv2d ~label:[ "wtr2" ] ~use_padding:true ~out_channels:2 () h1 in
    let%op loss = (h2 *. h2) ++ "...|... => |->0" in
    let params =
      Set.to_list loss.Tensor.params
      |> List.sort ~compare:(fun a b -> Int.compare a.Tensor.value.Tn.id b.Tensor.value.Tn.id)
    in
    List.iter params ~f:(fun p ->
        Train.set_materialized (Option.value_exn ~here:[%here] p.Tensor.diff).Tensor.grad);
    let ctx = Train.update_once (Context.auto ()) loss in
    let grads =
      List.map params ~f:(fun p ->
          ( Tn.debug_name p.Tensor.value,
            Context.get_values ctx (Option.value_exn ~here:[%here] p.Tensor.diff).Tensor.grad ))
    in
    let loss_value = (Context.get_values ctx loss.Tensor.value).(0) in
    let minted =
      List.length
        (List.dedup_and_sort ~compare:Tn.compare
           (List.filter (written (optimized_of loss).LL.llc) ~f:(fun tn ->
                Option.is_some (Winograd.minted_role tn))))
    in
    Winograd.set_enabled None;
    (loss_value, grads, minted)
  in
  let loss_d, grads_d, minted_d = train ~on:false in
  let loss_w, grads_w, minted_w = train ~on:true in
  p "the direct step's forward mints nothing" (minted_d = 0);
  p "the rewritten step's forward writes both convs' transforms (6 minted buffers)" (minted_w = 6);
  eprintf "loss: direct %.9g winograd %.9g (not part of the golden)\n%!" loss_d loss_w;
  p "the loss agrees within 1e-4 relative"
    Float.(abs (loss_w - loss_d) <= close_tol ~tol:1e-4 loss_d);
  printf "parameters with gradients: %d\n" (List.length grads_w);
  p "the same parameters carry gradients in both runs"
    (List.equal String.equal (List.map grads_w ~f:fst) (List.map grads_d ~f:fst));
  List.iter2_exn grads_w grads_d ~f:(fun (name, gw) (_, gd) ->
      let worst =
        Array.fold2_exn gw gd ~init:0. ~f:(fun acc a b ->
            Float.max acc (Float.abs (a -. b) /. Float.max 1. (Float.abs b)))
      in
      eprintf "%s.grad: worst relative difference %.3g (not part of the golden)\n%!" name worst;
      p_close (name ^ ".grad agrees within 1e-4 relative") gw gd ~tol:(close_tol ~tol:1e-4);
      p (name ^ ".grad is not identically zero") (Array.exists gd ~f:(fun v -> Float.(v <> 0.))))

(* {1 Leg 5: the tuned path} *)

let () =
  printf "--- leg 5: the tuner schedules the contractions ---\n";
  let build () =
    (* Every contraction extent a multiple of 8 -- 8 tile columns, 8 output and 8 input channels --
       so a GPU with 8x8x8 matrix tiles seeds its sketches too. *)
    let x = image ~tag:"wtu" ~b:[ 2 ] ~h:16 ~w:16 ~ic:8 ~f:(reals ~weights:[| 3; 1; 4; 7 |]) in
    let k = kernel ~tag:"wtu" ~ic:8 ~oc:8 ~f:(reals ~weights:[| 2; 5; 1; 3 |]) () in
    conv ~padded:true ~label:"wtuy" x k
  in
  Tensor.unsafe_reinitialize ();
  let want =
    let y = build () in
    let ctx = Train.forward_once (Context.auto ()) y in
    Context.get_values ctx y.Tensor.value
  in
  Tensor.unsafe_reinitialize ();
  Winograd.set_enabled (Some true);
  let y = build () in
  let cache_dir = "autotune_cache_winograd" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let reports = ref [] in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ())
      (named "wino_tuned" (Train.forward y))
      Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y.Tensor.value in
  Winograd.set_enabled None;
  (* Whole-routine or per fission segment: on a GPU the whole routine's seeds are refuted at
     companion coverage (the transform nests beside the contraction), and the contraction's own
     segment is where they come from. *)
  p_exists "the search seeded sketches for the contractions" !reports ~f:(fun r ->
      r.Autotune.sketch_candidates + r.Autotune.fiss_sketch_candidates > 0);
  p_close "the tuned Winograd conv matches the direct form within 1e-4 relative" got want
    ~tol:(close_tol ~tol:1e-4)

(* {1 Leg 6: declines} *)

let () =
  printf "--- leg 6: declines ---\n";
  let declined ~what build =
    let r = forward ~on:true build in
    p_none (what ^ ": left in the direct form, nothing minted") (written r.opt.LL.llc) ~f:(fun tn ->
        Option.is_some (Winograd.minted_role tn))
  in
  let mk ?(kh = 3) ?(kw = 3) ?(stride = 1) ?(dilation = 1) ?(prec = Ir.Ops.single) ~tag ~h ~w () ()
      =
    let x = image ~tag ~b:[ 2 ] ~h ~w ~ic:2 ~f:x_ints in
    let k = kernel ~kh ~kw ~prec ~tag ~ic:2 ~oc:3 ~f:k_ints () in
    let y = conv ~stride ~dilation ~padded:false ~label:(tag ^ "y") x k in
    (y, y, k)
  in
  declined ~what:"5x5 window" (mk ~kh:5 ~kw:5 ~tag:"wd5" ~h:9 ~w:9 ());
  declined ~what:"stride 2" (mk ~stride:2 ~tag:"wds" ~h:9 ~w:9 ());
  declined ~what:"dilation 2" (mk ~dilation:2 ~tag:"wdd" ~h:9 ~w:9 ());
  declined ~what:"3x1 window" (mk ~kw:1 ~tag:"wd31" ~h:8 ~w:6 ());
  declined ~what:"f16 kernel" (mk ~prec:Ir.Ops.half ~tag:"wdh" ~h:6 ~w:6 ());
  ()

(* {1 Leg 6b: hand-built raw lowerings the front end never emits, but the exposed pass accepts} *)

let n_o = 4
let n_b = 2
let n_ic = 2
let n_oc = 3
let node id dims label = Ll_test.node_factory ~first_id:id ~dims () label

(* [d[b, oh, ow, oc] += x[b, oh + kh, ow + kw, ic, ..extra..] * k[oc, kh, kw, ic]], where [extra]
   adds axes to the input read. *)
let conv_nest ~x ~k ~d ~extra =
  let module B = Ll_test in
  let b = B.sym () and oh = B.sym () and ow = B.sym () and oc = B.sym () and ic = B.sym () in
  let kh = B.sym () and kw = B.sym () in
  let di = [| B.iter b; B.iter oh; B.iter ow; B.iter oc |] in
  B.loop_n b n_b
    (B.loop_n oh n_o
       (B.loop_n ow n_o
          (B.loop_n oc n_oc
             (B.loop_n ic n_ic
                (B.loop_n kh 3
                   (B.loop_n kw 3
                      (B.set d di
                         (B.add (B.get d di)
                            (B.mul
                               (B.get x
                                  (Array.append
                                     [|
                                       B.iter b;
                                       B.aff [ (1, oh); (1, kh) ] 0;
                                       B.aff [ (1, ow); (1, kw) ] 0;
                                       B.iter ic;
                                     |]
                                     (extra ~kh)))
                               (B.get k [| B.iter oc; B.iter kh; B.iter kw; B.iter ic |]))))))))))

let () =
  printf "--- leg 6b: hand-built nests ---\n";
  let module B = Ll_test in
  let x = node 50600 [| n_b; n_o + 2; n_o + 2; n_ic |] "wax"
  and k = node 50601 [| n_oc; 3; 3; n_ic |] "wak"
  and d = node 50602 [| n_b; n_o; n_o; n_oc |] "wad"
  and x5 = node 50603 [| n_b; n_o + 2; n_o + 2; n_ic; 3 |] "wax5" in
  List.iter [ x; k; d; x5 ] ~f:B.materialize;
  (* Separability: the same conv, its input read at one more axis indexed by the window symbol. The
     tile transform has no value for that axis, so the nest must be left alone. *)
  let entangled = conv_nest ~x:x5 ~k ~d ~extra:(fun ~kh -> [| B.iter kh |]) in
  p "a conv whose input also reads the window symbol at another axis is left alone"
    (LL.equal (Winograd.rewrite entangled) entangled);
  (* Accumulation: a destination holding nonzero values keeps them. *)
  let llc = conv_nest ~x ~k ~d ~extra:(fun ~kh:_ -> [||]) in
  let rewritten = Winograd.rewrite llc in
  p "the control, the same conv without the extra read, is rewritten" (not (LL.equal rewritten llc));
  let n_x = n_b * (n_o + 2) * (n_o + 2) * n_ic
  and n_k = n_oc * 9 * n_ic
  and n_d = n_b * n_o * n_o * n_oc in
  let seed =
    [
      ( x,
        Array.init n_x
          ~f:
            (Ll_test.cycle_flat ~radix:2
               ~dims:[| n_b; n_o + 2; n_o + 2; n_ic |]
               ~modulus:9 ~offset:(-4.) ~stride:1.) );
      ( k,
        Array.init n_k
          ~f:
            (Ll_test.cycle_flat ~radix:3 ~dims:[| n_oc; 3; 3; n_ic |] ~modulus:7 ~offset:(-3.)
               ~stride:1.) );
      (* The destination's prior contents: nonzero and varying, so an overwrite shows. *)
      ( d,
        Array.init n_d
          ~f:
            (Ll_test.cycle_flat ~radix:2 ~dims:[| n_b; n_o; n_o; n_oc |] ~modulus:5 ~offset:10.
               ~stride:1.) );
    ]
  in
  let run name llc =
    let _o, got = B.optimize_and_execute ~name llc ~seed ~read:[ d ] in
    List.hd_exn got
  in
  let want = run "wino_acc_direct" llc in
  let got = run "wino_acc_rewritten" rewritten in
  p_all2 "the rewritten conv adds onto the destination's prior contents, bitwise" got want
    ~f:Float.equal
