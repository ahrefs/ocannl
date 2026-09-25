(* Padding lifecycle: the halo margins and their neutral element are part of a tensor node's
   identity, committed at the node's first compilation (when its [padding] lazy forces).

   Pinned here: - Locked-layout rejection: a data node created via [init] ([Keep_shape_no_padding])
   has its layout committed at creation; a padded conv reading it is REJECTED at shape-inference
   time (before the fix it silently lost the halo and read out of bounds at negative offsets). -
   Fresh-operand halo: a computed intermediate gets its halo from the padded consumer, the committed
   padding records the conv's neutral element, and the lowered conv is offset-free in buffer space
   (a valid conv over the padded buffer) — [detect_conv] sees [cx_offset = 0], so the autotune
   conv-seed gate admits healthy padded convs. - Compatible late demand: a second same-geometry
   padded conv on the already-committed operand is accepted (the resolved padding covers it). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments
open Verdict.Claims

let pr fmt = Stdio.printf fmt

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let make_x tag =
  NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
    ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
    ()

let padding_to_string (tn : Ir.Tnode.t) =
  if not (Lazy.is_val tn.Ir.Tnode.padding) then "UNFORCED"
  else
    match Lazy.force tn.Ir.Tnode.padding with
    | None -> "None"
    | Some (arr, elem) ->
        Printf.sprintf "[%s] elem=%s"
          (String.concat ~sep:"; "
             (Array.to_list arr
             |> List.map ~f:(fun Ir.Ops.{ left; right } -> Printf.sprintf "%d/%d" left right)))
          (Float.to_string elem)

let compile_conv ?ctx tag x =
  let conv =
    Nn_blocks.conv2d ~label:[ tag ] ~kernel_size:3 ~stride:1 ~use_padding:true ~out_channels:8 ()
  in
  let y = conv x in
  let site = ref None in
  let transform (opt : LL.optimized) =
    (match Autotune.detect_conv opt.LL.llc with Some s -> site := Some s | None -> ());
    opt
  in
  let ctx = match ctx with Some ctx -> ctx | None -> Context.auto () in
  let ctx = Train.init_params ctx Ir.Indexing.Empty y in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named (tag ^ "_fwd") (Train.forward y))
      Ir.Indexing.Empty
  in
  ignore (routine : Context.routine);
  (ctx, !site)

let () =
  (* === Locked layout: padded conv on an [init] data node is rejected === *)
  let x1 = make_x "locked" in
  p "locked: layout committed at creation" (Lazy.is_val x1.Tensor.value.Ir.Tnode.padding);
  (match compile_conv "locked" x1 with
  | exception Row.Shape_error (msg, _) -> pr "locked: REJECTED: %s\n" (String.prefix msg 60)
  | (_ : Context.t * Autotune.conv_site option) ->
      p "locked: padded conv on a committed-layout operand rejected" false);
  pr "---\n";
  (* === Fresh operand: halo granted, neutral committed, offset-free lowering === *)
  Tensor.unsafe_reinitialize ();
  let x2_src = make_x "fresh" in
  let x2 = NTDSL.O.einsum1 "... | h, w, c => ... | h, w, c" x2_src in
  (* The "fresh_again" section below re-reads [x2] from a second routine. Cross-routine reuse
     requires explicit materialization; before the gh-494 read-before-write decider flip this worked
     accidentally (the tracer's truncation spuriously flagged the padded operand recurrent, forcing
     it on-device). *)
  Train.set_materialized x2.Tensor.value;
  p "fresh: intermediate starts unforced" (not (Lazy.is_val x2.Tensor.value.Ir.Tnode.padding));
  let ctx, site = compile_conv "fresh" x2 in
  pr "fresh: committed padding = %s\n" (padding_to_string x2.Tensor.value);
  pr "fresh: buffer dims = [%s]\n"
    (String.concat ~sep:"; "
       (Array.to_list (Lazy.force x2.Tensor.value.Ir.Tnode.dims) |> List.map ~f:Int.to_string));
  (match site with
  | None -> p "fresh: conv site detected" false
  | Some s ->
      p_all "fresh: lowered conv is offset-free in buffer space" s.Autotune.c_axes ~f:(fun cx ->
          cx.Autotune.cx_offset = 0);
      p_all "fresh: window and stride detected" s.Autotune.c_axes ~f:(fun cx ->
          cx.Autotune.cx_stride = 1 && cx.Autotune.cx_nk = 3));
  (* === Compatible late demand on the now-committed operand is accepted === *)
  (match compile_conv ~ctx "fresh_again" x2 with
  | exception Row.Shape_error (msg, _) ->
      pr "fresh_again: unexpected rejection: %s\n" (String.prefix msg 60)
  | (_ : Context.t * Autotune.conv_site option) ->
      p "fresh_again: same-geometry padded conv on committed operand accepted" true);
  pr "---\n";
  (* === Wrapped-padded data: the creation-committed neutral element is enforced === *)
  (* [wrap_padded] commits both the margins and the [padded_value] at creation. A padded conv
     (neutral 0) reading margins committed to 1 must be rejected — without the reconciliation it
     would silently sum the 1s from the halo (Codex P1 on PR #173). *)
  let make_wrapped tag ~padded_value =
    let ndarray =
      Ir.Ndarray.init_array ~debug:(tag ^ "x") Ir.Ops.single ~dims:[| 2; 14; 14; 4 |] ~padding:None
        ~f:(fun _ -> padded_value)
    in
    Operation.wrap_padded ~grad_spec:Tensor.Prohibit_grad ~l:(tag ^ "x") ~b:[ 2 ] ~o:[ 11; 11; 4 ]
      ~padding:
        Ir.Ops.
          [|
            { left = 0; right = 0 };
            { left = 1; right = 2 };
            { left = 1; right = 2 };
            { left = 0; right = 0 };
          |]
      ~padded_value ndarray ()
  in
  Tensor.unsafe_reinitialize ();
  let xw1 = make_wrapped "wrapped_one" ~padded_value:1.0 in
  (match compile_conv "wrapped_one" xw1 with
  | exception Row.Shape_error (msg, _) -> pr "wrapped_one: REJECTED: %s\n" (String.prefix msg 42)
  | (_ : Context.t * Autotune.conv_site option) ->
      p "wrapped_one: conv's 0 neutral vs margins committed to 1 rejected" false);
  Tensor.unsafe_reinitialize ();
  let xw0 = make_wrapped "wrapped_zero" ~padded_value:0.0 in
  match compile_conv "wrapped_zero" xw0 with
  | exception Row.Shape_error (msg, _) ->
      pr "wrapped_zero: unexpected rejection: %s\n" (String.prefix msg 60)
  | _, site -> (
      p "wrapped_zero: conv on matching committed neutral accepted" true;
      match site with
      | Some s ->
          p_all "wrapped_zero: offset-free in buffer space" s.Autotune.c_axes ~f:(fun cx ->
              cx.Autotune.cx_offset = 0)
      | None -> p "wrapped_zero: offset-free in buffer space" false)
