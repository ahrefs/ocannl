open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
let test_div_compose a b =
  let r =
    (NTDSL.term ~label:("r" :: ((a.Tensor.value).Ir.Tnode.label))
       ?fetch_op:None) () in
  let _r =
    let open! NTDSL.O in
      let uncommented_comp =
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) r.Tensor.value)
           };
          if Tensor.is_fwd_root a
          then Tensor.take_forward_code a
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then Tensor.take_forward_code b
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast
                                 (([%ocaml.error
                                     "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `/` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `/`, or einsum notation for a custom contraction"]),
                                   (a.Tensor.shape), (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.Div;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b)
                         }));
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("_r", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  _r
let test_pow_compose a b =
  let r =
    (NTDSL.term ~label:("r" :: ((a.Tensor.value).Ir.Tnode.label))
       ?fetch_op:None) () in
  let _r =
    let open! NTDSL.O in
      let uncommented_comp =
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) r.Tensor.value)
           };
          if Tensor.is_fwd_root a
          then Tensor.take_forward_code a
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then Tensor.take_forward_code b
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast
                                 (([%ocaml.error
                                     "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `**` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `**`, or einsum notation for a custom contraction"]),
                                   (a.Tensor.shape), (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.ToPowOf;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b)
                         }));
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("_r", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  _r
let test_div_alias_compose a b =
  let r =
    (NTDSL.term ~label:("r" :: ((a.Tensor.value).Ir.Tnode.label))
       ?fetch_op:None) () in
  let _r =
    let open! NTDSL.O in
      let uncommented_comp =
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) r.Tensor.value)
           };
          if Tensor.is_fwd_root a
          then Tensor.take_forward_code a
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then Tensor.take_forward_code b
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast
                                 (([%ocaml.error
                                     "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `div` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `div`, or einsum notation for a custom contraction"]),
                                   (a.Tensor.shape), (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.Div;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b)
                         }));
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("_r", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  _r
let test_pow_alias_compose a b =
  let r =
    (NTDSL.term ~label:("r" :: ((a.Tensor.value).Ir.Tnode.label))
       ?fetch_op:None) () in
  let _r =
    let open! NTDSL.O in
      let uncommented_comp =
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) r.Tensor.value)
           };
          if Tensor.is_fwd_root a
          then Tensor.take_forward_code a
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then Tensor.take_forward_code b
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast
                                 (([%ocaml.error
                                     "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `pow` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `pow`, or einsum notation for a custom contraction"]),
                                   (a.Tensor.shape), (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.ToPowOf;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b)
                         }));
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("_r", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  _r
let test_add_compose_accepted a b =
  let r =
    (NTDSL.term ~label:("r" :: ((a.Tensor.value).Ir.Tnode.label))
       ?fetch_op:None) () in
  let _r =
    let open! NTDSL.O in
      let uncommented_comp =
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) r.Tensor.value)
           };
          if Tensor.is_fwd_root a
          then Tensor.take_forward_code a
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then Tensor.take_forward_code b
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast
                                 (Shape.Compose, (a.Tensor.shape),
                                   (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.Add;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b)
                         }));
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("_r", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  _r
