open Ocannl.Operation.DSL_modules
let test_raw_unop a =
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
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Transpose
                                 (Shape.Pointwise_un, (a.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Unop
                         {
                           op = Ir.Ops.Relu;
                           rhs =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a)
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
let test_raw_unop_transpose a =
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
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Transpose
                                 (Shape.Transpose, (a.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Unop
                         {
                           op = Ir.Ops.Identity;
                           rhs =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a)
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
let test_raw_unop_permute a =
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
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Transpose
                                 ((Shape.Permute ("ij=>ji", [])),
                                   (a.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Unop
                         {
                           op = Ir.Ops.Identity;
                           rhs =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a)
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
let test_raw_identity a =
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
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Transpose
                                 (Shape.Pointwise_un, (a.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Unop
                         {
                           op = Ir.Ops.Identity;
                           rhs =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a)
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
let test_raw_ternop_fma a b c =
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
          if Tensor.is_fwd_root c
          then Tensor.take_forward_code c
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast_tern
                                 (Shape.Compose_accumulate, (a.Tensor.shape),
                                   (b.Tensor.shape), (c.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Ternop
                         {
                           op = Ir.Ops.FMA;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b);
                           rhs3 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                c)
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
let test_raw_ternop_pointwise a b c =
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
          if Tensor.is_fwd_root c
          then Tensor.take_forward_code c
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast_tern
                                 (Shape.Pointwise_tern, (a.Tensor.shape),
                                   (b.Tensor.shape), (c.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Ternop
                         {
                           op = Ir.Ops.Where;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b);
                           rhs3 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                c)
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
let test_raw_ternop_einsum a b c =
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
          if Tensor.is_fwd_root c
          then Tensor.take_forward_code c
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Broadcast_tern
                                 ((Shape.Einsum_tern ("i;i;i=>i", [])),
                                   (a.Tensor.shape), (b.Tensor.shape),
                                   (c.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Ternop
                         {
                           op = Ir.Ops.Mul3;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                a);
                           rhs2 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                b);
                           rhs3 =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:false
                                c)
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
let test_merge_value_operand a =
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
          {
            Ir.Assignments.asgns =
              (Tensor.raw_accum ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:r ~lhs_is_grad:false
                 ~shape_logic:(Shape.Transpose
                                 (Shape.Pointwise_un, (a.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Unop
                         {
                           op = Ir.Ops.Identity;
                           rhs =
                             (Tensor.buffer_of ~is_grad:false ~is_merge:true
                                a)
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
let test_merge_grad_operand a b =
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
                                 (Shape.Pointwise_bin, (a.Tensor.shape),
                                   (b.Tensor.shape)))
                 ~rhs:(Ir.Assignments.Binop
                         {
                           op = Ir.Ops.Mul;
                           rhs1 =
                             (Tensor.buffer_of ~is_grad:true ~is_merge:true a);
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
let () =
  ignore
    (test_raw_unop, test_raw_unop_transpose, test_raw_unop_permute,
      test_raw_identity, test_raw_ternop_fma, test_raw_ternop_pointwise,
      test_raw_ternop_einsum, test_merge_value_operand,
      test_merge_grad_operand)
