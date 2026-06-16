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
          then (Tensor.remove_fwd_root a; a.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then (Tensor.remove_fwd_root b; b.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_binop ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false ~op:Ir.Ops.Div ~t1:a
                 ~rhs1_is_grad:false ~rhs1_is_merge:false ~t2:b
                 ~rhs2_is_grad:false ~rhs2_is_merge:false
                 ~logic:([%ocaml.error
                           "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `/` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `/`, or einsum notation for a custom contraction"]));
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
          then (Tensor.remove_fwd_root a; a.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then (Tensor.remove_fwd_root b; b.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_binop ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false ~op:Ir.Ops.ToPowOf ~t1:a
                 ~rhs1_is_grad:false ~rhs1_is_merge:false ~t2:b
                 ~rhs2_is_grad:false ~rhs2_is_merge:false
                 ~logic:([%ocaml.error
                           "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `**` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `**`, or einsum notation for a custom contraction"]));
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
          then (Tensor.remove_fwd_root a; a.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then (Tensor.remove_fwd_root b; b.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_binop ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false ~op:Ir.Ops.Div ~t1:a
                 ~rhs1_is_grad:false ~rhs1_is_merge:false ~t2:b
                 ~rhs2_is_grad:false ~rhs2_is_merge:false
                 ~logic:([%ocaml.error
                           "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `div` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `div`, or einsum notation for a custom contraction"]));
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
          then (Tensor.remove_fwd_root a; a.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then (Tensor.remove_fwd_root b; b.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_binop ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false ~op:Ir.Ops.ToPowOf ~t1:a
                 ~rhs1_is_grad:false ~rhs1_is_merge:false ~t2:b
                 ~rhs2_is_grad:false ~rhs2_is_merge:false
                 ~logic:([%ocaml.error
                           "ppx_ocannl %cd: `~logic:\"@\"` (Compose) with `pow` looks like matrix inverse/power but computes neither; use `~logic:\".\"` for pointwise `pow`, or einsum notation for a custom contraction"]));
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
          then (Tensor.remove_fwd_root a; a.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          if Tensor.is_fwd_root b
          then (Tensor.remove_fwd_root b; b.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_binop ~initialize_neutral:true ~accum:Ir.Ops.Add
                 ~t:r ~lhs_is_grad:false ~op:Ir.Ops.Add ~t1:a
                 ~rhs1_is_grad:false ~rhs1_is_merge:false ~t2:b
                 ~rhs2_is_grad:false ~rhs2_is_merge:false
                 ~logic:Shape.Compose);
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
