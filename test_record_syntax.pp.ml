[@@@ocaml.ppx.context
  {
    tool_name = "ppx_driver";
    include_dirs = [];
    hidden_include_dirs = [];
    load_path = ([], []);
    open_modules = [];
    for_package = None;
    debug = false;
    use_threads = false;
    use_vmthreads = false;
    recursive_types = false;
    principal = false;
    transparent_modules = false;
    unboxed_types = false;
    unsafe_string = false;
    cookies = []
  }]
open Ocannl
module Tensor = Tensor
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module PDSL = Operation.PDSL
let _test_op_uniform =
  let x =
    (TDSL.param ?more_label:None ?value:None ?values:None
       ?param_init:(Some (PDSL.uniform ())) "x") () in
  let open! TDSL.O in x
let _test_op_float =
  let y =
    (TDSL.param ?more_label:None ?value:(Some 0.5) ?values:None
       ?param_init:None "y") () in
  let open! TDSL.O in y
let _test_op_int =
  let z =
    (TDSL.param ?more_label:None ?value:(Some (Float.of_int 42)) ?values:None
       ?param_init:None "z") () in
  let open! TDSL.O in z
let _test_op_list =
  let weights =
    ((TDSL.param ?more_label:None ?value:None ?values:(Some [|0.1;0.2;0.3|])
        ?param_init:None "weights") ~input_dims:[] ~output_dims:[3]) () in
  let open! TDSL.O in weights
let _test_op_nested =
  let biases =
    ((TDSL.param ?more_label:None ?value:None
        ?values:(Some [|0.0;1.0;2.0;3.0|]) ?param_init:None "biases")
       ~input_dims:[] ~output_dims:[2; 2]) () in
  let open! TDSL.O in biases
let _test_op_with_dims =
  let w =
    ((TDSL.param ?more_label:None ?value:None ?values:None
        ?param_init:(Some (PDSL.uniform ())) "w") ~input_dims:[2; 3]
       ~output_dims:[4]) () in
  let open! TDSL.O in w
let _test_op_shorthands =
  let v =
    ((TDSL.param ?more_label:None ?value:None ?values:None
        ?param_init:(Some (PDSL.uniform ())) "v") ~input_dims:[5]
       ~output_dims:[6; 7]) () in
  let open! TDSL.O in v
let _test_cd_computation () =
  let temp = (NTDSL.term ~label:["temp"] ?fetch_op:None) () in
  let result =
    let open! NTDSL.O in
      let uncommented_comp =
        let nondiff__for_rhs1 = NTDSL.O.(!.) 2.0 in
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) temp.Tensor.value)
           };
          if Tensor.is_fwd_root nondiff__for_rhs1
          then
            (Tensor.remove_fwd_root nondiff__for_rhs1;
             nondiff__for_rhs1.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_unop ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:temp ~lhs_is_grad:false ~op:Ir.Ops.Identity
                 ~t1:nondiff__for_rhs1 ~rhs_is_grad:false ~rhs_is_merge:false
                 ~logic:Shape.Pointwise_un);
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("result", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  result
let _test_cd_with_dims () =
  let temp =
    (NTDSL.term ~label:["temp"] ?fetch_op:None ~output_dims:[3; 4]) () in
  let result =
    let open! NTDSL.O in
      let uncommented_comp =
        let nondiff__for_rhs1 = NTDSL.O.(!.) 1.0 in
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) temp.Tensor.value)
           };
          if Tensor.is_fwd_root nondiff__for_rhs1
          then
            (Tensor.remove_fwd_root nondiff__for_rhs1;
             nondiff__for_rhs1.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_unop ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:temp ~lhs_is_grad:false ~op:Ir.Ops.Identity
                 ~t1:nondiff__for_rhs1 ~rhs_is_grad:false ~rhs_is_merge:false
                 ~logic:Shape.Pointwise_un);
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("result", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  result
let _test_cd_shorthands () =
  let x = (NTDSL.term ~label:["x"] ?fetch_op:None ~output_dims:[10]) () in
  let result =
    let open! NTDSL.O in
      let uncommented_comp =
        let nondiff__for_rhs1 = NTDSL.O.(!.) 3.0 in
        Ir.Assignments.sequence
          [{
             Ir.Assignments.asgns = Ir.Assignments.Noop;
             embedded_nodes =
               (Base.Set.singleton (module Ir.Tnode) x.Tensor.value)
           };
          if Tensor.is_fwd_root nondiff__for_rhs1
          then
            (Tensor.remove_fwd_root nondiff__for_rhs1;
             nondiff__for_rhs1.Tensor.forward)
          else
            {
              Ir.Assignments.asgns = Ir.Assignments.Noop;
              embedded_nodes = (Base.Set.empty (module Ir.Tnode))
            };
          {
            Ir.Assignments.asgns =
              (Tensor.raw_unop ~initialize_neutral:false ~accum:Ir.Ops.Arg2
                 ~t:x ~lhs_is_grad:false ~op:Ir.Ops.Identity
                 ~t1:nondiff__for_rhs1 ~rhs_is_grad:false ~rhs_is_merge:false
                 ~logic:Shape.Pointwise_un);
            embedded_nodes = (Base.Set.empty (module Ir.Tnode))
          }] in
      {
        Ir.Assignments.embedded_nodes =
          (uncommented_comp.Ir.Assignments.embedded_nodes);
        asgns =
          (Ir.Assignments.Block_comment
             ("result", (uncommented_comp.Ir.Assignments.asgns)))
      } in
  result
let () =
  Stdio.printf "Test compilation successful!\n";
  Stdio.printf
    "Record syntax for both %%op and %%cd extensions works correctly.\n";
  Stdio.printf
    "All initialization patterns and shorthand notation supported.\n"
