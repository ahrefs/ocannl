open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

module type Backend = Ir.Backend_intf.Backend

let%expect_test "diagonal_tensor_initialization" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Create a diagonal tensor using einsum: i->ii *)
  let input = TDSL.range 5 in
  let%op diagonal = input ++ "i=>ii" in

  (* Ensure the diagonal tensor is hosted *)
  Train.set_hosted diagonal.value;
  ignore (Train.forward_once backend diagonal);

  (* Print the diagonal tensor *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false diagonal;
  [%expect
    {|
    HERE: test/einsum/surjectivity.ml:31:21
    ┌──────────────────────────────────────┐
    │[1]: =>_diagonal shape 0:6,1:6        │
    │┌──────┬─────────────────────────────┐│
    ││      │axis 1                       ││
    │├──────┼─────────────────────────────┤│
    ││axis 0│ 0.00  0.00  ...  0.00  0.00 ││
    ││      │ 0.00  1.00  ...  0.00  0.00 ││
    ││      │ ...   ...   ...  ...   ...  ││
    ││      │ 0.00  0.00  ...  4.00  0.00 ││
    ││      │ 0.00  0.00  ...  0.00  5.00 ││
    │└──────┴─────────────────────────────┘│
    └──────────────────────────────────────┘
    |}]

let%expect_test "sparse_assignment_with_fixed_indices" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Create a sparse tensor using fixed indices: i->i0j *)
  let input = TDSL.range 4 in
  let%op sparse = input ++ "i=>i0j" in

  Train.set_hosted sparse.value;
  ignore (Train.forward_once backend sparse);

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false sparse;
  [%expect
    {|
    HERE: test/einsum/surjectivity.ml:67:21
    ┌─────────────────────────────────┐
    │[1]: =>_sparse shape 0:5,1:1,2:1 │
    │┌──────┬──────┐                  │
    ││      │axis 2│                  │
    │├──────┼──────┤                  │
    ││0 @ 0 │ 0.00 │                  │
    ││axis 1│      │                  │
    │├──────┼──────┤                  │
    ││1 @ 0 │ 1.00 │                  │
    ││axis 1│      │                  │
    │├──────┼──────┤                  │
    ││2 @ 0 │ 2.00 │                  │
    ││axis 1│      │                  │
    │├──────┼──────┤                  │
    ││3 @ 0 │ 3.00 │                  │
    ││axis 1│      │                  │
    │├──────┼──────┤                  │
    ││4 @ 0 │ 4.00 │                  │
    ││axis 1│      │                  │
    │└──────┴──────┘                  │
    └─────────────────────────────────┘
    |}]

let%expect_test "multiple_sparse_axes" =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Test with multiple fixed indices: ij->i1j2 *)
  let input = TDSL.range_of_shape ~output_dims:[ 3; 4 ] () in
  let%op sparse_multi = input ++ "ij=>i1j2" in

  Train.set_hosted sparse_multi.value;
  ignore (Train.forward_once backend sparse_multi);

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false sparse_multi;
  [%expect
    {|
    HERE: test/einsum/surjectivity.ml:112:21
    ┌───────────────────────────────────────────┐
    │[1]: =>_sparse_multi shape 0:3,1:2,2:4,3:3 │
    │┌──────┬──────────────────┐                │
    ││0 @ 0 │axis 3            │                │
    │├──────┼──────────────────┤                │
    ││0 @ 1 │ 0.00  0.00  0.00 │                │
    ││axis 2│ 0.00  0.00  0.00 │                │
    ││      │ 0.00  0.00  0.00 │                │
    ││      │ 0.00  0.00  0.00 │                │
    │├──────┼──────────────────┤                │
    ││1 @ 1 │ 0.00  0.00  0.00 │                │
    ││axis 2│ 0.00  0.00  1.00 │                │
    ││      │ 0.00  0.00  2.00 │                │
    ││      │ 0.00  0.00  3.00 │                │
    │└──────┴──────────────────┘                │
    ├───────────────────────────────────────────┤
    │┌──────┬──────────────────┐                │
    ││1 @ 0 │axis 3            │                │
    │├──────┼──────────────────┤                │
    ││0 @ 1 │ 0.00  0.00  0.00 │                │
    ││axis 2│ 0.00  0.00  0.00 │                │
    ││      │ 0.00  0.00  0.00 │                │
    ││      │ 0.00  0.00  0.00 │                │
    │├──────┼──────────────────┤                │
    ││1 @ 1 │ 0.00  0.00  4.00 │                │
    ││axis 2│ 0.00  0.00  5.00 │                │
    ││      │ 0.00  0.00  6.00 │                │
    ││      │ 0.00  0.00  7.00 │                │
    │└──────┴──────────────────┘                │
    ├───────────────────────────────────────────┤
    │┌──────┬─────────────────────┐             │
    ││2 @ 0 │axis 3               │             │
    │├──────┼─────────────────────┤             │
    ││0 @ 1 │ 0.00  0.00  0.00    │             │
    ││axis 2│ 0.00  0.00  0.00    │             │
    ││      │ 0.00  0.00  0.00    │             │
    ││      │ 0.00  0.00  0.00    │             │
    │├──────┼─────────────────────┤             │
    ││1 @ 1 │ 0.00  0.00  8.00    │             │
    ││axis 2│ 0.00  0.00  9.00    │             │
    ││      │ 0.00  0.00  1.00e+1 │             │
    ││      │ 0.00  0.00  1.10e+1 │             │
    │└──────┴─────────────────────┘             │
    └───────────────────────────────────────────┘
    |}]
