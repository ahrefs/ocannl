open Base
open Ocannl
open Operation.DSL_modules
open Stdio

(* Regression test for per-channel conv biases: without a spec'd add, least-commitment shape
   inference would broadcast-unify the inline bias to the full feature map [oh; ow; oc] instead of
   the conventional per-channel [oc]. *)

let print_params ~name t =
  printf "%s params:\n" name;
  Set.to_list t.Tensor.params
  |> List.map ~f:(fun p ->
      Printf.sprintf "  %s: %s"
        (Ir.Tnode.label_name p.Tensor.value)
        (Ir.Tnode.dims_to_string p.Tensor.value))
  |> List.sort ~compare:String.compare
  |> List.iter ~f:(printf "%s\n")

let () =
  let module IDX = Train.IDX in
  (* 8x8 single-channel input through a 3x3 conv with 4 output channels. *)
  let input =
    TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8; 8; 1 ] ()
  in
  let%op conved = Nn_blocks.conv2d ~label:[ "test_conv" ] ~out_channels:4 () input in
  let ctx = Context.auto () in
  let _ctx = Train.forward_once ctx conved in
  print_params ~name:"conv2d" conved;
  printf "conv2d output dims: %s\n\n" (Ir.Tnode.dims_to_string conved.Tensor.value);

  (* Depthwise separable conv: pin the output channels to a single axis of dim 5 via a spec'd add of
     a fixed-shape tensor (a plain [+] would broadcast instead of pinning). *)
  let input2 =
    TDSL.range_of_shape ~label:[ "input2" ] ~batch_dims:[ 2 ] ~output_dims:[ 8; 8; 3 ] ()
  in
  let pin = TDSL.range_of_shape ~label:[ "pin" ] ~output_dims:[ 5 ] () in
  let%op dsconved =
    Nn_blocks.depthwise_separable_conv2d ~label:[ "test_dsconv" ] () input2
    +++ "... | h, w, c; |c => ... | h, w, c" pin
  in
  let ctx = Context.auto () in
  let _ctx = Train.forward_once ctx dsconved in
  print_params ~name:"depthwise_separable_conv2d" dsconved;
  printf "depthwise_separable_conv2d output dims: %s\n"
    (Ir.Tnode.dims_to_string dsconved.Tensor.value)
