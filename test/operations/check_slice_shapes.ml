open Base
open Ocannl
open Operation.DSL_modules
open Stdio
module IDX = Train.IDX

let () =
  let n_batches = 2 in
  let batch_size = 4 in
  let total_samples = n_batches * batch_size in

  (* Create test data *)
  let images_data =
    Ir.Ndarray.create_array ~debug:"images" Ir.Ops.single ~dims:[| total_samples; 3; 3; 1 |]
      ~padding:None
  in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let images = TDSL.rebatch ~l:"images" images_data () in

  let%op batch_images = images @| batch_n in
  let%op processed_images =
    Nn_blocks.conv2d ~label:[ "processed" ] ~use_padding:true ~out_channels:1 () batch_images
  in

  Train.set_hosted batch_images.value;
  let forward = Train.forward processed_images in
  (* Force shape inference and tensor allocation *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings processed_images in
  let routine = Train.to_routine ctx bindings forward in
  let batch_n_ref = IDX.find_exn (Context.bindings routine) batch_n in
  batch_n_ref := 0;
  Train.run ctx routine;

  printf "\n";
  printf "images.value dims: %s\n" (Ir.Tnode.dims_to_string images.value);
  printf "batch_images.value dims: %s\n" (Ir.Tnode.dims_to_string batch_images.value);
  printf "images.value padding: %s\n"
    (match Lazy.force images.value.Ir.Tnode.padding with None -> "None" | Some _ -> "Some");
  printf "batch_images.value padding: %s\n"
    (match Lazy.force batch_images.value.Ir.Tnode.padding with None -> "None" | Some _ -> "Some");
  ()
