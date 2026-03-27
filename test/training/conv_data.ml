(** Dataset conversion helpers for CNN training examples.

    Converts [int8_unsigned] bigarrays from [Dataprep] to [float32] bigarrays in [\[0, 1\]] range,
    suitable for use with OCANNL via [Ir.Ndarray.as_array Ir.Ops.Single]. *)

open Bigarray

(** Convert MNIST [int8_unsigned] images [\[N; 28; 28\]] to [float32] [\[N; 28; 28; 1\]] in [\[0,
    1\]]. Adds a trailing singleton channel dimension for [conv2d] compatibility. *)
let mnist_images_to_float32 raw_images =
  let n = (Genarray.dims raw_images).(0) in
  let result = Genarray.create Float32 c_layout [| n; 28; 28; 1 |] in
  for i = 0 to n - 1 do
    for r = 0 to 27 do
      for c = 0 to 27 do
        let v = Float.of_int (Genarray.get raw_images [| i; r; c |]) /. 255.0 in
        Genarray.set result [| i; r; c; 0 |] v
      done
    done
  done;
  result

(** Convert CIFAR-10 [int8_unsigned] images [\[N; 32; 32; 3\]] to [float32] [\[N; 32; 32; 3\]] in
    [\[0, 1\]]. *)
let cifar_images_to_float32 raw_images =
  let n = (Genarray.dims raw_images).(0) in
  let result = Genarray.create Float32 c_layout [| n; 32; 32; 3 |] in
  for i = 0 to n - 1 do
    for r = 0 to 31 do
      for c = 0 to 31 do
        for ch = 0 to 2 do
          let v = Float.of_int (Genarray.get raw_images [| i; r; c; ch |]) /. 255.0 in
          Genarray.set result [| i; r; c; ch |] v
        done
      done
    done
  done;
  result

(** Convert [int8_unsigned] 1D label array to a list of 0-based ints. *)
let labels_to_int_list labels =
  let n = (Genarray.dims labels).(0) in
  List.init n (fun i -> Genarray.get labels [| i |])

(** Take first [n] samples from a [float32] image genarray. Returns a contiguous copy. *)
let take_prefix_images ~n float_images =
  let dims = Genarray.dims float_images in
  let sub_dims = Array.copy dims in
  sub_dims.(0) <- n;
  let result = Genarray.create Float32 c_layout sub_dims in
  Genarray.blit (Genarray.sub_left float_images 0 n) result;
  result
