(** Dataset conversion helpers for CNN training examples.

    Converts [int8_unsigned] bigarrays from [Dataprep] to [float32] bigarrays in [\[0, 1\]] range,
    suitable for use with OCANNL via [Ir.Ndarray.as_array Ir.Ops.Single].

    Also provides [load_cifar10] which downloads and parses the binary distribution of CIFAR-10 (the
    [Dataprep.Cifar10] loader is broken because it downloads the Python pickle version). *)

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
    [\[-0.5, 0.5\]].

    Data centering is critical for CIFAR: unlike MNIST (sparse, mostly-zero background), CIFAR
    images are dense RGB with high mean (~0.45). Without centering, all-positive Xavier-uniform
    weights accumulate extreme activations through the network, causing softmax saturation and
    preventing learning. Centering to [\[-0.5, 0.5\]] ensures that convolution outputs are naturally
    zero-centered even with all-positive weight initialization. *)
let cifar_images_to_float32 raw_images =
  let n = (Genarray.dims raw_images).(0) in
  let result = Genarray.create Float32 c_layout [| n; 32; 32; 3 |] in
  for i = 0 to n - 1 do
    for r = 0 to 31 do
      for c = 0 to 31 do
        for ch = 0 to 2 do
          let v = Float.of_int (Genarray.get raw_images [| i; r; c; ch |]) /. 255.0 -. 0.5 in
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

(* --- CIFAR-10 binary loader ---
   Downloads and parses the binary distribution from U Toronto. The binary format stores each image
   as 1 label byte + 3072 pixel bytes (1024 R + 1024 G + 1024 B in row-major order within each
   channel plane). We convert from CHW to HWC layout for OCANNL's conv2d. *)

let cifar10_cache_dir () =
  let home =
    try Sys.getenv "HOME"
    with Not_found -> failwith "HOME environment variable not set"
  in
  home ^ "/.cache/ocaml-dataprep/datasets/cifar-10-bin/"

let cifar10_data_dir () = cifar10_cache_dir () ^ "cifar-10-batches-bin/"

let ensure_cifar10_binary () =
  let cache_dir = cifar10_cache_dir () in
  let data_dir = cifar10_data_dir () in
  let check_file = data_dir ^ "test_batch.bin" in
  if not (Sys.file_exists check_file) then begin
    let tar_path = cache_dir ^ "cifar-10-binary.tar.gz" in
    let url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" in
    (* Create cache dir *)
    (match Unix.system (Printf.sprintf "mkdir -p %s" (Filename.quote cache_dir)) with
     | Unix.WEXITED 0 -> ()
     | _ -> failwith ("Failed to create cache directory: " ^ cache_dir));
    (* Download if needed *)
    if not (Sys.file_exists tar_path) then begin
      Printf.printf "Downloading CIFAR-10 binary dataset...\n%!";
      (match
         Unix.system
           (Printf.sprintf "curl -L -o %s %s" (Filename.quote tar_path) (Filename.quote url))
       with
      | Unix.WEXITED 0 -> ()
      | _ -> failwith "Failed to download CIFAR-10 binary dataset")
    end;
    (* Extract *)
    Printf.printf "Extracting CIFAR-10...\n%!";
    (match
       Unix.system
         (Printf.sprintf "tar xzf %s -C %s" (Filename.quote tar_path) (Filename.quote cache_dir))
     with
    | Unix.WEXITED 0 -> ()
    | _ -> failwith "Failed to extract CIFAR-10 archive");
    if not (Sys.file_exists check_file) then
      failwith ("Extraction succeeded but check file not found: " ^ check_file)
  end

(** Read a single CIFAR-10 binary batch file. Each record is 1 label byte + 3072 pixel bytes
    (channel-first: 1024 R, 1024 G, 1024 B). Returns images in HWC layout [\[N; 32; 32; 3\]]. *)
let read_cifar10_batch filename =
  let ic = open_in_bin filename in
  let file_len = in_channel_length ic in
  let bytes_per_image = 3073 in
  if file_len mod bytes_per_image <> 0 then begin
    close_in ic;
    failwith (Printf.sprintf "CIFAR-10 binary file %s has unexpected size %d" filename file_len)
  end;
  let num_images = file_len / bytes_per_image in
  let images = Genarray.create int8_unsigned c_layout [| num_images; 32; 32; 3 |] in
  let labels = Genarray.create int8_unsigned c_layout [| num_images |] in
  for i = 0 to num_images - 1 do
    let label = input_byte ic in
    Genarray.set labels [| i |] label;
    (* Read CHW data and store as HWC *)
    for ch = 0 to 2 do
      for row = 0 to 31 do
        for col = 0 to 31 do
          let pixel = input_byte ic in
          Genarray.set images [| i; row; col; ch |] pixel
        done
      done
    done
  done;
  close_in ic;
  (images, labels)

(** Load CIFAR-10 from the binary distribution. Downloads on first call, then caches in
    [~/.cache/ocaml-dataprep/datasets/cifar-10-bin/]. Returns the same type as
    [Dataprep.Cifar10.load]: [((train_images, train_labels), (test_images, test_labels))] where
    images are [int8_unsigned] [\[N; 32; 32; 3\]] and labels are [int8_unsigned] [\[N\]]. *)
let load_cifar10 () =
  ensure_cifar10_binary ();
  let data_dir = cifar10_data_dir () in
  (* Read 5 training batches *)
  let train_batch_files =
    List.init 5 (fun i -> data_dir ^ Printf.sprintf "data_batch_%d.bin" (i + 1))
  in
  let train_batches = List.map read_cifar10_batch train_batch_files in
  let total_train = 50000 in
  let train_images = Genarray.create int8_unsigned c_layout [| total_train; 32; 32; 3 |] in
  let train_labels = Genarray.create int8_unsigned c_layout [| total_train |] in
  let offset = ref 0 in
  List.iter
    (fun (imgs, lbls) ->
      let n = (Genarray.dims lbls).(0) in
      Genarray.blit (Genarray.sub_left imgs 0 n) (Genarray.sub_left train_images !offset n);
      Genarray.blit (Genarray.sub_left lbls 0 n) (Genarray.sub_left train_labels !offset n);
      offset := !offset + n)
    train_batches;
  (* Read test batch *)
  let test_images, test_labels = read_cifar10_batch (data_dir ^ "test_batch.bin") in
  ((train_images, train_labels), (test_images, test_labels))
