(** Half moons synthetic dataset generation *)

open Bigarray

(** Configuration for the half moons dataset *)
module Config = struct
  type t = {
    noise_range : float;  (** Range of noise to add to the coordinates *)
    seed : int option;  (** Optional random seed for reproducibility *)
  }

  let default = { noise_range = 0.1; seed = None }
end

module Random = Rand.Random_for_tests

(** Internal helper function to generate half moons with specified precision.

    @param kind The bigarray kind (float32 or float64)
    @param config Configuration for noise and randomization
    @param len Number of samples per moon (total samples = len * 2)
    @return
      A tuple of (coordinates, labels) where:
      - coordinates is a bigarray of shape [len*2; 2] (batch_axis, output_axis)
      - labels is a bigarray of shape [len*2; 1] (batch_axis, output_axis)
      - First moon has label 1.0, second moon has label -1.0 *)
let generate_with_kind kind ?(config = Config.default) ~len () =
  (* Initialize random seed if specified *)
  (match config.seed with Some seed -> Random.init seed | None -> ());

  let noise () = Random.float_range ~-.(config.noise_range) config.noise_range in
  let total_samples = len * 2 in

  (* Create bigarrays with batch axis first, then output axis *)
  let coordinates = Genarray.create kind c_layout [| total_samples; 2 |] in
  let labels = Genarray.create kind c_layout [| total_samples; 1 |] in

  (* Generate first moon (label = 1.0) *)
  for i = 0 to len - 1 do
    let v = Float.of_int i *. Float.pi /. Float.of_int len in
    let c = Float.cos v and s = Float.sin v in
    let x = c +. noise () in
    let y = s +. noise () in
    let idx = i * 2 in
    Genarray.set coordinates [| idx; 0 |] x;
    Genarray.set coordinates [| idx; 1 |] y;
    Genarray.set labels [| idx; 0 |] 1.0
  done;

  (* Generate second moon (label = -1.0) *)
  for i = 0 to len - 1 do
    let v = Float.of_int i *. Float.pi /. Float.of_int len in
    let c = Float.cos v and s = Float.sin v in
    let x = 1.0 -. c +. noise () in
    let y = 0.5 -. s +. noise () in
    let idx = (i * 2) + 1 in
    Genarray.set coordinates [| idx; 0 |] x;
    Genarray.set coordinates [| idx; 1 |] y;
    Genarray.set labels [| idx; 0 |] (-1.0)
  done;

  (coordinates, labels)

(** Generate the half moons dataset with the specified parameters.

    @param config Configuration for noise and randomization
    @param len Number of samples per moon (total samples = len * 2)
    @return
      A tuple of (coordinates, labels) where:
      - coordinates is a bigarray of shape [len*2; 2] (batch_axis, output_axis)
      - labels is a bigarray of shape [len*2; 1] (batch_axis, output_axis)
      - First moon has label 1.0, second moon has label -1.0 *)
let generate ?(config = Config.default) ~len () = generate_with_kind float64 ~config ~len ()

(** Generate the half moons dataset with single precision floats.

    @param config Configuration for noise and randomization
    @param len Number of samples per moon (total samples = len * 2)
    @return
      A tuple of (coordinates, labels) where:
      - coordinates is a bigarray of shape [len*2; 2] (batch_axis, output_axis) with float32
        elements
      - labels is a bigarray of shape [len*2; 1] (batch_axis, output_axis) with float32 elements
      - First moon has label 1.0, second moon has label -1.0 *)
let generate_single_prec ?(config = Config.default) ~len () =
  generate_with_kind float32 ~config ~len ()

(** Generate half moons dataset using the old array-based approach for compatibility. This function
    is deprecated and provided for backwards compatibility.

    @param len Number of samples per moon
    @param noise_range Range of noise to add
    @return A tuple of (coordinates_array, labels_array) as flat arrays *)
let generate_arrays ?(noise_range = 0.1) ~len () =
  let noise () = Random.float_range ~-.noise_range noise_range in
  let coordinates =
    Array.concat
      (Array.to_list
         (Array.init len (fun _ ->
              let i = Random.int len in
              let v = Float.of_int i *. Float.pi /. Float.of_int len in
              let c = Float.cos v and s = Float.sin v in
              [| c +. noise (); s +. noise (); 1.0 -. c +. noise (); 0.5 -. s +. noise () |])))
  in
  let labels = Array.init (len * 2) (fun i -> if i mod 2 = 0 then 1. else -1.) in
  (coordinates, labels)
