(** Synthetic dataset for counting circles in images *)

open Bigarray

module Config = struct
  type t = {
    image_size : int;  (** Width and height of the generated images *)
    max_radius : int;  (** Maximum radius for generated circles *)
    min_radius : int;  (** Minimum radius for generated circles *)
    max_circles : int;  (** Maximum number of circles per image *)
    seed : int option;  (** Optional random seed for reproducibility *)
  }

  let default =
    { image_size = 32; max_radius = 8; min_radius = 2; max_circles = 5; seed = None }
end

module Random = Rand.Random_for_tests

(** Draw a filled circle on the image at (cx, cy) with radius r.
    Values are clamped to [0, 1] range. *)
let draw_circle ~image_size image cx cy r =
  for y = 0 to image_size - 1 do
    for x = 0 to image_size - 1 do
      let dx = x - cx in
      let dy = y - cy in
      if (dx * dx) + (dy * dy) <= r * r then
        Genarray.set image [| y; x; 0 |] 1.0
    done
  done

(** Generate circle counting dataset with specified precision.

    @param kind The bigarray kind (float32 or float64)
    @param config Configuration for image and circle parameters
    @param len Number of images to generate
    @return
      A tuple of (images, labels) where:
      - images is a bigarray of shape [len; image_size; image_size; 1] (batch, height, width, channels)
      - labels is a bigarray of shape [len; 1] (batch, output) containing the circle count *)
let generate_with_kind kind ?(config = Config.default) ~len () =
  (match config.seed with Some seed -> Random.init seed | None -> ());

  let image_size = config.image_size in
  let max_radius = config.max_radius in
  let min_radius = config.min_radius in
  let max_circles = config.max_circles in
  let radius_range = max_radius - min_radius + 1 in

  (* Create bigarrays: batch, height, width, channels *)
  let images = Genarray.create kind c_layout [| len; image_size; image_size; 1 |] in
  let labels = Genarray.create kind c_layout [| len; 1 |] in

  (* Initialize images and labels to zero *)
  for i = 0 to len - 1 do
    Genarray.set labels [| i; 0 |] 0.0;
    for y = 0 to image_size - 1 do
      for x = 0 to image_size - 1 do
        Genarray.set images [| i; y; x; 0 |] 0.0
      done
    done
  done;

  (* Generate each image *)
  for i = 0 to len - 1 do
    (* Random number of circles (1 to max_circles) *)
    let num_circles = 1 + Random.int max_circles in

    (* Draw each circle *)
    for _ = 1 to num_circles do
      let r = min_radius + Random.int radius_range in
      (* Ensure circle center is within bounds so circle is at least partially visible *)
      let margin = max 0 (r - (image_size / 4)) in
      let cx = margin + Random.int (image_size - (2 * margin)) in
      let cy = margin + Random.int (image_size - (2 * margin)) in

      (* Draw into a view of this image *)
      let image_slice = Genarray.slice_left images [| i |] in
      draw_circle ~image_size image_slice cx cy r
    done;

    (* Store the count as a float *)
    Genarray.set labels [| i; 0 |] (Float.of_int num_circles)
  done;

  (images, labels)

(** Generate circle counting dataset with double precision.

    @param config Configuration for image and circle parameters
    @param len Number of images to generate
    @return
      A tuple of (images, labels) where:
      - images is a bigarray of shape [len; image_size; image_size; 1]
      - labels is a bigarray of shape [len; 1] containing the circle count *)
let generate ?(config = Config.default) ~len () = generate_with_kind float64 ~config ~len ()

(** Generate circle counting dataset with single precision.

    @param config Configuration for image and circle parameters
    @param len Number of images to generate
    @return
      A tuple of (images, labels) where:
      - images is a bigarray of shape [len; image_size; image_size; 1] with float32 elements
      - labels is a bigarray of shape [len; 1] with float32 elements containing the circle count *)
let generate_single_prec ?(config = Config.default) ~len () =
  generate_with_kind float32 ~config ~len ()
