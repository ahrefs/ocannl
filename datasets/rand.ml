(** Random number generator library with low quality but very reliable reproducibility. *)

module type Random = sig
  val init : int -> unit
  val float_range : float -> float -> float
  val char : unit -> char
  val int : int -> int
end

module Random_for_tests : Random = struct
  let rand = ref (1l : Int32.t)

  let rand_int32 () =
    let open Int32 in
    rand := logxor !rand (shift_left !rand 13);
    rand := logxor !rand (shift_right_logical !rand 17);
    rand := logxor !rand (shift_left !rand 5);
    !rand

  let init seed = rand := Int32.(add (of_int seed) 1l)

  let float_range low high =
    let raw = Int32.(to_float @@ rem (rand_int32 ()) 10000l) in
    (raw /. 10000. *. (high -. low)) +. low

  let char () = Char.chr @@ Int32.(to_int @@ rem (rand_int32 ()) 256l)
  let int high =
    (* Use abs to handle negative random values from xor-shift RNG *)
    Int32.(to_int @@ rem (abs (rand_int32 ())) @@ of_int high)
end

module Random_for_dummy_tests : Random = struct
  let rand = ref (1l : Int32.t)

  let rand_int32 () =
    let open Int32 in
    rand := add !rand 1l;
    if equal !rand 10000l then rand := 1l;
    !rand

  let init seed = rand := Int32.(add (of_int seed) 1l)

  let float_range low high =
    let raw = Int32.(to_float @@ rand_int32 ()) in
    (raw /. 10000. *. (high -. low)) +. low

  let char () = Char.chr @@ Int32.(to_int @@ rem (rand_int32 ()) 256l)
  let int high = Int32.(to_int @@ rem (rand_int32 ()) @@ of_int high)
end
