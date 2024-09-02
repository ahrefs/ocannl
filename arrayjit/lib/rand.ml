open Base

module type Random = sig
  val init : int -> unit
  val float_range : float -> float -> float
  val char : unit -> char
  val int : int -> int
end

module Random_for_tests = struct
  let rand = ref (1l : Int32.t)

  let rand_int32 () =
    let open Int32 in
    rand := !rand lxor shift_left !rand 13;
    rand := !rand lxor shift_right_logical !rand 17;
    rand := !rand lxor shift_left !rand 5;
    !rand

  let init seed = rand := Int32.(of_int_trunc seed + 1l)

  let float_range low high =
    let raw = Int32.(to_float @@ (rand_int32 () % 10000l)) in
    (raw /. 10000. *. (high -. low)) +. low

  let char () = Char.of_int_exn @@ Int32.(to_int_trunc @@ (rand_int32 () % 256l))
  let int high = Int32.(to_int_trunc @@ (rand_int32 () % of_int_trunc high))
end

let random_config = Utils.get_global_arg ~arg_name:"randomness_lib" ~default:"stdlib"

let random_lib =
  match random_config with
  | "stdlib" -> (module Base.Random : Random)
  | "for_tests" -> (module Random_for_tests : Random)
  | _ ->
      invalid_arg
      @@ "Rand.random_lib: invalid setting of the global argument randomness_lib, expected one of: \
          stdlib, for_tests; found: " ^ random_config

module Lib = (val random_lib)
