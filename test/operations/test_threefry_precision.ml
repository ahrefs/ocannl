open Base
open Ocannl

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.log_level <- 1;
  let open Operation.DSL_modules in
  (* Create a simple Threefry4x32 operation *)
  let key = TDSL.number ~label:[ "key" ] 42.0 in
  let counter = TDSL.number ~label:[ "counter" ] 1.0 in
  let rng_result = TDSL.threefry4x32 ~label:[ "rng_result" ] key counter () in

  (* Print the precision of the result *)
  Stdlib.Printf.printf "Threefry4x32 result precision: %s\n"
    (Ir.Ops.prec_string (Lazy.force rng_result.value.prec));

  (* Try to use it in a computation - this should trigger the error *)
  let uniform_result = TDSL.uint4x32_to_prec_uniform ~label:[ "uniform" ] rng_result () in
  Stdlib.Printf.printf "Uniform result precision: %s\n"
    (Ir.Ops.prec_string (Lazy.force uniform_result.value.prec));
  let module Backend = (val Backends.fresh_backend ()) in
  try
    let _ctx = Train.forward_once (module Backend) uniform_result in
    Stdlib.Printf.printf "Compilation successful!\n";
    (* Also check the actual value precision in the context *)
    let tn = rng_result.value in
    Stdlib.Printf.printf "Actual tensor precision in context: %s\n"
      (Ir.Ops.prec_string (Lazy.force tn.prec))
  with
  | Utils.User_error msg -> Stdlib.Printf.printf "Error: %s\n" msg
  | e -> Stdlib.Printf.printf "Unexpected error: %s\n" (Exn.to_string e)
