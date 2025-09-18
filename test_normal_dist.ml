open Base
open Ocannl.Operation.DSL_modules

let () =
  let open TDSL in
  let open O in
  
  (* Test normal distribution generation *)
  let n1 = normal () in
  let n2 = normal1 () in
  
  let counter = !@"counter" in
  let n3 = normal_at counter in
  let n4 = normal_at1 counter in
  
  Printf.printf "Normal distribution functions created successfully\n"