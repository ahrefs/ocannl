(* A visible Dune executable entry point for the Git fixture. The reader is passed as data; no shell
   changes Dune's working directory or configuration lookup. *)
let () =
  match Array.to_list Sys.argv with
  | [ _; fixture; reader ] -> Unix.execvp "python3" [| "python3"; fixture; reader |]
  | _ -> failwith "expected the fixture and API-reader executable paths"
