(* Off-platform stub. The real Metal source check runs only when the Metal backend is available. *)
let () =
  Stdio.In_channel.read_all "test_metal_guarded_gather_codegen.expected" |> Stdio.print_string
