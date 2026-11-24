let () =
  match Sys.argv with
  | [| _; dune_file |] ->
      Stdio.In_channel.read_all dune_file
      |> Test_scan.Policy_invariance_scan.sweep_inventory
      |> Stdio.Out_channel.output_string Stdio.stdout
  | _ -> failwith "usage: sweep_policy_inventory DUNE_FILE"
