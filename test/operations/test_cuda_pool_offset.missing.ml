(* Off-platform stub. cudajit is absent on this host so dune's [select] (in this test's
   [libraries]) picks this module; it reproduces the recorded golden output so the test passes
   trivially. The real CUDA pool-offset test runs only on CUDA hosts (minipc-wsl), where
   test_cuda_pool_offset.real.ml is selected. *)
let () =
  Stdio.In_channel.read_all "test_cuda_pool_offset.expected" |> Stdio.print_string
