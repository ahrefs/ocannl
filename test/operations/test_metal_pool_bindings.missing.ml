(* Off-platform stub. The Metal backend is macOS-only, so dune's [select] (in this test's
   [libraries]) picks this module wherever [metal] is unavailable. It reproduces the recorded golden
   output so the test passes trivially; the real Metal pooled-binding path runs only on macOS, where
   [test_metal_pool_bindings.real.ml] is selected instead. *)
let () =
  Stdio.In_channel.read_all "test_metal_pool_bindings.expected" |> Stdio.print_string
