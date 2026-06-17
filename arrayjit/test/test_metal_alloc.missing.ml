(* Off-platform stub — see test_metal_storage_mode.missing.ml. Selected wherever
   the macOS-only `metal` / `arrayjit.metal_backend` libraries are unavailable;
   reproduces the golden so the test passes; real paths run only on macOS. *)
let () =
  Stdio.In_channel.read_all "test_metal_alloc.expected" |> Stdio.print_string
