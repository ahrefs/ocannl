(* Off-platform stub. The Metal backend is macOS-only, so dune's [select] picks this module wherever
   [metal] is unavailable; it reproduces the recorded golden output. The real check runs only on
   macOS via metal_gpu_core_count.real.ml. *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "metal_gpu_core_count.expected" |> Stdio.print_string
