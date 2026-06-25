(* Off-platform stub. The Metal backend is macOS-only, so dune's [select] picks this module wherever
   [metal] is unavailable; it reproduces the recorded golden output. The real Metal slot-width check
   runs only on macOS via test_metal_pool_slot_width.real.ml. *)
let () = Stdio.In_channel.read_all "test_metal_pool_slot_width.expected" |> Stdio.print_string
