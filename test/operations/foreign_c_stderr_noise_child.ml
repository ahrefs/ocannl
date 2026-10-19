(* gh-ocannl-1370: the child [foreign_c_stderr_noise] runs. It installs the collapsing C stream the
   way the HIP backend does -- by forcing [Utils.c_stderr_noise_collapsed], so the setting the
   parent passes on the commandline decides -- reports on stdout whether it took, then makes the
   scripted writes through the C runtime's [stderr] and exits normally, so the exit-time summary
   runs too. *)

external c_stderr : string -> unit = "ocannl_test_noise_c_stderr"
external close_c_stderr : unit -> unit = "ocannl_test_noise_close_c_stderr"

(* With [close] among its arguments the child then [fclose]s the C stream, as daemonizing code may:
   the count must be reported at the close, and the exit hook must not touch the closed FILE. *)
let () =
  let installed = Lazy.force Utils.c_stderr_noise_collapsed in
  Stdio.print_endline (if installed then "stream: collapsing" else "stream: unchanged");
  List.iter c_stderr Foreign_c_stderr_noise_script.writes;
  if Array.exists (String.equal "close") Sys.argv then close_c_stderr ()
