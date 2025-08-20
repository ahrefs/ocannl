open Base
open Stdio

let extract_config arg =
  let prefixes = [ "--read="; "--read-"; "--read_" ] in
  List.find_map prefixes ~f:(fun prefix ->
      Option.map (String.chop_prefix arg ~prefix) ~f:(fun config -> config))

let () =
  let config_opt = Array.find_map Stdlib.Sys.argv ~f:extract_config in
  match config_opt with
  | Some config -> (
      let value = Utils.get_global_arg ~default:"" ~arg_name:config in
      let filename = "ocannl_" ^ config ^ ".txt" in
      try
        Out_channel.write_all filename ~data:value;
        printf "Wrote value of '%s' to %s\n" config filename
      with exn -> eprintf "Error writing to %s: %s\n" filename (Exn.to_string exn))
  | None -> printf "No --read=<config>, --read-<config>, or --read_<config> argument found.\n"
