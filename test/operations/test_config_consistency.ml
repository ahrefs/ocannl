open Base
open Stdio

let extract_keys filename =
  In_channel.read_lines filename
  |> List.filter_map ~f:(fun line ->
         let line = String.strip line in
         if String.is_empty line
            || String.is_prefix ~prefix:"#" line
            || String.is_prefix ~prefix:"~~" line
         then None
         else
           match String.lsplit2 line ~on:'=' with
           | Some (key, _) ->
               let key =
                 String.lowercase
                 @@ String.strip ~drop:(fun c -> Char.equal '-' c || Char.equal ' ' c) key
               in
               let key =
                 if String.is_prefix key ~prefix:"ocannl" then
                   String.drop_prefix key 6 |> String.strip ~drop:(Char.equal '_')
                 else key
               in
               if String.is_empty key then None else Some key
           | None -> None)
  |> Set.of_list (module String)

let () =
  let reference_file = Stdlib.Sys.argv.(1) in
  let file_keys = extract_keys reference_file in
  let code_keys = Utils.known_config_keys in
  let extra_in_file = Set.diff file_keys code_keys in
  let extra_in_code = Set.diff code_keys file_keys in
  if not (Set.is_empty extra_in_file) then
    printf "Keys in reference file but not in code: %s\n"
      (String.concat ~sep:", " @@ Set.to_list extra_in_file);
  if not (Set.is_empty extra_in_code) then
    printf "Keys in code but not in reference file: %s\n"
      (String.concat ~sep:", " @@ Set.to_list extra_in_code);
  if Set.is_empty extra_in_file && Set.is_empty extra_in_code then
    printf "OK: all %d keys match between code and ocannl_config.reference.\n"
      (Set.length code_keys)
