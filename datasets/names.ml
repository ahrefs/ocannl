open Base
open Stdio

(* Access the site locations to find names.txt *)
let read_names () = 
  let data_locations : string list = Dataset_sites.Sites.data in
  let names_file = "names.txt" in
  let rec find_file = function
    | [] -> None
    | dir :: rest ->
        let filepath = Stdlib.Filename.concat dir names_file in
        if Stdlib.Sys.file_exists filepath then Some filepath
        else find_file rest
  in
  let filepath = match find_file data_locations with
    | Some path -> path
    | None ->
        (* Fallback for testing: try to find the file in common locations *)
        let fallback_paths = [
          names_file;  (* current directory *)
          "../../datasets/names.txt";  (* from test/training/ *)
          "../datasets/names.txt";     (* from test/ *)
          "datasets/names.txt";        (* from project root *)
        ] in
        let rec try_fallbacks = function
          | [] -> failwith (Printf.sprintf "Could not find %s in any location (sites: %s)" names_file (String.concat ~sep:"; " data_locations))
          | path :: rest ->
              if Stdlib.Sys.file_exists path then path
              else try_fallbacks rest
        in
        try_fallbacks fallback_paths
  in
  In_channel.read_lines filepath

let bigrams s =
  let chars = String.to_list s in
  let front = '.' :: chars in
  let back = chars @ [ '.' ] in
  List.zip_exn front back

let get_all_bigrams () = List.(read_names () >>| bigrams |> concat)
let letters = List.init 26 ~f:(fun i -> Char.of_int_exn (Char.to_int 'a' + i))

(* Round the number of tokens up to 28 so it's divisible by 4 as we are using the bit-efficient
   random number generator. *)
(* TODO: double check if this is necessary. *)
let letters_with_dot = '.' :: ' ' :: letters

let char_to_index_tbl =
  let tbl = Hashtbl.create (module Char) in
  List.iteri letters_with_dot ~f:(fun i c -> Hashtbl.set tbl ~key:c ~data:i);
  tbl

let char_index c =
  match Hashtbl.find char_to_index_tbl c with
  | Some i -> i
  | None -> failwith (Printf.sprintf "Character not found: %c" c)

let bigrams_to_indices bigrams = List.(bigrams >>| fun (c1, c2) -> (char_index c1, char_index c2))
let dict_size = List.length letters_with_dot

let char_to_one_hot c =
  let c_index = char_index c in
  let arr = Array.create ~len:dict_size 0. in
  arr.(c_index) <- 1.;
  arr