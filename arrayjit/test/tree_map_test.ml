open Base
open Stdio
open Utils

(* Demo of Tree_map with tree-preserving sexp output *)
let () =
  let open Tree_map in

  (* Create a tree by inserting values *)
  let tree =
    empty
    |> add ~compare:Int.compare ~key:5 ~data:"five"
    |> add ~compare:Int.compare ~key:3 ~data:"three"
    |> add ~compare:Int.compare ~key:7 ~data:"seven"
    |> add ~compare:Int.compare ~key:2 ~data:"two"
    |> add ~compare:Int.compare ~key:4 ~data:"four"
    |> add ~compare:Int.compare ~key:6 ~data:"six"
    |> add ~compare:Int.compare ~key:8 ~data:"eight"
  in

  (* Print the tree structure *)
  printf "Tree structure:\n";
  printf "%s\n\n" (Sexp.to_string_hum (sexp_of_t Int.sexp_of_t String.sexp_of_t tree));

  (* Test lookups *)
  printf "Finding key 4: %s\n" (Option.value ~default:"not found" (find ~compare:Int.compare ~key:4 tree));
  printf "Finding key 10: %s\n" (Option.value ~default:"not found" (find ~compare:Int.compare ~key:10 tree));

  (* Print as association list *)
  printf "\nAs alist (in-order): ";
  printf "%s\n" (Sexp.to_string_hum ([%sexp_of: (int * string) list] (to_alist tree)))
