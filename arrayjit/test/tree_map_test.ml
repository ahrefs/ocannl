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
  printf "Finding key 4: %s\n"
    (Option.value ~default:"not found" (find ~compare:Int.compare ~key:4 tree));
  printf "Finding key 10: %s\n"
    (Option.value ~default:"not found" (find ~compare:Int.compare ~key:10 tree));

  (* Print as association list *)
  printf "\nAs alist (in-order): ";
  printf "%s\n" (Sexp.to_string_hum ([%sexp_of: (int * string) list] (to_alist tree)))

(* gh-ocannl-1904: [sparse_mapi] is [mapi] restricted to its targets -- the same calls in the same
   order, the same result on the same tree shape -- and shares what it does not visit. *)
let () =
  let open Tree_map in
  let open Verdict.Claims in
  let compare = Int.compare in
  let keys = List.init 300 ~f:(fun i -> i * 7919 % 1000) in
  let tree = List.fold keys ~init:empty ~f:(fun t k -> add ~compare ~key:k ~data:k t) in
  let targets = List.filter keys ~f:(fun k -> k % 3 = 0) @ [ 1001; 1002 ] in
  let is_target k = List.mem targets k ~equal:Int.equal in
  let calls f =
    let seen = ref [] in
    let result =
      f ~f:(fun ~key ~data ->
          seen := key :: !seen;
          data + 1)
    in
    (result, List.rev !seen)
  in
  let whole, whole_calls =
    calls (fun ~f -> mapi tree ~f:(fun ~key ~data -> if is_target key then f ~key ~data else data))
  in
  let sparse, sparse_calls = calls (fun ~f -> sparse_mapi ~compare ~targets tree ~f) in
  let rec shape = function Empty -> [] | Node n -> n.key :: (shape n.left @ shape n.right) in
  p "sparse_mapi calls f on the targets in mapi's order"
    (List.equal Int.equal whole_calls sparse_calls);
  p "sparse_mapi leaves the entries mapi restricted to the targets leaves"
    ([%equal: (int * int) list] (to_alist whole) (to_alist sparse));
  p "sparse_mapi keeps the tree's shape" (List.equal Int.equal (shape tree) (shape sparse));
  p "sparse_mapi without targets returns the tree itself"
    (phys_equal tree (sparse_mapi ~compare ~targets:[] tree ~f:(fun ~key:_ ~data -> data)))
