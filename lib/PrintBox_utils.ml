open Base

type box = PrintBox.t
let sexp_of_box _ = String.sexp_of_t "<opaque>"

type dag =
  [ `Empty
  | `Pad of dag
  | `Frame of dag
  | `Align of [`Left | `Center | `Right] * [`Top | `Center | `Bottom] * dag
  | `Text of string
  | `Box of box
  | `Vlist of bool * dag list
  | `Hlist of bool * dag list
  | `Table of dag array array
  | `Tree of dag * dag list
  | `Embed_subtree_ID of string
  | `Subtree_with_ID of string * dag
  ]
[@@deriving sexp_of]

let rec boxify (depth: int) (b: dag): dag  = match b with
  | b when depth <= 0 -> b
  | `Tree (n, bs) when depth > 0 ->
    `Vlist (false, [`Align (`Center, `Bottom, n);
                    `Hlist (true, List.map ~f:(boxify @@ depth - 1) bs)])
  | `Hlist (bars, bs) -> `Hlist (bars, List.map ~f:(boxify @@ depth - 1) bs)
  | `Vlist (bars, bs) -> `Vlist (bars, List.map ~f:(boxify @@ depth - 1) bs)
  | `Pad b -> `Pad (boxify depth b)
  | `Frame b -> `Frame (boxify depth b)
  | `Align (h, v, b) -> `Align (h, v, boxify depth b)
  | `Subtree_with_ID (id, b) -> `Subtree_with_ID (id, boxify depth b)
  | b -> b

let dag_to_box (b: dag) =
  let s: ('a, 'cmp) Comparator.Module.t = (module String) in
  let rec reused = function
  | `Embed_subtree_ID id -> Set.singleton s id
  | `Subtree_with_ID (_, b) -> reused b
  | `Pad b | `Frame b | `Align (_, _, b) -> reused b
  | `Empty | `Text _ | `Box _ -> Set.empty s
  | `Tree (n, bs) -> Set.union_list s (reused n::List.map ~f:reused bs)
  | `Hlist (_, bs) -> Set.union_list s @@ List.map ~f:reused bs
  | `Vlist (_, bs) -> Set.union_list s @@ List.map ~f:reused bs
  | `Table bss ->
    Set.union_list s @@ Array.to_list @@ Array.concat_map bss
      ~f:(fun bs -> Array.map ~f:reused bs) in
  let reused = reused b in
  let open PrintBox in
  let rec convert = function
    | `Embed_subtree_ID id -> text ("["^id^"]")
    | `Tree (n, bs) -> tree (convert n) (List.map ~f:convert bs)
    | `Subtree_with_ID (id, `Text x) when Set.mem reused id ->
      text ("["^id^"] "^x)
    | `Subtree_with_ID (id, `Tree (`Text x, bs)) when Set.mem reused id ->
      convert @@ `Tree (`Text ("["^id^"] "^x), bs)
    | `Subtree_with_ID (_, b) -> convert b
    | `Box b -> b
    | `Empty -> empty
    | `Pad b -> pad @@ convert b
    | `Frame b -> frame @@ convert b
    | `Align (h, v, b) -> align ~h ~v @@ convert b
    | `Text t -> text t
    | `Vlist (bars, l) -> vlist ~bars (List.map ~f:convert l)
    | `Hlist (bars, l) -> hlist ~bars (List.map ~f:convert l)
    | `Table a -> grid (map_matrix convert a) in
  convert b

let reformat_dag box_depth b =
  boxify box_depth b |> dag_to_box
