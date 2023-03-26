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

type plot_spec =
  | Scatterplot of {points: (float * float) array; pixel: string}
  | Line_plot of {points: float array; pixel: string}

let plot ?canvas ?size specs =
  let open Float in
  (* Unfortunately "x" and "y" of a "matrix" are opposite to how we want them displayed --
     the first dimension (i.e. "x") as the horizontal axis. *)
  let dimx, dimy, canvas =
    match canvas, size with
    | None, None -> invalid_arg "PrintBox_utils.plot: provide ~canvas or ~size"
    | None, Some (dimx, dimy) -> dimx, dimy, Array.make_matrix ~dimx:dimy ~dimy:dimx " "
    | Some canvas, None ->
      let dimy = Array.length canvas in
      let dimx = Array.length canvas.(0) in
      dimx, dimy, canvas
    | Some canvas, Some (dimx, dimy) ->
      assert Int.(dimy = Array.length canvas);
      assert Int.(dimx = Array.length canvas.(0));
      dimx, dimy, canvas in
  let specs = Array.of_list specs in
  let all_x_points = Array.concat_map specs ~f:(function
      | Scatterplot {points; _} -> Array.map ~f:fst points
      | Line_plot _ -> [||]) in
  let all_y_points = Array.concat_map specs ~f:(function
      | Scatterplot {points; _} -> Array.map ~f:snd points
      | Line_plot {points; _} -> points) in
  let minx = Array.reduce_exn all_x_points ~f:min in
  let miny = Array.reduce_exn all_y_points ~f:min in
  let maxx = Array.reduce_exn all_x_points ~f:max in
  let maxy = Array.reduce_exn all_y_points ~f:max in
  let spanx = maxx - minx in
  let spany = maxy - miny in
  let scale_1d y = to_int @@ of_int Int.(dimy - 1) * (y - miny) / spany in
  let scale_2d (x, y) =
    to_int @@ of_int Int.(dimx - 1) * (x - minx) / spanx,
    to_int @@ of_int Int.(dimy - 1) * (y - miny) / spany in
  Array.iter specs ~f:(function
      | Scatterplot {points; pixel} ->
        let points = Array.map points ~f:scale_2d in
        Array.iter points ~f:Int.(fun (i, j) -> canvas.(dimy - 1 - j).(i) <- pixel)
      | Line_plot {points; pixel} ->
        let points = Array.map points ~f:scale_1d in
        let rescale_x i = to_int @@ of_int i * spanx / of_int (Array.length points) in
        (* TODO: implement interpolation if not enough points. *)
        Array.iteri points ~f:Int.(fun i j -> canvas.(dimy - 1 - j).(rescale_x i) <- pixel));
  canvas
