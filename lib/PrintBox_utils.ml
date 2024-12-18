open Base
module Utils = Arrayjit.Utils
module Debug_runtime = Utils.Debug_runtime

type box = PrintBox.t

let sexp_of_box _ = String.sexp_of_t "<opaque>"

type dag =
  [ `Empty
  | `Pad of dag
  | `Frame of dag
  | `Align of [ `Left | `Center | `Right ] * [ `Top | `Center | `Bottom ] * dag
  | `Text of string
  | `Box of box
  | `Vlist of bool * dag list
  | `Hlist of bool * dag list
  | `Table of dag array array
  | `Tree of dag * dag list
  | `Embed_subtree_ID of string
  | `Subtree_with_ID of string * dag ]
[@@deriving sexp_of]

let rec boxify (depth : int) (b : dag) : dag =
  match b with
  | b when depth <= 0 -> b
  | `Tree (n, bs) when depth > 0 ->
      `Vlist
        ( false,
          [ `Align (`Center, `Bottom, n); `Hlist (true, List.map ~f:(boxify @@ (depth - 1)) bs) ] )
  | `Hlist (bars, bs) -> `Hlist (bars, List.map ~f:(boxify @@ (depth - 1)) bs)
  | `Vlist (bars, bs) -> `Vlist (bars, List.map ~f:(boxify @@ (depth - 1)) bs)
  | `Pad b -> `Pad (boxify depth b)
  | `Frame b -> `Frame (boxify depth b)
  | `Align (h, v, b) -> `Align (h, v, boxify depth b)
  | `Subtree_with_ID (id, b) -> `Subtree_with_ID (id, boxify depth b)
  | b -> b

let dag_to_box (b : dag) =
  let s : ('a, 'cmp) Comparator.Module.t = (module String) in
  let rec reused = function
    | `Embed_subtree_ID id -> Set.singleton s id
    | `Subtree_with_ID (_, b) -> reused b
    | `Pad b | `Frame b | `Align (_, _, b) -> reused b
    | `Empty | `Text _ | `Box _ -> Set.empty s
    | `Tree (n, bs) -> Set.union_list s (reused n :: List.map ~f:reused bs)
    | `Hlist (_, bs) -> Set.union_list s @@ List.map ~f:reused bs
    | `Vlist (_, bs) -> Set.union_list s @@ List.map ~f:reused bs
    | `Table bss ->
        Set.union_list s @@ Array.to_list
        @@ Array.concat_map bss ~f:(fun bs -> Array.map ~f:reused bs)
  in
  let reused = reused b in
  let open PrintBox in
  let rec convert = function
    | `Embed_subtree_ID id -> text ("[" ^ id ^ "]")
    | `Tree (n, bs) -> tree (convert n) (List.map ~f:convert bs)
    | `Subtree_with_ID (id, `Text x) when Set.mem reused id -> text ("[" ^ id ^ "] " ^ x)
    | `Subtree_with_ID (id, `Tree (`Text x, bs)) when Set.mem reused id ->
        convert @@ `Tree (`Text ("[" ^ id ^ "] " ^ x), bs)
    | `Subtree_with_ID (_, b) -> convert b
    | `Box b -> b
    | `Empty -> empty
    | `Pad b -> pad @@ convert b
    | `Frame b -> frame @@ convert b
    | `Align (h, v, b) -> align ~h ~v @@ convert b
    | `Text t -> text t
    | `Vlist (bars, l) -> vlist ~bars (List.map ~f:convert l)
    | `Hlist (bars, l) -> hlist ~bars (List.map ~f:convert l)
    | `Table a -> grid (map_matrix convert a)
  in
  convert b

let reformat_dag box_depth b = boxify box_depth b |> dag_to_box

type plot_spec =
  | Scatterplot of { points : (float * float) array; pixel : string }
  | Line_plot of { points : float array; pixel : string }
  | Boundary_map of { callback : float * float -> bool; pixel_true : string; pixel_false : string }
  | Line_plot_adaptive of {
      callback : float -> float;
      mutable cache : float Map.M(Float).t;
      pixel : string;
    }
[@@deriving sexp_of]

let plot_canvas ?canvas ?(size : (int * int) option) (specs : plot_spec list) :
    float * float * float * float * _ =
  let open Float in
  (* Unfortunately "x" and "y" of a "matrix" are opposite to how we want them displayed -- the first
     dimension (i.e. "x") as the horizontal axis. *)
  let (dimx, dimy, canvas) : int * int * _ =
    match (canvas, size) with
    | None, None -> invalid_arg "PrintBox_utils.plot: provide ~canvas or ~size"
    | None, Some (dimx, dimy) -> (dimx, dimy, Array.make_matrix ~dimx:dimy ~dimy:dimx " ")
    | Some canvas, None ->
        let dimy = Array.length canvas in
        let dimx = Array.length canvas.(0) in
        (dimx, dimy, canvas)
    | Some canvas, Some (dimx, dimy) ->
        assert (Int.(dimy = Array.length canvas));
        assert (Int.(dimx = Array.length canvas.(0)));
        (dimx, dimy, canvas)
  in
  let specs = Array.of_list specs in
  let all_x_points =
    Array.concat_map specs ~f:(function
      | Scatterplot { points; _ } -> Array.map ~f:fst points
      | Line_plot _ -> [||]
      | Boundary_map _ -> [||]
      | Line_plot_adaptive _ -> [||])
  in
  let given_y_points =
    Array.concat_map specs ~f:(function
      | Scatterplot { points; _ } -> Array.map ~f:snd points
      | Line_plot { points; _ } -> points
      | Boundary_map _ -> [||]
      | Line_plot_adaptive _ -> [||])
  in
  let extra_y_points =
    Array.concat_map specs ~f:(function
      | Line_plot_adaptive ({ callback; cache; _ } as plot) ->
          let cache =
            Array.fold all_x_points ~init:cache ~f:(fun cache x ->
                Map.update cache x ~f:(fun _ -> callback x))
          in
          plot.cache <- cache;
          Array.of_list @@ Map.data cache
      | _ -> [||])
  in
  let all_y_points = Array.append given_y_points extra_y_points in
  let minx = if Array.is_empty all_x_points then 0. else Array.reduce_exn all_x_points ~f:min in
  let miny = if Array.is_empty all_y_points then 0. else Array.reduce_exn all_y_points ~f:min in
  let maxx =
    if Array.is_empty all_x_points then of_int Int.(Array.length all_y_points - 1)
    else Array.reduce_exn all_x_points ~f:max
  in
  let maxy =
    if Array.is_empty all_y_points then maxx - minx else Array.reduce_exn all_y_points ~f:max
  in
  let spanx = maxx - minx in
  let spanx = Float.(if spanx < epsilon_float then 1.0 else spanx) in
  let spany = maxy - miny in
  let spany = Float.(if spany < epsilon_float then 1.0 else spany) in
  let scale_1d y =
    try Some (to_int @@ (of_int Int.(dimy - 1) * (y - miny) / spany))
    with Invalid_argument _ -> None
  in
  let scale_2d (x, y) =
    try
      Some
        ( to_int @@ (of_int Int.(dimx - 1) * (x - minx) / spanx),
          to_int @@ (of_int Int.(dimy - 1) * (y - miny) / spany) )
    with Invalid_argument _ -> None
  in
  Array.iter specs ~f:(function
    | Scatterplot { points; pixel } ->
        let points = Array.filter_map points ~f:scale_2d in
        Array.iter points ~f:Int.(fun (i, j) -> canvas.(dimy - 1 - j).(i) <- pixel)
    | Line_plot { points; pixel } ->
        let points = Array.map points ~f:scale_1d in
        let rescale_x i = to_int @@ (of_int i * of_int dimx / of_int (Array.length points)) in
        (* TODO: implement interpolation if not enough points. *)
        Array.iteri points
          ~f:Int.(fun i -> Option.iter ~f:(fun j -> canvas.(dimy - 1 - j).(rescale_x i) <- pixel))
    | Boundary_map { callback; pixel_true; pixel_false } ->
        Array.iteri canvas ~f:(fun dmj ->
            Array.iteri ~f:(fun i pix ->
                if String.is_empty @@ String.strip pix then
                  let x = (of_int i * spanx / of_int Int.(dimx - 1)) + minx in
                  let y = (of_int Int.(dimy - 1 - dmj) * spany / of_int Int.(dimy - 1)) + miny in
                  canvas.(dmj).(i) <- (if callback (x, y) then pixel_true else pixel_false)))
    | Line_plot_adaptive ({ callback; cache; pixel } as plot) ->
        Array.iteri canvas.(0) ~f:(fun i pix ->
            if String.is_empty @@ String.strip pix then
              let x = (of_int i * spanx / of_int Int.(dimx - 1)) + minx in
              let y =
                match Map.find cache x with
                | Some y -> y
                | None ->
                    let y = callback x in
                    plot.cache <- Map.add_exn cache ~key:x ~data:y;
                    y
              in
              Option.iter (scale_1d y) ~f:Int.(fun j -> canvas.(dimy - 1 - j).(i) <- pixel)));
  (minx, miny, maxx, maxy, canvas)

let concise_float = Arrayjit.Ndarray.concise_float

let plot ?(prec = 3) ?(no_axes = false) ?canvas ?size ?(x_label = "x") ?(y_label = "y") specs =
  let minx, miny, maxx, maxy, canvas = plot_canvas ?canvas ?size specs in
  let open PrintBox in
  let y_label_l = List.map ~f:String.of_char @@ String.to_list y_label in
  if no_axes then grid_text ~bars:false canvas
  else
    grid_l
      [
        [
          hlist ~bars:false
            [
              align ~h:`Left ~v:`Center @@ lines y_label_l;
              vlist ~bars:false
                [
                  line @@ concise_float ~prec maxy; align_bottom @@ line @@ concise_float ~prec miny;
                ];
            ];
          grid_text ~bars:false canvas;
        ];
        [
          empty;
          vlist ~bars:false
            [
              hlist ~bars:false
                [
                  line @@ concise_float ~prec minx; align_right @@ line @@ concise_float ~prec maxx;
                ];
              align ~h:`Center ~v:`Bottom @@ line x_label;
            ];
        ];
      ]

type table_row_spec =
  | Benchmark of {
      bench_title : string;
      time_in_sec : float;
      mem_in_bytes : int;
      result_label : string;
      result : Sexp.t;
    }
[@@deriving sexp_of]

let nolines = String.substr_replace_all ~pattern:"\n" ~with_:";"

let table rows =
  if List.is_empty rows then PrintBox.empty
  else
    let titles = List.map rows ~f:(fun (Benchmark { bench_title; _ }) -> nolines bench_title) in
    let times = List.map rows ~f:(fun (Benchmark { time_in_sec; _ }) -> time_in_sec) in
    let sizes = List.map rows ~f:(fun (Benchmark { mem_in_bytes; _ }) -> mem_in_bytes) in
    let max_time = List.reduce_exn ~f:Float.max times in
    let max_size = List.reduce_exn ~f:Int.max sizes in
    let speedups = List.map times ~f:(fun x -> max_time /. x) in
    let mem_gains = List.map sizes ~f:Float.(fun x -> of_int max_size / of_int x) in
    let small_float = Fn.compose PrintBox.line (Printf.sprintf "%.3f") in
    let results =
      List.map rows ~f:(fun (Benchmark { result; _ }) -> nolines @@ Sexp.to_string_hum result)
    in
    let result_labels =
      List.map rows ~f:(fun (Benchmark { result_label; _ }) -> nolines result_label)
    in
    (* TODO(#140): partition by unique result_label and output a vlist of records. *)
    PrintBox.(
      frame
      @@ record
           [
             ("Benchmarks", vlist_map ~bars:false line titles);
             ("Time in sec", vlist_map ~bars:false float_ times);
             ("Memory in bytes", vlist_map ~bars:false int_ sizes);
             ("Speedup", vlist_map ~bars:false small_float speedups);
             ("Mem gain", vlist_map ~bars:false small_float mem_gains);
             (List.hd_exn result_labels, vlist_map ~bars:false line results);
           ])
