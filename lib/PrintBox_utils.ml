open Base
module Utils = Arrayjit.Utils
module BPlot = PrintBox_ext_plot
module B = PrintBox
module Debug_runtime = Utils.Debug_runtime

type box = B.t

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
let concise_float = Arrayjit.Ndarray.concise_float
let () = BPlot.concise_float := concise_float

let plot ?(as_canvas = false) ?x_label ?y_label ?axes ?size ?(small = false) specs =
  let default = BPlot.default_config in
  let x_label = if as_canvas then "" else Option.value x_label ~default:default.x_label in
  let y_label = if as_canvas then "" else Option.value y_label ~default:default.y_label in
  let axes = if as_canvas then false else Option.value axes ~default:default.axes in
  let size = Option.value size ~default:default.size in
  let size = if small then (fst size / 4, snd size / 4) else size in
  BPlot.box
    { BPlot.size; prec = Utils.settings.print_decimals_precision; axes; x_label; y_label; specs }

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
