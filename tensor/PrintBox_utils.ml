open Base
module BPlot = PrintBox_ext_plot
module B = PrintBox

type box = B.t

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
let concise_float = Ir.Ndarray.concise_float
let () = BPlot.concise_float := concise_float

(* Plot axes and legends (gh-ocannl-103). [printbox-ext-plot] 0.12 labels only the two ends of each
   axis and has no legend. Rather than forking its renderer, [plot] renders the upstream canvas
   alone (its [axes = false] form) and lays the tick labels and the legend out around that
   rendering, in a PrintBox extension of our own with a handler per backend. The tick positions need
   the data window the canvas was drawn in, which the upstream module computes but does not export:
   [data_window] restates that computation, and the focused test [plot_axes] pins the two together
   by placing markers exactly at tick values and checking they land on the labeled rows and
   columns. *)

(* The [(minx, miny, maxx, maxy)] window [PrintBox_ext_plot.plot_canvas] maps onto a canvas [dimx]
   cells wide; [sparse] is the HTML backend's sampling of adaptive line plots. Evaluating an
   adaptive plot's callback here goes through the plot's own cache, so the canvas sees the same
   values. *)
let data_window ~dimx ~sparse (specs : BPlot.plot_spec list) =
  let all_x_points =
    Array.concat
    @@ List.map specs ~f:(function
      | BPlot.Scatterplot { points; _ } -> Array.map points ~f:fst
      | Scatterbag { points } -> Array.map points ~f:(fun ((x, _), _) -> x)
      | Line_plot _ | Map _ | Boundary_map _ | Line_plot_adaptive _ -> [||])
  in
  let given_y_points =
    Array.concat
    @@ List.map specs ~f:(function
      | BPlot.Scatterplot { points; _ } -> Array.map points ~f:snd
      | Scatterbag { points } -> Array.map points ~f:(fun ((_, y), _) -> y)
      | Line_plot { points; _ } -> points
      | Map _ | Boundary_map _ | Line_plot_adaptive _ -> [||])
  in
  let fold_min a = Array.fold a ~init:a.(0) ~f:Float.min in
  let fold_max a = Array.fold a ~init:a.(0) ~f:Float.max in
  let minx = if Array.is_empty all_x_points then 0. else fold_min all_x_points in
  let maxx =
    if Array.is_empty given_y_points then 1.
    else if Array.is_empty all_x_points then Float.of_int (Array.length given_y_points - 1)
    else fold_max all_x_points
  in
  let spanx = maxx -. minx in
  let spanx = if Float.(spanx <= epsilon_float) then 1.0 else spanx in
  let unscale_x i = (Float.of_int i *. spanx /. Float.of_int (dimx - 1)) +. minx in
  let extra_y_points =
    Array.concat
    @@ List.map specs ~f:(function
      | BPlot.Line_plot_adaptive { callback; cache; _ } ->
          Array.init
            (if sparse then dimx / 5 else dimx)
            ~f:(fun i ->
              let x = unscale_x (if sparse then i * 5 else i) in
              match Stdlib.Hashtbl.find_opt cache x with
              | Some y -> y
              | None ->
                  let y = callback x in
                  Stdlib.Hashtbl.add cache x y;
                  y)
      | _ -> [||])
  in
  let all_y_points = Array.append given_y_points extra_y_points in
  let miny = if Array.is_empty all_y_points then 0. else fold_min all_y_points in
  let maxy = if Array.is_empty all_y_points then maxx -. minx else fold_max all_y_points in
  (minx, miny, maxx, maxy)

type tick = { cell : int; label : string }

(* The cell a value maps to along an axis of [cells] cells: the upstream canvas's truncating
   scaling, so a marker placed at a labeled value lands on the labeled cell. *)
let cell_of ~cells ~lo ~span v = Float.to_int (Float.of_int (cells - 1) *. (v -. lo) /. span)

(* Round [raw] to the nearest of 1, 2 and 5 times a power of ten, in log scale; also returns the
   number of decimals that prints every multiple of the result exactly. *)
let nice_step raw =
  let e = Float.round_down (Float.log10 raw) in
  let m = raw /. (10. **. e) in
  let mantissa, e =
    if Float.(m < sqrt 2.) then (1., e)
    else if Float.(m < sqrt 10.) then (2., e)
    else if Float.(m < sqrt 50.) then (5., e)
    else (1., e +. 1.)
  in
  (mantissa *. (10. **. e), Int.max 0 (-Float.to_int e))

(* Tick labels along one axis of [cells] cells over the window [lo, hi]. The two ends are always
   labeled; interior ticks are the multiples of a nice step, about [target] intervals per axis, each
   kept only if its label clears its neighbours by [min_gap] cells. [extent label] is the number of
   cells a label occupies: an interior label is centered on its cell, the end labels extend
   inward. *)
let axis_ticks ~cells ~lo ~hi ~prec ~target ~extent ~min_gap =
  let span = hi -. lo in
  let span = if Float.(span <= epsilon_float) then 1.0 else span in
  let hi = lo +. span in
  let concise v = concise_float ~prec v in
  let step =
    if Float.is_finite span && Float.is_finite lo && cells > 1 then
      Some (nice_step (span /. Float.of_int (Int.max 1 target)))
    else None
  in
  (* Within moderate magnitudes the labels are plain decimals: the multiples of the step with just
     the step's decimals, the window's ends with the precision's significant digits besides; beyond
     them, every label takes the concise scientific form. *)
  let plain = Float.(max (abs lo) (abs hi) < 1e5) in
  let fixed decimals v = Printf.sprintf "%.*f" decimals (if Float.(v = 0.) then 0. else v) in
  let tick_label =
    match step with
    | Some (step, decimals) when plain && Float.(step >= 1e-4) -> fixed decimals
    | _ -> concise
  in
  let end_label v =
    match step with
    | Some (step, decimals) when plain && Float.(step >= 1e-4) ->
        if Float.(abs ((v /. step) -. round (v /. step)) < 1e-6) then fixed decimals v
        else
          let magnitude = Float.(to_int (round_down (log10 (abs v)))) in
          fixed (Int.min 6 (Int.max decimals (prec - magnitude))) v
    | _ -> concise v
  in
  let lo_tick = { cell = 0; label = end_label lo } in
  let hi_tick = { cell = cells - 1; label = end_label hi } in
  let hi_start = cells - extent hi_tick.label in
  let interior =
    match step with
    | None -> []
    | Some (step, _) ->
        let first = Float.to_int (Float.round_up (lo /. step)) in
        let last = Float.to_int (Float.round_down (hi /. step)) in
        let rec loop k last_end acc =
          if k > last then List.rev acc
          else
            let v = Float.of_int k *. step in
            let cell = cell_of ~cells ~lo ~span v in
            let label = tick_label v in
            let start = cell - (extent label / 2) in
            let fin = start + extent label in
            if
              start - last_end >= min_gap
              && hi_start - fin >= min_gap
              && cell > 0
              && cell < cells - 1
            then loop (k + 1) fin ({ cell; label } :: acc)
            else loop (k + 1) last_end acc
        in
        loop first (extent lo_tick.label) []
  in
  if cells <= 1 then [ lo_tick ] else (lo_tick :: interior) @ [ hi_tick ]

type labeled_plot = { graph : BPlot.graph; legend : (string * string) list }
type PrintBox.ext += Labeled_plot of labeled_plot

let display_width s = PrintBox_text.str_display_width s 0 (String.length s)
let spaces n = String.make (Int.max 0 n) ' '

let legend_box legend =
  B.frame @@ B.vlist ~bars:false
  @@ List.map legend ~f:(fun (marker, label) -> B.line (marker ^ " " ^ label))

(* The text layout: the y-axis label above, the framed canvas with tick labels left of its rows and
   below its columns (each labeled row and column marked on the frame), the x-axis label centered
   below, and the legend box right-aligned under the plot. *)
let text_layout ~style { graph; legend } =
  let sx, sy = graph.size in
  let cx, cy = !BPlot.scale_size_for_text in
  let dimx = Float.(to_int (cx *. of_int sx)) and dimy = Float.(to_int (cy *. of_int sy)) in
  let minx, miny, maxx, maxy = data_window ~dimx ~sparse:false graph.specs in
  let canvas =
    String.split ~on:'\n'
    @@ PrintBox_text.to_string_with ~style
         (BPlot.box { graph with axes = false; x_label = ""; y_label = "" })
  in
  let canvas = Array.of_list canvas in
  if Array.length canvas <> dimy + 2 then
    invalid_arg
      (Printf.sprintf "PrintBox_utils.plot: expected the canvas to render as %d lines, got %d"
         (dimy + 2) (Array.length canvas));
  let y_ticks =
    axis_ticks ~cells:dimy ~lo:miny ~hi:maxy ~prec:graph.prec ~target:(dimy / 3)
      ~extent:(fun _ -> 1)
      ~min_gap:1
  in
  let x_label_width =
    List.fold [ minx; maxx ] ~init:4 ~f:(fun acc v ->
        Int.max acc (String.length (concise_float ~prec:graph.prec v)))
  in
  let x_ticks =
    axis_ticks ~cells:dimx ~lo:minx ~hi:maxx ~prec:graph.prec
      ~target:(dimx / (x_label_width + 6))
      ~extent:String.length ~min_gap:2
  in
  let leftw = List.fold y_ticks ~init:0 ~f:(fun acc t -> Int.max acc (String.length t.label)) in
  let row_label = Array.create ~len:dimy None in
  List.iter y_ticks ~f:(fun t -> row_label.(dimy - 1 - t.cell) <- Some t.label);
  let rows =
    List.init dimy ~f:(fun dmj ->
        let line = canvas.(dmj + 1) in
        match row_label.(dmj) with
        | None -> spaces leftw ^ line
        | Some label ->
            let line =
              match String.chop_prefix line ~prefix:"│" with
              | Some rest -> "┤" ^ rest
              | None -> line
            in
            spaces (leftw - String.length label) ^ label ^ line)
  in
  let bottom =
    let plain = "└" ^ String.concat (List.init dimx ~f:(fun _ -> "─")) ^ "┘" in
    let line = canvas.(dimy + 1) in
    if String.equal line plain then (
      let marked = Array.create ~len:dimx false in
      List.iter x_ticks ~f:(fun t -> marked.(t.cell) <- true);
      "└" ^ String.concat (List.init dimx ~f:(fun i -> if marked.(i) then "┬" else "─")) ^ "┘")
    else line
  in
  let x_tick_line =
    let buf = Bytes.make dimx ' ' in
    let n = List.length x_ticks in
    List.iteri x_ticks ~f:(fun idx t ->
        let len = String.length t.label in
        let start = if idx = 0 then 0 else if idx = n - 1 then dimx - len else t.cell - (len / 2) in
        let start = Int.max 0 (Int.min start (dimx - len)) in
        Bytes.From_string.blit ~src:t.label ~src_pos:0 ~dst:buf ~dst_pos:start
          ~len:(Int.min len dimx));
    String.rstrip (Bytes.to_string buf)
  in
  let total_width = leftw + dimx + 2 in
  let legend_lines =
    if List.is_empty legend then []
    else
      let lines =
        String.split ~on:'\n' @@ PrintBox_text.to_string_with ~style (legend_box legend)
      in
      let w = List.fold lines ~init:0 ~f:(fun acc l -> Int.max acc (display_width l)) in
      List.map lines ~f:(fun l -> spaces (total_width - w) ^ l)
  in
  List.concat
    [
      (if String.is_empty graph.y_label then [] else [ graph.y_label ]);
      [ spaces leftw ^ canvas.(0) ];
      rows;
      [ spaces leftw ^ bottom; spaces (leftw + 1) ^ x_tick_line ];
      (if String.is_empty graph.x_label then []
       else [ spaces (leftw + 1 + ((dimx - display_width graph.x_label) / 2)) ^ graph.x_label ]);
      legend_lines;
    ]

module H = Tyxml.Html

(* The HTML layout mirrors the text one, in pixels: the canvas is [size] pixels, the tick labels are
   absolutely positioned against it, and their spacing assumes a monospace font of about 8 by 16
   pixels. *)
let html_layout config { graph; legend } =
  let dimx, dimy = graph.size in
  let minx, miny, maxx, maxy = data_window ~dimx ~sparse:true graph.specs in
  let char_px = 8 and line_px = 16 in
  let y_ticks =
    axis_ticks ~cells:dimy ~lo:miny ~hi:maxy ~prec:graph.prec ~target:(dimy / 60)
      ~extent:(fun _ -> line_px)
      ~min_gap:(line_px / 2)
  in
  let x_label_width =
    List.fold [ minx; maxx ] ~init:4 ~f:(fun acc v ->
        Int.max acc (String.length (concise_float ~prec:graph.prec v)))
  in
  let x_ticks =
    axis_ticks ~cells:dimx ~lo:minx ~hi:maxx ~prec:graph.prec
      ~target:(dimx / ((x_label_width + 6) * char_px))
      ~extent:(fun l -> String.length l * char_px)
      ~min_gap:(2 * char_px)
  in
  let px n = Int.to_string n ^ "px" in
  let leftw = List.fold y_ticks ~init:0 ~f:(fun acc t -> Int.max acc (String.length t.label)) in
  let positioned style label =
    H.div ~a:[ H.a_style ("position:absolute;white-space:nowrap;" ^ style) ] [ H.txt label ]
  in
  let n_x = List.length x_ticks and n_y = List.length y_ticks in
  let y_column =
    H.div
      ~a:
        [
          H.a_style
            ("position:relative;height:" ^ px dimy ^ ";min-width:"
            ^ Int.to_string (leftw + 1)
            ^ "ch");
        ]
      (List.mapi y_ticks ~f:(fun idx t ->
           let top = dimy - 1 - t.cell in
           let shift =
             if idx = 0 then "translateY(-100%)"
             else if idx = n_y - 1 then "none"
             else "translateY(-50%)"
           in
           positioned ("right:0.5ch;top:" ^ px top ^ ";transform:" ^ shift) (t.label ^ " ─")))
  in
  let x_row =
    H.div
      ~a:[ H.a_style ("position:relative;height:1.5em;width:" ^ px dimx) ]
      (List.mapi x_ticks ~f:(fun idx t ->
           let shift =
             if idx = 0 then "none"
             else if idx = n_x - 1 then "translateX(-100%)"
             else "translateX(-50%)"
           in
           positioned ("top:0;left:" ^ px t.cell ^ ";transform:" ^ shift) t.label))
  in
  let canvas =
    PrintBox_html.to_html ~config
      (BPlot.box { graph with axes = false; x_label = ""; y_label = "" })
  in
  let cell ?(style = "") content = H.div ~a:[ H.a_style style ] content in
  let empty = cell [] in
  H.div
    ~a:
      [
        H.a_style
          "display:inline-grid;grid-template-columns:auto \
           auto;column-gap:0.25ch;font-family:monospace";
      ]
    (List.concat
       [
         (if String.is_empty graph.y_label then []
          else [ cell ~style:"grid-column:1 / span 2" [ H.txt graph.y_label ] ]);
         [ y_column; cell [ canvas ]; empty; x_row ];
         (if String.is_empty graph.x_label then []
          else [ empty; cell ~style:"text-align:center" [ H.txt graph.x_label ] ]);
         (if List.is_empty legend then []
          else
            [
              empty;
              cell ~style:"justify-self:end" [ PrintBox_html.to_html ~config (legend_box legend) ];
            ]);
       ])

let () =
  PrintBox_text.register_extension ~key:"Ocannl_labeled_plot" (fun ~style -> function
    | Labeled_plot p -> String.concat ~sep:"\n" (text_layout ~style p)
    | _ -> invalid_arg "PrintBox_utils: unrecognized plot extension");
  PrintBox_md.register_extension ~key:"Ocannl_labeled_plot" (fun config -> function
    | Labeled_plot p -> PrintBox_md.to_string config (B.lines (text_layout ~style:false p))
    | _ -> invalid_arg "PrintBox_utils: unrecognized plot extension");
  PrintBox_html.register_extension ~key:"Ocannl_labeled_plot" (fun config -> function
    | Labeled_plot p -> (html_layout config p :> PrintBox_html.toplevel_html)
    | _ -> invalid_arg "PrintBox_utils: unrecognized plot extension")

let plot ?(as_canvas = false) ?x_label ?y_label ?axes ?size ?(small = false) ?(legend = []) specs =
  let default = BPlot.default_config in
  let x_label = if as_canvas then "" else Option.value x_label ~default:default.x_label in
  let y_label = if as_canvas then "" else Option.value y_label ~default:default.y_label in
  let axes = if as_canvas then false else Option.value axes ~default:default.axes in
  let size = Option.value size ~default:default.size in
  let size = if small then (fst size / 4, snd size / 4) else size in
  let graph =
    { BPlot.size; prec = Utils.settings.print_decimals_precision; axes; x_label; y_label; specs }
  in
  if axes then B.extension ~key:"Ocannl_labeled_plot" (Labeled_plot { graph; legend })
  else if List.is_empty legend then BPlot.box graph
  else B.vlist ~bars:false [ BPlot.box graph; B.align_right (legend_box legend) ]

type table_row_spec =
  | Benchmark of {
      bench_title : string;
      time_in_sec : float;
      mem_in_bytes : int;
      result_label : string;
      result : Sexp.t;
    }

let nolines = String.substr_replace_all ~pattern:"\n" ~with_:";"

let render_group result_label group_rows =
  let titles = List.map group_rows ~f:(fun (Benchmark { bench_title; _ }) -> nolines bench_title) in
  let times = List.map group_rows ~f:(fun (Benchmark { time_in_sec; _ }) -> time_in_sec) in
  let sizes = List.map group_rows ~f:(fun (Benchmark { mem_in_bytes; _ }) -> mem_in_bytes) in
  let max_time = List.reduce_exn ~f:Float.max times in
  let max_size = List.reduce_exn ~f:Int.max sizes in
  let speedups = List.map times ~f:(fun x -> max_time /. x) in
  let mem_gains = List.map sizes ~f:Float.(fun x -> of_int max_size / of_int x) in
  let small_float = Fn.compose PrintBox.line (Printf.sprintf "%.3f") in
  let results =
    List.map group_rows ~f:(fun (Benchmark { result; _ }) -> nolines @@ Sexp.to_string_hum result)
  in
  PrintBox.(
    record
      [
        ("Benchmarks", vlist_map ~bars:false line titles);
        ("Time in sec", vlist_map ~bars:false float_ times);
        ("Memory in bytes", vlist_map ~bars:false int_ sizes);
        ("Speedup", vlist_map ~bars:false small_float speedups);
        ("Mem gain", vlist_map ~bars:false small_float mem_gains);
        (result_label, vlist_map ~bars:false line results);
      ])

let table rows =
  if List.is_empty rows then PrintBox.empty
  else
    (* Accumulate all rows with the same label into one group, preserving first-seen label order.
       Using Map rather than List.group so that interleaved labels (ms, MB, ms) don't produce
       duplicate sub-tables with metrics computed against different local maxima. *)
    let groups =
      let map, order_rev =
        List.fold rows
          ~init:(Map.empty (module String), [])
          ~f:(fun (acc, order) row ->
            let (Benchmark { result_label; _ }) = row in
            let label = nolines result_label in
            let order' = if Map.mem acc label then order else label :: order in
            let acc' =
              Map.update acc label ~f:(function None -> [ row ] | Some rs -> row :: rs)
            in
            (acc', order'))
      in
      List.rev_map order_rev ~f:(fun label -> (label, List.rev (Map.find_exn map label)))
    in
    let group_boxes =
      List.map groups ~f:(fun (label, group_rows) -> render_group label group_rows)
    in
    PrintBox.(
      frame @@ match group_boxes with [ single ] -> single | multiple -> vlist ~bars:true multiple)
