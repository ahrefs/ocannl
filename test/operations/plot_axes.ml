(* Plot tick labels and legends (gh-ocannl-103). [PrintBox_utils.plot] lays its tick labels out
   around the upstream [printbox-ext-plot] canvas, whose data window it restates rather than reads
   (the upstream module does not export it). The claims below pin the two together: markers placed
   exactly at round values must land on the rows and columns the axes label with those values. The
   diagonal y = x makes the two axes check each other: under every x tick mark, the marker sits on
   the row whose y label is the same number. The markers are 11 points over a 40-row canvas, so a
   window or a scaling that drifted by one cell would leave a labeled row without its marker. *)

open Base
open Ocannl
open Verdict.Claims
module BPlot = PrintBox_ext_plot

(* The display characters of a rendered line: the canvas frame is drawn with multi-byte box
   characters, and the claims index lines by column. *)
let chars_of line =
  let n = String.length line in
  let rec loop pos acc =
    if pos >= n then List.rev acc
    else
      let len = ref 1 in
      while pos + !len < n && Char.to_int line.[pos + !len] land 0xC0 = 0x80 do
        Int.incr len
      done;
      loop (pos + !len) (String.sub line ~pos ~len:!len :: acc)
  in
  Array.of_list (loop 0 [])

let render box = PrintBox_text.to_string box
let lines_of s = Array.of_list (String.split ~on:'\n' s)

let diagonal =
  [
    BPlot.Scatterplot
      {
        points = Array.init 11 ~f:(fun i -> (Float.of_int i, Float.of_int i));
        content = PrintBox.line "#";
      };
  ]

let () =
  let text = render (PrintBox_utils.plot ~x_label:"x" ~y_label:"y" diagonal) in
  Stdio.print_endline "=== Diagonal y = x, markers at the integers 0..10 ===";
  Stdio.print_endline text;
  let lines = Array.map (lines_of text) ~f:chars_of in
  (* Rows of the canvas: the lines with a left frame character, labeled ones marked with [┤]. *)
  let frame_col =
    Array.find_map lines ~f:(fun l -> Array.findi l ~f:(fun _ c -> String.equal c "┌"))
    |> Option.value_exn ~message:"no canvas frame"
    |> fst
  in
  let is_row l =
    Array.length l > frame_col && (String.equal l.(frame_col) "│" || String.equal l.(frame_col) "┤")
  in
  let rows = Array.filter lines ~f:is_row in
  let label_of l =
    String.strip (String.concat (Array.to_list (Array.sub l ~pos:0 ~len:frame_col)))
  in
  let labeled = Array.filter rows ~f:(fun l -> String.equal l.(frame_col) "┤") in
  let labeled = Array.to_list labeled in
  p "the canvas has 40 rows" (Array.length rows = 40);
  p_all ~min:3 "every labeled y row carries a marker" labeled ~f:(fun l ->
      Array.exists l ~f:(String.equal "#"));
  p_all ~min:3 "every y label is a number" labeled ~f:(fun l ->
      Option.is_some (Float.of_string_opt (label_of l)));
  (* The bottom frame line: x tick marks [┬], with their labels on the next line. *)
  let bottom_idx =
    Array.findi lines ~f:(fun _ l -> Array.length l > frame_col && String.equal l.(frame_col) "└")
    |> Option.value_exn ~message:"no bottom frame"
    |> fst
  in
  let bottom = lines.(bottom_idx) and tick_line = lines.(bottom_idx + 1) in
  let x_ticks =
    List.filter_map
      (List.init (Array.length bottom) ~f:Fn.id)
      ~f:(fun col -> if String.equal bottom.(col) "┬" then Some col else None)
  in
  (* The label covering a column of the tick line: the maximal run of non-blank characters. *)
  let token_at col =
    let blank i = i < 0 || i >= Array.length tick_line || String.equal tick_line.(i) " " in
    if blank col then None
    else
      let rec left i = if blank (i - 1) then i else left (i - 1) in
      let rec right i = if blank (i + 1) then i else right (i + 1) in
      let l = left col and r = right col in
      Some (String.concat (Array.to_list (Array.sub tick_line ~pos:l ~len:(r - l + 1))))
  in
  p_all ~min:3 "every x tick mark has a label over its column" x_ticks ~f:(fun col ->
      Option.is_some (token_at col));
  p_all ~min:3 "under every x tick mark, the marker sits on the row with the same y label" x_ticks
    ~f:(fun col ->
      match token_at col with
      | None -> false
      | Some x_label -> (
          match
            List.filter (Array.to_list rows) ~f:(fun l ->
                col < Array.length l && String.equal l.(col) "#")
          with
          | [ row ] ->
              String.equal row.(frame_col) "┤"
              && Float.equal (Float.of_string (label_of row)) (Float.of_string x_label)
          | _ -> false))

(* A documentation example: two series with a legend, over a window whose ends are not round, so the
   end labels keep the concise format while the interior ticks are round. *)
let parabola_specs =
  let xs = Array.init 41 ~f:(fun i -> -1.9 +. (0.095 *. Float.of_int i)) in
  [
    BPlot.Scatterplot
      { points = Array.map xs ~f:(fun x -> (x, x *. x)); content = PrintBox.line "#" };
    BPlot.Scatterplot
      { points = Array.map xs ~f:(fun x -> (x, 2. *. x)); content = PrintBox.line "*" };
  ]

let parabola_legend = [ ("#", "x^2"); ("*", "2x, its derivative") ]

let () =
  Stdio.print_endline "=== Two series with a legend, small ===";
  Stdio.print_endline
    (render
       (PrintBox_utils.plot ~small:true ~x_label:"x" ~y_label:"f(x)" ~legend:parabola_legend
          parabola_specs))

let () =
  (* [~as_canvas] keeps the upstream canvas rendering, unlabeled. *)
  let upstream =
    BPlot.box
      {
        BPlot.default_config with
        axes = false;
        x_label = "";
        y_label = "";
        prec = Utils.settings.print_decimals_precision;
        specs = diagonal;
      }
  in
  p "as_canvas renders exactly the upstream canvas"
    (String.equal (render (PrintBox_utils.plot ~as_canvas:true diagonal)) (render upstream))

let () =
  (* The HTML rendering carries the same layout: the upstream canvas, positioned tick labels at both
     ends of each axis and between them, and the legend. *)
  let html = PrintBox_html.to_string (PrintBox_utils.plot ~legend:parabola_legend parabola_specs) in
  let count sub = List.length (String.substr_index_all html ~may_overlap:false ~pattern:sub) in
  p "the HTML embeds the upstream canvas" (String.is_substring html ~substring:"border:thin dotted");
  p_all "the HTML legend lists every description" parabola_legend ~f:(fun (_, d) ->
      String.is_substring html ~substring:d);
  p "the HTML x axis labels its left end"
    (String.is_substring html ~substring:"top:0;left:0px;transform:none");
  p "the HTML x axis labels its right end"
    (String.is_substring html ~substring:"top:0;left:799px;transform:translateX(-100%)");
  p "the HTML x axis has interior ticks" (count "translateX(-50%)" >= 2);
  p "the HTML y axis has interior ticks" (count "translateY(-50%)" >= 2)

let () =
  let md =
    PrintBox_md.to_string PrintBox_md.Config.default
      (PrintBox_utils.plot ~legend:parabola_legend parabola_specs)
  in
  p "the Markdown rendering carries the legend"
    (String.is_substring md ~substring:"2x, its derivative")
