(* gh-ocannl-1258: [Ndarray.set_flat_values] -- the host writer behind [Context.set_values] and
   multi-value host initialization -- against the per-cell [Ndarray.set_from_float] it must agree
   with, in every storage precision, padded and unpadded. The per-cell oracle walks the
   before-padding index space with nested loops and lets [set_from_float] apply the margins, so it
   shares no run arithmetic with the writer. Margins and cells past the end of the values keep a
   sentinel both buffers were filled with; the raw buffers, margins included, must be equal. *)
open Base
open Verdict.Claims
module Nd = Ir.Ndarray
module Ops = Ir.Ops

let sentinel = 7.

(* Small non-negative values, so [byte] and [uint16] can hold them; the fractional part exercises
   truncation for the integer precisions and rounding for the narrow floats. *)
let make_values count = Array.init count ~f:(fun k -> Float.of_int (k * 37 % 251) +. 0.25)

let fresh prec ~dims ~pads =
  let padding = Option.map pads ~f:(fun pads -> (pads, sentinel)) in
  let nd = Nd.create_array ~debug:"flat_writer" prec ~dims ~padding in
  (* An unpadded buffer starts uninitialized: give every cell the sentinel, margins included. *)
  Nd.fill_from_float nd sentinel;
  nd

let interior ~dims ~pads =
  match pads with
  | None -> dims
  | Some pads ->
      Array.mapi dims ~f:(fun axis d ->
          if axis < Array.length pads then d - pads.(axis).Ops.left - pads.(axis).right else d)

(* The old per-cell path, spelled independently of the writer's run arithmetic. *)
let per_cell nd ~dims ~pads values =
  let count = Array.length values in
  if Array.is_empty dims then Nd.set_from_float ?padding:pads nd [||] values.(0)
  else
    let inner = interior ~dims ~pads in
    let idx = Array.create ~len:(Array.length dims) 0 in
    let k = ref 0 in
    let rec go axis =
      if axis = Array.length dims then (
        if !k < count then Nd.set_from_float ?padding:pads nd (Array.copy idx) values.(!k);
        Int.incr k)
      else
        for i = 0 to inner.(axis) - 1 do
          idx.(axis) <- i;
          go (axis + 1)
        done
    in
    go 0

let agrees ~label ~dims ~pads ~count prec =
  let values = make_values count in
  let written = fresh prec ~dims ~pads and expected = fresh prec ~dims ~pads in
  Nd.set_flat_values ?padding:pads written values;
  per_cell expected ~dims ~pads values;
  let got = Nd.retrieve_flat_values written and want = Nd.retrieve_flat_values expected in
  let ok = Array.equal Float.equal got want in
  if not ok then
    Stdio.eprintf "%s, %d values: %s disagrees with the per-cell writer\n" label count
      (Ops.prec_string prec);
  ok

let case label ~dims ~pads =
  let numel = Array.fold (interior ~dims ~pads) ~init:1 ~f:( * ) in
  List.iter
    [ ("as many values as cells", numel); ("fewer values", numel / 2); ("more values", numel + 4) ]
    ~f:(fun (count_label, count) ->
      p_all
        (Printf.sprintf "%s, %s: flat writer matches per-cell writes in every precision" label
           count_label)
        Ops.storage_precs
        ~f:(agrees ~label ~dims ~pads ~count))

(* Heap words [set_flat_values] allocates over [numel] interior cells: on stderr, as the cost of a
   host write is a property of this build, not of the golden. Measured on a buffer of 512 runs of 128
   cells (padded) and on one unpadded run of the same size. *)
let allocated_words ~dims ~pads prec =
  let numel = Array.fold (interior ~dims ~pads) ~init:1 ~f:( * ) in
  let values = make_values numel in
  let nd = fresh prec ~dims ~pads in
  let before = Stdlib.Gc.allocated_bytes () in
  Nd.set_flat_values ?padding:pads nd values;
  let words = (Stdlib.Gc.allocated_bytes () -. before) /. Float.of_int (Stdlib.Sys.word_size / 8) in
  Stdio.eprintf "allocation witness %s %s: %.0f words for %d cells (not part of the golden)\n"
    (Ops.prec_string prec)
    (if Option.is_some pads then "padded" else "unpadded")
    words numel;
  words

let witness_padded_dims = [| 514; 130; 1 |]

let witness_pads =
  Ops.[| { left = 1; right = 1 }; { left = 1; right = 1 }; { left = 0; right = 0 } |]

let witness_unpadded_dims = [| 512 * 128 |]

let () =
  Ops.(
    case "uneven margins on all three axes" ~dims:[| 4; 5; 6 |]
      ~pads:(Some [| { left = 1; right = 0 }; { left = 0; right = 2 }; { left = 2; right = 1 } |]);
    case "unpadded innermost axis" ~dims:[| 4; 5; 3 |]
      ~pads:(Some [| { left = 1; right = 0 }; { left = 2; right = 1 }; { left = 0; right = 0 } |]);
    case "padding shorter than the rank" ~dims:[| 3; 4; 2 |]
      ~pads:(Some [| { left = 1; right = 1 } |]);
    case "padded rank 1" ~dims:[| 10 |] ~pads:(Some [| { left = 3; right = 2 } |]);
    case "unpadded rank 3" ~dims:[| 2; 3; 4 |] ~pads:None);
  p_all "rank 0: flat writer matches the per-cell write in every precision" Ops.storage_precs
    ~f:(agrees ~label:"rank 0" ~dims:[||] ~pads:None ~count:1);
  List.iter Ops.storage_precs ~f:(fun prec ->
      let _ : float = allocated_words ~dims:witness_padded_dims ~pads:(Some witness_pads) prec in
      let _ : float = allocated_words ~dims:witness_unpadded_dims ~pads:None prec in
      ())
