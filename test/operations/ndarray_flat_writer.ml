(* gh-ocannl-1258: [Ndarray.set_flat_values] -- the host writer behind [Context.set_values] and
   multi-value host initialization -- against the per-cell [Ndarray.set_from_float] it must agree
   with, in every storage precision, padded and unpadded, with and without [~cycle]. The per-cell
   oracle walks the before-padding index space with nested loops and lets [set_from_float] apply the
   margins, so it shares no run arithmetic with the writer. Margins and cells past the end of the
   values keep a sentinel both buffers were filled with; the raw buffers, margins included, must be
   equal. Then: values the conversions reject or wrap, and the writer's heap allocation. *)
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
let per_cell ~cycle nd ~dims ~pads values =
  let count = Array.length values in
  if Array.is_empty dims then Nd.set_from_float ?padding:pads nd [||] values.(0)
  else
    let inner = interior ~dims ~pads in
    let idx = Array.create ~len:(Array.length dims) 0 in
    let k = ref 0 in
    let rec go axis =
      if axis = Array.length dims then (
        if cycle then Nd.set_from_float ?padding:pads nd (Array.copy idx) values.(!k % count)
        else if !k < count then Nd.set_from_float ?padding:pads nd (Array.copy idx) values.(!k);
        Int.incr k)
      else
        for i = 0 to inner.(axis) - 1 do
          idx.(axis) <- i;
          go (axis + 1)
        done
    in
    go 0

let agrees ?(cycle = false) ~label ~dims ~pads ~count prec =
  let values = make_values count in
  let written = fresh prec ~dims ~pads and expected = fresh prec ~dims ~pads in
  Nd.set_flat_values ?padding:pads ~cycle written values;
  per_cell ~cycle expected ~dims ~pads values;
  let got = Nd.retrieve_flat_values written and want = Nd.retrieve_flat_values expected in
  let ok = Array.equal Float.equal got want in
  if not ok then
    Stdio.eprintf "%s, %d values: %s disagrees with the per-cell writer\n" label count
      (Ops.prec_string prec);
  ok

(* The runs both host writers walk ([Ndarray.iter_interior_runs]): non-empty, in increasing order,
   never adjacent (adjacent runs would be one maximal run, so a run count proportional to the cells
   would show here), and covering exactly the linear indices inside the margins -- decomposed
   independently, innermost axis first. *)
let check_runs label ~dims ~pads =
  let runs = ref [] in
  Nd.iter_interior_runs ?padding:pads dims ~f:(fun lo hi -> runs := (lo, hi) :: !runs);
  let runs = List.rev !runs in
  let numel = Array.fold dims ~init:1 ~f:( * ) in
  let axes =
    Array.mapi dims ~f:(fun axis d ->
        match pads with
        | Some pads when axis < Array.length pads -> (d, pads.(axis))
        | _ -> (d, Ops.{ left = 0; right = 0 }))
  in
  let inside k =
    snd
      (Array.fold_right axes ~init:(k, true) ~f:(fun (d, Ops.{ left; right }) (rest, inside) ->
           let i = rest % d in
           (rest / d, inside && i >= left && i < d - right)))
  in
  let covered = Array.create ~len:numel false in
  let in_bounds = ref true in
  List.iter runs ~f:(fun (lo, hi) ->
      if not (0 <= lo && lo < hi && hi <= numel) then in_bounds := false
      else
        for j = lo to hi - 1 do
          covered.(j) <- true
        done);
  let rec separated = function
    | (_, hi) :: ((lo, _) :: _ as rest) -> hi < lo && separated rest
    | _ -> true
  in
  pf "%s: interior runs are non-empty, in bounds, increasing and never adjacent" label
    ((not (List.is_empty runs)) && !in_bounds && separated runs);
  p_alli (Printf.sprintf "%s: interior runs cover exactly the cells inside the margins" label)
    (Array.to_list covered) ~f:(fun k c -> Bool.equal c (inside k))

let case label ~dims ~pads =
  check_runs label ~dims ~pads;
  let numel = Array.fold (interior ~dims ~pads) ~init:1 ~f:( * ) in
  (* [numel / 2 + 1] values divide no interior here, so cycling wraps in the middle of a run. *)
  List.iter
    [
      ("as many values as cells", false, numel);
      ("fewer values", false, numel / 2);
      ("more values", false, numel + 4);
      ("cycling fewer values", true, (numel / 2) + 1);
      ("cycling more values", true, numel + 4);
    ]
    ~f:(fun (count_label, cycle, count) ->
      p_all
        (Printf.sprintf "%s, %s: flat writer matches per-cell writes in every precision" label
           count_label)
        Ops.storage_precs
        ~f:(agrees ~cycle ~label ~dims ~pads ~count))

(* Values a conversion rejects (NaN, out of range: Base's checked [of_float]s, [Char.of_int_exn]) or
   wraps (the unsigned and narrow ones): each written alone, through both writers, must end the same
   way -- the same exception, or the same stored cell. *)
let edge_values = [| Float.nan; 300.; -1.; 3e9; -3e9; 1e19; -1e19; 1e30; -1e30; Float.infinity |]

let same_outcome prec v =
  let outcome write =
    let nd = fresh prec ~dims:[| 1 |] ~pads:None in
    match write nd with
    | () -> Ok (Nd.retrieve_flat_values nd).(0)
    | exception exn -> Error (Exn.to_string exn)
  in
  let flat = outcome (fun nd -> Nd.set_flat_values nd [| v |]) in
  let cell = outcome (fun nd -> Nd.set_from_float nd [| 0 |] v) in
  let ok =
    match (flat, cell) with
    | Ok a, Ok b -> Float.(equal a b || (is_nan a && is_nan b))
    | Error a, Error b -> String.equal a b
    | _ -> false
  in
  if not ok then
    Stdio.eprintf "edge value %h: %s outcomes differ between the writers\n" v (Ops.prec_string prec);
  ok

(* Heap words [set_flat_values] allocates over [numel] interior cells: on stderr, as the cost of a
   host write is a property of this build, not of the golden. Measured on a buffer of 512 runs of
   128 cells (padded) and on one unpadded run of the same size. *)
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
  p_all "edge values: flat writer raises or stores as the per-cell writer does, in every precision"
    (List.cartesian_product Ops.storage_precs (Array.to_list edge_values))
    ~f:(fun (prec, v) -> same_outcome prec v);
  (* Both witness shapes, for every precision. *)
  let writes precs =
    List.concat_map precs ~f:(fun prec ->
        [
          allocated_words ~dims:witness_padded_dims ~pads:(Some witness_pads) prec;
          allocated_words ~dims:witness_unpadded_dims ~pads:None prec;
        ])
  in
  let packed_precs, scalar_precs =
    List.partition_tf Ops.storage_precs ~f:(fun prec ->
        match Ops.prec_family prec with Packed_integer -> true | _ -> false)
  in
  p_all
    "a 65536-cell flat write allocates under 512 heap words, padded or not, in every scalar \
     precision"
    (writes scalar_precs) ~f:(fun words -> Float.(words < 512.));
  (* A [uint4x32] cell is an OCaml [Complex.t] record, which the store reads its two halves from:
     the compiler does not elide that 3-word record, so the bound is the record per cell. *)
  p_all "a packed uint4x32 flat write allocates no more than its 3-word record per cell"
    (writes packed_precs) ~f:(fun words -> Float.(words < 512. + (3. * 65536.)))
