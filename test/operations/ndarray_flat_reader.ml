(* gh-ocannl-1333, the read-side mirror of gh-ocannl-1258: the host readers --
   [Ndarray.retrieve_flat_values] (behind [Context.get_values]), [Ndarray.convert] (gh-ocannl-492's
   load-time precision conversion), [Ndarray.init_array] and [Ndarray.fold_bigarray] -- against
   per-cell oracles, in every storage precision, padded and unpadded. The oracles walk the
   before-padding index space with nested loops and read or write one cell at a time through
   [get_as_float]/[set_from_float], so they share no run arithmetic with the readers. Values are
   compared bit for bit. Then: the readers' heap allocation, and that of the host's native precision
   codecs. *)
open Base
open Verdict.Claims
module Nd = Ir.Ndarray
module Ops = Ir.Ops

let sentinel = 7.

(* Small non-negative values, so [byte] and [uint16] can hold them even after rounding to a narrow
   float ([fp8] rounds 126.25 up to 128, but 250.25 up to 256); the fractional part exercises
   truncation for the integer precisions and rounding for the narrow floats. *)
let make_values count = Array.init count ~f:(fun k -> Float.of_int (k * 37 % 127) +. 0.25)
let numel dims = Array.fold dims ~init:1 ~f:( * )

let interior ~dims ~pads =
  match pads with
  | None -> dims
  | Some pads ->
      Array.mapi dims ~f:(fun axis d ->
          if axis < Array.length pads then d - pads.(axis).Ops.left - pads.(axis).right else d)

let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let same_floats = Array.equal same_bits

(* Calls [f] on every index of [dims] in row-major order, with a fresh copy each time. *)
let for_each_index dims ~f =
  let rank = Array.length dims in
  let idx = Array.create ~len:rank 0 in
  let rec go axis =
    if axis = rank then f (Array.copy idx)
    else
      for i = 0 to dims.(axis) - 1 do
        idx.(axis) <- i;
        go (axis + 1)
      done
  in
  go 0

(* The per-cell reader over the interior: the oracle every reader here answers to. *)
let per_cell_read ?pads nd =
  let out = ref [] in
  for_each_index
    (interior ~dims:(Nd.dims nd) ~pads)
    ~f:(fun idx -> out := Nd.get_as_float ?padding:pads nd idx :: !out);
  Array.of_list_rev !out

(* Every cell of the raw buffer, margins included. *)
let raw_cells nd = per_cell_read nd

(* A buffer whose margins hold [sentinel] and whose interior holds [make_values]. *)
let filled prec ~dims ~pads =
  let padding = Option.map pads ~f:(fun pads -> (pads, sentinel)) in
  let nd = Nd.create_array ~debug:"flat_reader" prec ~dims ~padding in
  Nd.fill_from_float nd sentinel;
  let inner = interior ~dims ~pads in
  let values = make_values (numel inner) in
  let k = ref 0 in
  for_each_index inner ~f:(fun idx ->
      Nd.set_from_float ?padding:pads nd idx values.(!k);
      Int.incr k);
  nd

let reads_agree ~label ~dims ~pads prec =
  let nd = filled prec ~dims ~pads in
  let ok = same_floats (Nd.retrieve_flat_values ?padding:pads nd) (per_cell_read ?pads nd) in
  if not ok then
    Stdio.eprintf "%s: %s retrieve_flat_values disagrees with the per-cell reader\n" label
      (Ops.prec_string prec);
  ok

(* [fold_as_float] (over [fold_bigarray]) visits the interior cells in row-major order, with their
   before-padding indices. *)
let fold_agrees ~label ~dims ~pads prec =
  let nd = filled prec ~dims ~pads in
  let visited =
    Nd.fold_as_float ?padding:pads nd ~init:[] ~f:(fun acc idx v -> (Array.copy idx, v) :: acc)
    |> List.rev
  in
  let expected = ref [] in
  for_each_index (interior ~dims ~pads) ~f:(fun idx ->
      expected := (idx, Nd.get_as_float ?padding:pads nd idx) :: !expected);
  let ok =
    List.equal
      (fun (i1, v1) (i2, v2) -> Array.equal Int.equal i1 i2 && same_bits v1 v2)
      visited (List.rev !expected)
  in
  if not ok then
    Stdio.eprintf "%s: %s fold_as_float disagrees with the per-cell walk\n" label
      (Ops.prec_string prec);
  ok

(* [init_array] against per-cell writes of the same function of the before-padding index, raw
   buffers compared margins included. The function varies with every axis. *)
let init_value idx =
  Float.of_int (Array.foldi idx ~init:0 ~f:(fun axis acc i -> (acc * 7) + i + axis) % 127) +. 0.25

let init_agrees ~label ~dims ~pads prec =
  let padding = Option.map pads ~f:(fun pads -> (pads, sentinel)) in
  let got = Nd.init_array ~debug:"flat_reader" prec ~dims ~padding ~f:init_value in
  let want = Nd.create_array ~debug:"flat_reader" prec ~dims ~padding in
  Nd.fill_from_float want sentinel;
  for_each_index (interior ~dims ~pads) ~f:(fun idx ->
      Nd.set_from_float ?padding:pads want idx (init_value idx));
  let ok = same_floats (raw_cells got) (raw_cells want) in
  if not ok then
    Stdio.eprintf "%s: %s init_array disagrees with per-cell writes\n" label (Ops.prec_string prec);
  ok

(* [convert] against the per-cell conversion it replaced: [get_as_float] then [set_from_float] at
   every raw index, or [src] itself when the precision already matches. An exception must be the
   same one. *)
let per_cell_convert prec src =
  if Ops.equal_prec prec (Nd.get_prec src) then src
  else
    let dims = Nd.dims src in
    let dst = Nd.create_array ~debug:"flat_reader" prec ~dims ~padding:None in
    for_each_index dims ~f:(fun idx -> Nd.set_from_float dst idx (Nd.get_as_float src idx));
    dst

let outcome f =
  match f () with nd -> Ok (raw_cells nd) | exception exn -> Error (Exn.to_string exn)

let same_outcome a b =
  match (a, b) with
  | Ok a, Ok b -> same_floats a b
  | Error a, Error b -> String.equal a b
  | _ -> false

let convert_agrees ~label src dst_prec =
  let ok =
    same_outcome
      (outcome (fun () -> Nd.convert dst_prec src))
      (outcome (fun () -> per_cell_convert dst_prec src))
  in
  if not ok then
    Stdio.eprintf "%s: convert %s -> %s disagrees with per-cell conversion\n" label
      (Nd.precision_string src) (Ops.prec_string dst_prec);
  ok

let case label ~dims ~pads =
  p_all
    (Printf.sprintf "%s: flat reader matches per-cell reads in every precision" label)
    Ops.storage_precs ~f:(reads_agree ~label ~dims ~pads);
  p_all
    (Printf.sprintf "%s: fold visits the interior in row-major order in every precision" label)
    Ops.storage_precs ~f:(fold_agrees ~label ~dims ~pads);
  p_all
    (Printf.sprintf "%s: init_array matches per-cell writes in every precision" label)
    Ops.storage_precs ~f:(init_agrees ~label ~dims ~pads);
  (* [convert] carries no padding: it converts the raw buffer, margins included. *)
  p_all (Printf.sprintf "%s: convert matches per-cell conversion for every precision pair" label)
    (List.cartesian_product Ops.storage_precs Ops.storage_precs) ~f:(fun (src_prec, dst_prec) ->
      convert_agrees ~label (filled src_prec ~dims ~pads) dst_prec)

(* Raw codes the readers must widen exactly as the per-cell reader does: every code of the 8- and
   16-bit formats (NaN, infinity and subnormal encodings among them), the extremes of the wide
   integers -- the unsigned ones with the sign bit set, which a signed read would turn negative --
   and the float specials. Written straight into the bigarray, bypassing any host conversion. *)
let code_buffers () =
  let module A = Bigarray.Genarray in
  let make kind n set =
    let arr = A.create kind Bigarray.c_layout [| n |] in
    for i = 0 to n - 1 do
      A.set arr [| i |] (set i)
    done;
    arr
  in
  let int32s = [| Int32.min_value; -1l; 0l; 1l; Int32.max_value; 0x8000_0001l |] in
  let int64s = [| Int64.min_value; -1L; 0L; 1L; Int64.max_value; 0x8000_0000_0000_0401L |] in
  let floats =
    [|
      0.;
      -0.;
      1.5;
      -2.25;
      Float.nan;
      Float.infinity;
      Float.neg_infinity;
      1e-40;
      3e38;
      65504.;
      6e-8;
      1e300;
    |]
  in
  Nd.
    [
      Byte_nd (make Bigarray.Char 256 Char.of_int_exn);
      Fp8_nd (make Bigarray.Char 256 Char.of_int_exn);
      Uint16_nd (make Bigarray.Int16_unsigned 65536 Fn.id);
      Bfloat16_nd (make Bigarray.Int16_unsigned 65536 Fn.id);
      Int32_nd (make Bigarray.Int32 (Array.length int32s) (Array.get int32s));
      Uint32_nd (make Bigarray.Int32 (Array.length int32s) (Array.get int32s));
      Int64_nd (make Bigarray.Int64 (Array.length int64s) (Array.get int64s));
      Uint64_nd (make Bigarray.Int64 (Array.length int64s) (Array.get int64s));
      Half_nd (make Bigarray.Float16 (Array.length floats) (Array.get floats));
      Single_nd (make Bigarray.Float32 (Array.length floats) (Array.get floats));
      Double_nd (make Bigarray.Float64 (Array.length floats) (Array.get floats));
      Uint4x32_nd
        (make Bigarray.Complex64 (Array.length floats) (fun i ->
             { Stdlib.Complex.re = floats.(i); im = 1. }));
    ]

(* Values the writers reject (NaN, out of range) or wrap, converted from double into every
   precision: the same exception, or the same stored cells. *)
let edge_values = [| Float.nan; 300.; -1.; 3e9; -3e9; 1e19; -1e19; 1e30; -1e30; Float.infinity |]

(* Heap words an operation allocates, on stderr: the cost of a host read is a property of this
   build, not of the golden. *)
let word_bytes = Float.of_int (Stdlib.Sys.word_size / 8)

let measure what f =
  let before = Stdlib.Gc.allocated_bytes () in
  f ();
  let words = (Stdlib.Gc.allocated_bytes () -. before) /. word_bytes in
  Stdio.eprintf "allocation witness %s: %.0f words (not part of the golden)\n" what words;
  words

(* 512 runs of 128 cells (padded), and one unpadded run of the same 65536 cells. *)
let witness_padded_dims = [| 514; 130; 1 |]

let witness_pads =
  Ops.[| { left = 1; right = 1 }; { left = 1; right = 1 }; { left = 0; right = 0 } |]

let witness_unpadded_dims = [| 512 * 128 |]
let witness_cells = 512 * 128

let witness_shapes =
  [ ("padded", witness_padded_dims, Some witness_pads); ("unpadded", witness_unpadded_dims, None) ]

let read_words prec =
  List.map witness_shapes ~f:(fun (shape, dims, pads) ->
      let nd = filled prec ~dims ~pads in
      measure
        (Printf.sprintf "retrieve_flat_values %s %s" (Ops.prec_string prec) shape)
        (fun () -> ignore (Nd.retrieve_flat_values ?padding:pads nd : float array)))

let init_words prec =
  List.map witness_shapes ~f:(fun (shape, dims, pads) ->
      let padding = Option.map pads ~f:(fun pads -> (pads, sentinel)) in
      (* A callback returning a constant allocates nothing, so what is left is [init_array]'s. *)
      measure
        (Printf.sprintf "init_array %s %s" (Ops.prec_string prec) shape)
        (fun () -> ignore (Nd.init_array ~debug:"witness" prec ~dims ~padding ~f:(fun _ -> 0.5))))

let convert_words (src_prec, dst_prec) =
  let src = filled src_prec ~dims:witness_unpadded_dims ~pads:None in
  measure
    (Printf.sprintf "convert %s -> %s" (Ops.prec_string src_prec) (Ops.prec_string dst_prec))
    (fun () -> ignore (Nd.convert dst_prec src : Nd.t))

let is_packed prec = match Ops.prec_family prec with Packed_integer -> true | _ -> false

let () =
  Ops.(
    case "uneven margins on all three axes" ~dims:[| 4; 5; 6 |]
      ~pads:(Some [| { left = 1; right = 0 }; { left = 0; right = 2 }; { left = 2; right = 1 } |]);
    case "unpadded innermost axis" ~dims:[| 4; 5; 3 |]
      ~pads:(Some [| { left = 1; right = 0 }; { left = 2; right = 1 }; { left = 0; right = 0 } |]);
    case "padding shorter than the rank" ~dims:[| 3; 4; 2 |]
      ~pads:(Some [| { left = 1; right = 1 } |]);
    case "padded rank 1" ~dims:[| 10 |] ~pads:(Some [| { left = 3; right = 2 } |]);
    case "margins covering an axis" ~dims:[| 3; 2; 4 |]
      ~pads:(Some [| { left = 0; right = 0 }; { left = 1; right = 1 } |]);
    case "unpadded rank 3" ~dims:[| 2; 3; 4 |] ~pads:None;
    case "rank 0" ~dims:[||] ~pads:None);
  let codes = code_buffers () in
  p_all "raw codes: flat reader widens every code as the per-cell reader does, in every precision"
    codes ~f:(fun nd ->
      let ok = same_floats (Nd.retrieve_flat_values nd) (per_cell_read nd) in
      if not ok then Stdio.eprintf "raw codes: %s reader disagrees\n" (Nd.precision_string nd);
      ok);
  p_all "raw codes: convert matches per-cell conversion for every precision pair"
    (List.cartesian_product codes Ops.storage_precs) ~f:(fun (src, dst_prec) ->
      convert_agrees ~label:"raw codes" src dst_prec);
  let edges = Nd.create_array ~debug:"flat_reader" Ops.double ~dims:[| 1 |] ~padding:None in
  p_all
    "edge values: convert from double raises or stores as per-cell conversion, in every precision"
    (List.cartesian_product Ops.storage_precs (Array.to_list edge_values))
    ~f:(fun (dst_prec, v) ->
      Nd.set_from_float edges [| 0 |] v;
      convert_agrees ~label:(Printf.sprintf "edge value %h" v) edges dst_prec);
  (* The allocation bounds. Before gh-ocannl-1333 every one of these took heap words per cell. *)
  let packed_precs, scalar_precs = List.partition_tf Ops.storage_precs ~f:is_packed in
  let result_words = Float.of_int (witness_cells + 1) in
  p_all
    "a 65536-cell flat read allocates its float array and under 512 heap words more, padded or \
     not, in every scalar precision" (List.concat_map scalar_precs ~f:read_words) ~f:(fun words ->
      Float.(words < result_words + 512.));
  (* A [uint4x32] cell is an OCaml [Complex.t] record: the read takes its real part, and the
     compiler does not elide the record, so the bound is the record per cell. *)
  p_all "a packed uint4x32 flat read allocates no more than its 3-word record per cell"
    (List.concat_map packed_precs ~f:read_words) ~f:(fun words ->
      Float.(words < result_words + 512. + (3. * of_int witness_cells)));
  (* [init_array] and [convert] carry their values through a float buffer of [Nd.host_chunk_cells]
     cells, far fewer than the witness's. *)
  let chunk_words = Float.of_int (Nd.host_chunk_cells + 1) in
  p "the chunk buffer is far smaller than the witness" (Nd.host_chunk_cells * 16 <= witness_cells);
  p_all
    "a 65536-cell init_array with a non-allocating callback allocates its chunk buffer and under \
     512 heap words more, padded or not, in every scalar precision"
    (List.concat_map scalar_precs ~f:init_words) ~f:(fun words ->
      Float.(words < chunk_words + 512.));
  let pairs =
    List.cartesian_product Ops.storage_precs Ops.storage_precs
    |> List.filter ~f:(fun (a, b) -> not (Ops.equal_prec a b))
  in
  let packed_pairs, scalar_pairs =
    List.partition_tf pairs ~f:(fun (a, b) -> is_packed a || is_packed b)
  in
  p_all
    "a 65536-cell convert allocates its chunk buffer and under 512 heap words more between every \
     two scalar precisions" (List.map scalar_pairs ~f:convert_words) ~f:(fun words ->
      Float.(words < chunk_words + 512.));
  p_all "a 65536-cell convert to or from uint4x32 allocates no more than its 3-word record per cell"
    (List.map packed_pairs ~f:convert_words) ~f:(fun words ->
      Float.(words < chunk_words + 512. + (3. * of_int witness_cells)));
  (* The padded fold no longer rebuilds the adjusted index per cell; a byte read and an int
     accumulator box nothing, so what is left is the fold's own. *)
  let fold_nd = filled Ops.byte ~dims:witness_padded_dims ~pads:(Some witness_pads) in
  let fold_words =
    measure "fold_bigarray byte padded" (fun () ->
        match fold_nd with
        | Nd.Byte_nd arr ->
            ignore
              (Nd.fold_bigarray ~padding:witness_pads arr ~init:0 ~f:(fun acc _ c ->
                   acc + Char.to_int c)
                : int)
        | _ -> assert false)
  in
  p "a 65536-cell padded fold allocates under 512 heap words" Float.(fold_words < 512.);
  (* The native codecs, called per cell into preallocated arrays. *)
  let floats = Stdlib.Array.create_float witness_cells
  and ints = Array.create ~len:witness_cells 0 in
  let codec_words =
    [
      measure "bfloat16_to_single" (fun () ->
          for i = 0 to witness_cells - 1 do
            floats.(i) <- Ops.bfloat16_to_single i
          done);
      measure "fp8_to_single" (fun () ->
          for i = 0 to witness_cells - 1 do
            floats.(i) <- Ops.fp8_to_single (i land 255)
          done);
      measure "single_to_fp8" (fun () ->
          for i = 0 to witness_cells - 1 do
            ints.(i) <- Ops.single_to_fp8 (Float.of_int i)
          done);
    ]
  in
  p_all
    "65536 calls of each of bfloat16_to_single, fp8_to_single and single_to_fp8 allocate under 512 \
     heap words"
    codec_words ~f:(fun words -> Float.(words < 512.))
