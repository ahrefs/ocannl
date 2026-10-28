open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* gh-ocannl-1673: a byte tensor converted to a [uint4x32] counter on the device gets the same four
   words as the host's [Ops.byte_to_uint4x32], for every byte value. Metal's table once took
   [int8_t] (fixed by gh-ocannl-1580), so every byte from 0x80 up was sign-extended into a counter
   no other backend produced -- a random stream keyed by such a byte tensor then differed on Metal
   -- and nothing executed the conversion on a device to notice.

   The bytes are all 256 values, [range 256] stored at byte precision: both sides of 0x80. The byte
   tensor is materialized and read back, so the device's input is checked rather than assumed, and
   the counter is a materialized [cast] of it, so the conversion is the store of the byte operand
   into the [uint4x32] cell, which the generated code must show as a [byte_to_uint4x32] call (not a
   folded constant, not a wider source's conversion). Backend-uniform: the claims hold on every
   backend, and the backend the run used goes to stderr. *)

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let routine = "cast_b2u_counter_fwd"

(* The four 32-bit words of a [uint4x32] cell, stored as a [Complex.t]: the real part holds words 0
   and 1, the imaginary part words 2 and 3, each pair little-endian. *)
let words_of_cell (c : Stdlib.Complex.t) =
  let lo = Int64.bits_of_float c.re and hi = Int64.bits_of_float c.im in
  let word bits k =
    Int64.to_int_exn (Int64.bit_and (Int64.shift_right_logical bits (32 * k)) 0xFFFF_FFFFL)
  in
  [| word lo 0; word lo 1; word hi 0; word hi 1 |]

let hex_words ws = String.concat_array ~sep:" " (Array.map ws ~f:(Printf.sprintf "%08x"))

let () =
  Stdio.eprintf "byte_counter_conversion: backend %s (not part of the golden)\n%!" backend_name;
  let ctx = Context.auto () in
  let bytes = NTDSL.range ~label:[ "b2u_bytes" ] 256 in
  Ir.Tnode.update_prec bytes.value Ir.Ops.byte;
  let counter = NTDSL.cast bytes ~label:[ "b2u_counter" ] () in
  Ir.Tnode.update_prec counter.value Ir.Ops.uint4x32;
  Train.set_materialized bytes.value;
  Train.set_materialized counter.value;
  Generated.arm routine;
  let ctx = Train.forward_once ctx counter in
  Generated.assert_emits ~routine ~contains:"byte_to_uint4x32("
    "the counter is converted from its byte operand by byte_to_uint4x32";
  let stored_bytes =
    match Context.to_host ctx bytes.value with
    | Ir.Ndarray.Byte_nd arr ->
        let n = Bigarray.Genarray.nth_dim arr 0 in
        List.init n ~f:(fun i -> Char.to_int (Bigarray.Genarray.get arr [| i |]))
    | nd -> failwith ("byte tensor read back as " ^ Ir.Ndarray.precision_string nd)
  in
  let cells =
    match Context.to_host ctx counter.value with
    | Ir.Ndarray.Uint4x32_nd arr ->
        let n = Bigarray.Genarray.nth_dim arr 0 in
        List.init n ~f:(fun i -> words_of_cell (Bigarray.Genarray.get arr [| i |]))
    | nd -> failwith ("counter read back as " ^ Ir.Ndarray.precision_string nd)
  in
  Stdio.printf "%d bytes converted\n" (List.length stored_bytes);
  p "the byte tensor holds every byte value once, in order"
    (List.equal Int.equal stored_bytes (List.init 256 ~f:Fn.id));
  p_exists "some probed byte is below 0x80" stored_bytes ~f:(fun b -> b < 0x80);
  p_exists "some probed byte is 0x80 or above" stored_bytes ~f:(fun b -> b >= 0x80);
  let want = Array.of_list_map stored_bytes ~f:Ir.Ops.byte_to_uint4x32 in
  let got = Array.of_list cells in
  (* The first few differing cells, as words, for diagnosis; the claim below carries the verdict. *)
  if Array.length got = Array.length want then
    List.filter
      (List.init (Array.length want) ~f:Fn.id)
      ~f:(fun i -> not (Array.equal Int.equal got.(i) want.(i)))
    |> Fn.flip List.take 4
    |> List.iter ~f:(fun i ->
        Stdio.eprintf "  byte 0x%02x: got %s want %s (not part of the golden)\n"
          (List.nth_exn stored_bytes i)
          (hex_words got.(i))
          (hex_words want.(i)));
  p_all2 "every counter cell equals the host's byte_to_uint4x32 of its byte" got want
    ~f:(Array.equal Int.equal)
