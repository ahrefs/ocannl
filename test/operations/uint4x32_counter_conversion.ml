open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* gh-ocannl-1704 (from gh-ocannl-1673): a tensor of every source precision converted to a
   [uint4x32] counter on the device gets the same four words as the host's [Ops.<prec>_to_uint4x32]
   stub, for every value of the source sample. Metal's table once took [int8_t] for
   [byte_to_uint4x32] (fixed by gh-ocannl-1580), so every byte from 0x80 up was sign-extended into a
   counter no other backend produced -- a random stream keyed by such a tensor then differed on
   Metal -- and nothing executed any of the conversions on a device to notice.

   One leg per source precision. Each leg uploads its sample as the host initialization of a
   materialized source tensor, at the source precision, and reads it back, so the device's input is
   checked rather than assumed. The counter is a materialized [cast] of it, so the conversion is the
   store of the source operand into the [uint4x32] cell, which the generated code must show as a
   [<prec>_to_uint4x32] call (not a folded constant, not another source's conversion). The source
   values are handled as bit patterns throughout: every stub's argument is the stored bits of its
   value ([half]/[bfloat16]/[fp8] pass their storage bits; [single]/[double] the float itself, whose
   stub copies its bits).

   The samples. The 8- and 16-bit sources are exhaustive: [byte], [fp8], [uint16] and [bfloat16]
   every bit pattern; [half] every non-NaN pattern (63490 of 65536), as OCaml's [float16] bigarray
   is written from a [float] and cannot carry a NaN payload through. The 32- and 64-bit sources,
   where 2^32 is infeasible, take a structured sample: zero, all ones, every single bit [1 << k] and
   its complement, every low mask [(1 << k) - 1], and 1024 splitmix64 values (the float sources keep
   only the non-NaN ones, plus the named float boundaries). It discriminates because every
   conversion here is a function of its source's bits -- a copy, an XOR with a constant, a 16-bit
   rotation, a split into two words. A per-position defect (truncation through a narrower type, sign
   extension, a swapped or misplaced word, a wrong rotation) changes the result for a value with one
   bit set or one bit clear at that position, and the singles and their complements put every
   position in both states alone. A value-dependent defect (a numeric conversion in place of a bit
   reinterpretation, a flushed subnormal, a canonicalized sign) changes the dense random values and
   the masks, which such a conversion moves off their bits. Each leg claims both coverage
   conditions: every bit position is set in some sampled value and clear in another.

   [double] has no Metal storage ([Metal_backend]'s [typ_of_prec] refuses it), so that leg is a
   backend skip there. Backend-uniform otherwise: the claims hold on every backend, and the backend
   the run used goes to stderr. *)

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* A source precision's leg. Values are the source's bit patterns, as unsigned [Int64.t]s of [width]
   bits. *)
type leg = {
  prec : Ir.Ops.prec;
  width : int;
  sample : string;  (** How the bit patterns were chosen, for the golden. *)
  patterns : Int64.t list;
  upload : Int64.t array -> Ir.Ndarray.t;  (** A fresh host array holding the patterns. *)
  read_back : Ir.Ndarray.t -> Int64.t array option;
      (** The patterns a read-back array holds, or [None] for an array of another precision. *)
  host : Int64.t -> int array;  (** The host stub, applied to a pattern. *)
  leg : Verdict.leg;
}

let bigarray kind n = Bigarray.Genarray.create kind Bigarray.c_layout [| n |]

let fill kind prec ~f patterns =
  let arr = bigarray kind (Array.length patterns) in
  Array.iteri patterns ~f:(fun i bits -> Bigarray.Genarray.set arr [| i |] (f bits));
  Ir.Ndarray.as_array prec arr

let read arr ~f =
  Array.init (Bigarray.Genarray.nth_dim arr 0) ~f:(fun i -> f (Bigarray.Genarray.get arr [| i |]))

(* [width] < 64 masks; a shift by 64 is not a shift in OCaml (it is taken modulo 64). *)
let low_bits width bits =
  if width >= 64 then bits else Int64.bit_and bits (Int64.pred (Int64.shift_left 1L width))

let to_int = Int64.to_int_exn
let all_patterns width = List.init (1 lsl width) ~f:Int64.of_int

(* splitmix64, so the dense sample is the same on every platform and run. *)
let splitmix64 n =
  let state = ref 0x0123_4567_89AB_CDEFL in
  List.init n ~f:(fun _ ->
      state := Int64.( + ) !state 0x9E37_79B9_7F4A_7C15L;
      let z = !state in
      let z =
        Int64.( * ) (Int64.bit_xor z (Int64.shift_right_logical z 30)) 0xBF58_476D_1CE4_E5B9L
      in
      let z =
        Int64.( * ) (Int64.bit_xor z (Int64.shift_right_logical z 27)) 0x94D0_49BB_1331_11EBL
      in
      Int64.bit_xor z (Int64.shift_right_logical z 31))

let structured width =
  let bit k = Int64.shift_left 1L k in
  let singles = List.init width ~f:bit in
  List.concat
    [
      [ 0L; Int64.bit_not 0L ];
      singles;
      List.map singles ~f:Int64.bit_not;
      List.init width ~f:(fun k -> Int64.pred (bit k));
      splitmix64 1024;
    ]
  |> List.map ~f:(low_bits width)
  |> List.dedup_and_sort ~compare:Int64.compare

let is_nan_bits ~exp_bits ~mant_bits bits =
  let exp =
    Int64.bit_and
      (Int64.shift_right_logical bits mant_bits)
      (Int64.pred (Int64.shift_left 1L exp_bits))
  in
  Int64.equal exp (Int64.pred (Int64.shift_left 1L exp_bits))
  && not (Int64.equal (low_bits mant_bits bits) 0L)

let open_leg = Verdict.leg ~when_:true ~on:backend_name ()

let legs =
  let structured_desc =
    "zero, all ones, single bits, their complements, low masks, 1024 splitmix64"
  in
  [
    {
      prec = Ir.Ops.byte;
      width = 8;
      sample = "every bit pattern";
      patterns = all_patterns 8;
      upload = fill Bigarray.Char Ir.Ops.Byte ~f:(fun b -> Char.of_int_exn (to_int b));
      read_back =
        (function
        | Ir.Ndarray.Byte_nd a -> Some (read a ~f:(fun c -> Int64.of_int (Char.to_int c)))
        | _ -> None);
      host = (fun b -> Ir.Ops.byte_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.fp8;
      width = 8;
      sample = "every bit pattern";
      patterns = all_patterns 8;
      upload = fill Bigarray.Char Ir.Ops.Fp8 ~f:(fun b -> Char.of_int_exn (to_int b));
      read_back =
        (function
        | Ir.Ndarray.Fp8_nd a -> Some (read a ~f:(fun c -> Int64.of_int (Char.to_int c)))
        | _ -> None);
      host = (fun b -> Ir.Ops.fp8_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.uint16;
      width = 16;
      sample = "every bit pattern";
      patterns = all_patterns 16;
      upload = fill Bigarray.Int16_unsigned Ir.Ops.Uint16 ~f:to_int;
      read_back = (function Ir.Ndarray.Uint16_nd a -> Some (read a ~f:Int64.of_int) | _ -> None);
      host = (fun b -> Ir.Ops.uint16_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.bfloat16;
      width = 16;
      sample = "every bit pattern";
      patterns = all_patterns 16;
      upload = fill Bigarray.Int16_unsigned Ir.Ops.Bfloat16 ~f:to_int;
      read_back = (function Ir.Ndarray.Bfloat16_nd a -> Some (read a ~f:Int64.of_int) | _ -> None);
      host = (fun b -> Ir.Ops.bfloat16_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.half;
      width = 16;
      sample = "every non-NaN bit pattern";
      patterns = List.filter (all_patterns 16) ~f:(Fn.non (is_nan_bits ~exp_bits:5 ~mant_bits:10));
      upload = fill Bigarray.Float16 Ir.Ops.Half ~f:(fun b -> Ir.Ops.half_to_single (to_int b));
      read_back =
        (function
        | Ir.Ndarray.Half_nd a -> Some (read a ~f:(fun v -> Int64.of_int (Ir.Ops.single_to_half v)))
        | _ -> None);
      host = (fun b -> Ir.Ops.half_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.int32;
      width = 32;
      sample = structured_desc;
      patterns = structured 32;
      upload = fill Bigarray.Int32 Ir.Ops.Int32 ~f:Int64.to_int32_trunc;
      read_back =
        (function
        | Ir.Ndarray.Int32_nd a -> Some (read a ~f:(fun v -> low_bits 32 (Int64.of_int32 v)))
        | _ -> None);
      host = (fun b -> Ir.Ops.int32_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.uint32;
      width = 32;
      sample = structured_desc;
      patterns = structured 32;
      upload = fill Bigarray.Int32 Ir.Ops.Uint32 ~f:Int64.to_int32_trunc;
      read_back =
        (function
        | Ir.Ndarray.Uint32_nd a -> Some (read a ~f:(fun v -> low_bits 32 (Int64.of_int32 v)))
        | _ -> None);
      host = (fun b -> Ir.Ops.uint32_to_uint4x32 (to_int b));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.int64;
      width = 64;
      sample = structured_desc;
      patterns = structured 64;
      upload = fill Bigarray.Int64 Ir.Ops.Int64 ~f:Fn.id;
      read_back = (function Ir.Ndarray.Int64_nd a -> Some (read a ~f:Fn.id) | _ -> None);
      host = Ir.Ops.int64_to_uint4x32;
      leg = open_leg;
    };
    {
      prec = Ir.Ops.uint64;
      width = 64;
      sample = structured_desc;
      patterns = structured 64;
      upload = fill Bigarray.Int64 Ir.Ops.Uint64 ~f:Fn.id;
      read_back = (function Ir.Ndarray.Uint64_nd a -> Some (read a ~f:Fn.id) | _ -> None);
      host = Ir.Ops.uint64_to_uint4x32;
      leg = open_leg;
    };
    {
      prec = Ir.Ops.single;
      width = 32;
      sample =
        "the structured sample's non-NaN patterns, infinities, extreme normals and subnormals";
      patterns =
        List.filter
          (structured 32
          @ [ 0x7F80_0000L; 0xFF80_0000L; 0x7F7F_FFFFL; 0x0080_0000L; 0x8000_0001L; 0x3F80_0000L ])
          ~f:(Fn.non (is_nan_bits ~exp_bits:8 ~mant_bits:23))
        |> List.dedup_and_sort ~compare:Int64.compare;
      upload =
        fill Bigarray.Float32 Ir.Ops.Single ~f:(fun b ->
            Int32.float_of_bits (Int64.to_int32_trunc b));
      read_back =
        (function
        | Ir.Ndarray.Single_nd a ->
            Some (read a ~f:(fun v -> low_bits 32 (Int64.of_int32 (Int32.bits_of_float v))))
        | _ -> None);
      host = (fun b -> Ir.Ops.single_to_uint4x32 (Int32.float_of_bits (Int64.to_int32_trunc b)));
      leg = open_leg;
    };
    {
      prec = Ir.Ops.double;
      width = 64;
      sample =
        "the structured sample's non-NaN patterns, infinities, extreme normals and subnormals";
      patterns =
        List.filter
          (structured 64
          @ [
              0x7FF0_0000_0000_0000L;
              0xFFF0_0000_0000_0000L;
              0x7FEF_FFFF_FFFF_FFFFL;
              0x0010_0000_0000_0000L;
              0x8000_0000_0000_0001L;
              0x3FF0_0000_0000_0000L;
            ])
          ~f:(Fn.non (is_nan_bits ~exp_bits:11 ~mant_bits:52))
        |> List.dedup_and_sort ~compare:Int64.compare;
      upload = fill Bigarray.Float64 Ir.Ops.Double ~f:Int64.float_of_bits;
      read_back =
        (function Ir.Ndarray.Double_nd a -> Some (read a ~f:Int64.bits_of_float) | _ -> None);
      host = (fun b -> Ir.Ops.double_to_uint4x32 (Int64.float_of_bits b));
      leg = Verdict.leg ~when_:(not (String.equal backend_name "metal")) ~on:backend_name ();
    };
  ]

let ctx = ref (Context.auto ())

(* Compiles and runs one leg's conversion, returning the source patterns the device holds and the
   counter cells' words. Forced only inside an open leg. *)
let convert ~name ~prec ~upload ~read_back patterns =
  let n = Array.length patterns in
  let label = name ^ "_counter" in
  let routine = "cast_" ^ label ^ "_fwd" in
  let source =
    Tensor.term ~grad_spec:Prohibit_grad
      ~init_data:(Ir.Assignments.Keep_shape_no_padding (upload patterns))
      ~label:[ name ^ "_source" ]
      ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()
  in
  Ir.Tnode.update_prec source.value prec;
  let counter = NTDSL.cast source ~label:[ label ] () in
  Ir.Tnode.update_prec counter.value Ir.Ops.uint4x32;
  Train.set_materialized source.value;
  Train.set_materialized counter.value;
  Generated.arm routine;
  ctx := Train.forward_once !ctx counter;
  let stored =
    let nd = Context.to_host !ctx source.value in
    match read_back nd with
    | Some patterns -> patterns
    | None -> failwith (name ^ " source read back as " ^ Ir.Ndarray.precision_string nd)
  in
  let cells =
    match Context.to_host !ctx counter.value with
    | Ir.Ndarray.Uint4x32_nd arr -> read arr ~f:Test_utils.uint4x32_words
    | nd -> failwith (name ^ " counter read back as " ^ Ir.Ndarray.precision_string nd)
  in
  (routine, stored, cells)

let run_leg { prec; width; sample; patterns; upload; read_back; host; leg } =
  let name = Ir.Ops.prec_string prec in
  let patterns = Array.of_list patterns in
  let claim fmt = Printf.ksprintf (fun s -> name ^ ": " ^ s) fmt in
  Stdio.printf "%s: %d source values, %s\n" name (Array.length patterns) sample;
  (* Coverage of the sample itself: holds whatever the backend. *)
  p_all (claim "every bit position is both set and clear in some sampled value")
    (List.init width ~f:Fn.id) ~f:(fun k ->
      let set bits = not (Int64.equal (Int64.bit_and bits (Int64.shift_left 1L k)) 0L) in
      Array.exists patterns ~f:set && Array.exists patterns ~f:(Fn.non set));
  let result = lazy (convert ~name ~prec ~upload ~read_back patterns) in
  Verdict.in_leg ~leg (claim "the counter is converted by %s_to_uint4x32 from its source" name)
    (fun label ->
      let routine, _, _ = Lazy.force result in
      Generated.assert_emits ~routine ~contains:(name ^ "_to_uint4x32(") label);
  Verdict.in_leg ~leg (claim "the device holds the sampled bit patterns, in order") (fun label ->
      let _, stored, _ = Lazy.force result in
      p label (Array.equal Int64.equal stored patterns));
  Verdict.in_leg ~leg
    (claim "every counter cell equals the host's %s_to_uint4x32 of its source" name) (fun label ->
      let _, _, got = Lazy.force result in
      let want = Array.map patterns ~f:host in
      (* The first few differing cells, as words, for diagnosis; the claim carries the verdict. *)
      if Array.length got = Array.length want then
        List.filter
          (List.init (Array.length want) ~f:Fn.id)
          ~f:(fun i -> not (Array.equal Int.equal got.(i) want.(i)))
        |> Fn.flip List.take 4
        |> List.iter ~f:(fun i ->
            Stdio.eprintf "  %s 0x%Lx: got %s want %s (not part of the golden)\n" name patterns.(i)
              (Test_utils.hex_words got.(i))
              (Test_utils.hex_words want.(i)));
      p_all2 label got want ~f:(Array.equal Int.equal))

let () =
  Stdio.eprintf "uint4x32_counter_conversion: backend %s (not part of the golden)\n%!" backend_name;
  List.iter legs ~f:(fun leg -> Verdict.case (Ir.Ops.prec_string leg.prec) (fun () -> run_leg leg))
