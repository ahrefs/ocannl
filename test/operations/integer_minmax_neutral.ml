(* The neutral element of a [Max]/[Min] reduction, at the precision of the node it lands in
   (gh-ocannl-2081).

   [Ops.neutral_elem] answers infinities, which only a float slot holds. At an integer precision
   [(unsigned char)(-INFINITY)] is undefined behaviour: a byte max over values at most 4 read 165 on
   cc and 255 on Metal, from a plain [%op] reduction. Lowering now stores the slot's own extreme.

   Each precision runs every route the neutral element takes, against values written out by hand:

   - the whole-operand reduction, over values at the end of the range the neutral element sits at,
   so a neutral element that is anything but the extreme wins; - the same reduction into a wider
   result, whose cells no contribution reaches read the neutral element itself. Read through the raw
   buffer, not through a float: 2^63-1 and 2^64-1 have no double, and that is the reason the upper
   bounds are bit constants; - a padded window, whose out-of-range positions read the neutral
   element through the clamp's [Where] (gh-504: a non-finite identity demands no margins, so no
   padding carries one); - a reduction virtualized into its reader, which holds the neutral element
   in a scope; - a [Split_reduce], whose partials start from it.

   The float legs are the control: their neutral elements stay the infinities. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Ops = Ir.Ops
module Nd = Ir.Ndarray
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
module A = Stdlib.Bigarray.Genarray
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Stdio.eprintf "integer-minmax-neutral backend: %s\n" backend_name

type case = {
  name : string;
  prec : Ops.prec;
  low : string array;  (** eight increasing values at the bottom of the range *)
  high : string array;  (** eight increasing values at the top of the range *)
  bottom : string;  (** [Max]'s neutral element *)
  top : string;  (** [Min]'s neutral element *)
}

let ints base = Array.init 8 ~f:(fun k -> Int.to_string (base + k))

let cases =
  [
    { name = "byte"; prec = Ops.byte; low = ints 0; high = ints 248; bottom = "0"; top = "255" };
    {
      name = "uint16";
      prec = Ops.uint16;
      low = ints 0;
      high = ints 65528;
      bottom = "0";
      top = "65535";
    };
    {
      name = "int32";
      prec = Ops.int32;
      low = ints (-2147483648);
      high = ints 2147483640;
      bottom = "-2147483648";
      top = "2147483647";
    };
    {
      name = "uint32";
      prec = Ops.uint32;
      low = ints 0;
      high = ints 4294967288;
      bottom = "0";
      top = "4294967295";
    };
    (* The doubles nearest the int64 limits are 1024 apart, and nearest 2^64 2048 apart. *)
    {
      name = "int64";
      prec = Ops.int64;
      low =
        [|
          "-9223372036854775808";
          "-9223372036854774784";
          "-9223372036854773760";
          "-9223372036854772736";
          "-9223372036854771712";
          "-9223372036854770688";
          "-9223372036854769664";
          "-9223372036854768640";
        |];
      high =
        [|
          "9223372036854767616";
          "9223372036854768640";
          "9223372036854769664";
          "9223372036854770688";
          "9223372036854771712";
          "9223372036854772736";
          "9223372036854773760";
          "9223372036854774784";
        |];
      bottom = "-9223372036854775808";
      top = "9223372036854775807";
    };
    {
      name = "uint64";
      prec = Ops.uint64;
      low = ints 0;
      high =
        [|
          "18446744073709535232";
          "18446744073709537280";
          "18446744073709539328";
          "18446744073709541376";
          "18446744073709543424";
          "18446744073709545472";
          "18446744073709547520";
          "18446744073709549568";
        |];
      bottom = "0";
      top = "18446744073709551615";
    };
    {
      name = "single";
      prec = Ops.single;
      low = Array.init 8 ~f:(fun k -> Int.to_string (k - 8));
      high = ints 1;
      bottom = "-infinity";
      top = "infinity";
    };
  ]

(* Spelled like the table: an integer cell exactly, off the raw buffer, a float cell through its
   integer spelling or the sign of its infinity. *)
let cell (nd : Nd.t) i =
  let idx = if Array.is_empty (Nd.dims nd) then [||] else [| i |] in
  let float_cell v =
    if Float.is_inf v then if Float.(v < 0.) then "-infinity" else "infinity"
    else Printf.sprintf "%.0f" v
  in
  match nd with
  | Byte_nd a -> Int.to_string (Char.to_int (A.get a idx))
  | Uint16_nd a -> Int.to_string (A.get a idx)
  | Int32_nd a -> Printf.sprintf "%ld" (A.get a idx)
  | Uint32_nd a -> Printf.sprintf "%lu" (A.get a idx)
  | Int64_nd a -> Printf.sprintf "%Ld" (A.get a idx)
  | Uint64_nd a -> Printf.sprintf "%Lu" (A.get a idx)
  | Single_nd a -> float_cell (A.get a idx)
  | _ -> failwith "integer_minmax_neutral: a precision outside the table"

let cells nd = Array.init (Array.fold (Nd.dims nd) ~init:1 ~f:( * )) ~f:(cell nd)

let leaf ~prec ~label values =
  let n = Array.length values in
  Operation.init ~l:label ~prec ~o:[ n ]
    ~f:(fun idx -> Float.of_string values.(idx.(0)))
    ~grad_spec:Tensor.Prohibit_grad ()

(* The whole value of a table entry is [Float.of_string]'s: every entry is a double exactly. *)
let filled ~prec ~label ~n value = leaf ~prec ~label (Array.create ~len:n value)
let%cd min1_asn ~t ~t1 ~projections = v =:@- v1
let%cd min2_asn ~t ~t1 ~t2 ~projections = v =:@- v1 + v2
let%cd max2_asn ~t ~t1 ~t2 ~projections = v =:@^ v1 + v2
let no_grad1 ~t:_ ~g:_ ~t1:_ ~projections:_ = Asgns.empty_comp
let no_grad2 ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ = Asgns.empty_comp

let einmin1 spec =
  Tensor.unop
    ~transpose_op:(Shape.Permute (Shape.einsum_spec spec, []))
    ~op_asn:min1_asn ~grad_asn:no_grad1 ~op_label:"@-=>" ~grad_spec:Tensor.Prohibit_grad

let einmax1 = NTDSL.einmax1

(* [Operation.tropical] and its [Min] mirror, with no gradient to generate. *)
let window2 op_asn spec =
  Tensor.binop
    ~compose_op:(Shape.Einsum (Shape.einsum_spec spec, []))
    ~op_asn ~grad_asn:no_grad2 ~op_label:"win" ~grad_spec:Tensor.Prohibit_grad

let routine = ref 0

let run ?(transform = Fn.id) ~label (nodes : Tensor.t list) (result : Tensor.t) =
  Int.incr routine;
  let name = Printf.sprintf "imn_%s_%d" label !routine in
  let ctx = Context.auto () in
  List.iter nodes ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let comp = Train.forward result in
  let comp = { comp with asgns = Asgns.Block_comment (name, comp.Asgns.asgns) } in
  let ctx, r = Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Idx.Empty in
  let ctx = Context.run ctx r in
  cells (Context.to_host ctx result.Tensor.value)

let check case leg got want =
  Stdio.eprintf "%s %s: got [%s] (not part of the golden)\n" case.name leg
    (String.concat_array ~sep:" " got);
  pf "%s %s reads the hand-computed cells" case.name leg (Array.equal String.equal got want)

let first_loop_of_extent ~n (llc : LL.t) =
  Option.map
    (Ll_test.find_loop ~in_scopes:false llc ~f:(fun site -> site.Ll_test.ls_extent = n))
    ~f:(fun site -> site.Ll_test.ls_index)

(* Ranks 0-7 in a shuffled order, so that neither end of the operand is an end of the range. *)
let shuffled values = Array.map [| 3; 1; 4; 0; 6; 2; 7; 5 |] ~f:(fun r -> values.(r))

let run_case case =
  let prec = case.prec in
  let low = shuffled case.low and high = shuffled case.high in
  let at_low r = case.low.(r) and at_high r = case.high.(r) in
  Verdict.case (case.name ^ " whole reduction") (fun () ->
      let x = leaf ~prec ~label:"x" low in
      let y = einmax1 "i => 0" x () in
      check case "max" (run ~label:"max" [ y ] y) [| at_low 7 |];
      let x = leaf ~prec ~label:"x" high in
      let y = einmin1 "i => 0" x () in
      check case "min" (run ~label:"min" [ y ] y) [| at_high 0 |]);
  Verdict.case (case.name ^ " neutral cells") (fun () ->
      let x = leaf ~prec ~label:"x" low in
      let y = einmax1 "i => 0" x ~output_dims:[ 3 ] () in
      check case "max neutral" (run ~label:"maxn" [ y ] y) [| at_low 7; case.bottom; case.bottom |];
      let x = leaf ~prec ~label:"x" high in
      let y = einmin1 "i => 0" x ~output_dims:[ 3 ] () in
      check case "min neutral" (run ~label:"minn" [ y ] y) [| at_high 0; case.top; case.top |]);
  (* N=8, stride 2, window 5: y_o reduces x[2o-2 .. 2o+2], clipped to the operand. Over the ranks:
     windows {3 1 4} {3 1 4 0 6} {4 0 6 2 7} {6 2 7 5}. *)
  Verdict.case (case.name ^ " clamped window") (fun () ->
      let spec = "2*o=+w; w => o" in
      let zero = "0" in
      let x = leaf ~prec ~label:"x" low in
      let k = filled ~prec ~label:"k" ~n:5 zero in
      let y = window2 max2_asn spec x k () in
      check case "window max" (run ~label:"wmax" [ y ] y)
        [| at_low 4; at_low 6; at_low 7; at_low 7 |];
      pf "%s the clamped window leaves its operand unpadded" case.name
        (Option.is_none (Lazy.force x.Tensor.value.Ir.Tnode.padding));
      let x = leaf ~prec ~label:"x" high in
      let k = filled ~prec ~label:"k" ~n:5 zero in
      let y = window2 min2_asn spec x k () in
      check case "window min" (run ~label:"wmin" [ y ] y)
        [| at_high 1; at_high 0; at_high 0; at_high 2 |]);
  (* A scalar reduction has no capture site, so it is the row-wise form that stays virtual
     (virtualization-and-inlining.md); its two rows are reverses of each other. *)
  Verdict.case (case.name ^ " virtualized") (fun () ->
      let rows = [| high; Array.rev high |] in
      let x =
        Operation.init ~l:"x" ~prec ~o:[ 2; 8 ]
          ~f:(fun idx -> Float.of_string rows.(idx.(0)).(idx.(1)))
          ~grad_spec:Tensor.Prohibit_grad ()
      in
      let m = einmin1 "ij => i" x () in
      let zero = filled ~prec ~label:"zero" ~n:2 "0" in
      let y =
        Tensor.binop ~op_asn:max2_asn ~grad_asn:no_grad2 ~grad_spec:Tensor.Prohibit_grad m zero ()
      in
      let virtualized = ref false in
      let got =
        run ~label:"virt" [ y ] y ~transform:(fun o ->
            virtualized := Ll_test.known_virtual o m.Tensor.value;
            o)
      in
      check case "min read through a scope" got [| at_high 0; at_high 0 |];
      pf "%s the reduction is virtualized into its reader" case.name !virtualized);
  Verdict.case (case.name ^ " split reduce") (fun () ->
      let split y o =
        let axis = Option.value_exn ~here:[%here] (first_loop_of_extent ~n:8 o.LL.llc) in
        let op, _, _, _ = Sched.split_reduce ~axis ~target:y.Tensor.value ~num_blocks:3 in
        Sched.apply [ op ] o
      in
      let x = leaf ~prec ~label:"x" low in
      let y = einmax1 "i => 0" x () in
      check case "split max" (run ~label:"smax" [ y ] y ~transform:(split y)) [| at_low 7 |];
      let x = leaf ~prec ~label:"x" high in
      let y = einmin1 "i => 0" x () in
      check case "split min" (run ~label:"smin" [ y ] y ~transform:(split y)) [| at_high 0 |])

let () =
  Tensor.unsafe_reinitialize ();
  List.iter cases ~f:run_case
