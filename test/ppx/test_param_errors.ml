open Base
open Ocannl
open Nn_blocks.DSL_modules

(* gh-ocannl-1987: parameters %op cannot honour where they are written, refused with the fix named.
   This file is only expanded: each case is an error node in the golden.

   An inline declaration is created where the enclosing [()] is applied -- without one, when the
   function is defined -- so a field naming a parameter bound later than that was OCaml's "Unbound
   value", or a capture of an outer value of that name. *)
let%op late_field hid_dim x = ({ w } * x) + { b; o = [ hid_dim ] }
let%op late_field_labelled ~hid_dim x = ({ w } * x) + { b; o = [ hid_dim ] }
let%op late_init ~scale x = { w = uniform_at scale } * x
let%op field_after_unit () ~hid_dim x = ({ w } * x) + { b; o = [ hid_dim ] }

let%op field_in_nested () x =
  let g n y = y + { b; o = [ n ] } in
  g 3 x

(* Bound before [()], or rebound inside the field, the name is in scope. *)
let%op field_before_unit ~hid_dim () x = ({ w } * x) + { b; o = [ hid_dim ] }
let%op field_rebinds hid_dim x = x + { b; o = List.map [ 3 ] ~f:(fun hid_dim -> hid_dim) }

(* An unlabelled parameter is taken for a tensor: a label source without [()], a scheme's argument
   after it. One the body uses as an integer or a float was a type error at that use. *)
let%op spec_stride stride x = x +* "stride*o<+k; k => o" { kernel }
let%op spec_dilation dil x = x +* "o<+dil*k; k => o" { kernel }
let%op embedded_float scale x = x *. !.scale
let%op embedded_int count x = x *. !..count
let%op float_exponent n x = x **. n
let%op stride_after_unit () stride x = x +* "stride*o<+k; k => o" { kernel }

(* Not refused: a name the body rebinds (the outer [n] is a tensor), and a function that gets no
   scheme. *)
let%op exponent_rebound n x =
  let pow n = x **. n in
  pow [%oc 2.0] + n

let%op stride_unschemed () stride x = x +* "stride*o<+k; k => o" { kernel } [@@ocannl.no_scheme]
