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

(* gh-ocannl-2058: a name the body binds -- by a let, a match or function case, a for loop -- is
   bound below the lift point as well. A let rebinding a parameter bound before [()] was a silent
   read of the parameter. *)
let%op field_let_bound ~dims () x =
  let n = List.length dims in
  x + { b; o = [ n ] }

let%op field_let_bound_no_unit x =
  let n = [%oc 3] in
  x + { b; o = [ n ] }

let%op field_let_shadows ~n () x =
  let n = [%oc n + 1] in
  x + { b; o = [ n ] }

let%op init_let_bound () x =
  let scale = [%oc 7] in
  { w = uniform_at scale } * x

let%op field_let_tuple () x =
  let ((_, n) as _pair) = [%oc 2, 3] in
  x + { b; o = [ n ] }

let%op field_let_rec () x =
  let rec n = [%oc 3] and f y = y + { b; o = [ n ] } in
  f x

(* Without a [()] directly under [%op], the declarations are created at the definition, above any
   let. *)
let%op let_above_unit =
  let n = [%oc 3] in
  fun () x -> x + { b; o = [ n ] }

let%op field_match_case ~dims () x = match dims with n :: _ -> x + { b; o = [ n ] } | [] -> x

let%op field_function_case ~dims () x =
  List.fold dims ~init:x ~f:(fun x -> function n -> x + { b; o = [ n ] })

let%op field_for_loop () x =
  for n = 1 to 2 do
    ignore (x + { b; o = [ n ] })
  done;
  x

(* Accepted: a binder of the body no field names, and one bound inside the field itself. *)
let%op let_unnamed ~hid_dim () x =
  let n = [%oc 2 * hid_dim] in
  let y = x + { b; o = [ hid_dim ] } in
  y + !..n

let%op let_in_field () x =
  x
  + {
      b;
      o =
        (let n = 3 in
         [ n ]);
    }

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
