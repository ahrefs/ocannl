open Base
open Ocannl
open Nn_blocks.DSL_modules

let%op y0 = (2 *. { hey1 }) + 3
let%op y1 x = ({ hey2 } * 2) + x
let%op y2 x1 x2 = (x1 *. { hey3 }) + x2
let%op a = [ (1, 2, 3); (4, 5, 6) ]
let%op b = [| [ 7; 8 ]; [ 9; 10 ] |]
let%op y = ({ hey4 } * ([ 2.0 ] : q)) + ([ 1.0 ] : p)
let%op z = (([ 2.0 ] : q) *. { hey5 }) + ({ hey6 } *. ([ 1.0 ] : q))

let stride = 2
and dilation = 3
and use_padding = true

let%op z2 = { hey7 } +* "stride*a+dilation*b,;b=>a," { hey8 }

let z3 =
  let s = 2 and d = 3 in
  [%op { hey9 } +* "i, s*a+d*bc; bc => i, a, c" { hey10 }]

let%op concat_single = { hey11 } ++^ "i=>i"
let%op concat_pair = ({ hey12 }, { hey13 }) ++^ "i; j => i^j" [ "i"; "j" ]
let%op initialized_no_unit x = relu ({ init_w = kaiming ~scale_sq:2.0 normal1 () } * x)

let initialized_expr =
  [%op
    let f x = relu ({ expr_w = kaiming normal1 () } * x) in
    f]

(* gh-ocannl-2007: an initializer's applications are not wrapped in [at_site], at its root nor in a
   conditional's branches or a [let]'s body, where the expansion qualifies their heads with
   NTDSL. *)
let use_kaiming = true

let%op initialized_compound x =
  relu
    ({ cond_w = (if use_kaiming then kaiming normal1 () else uniform ()) }
    * ({
         let_w =
           (let base = normal1 in
            kaiming base ());
       }
      * x))

let () = ignore (y0, y1, y2, a, b, y)
let () = ignore initialized_compound
let () = ignore (z, z2, z3)
let () = ignore (concat_single, concat_pair)
let () = ignore (initialized_no_unit, initialized_expr)
let%op mlp_layer ~label ~hid_dim () ~x = relu (({ w } * x) + { b; o = [ hid_dim ] })

let%op _use_layer =
  let l1 = mlp_layer ~label:[ "L" ] ~hid_dim:3 () in
  let l2 = mlp_layer ~label:[ "L2" ] ~hid_dim:3 () in
  fun x -> l1 ~x:(l2 ~x)

let%op _config_layer ~label () =
  let l = mlp_layer ~label:(label @ [ "L" ]) ~hid_dim:3 () in
  fun x -> l ~x

let%op _three_layer_perceptron ~label ~dim1 ~dim2 ~dim3 () =
  let l1 = mlp_layer ~label:(label @ [ "L1" ]) ~hid_dim:dim1 () in
  let l2 = mlp_layer ~label:(label @ [ "L2" ]) ~hid_dim:dim2 () in
  let l3 = mlp_layer ~label:(label @ [ "L3" ]) ~hid_dim:dim3 () in
  fun x -> l3 ~x:(l2 ~x:(l1 ~x))

(* Dimension captures are bound per call, at the top of their innermost enclosing function, while
   inline declarations keep their lifting to the unit parameter or the definition
   (gh-ocannl-1507). *)
let%op _capture_no_unit x = x ++ "ab => a" [ "b" ]

let%op _capture_with_unit () x =
  let y = { cw } * x in
  (y ++ "ab => a" [ "b" ]) /. dim b

let%op _capture_siblings =
  let f x = x ++ "ab => a" [ "b" ] in
  let g x = x ++ "ab => b" [ "b" ] in
  fun x -> f x + g x

(* A nested function's capture shadows the enclosing one of the same name, as any binding does; that
   is not a clash (each names its own call's dimension). Both forms: under a function, and at the
   extension point. *)
let%op _capture_nested x =
  let g y = y ++ "ab => a" [ "b" ] in
  x ++ "ab => a" [ "b" ] + g x

let%op _capture_nested_toplevel =
  let g y = y ++ "ab => a" [ "b" ] in
  g ({ cn } ++ "abc => ab" [ "b" ])

(* Automatic shape schemes (gh-ocannl-1751). The returned function of a [()] stage gets one when its
   parameters are tensors -- unlabelled, or labelled with an operand use -- including a returned
   function after the sub-layers; a non-tensor parameter after [()], an optional one, a rebound
   labelled one, [let rec] and [[@@ocannl.no_scheme]] leave the expansion as it was. *)
let%op _scheme_labelled () ~gamma ~beta x = (gamma *. x) + beta

let%op _scheme_tail () =
  let l = mlp_layer ~label:[ "tail" ] ~hid_dim:2 () in
  fun x -> l ~x

let%op _scheme_step () ~train_step x = match train_step with Some () -> x + 1.0 | None -> x
let%op _scheme_mask () ?mask x = match mask with None -> x | Some m -> x *. m

(* gh-ocannl-1792: annotated option-valued tensors (labelled, optional) get one scheme per presence
   pattern, through the [_variants] entry points; a [Tensor.t] annotation is evidence for a labelled
   tensor that is only passed on; a function without [()] reads no label from an option. *)
let%op _scheme_variants () ~(train_step : Tensor.t option) ?(mask : Tensor.t option) x =
  let x = match train_step with Some s -> x + s | None -> x in
  match mask with Some m -> x *. m | None -> x

let%op _scheme_forwarded () ~(gamma : Tensor.t) x = _scheme_labelled () ~gamma ~beta:gamma x
let%op _scheme_late ?(mask : Tensor.t option) x = match mask with Some m -> x *. m | None -> x

(* Without [()], an annotated non-tensor is not a label source either; the function stays
   unschemed. *)
let%op _late_config ~(scale : float) x = x *. !.scale

let%op _scheme_defaulted () ?(scale : Tensor.t option = None) x =
  match scale with Some s -> x *. s | None -> x

let%op _scheme_opted_out () x = { ow } * x [@@ocannl.no_scheme]
let%op rec _scheme_rec () x = if false then _scheme_rec () x else x + 1.0

(* Without a [()] stage every parameter is a label source -- its tensor's label enriches the
   result's -- except an optional one (gh-ocannl-1787): [stride] stays an int and [scale] an option,
   neither read as a tensor. *)
let%op _no_unit_optional ?(stride = 2) ?scale x =
  let y = match scale with None -> x | Some s -> x *. s in
  ignore (stride : int);
  y + 1.0

(* gh-ocannl-1885: without [()], a labelled parameter is a label source by tensor evidence only -- a
   direct operand use ([gamma]) or a [Tensor.t] annotation ([beta], only passed on); [scale] stays a
   float and [label] a string list, neither read as a tensor. *)
let%op _late_gamma ~gamma x = gamma *. x
let%op _late_beta ~(beta : Tensor.t) x = _late_gamma ~gamma:beta x
let%op _late_scale ~scale x = x *. !.scale
let%op _late_label ~label x = mlp_layer ~label:("late" :: label) ~hid_dim:2 () ~x

(* gh-ocannl-1987: without [()], the parameter labelled [label] enriches the result's label, at its
   position among the label sources -- the list for [~label] and a defaulted [?(label = ...)], the
   option's contents for [?label]. The inline declarations are created at definition and cannot take
   it. A [label] with tensor evidence is not this list: labelled it is a tensor like any other,
   optional it is left alone like any other. *)
let%op _late_label_result ~label x = ({ llw } * x) + 1.0
let%op _late_opt_label ?label x = x + 1.0
let%op _late_label_default ?(label = [ "dflt" ]) ~(g : Tensor.t) x = x + g
let%op _late_label_annot ~(label : string list) x = relu x
let%op _late_label_tensor ~label x = x + label
let%op _late_label_default_tensor ?(label = !.1.0) x = x + label

(* gh-ocannl-1987: an integer or a float that is labelled, or bound before [()], stays what it
   is. *)
let%op _late_stride ~stride x = x +* "stride*o<+k; k => o" { ls_kernel }
let%op _unit_stride stride () x = x +* "stride*o<+k; k => o" { us_kernel }
let%op _late_exponent ~n x = x **. n

(* gh-ocannl-1885: [?label] before [()] enriches the inline declarations' labels as [~label] does;
   un-defaulted it is the option itself, defaulted the list. *)
let%op _opt_label ?label ~hid_dim () x = ({ olw } * x) + { olb; o = [ hid_dim ] }
let%op _opt_label_default ?(label = [ "dflt" ]) () x = { odw } * x

(* gh-ocannl-1986: the uncurried forms [op (a, b)] and [op (a, b, c)] of the primitive operations
   expand to the same curried call as [op a b] and [op a b c]: the operator alone is the call head,
   and the operands are translated, so they may hold inline declarations. *)
let%op _uncurried_where c x y = where (c, x, y)
let%op _curried_where c x y = where c x y
let%op _uncurried_fma x = fma (x, { uf_w }, { uf_b })
let%op _uncurried_plus x y = ( + ) (x, y *. { up_w })
let%op _uncurried_nested c x y = where (( < ) (x, y), relu x, fma (c, x, y))
