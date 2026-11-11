open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* gh-ocannl-1987: where the [string list] of a parameter labelled [label] goes. Before a [()] it
   enriches the result and inline declarations, created at that application; without a [()] they are
   created once, at definition, and it enriches the result's label, at its position among the
   parameters. *)

let%op unit_layer ~label () x = { ul_w } * x
let%op unit_opt ?label () x = x + 1.0
let%op unit_default ?(label = [ "dflt" ]) () x = x + 1.0

let%op unit_shadow ~label () x =
  let label = { shadow_w } * x in
  label + x

let%op unit_after ~label () label = label + 1.0
let%op unit_tensor ~(label : Tensor.t) () x = label + x
let%op unit_operand ~label () x = label + x

let%op unit_tensor_opt ?(label : Tensor.t option) () x =
  match label with Some label -> label + x | None -> x + 1.0

let fallback = TDSL.range 3
let%op unit_tensor_default ?(label = fallback) () x = label + x
let%op unit_ignored ~label:_ () x = relu x

let%op unit_private_name ~label () x =
  let __ocannl_result_label = x + 1.0 in
  __ocannl_result_label + x

let%op late_layer ~label x = { ll_w } * x
let%op late_opt ?label x = x + 1.0
let%op late_default ?(label = [ "dflt" ]) x = x + 1.0
let%op late_after x ~label = relu x
let label (t : Tensor.t) = t.value.label
let show what l = Stdio.printf "%s: [%s]\n" what (String.concat "; " l)

let declared (t : Tensor.t) name =
  List.find (fun p -> List.mem name (label p)) (Base.Set.to_list t.params)

let claim what t expected =
  show what (label t);
  p (what ^ ": the label is as stated") (label t = expected)

(* A parameter's label also names its initialization: only the membership of [l] is claimed. *)
let claim_declared what t name ~has l =
  let d = declared t name in
  show what (label d);
  p
    (Printf.sprintf "%s: %s %s" what (if has then "it carries" else "it does not carry") l)
    (List.mem l (label d) = has)

let () =
  let x = TDSL.range ~label:[ "x" ] 3 in
  let lx = label x in
  let y = unit_layer ~label:[ "L" ] () x in
  claim "with (): the result" y [ "*"; "unit_layer"; "L" ];
  claim_declared "with (): the inline declaration" y "ul_w" ~has:true "L";
  claim "with (): ?label given" (unit_opt ~label:[ "O" ] () x) [ "+"; "unit_opt"; "O" ];
  claim "with (): ?label omitted" (unit_opt () x) [ "+"; "unit_opt" ];
  claim "with (): default omitted" (unit_default () x) [ "+"; "unit_default"; "dflt" ];
  claim "with (): default given" (unit_default ~label:[ "G" ] () x) [ "+"; "unit_default"; "G" ];
  let y = unit_shadow ~label:[ "S" ] () x in
  claim "with (): shadowed label" y [ "+"; "unit_shadow"; "S" ];
  claim_declared "with (): shadowed inline declaration" y "shadow_w" ~has:true "S";
  claim "with (): tensor named label after unit" (unit_after ~label:[ "A" ] () x)
    [ "+"; "unit_after"; "A" ];
  claim "with (): annotated tensor label" (unit_tensor ~label:x () x) [ "+"; "unit_tensor" ];
  claim "with (): operand tensor label" (unit_operand ~label:x () x) [ "+"; "unit_operand" ];
  claim "with (): optional tensor given" (unit_tensor_opt ~label:x () x) [ "+"; "unit_tensor_opt" ];
  claim "with (): optional tensor omitted" (unit_tensor_opt () x) [ "+"; "unit_tensor_opt" ];
  claim "with (): default tensor omitted" (unit_tensor_default () x) [ "+"; "unit_tensor_default" ];
  claim "with (): default tensor given"
    (unit_tensor_default ~label:x () x)
    [ "+"; "unit_tensor_default" ];
  claim "with (): ignored label" (unit_ignored ~label:[ "I" ] () x) [ "relu"; "unit_ignored" ];
  claim "with (): capture avoids private-name binders"
    (unit_private_name ~label:[ "P" ] () x)
    [ "+"; "unit_private_name"; "P" ];
  let y = late_layer ~label:[ "L" ] x in
  claim "without (): the result" y ([ "*"; "late_layer"; "L" ] @ lx);
  claim_declared "without (): the inline declaration" y "ll_w" ~has:false "L";
  claim "?label given" (late_opt ~label:[ "O" ] x) ([ "+"; "late_opt"; "O" ] @ lx);
  claim "?label omitted" (late_opt x) ([ "+"; "late_opt" ] @ lx);
  claim "?(label = ...) omitted" (late_default x) ([ "+"; "late_default"; "dflt" ] @ lx);
  claim "?(label = ...) given" (late_default ~label:[ "G" ] x) ([ "+"; "late_default"; "G" ] @ lx);
  claim "~label after the tensor" (late_after x ~label:[ "A" ])
    (([ "relu"; "late_after" ] @ lx) @ [ "A" ])
