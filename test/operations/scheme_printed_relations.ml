(* gh-ocannl-1988: what [Tensor.Scheme.to_string] prints among a scheme's boundary variables.
   Arguments related only through an eliminated intermediate -- consumed together by a reduction, so
   that no boundary or fixed variable refines them both -- print as a common refinement [_ ⊑ a, b]
   (an anonymous witness that the two broadcast together), and the relations print grouped by left
   side in a canonical order rather than the solver's traversal order. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

(* The product is reduced away: [x] and [y] are joined only through it. *)
let%op joined () x y = (x *. y) ++ "...|...->... => |->0"

(* The product is the result: it refines both arguments, so it witnesses their relation. *)
let%op joined_kept () x y = x *. y

(* The result refines several fixed rows (the parameters'): one line per left side. *)
let%op biased () x = x + { w1 } + { w2 } + { w3 }

let%op sized () x =
  let y = x ++ "...|...->i => ...|...->i" [ "i" ] in
  [%oc Shape.set_dim i 3];
  y

let traced build =
  let _, schemes = Tensor.Scheme.traced_by build in
  List.map schemes ~f:(fun s ->
      let text = Tensor.Scheme.to_string s in
      Stdio.print_endline text;
      text)

let relation_lines text =
  String.split_lines text
  |> List.filter ~f:(fun l -> String.is_prefix l ~prefix:"  ")
  |> List.map ~f:String.strip

let left_side line = String.lsplit2_exn line ~on:'\xe2' |> fst |> String.strip

let () =
  Verdict.case "arguments joined only through an eliminated intermediate" (fun () ->
      Tensor.unsafe_reinitialize ();
      let text = List.hd_exn (traced (fun () -> joined ())) in
      let commons = List.filter (relation_lines text) ~f:(String.is_prefix ~prefix:"_ ⊑") in
      p "each of the three rows of x and y prints as a common refinement"
        (List.length commons = 3
        && List.for_all commons ~f:(fun l -> String.count l ~f:(Char.equal ',') = 1)));
  Verdict.case "a result refining both arguments witnesses their relation" (fun () ->
      Tensor.unsafe_reinitialize ();
      let text = List.hd_exn (traced (fun () -> joined_kept ())) in
      p_none "no common refinement prints where the result refines both arguments"
        (relation_lines text)
        ~f:(String.is_prefix ~prefix:"_ ⊑"));
  Verdict.case "relations print grouped by left side" (fun () ->
      Tensor.unsafe_reinitialize ();
      let texts = traced (fun () -> biased ()) in
      let lines = List.concat_map texts ~f:relation_lines in
      let lefts =
        List.filter_map lines ~f:(fun l ->
            if String.is_prefix l ~prefix:"_ " then None else Some (left_side l))
      in
      p "no two lines share a left side" (not (List.contains_dup lefts ~compare:String.compare));
      p_exists "a line lists several upper bounds" lines ~f:(fun l ->
          String.is_substring l ~substring:", "));
  Verdict.case "numeric capture setters remain visible before their basis settles" (fun () ->
      Tensor.unsafe_reinitialize ();
      let text = List.hd_exn (traced (fun () -> sized ())) in
      p_exists "the signature shows its held size of 3" (relation_lines text) ~f:(fun line ->
          String.is_prefix line ~prefix:"size(" && String.is_suffix line ~suffix:" = 3"))
