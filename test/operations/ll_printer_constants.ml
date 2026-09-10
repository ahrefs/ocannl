(** What the IR dumps show for a floating-point constant (gh-ocannl-713).

    [build_files/<routine>.ll] and the [.cd] dumps are the surface a constant bug is chased on, so
    what they print about a constant has to BE the constant. Rendered with [%.16g] it was not, in
    two ways. A value with no fractional part lost its radix point, so a dumped [-0.] came out as
    the token [-0]: the integer zero, which is what any C-family dialect reading the dump back would
    make of it, and the exact shape of the bug gh-ocannl-615 chased. And 16 significant digits do
    not recover every double, so a constant whose 17th digit mattered was displayed as its 16-digit
    neighbour — [0.1 +. 0.2] and [0.3] printed the same text, which is the harder failure to
    suspect, because the view and the value disagree while both look ordinary. Both printers now
    share the renderer that already settled this for kernel text ([Utils.decimal_float_literal],
    gh-ocannl-623), minus its C-dialect spellings: an IR dump is not C, so the specials stay the
    words [%.16g] gives them.

    The table below is the dump surface itself: each row is what the two printers write for one
    constant, keyed by its exact hexadecimal spelling — decimal-rounding-free, so the row label
    cannot itself lose the distinction it is about. The claims under it are the properties that make
    the table trustworthy: every token parses back to the very double it names, no finite token is
    an integer literal, and the tokens tell apart values the printers used to conflate. *)

open Base
module LL = Ir.Low_level
module Ll = Ll_test
open Verdict.Claims

let tn = Ll.node_factory ~first_id:7130 ~dims:[| 1 |] () "konst"

let render doc =
  let b = Buffer.create 256 in
  PPrint.ToBuffer.pretty 0.7 100 b doc;
  Buffer.contents b

(** The rendered constant, lifted out of the single statement it sits in: everything between the
    assignment arrow and the terminator. Reading the token back out of a real printed statement —
    rather than calling the renderer directly — is what makes this a test of the dump. *)
let token_of statement =
  let arrow = " := " in
  let at = Option.value_exn (String.substr_index statement ~pattern:arrow) in
  String.subo statement ~pos:(at + String.length arrow)
  |> String.strip
  |> String.rstrip ~drop:(Char.equal ';')
  |> String.strip

let cd_token v = token_of (render (LL.to_doc () (Ll.set_at tn (Ll.fixed 0) (Ll.c v))))
let ll_token v = token_of (render (LL.to_doc_cstyle () (Ll.set_at tn (Ll.fixed 0) (Ll.c v))))

(** Whether [token] names exactly [v] — the same bits, so that the sign of a zero counts as a
    difference the way it does everywhere else in this issue. A NaN has no bit pattern worth
    preserving through text, so for it the question is only whether the token is still a NaN. *)
let round_trips v token =
  match Float.of_string token with
  | parsed ->
      if Float.is_nan v then Float.is_nan parsed
      else Int64.equal (Int64.bits_of_float parsed) (Int64.bits_of_float v)
  | exception _ -> false

let values =
  [
    0.0;
    -0.0;
    2.0;
    -4.0;
    0.1;
    0.1 +. 0.2;
    0.3;
    1e20;
    1e-300;
    Float.max_finite_value;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
  ]

let () =
  Stdio.printf "%-26s %-26s %s\n" "exact value" "%cd dump" "C-style dump";
  List.iter values ~f:(fun v ->
      Stdio.printf "%-26s %-26s %s\n" (Printf.sprintf "%h" v) (cd_token v) (ll_token v));
  let tokens = List.concat_map values ~f:(fun v -> [ (v, cd_token v); (v, ll_token v) ]) in
  p_all "every dumped constant parses back to the double it names" tokens ~f:(fun (v, token) ->
      round_trips v token);
  (* A finite token that is neither is an integer literal, which is how the radix point went missing
     in the first place. *)
  p_all "every finite constant is dumped as a floating literal, with a radix point or an exponent"
    tokens ~f:(fun (v, token) ->
      (not (Float.is_finite v))
      || String.exists token ~f:(function '.' | 'e' | 'E' -> true | _ -> false));
  p "the two zeros are dumped differently"
    (String.(cd_token 0.0 <> cd_token (-0.0)) && String.(ll_token 0.0 <> ll_token (-0.0)));
  p "a constant needing a 17th digit is dumped differently from its 16-digit neighbour"
    (String.(cd_token (0.1 +. 0.2) <> cd_token 0.3)
    && String.(ll_token (0.1 +. 0.2) <> ll_token 0.3))
