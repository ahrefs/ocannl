(* A floating constant an integer slot cannot hold is refused at lowering (gh-ocannl-2085).

   C converts a floating constant to an integer type by truncating it toward zero, and the behaviour
   is undefined when the result does not fit (C11 6.3.1.4). [(unsigned char)300.0] made clang drop
   the store, so the cell kept the [0xA5] fill of [debug_fill_fresh_outputs], with no error. [where]
   over [Nn_blocks.default_mask_fill] at byte read 0 in the masked cell. On Metal, an out-of-range
   float-to-uint32 conversion stalled a kernel. [Low_level.simplify_llc] now raises
   [Unrepresentable_integer_constant] for such a constant, judged at the precision the slot renders
   it at, whatever [check_half_prec_constants_cutoff] says.

   The refused cases are compiled, never run: an undefined conversion is no oracle, and on Metal it
   can hang. The accepted cases run, with per-cell values distinct from each other and from the
   fill, at the range endpoints and at the fractional constants whose truncation lands on them.
   Hand-built IR then pins which precision counts: the slot's, not the destination's. *)
open Base
module Train = Ocannl.Train
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode
open Verdict.Claims

let n = 2
let backend_name = Context.backend_name (Context.auto ())
let () = Stdio.eprintf "integer_constant_range: backend %s (not part of the golden)\n" backend_name
let same_value a b = (Float.is_nan a && Float.is_nan b) || Float.equal a b

type outcome =
  | Refused of { tn : Tn.t option; scope : Tn.t option; prec : Ops.prec; value : float }
  | Accepted
  | Raised of string

let outcome f =
  match f () with
  | _ -> Accepted
  | exception LL.Unrepresentable_integer_constant { tn; scope; prec; value } ->
      Refused { tn; scope; prec; value }
  | exception e -> Raised (Exn.to_string e)

(* Diagnostics for a failed claim, on stderr. *)
let report label = function
  | Raised msg -> Stdio.eprintf "%s: raised %s\n" label msg
  | Accepted -> Stdio.eprintf "%s: accepted\n" label
  | Refused { tn = Some tn; prec; _ } ->
      Stdio.eprintf "%s: refused naming %s at %s\n" label (Tn.debug_name tn) (Ops.prec_string prec)
  | Refused { tn = None; prec; _ } ->
      Stdio.eprintf "%s: refused naming no node at %s\n" label (Ops.prec_string prec)

(** [label]'s outcome is the refusal of [value] at [prec], naming [tn] (and the inlined [scope]). *)
let refused ?scope label ~tn ~prec ~value got =
  let ok =
    match got with
    | Refused r ->
        Option.equal Tn.equal r.tn tn
        && Option.for_all scope ~f:(fun scope -> Option.equal Tn.equal r.scope (Some scope))
        && Ops.equal_prec r.prec prec && same_value r.value value
    | Accepted | Raised _ -> false
  in
  if not ok then report label got;
  p (label ^ ": refused, naming the node, the precision and the value") ok

let accepted label got =
  let ok = match got with Accepted -> true | Refused _ | Raised _ -> false in
  if not ok then report label got;
  p (label ^ ": accepted") ok

let input ?(l = "a") prec values = NTDSL.init ~l ~prec ~o:[ n ] ~f:(fun idx -> values.(idx.(0))) ()

let compile (t : Tensor.t) () =
  Train.to_routine (Context.auto ()) Ir.Indexing.Empty (Train.forward t)

let values (t : Tensor.t) = Context.get_values (Train.forward_once (Context.auto ()) t) t.value

(* The 64-bit precisions have no fractional floats near their endpoints, and their largest floats
   below the upper endpoint are 2^63 - 1024 and 2^64 - 2048. *)
let wide prec = match prec with Ops.Int64_prec _ | Ops.Uint64_prec _ -> true | _ -> false

let range prec =
  match Ops.integer_range prec with
  | Some (lo, hi) ->
      let hi =
        match prec with
        | Ops.Int64_prec _ -> hi -. 1024.
        | Ops.Uint64_prec _ -> hi -. 2048.
        | _ -> hi
      in
      (lo, hi)
  | None -> assert false

(* The nearest floats past either endpoint. *)
let below prec =
  match prec with Ops.Int64_prec _ -> -9223372036854777856. | _ -> fst (range prec) -. 1.

let above prec =
  match prec with
  | Ops.Int64_prec _ -> 9223372036854775808.
  | Ops.Uint64_prec _ -> 18446744073709551616.
  | _ -> snd (range prec) +. 1.

let step prec = if wide prec then 2048. else 1.

(* Refused at every integer precision: a literal in [a + c], compiled. *)
let () =
  List.iter Ops.integer_precs ~f:(fun prec ->
      let ps = Ops.prec_string prec in
      let cases =
        [
          ("NaN", Float.nan);
          ("+inf", Float.infinity);
          ("-inf", Float.neg_infinity);
          ("just below the range", below prec);
          ("just above the range", above prec);
        ]
        @
        if wide prec then []
        else
          [
            ("a fraction truncating below the range", fst (range prec) -. 1.5);
            ("a fraction truncating above the range", snd (range prec) +. 1.25);
          ]
      in
      List.iter cases ~f:(fun (what, c) ->
          let a = input prec [| 0.; 1. |] in
          let%op r = a + !.c in
          refused
            (Printf.sprintf "%s a + c, c %s" ps what)
            ~tn:(Some r.value) ~prec ~value:c
            (outcome (compile r))))

(* The registered printer names the node and the precision. *)
let () =
  let a = input Ops.byte [| 0.; 1. |] in
  let%op r = a + 300. in
  let message = match compile r () with _ -> "" | exception e -> Exn.to_string e in
  p "the refusal's message names the stored node and the precision"
    (String.is_substring message ~substring:("tensor node " ^ Tn.debug_name r.value)
    && String.is_substring message ~substring:"renders at byte")

(* The cutoff key does not gate the refusal. *)
let () =
  let saved = Utils.settings.check_half_prec_constants_cutoff in
  Utils.settings.check_half_prec_constants_cutoff <- None;
  List.iter Ops.integer_precs ~f:(fun prec ->
      let a = input prec [| 0.; 1. |] in
      let c = above prec in
      let%op r = a + !.c in
      refused
        (Ops.prec_string prec ^ " a + c above the range, with the fp16 cutoff off")
        ~tn:(Some r.value) ~prec ~value:c
        (outcome (compile r)));
  Utils.settings.check_half_prec_constants_cutoff <- saved

(* The user routes the issue names: a mask fill at an integer precision, and the library's unary
   minus over an unsigned tensor, which multiplies by the literal -1. *)
let () =
  List.iter Ops.integer_precs ~f:(fun prec ->
      let ps = Ops.prec_string prec in
      let a = input prec [| 0.; 1. |] in
      let%op w = where (a < 1.) a !.Ocannl.Nn_blocks.default_mask_fill in
      refused
        (ps ^ " where with the default mask fill")
        ~tn:(Some w.value) ~prec ~value:Float.neg_infinity
        (outcome (compile w));
      match Ops.integer_range prec with
      | Some (0., _) ->
          let a = input prec [| 0.; 1. |] in
          let negated = NTDSL.O.( ~- ) a in
          refused
            (ps ^ " ~- over an unsigned tensor")
            ~tn:(Some negated.value) ~prec ~value:(-1.)
            (outcome (compile negated))
      | _ -> ());
  let x = input ~l:"x" Ops.single [| 0.; 1. |] in
  let%op w = where (x < 1.) x !.Ocannl.Nn_blocks.default_mask_fill in
  let got = values w in
  p "single where with the default mask fill keeps its infinity"
    (Float.equal got.(0) 0. && Float.equal got.(1) Float.neg_infinity)

(* Accepted and executed: each endpoint, and a fraction truncating onto it. *)
let () =
  List.iter Ops.integer_precs ~f:(fun prec ->
      let ps = Ops.prec_string prec in
      let lo, hi = range prec in
      let s = step prec in
      let run what c ~down want =
        let a = input prec [| 0.; s |] in
        let r =
          if down then
            let%op r = !.c - a in
            r
          else
            let%op r = !.c + a in
            r
        in
        let got = values r in
        p
          (Printf.sprintf "%s %s gives the truncated value per cell" ps what)
          (Array.equal Float.equal got [| want; (if down then want -. s else want +. s) |])
      in
      run "the top endpoint minus a" hi ~down:true hi;
      run "the bottom endpoint plus a" lo ~down:false lo;
      if not (wide prec) then (
        run "a fraction above the top endpoint, minus a" (hi +. 0.75) ~down:true hi;
        run "a fraction below the bottom endpoint, plus a" (lo -. 0.75) ~down:false lo))

(* Hand-built IR: the precision that counts is the one the slot renders the constant at. *)
let next_first_id = ref 27000

let node prec label =
  next_first_id := !next_first_id + 1;
  Ll_test.node_factory ~prec ~first_id:!next_first_id ~dims:[| n |] () label

let simplified llc () = LL.simplify_llc [] llc

let () =
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let byte = Ops.byte and single = Ops.single in
  let x_single = node single "icr_x_single" and x_byte = node byte "icr_x_byte" in
  let out_byte = node byte "icr_out_byte" and out_single = node single "icr_out_single" in
  List.iter [ x_single; x_byte; out_byte; out_single ] ~f:Ll_test.materialize;
  let cmp prec x c =
    (LL.Binop (Ops.Cmplt, (Ll_test.get x at, prec), (LL.Constant c, prec)), prec)
  in
  let select cond = LL.Ternop (Ops.Where, cond, (LL.Constant 7., byte), (LL.Constant 9., byte)) in
  (* 300 is what the destination's byte could not hold: judged at the store's precision, it would be
     refused. *)
  List.iter
    [ ("300", 300.); ("+inf", Float.infinity) ]
    ~f:(fun (what, c) ->
      accepted
        ("a float condition's " ^ what ^ " in a byte store")
        (outcome
           (simplified
              (Ll_test.loop_n s n (Ll_test.set out_byte at (select (cmp single x_single c)))))));
  refused "a byte condition's 300 in a float store" ~tn:(Some out_single) ~prec:byte ~value:300.
    (outcome
       (simplified
          (Ll_test.loop_n s n
             (Ll_test.set out_single at
                (LL.Ternop
                   ( Ops.Where,
                     cmp byte x_byte 300.,
                     (LL.Constant 7., single),
                     (LL.Constant 9., single) ))))));
  refused "a float-annotated 300 in a homogeneous byte Add" ~tn:(Some out_byte) ~prec:byte
    ~value:300.
    (outcome
       (simplified
          (Ll_test.loop_n s n
             (Ll_test.set out_byte at
                (LL.Binop (Ops.Add, (Ll_test.get x_byte at, byte), (LL.Constant 300., single)))))));
  refused "a byte guard's 300" ~tn:None ~prec:byte ~value:300.
    (outcome
       (simplified
          (Ll_test.loop_n s n
             (Ll_test.if_at byte
                (fst (cmp byte x_byte 300.))
                (Ll_test.set out_byte at (LL.Constant 7.))))));
  let scoped = node byte "icr_scoped" in
  Ll_test.virtualize scoped;
  refused "a byte scope's 300 read into a float slot" ~scope:scoped ~tn:(Some out_single) ~prec:byte
    ~value:300.
    (outcome
       (simplified
          (Ll_test.loop_n s n
             (Ll_test.set out_single at
                (Ll_test.scope scoped
                   (LL.Binop (Ops.Add, (Ll_test.get x_byte at, byte), (LL.Constant 300., byte))))))));
  let state = node byte "icr_state" in
  Ll_test.virtualize state;
  let cr = Ll_test.carry ~init:(LL.Constant 300.) state in
  refused "a byte scan's initial 300" ~tn:(Some state) ~prec:byte ~value:300.
    (outcome
       (simplified
          (Ll_test.scan ~upto:(n - 1) s ~carried:[ cr ]
             (Ll_test.seq
                (Ll_test.set_next cr
                   (LL.Binop (Ops.Add, (Ll_test.prev cr, byte), (LL.Constant 1., byte))))
                (Ll_test.set out_byte at (Ll_test.next cr))))));
  (* The fp16 headroom check, which shares the walk, still refuses at its cutoff. *)
  let out_half = node Ops.half "icr_out_half" in
  Ll_test.materialize out_half;
  p "a half store of 20000 is still refused by the fp16 cutoff"
    (match simplified (Ll_test.loop_n s n (Ll_test.set out_half at (LL.Constant 20000.))) () with
    | _ -> false
    | exception Utils.User_error _ -> true)
