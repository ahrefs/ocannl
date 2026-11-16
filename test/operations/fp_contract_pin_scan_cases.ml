(* Typed source fixtures for the fp-contract subject reader. *)
let cases =
  [
    ( "equal wording over output bits",
      true,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = p "equal" (same_bits got.(0) 1.)
|ocaml}
    );
    ( "local comparator alias",
      true,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = let eq = same_bits in p "equal" (eq got.(0) 1.)
|ocaml}
    );
    ( "non-bit comparator shadow",
      false,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = let same_bits x y = x < y in p "equal" (same_bits got.(0) 1.)
|ocaml}
    );
    ( "host arithmetic only",
      false,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let () = p "equal" (same_bits (1. +. 2.) 3.)
|ocaml}
    );
    ( "numeric equality",
      false,
      {ocaml|
open Verdict.Claims
let got = Context.get_values ctx tensor
let () = p "equal" (Float.equal got.(0) 1.)
|ocaml}
    );
    ( "array comparator and read helper",
      true,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let equal = Array.equal same_bits
let read () = Context.get_values ctx tensor
let () = p "equal" (equal (read ()) [|1.|])
|ocaml}
    );
    ( "qualified claim and module aliases",
      true,
      {ocaml|
module V = Verdict
module C = Context
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let () = V.claim "equal" (eq (C.get_values ctx tensor).(0) 1.)
|ocaml}
    );
    ( "gated claim",
      true,
      {ocaml|
open Verdict.Claims
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = gated ~when_:true ~on:"cc" "equal" (eq got.(0) 1.)
|ocaml}
    );
    ( "quantified predicate and output collection",
      true,
      {ocaml|
open Verdict.Claims
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let () = p_all "equal" [Context.get_values ctx tensor] ~f:(Array.equal eq [|1.|])
|ocaml}
    );
    ( "unused comparator is not applied",
      false,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = p "equal" (let unused = same_bits in Float.equal got.(0) 1.)
|ocaml}
    );
    ( "shadowed claim",
      false,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let p _ _ = ()
let () = p "equal" (same_bits got.(0) 1.)
|ocaml}
    );
    ( "shadowed output module",
      false,
      {ocaml|
open Verdict.Claims
let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
module Context = struct let get_values _ _ = [|1.|] end
let got = Context.get_values ctx tensor
let () = p "equal" (same_bits got.(0) 1.)
|ocaml}
    );
  ]
