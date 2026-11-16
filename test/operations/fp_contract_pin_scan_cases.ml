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
    ( "numeric collection predicate",
      false,
      {ocaml|
open Verdict.Claims
let got = Context.get_values ctx tensor
let () = p "equal" (Array.equal Float.equal got [|1.|])
|ocaml}
    );
    ( "indexed claim over low-level executed outputs",
      true,
      {ocaml|
open Verdict.Claims
let eq a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let got = List.hd (Ll_test.execute ~name:"probe" () ~seed:[] ~read:[tensor])
let () = p_alli "rounding" [0;1] ~f:(fun i _ -> eq got.(i) got.(i))
|ocaml}
    );
    ( "qualified top-level p",
      true,
      {ocaml|
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = Context.get_values ctx tensor
let () = Verdict.p "equal" (eq got.(0) 1.)
|ocaml}
    );
    ( "Int32 bit pattern against encoding",
      true,
      {ocaml|
open Verdict.Claims
let got = Context.get_values ctx tensor
let () = claim "rounding" (Int32.equal (Int32.bits_of_float got.(0)) 0x3e396287l)
|ocaml}
    );
    ( "gated pairwise claim and census reader",
      true,
      {ocaml|
open Verdict.Claims
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let values, _ = Ll_test.execute_with_census ~name:"probe" () ~seed:[] ~read:[tensor]
let got = List.hd values
let () = gated_all ~when_:true ~on:"cc" "equal" [got] ~f:(Array.equal eq [|1.|])
|ocaml}
    );
    ( "shadowed low-level output module",
      false,
      {ocaml|
open Verdict.Claims
module Ll_test = struct let execute ~name:_ () ~seed:_ ~read:_ = [[|1.|]] end
let eq x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let got = List.hd (Ll_test.execute ~name:"probe" () ~seed:[] ~read:[tensor])
let () = p "equal" (eq got.(0) 1.)
|ocaml}
    );
  ]
