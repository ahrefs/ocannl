open Base
open Verdict.Claims
module Scan = Premise_reader.Premise_scan

let protocol =
  {ocaml|
module L = Ll_test
let a = L.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 1.)
let b = L.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 2.)
let products = L.matmul_products ~m:2 ~n:2 ~k:2 a b
let () = p_all "exact" products ~f:(L.exact_partials ~sig_bits:24)
let reference = List.hd_exn (L.execute ~name:"ref" program ~seed:[] ~read:[])
let () = p_all2 "reference" reference
  (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))) ~f:Float.equal
|ocaml}

let read source =
  Test_scan.Scan_attempt.attempted ~what:"premise scan" ~name:"control" ~default:[] (fun () ->
      Scan.read source)

let () =
  let malformed = Scan.read_or_refusal ~reader:Scan.read ~source:"control.ml" "let =" in
  p "parse failures name the input"
    (match malformed with
    | Error s -> String.is_substring s ~substring:"premise input does not parse:"
    | Ok _ -> false);
  let raised =
    Scan.read_or_refusal
      ~reader:(fun _ -> invalid_arg "planted scanner defect")
      ~source:"control.ml" protocol
  in
  p "scanner defects retain their exception"
    (match raised with
    | Error s ->
        String.is_substring s
          ~substring:"premise scanner raised: Invalid_argument(\"planted scanner defect\")"
    | Ok _ -> false);
  let covered = read protocol in
  p "aliased operand and exact_partials are checked"
    (List.length covered = 1 && Scan.uncovered covered = 0);
  let unchecked =
    String.substr_replace_all protocol ~pattern:"L.operand ~b:[] ~o:[2] ~i:[2]" ~with_:"Fun.id"
    |> read
  in
  p "a new unchecked site fails with no debt row"
    (List.length (Scan.violations ~debt:[] [ ("control.ml", unchecked) ]) = 1);
  p "conversion without lowering its row fails stale"
    (List.length (Scan.violations ~debt:[ ("control.ml", 1) ] [ ("control.ml", covered) ]) = 1);
  p "a zero debt row is stale"
    (List.length (Scan.violations ~debt:[ ("control.ml", 0) ] [ ("control.ml", covered) ]) = 1);
  p "a removed source leaves a stale row"
    (List.length (Scan.violations ~debt:[ ("removed.ml", 1) ] []) = 1);
  let shadowed_claim = read ("let p_all2 _ _ _ ~f:_ = ()\n" ^ protocol) in
  p "a local comparison function cannot confer coverage" (Scan.uncovered shadowed_claim = 1);
  let shadowed_collection = read ("module List = List\n" ^ protocol) in
  p "an unmodelled collection alias cannot confer coverage" (Scan.uncovered shadowed_collection = 1);
  let without_reference =
    String.substr_replace_all protocol ~pattern:"p_all2" ~with_:"ignore_comparison" |> read
  in
  p "checked operands alone do not cover a premise" (Scan.uncovered without_reference = 1);
  let shadowed =
    String.substr_replace_all protocol ~pattern:"module L = Ll_test" ~with_:"module L = Other"
    |> read
  in
  p "a shadowed module is outside the Ll_test census" (List.length shadowed = 0);
  let direct = read "let f cell = Ll_test.exact_partials ~sig_bits:24 cell" in
  p "fully applied sites are counted once" (List.length direct = 1 && Scan.uncovered direct = 1);
  let unrelated =
    "let unused = Ll_test.operand ~b:[] ~o:[] ~i:[] (fun _ -> 1.)\n"
    ^ String.substr_replace_all protocol ~pattern:"L.operand ~b:[] ~o:[2] ~i:[2]" ~with_:"Fun.id"
  in
  p "an unrelated checked operand cannot cover a site" (Scan.uncovered (read unrelated) = 1)
