open Base
open Verdict.Claims
module Scan = Test_scan.Premise_scan

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

let wrapped_protocol =
  {ocaml|
module L = Ll_test
let a = L.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 1.)
let b = L.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 2.)
let seeded seed idcs =
  let _layout = [2; 2] in
  L.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> seed) idcs
let check_products label products want =
  p_all2 label want (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))) ~f:Float.equal
let run () =
  let ctx, routine = Context.compile () in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx () in
  Context.release ctx;
  (got, ())
let reference () = lazy (fst (run ()))
let first = L.reduction_cells ~out:[|2;2|] ~over:[|2|] (fun o r ->
  seeded 1. [|o.(0);r.(0)|] *. seeded 2. [|r.(0);o.(1)|])
let second = L.matmul_products ~m:2 ~n:2 ~k:2 a b
let () = List.iter [ ("first", first); ("second", second) ] ~f:(fun (tag, products) ->
  p_all tag products ~f:(L.exact_partials ~sig_bits:24);
  let want = reference () in
  check_products tag products (Lazy.force want))
|ocaml}

(* gh-ocannl-2334: a support module's helpers, and a test calling them across files. *)
let support_source =
  {ocaml|
let with_zero vals = Ll_test.operand ~b:[] ~o:[ 3 ] ~i:[] (fun ix -> vals.(ix.(0)))
let check_sums label products got =
  p_all2 label got
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))) ~f:Float.equal
let check1 label products got = check_sums label products got
let check2 label products got = check1 label products got
|ocaml}

let cross_module =
  {ocaml|
let a = Exact_sums.with_zero [| 1.; 2. |]
let b = Exact_sums.with_zero [| 3.; 4. |]
let products = Ll_test.matmul_products ~m:2 ~n:2 ~k:2 a b
let () = p_all "exact" products ~f:(Ll_test.exact_partials ~sig_bits:24)
let run () =
  let ctx, routine = Context.compile () in
  Context.run ctx routine
let () = Exact_sums.check_sums "reference" products (Context.get_values (run ()) ())
|ocaml}

(* A support helper comparing products it builds itself, and a caller whose own builder is
   unchecked: aligned at one offset, the two builders must stay apart. *)
let self_checking_support =
  {ocaml|
let a = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 1.)
let b = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 2.)
let self_check () =
  let products = Ll_test.matmul_products ~m:2 ~n:2 ~k:2 a b in
  p_all2 "self" (Context.get_values (Context.run () ()) ())
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))) ~f:Float.equal
|ocaml}

let unchecked_caller =
  {ocaml|
let a = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 1.)
let b = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 2.)
let products = Ll_test.matmul_products ~m:2 ~n:2 ~k:2 a b
let () = p_all "exact" products ~f:(Ll_test.exact_partials ~sig_bits:24)
let () = Exact_sums.self_check ()
|ocaml}

(* gh-ocannl-2334: a Train.*_once context as the executed reference. *)
let once_protocol =
  {ocaml|
let a = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 1.)
let b = Ll_test.operand ~b:[] ~o:[2] ~i:[2] (fun _ -> 2.)
let products = Ll_test.matmul_products ~m:2 ~n:2 ~k:2 a b
let () = p_all "exact" products ~f:(Ll_test.exact_partials ~sig_bits:24)
let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad
let () =
  let t = build () in
  let ctx = Train.forward_once (Context.auto ()) t in
  p_all2 "reference" (Context.get_values ctx t.Tensor.value)
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))) ~f:Float.equal
|ocaml}

let read ?support source =
  Test_scan.Scan_attempt.attempted ~what:"premise scan" ~name:"control" ~default:[] (fun () ->
      Scan.read ?support source)

let support_of source =
  Test_scan.Scan_attempt.attempted ~what:"premise scan" ~name:"support" ~default:Scan.no_support
    (fun () -> Scan.add_support Scan.no_support ~name:"Exact_sums" source)

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
  let other_products =
    String.substr_replace_all protocol ~pattern:"let () = p_all2"
      ~with_:"let () = let products = L.matmul_products ~m:2 ~n:2 ~k:2 a b in p_all2"
    |> read
  in
  p "a different products binding cannot cover the premise" (Scan.uncovered other_products = 1);
  let host_reference =
    String.substr_replace_all protocol
      ~pattern:"List.hd_exn (L.execute ~name:\"ref\" program ~seed:[] ~read:[])"
      ~with_:"Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))"
    |> read
  in
  p "host sums cannot impersonate an executed reference" (Scan.uncovered host_reference = 1);
  let wrapped = read wrapped_protocol in
  p "checked reduction, comparison and lazy reference wrappers cover tuple dispatch"
    (List.length wrapped = 1 && Scan.uncovered wrapped = 0);
  let without_run =
    String.substr_replace_all wrapped_protocol ~pattern:"Context.run ctx routine" ~with_:"ctx"
    |> read
  in
  p "reading a context without running it is not an executed reference"
    (Scan.uncovered without_run = 1);
  let unchecked_dispatch =
    String.substr_replace_all wrapped_protocol ~pattern:"~k:2 a b" ~with_:"~k:2 (fun _ -> 1.) b"
    |> read
  in
  p "one unchecked dispatched products arm keeps the shared site uncovered"
    (Scan.uncovered unchecked_dispatch = 1);
  let builder_wrapper =
    String.substr_replace_all protocol
      ~pattern:"let products = L.matmul_products ~m:2 ~n:2 ~k:2 a b"
      ~with_:
        "let make_products a b = L.matmul_products ~m:2 ~n:2 ~k:2 a b\n\
         let products = make_products a b"
    |> read
  in
  p "opaque product builders cannot export a shared lexical identity"
    (Scan.uncovered builder_wrapper = 1);
  let unrelated =
    "let unused = Ll_test.operand ~b:[] ~o:[] ~i:[] (fun _ -> 1.)\n"
    ^ String.substr_replace_all protocol ~pattern:"L.operand ~b:[] ~o:[2] ~i:[2]" ~with_:"Fun.id"
  in
  p "an unrelated checked operand cannot cover a site" (Scan.uncovered (read unrelated) = 1);
  let support = support_of support_source in
  let covered_across = read ~support cross_module in
  p "a premise checked through a support module's reader and comparison is covered"
    (List.length covered_across = 1 && Scan.uncovered covered_across = 0);
  p "without the support source its helpers confer nothing" (Scan.uncovered (read cross_module) = 1);
  let mutant_support ~pattern ~with_ =
    let mutant = String.substr_replace_first support_source ~pattern ~with_ in
    if String.equal mutant support_source then
      Verdict.fail ("premise scan -- support mutant: pattern not found: " ^ pattern);
    support_of mutant
  in
  let unchecked_comparison =
    mutant_support ~pattern:"got =\n  p_all2"
      ~with_:"got =\n  if String.equal label \"\" then () else p_all2"
  in
  p "a support comparison with a branch that skips the check confers nothing"
    (Scan.uncovered (read ~support:unchecked_comparison cross_module) = 1);
  let unchecked_reader =
    mutant_support ~pattern:"let with_zero vals = Ll_test.operand"
      ~with_:
        "let with_zero vals = if Array.length vals = 0 then (fun ix -> vals.(ix.(0))) else \
         Ll_test.operand"
  in
  p "a support reader with a branch that is no checked reader confers nothing"
    (Scan.uncovered (read ~support:unchecked_reader cross_module) = 1);
  let opened =
    String.substr_replace_all cross_module ~pattern:"Exact_sums." ~with_:""
    |> String.( ^ ) "open Exact_sums\n" |> read ~support
  in
  p "an open of the support module brings its helpers" (Scan.uncovered opened = 0);
  let aliased =
    String.substr_replace_all cross_module ~pattern:"Exact_sums." ~with_:"E."
    |> String.( ^ ) "module E = Exact_sums\n"
    |> read ~support
  in
  p "a module alias of the support module brings its helpers" (Scan.uncovered aliased = 0);
  let included_caller =
    String.substr_replace_all cross_module ~pattern:"Exact_sums." ~with_:""
    |> String.( ^ ) "include Exact_sums\n"
    |> read ~support
  in
  p "an include of the support module brings its helpers" (Scan.uncovered included_caller = 0);
  p "a rebound support module name confers nothing"
    (Scan.uncovered (read ~support ("module Exact_sums = Other_sums\n" ^ cross_module)) = 1);
  let included = support_of ("include Other\n" ^ support_source) in
  p "a support source with a top-level include exports nothing"
    (Scan.uncovered (read ~support:included cross_module) = 1);
  let via ~check =
    read ~support
      (String.substr_replace_first cross_module ~pattern:"Exact_sums.check_sums \"reference\""
         ~with_:("Exact_sums." ^ check ^ " \"reference\""))
  in
  p "a support wrapper chain within the three-call bound is followed"
    (Scan.uncovered (via ~check:"check2") = 0);
  let past_bound =
    support_of (support_source ^ "let check3 label products got = check2 label products got\n")
  in
  let local_open =
    read ~support
      (String.substr_replace_all cross_module ~pattern:"Exact_sums.with_zero [| 1.; 2. |]"
         ~with_:"Exact_sums.(with_zero [| 1.; 2. |])")
  in
  p "a local open of the support module brings its helpers" (Scan.uncovered local_open = 0);
  let aligned_support, aligned_caller =
    let builder = "Ll_test.matmul_products" in
    let at source = Option.value_exn (String.substr_index source ~pattern:builder) in
    let gap = at self_checking_support - at unchecked_caller in
    let pad n = String.make (Int.max n 0) ' ' in
    (pad (-gap) ^ self_checking_support, pad gap ^ unchecked_caller)
  in
  p "the collision control aligns the two builders at one offset"
    (Option.equal Int.equal
       (String.substr_index aligned_support ~pattern:"Ll_test.matmul_products")
       (String.substr_index aligned_caller ~pattern:"Ll_test.matmul_products"));
  p "a support helper's own compared products cannot cover a caller's builder at their offset"
    (Scan.uncovered (read ~support:(support_of aligned_support) aligned_caller) = 1);
  p "the three-call bound spans modules: one call deeper confers nothing"
    (Scan.uncovered
       (read ~support:past_bound
          (String.substr_replace_first cross_module ~pattern:"Exact_sums.check_sums \"reference\""
             ~with_:"Exact_sums.check3 \"reference\""))
    = 1);
  let once = read once_protocol in
  p "the forward_once result read at its own tensor's value is an executed reference"
    (List.length once = 1 && Scan.uncovered once = 0);
  let once_variant ~pattern ~with_ =
    let variant = String.substr_replace_first once_protocol ~pattern ~with_ in
    if String.equal variant once_protocol then
      Verdict.fail ("premise scan -- once variant: pattern not found: " ^ pattern);
    read variant
  in
  p "update_once, Ocannl-qualified and with options, is an executed reference too"
    (Scan.uncovered
       (once_variant ~pattern:"Train.forward_once (Context.auto ())"
          ~with_:"Ocannl.Train.update_once ~output_cd_file:false (Context.auto ())")
    = 0);
  p "a forward_once context read at another tensor confers nothing"
    (Scan.uncovered
       (once_variant ~pattern:"Context.get_values ctx t.Tensor.value"
          ~with_:"let u = build () in Context.get_values ctx u.Tensor.value")
    = 1);
  p "a forward_once context read at a rebound name confers nothing"
    (Scan.uncovered
       (once_variant ~pattern:"  p_all2 \"reference\""
          ~with_:"  let t = build () in\n  p_all2 \"reference\"")
    = 1);
  p "a compile without forward_once's run confers nothing"
    (Scan.uncovered
       (once_variant ~pattern:"Train.forward_once (Context.auto ()) t"
          ~with_:"fst (Context.compile (ignore t))")
    = 1);
  p "a rebound Train module confers nothing"
    (Scan.uncovered (read ("module Train = Other_train\n" ^ once_protocol)) = 1);
  let gradient ~before ~after =
    once_variant
      ~pattern:
        "  let ctx = Train.forward_once (Context.auto ()) t in\n\
        \  p_all2 \"reference\" (Context.get_values ctx t.Tensor.value)"
      ~with_:
        (before ^ "  let ctx = Train.update_once (Context.auto ()) t in\n" ^ after
       ^ "  p_all2 \"reference\" (Context.get_values ctx (grad_of t))")
  in
  p "a gradient materialized before update_once is an executed reference"
    (Scan.uncovered (gradient ~before:"  Train.set_materialized (grad_of t);\n" ~after:"") = 0);
  p "a gradient not materialized first confers nothing"
    (Scan.uncovered (gradient ~before:"" ~after:"") = 1);
  p "a gradient materialized only after the run confers nothing"
    (Scan.uncovered (gradient ~before:"" ~after:"  Train.set_materialized (grad_of t);\n") = 1);
  p "another node's materialization confers nothing"
    (Scan.uncovered
       (gradient ~before:"  let u = build () in\n  Train.set_materialized (grad_of u);\n" ~after:"")
    = 1);
  p "a materialized node whose name was rebound since confers nothing"
    (Scan.uncovered
       (gradient ~before:"  Train.set_materialized (grad_of t);\n  let t = build () in\n" ~after:"")
    = 1);
  let values_wrapper =
    once_variant
      ~pattern:
        "let () =\n\
        \  let t = build () in\n\
        \  let ctx = Train.forward_once (Context.auto ()) t in\n\
        \  p_all2 \"reference\" (Context.get_values ctx t.Tensor.value)"
      ~with_:
        "let values t =\n\
        \  let ctx = Train.forward_once (Context.auto ()) t in\n\
        \  Context.get_values ctx t.Tensor.value\n\
         let () =\n\
        \  p_all2 \"reference\" (values (build ()))"
  in
  p "a local wrapper reading its own forward_once tensor is an executed reference"
    (Scan.uncovered values_wrapper = 0)
