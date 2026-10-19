(* gh-ocannl-1309: the tensorized-census reader's rule, put to synthesized sources -- each form it
   recognises beside the nearest text it must not read -- and the shipping scanner run on a
   synthetic tree holding a parity-only tensorized claim it must refuse. *)
open Base
open Stdio
module Scan = Test_utils.Tensorized_census_scan
open Verdict.Claims

let read source = List.length (Scan.tensorized_executed source)
let refused source = List.length (Scan.unjudged source)

(* The executed parity-only shape gh-ocannl-1283 found eleven times. *)
let parity_only =
  "let () =\n\
  \  let got = Ll_test.execute ~name:\"t\" o ~seed ~read:[ c ] in\n\
  \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want)\n"

let judged =
  "let () =\n\
  \  let got, census = Ll_test.execute_with_census ~name:\"t\" o ~seed ~read:[ c ] in\n\
  \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want);\n\
  \  Verdict.p \"the matmul's routine renders tensorized\" (Ll_test.rendered_tensorized census)\n"

let () =
  (* The rule. *)
  p "an executed parity-only tensorized claim is refused" (refused parity_only = 1);
  p "a census judgment in the same item answers it" (read judged = 1 && refused judged = 0);
  p "obtaining the census and discarding it is still parity-only"
    (refused
       "let () =\n\
       \  let got, _ = Ll_test.execute_with_census ~name:\"t\" o ~seed ~read:[ c ] in\n\
       \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want)\n"
    = 1);
  p "a census read only to print it is not a judgment"
    (refused
       "let () =\n\
       \  let ctx = Context.run ctx routine in\n\
       \  Stdio.eprintf \"%s\\n\" (Ir.C_syntax.mma_summary_string routine.Context.mma);\n\
       \  Verdict.p_all2 \"packed+tensorized matmul matches the serial twin bitwise\" (got ctx) \
        want ~f:Float.equal\n"
    = 1);
  (* The judgment forms. *)
  let with_judgment judgment =
    "let () =\n\
    \  let ctx = Context.run ctx routine in\n\
    \  Verdict.p \"tensorized matmul matches the serial twin\" (ok ctx);\n\
    \  Verdict.p \"rendered\" (" ^ judgment ^ ")\n"
  in
  p_all "every census predicate, constructor and field form is a judgment"
    [
      "Ll_test.all_tensorized ~what:\"x\" (Ll_test.renderings routine.Context.mma)";
      "Ll_test.census_agrees_with_capability ~backend_name ~what:\"x\" ~a_prec ~b_prec ~d_prec c";
      "Ir.C_syntax.equal_tensorization routine.Context.mma.tensorization Ir.C_syntax.Tensorized";
      "match census.tensorization with Tensorized -> true | _ -> false";
      "census.Ir.C_syntax.scalar_fallbacks = 0";
      "List.for_all (Ll_test.renderings c) ~f:Ir.C_syntax.is_tensorized_rendering";
      "Option.equal Ir.C_syntax.equal_tensorization r.Autotune.best_tensorization (Some \
       Ir.C_syntax.Tensorized)";
    ] ~f:(fun judgment -> read (with_judgment judgment) = 1 && refused (with_judgment judgment) = 0);
  p "a constructor of another module is not the census's"
    (refused (with_judgment "Poly.equal x Other.Tensorized") = 1);
  (* Reach through the file's own top-level bindings. *)
  (let source =
     "let renders census = Ll_test.all_tensorized ~what:\"x\" census\n"
     ^ with_judgment "renders census"
   in
   p "a top-level helper that judges counts for the item that references it"
     (read source = 1 && refused source = 0));
  p "a top-level helper that executes makes its caller an executing item"
    (refused
       "let run_it o = Ll_test.execute ~name:\"t\" o ~seed ~read:[ c ]\n\
        let () = Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal (run_it o) \
        want)\n"
    = 1);
  p "a judgment in ANOTHER item does not reach this one"
    (refused ("let () = Verdict.p \"x\" (Ll_test.rendered_tensorized census)\n" ^ parity_only) = 1);
  (* The executor forms, and their spellings. *)
  p_all "every executor form makes an item executing"
    [
      "Ll_test.run ~name:\"t\" o ~seed";
      "Ll_test.run_linked (ctx, routine) ~seed";
      "Ll_test.optimize_and_execute ~name:\"t\" llc ~seed ~read";
      "Train.forward_once ctx t";
      "Context.run ctx routine";
    ] ~f:(fun execution ->
      refused
        ("let () =\n  ignore (" ^ execution
       ^ ");\n  Verdict.p \"tensorized matmul matches the serial twin\" ok\n")
      = 1);
  p "a module alias of Ll_test is followed"
    (refused
       "module L = Test_utils.Ll_test\n\
        let () =\n\
       \  ignore (L.execute ~name:\"t\" o ~seed ~read);\n\
       \  Verdict.p \"tensorized matmul matches the serial twin\" ok\n"
    = 1);
  p "an open of Ll_test brings its executors and judgments into scope"
    (refused
       "open Ll_test\n\
        let () =\n\
       \  let got, census = execute_with_census ~name:\"t\" o ~seed ~read in\n\
       \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want)\n"
     = 1
    && refused
         "open Ll_test\n\
          let () =\n\
         \  let got, census = execute_with_census ~name:\"t\" o ~seed ~read in\n\
         \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want);\n\
         \  Verdict.p \"rendered\" (rendered_tensorized census)\n"
       = 0);
  p "a source's own binding shadowing an opened judgment is not the judgment"
    (refused
       "open Ll_test\n\
        let rendered_tensorized _ = true\n\
        let () =\n\
       \  let got, census = execute_with_census ~name:\"t\" o ~seed ~read in\n\
       \  Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want);\n\
       \  Verdict.p \"rendered\" (rendered_tensorized census)\n"
    = 1);
  p "a module the source defines under a library's name is the source's, not the library's"
    (refused
       ("module Ll_test = struct let rendered_tensorized _ = true end\n"
       ^ with_judgment "Ll_test.rendered_tensorized census")
     = 1
    && refused
         ("module C_syntax = struct type t = Tensorized end\n"
         ^ with_judgment "Poly.equal x C_syntax.Tensorized")
       = 1);
  (* Which claims are read. *)
  p "an item that executes nothing is not read"
    (read "let () = Verdict.p \"tensorized matmul matches the serial twin\" (Poly.equal got want)\n"
    = 0);
  p "a tensorized claim that asserts no parity is not read"
    (read
       "let () =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p \"the tensorized seeds are proposed\" ok;\n\
       \  Verdict.p_all \"every tensorized twin constructs and validates\" xs ~f\n"
    = 0);
  p "a parity claim whose label names no tensorization is not read"
    (read
       "let () =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p \"the comma-separated values match the reference\" ok\n"
    = 0);
  p_all "the mma word reads in Tile_mma, sk_mma, wmma and a tag"
    [ "the Tile_mma twin matches"; "sk_mma seeds match"; "qkv_mma: matches"; "wmma parity" ]
    ~f:(fun label ->
      read ("let () =\n  ignore (Context.run ctx routine);\n  Verdict.p \"" ^ label ^ "\" ok\n") = 1);
  p "the mma word does not read inside another word"
    (read
       "let () =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p \"the gemma summary matches\" ok\n"
    = 0);
  p "a label's fragments are followed through let-bound names"
    (refused
       "let () =\n\
       \  let tag = \"qkv_mma\" in\n\
       \  let label = tag ^ \": matches the reference\" in\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p label ok\n"
    = 1);
  p "a name the label binds itself shadows the outer binding of that name"
    (read
       "let () =\n\
       \  let tag = \"tensorized\" in\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p (let tag = \"scalar\" in tag ^ \" matmul matches the reference\") ok\n"
     = 0
    && read
         "let () =\n\
         \  ignore (Context.run ctx routine);\n\
         \  Verdict.p (let tag = \"tensorized\" in tag ^ \" matmul matches the reference\") ok\n"
       = 1);
  p "a pf format is the label, and pf_all2 and gated forms are claims"
    (refused
       "let () =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.pf \"%s tensorized matmul matches the serial twin\" tag ok;\n\
       \  Verdict.pf_all2 \"%s packed+tensorized parity\" tag got want ~f:Float.equal;\n\
       \  Verdict.gated ~when_:on_gpu ~on:backend_name \"tensorized matmul parity (GPU)\" ok\n"
    = 3);
  p "skipped, fail and a source's own p are not claims this reader reads"
    (read
       "let () =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.skipped ~backend:\"cc\" \"tensorized matmul matches the serial twin\";\n\
       \  Verdict.fail \"tensorized matmul matches the serial twin\";\n\
       \  let p _ _ = () in\n\
       \  p \"tensorized matmul matches the serial twin\" ok\n"
    = 0);
  p "boundary: a tensorization word arriving through a parameter is not read"
    (read
       "let leg what =\n\
       \  ignore (Context.run ctx routine);\n\
       \  Verdict.p (what ^ \" matmul matches the serial twin\") ok\n\
        let () = leg \"tensorized\"\n"
    = 0);
  (* The exemption rows. *)
  let rows = [ ("test/x.ml", parity_only) ] in
  let row label = { Scan.path = "test/x.ml"; label; reason = "r" } in
  let key = "\"tensorized matmul matches the serial twin\"" in
  p_empty "a row naming the claim's label absorbs it" ~over:rows
    (Scan.violations ~exemptions:[ row key ] rows);
  p "a row naming another label absorbs nothing, and is stale"
    (List.length (Scan.violations ~exemptions:[ row "\"other\"" ] rows) = 2);
  p "a row over a judged file is stale"
    (match Scan.violations ~exemptions:[ row key ] [ ("test/x.ml", judged) ] with
    | [ msg ] -> String.is_substring msg ~substring:"(stale)"
    | _ -> false);
  (* The shipping scanner on a synthetic tree: the negative control it must refuse. *)
  let exe = Stdlib.Sys.argv.(1) in
  let root = Stdlib.Filename.temp_dir "tensorized census control " "" in
  let write path data = Out_channel.write_all (Stdlib.Filename.concat root path) ~data in
  Unix.mkdir (Stdlib.Filename.concat root "test") 0o700;
  let run ?(exempt = false) () =
    Fresh_process.run ~exe [ (if exempt then "--fixture-exempt" else "--fixture"); root ]
  in
  let check label ~exit ~message child =
    let ok = Fresh_process.matches ~exit ~contains:[ message ] child in
    if not ok then Fresh_process.report ~label child;
    p label ok
  in
  write "test/judged.ml" judged;
  write "test/new.ml" parity_only;
  check "shipping scanner refuses the parity-only tensorized claim" ~exit:1
    ~message:"test/new.ml:3: the tensorized claim \"tensorized matmul matches the serial twin\""
    (run ());
  check "shipping scanner accepts the claim under a named exemption" ~exit:0
    ~message:"control exemption" (run ~exempt:true ());
  write "test/new.ml" judged;
  check "shipping scanner accepts the claim once its item judges the census" ~exit:0
    ~message:"judges the census: true" (run ());
  check "shipping scanner refuses the exemption the judgment left stale" ~exit:1
    ~message:"absorbs no unjudged claim (stale)" (run ~exempt:true ());
  Unix.unlink (Stdlib.Filename.concat root "test/new.ml");
  Unix.unlink (Stdlib.Filename.concat root "test/judged.ml");
  check "shipping scanner refuses a tree where it read no executed tensorized parity claim" ~exit:1
    ~message:"(empty): false" (run ());
  Unix.rmdir (Stdlib.Filename.concat root "test");
  Unix.rmdir root
