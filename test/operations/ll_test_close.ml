open Base
open Verdict.Claims

let () =
  p "two empty readbacks still match" (Ll_test.close [||] [||]);
  p "two empty readback lists still match" (Ll_test.same [] []);
  p "the tolerance boundary is inclusive" (Ll_test.close ~tol:0.25 [| 1.25 |] [| 1. |]);
  let holds, diagnostic =
    Stderr_capture.capture ~echo:false (fun () ->
        Ll_test.same ~tol:0.25
          [ [| 0. |]; [| 1.; 2.5; 3.; 4. |] ]
          [ [| 0. |]; [| 1.; 2.; 3.; 8. |] ])
  in
  p "a differing readback fails" (not holds);
  p "the shared check names both differing cells and the worst cell's values"
    (String.is_substring diagnostic ~substring:"2 of 4 cells differ"
    && String.is_substring diagnostic ~substring:"first at 1, worst at 3: got 4 vs want 8"
    && String.is_substring diagnostic ~substring:"max |want| 8");
  p "diagnostics identify the failing readback"
    (String.is_substring diagnostic ~substring:"Ll_test.close (readback 1)");
  let holds, diagnostic =
    Stderr_capture.capture ~echo:false (fun () -> Ll_test.close [| 1. |] [||])
  in
  p "a length mismatch fails with its lengths"
    ((not holds) && String.is_substring diagnostic ~substring:"length 1 vs 0");
  p "a mismatched readback list remains a test error"
    (try
       ignore (Ll_test.same [] [ [||] ] : bool);
       false
     with Invalid_argument _ -> true);
  let legacy, _ =
    Stderr_capture.capture ~echo:false (fun () ->
        (not (Ll_test.close [| Float.infinity |] [| Float.infinity |]))
        && (not (Ll_test.close ~tol:(-1.) [| 1. |] [| 1. |]))
        && Ll_test.close ~tol:Float.infinity [| Float.infinity |] [| 1. |])
  in
  p "legacy subtraction semantics also hold for infinities and negative tolerances" legacy
