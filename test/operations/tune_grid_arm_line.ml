(* gh-ocannl-1902: pin the Grid census suffix from a real placement arm, without copying its
   formatter or putting measured timings in the golden. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let () =
  let a =
    TDSL.ndarray (Array.init 32 ~f:Float.of_int) ~label:[ "grid_arm_a" ] ~output_dims:[ 32 ] ()
  in
  let%op c = a + a in
  let comp = Train.forward c in
  let ((), _), lines =
    Stderr_capture.lines (fun () ->
        Scripted_search.scripted
        (* Equal scripted windows keep the initial Grid preset crowned. The tiny pointwise loop is
           below the pinned fork/join floor. *)
          ~settings:[ ("cc_parallel_chunks", "2"); ("cc_grid_min_updates", "16384") ]
          ~measure:(fun _ -> 1.)
          (fun () ->
            let ctx, _ =
              Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
                ~placement_store:false ~inline_flips:0 ~log:true ~progress:false (Context.cpu ()) c
                comp Ir.Indexing.Empty
            in
            Context.release ctx))
  in
  let arm =
    List.find lines
      ~f:(String.is_prefix ~prefix:"tune_placements: arm A (default placements) best:")
  in
  let suffix =
    Option.bind arm ~f:(fun line ->
        Option.bind (String.substr_index line ~pattern:"[grid:") ~f:(fun start ->
            let rest = String.drop_prefix line start in
            Option.map (String.index rest ']') ~f:(fun stop -> String.prefix rest (stop + 1))))
  in
  p "arm A reports a Grid census" (Option.is_some suffix);
  Option.iter suffix ~f:Stdio.print_endline
