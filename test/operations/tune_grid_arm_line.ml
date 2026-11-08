(* gh-ocannl-1902: pin the Grid census suffix from a real placement arm, without copying its
   formatter or putting measured timings in the golden. The exact label pins [i3], from the
   process-wide symbol counter: a symbol-renumbering change needs a source-label edit as well as a
   golden update, since a failed Verdict claim cannot be promoted. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let capture_suffix () =
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
          ~settings:[ ("cc_grid_min_updates", "16384") ]
          ~measure:(fun _ -> 1.)
          (fun () ->
            let ctx, _ =
              Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
                ~placement_store:false ~inline_flips:0 ~log:true ~progress:false (Context.auto ()) c
                comp Ir.Indexing.Empty
            in
            Context.release ctx))
  in
  let arm =
    List.find lines
      ~f:(String.is_prefix ~prefix:"tune_placements: arm A (default placements) best:")
  in
  Option.bind arm ~f:(fun line ->
      Option.bind (String.substr_index line ~pattern:"[grid:") ~f:(fun start ->
          let rest = String.drop_prefix line start in
          Option.map (String.index rest ']') ~f:(fun stop -> String.prefix rest (stop + 1))))

let () =
  Scripted_search.with_settings [ ("cc_parallel_chunks", "2") ] @@ fun () ->
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"") in
  let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend) in
  let backend_leg = Verdict.leg ~when_:on_cpu ~on:backend () in
  let leg =
    Verdict.sub_leg backend_leg ~aggregation:`Environment
      ~when_:(lazy (Context.Cc_backend.pool_parallel_grid ()))
      ~on:"cc without a native pool"
  in
  in_leg ~leg
    "[grid: 0 of 1 on the pool; i3 serial: native fork/join has only 32 estimated scalar updates \
     per launch, below cc_grid_min_updates = 16384]" (fun expected ->
      p expected (Option.value_map (capture_suffix ()) ~default:false ~f:(String.equal expected)))
