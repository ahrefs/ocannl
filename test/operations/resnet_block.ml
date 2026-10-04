open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let input_value b h w c =
  1. +. Float.of_int b
  +. (0.25 *. Float.of_int h)
  +. (0.125 *. Float.of_int w)
  +. (0.5 *. Float.of_int c)

let run tag ~stride ?out_channels () =
  Verdict.case tag (fun () ->
      Tensor.unsafe_reinitialize ();
      let x =
        NTDSL.reshape ~l:(tag ^ "_input") ~b:[ 2 ] ~o:[ 4; 4; 2 ]
          (Ir.Ndarray.init_array ~debug:tag Ir.Ops.single ~dims:[| 2; 4; 4; 2 |] ~padding:None
             ~f:(fun ij -> input_value ij.(0) ij.(1) ij.(2) ij.(3)))
          ()
      in
      let block = Nn_blocks.resnet_block ~label:[ tag ] ~stride ?out_channels () in
      let y = block ~train_step:None x in
      Train.set_materialized y.Tensor.value;
      Ir.Tnode.set_observable y.Tensor.value;
      let ctx = Context.auto () in
      let ctx = Train.init_params ctx Ir.Indexing.Empty y in
      (* Zero the main branch. The projection sums input channels, then its batch norm uses unit
         gamma; the unprojected shortcut retains the positive input. *)
      let ctx =
        Set.fold y.Tensor.params ~init:ctx ~f:(fun ctx p ->
            let tn = p.Tensor.value in
            let name = Ir.Tnode.debug_name tn in
            let value =
              if
                String.is_prefix name ~prefix:"kernel_downsample_"
                || String.is_prefix name ~prefix:"gamma_downsample_bn_"
              then 1.
              else 0.
            in
            let count = Array.fold (Lazy.force tn.Ir.Tnode.dims) ~init:1 ~f:( * ) in
            Context.set_values ctx tn (Array.create ~len:count value))
      in
      let ctx, routine = Context.compile ctx (Train.forward y) Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let values = Context.get_values ctx y.Tensor.value in
      let side = 4 / stride and channels = Option.value out_channels ~default:2 in
      pf "%s output dimensions" tag
        (Array.equal Int.equal
           (Lazy.force y.Tensor.value.Ir.Tnode.dims)
           [| 2; side; side; channels |]);
      let projected = stride > 1 || Option.is_some out_channels in
      let samples =
        Array.init
          (2 * side * side)
          ~f:(fun i ->
            let b = i / (side * side) and h = i / side % side and w = i % side in
            input_value b (stride * h) (stride * w) 0 +. input_value b (stride * h) (stride * w) 1)
      in
      let mean = Array.fold samples ~init:0. ~f:( +. ) /. Float.of_int (Array.length samples) in
      let variance =
        Array.fold samples ~init:0. ~f:(fun acc v -> acc +. ((v -. mean) **. 2.))
        /. Float.of_int (Array.length samples)
      in
      p_alli (tag ^ " executed shortcut matches oracle") (Array.to_list values) ~f:(fun i actual ->
          let cell = i / channels in
          let expected =
            if projected then
              Float.max 0. ((samples.(cell) -. mean) /. Float.sqrt (variance +. 1e-5))
            else input_value (cell / (side * side)) (cell / side % side) (cell % side) (i % channels)
          in
          if Float.(abs (actual -. expected) >= 1e-4) then
            Stdio.eprintf "%s cell %d: actual %f expected %f (not part of the golden)\n" tag i
              actual expected;
          Float.is_finite actual && Float.(abs (actual -. expected) < 1e-4));
      Context.release ctx)

let () =
  run "identity" ~stride:1 ();
  run "downsample" ~stride:2 ();
  run "channel_change" ~stride:1 ~out_channels:3 ();
  run "downsample_channel_change" ~stride:2 ~out_channels:3 ()
