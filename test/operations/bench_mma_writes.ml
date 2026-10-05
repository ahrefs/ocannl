(* gh-ocannl-1176 item 3: the shipped per-kernel table must attribute a Tile_mma's accumulator
   write. Keep zeroing in its own kernel: attributing the whole routine would hide the formerly
   empty MMA row behind the ordinary zeroing write. Two distinct outputs also catch attribution
   leaking between kernels/routines. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module H = Bench_harness

let compile ctx ~name output =
  let transform (opt : LL.optimized) =
    let axes = List.find_exn (Ll_test.nest_paths opt.llc) ~f:(fun p -> List.length p = 3) in
    let i, j, k = match axes with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let tensorize, _ = Ir.Schedule.tensorize ~i ~j ~k ~simd_width:32 () in
    let opt = Ir.Schedule.apply [ tensorize ] opt in
    LL.flat_lines [ opt.llc ]
    |> List.filter ~f:(function LL.Comment _ | LL.Noop -> false | _ -> true)
    |> List.map ~f:(fun llc -> { opt with llc })
  in
  Context.compile ~name ~lowered_transform:transform ctx (Train.forward output) Ir.Indexing.Empty

let census routines =
  let path = Stdlib.Filename.temp_file "bench_mma_writes" ".txt" in
  Exn.protect
    ~finally:(fun () -> Stdlib.Sys.remove path)
    ~f:(fun () ->
      Stdio.Out_channel.with_file path ~f:(fun out -> H.print_shipped_census ~out routines);
      Stdio.In_channel.read_all path)

let () =
  Verdict.case "shipped MMA write attribution" (fun () ->
      let input label modulus =
        TDSL.ndarray
          (Array.init (16 * 16)
             ~f:(Ll_test.cycle_flat ~dims:[| 16; 16 |] ~modulus ~offset:1. ~stride:0.25))
          ~label:[ label ] ~input_dims:[ 16 ] ~output_dims:[ 16 ] ()
      in
      let a = input "mma_writes_a" 13 and b = input "mma_writes_b" 17 in
      let%op c = a * b in
      let%op d = b * a in
      let ctx, rc = compile (Context.auto ()) ~name:"mma_writes_c" c in
      let ctx, rd = compile ctx ~name:"mma_writes_d" d in
      Exn.protect
        ~finally:(fun () -> Context.release ctx)
        ~f:(fun () ->
          let outputs = [ (rc, c.Tensor.value); (rd, d.Tensor.value) ] in
          List.iter outputs ~f:(fun (r, _) ->
              Stdio.eprintf "bench_mma_writes: %s %s\n%!" r.Context.name
                (Ir.C_syntax.mma_summary_string r.Context.mma));
          p "the constructed destinations are distinct"
            (not (Tn.equal c.Tensor.value d.Tensor.value));
          p_all "each routine ships separate zeroing and MMA kernels" outputs ~f:(fun (r, out) ->
              match List.map r.Context.segments ~f:(fun seg -> seg.LL.llc) with
              | [ LL.Zero_out zero; LL.For_loop { body = LL.Tile_mma { d = dest, _; _ }; _ } ] ->
                  Tn.equal zero out && Tn.equal dest out && r.Context.mma.statements = 1
              | _ -> false);
          p_all "every shipped kernel attributes exactly its constructed destination" outputs
            ~f:(fun (r, out) ->
              List.equal (List.equal Tn.equal)
                (List.map r.Context.segments ~f:(fun seg -> H.writes_of seg.LL.llc))
                [ [ out ]; [ out ] ]);
          let text = census [ rc; rd ] in
          Stdio.eprintf "bench_mma_writes: backend=%s\n%s%!"
            (Utils.get_global_arg ~arg_name:"backend" ~default:"auto")
            text;
          let rows =
            String.split_lines text
            |> List.filter_map ~f:(fun line ->
                Option.map (String.substr_index line ~pattern:" w:") ~f:(fun pos ->
                    String.drop_prefix line (pos + String.length " w:")))
          in
          let expected =
            List.concat_map outputs ~f:(fun (_, out) -> [ Tn.debug_name out; Tn.debug_name out ])
          in
          p "the printed per-kernel rows retain exact destination ownership in launch order"
            (List.equal String.equal rows expected)))
