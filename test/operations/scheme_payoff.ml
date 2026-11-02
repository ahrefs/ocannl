(* gh-ocannl-1752: what shape schemes buy in inference cost, on stacks where one block repeats N
   times, under op_shape_schemes=off|check. Each workload is built in a fresh session; the closure
   that ends it is forced, and the registration statistics say what each source added to the
   constraint store and where the shape-registration time went. Nothing is compiled.

   The deterministic counts are claimed (stdout); the full count and timing tables go to stderr,
   which is not part of the golden. [--bench REPS N...] times the workloads instead: the measurement
   behind docs/shape_inference.md's numbers; [--bench-result N...] is its form for the A/B
   driver. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Nn = Ocannl.Nn_blocks

type mode = [ `Off | `Check ]

let modes : mode list = [ `Off; `Check ]
let mode_name = function `Off -> "off" | `Check -> "check"
let d_model = 16

(* The repeated block: schemed at its [()] stage, made of schemed nn_blocks layers. *)
let%op block ~label () =
  let ffn = Nn.mlp_layer ~label:("ffn" :: label) ~hid_dim:d_model () in
  let ln = Nn.layer_norm ~label:("ln" :: label) () in
  fun x -> ln (x + ffn x)

let input () =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ "x" ] ~batch_dims:[ 8 ] ~output_dims:[ d_model ] ()

(* A workload builds a model of depth [n] on a fresh input; it returns the result. *)
type workload = { w_name : string; build : int -> Tensor.t }

let workloads =
  [
    {
      w_name = "distinct";
      (* [n] blocks, each its own closure (own parameters) applied once: a plain deep stack. *)
      build =
        (fun n ->
          let layers = List.init n ~f:(fun i -> block ~label:[ "l" ^ Int.to_string i ] ()) in
          List.fold layers ~init:(input ()) ~f:(fun x layer -> layer x));
    };
    {
      w_name = "shared";
      (* One closure applied [n] times: a weight-tied stack (an unrolled recurrence). *)
      build =
        (fun n ->
          let layer = block ~label:[ "tied" ] () in
          Fn.apply_n_times ~n layer (input ()));
    };
    {
      w_name = "decoder";
      (* nn_blocks' decoder-only transformer: its blocks stay unschemed (~train_step and ~mask come
         after (), gh-ocannl-1792), their layer norms and feed-forward layers are schemed. *)
      build =
        (fun n ->
          let model =
            Nn.decoder_only ~label:[ "dec" ] ~num_layers:n ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
          in
          let x =
            Tensor.term ~grad_spec:Prohibit_grad ~label:[ "x" ] ~batch_dims:[ 2; 8 ]
              ~output_dims:[ d_model ] ()
          in
          let mask =
            NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ 8 ] ~i:[ 8 ] ~o:[]
              ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> 0.)
              ()
          in
          model ~train_step:None x ~mask);
    };
  ]

type run = {
  counts : Shape.registration_counts;
  times : Shape.registration_times;
  construction_s : float;
  total_s : float;
  dims : int list list;  (** The closed dims of the result and of every parameter. *)
}

let seconds started = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9

(* [~unbounded]: with the rank-cycle search walking everything a new edge's target reaches, as
   before gh-ocannl-1802 bounded it by the edge's endpoints: the reference for its work counter. *)
let run_one ?(unbounded = false) (mode : mode) w n =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Tensor.Scheme.reset_registration_counts ();
  Tensor.Scheme.reset_registration_times ();
  let started = Mtime_clock.counter () in
  let infer () =
    let y = w.build n in
    let construction_s = seconds started in
    (y, construction_s, Array.to_list (Shape.to_dims y.Tensor.shape))
  in
  let y, construction_s, result_dims =
    if unbounded then Ocannl.Row.with_unbounded_rank_search infer else infer ()
  in
  let total_s = seconds started in
  let params =
    Set.to_list y.Tensor.params
    |> List.map ~f:(fun p -> Array.to_list (Shape.to_dims p.Tensor.shape))
  in
  {
    counts = Tensor.Scheme.registration_counts ();
    times = Tensor.Scheme.registration_times ();
    construction_s;
    total_s;
    dims = result_dims :: params;
  }

let row_header =
  "workload depth mode | checked inline | C' E' boundary inline_c | closure_constraints | \
   rank_search_visits"

let count_row w n mode (c : Shape.registration_counts) =
  Printf.sprintf "%s %d %s | %d %d | %d %d %d %d | %d | %d" w n (mode_name mode)
    c.checked_applications c.inline_registrations c.instantiated_constraints
    c.instantiated_solutions c.instantiated_boundary c.inline_constraints c.closure_constraints
    c.rank_search_visits

let ms x = Printf.sprintf "%.2f" (x *. 1000.)

let time_header =
  "workload depth mode | construction_ms closure_ms total_ms | tracing instantiation inline_steps \
   (ms)"

let time_row w n mode r =
  let t = r.times in
  String.concat ~sep:" "
    [
      w;
      Int.to_string n;
      mode_name mode;
      "|";
      ms r.construction_s;
      ms t.closure_s;
      ms r.total_s;
      "|";
      ms t.tracing_s;
      ms t.instantiation_s;
      ms t.inline_steps_s;
    ]

(* The test's depths: the counts' relationships show by depth 4; the measurement goes deeper. *)
let depths = [ 1; 2; 4 ]
let find_workload name = List.find_exn workloads ~f:(fun x -> String.equal x.w_name name)

let median xs =
  let a = List.sort xs ~compare:Float.compare |> Array.of_list in
  a.(Array.length a / 2)

(* The measurement: per workload, depth and mode, one warm-up run and the median of [reps] runs
   (each column's own median), with the range of the total. *)
let bench reps cells =
  Stdio.printf "%s | total range (ms)\n%!" time_header;
  let counts = ref [] in
  List.iter cells ~f:(fun (w, ns) ->
      List.iter ns ~f:(fun n ->
          List.iter modes ~f:(fun mode ->
              ignore (run_one mode w n : run);
              let rs = List.init reps ~f:(fun _ -> run_one mode w n) in
              let med f = median (List.map rs ~f) in
              let r =
                {
                  (List.hd_exn rs) with
                  construction_s = med (fun r -> r.construction_s);
                  total_s = med (fun r -> r.total_s);
                  times =
                    {
                      tracing_s = med (fun r -> r.times.tracing_s);
                      instantiation_s = med (fun r -> r.times.instantiation_s);
                      inline_steps_s = med (fun r -> r.times.inline_steps_s);
                      closure_s = med (fun r -> r.times.closure_s);
                    };
                }
              in
              counts := count_row w.w_name n mode r.counts :: !counts;
              let totals = List.map rs ~f:(fun r -> r.total_s) in
              let lo = List.reduce_exn totals ~f:Float.min
              and hi = List.reduce_exn totals ~f:Float.max in
              Stdio.printf "%s | total range %s-%s\n%!" (time_row w.w_name n mode r) (ms lo) (ms hi))));
  Stdio.printf "%s\n" row_header;
  List.iter (List.rev !counts) ~f:(Stdio.printf "%s\n")

(* The A/B driver's form of the measurement (benchmarks/ab.py, bin_cells.py): per workload and
   depth, one warm-up and one timed build with schemes off, as a [bench-result] line. *)
let bench_result cells =
  (* Nothing here is compiled, but the A/B driver checks the backend the cell resolved: read it. *)
  Stdio.eprintf "scheme_payoff: backend %s\n%!" (Context.backend_name (Context.auto ()));
  List.iter cells ~f:(fun (w, ns) ->
      List.iter ns ~f:(fun n ->
          ignore (run_one `Off w n : run);
          let r = run_one `Off w n in
          Stdio.printf "%s\n" (time_row w.w_name n `Off r);
          Bench_out.result
            ~variant:(Printf.sprintf "%s-%d-off" w.w_name n)
            ~check:(Unchecked "an inference timing: the test claims the shapes it closes")
            ~seconds:r.total_s))

let parse_cells =
  List.map ~f:(fun cell ->
      match String.lsplit2 cell ~on:'=' with
      | Some (w, ns) -> (find_workload w, List.map (String.split ns ~on:',') ~f:Int.of_string)
      | None -> failwith ("scheme_payoff: expected workload=depths, got " ^ cell))

let () =
  (* The A/B driver's cells lead with their [--ocannl_*] pins: settings, not this program's mode. *)
  let argv =
    List.filter
      (Array.to_list (Sys.get_argv ()))
      ~f:(fun a -> not (String.is_prefix a ~prefix:"--ocannl_"))
  in
  match argv with
  | [ _; "--one"; w; n; mode ] ->
      (* One run, for profiling. *)
      let w = find_workload w and n = Int.of_string n in
      let mode = List.find_exn modes ~f:(fun m -> String.equal (mode_name m) mode) in
      let r = run_one mode w n in
      Stdio.printf "%s\n%s\n" (count_row w.w_name n mode r.counts) (time_row w.w_name n mode r)
  | _ :: "--bench" :: reps :: cells ->
      (* [--bench REPS distinct=1,4,16 shared=1,4,16 decoder=1,4,8] *)
      bench (Int.of_string reps) (parse_cells cells)
  | _ :: "--bench-result" :: cells -> bench_result (parse_cells cells)
  | _ ->
      let results =
        List.concat_map workloads ~f:(fun w ->
            List.concat_map depths ~f:(fun n ->
                List.map modes ~f:(fun mode -> ((w.w_name, n, mode), run_one mode w n))))
      in
      let unbounded =
        List.concat_map workloads ~f:(fun w ->
            List.map depths ~f:(fun n -> ((w.w_name, n), run_one ~unbounded:true `Off w n)))
      in
      Tensor.For_testing.set_op_shape_schemes None;
      Stdio.eprintf "(not part of the golden) %s\n" row_header;
      List.iter results ~f:(fun ((w, n, mode), r) ->
          Stdio.eprintf "(not part of the golden) %s\n" (count_row w n mode r.counts));
      List.iter unbounded ~f:(fun ((w, n), r) ->
          Stdio.eprintf "(not part of the golden) unbounded rank search: %s\n"
            (count_row w n `Off r.counts));
      Stdio.eprintf "(not part of the golden) %s\n" time_header;
      List.iter results ~f:(fun ((w, n, mode), r) ->
          Stdio.eprintf "(not part of the golden) %s\n" (time_row w n mode r));
      let get w n mode = List.Assoc.find_exn results (w, n, mode) ~equal:Poly.equal in
      let cells = List.filter results ~f:(fun ((_, _, mode), _) -> Poly.equal mode `Off) in
      Verdict.case "what each mode adds to inference" (fun () ->
          p_all "every mode closes the shapes it closes without schemes" results
            ~f:(fun ((w, n, _), r) -> List.equal (List.equal Int.equal) r.dims (get w n `Off).dims);
          p_all "checking leaves the closure the constraint store it has without schemes" cells
            ~f:(fun ((w, n, _), off) ->
              (get w n `Check).counts.closure_constraints = off.counts.closure_constraints);
          p_all "checking leaves the body's inline registrations those it has without schemes" cells
            ~f:(fun ((w, n, _), off) ->
              let c = (get w n `Check).counts in
              c.inline_registrations = off.counts.inline_registrations
              && c.inline_constraints = off.counts.inline_constraints);
          p_all "checking instantiates a scheme at every depth" cells ~f:(fun ((w, n, _), _) ->
              let c = (get w n `Check).counts in
              c.checked_applications > 0 && c.instantiated_constraints > 0));
      (* gh-ocannl-1802: the rank-cycle search bounded by the new edge's endpoints' levels, against
         the unbounded walk it replaced. *)
      let unbounded_visits w n =
        (List.Assoc.find_exn unbounded (w, n) ~equal:Poly.equal).counts.rank_search_visits
      in
      let saving w n = unbounded_visits w n - (get w n `Off).counts.rank_search_visits in
      Verdict.case "the rank-cycle search, bounded by the new edge's endpoints" (fun () ->
          p_all "the bounded search closes the shapes the unbounded one closes" unbounded
            ~f:(fun ((w, n), r) -> List.equal (List.equal Int.equal) r.dims (get w n `Off).dims);
          p_all "the bounded search visits fewer states than the unbounded one" unbounded
            ~f:(fun ((w, n), _) -> saving w n > 0);
          let deeper =
            List.cartesian_product workloads
              (List.zip_exn (List.drop_last_exn depths) (List.tl_exn depths))
          in
          p_all "the bounded search's saving grows with depth" deeper ~f:(fun (w, (n, n')) ->
              saving w.w_name n < saving w.w_name n'))
