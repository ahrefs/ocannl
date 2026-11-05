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

(* The transformer workloads' input, causal mask and training step counter. *)
let seq_input () =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ "x" ] ~batch_dims:[ 2; 8 ] ~output_dims:[ d_model ]
    ()

let causal_mask () =
  NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ 8 ] ~i:[ 8 ] ~o:[]
    ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> 0.)
    ()

let step () =
  let step_n, _ = Ocannl.Train.IDX.get_static_symbol Ocannl.Train.IDX.empty in
  NTDSL.embed_symbol step_n

let decoder ~train_step n =
  let model =
    Nn.decoder_only ~label:[ "dec" ] ~num_layers:n ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
  in
  model ~train_step (seq_input ()) ~mask:(causal_mask ())

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
    (* nn_blocks' transformer stacks, whose blocks are schemed per presence pattern of [~train_step]
       and attention's [?mask] (gh-ocannl-1792): evaluation and training of the decoder-only stack
       (masked attention), and training of the encoder stack (unmasked). *)
    { w_name = "decoder"; build = (fun n -> decoder ~train_step:None n) };
    { w_name = "decoder_train"; build = (fun n -> decoder ~train_step:(Some (step ())) n) };
    {
      w_name = "encoder_train";
      build =
        (fun n ->
          let model =
            Nn.transformer_encoder ~label:[ "enc" ] ~num_layers:n ~num_heads:2 ~d_k:8 ~d_v:8
              ~d_ff:32 ()
          in
          model ~train_step:(Some (step ())) (seq_input ()));
    };
  ]

type run = {
  counts : Shape.registration_counts;
  times : Shape.registration_times;
  construction_s : float;
  total_s : float;
  dims : int list list;  (** The closed dims of the result and of every parameter. *)
  checked : string list;  (** The labels of the schemes the applications were checked against. *)
  alloc_words : float option;
      (** With [~alloc]: the heap words construction and closure allocated (gh-ocannl-1904). *)
}

let seconds started = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9

(* [~unbounded]: with the rank-cycle search walking everything a new edge's target reaches, as
   before gh-ocannl-1802 bounded it by the edge's endpoints: the reference for its work counter.
   [~whole_env]: with every solution substituted into the whole environment, as before
   gh-ocannl-1904 indexed the entries it can change. [~alloc]: counting the heap words construction
   and closure allocate, in a window opened after the earlier garbage's finalisers ran. *)
let run_one ?(unbounded = false) ?(whole_env = false) ?(alloc = false) (mode : mode) w n =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Tensor.Scheme.reset_registration_counts ();
  Tensor.Scheme.reset_registration_times ();
  let started = Mtime_clock.counter () in
  let checked = ref [] in
  let infer () =
    let y =
      Tensor.For_testing.observe_schemes
        (fun () -> w.build n)
        ~on_event:(function Tensor.Scheme.Checked l -> checked := l :: !checked | _ -> ())
    in
    let construction_s = seconds started in
    (y, construction_s, Array.to_list (Shape.to_dims y.Tensor.shape))
  in
  let infer = if unbounded then fun () -> Ocannl.Row.with_unbounded_rank_search infer else infer in
  let infer = if whole_env then fun () -> Ocannl.Row.with_whole_env_substitution infer else infer in
  let (y, construction_s, result_dims), alloc_words =
    if alloc then
      let result, { Gc_witness.words; drained = _ } = Gc_witness.drained_finalisers infer in
      (result, Some words)
    else (infer (), None)
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
    checked = List.dedup_and_sort !checked ~compare:String.compare;
    alloc_words;
  }

(* gh-ocannl-1891: one trace of a whole stack, the scheme nn_blocks' stacks leave off. The closure
   is built before the window, in a fresh session, so the window holds the trace alone: the body on
   unsized arguments, its end validation and the scheme's export, not a build's construction and
   closure. [args] are concrete arguments for one build of the same model. *)
type stack = {
  s_name : string;
  depths : int list;
  arity : int;
  model : int -> Tensor.t list -> Tensor.t;
  args : unit -> Tensor.t list;
}

let on_x_mask f = function [ x; mask ] -> f x mask | _ -> assert false
let on_x f = function [ x ] -> f x | _ -> assert false
let decoder_args () = [ seq_input (); causal_mask () ]

let decoder_stack ~train_step n =
  let model =
    Nn.decoder_only ~label:[ "dec" ] ~num_layers:n ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
  in
  on_x_mask (fun x mask -> model ~train_step:(train_step ()) x ~mask)

let stack_depths = [ 2; 4; 8; 16 ]

let stacks =
  [
    {
      s_name = "decoder";
      depths = stack_depths;
      arity = 2;
      model = decoder_stack ~train_step:(fun () -> None);
      args = decoder_args;
    };
    {
      s_name = "decoder_train";
      depths = stack_depths;
      arity = 2;
      model =
        (fun n ->
          let step = step () in
          decoder_stack ~train_step:(fun () -> Some step) n);
      args = decoder_args;
    };
    (* One closure applied [n] times: depth without the captures growing with it. *)
    {
      s_name = "decoder_tied";
      depths = stack_depths;
      arity = 2;
      model =
        (fun n ->
          let layer =
            Nn.decoder_only_block ~label:[ "tied" ] ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
          in
          on_x_mask (fun x mask -> Fn.apply_n_times ~n (fun x -> layer ~train_step:None x ~mask) x));
      args = decoder_args;
    };
    (* Convolutions and batch norms, whose rows stay unsolved longer than attention's. *)
    {
      s_name = "resnet";
      depths = [ 2; 4; 8 ];
      arity = 1;
      model =
        (fun n ->
          let layers =
            List.init n ~f:(fun i ->
                Nn.resnet_block ~label:[ "r" ^ Int.to_string i ] ~out_channels:8 ())
          in
          on_x (fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step:None x)));
      args =
        (fun () ->
          [
            Tensor.term ~grad_spec:Prohibit_grad ~label:[ "img" ] ~batch_dims:[ 2 ]
              ~output_dims:[ 32; 32; 8 ] ();
          ]);
    };
  ]

type stack_run = {
  t_counts : Shape.registration_counts;
  t_words : float;  (** The heap words the window allocated. *)
  t_s : float;  (** The window's time: on stderr, never claimed. *)
  t_dims : int list list;  (** The closed dims of the result and of every parameter. *)
}

let closed_dims (y : Tensor.t) =
  Array.to_list (Shape.to_dims y.shape)
  :: (Set.to_list y.params |> List.map ~f:(fun p -> Array.to_list (Shape.to_dims p.Tensor.shape)))

(* One build of the stack without schemes: what a trace is held against. *)
let stack_build s n =
  Tensor.For_testing.set_op_shape_schemes (Some `Off);
  Tensor.unsafe_reinitialize ();
  Tensor.Scheme.reset_registration_counts ();
  let started = Mtime_clock.counter () in
  let dims, { Gc_witness.words; drained = _ } =
    Gc_witness.drained_finalisers (fun () -> closed_dims (s.model n (s.args ())))
  in
  {
    t_counts = Tensor.Scheme.registration_counts ();
    t_words = words;
    t_s = seconds started;
    t_dims = dims;
  }

(* One trace, then the scheme applied to the concrete arguments, so that a trace that lost a
   constraint or a solution shows in what closes. [~unmemoized]: with the dimension-cycle search as
   it was before gh-ocannl-1891. *)
let stack_trace ?(unmemoized = false) s n =
  Tensor.For_testing.set_op_shape_schemes (Some `Check);
  Tensor.unsafe_reinitialize ();
  let model = s.model n in
  Tensor.Scheme.reset_registration_counts ();
  let started = ref (Mtime_clock.counter ()) in
  let trace () =
    started := Mtime_clock.counter ();
    Tensor.Scheme.trace ~label:s.s_name s.arity (fun args -> [ model args ])
  in
  let trace =
    if unmemoized then fun () -> Ocannl.Row.with_unmemoized_cycle_search trace else trace
  in
  let scheme, { Gc_witness.words; drained = _ } = Gc_witness.drained_finalisers trace in
  let t_s = seconds !started in
  let t_counts = Tensor.Scheme.registration_counts () in
  let args = s.args () in
  let y = model args in
  Tensor.Scheme.instantiate scheme
    ~args:(List.map args ~f:(fun t -> t.Tensor.shape))
    ~results:[ y.shape ];
  { t_counts; t_words = words; t_s; t_dims = closed_dims y }

let row_header =
  "workload depth mode | checked inline | C' E' boundary inline_c | closure_constraints | \
   rank_search_visits dim_cycle_visits subst_visits"

let count_row w n mode (c : Shape.registration_counts) =
  Printf.sprintf "%s %d %s | %d %d | %d %d %d %d | %d | %d %d %d" w n (mode_name mode)
    c.checked_applications c.inline_registrations c.instantiated_constraints
    c.instantiated_solutions c.instantiated_boundary c.inline_constraints c.closure_constraints
    c.rank_search_visits c.dim_cycle_visits c.subst_visits

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

(* gh-ocannl-1904's scaling cells: deep enough for the whole environment's quadratic term to show in
   each doubling. *)
let scaling_cells =
  [ ("distinct", [ 4; 8; 16 ]); ("shared", [ 4; 8; 16 ]); ("decoder", [ 2; 4; 8 ]) ]

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
      (* gh-ocannl-1904: deeper stacks, schemes off, with the allocation counted, indexed and with
         every solution substituted into the whole environment (the control). *)
      let scaling =
        List.concat_map scaling_cells ~f:(fun (w, ns) ->
            let w = find_workload w in
            List.concat_map ns ~f:(fun n ->
                List.map [ false; true ] ~f:(fun whole_env ->
                    ((w.w_name, n, whole_env), run_one ~whole_env ~alloc:true `Off w n))))
      in
      (* gh-ocannl-1891: whole-stack traces, and one build of each model without schemes. *)
      let traced =
        List.concat_map stacks ~f:(fun s ->
            List.map s.depths ~f:(fun n -> ((s.s_name, n), (stack_trace s n, stack_build s n))))
      in
      (* The dimension-cycle walk's control: depths 2 and 3 of the decoder only, since it grows by
         tens of times per layer. *)
      let decoder_stack_cell = List.hd_exn stacks in
      let walk_depths = [ 2; 3 ] in
      let cycle_walk =
        List.map walk_depths ~f:(fun n ->
            ( n,
              (stack_trace decoder_stack_cell n, stack_trace ~unmemoized:true decoder_stack_cell n)
            ))
      in
      Tensor.For_testing.set_op_shape_schemes None;
      Stdio.eprintf "(not part of the golden) %s\n" row_header;
      List.iter results ~f:(fun ((w, n, mode), r) ->
          Stdio.eprintf "(not part of the golden) %s\n" (count_row w n mode r.counts));
      List.iter unbounded ~f:(fun ((w, n), r) ->
          Stdio.eprintf "(not part of the golden) unbounded rank search: %s\n"
            (count_row w n `Off r.counts));
      List.iter scaling ~f:(fun ((w, n, whole_env), r) ->
          Stdio.eprintf "(not part of the golden) %s substitution: %s | %.0f words\n"
            (if whole_env then "whole-env" else "indexed")
            (count_row w n `Off r.counts) (Option.value_exn r.alloc_words));
      List.iter traced ~f:(fun ((w, n), (t, b)) ->
          Stdio.eprintf
            "(not part of the golden) whole-stack trace: %s | %.0f words %.1f ms | build %.0f \
             words %.1f ms | trace/build %.3f\n"
            (count_row w n `Check t.t_counts) t.t_words (t.t_s *. 1000.) b.t_words (b.t_s *. 1000.)
            (t.t_words /. b.t_words));
      List.iter cycle_walk ~f:(fun (n, (t, u)) ->
          Stdio.eprintf
            "(not part of the golden) dimension-cycle search at depth %d: %d visits, walking paths \
             %d\n"
            n t.t_counts.dim_cycle_visits u.t_counts.dim_cycle_visits);
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
              c.checked_applications > 0 && c.instantiated_constraints > 0);
          (* The transformer blocks' own schemes, by pattern: the counts above are already positive
             through the layer norms and feed-forward layers alone. *)
          let checks_at_every_depth w labels =
            p_all
              (w ^ ": checking reaches " ^ String.concat ~sep:", " labels)
              depths
              ~f:(fun n ->
                List.for_all labels ~f:(fun l ->
                    List.mem (get w n `Check).checked l ~equal:String.equal))
          in
          checks_at_every_depth "decoder"
            [ "decoder_only_block [no train_step]"; "multi_head_attention [no train_step, mask]" ];
          checks_at_every_depth "decoder_train"
            [ "decoder_only_block [train_step]"; "multi_head_attention [train_step, mask]" ];
          checks_at_every_depth "encoder_train"
            [
              "transformer_encoder_block [train_step]"; "multi_head_attention [train_step, no mask]";
            ];
          p_all "schemes off: no application is checked" results ~f:(fun ((_, _, mode), r) ->
              Poly.equal mode `Check || List.is_empty r.checked));
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
              saving w.w_name n < saving w.w_name n'));
      (* gh-ocannl-1904: a solution visits the entries it can change, not the whole environment. Per
         depth doubling, the indexed substitution's work doubles and the inference's allocation
         stays well below the whole environment's growth, which the control shows near 4x. The
         rank-cycle search still grows about 4x per doubling, which is what keeps the allocation's
         ratio above 2 here and is why it is bounded only to the depths below. *)
      let scaled w n whole_env = List.Assoc.find_exn scaling (w, n, whole_env) ~equal:Poly.equal in
      let doublings =
        List.concat_map scaling_cells ~f:(fun (w, ns) ->
            List.map
              (List.zip_exn (List.drop_last_exn ns) (List.tl_exn ns))
              ~f:(fun (n, n') -> (w, n, n')))
      in
      let ratio f w n n' whole_env = f (scaled w n' whole_env) /. f (scaled w n whole_env) in
      let visits r = Float.of_int r.counts.subst_visits in
      let words r = Option.value_exn r.alloc_words in
      (* Under the cross-check setting every solution also visits the whole environment, so the
         allocation is the whole environment's again: those claims need the setting off. *)
      let leg =
        Verdict.leg ~aggregation:`Environment
          ~when_:(not (Ocannl.Row.substitution_cross_checked ()))
          ~on:"shape_substitution_cross_check=true" ()
      in
      Verdict.case "a solution substitutes itself into the entries it can change" (fun () ->
          p_all "indexed substitution closes the shapes whole-environment substitution closes"
            scaling ~f:(fun ((w, n, _), r) ->
              List.equal (List.equal Int.equal) r.dims (scaled w n true).dims);
          p_all "indexed substitution visits fewer entries than whole-environment substitution"
            doublings ~f:(fun (w, _, n') ->
              (scaled w n' false).counts.subst_visits < (scaled w n' true).counts.subst_visits);
          p_all "per depth doubling, the indexed substitution's visits double (1.8x-2.2x)" doublings
            ~f:(fun (w, n, n') -> Float.between (ratio visits w n n' false) ~low:1.8 ~high:2.2);
          p_all "per depth doubling, whole-environment substitution's visits grow at least 3x"
            doublings ~f:(fun (w, n, n') -> Float.(ratio visits w n n' true >= 3.));
          p "the allocation window counts a known allocation" (Gc_witness.window_sees_allocation ());
          in_leg ~leg
            "per depth doubling, the inference allocates 1.8x-2.6x with indexed substitution"
            (fun label ->
              p_all label doublings ~f:(fun (w, n, n') ->
                  Float.between (ratio words w n n' false) ~low:1.8 ~high:2.6));
          in_leg ~leg
            "per depth doubling, the inference allocates at least 3x with whole-environment \
             substitution" (fun label ->
              p_all label doublings ~f:(fun (w, n, n') -> Float.(ratio words w n n' true >= 3.))));
      (* gh-ocannl-1891: a trace leaves every dimension unsolved, so its end validation repeats each
         inequality over the whole bounds graph. *)
      Verdict.case "a whole-stack trace" (fun () ->
          p_all "a whole stack's scheme, applied, closes the shapes one build closes" traced
            ~f:(fun (_, (t, b)) -> List.equal (List.equal Int.equal) t.t_dims b.t_dims);
          let cycle_visits (t : stack_run) = Float.of_int t.t_counts.dim_cycle_visits in
          let grown f = function [ (_, a); (_, b) ] -> f b /. f a | _ -> assert false in
          p "one more layer: the dimension-cycle search visits at most 3x"
            Float.(grown (fun (t, _) -> cycle_visits t) cycle_walk <= 3.);
          p "one more layer: walking every path visits at least 10x"
            Float.(grown (fun (_, u) -> cycle_visits u) cycle_walk >= 10.))
