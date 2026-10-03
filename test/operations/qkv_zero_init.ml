(* gh-ocannl-1175: expand covering reduction zeros before fission so the existing aligned companion
   rules can keep them in the accumulation kernel. Forward that zero directly into a private
   accumulator with the serial localizer's proof, rather than loading the output. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "qkv_zero_init backend=%s\n" backend_name

let projection ~b ~s ~h ~j ~k =
  let x =
    NTDSL.init ~l:"zi_x" ~prec:Ir.Ops.single ~o:[ b; s; k ]
      ~f:(fun ix -> Float.of_int (1 + (3 * ix.(0)) + (5 * ix.(1)) + ix.(2)) /. 16.)
      ()
  in
  let w =
    NTDSL.init ~l:"zi_w" ~prec:Ir.Ops.single ~o:[ h; j; k ]
      ~f:(fun ix -> Float.of_int (1 + (7 * ix.(0)) + (3 * ix.(1)) + ix.(2)) /. 32.)
      ()
  in
  let%op out = x +* "bsk;hjk=>bshj" w in
  Train.set_materialized out.Tensor.value;
  out

let capture ~name out =
  let captured = ref None in
  let _, _ =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) (Train.forward out) Ir.Indexing.Empty
  in
  Option.value_exn !captured

let fission opt =
  let limits = Ir.Backend_intf.no_hardware_limits in
  Sched.fission_scheduled ~keep_mapping:(Sched.default_gpu ~limits)
    ~preset:(Sched.default_gpu ~limits) ~zero_sched:(Sched.zero_expansion ~limits)
    ~static_indices:[] opt

(* Schedule application registers scratch nodes and placements. Each prototype owns a copy, so
   another prototype's scratch cannot leak into its interface or declarations. *)
let apply schedule opt =
  Sched.apply schedule
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }

let () =
  let out = projection ~b:32 ~s:32 ~h:4 ~j:32 ~k:128 in
  let opt = capture ~name:"zi_shape" out in
  let parts = fission opt in
  p "gpt2 qkv projection has one kernel including initialization" (List.length parts = 1);
  p_all "qkv kernel retains GPU parallelism" parts ~f:(fun (_, _, _, o) ->
      not (List.is_empty (LL.hardware_axes o.LL.llc)));
  let _, pre, _, _ = List.hd_exn parts in
  let seeds =
    Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false
      ~limits:Ir.Backend_intf.no_hardware_limits pre
  in
  p "qkv init companion preserves tiled sketch eligibility" (not (List.is_empty seeds));
  p_all "qkv tiled sketches construct and validate" seeds ~f:(fun seed ->
      match apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:seed pre) pre with
      | o -> (
          match LL.validate_parallel o.LL.optimize_ctx.placements o.LL.llc with
          | () -> true
          | exception exn ->
              Stdio.eprintf "qkv sketch validation FAILED: %s\n" (Exn.to_string exn);
              false)
      | exception exn ->
          Stdio.eprintf "qkv sketch construction FAILED: %s\n" (Exn.to_string exn);
          false);
  p_exists "a tiled qkv sketch forwards zero into the private accumulator" seeds ~f:(fun seed ->
      let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:seed pre) pre in
      Ll_test.count_get o out.Tensor.value = 0 && Ll_test.count_set o out.Tensor.value = 1)

let () =
  let out = projection ~b:2 ~s:32 ~h:2 ~j:32 ~k:64 in
  let opt = capture ~name:"zi_parity_capture" out in
  let seed = [ (out.Tensor.value, Array.create ~len:(2 * 32 * 2 * 32) (-999.)) ] in
  let want =
    List.hd_exn (Ll_test.execute ~name:"zi_materialized" opt ~seed ~read:[ out.Tensor.value ])
  in
  p_all "qkv reference differs from zero and the entry sentinel" (Array.to_list want) ~f:(fun v ->
      Float.(v > 0.));
  let parts = fission opt in
  let _, pre, _, _ = List.hd_exn parts in
  let site = Option.value_exn (Autotune.detect_matmul pre.LL.llc) in
  let tiled =
    apply [ Sched.privatize ~accum_prec:Fn.id ~target:site.Autotune.m_d ~over:site.m_k ] pre
  in
  p "private qkv accumulator opens from zero without output reads"
    (Ll_test.count_get tiled out.Tensor.value = 0);
  p "private qkv accumulator drops the covering init store"
    (Ll_test.count_set tiled out.Tensor.value = 1);
  let got =
    List.hd_exn (Ll_test.execute ~name:"zi_private" tiled ~seed ~read:[ out.Tensor.value ])
  in
  p_all2 "private qkv matches the materialized run with discriminating operands" got want
    ~f:Float.equal;
  let staged_label = "staged tiled qkv with zero forwarded matches the materialized run" in
  if Sched.backend_is_gpu backend_name then begin
    let staged =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false
        ~limits:Ir.Backend_intf.no_hardware_limits pre
      |> List.find_map ~f:(fun p ->
          let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p pre) pre in
          Option.some_if (Ll_test.count_get o out.Tensor.value = 0) o)
      |> Option.value_exn
    in
    let got =
      List.hd_exn (Ll_test.execute ~name:"zi_staged" staged ~seed ~read:[ out.Tensor.value ])
    in
    p_all2 staged_label got want ~f:Float.equal
  end
  else Verdict.skipped ~backend:backend_name staged_label;
  let ctx, routine =
    Context.compile ~name:"zi_default" ~prelowered:opt
      ~lowered_transform:(fun o -> List.map (fission o) ~f:(fun (_, _, _, scheduled) -> scheduled))
      (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed in
  let got = Context.get_values ctx out.Tensor.value in
  p_all2 "fission qkv matches the materialized run with discriminating operands" got want
    ~f:Float.equal;
  let ctx = Context.run ctx routine in
  p_all2 "fission qkv resets the accumulator on every call"
    (Context.get_values ctx out.Tensor.value)
    want ~f:Float.equal

(* An enclosing reduction repeats each cell. Its inner private tile must load the previous partial
   sum, and the whole-node zero remains live. *)
let () =
  let open Ll_test in
  let node = node_factory ~first_id:117500 ~dims:[| 4 |] () in
  let out = node "zi_repeated" in
  materialize out;
  let outer = sym () and i = sym () and k = sym () in
  let update = set_at out (iter i) (add (get out [| iter i |]) (add (tag outer i) (tick k))) in
  let code = seq (zero out) (loop_n outer 3 (loop_n i 4 (loop_n k 5 update))) in
  let opt = optimize ~name:"zi_repeat_capture" code in
  let priv = apply [ Sched.privatize ~accum_prec:Fn.id ~target:out ~over:k ] opt in
  p "an enclosing repeated-cell loop retains initialization and opening reads"
    (count_set priv out = 2 && count_get priv out = 1);
  let seed = [ (out, Array.create ~len:4 (-999.)) ] in
  let want = List.hd_exn (execute ~name:"zi_repeat_materialized" opt ~seed ~read:[ out ]) in
  let got = List.hd_exn (execute ~name:"zi_repeat_private" priv ~seed ~read:[ out ]) in
  p_all2 "repeated-cell private reduction preserves earlier contributions" got want ~f:Float.equal

let () =
  let open Ll_test in
  let node = node_factory ~first_id:117600 ~dims:[| 4 |] () in
  let out = node "zi_refusal" in
  let i = sym () in
  let full = loop_n i 4 (set_at out (iter i) (c 0.)) in
  p "a full per-cell zero is recognized" (Option.is_some (LL.zero_initializer_target full));
  let refused =
    [
      loop_n i 3 (set_at out (iter i) (c 0.));
      if_ (tick i) full;
      loop_n i 0 (set_at out (iter i) (c 0.));
      loop_n i 4 (set_at out (iter i) (c (-0.)));
    ]
  in
  p_all "partial, guarded, dead and negative-zero initializers are retained" refused ~f:(fun init ->
      Option.is_none (LL.zero_initializer_target init))

(* Reusing an immutable statement at two positions must not make zero DSE remove both. *)
let () =
  let open Ll_test in
  let node = node_factory ~first_id:117700 ~dims:[| 4 |] () in
  let out = node "zi_reused_zero" in
  materialize out;
  let i = sym () and k = sym () in
  let cell = [| iter i |] in
  let shared_zero = zero out in
  let update = set out cell (add (get out cell) (add (tick i) (tick k))) in
  let program = seq shared_zero (seq (loop_n i 4 (loop_n k 5 update)) shared_zero) in
  let opt = optimize_scoped ~materialized:[ out ] ~name:"zi_reused_capture" ~raw:program program in
  let priv = apply [ Sched.privatize ~accum_prec:Fn.id ~target:out ~over:k ] opt in
  p "private zero forwarding removes only the proven initializer occurrence"
    (List.count (LL.flat_lines [ priv.LL.llc ]) ~f:(function
       | LL.Zero_out tn -> Ir.Tnode.equal tn out
       | _ -> false)
    = 1);
  let seed = [ (out, Array.create ~len:4 (-999.)) ] in
  let want = List.hd_exn (execute ~name:"zi_reused_materialized" opt ~seed ~read:[ out ]) in
  let got = List.hd_exn (execute ~name:"zi_reused_private" priv ~seed ~read:[ out ]) in
  p_all2 "reused final zero preserves executed parity with the materialized run" got want
    ~f:Float.equal
