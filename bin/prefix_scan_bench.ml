(* gh-ocannl-1858: what the prefix-scan rewrite's serial [Scan_loop] costs, against the composed
   triangular-mask form it replaces, at the sizes the rewrite targets.

   A [Scan_loop] renders serially on every backend: on a GPU each row's scan is one thread's loop,
   and a row-less scan is one thread for the whole axis (gh-ocannl-1134), while the composed form's
   O(n^2) nest spreads over the device. This bench says whether that matters. Workloads, each at
   every [n] given:

   - [cumsum n rows=1]: puzzle 7's selector contraction over a row-less [n] input; - [cumsum n
   rows=64]: the same along the last axis of a [64, n] input (64 independent scans); - [cummax-step
   n]: a training step through [where tri x -inf @^^ "i o => o"], forward and backward: with the
   rewrite, a forward scan and the gradient gate's reverse scan.

   The treatment is the [prefix_scan] key, stated on the command line
   ([--ocannl_prefix_scan=true|false]): measure through benchmarks/bin_cells.py with one arm per
   setting, so the arms are whole processes and their positions balanced. Every variant's values are
   checked against a host oracle on integer data, exact in either form; a variant whose lowering
   holds no scan under [true] (or one under [false]) fails its check, since a timing labeled with
   the wrong form is a false number.

   Usage: prefix_scan_bench [repeats] [n ...] (defaults 10, and 256 1024 4096). *)

open Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module LL = Ir.Low_level
module Idx = Ir.Indexing

let p fmt = Bench_out.p fmt

let rec scans (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> scans a + scans b
  | LL.For_loop { body; _ } -> scans body
  | LL.If { body; _ } -> scans body
  | LL.Scan_loop { body; _ } -> 1 + scans body
  | _ -> 0

let now () = Time_now.nanoseconds_since_unix_epoch ()
let seconds a b = Float.of_int63 Int63.(b - a) /. 1e9
let data len = Array.init len ~f:(fun k -> Float.of_int ((k * 7 % 11) - 4))

let lower_triangle n =
  let ri = TDSL.range n and co = TDSL.range n in
  let%op tri = not (co ++ "j=>0j" < ri ++ "i=>i0") in
  tri

let cumsum xs = Array.folding_map xs ~init:0. ~f:(fun s x -> (s +. x, s +. x))

(* Compile [comp], run it [repeats] times after one warm run, and read [read] back. The compiled
   context and its parent [ctx] are released before the next case: device pools stay resident until
   then, and a size sweep would otherwise accumulate every case's buffers. *)
let time ~repeats ~name ctx comp read =
  let parent = ctx in
  let lowered = ref [] in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o ->
        lowered := o :: !lowered;
        [ o ])
      ctx comp Idx.Empty
  in
  let ctx = Context.run ctx routine in
  ignore (Context.get_values ctx read : float array);
  let start = now () in
  let ctx = Fn.apply_n_times ~n:repeats (fun ctx -> Context.run ctx routine) ctx in
  Context.sync ctx;
  let stop = now () in
  let values = Context.get_values ctx read in
  Context.release ctx;
  Context.release parent;
  let n_scans = List.sum (module Int) !lowered ~f:(fun o -> scans o.LL.llc) in
  (seconds start stop /. Float.of_int repeats, values, n_scans)

let report ~variant ~seconds ~scans ~want_scans ~ok =
  let check =
    if (want_scans && scans = 0) || ((not want_scans) && scans > 0) then
      Bench_out.Failed
        (Printf.sprintf "the lowering holds %d scan(s) under prefix_scan=%b" scans want_scans)
    else if ok then Bench_out.Passed
    else Bench_out.Failed "values differ from the host oracle"
  in
  p "%-28s %10.4f ms  scans=%d  %s\n" variant (seconds *. 1e3) scans
    (match check with Bench_out.Passed -> "ok" | _ -> "FAILED");
  Bench_out.result ~variant ~check ~seconds;
  match check with Bench_out.Failed _ -> false | _ -> true

let cumsum_case ~repeats ~on n rows =
  Tensor.unsafe_reinitialize ();
  let xs = data (rows * n) in
  let x =
    if rows = 1 then
      Tensor.term_init xs ~label:[ "x" ] ~grad_spec:Prohibit_grad ~output_dims:[ n ] ()
    else
      NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ rows; n ]
        ~f:(function [| b; k |] -> xs.((b * n) + k) | _ -> assert false)
        ()
  in
  let tri = lower_triangle n in
  let y =
    if rows = 1 then
      let%op y = x +* "i; i o => o" tri in
      y
    else
      let%op y = x +* "b i; i o => b o" tri in
      y
  in
  Train.set_materialized y.Tensor.value;
  let ctx = Train.init_params (Context.auto ()) Idx.Empty y in
  let seconds, got, scans =
    time ~repeats ~name:(Printf.sprintf "cumsum_%d_%d" n rows) ctx y.Tensor.forward y.Tensor.value
  in
  let want =
    Array.concat (List.init rows ~f:(fun b -> cumsum (Array.sub xs ~pos:(b * n) ~len:n)))
  in
  report
    ~variant:(Printf.sprintf "cumsum n=%d rows=%d" n rows)
    ~seconds ~scans ~want_scans:on
    ~ok:(Array.equal Float.equal got want)

let cummax_step_case ~repeats ~on n =
  Tensor.unsafe_reinitialize ();
  let xs = data n in
  let ws = Array.init n ~f:(fun o -> Float.of_int (1 + (o * 3 % 7))) in
  let x = Tensor.term_init xs ~label:[ "x" ] ~grad_spec:Require_grad ~output_dims:[ n ] () in
  let w = Tensor.term_init ws ~label:[ "w" ] ~grad_spec:Prohibit_grad ~output_dims:[ n ] () in
  let tri = lower_triangle n in
  let neg_inf = Float.neg_infinity in
  let%op y = where tri (x ++ "i=>i0") !.neg_inf @^^ "i o => o" in
  let%op loss = (y *. w) ++ "... | ... => 0" in
  let x_grad = (Option.value_exn x.Tensor.diff).Tensor.grad in
  Train.set_materialized x_grad;
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) Idx.Empty loss in
  let seconds, got, scans =
    time ~repeats ~name:(Printf.sprintf "cummax_step_%d" n) ctx update x_grad
  in
  (* The tie rule: [x.grad[i]] sums [w[o]] over the [o >= i] whose prefix maximum [x[i]] attains. *)
  let ys = Array.folding_map xs ~init:neg_inf ~f:(fun m x -> (Float.max m x, Float.max m x)) in
  let want =
    Array.mapi xs ~f:(fun i x ->
        Array.foldi ys ~init:0. ~f:(fun o acc y ->
            if o >= i && Float.equal y x then acc +. ws.(o) else acc))
  in
  report
    ~variant:(Printf.sprintf "cummax-step n=%d" n)
    ~seconds ~scans ~want_scans:on
    ~ok:(Array.equal Float.equal got want)

let () =
  let args = List.drop (Array.to_list (Sys.get_argv ())) 1 in
  let args = List.filter args ~f:(fun a -> not (String.is_prefix a ~prefix:"--")) in
  let repeats, sizes =
    match List.map args ~f:Int.of_string with
    | [] -> (10, [ 256; 1024; 4096 ])
    | [ r ] -> (r, [ 256; 1024; 4096 ])
    | r :: ns -> (r, ns)
  in
  let on = Ir.Prefix_scan.enabled () in
  p "prefix_scan_bench: backend %s, prefix_scan=%b, %d repeats\n"
    (Utils.get_global_arg ~arg_name:"backend" ~default:"")
    on repeats;
  let ok =
    List.for_all
      (List.concat_map sizes ~f:(fun n ->
           [
             (fun () -> cumsum_case ~repeats ~on n 1);
             (fun () -> cumsum_case ~repeats ~on n 64);
             (fun () -> cummax_step_case ~repeats ~on n);
           ]))
      ~f:(fun case -> case ())
  in
  if not ok then Stdlib.exit 1
