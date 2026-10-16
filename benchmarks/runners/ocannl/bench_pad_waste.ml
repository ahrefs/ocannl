(* gh-ocannl-740: what a wasteful pad costs the search. The matmul sketch family pads a non-multiple
   extent to its block size wherever both operands are staged (gh-ocannl-485, gh-ocannl-730), and
   config [autotune_pad_waste_max] excludes a geometry whose padded work is at least that many times
   that of its menu's least-padded geometry. This runner tunes one plain matmul per site, so an arm
   with the threshold lifted ([--ocannl_autotune_pad_waste_max=inf]) and an arm at a threshold
   compare the same searches with and without the wasteful candidates.

   Usage: bench_pad_waste.exe SITE... [ocannl configuration flags], each SITE as [name=b,m,n,k]:
   d[b,m,n] = a[b,m,k] * w[k,n] ([b = 1] drops the batch axis). Every search runs from an empty
   schedule cache ([~cache_dir:""]) with the configured search shape.

   Prints, per site, one [padwaste:] line: the sketch tree's leaves and its pad-waste exclusions as
   construction judged them (read off [Autotune.matmul_sketch_tree] of the site's base lowering, so
   they are the seeding's own verdicts), the search's wall time and counters, the crowned label and
   its time, and whether the crowned routine's output matches a host oracle exactly. Inputs are
   small integers, so every product and sum is exact in f32 whatever the schedule, and a padded
   candidate's masked slots must contribute exact zeros. Per-candidate times are the [autotune_log]
   lines on stderr ([--ocannl_autotune_log=true]). Exits 1 when a crowned routine disagrees with the
   oracle. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sspace = Ir.Schedule_space

let parse_site s =
  match String.lsplit2 s ~on:'=' with
  | Some (name, dims) -> (
      match List.map (String.split dims ~on:',') ~f:Int.of_string with
      | [ b; m; n; k ] when b >= 1 && m >= 1 && n >= 1 && k >= 1 -> (name, b, m, n, k)
      | _ -> failwith ("site dims must be b,m,n,k (positive): " ^ s))
  | None -> failwith ("site must be name=b,m,n,k: " ^ s)

(* Every coefficient is a unit modulo its modulus, so each operand varies along every axis it has (a
   coefficient the modulus divides would make the axis cancel, and a 20-deep contraction of such
   cells sums to zero everywhere -- an oracle a routine that only zeroes its output would pass). *)
let a_val bi i kk = Float.of_int (((bi * 5) + (i * 3) + (kk * 2)) % 7) -. 3.
let w_val kk j = Float.of_int (((kk * 3) + (j * 2)) % 5) -. 2.

let bench_site ~failed (name, b, m, n, k) =
  let batch = if b > 1 then [ b ] else [] in
  let a =
    NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.single ~b:batch ~o:[ m; k ]
      ~f:(fun idx -> if b > 1 then a_val idx.(0) idx.(1) idx.(2) else a_val 0 idx.(0) idx.(1))
      ()
  in
  let w =
    NTDSL.init ~l:(name ^ "_w") ~prec:Ir.Ops.single ~o:[ k; n ]
      ~f:(fun idx -> w_val idx.(0) idx.(1))
      ()
  in
  (* The weight has no batch axis, so the batched spec names it: a row variable on both operands
     would force the weight's batch row to the activation's. *)
  let d = if b > 1 then [%op a +* "b|ik;kj=>b|ij" w] else [%op a +* "ik;kj=>ij" w] in
  let comp =
    { (Train.forward d) with asgns = Ir.Assignments.Block_comment (name, (Train.forward d).asgns) }
  in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let is_gpu = Ir.Schedule.backend_is_gpu backend in
  let captured = ref None in
  let _ =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp Ir.Indexing.Empty
  in
  let leaves, pad_excluded =
    match
      Option.bind !captured ~f:(fun opt ->
          Autotune.matmul_sketch_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits opt)
    with
    | None -> (0, 0)
    | Some tree ->
        ( List.length (Sspace.leaves tree),
          List.count (Sspace.exclusions tree) ~f:(fun (_, w) ->
              String.is_prefix w ~prefix:"pad waste:") )
  in
  let report = ref None in
  let t0 = Unix.gettimeofday () in
  let ctx, routine =
    Autotune.tune ~name ~search:true ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx comp Ir.Indexing.Empty
  in
  let wall_s = Unix.gettimeofday () -. t0 in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx d.Tensor.value in
  let exact =
    let ok = ref (Array.length got = b * m * n) in
    for bi = 0 to b - 1 do
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          if !ok then (
            let acc = ref 0. in
            for kk = 0 to k - 1 do
              acc := !acc +. (a_val bi i kk *. w_val kk j)
            done;
            let cell = got.((((bi * m) + i) * n) + j) in
            if Float.(cell <> !acc) then ok := false)
        done
      done
    done;
    (* An all-zero oracle discriminates nothing: refuse it rather than pass it. *)
    !ok && Array.exists got ~f:(fun x -> Float.(x <> 0.))
  in
  if not exact then failed := true;
  let r = Option.value_exn ~here:[%here] !report in
  Stdio.printf
    "padwaste: backend=%s site=%s b=%d m=%d n=%d k=%d threshold=%g leaves=%d pad_excluded=%d \
     wall_s=%.2f outcome=%s sketch_candidates=%d timed=%d failed=%d crowned=%s best_ms=%.4f \
     default_ms=%s exact=%b\n\
     %!"
    backend name b m n k (Autotune.pad_waste_max ()) leaves pad_excluded wall_s
    (Autotune.outcome_name r.Autotune.outcome)
    r.Autotune.sketch_candidates r.Autotune.candidates_timed r.Autotune.candidates_failed
    r.Autotune.best_label r.Autotune.best_ms
    (Option.value_map r.Autotune.default_ms ~default:"-" ~f:(Printf.sprintf "%.4f"))
    exact

let () =
  let sites =
    List.filter_map
      (List.tl_exn (Array.to_list (Sys.get_argv ())))
      ~f:(fun arg -> if String.is_prefix arg ~prefix:"-" then None else Some (parse_site arg))
  in
  if List.is_empty sites then
    failwith "usage: bench_pad_waste.exe name=b,m,n,k ... [ocannl configuration flags]";
  let failed = ref false in
  List.iter sites ~f:(bench_site ~failed);
  if !failed then Stdlib.exit 1
