(* gh-ocannl-1562: [bin/schedule_bench] itself, run as the shipped executable on cc, at extents its
   GEBP blocking (64 x 64) does not divide. [gebp_padded_parity] executes the shared builder
   ([Bench_gebp]) at every layout; this one executes the bench — its gates, its cell-by-cell
   comparison against the unscheduled computation, its census report and its padding header — so a
   gate left in place, a variant still built by a private copy, or a report that stopped saying what
   ran is caught here and could not be caught there.

   Before gh-ocannl-1562 every one of these runs skipped all five packed variants ("this size is not
   schedulable with this blocking"), so each claim below about a packed row at a padded site fails
   on the base. The dividing control holds what the change must not lose: every packed variant,
   pm_hoist included, still runs there, with no padding stated.

   The naive timing is skipped (fifth argument 0): the oracle is then one untimed run of the
   unscheduled computation, which is what every packed row's [= reference] compares against, cell
   for cell, and the bench exits 1 on any disagreement. Each run's whole output goes to stderr,
   tagged as not part of the golden: timings are not stable, and the claims read only the parts that
   are. *)

open Base
open Stdio
open Verdict.Claims

let bench_exe = (Sys.get_argv ()).(1)
let gebp_bm, gebp_bk = (64, 64)

let run_bench args =
  let status, out, err = Fresh_process.run ~exe:bench_exe (args @ [ "--ocannl_backend=cc" ]) in
  eprintf "schedule_bench %s: %s (not part of the golden):\n%s\n%!" (String.concat ~sep:" " args)
    (Fresh_process.describe_status status)
    (Fresh_process.prefixed (out ^ err));
  (status, String.split_lines out)

let has line s = String.is_substring line ~substring:s

(* [variant] ran, agreed with the unscheduled computation cell by cell, and every one of its
   [Tile_mma]s rendered the register tiling. The row's leading column is the variant's name padded
   to a fixed width, so the prefix [variant ^ " "] tells [packmma] from [packmma_par]. *)
let row_ok ~variant line =
  String.is_prefix line ~prefix:(variant ^ " ")
  && has line "= reference)"
  && has line "[tensorized: Mma_register_tiled"

(* The packed variants that stage both operands, which pad; pm_hoist reads A in place. *)
let staged = [ "packmma"; "packmma_par"; "pm_mixed"; "pm_bpk" ]

(* [m x k] times [k x n], in the bench's positional order [n repeats m k naive_repeats]. *)
let check ~label ~m ~n ~k =
  let status, lines = run_bench [ Int.to_string n; "1"; Int.to_string m; Int.to_string k; "0" ] in
  let lbl what = Printf.sprintf "%s: %s" label what in
  let pads = Bench_gebp.pads ~bm:gebp_bm ~bk:gebp_bk ~ni:m ~nk:k in
  p (lbl "the bench exits 0") (match status with Unix.WEXITED 0 -> true | _ -> false);
  List.iter staged ~f:(fun variant ->
      p_exists
        (lbl (variant ^ " ran, matched the unscheduled computation, and is register-tiled"))
        lines ~f:(row_ok ~variant));
  match Bench_gebp.in_place_a_refusal ~bm:gebp_bm ~bk:gebp_bk ~ni:m ~nk:k with
  | Some why ->
      p_exists (lbl "pm_hoist is skipped, with the builder's refusal as the reason") lines
        ~f:(fun l -> has l why);
      p_exists (lbl "the header states the padded extents") lines ~f:(fun l ->
          String.is_prefix l ~prefix:("padded: " ^ Bench_gebp.describe_pads pads ^ " "))
  | None ->
      p_exists
        (lbl "pm_hoist ran, matched the unscheduled computation, and is register-tiled")
        lines ~f:(row_ok ~variant:"pm_hoist");
      p_none (lbl "no padding is stated") lines ~f:(fun l -> String.is_prefix l ~prefix:"padded:")

let () =
  check ~label:"100x76x130, rows and k padded" ~m:100 ~n:76 ~k:130;
  check ~label:"37x40x45, below both blocks" ~m:37 ~n:40 ~k:45;
  check ~label:"128x76x130, k-only remainder" ~m:128 ~n:76 ~k:130;
  check ~label:"64x40x64, dividing control" ~m:64 ~n:40 ~k:64
