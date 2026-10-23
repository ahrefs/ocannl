(* gh-ocannl-627: [bin/narrow_gebp_bench] itself, run as the shipped executable, at extents its
   blocking does not divide. [gebp_padded_parity] executes the bench's schedule builder; this one
   executes the bench — its argument parsing, its gate, its cross-variant comparisons and its census
   report — so a gate left in place, or a report that stopped saying what ran, is caught here and
   could not be caught there.

   Before gh-ocannl-627 every one of these runs skipped both packed variants ("this n is not
   schedulable with this blocking") and printed the naive line alone, exiting 0 — except the
   requested-geometry run, which exited 1 because nothing rendered the geometry. So each claim below
   about a packed row fails on the base.

   Each run's whole output goes to stderr, tagged as not part of the golden: timings are not stable,
   and the claims read only the parts that are — which rows ran, what each was compared against and
   whether it agreed, how the census says its [Tile_mma]s rendered, and the padding the header
   states. *)

open Base
open Stdio
open Verdict.Claims

let bench_exe = (Sys.get_argv ()).(1)

let run_bench args =
  let status, out, err = Fresh_process.run ~exe:bench_exe (args @ [ "--ocannl_backend=cc" ]) in
  eprintf "narrow_gebp_bench %s: %s (not part of the golden):\n%s\n%!" (String.concat ~sep:" " args)
    (Fresh_process.describe_status status)
    (Fresh_process.prefixed (out ^ err));
  (status, String.split_lines out)

let has line s = String.is_substring line ~substring:s

(* [variant] ran, agreed with [against] cell by cell, and every one of its [Tile_mma]s rendered the
   register tiling. The row's leading column is the variant's name padded to a fixed width, so the
   prefix [variant ^ " "] tells [packmma] from [packmma_par]. *)
let row_ok ~variant ~against line =
  String.is_prefix line ~prefix:(variant ^ " ")
  && has line ("= " ^ against ^ ")")
  && has line "[tensorized: Mma_register_tiled"

let check ~label ~prec ~n ?blocking ?(extra = []) () =
  let bm, bk = Option.value blocking ~default:(64, 256) in
  let args =
    [ prec; Int.to_string n; "1" ]
    @ (match blocking with None -> [] | Some (bm, bk) -> [ Int.to_string bm; Int.to_string bk ])
    @ extra
  in
  let status, lines = run_bench args in
  let lbl what = Printf.sprintf "%s: %s" label what in
  p (lbl "the bench exits 0") (match status with Unix.WEXITED 0 -> true | _ -> false);
  p_none (lbl "no packed variant is skipped") lines ~f:(fun l -> has l "skipping the packed");
  p_exists
    (lbl "packmma ran, matched naive, and is register-tiled")
    lines
    ~f:(row_ok ~variant:"packmma" ~against:"naive");
  p_exists
    (lbl "packmma_par ran, matched packmma, and is register-tiled")
    lines
    ~f:(row_ok ~variant:"packmma_par" ~against:"packmma");
  (* The padding the header states is the padding [Bench_gebp] schedules at this blocking: the same
     [pads] the builder consults, rendered here only to find the line. *)
  let pads = Bench_gebp.pads ~bm ~bk ~ni:n ~nk:n in
  p_exists (lbl "the header states the padded extents") lines ~f:(fun l ->
      String.is_prefix l ~prefix:"padded: "
      && List.for_all pads ~f:(fun (axis, extent, padded) ->
          has l (Printf.sprintf "%s %d to %d" axis extent padded)))

let () =
  check ~label:"f32 n=76 at the default 64x256 blocking" ~prec:"f32" ~n:76 ();
  check ~label:"bf16 n=77 at 16x32" ~prec:"bf16" ~n:77 ~blocking:(16, 32) ();
  (* A requested register-tile geometry at a padded site: before gh-ocannl-627 no packed site
     existed at n = 76 with a 16-row block, and the bench refused to report a timing for a geometry
     nothing rendered (exit 1). [--rn=2] derives the rest of the geometry from the machine. *)
  check ~label:"f32 n=76 at 16x32 with --rn=2" ~prec:"f32" ~n:76 ~blocking:(16, 32)
    ~extra:[ "--rn=2" ] ()
