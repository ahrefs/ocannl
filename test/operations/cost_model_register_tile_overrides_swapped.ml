(* gh-ocannl-2262: [cost_model_register_tile_overrides] with the directions swapped -- [none] for
   the register-reuse rate and a value for the lane group, over the same advised limits. This is the
   resolution the acceptance window's [noshl] arm takes in [benchmarks/gh1253_cells.sh]
   ([model_shared_load_rate=none] beside the shipped arm's stated [model_fill_lane_width]): dropped,
   the [none] would fall back to the advisory rate and [noshl] would score as the shipped arm. The
   sibling pins the other two directions; together the pair covers both keys in both forms.

   One process per resolution (the overrides are module-level lazies, read once): the dune rule
   states [none] for the rate and [16] for the lane group as command-line flags, which outrank an
   ambient environment variable and every config file, over limits whose advisories are both set
   (and differ from the stated lane group). *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf

let () =
  let advised =
    {
      BI.no_hardware_limits with
      multiprocessors = Some 46;
      fill_threads = Some 256;
      shared_load_rate = Some 2.5e12;
      fill_lane_width = Some 32;
    }
  in
  p "none beats the backend's advisory shared-load rate: no register-reuse term"
    (Option.is_none (Autotune.ranking_shared_load_rate ~limits:advised));
  p "a stated lane group beats the backend's advisory one"
    (Autotune.ranking_fill_lane_width ~limits:advised = 16);
  p "the fill model the score reads carries the overridden lane group"
    (match Autotune.ranking_fill_model ~limits:advised with
    | Some fm -> fm.Ir.Cost_model.fm_lane_width = 16
    | None -> false)
