(* gh-ocannl-2262: the two config keys beat a backend's advisory register-reuse rate and lane group
   -- a value and [none] -- through the same resolution every other ranking rate takes. The
   acceptance window's [noshl] / [nolane] arms are exactly these overrides: dropped, they would
   score as the shipped arm and the window could not tell the terms from their absence.

   One process per resolution (the overrides are module-level lazies, read once): the dune rule
   states a value for the rate and [none] for the lane group as command-line flags, which outrank an
   ambient environment variable and every config file, over limits whose advisories are both set. *)

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
  p "a stated rate beats the backend's advisory shared-load rate"
    (Option.equal Float.equal (Autotune.ranking_shared_load_rate ~limits:advised) (Some 7e12));
  p "none beats the backend's advisory lane group: the thread-only fill"
    (Autotune.ranking_fill_lane_width ~limits:advised = 1);
  p "the fill model the score reads carries the overridden lane group"
    (match Autotune.ranking_fill_model ~limits:advised with
    | Some fm -> fm.Ir.Cost_model.fm_lane_width = 1
    | None -> false)
