(* gh-ocannl-1766: [Autotune.resolve_cache_dir] with [autotune_cache_dir] unset, as a tuned bench
   cell that leaves the key defaulted runs it. The directory a search uses and the key's record in
   the resolved-configuration stream are one read, so they agree, and the stream carries one record
   for the key. The rule runs this with [--ocannl_log_config_json=true] and hands its stderr to
   [benchmarks/config_records.py show], which refuses a stream that resolved one key two ways: the
   end-to-end half of the claim, through the reader every bench driver checks its cells with.

   The search-off rule itself -- a CHOSEN directory replays, the built-in default does not -- is
   exercised end to end, over a real cache, by autotune_smoke; the calls here are the resolution
   alone. *)

open Base
open Verdict.Claims

let () =
  let key = "autotune_cache_dir" in
  let records () =
    List.filter (Utils.resolved_config ()) ~f:(fun r -> String.equal r.Utils.key key)
  in
  let searched = Autotune.resolve_cache_dir ~search:true () in
  let unsearched = Autotune.resolve_cache_dir ~search:false () in
  let passed = Autotune.resolve_cache_dir ~cache_dir:"autotune_cache_passed" ~search:false () in
  p "a search uses the built-in default directory" (String.equal searched "autotune_cache");
  p "with the search off, the built-in default directory is no cache" (String.equal unsearched "");
  p "with the search off, a passed directory is still used"
    (String.equal passed "autotune_cache_passed");
  p "the key has one record, the built-in default, naming the directory a search uses"
    (match records () with
    | [ { value; source = Utils.From_default; _ } ] -> String.equal value searched
    | _ -> false)
