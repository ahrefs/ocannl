(* gh-ocannl-685: the action menu's per-unit cap is shared across its categories, not spent in
   category order.

   [Autotune.menu] builds five category lists — tensorizes, splits, swaps, unrolls, vectorizes — and
   used to cap them with [List.take (tensorizes @ splits @ ...) max_actions_per_unit]. The list is a
   concatenation ordered by category and is NOT ranked (contrast the placement surface's prefix over
   [rank_flip_candidates], where top-N is the intended semantics), so the prefix was arbitrary: a
   unit whose tensorizes alone reach the cap — tensorize proposals are up to 6 role assignments per
   tightly-nested innermost serial triple, so the count scales with the number of matmul-shaped
   nests — offered the search no split, swap, unroll or vectorize action at all. Not fewer of them:
   none. And those are exactly the categories a unit needs when its tensorizes turn out [Op_illegal]
   or unprofitable. Silent search-space loss, with a [menu:] log line reporting the counts BEFORE
   the take, so a truncated menu logged the same numbers as an untruncated one.

   [Autotune.share_cap] is the fix and this is its pin, over synthetic categories (integers stand in
   for proposals) so the claims are about the sharing rule rather than about whichever menu a
   particular lowering happens to produce. The negative control is stated as its own claim: the
   plain prefix the cap used to be starves every category after the first on the same input, so a
   revert fails the representation claim rather than quietly passing. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
open Verdict.Claims

let cap = 48

(* Category sizes: one overflowing category, one comfortably small, one empty, and two in between —
   100 + 6 + 0 + 10 + 5 = 121 proposals competing for 48 slots. *)
let sizes = [ ("tensorize", 100); ("split", 6); ("swap", 0); ("unroll", 10); ("vectorize", 5) ]

(* Each synthetic proposal remembers its category, so "represented" is checkable. *)
let categories = List.map sizes ~f:(fun (name, n) -> (name, List.init n ~f:(fun i -> (name, i))))
let total = List.sum (module Int) sizes ~f:snd
let non_empty = List.filter_map sizes ~f:(fun (name, n) -> if n > 0 then Some name else None)
let represented kept name = List.exists kept ~f:(fun (c, _) -> String.equal c name)

let () =
  let kept, dropped = Autotune.share_cap ~cap categories in
  p "the overflowing menu is capped to exactly the budget" (List.length kept = cap);
  p_all "every non-empty category is represented under the cap" non_empty ~f:(represented kept);
  p_none "an empty category is not conjured into the menu" kept ~f:(fun (category, _) ->
      String.equal category "swap");
  (* Remainder spill: with 4 non-empty categories an equal share is 12, so the two categories
     smaller than that keep everything and their unused share goes to the larger ones. *)
  let kept_of name = List.count kept ~f:(fun (c, _) -> String.equal c name) in
  p "a category smaller than its share is kept whole (split: 6 of 6)" (kept_of "split" = 6);
  p "a category smaller than its share is kept whole (vectorize: 5 of 5)" (kept_of "vectorize" = 5);
  p "the spilled remainder goes to the categories that still have proposals"
    (kept_of "tensorize" + kept_of "unroll" = cap - 6 - 5);
  p_all "no category is starved to zero while it has proposals and budget remains" non_empty
    ~f:(fun name -> kept_of name > 0);
  (* Survivors keep their category order and their within-category order, so the fix is invisible to
     anything downstream that fits under the cap. *)
  p "survivors stay in category order and in each category's own order"
    (List.equal
       (fun (a, i) (b, j) -> String.equal a b && i = j)
       kept
       (List.concat_map sizes ~f:(fun (name, _) ->
            List.filter kept ~f:(fun (c, _) -> String.equal c name))));
  (* What the cap dropped is reported, per category, rather than what it was offered. *)
  p "the drop report accounts for every withheld proposal"
    (List.sum (module Int) dropped ~f:snd = total - cap);
  p_all "the drop report names only categories that actually lost proposals" dropped
    ~f:(fun (name, d) ->
      d > 0 && d = List.Assoc.find_exn sizes name ~equal:String.equal - kept_of name);
  (* An under-full menu is returned untouched, with nothing reported dropped. *)
  let small =
    List.map sizes ~f:(fun (name, n) -> (name, List.init (min n 3) ~f:(fun i -> (name, i))))
  in
  let small_kept, small_dropped = Autotune.share_cap ~cap small in
  p "a menu that fits under the cap is unchanged"
    (List.equal
       (fun (a, i) (b, j) -> String.equal a b && i = j)
       small_kept (List.concat_map small ~f:snd));
  p_empty "a menu that fits under the cap reports no drops" ~over:small small_dropped;
  (* The negative control, stated as a claim: on this very input the plain prefix the cap used to be
     represents the first category and nothing else. *)
  let prefix = List.take (List.concat_map categories ~f:snd) cap in
  p_all "the plain prefix this replaces would have starved every category after the first" non_empty
    ~f:(fun name -> String.equal name "tensorize" || not (represented prefix name));
  Stdio.printf "\n%!"

(* === the cap's altitude (PR #424 review, P2): [Autotune.menu]'s [admits] runs BEFORE the cap ===

   The beam expands a GPU incumbent that binds no hardware dimension only through moves that can
   bind one; every other move provably yields another undispatchable candidate. That refusal used to
   run AFTER [menu] had already capped, so a tensorize-rich unit spent its 48 slots across five
   categories and kept only its share of the one category the beam could use — where the old plain
   prefix, by accident of ordering, handed all 48 to the tensorizes. Filtering first makes that the
   rule rather than the accident.

   The pin needs a menu that genuinely overflows, so: twelve sibling loops of extent 8, each
   innermost and zero-origin, which draw two Splits, two Unrolls and one Vectorized retype apiece.
   The discriminator is that admitting ONE category yields MORE of that category than the uncapped
   sharing leaves it — impossible if the filter ran after the cap. *)

let overflowing_unit () =
  let node = Ll_test.node_factory ~first_id:7300 ~dims:[| 8 |] () in
  let x = node "amc_x" in
  Ll_test.materialize x;
  let outs = List.init 12 ~f:(fun k -> node (Printf.sprintf "amc_out%d" k)) in
  List.iter outs ~f:Ll_test.materialize;
  let body =
    List.foldi outs ~init:LL.Noop ~f:(fun k acc out ->
        let s = Ll_test.sym () in
        LL.Seq
          ( acc,
            Ll_test.loop_n s 8
              (Ll_test.set out
                 [| Ll_test.iter s |]
                 (Ll_test.add (Ll_test.get x [| Ll_test.iter s |]) (Ll_test.c (Float.of_int k)))) ))
  in
  let o = Ll_test.optimize ~materialized:(x :: outs) ~name:"amc_overflow" body in
  let canon = SC.canonicalize ~static_indices:[] ~with_placements:false o in
  (SC.base_registry canon, o)

let () =
  let registry, o = overflowing_unit () in
  let limits =
    { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32; simd_vector_registers = 16 }
  in
  let build ?admits () = Autotune.menu ?admits ~is_cpu:true ~is_gpu:false ~limits ~registry o in
  let is_unroll = function SC.Unroll _ -> true | _ -> false in
  let full = build () in
  let unrolls_only = build ~admits:is_unroll () in
  p "the probe unit really does overflow the per-unit cap" (List.length full = cap);
  p_all "admitting one category yields only that category" unrolls_only ~f:is_unroll;
  p "the admitted category is not itself capped (so it is the complete admitted set)"
    (List.length unrolls_only < cap);
  (* The discriminator. Were [admits] applied to the capped menu, this count could only ever be the
     share the cap left that category -- never more. *)
  p "admitting one category yields more of it than sharing the cap leaves it"
    (List.length unrolls_only > List.count full ~f:is_unroll);
  Stdio.printf "\nDone.\n%!"
