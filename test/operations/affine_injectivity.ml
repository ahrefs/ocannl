(* gh-133 Stage B: unit coverage for affine injectivity analysis.

   [Ir.Indexing.affine_injective] implements the proposal's mixed-radix per-position criterion plus a
   whole-LHS pinning fixpoint; [Ir.Indexing.is_injective] wraps it with the product-iterator coverage
   check (a contraction symbol absent from the LHS makes the map non-injective regardless of the
   affine analysis). We assert both the accept/reject cases from the acceptance criteria and the
   known-incomplete [3*a + 4*b] case. *)

open Base
module Idx = Ir.Indexing

let sym () = Idx.get_symbol ()
let aff terms offset = Idx.Affine { symbols = terms; offset }

let ranges pairs s =
  List.Assoc.find pairs s ~equal:Idx.equal_symbol |> Option.value ~default:1

let p name b = Stdio.printf "%s: %b\n" name b

(* Minimal projections: one single-symbol product axis per [product_axes] entry, [project_lhs] over
   those symbols. [is_injective] reads only product_iterators / product_space / project_lhs;
   [is_surjective] additionally reads [lhs_dims] (length must match [project_lhs]). *)
let mk_proj ?(lhs_dims = [||]) product_axes project_lhs : Idx.projections =
  {
    product_space = Array.of_list_map product_axes ~f:(fun (_, d) -> [ d ]);
    lhs_dims;
    rhs_dims = [| [||] |];
    product_iterators = Array.of_list_map product_axes ~f:(fun (s, _) -> [ s ]);
    project_lhs;
    project_rhs = [| project_lhs |];
    debug_info = { Idx.spec = ""; derived_for = Sexp.Atom ""; trace = [] };
  }

let () =
  Stdio.printf "=== affine_injective (per-position + pinning) ===\n";

  (* Accept: 2*oh + wh, wh range 2. *)
  let oh = sym () and wh = sym () in
  p "2*oh+wh (wh range 2) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (oh, 4); (wh, 2) ])
       [| aff [ (2, oh); (1, wh) ] 0 |]);

  (* Accept: K*i + k, k range <= K (K=3, k range 3). *)
  let i = sym () and k = sym () in
  p "K*i+k (k range <= K) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (i, 5); (k, 3) ])
       [| aff [ (3, i); (1, k) ] 0 |]);

  (* Accept: 3*i + j, j range <= 3. *)
  let i2 = sym () and j2 = sym () in
  p "3*i+j (j range <= 3) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (i2, 5); (j2, 3) ])
       [| aff [ (3, i2); (1, j2) ] 0 |]);

  (* Accept: triangular (s1, s1 + s2). *)
  let s1 = sym () and s2 = sym () in
  p "triangular (s1, s1+s2) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (s1, 4); (s2, 3) ])
       [| Idx.Iterator s1; aff [ (1, s1); (1, s2) ] 0 |]);

  (* Reject: i + j, both ranges > 1. *)
  let a = sym () and b = sym () in
  p "i+j (both ranges > 1) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (a, 3); (b, 3) ])
       [| aff [ (1, a); (1, b) ] 0 |]);

  (* Reject: stride*o + k, k range > stride (stride=2, k range 3). *)
  let o = sym () and kk = sym () in
  p "stride*o+k (k range > stride) injective"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (o, 5); (kk, 3) ])
       [| aff [ (2, o); (1, kk) ] 0 |]);

  (* Known-incomplete: 3*a + 4*b over ranges (3, 2) may remain false. *)
  let a2 = sym () and b2 = sym () in
  p "3*a+4*b over (3,2) injective (known-incomplete)"
    (Idx.affine_injective
       ~symbol_range:(ranges [ (a2, 3); (b2, 2) ])
       [| aff [ (3, a2); (4, b2) ] 0 |]);

  Stdio.printf "=== is_injective (with product-iterator coverage) ===\n";

  (* Accept through is_injective: same 2*oh+wh, both symbols are product iterators on the LHS. *)
  let oh3 = sym () and wh3 = sym () in
  p "is_injective 2*oh+wh"
    (Idx.is_injective (mk_proj [ (oh3, 4); (wh3, 2) ] [| aff [ (2, oh3); (1, wh3) ] 0 |]));

  (* Reject through is_injective: non-injective i+j. *)
  let a3 = sym () and b3 = sym () in
  p "is_injective i+j"
    (Idx.is_injective (mk_proj [ (a3, 3); (b3, 3) ] [| aff [ (1, a3); (1, b3) ] 0 |]));

  (* Reject through is_injective: a contraction symbol [c] is a product iterator but never appears on
     the LHS, so multiple product points collapse to one cell. The affine analysis alone would accept
     [Iterator i], but coverage must reject the whole map. *)
  let i4 = sym () and c4 = sym () in
  p "is_injective with uncovered contraction symbol"
    (Idx.is_injective (mk_proj [ (i4, 4); (c4, 3) ] [| Idx.Iterator i4 |]));

  Stdio.printf "=== lowering payoff: injective + surjective scatter skips neutral init ===\n";

  (* Pool-backward with stride = window: the input-gradient index [2*oh+wh] (oh range 3, wh range 2)
     covers [0, 6) exactly. Both injective and surjective, so the lowering in [assignments.ml] skips
     the neutral-init pass and uses a plain setter. *)
  let oh5 = sym () and wh5 = sym () in
  let pool_back = mk_proj ~lhs_dims:[| 6 |] [ (oh5, 3); (wh5, 2) ] [| aff [ (2, oh5); (1, wh5) ] 0 |] in
  p "pool-backward scatter injective" (Idx.is_injective pool_back);
  p "pool-backward scatter surjective" (Idx.is_surjective pool_back);
  p "pool-backward scatter skips neutral-init (surjective && injective)"
    (Idx.is_surjective pool_back && Idx.is_injective pool_back);

  Stdio.printf "%!"
