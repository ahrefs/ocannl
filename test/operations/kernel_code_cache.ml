(* gh-ocannl-1314: the kernel code cache's contract, on any backend -- the cache itself is
   backend-agnostic, so its invariants are pinned here without the hardware the HIP backend that
   uses it needs. Keys: exact, and unambiguous over the list of parts. Retention: by group, bounded
   by bytes, oldest group evicted first; a value already handed out is untouched by its eviction; a
   key re-pointed by a later group survives the older group's eviction; a group larger than the
   capacity is not retained; capacity 0 disables the cache; nothing is retained unless published,
   which is how a failed compile (which publishes nothing) leaves no entry. *)

open Base
open Verdict.Claims
module K = Ir.Kernel_code_cache

let () =
  let key parts = K.key ~parts in
  p "equal parts give equal keys" (String.equal (key [ "-O3"; "src" ]) (key [ "-O3"; "src" ]));
  p "a changed source changes the key"
    (not (String.equal (key [ "-O3"; "a" ]) (key [ "-O3"; "b" ])));
  p "a changed option changes the key"
    (not
       (String.equal
          (key [ "--offload-arch=gfx1102"; "s" ])
          (key [ "--offload-arch=gfx1151"; "s" ])));
  p "parts do not collide by concatenation"
    (not (String.equal (key [ "ab"; "c" ]) (key [ "a"; "bc" ])));
  p "an added option changes the key" (not (String.equal (key [ "s" ]) (key [ "-g"; "s" ])));
  let capacity = ref 250 in
  let c : string ref K.t = K.create ~capacity_bytes:(fun () -> !capacity) in
  p "enabled at a positive capacity" (K.enabled c);
  p "nothing is found before anything is published" (Option.is_none (K.find c "a1"));
  let va = ref "program A" and vb = ref "program B" and vc = ref "program C" in
  K.publish c ~keys:[ "a1"; "a2" ] ~bytes:100 va;
  K.publish c ~keys:[ "b1" ] ~bytes:100 vb;
  let held_a = K.find c "a1" in
  p "a published group answers every one of its keys with its value"
    (Option.exists held_a ~f:(fun v -> phys_equal v va)
    && Option.exists (K.find c "a2") ~f:(fun v -> phys_equal v va)
    && Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vb));
  K.publish c ~keys:[ "c1" ] ~bytes:100 vc;
  p "over capacity, the oldest group is evicted whole"
    (Option.is_none (K.find c "a1") && Option.is_none (K.find c "a2"));
  p "the younger groups stay" (Option.is_some (K.find c "b1") && Option.is_some (K.find c "c1"));
  p "a value handed out before its eviction is still the value published"
    (Option.exists held_a ~f:(fun v -> phys_equal v va && String.equal !v "program A"));
  let st = K.stats c in
  p "the retained bytes are the surviving groups'" (st.n_bytes = 200 && st.n_groups = 2);
  p "one group was evicted" (st.n_evicted_groups = 1);
  (* A key re-pointed by a later group outlives the eviction of the group that first held it. *)
  let vd = ref "program D" in
  K.publish c ~keys:[ "b1"; "d1" ] ~bytes:100 vd;
  p "a re-published key answers with the newer group's value"
    (Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vd));
  p "evicting the older group (B) left the re-pointed key alone"
    (Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vd)
    && Option.exists (K.find c "d1") ~f:(fun v -> phys_equal v vd)
    && Option.exists (K.find c "c1") ~f:(fun v -> phys_equal v vc));
  K.publish c ~keys:[ "huge" ] ~bytes:251 (ref "too big");
  p "a group larger than the capacity is not retained" (Option.is_none (K.find c "huge"));
  p "and evicted nothing to make room for it" (Option.is_some (K.find c "c1"));
  (* The capacity is read on every use: lowering it evicts down to it. *)
  capacity := 100;
  p "a lowered capacity evicts the oldest groups down to it"
    (Option.is_none (K.find c "c1") && Option.is_some (K.find c "d1"));
  capacity := 0;
  p "capacity 0 disables the cache" (not (K.enabled c));
  p "a disabled cache finds nothing" (Option.is_none (K.find c "d1"));
  K.publish c ~keys:[ "e1" ] ~bytes:1 (ref "program E");
  capacity := 250;
  p "and retains nothing published while disabled" (Option.is_none (K.find c "e1"));
  (* The per-kernel renaming of loop symbols the HIP cache keys on. *)
  let canon = K.canonical_symbols in
  p "symbols are renamed by first occurrence"
    (String.equal
       (canon "for (int i12 = 0; i12 < 4; ++i12) a[i12 + i3] = i3;")
       "for (int ocannl_k0 = 0; ocannl_k0 < 4; ++ocannl_k0) a[ocannl_k0 + ocannl_k1] = ocannl_k1;");
  p "kernels differing only in symbol numbers render alike"
    (String.equal (canon "x[i7 * 4 + i9] = y[i9];") (canon "x[i70 * 4 + i2] = y[i2];"));
  p "kernels differing in structure stay apart"
    (not (String.equal (canon "x[i7 * 4 + i9] = y[i9];") (canon "x[i7 * 4 + i9] = y[i7];")));
  p "string and character literals are left alone"
    (String.equal
       (canon {|printf("i12 \" i3", i12); c = 'i';|})
       {|printf("i12 \" i3", ocannl_k0); c = 'i';|});
  p "identifiers and numbers merely containing the spelling are left alone"
    (String.equal
       (canon "xi12 + i12x + i_12 + n12 + 1e12 + i + i1")
       "xi12 + i12x + i_12 + n12 + 1e12 + i + ocannl_k0");
  p "comments are renamed like code"
    (String.equal (canon "/* loop i5 */ a[i5] = 0;") "/* loop ocannl_k0 */ a[ocannl_k0] = 0;");
  p "scope-local names are renamed in the same numbering, keeping their node suffix"
    (String.equal
       (canon "float v12_z; v12_z = x[i4]; y = v12_z + v3_r;")
       "float ocannl_k0_z; ocannl_k0_z = x[ocannl_k1]; y = ocannl_k0_z + ocannl_k2_r;");
  p "two lowerings' locals and symbols render alike"
    (String.equal (canon "v1_z = v1_z + a[i20];") (canon "v3_z = v3_z + a[i31];"));
  p "a v-prefixed name without the scope shape is left alone"
    (String.equal (canon "v12 + v_1 + vx_2 + v1x_y") "v12 + v_1 + vx_2 + v1x_y");
  p "a quote inside a comment is prose: code after it is still renamed"
    (String.equal
       (canon "/* don't */ a[i5] = 0; // it's\nb[i5] = 1;")
       "/* don't */ a[ocannl_k0] = 0; // it's\nb[ocannl_k0] = 1;");
  p "a kept name is left alone whatever its spelling"
    (String.equal
       (K.canonical_symbols ~keep:[ "i123" ] "void i123(int i4) { x[i4] = 0; }")
       "void i123(int ocannl_k0) { x[ocannl_k0] = 0; }")
