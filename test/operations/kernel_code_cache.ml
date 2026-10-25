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
  K.publish c ~entries:[ ("a1", va); ("a2", va) ] ~bytes:100;
  K.publish c ~entries:[ ("b1", vb) ] ~bytes:100;
  let held_a = K.find c "a1" in
  p "a published group answers every one of its keys with its value"
    (Option.exists held_a ~f:(fun v -> phys_equal v va)
    && Option.exists (K.find c "a2") ~f:(fun v -> phys_equal v va)
    && Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vb));
  K.publish c ~entries:[ ("c1", vc) ] ~bytes:100;
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
  K.publish c ~entries:[ ("b1", vd); ("d1", vd) ] ~bytes:100;
  p "a re-published key answers with the newer group's value"
    (Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vd));
  p "evicting the older group (B) left the re-pointed key alone"
    (Option.exists (K.find c "b1") ~f:(fun v -> phys_equal v vd)
    && Option.exists (K.find c "d1") ~f:(fun v -> phys_equal v vd)
    && Option.exists (K.find c "c1") ~f:(fun v -> phys_equal v vc));
  K.publish c ~entries:[ ("huge", ref "too big") ] ~bytes:251;
  p "a group larger than the capacity is not retained" (Option.is_none (K.find c "huge"));
  p "and evicted nothing to make room for it" (Option.is_some (K.find c "c1"));
  (* The capacity is read on every use: lowering it evicts down to it. *)
  capacity := 100;
  p "a lowered capacity evicts the oldest groups down to it"
    (Option.is_none (K.find c "c1") && Option.is_some (K.find c "d1"));
  capacity := 0;
  p "capacity 0 disables the cache" (not (K.enabled c));
  ignore (K.find c "d1" : string ref option);
  p "a lookup at capacity 0 releases everything retained"
    ((K.stats c).n_bytes = 0 && (K.stats c).n_entries = 0);
  p "a disabled cache finds nothing" (Option.is_none (K.find c "d1"));
  K.publish c ~entries:[ ("e1", ref "program E") ] ~bytes:1;
  capacity := 250;
  p "and retains nothing published while disabled" (Option.is_none (K.find c "e1"));
  (* The per-kernel renaming the GPU caches key on. *)
  let canon ?kernel_name ?keep ?bound ?free text =
    K.canonical_kernel ?kernel_name ?keep ?bound ~derivations:Ir.C_syntax.minted_derivations ?free
      text
  in
  let same ?kernel_name ?bound a b =
    String.equal (canon ?kernel_name ?bound a) (canon ?kernel_name ?bound b)
  in
  p "symbols are renamed by first occurrence"
    (String.equal
       (canon "for (int i12 = 0; i12 < 4; ++i12) a[i12 + i3] = i3;")
       "for (int ocannl_k0 = 0; ocannl_k0 < 4; ++ocannl_k0) a[ocannl_k0 + ocannl_k1] = ocannl_k1;");
  p "kernels differing only in symbol numbers render alike"
    (same "x[i7 * 4 + i9] = y[i9];" "x[i70 * 4 + i2] = y[i2];");
  p "kernels differing in structure stay apart"
    (not (same "x[i7 * 4 + i9] = y[i9];" "x[i7 * 4 + i9] = y[i7];"));
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
  p "scope-local names are renamed whole, in the same numbering"
    (String.equal
       (canon "float v12_z; v12_z = x[i4]; y = v12_z + v3_r;")
       "float ocannl_k0; ocannl_k0 = x[ocannl_k1]; y = ocannl_k0 + ocannl_k2;");
  p "two lowerings' locals and symbols render alike"
    (same "v1_z = v1_z + a[i20];" "v3_z = v3_z + a[i31];");
  p "a local whose node suffix carries a fresh tile id renders alike"
    (same "v4_tile__n812_acc = 0; o[0] = v4_tile__n812_acc;"
       "v9_tile__n977_acc = 0; o[0] = v9_tile__n977_acc;");
  p "a v-prefixed name without the scope shape is left alone"
    (String.equal (canon "v12 + v_1 + vx_2 + v1x_y") "v12 + v_1 + vx_2 + v1x_y");
  p "a quote inside a comment is prose: code after it is still renamed"
    (String.equal
       (canon "/* don't */ a[i5] = 0; // it's\nb[i5] = 1;")
       "/* don't */ a[ocannl_k0] = 0; // it's\nb[ocannl_k0] = 1;");
  p "a kept name is left alone whatever its spelling"
    (String.equal
       (canon ~keep:[ "i123" ] "void i123(int i4) { x[i4] = 0; }")
       "void i123(int ocannl_k0) { x[ocannl_k0] = 0; }");
  (* gh-ocannl-1525: the kernel's own name and its nodes' identifiers. *)
  let k1 = "void r__seg3(float *n12_x, float *w) { n12_x[i4] = w[i4]; }" in
  let k2 = "void r__seg7(float *n57_x, float *w) { n57_x[i9] = w[i9]; }" in
  p "the same kernel under two names and two node numberings renders alike"
    (String.equal
       (canon ~kernel_name:"r__seg3" ~bound:[ "n12_x"; "w" ] k1)
       (canon ~kernel_name:"r__seg7" ~bound:[ "n57_x"; "w" ] k2));
  p "a segment's positional index is dropped, its routine's name kept"
    (String.equal (K.kernel_token (K.segment_kernel_name ~routine:"r" 3)) "r__seg"
    && String.equal (K.kernel_token "r") "r"
    && String.equal (K.kernel_token "r__segx") "r__segx");
  p "kernels of two routines stay apart"
    (not
       (String.equal
          (canon ~kernel_name:"r__seg3" ~bound:[ "w" ] "void r__seg3(float *w) { w[0] = 0; }")
          (canon ~kernel_name:"q__seg3" ~bound:[ "w" ] "void q__seg3(float *w) { w[0] = 0; }")));
  p "a node's bare and disambiguated spellings render alike"
    (String.equal
       (canon ~bound:[ "x"; "w" ] "void k(float *x, float *w) { x[0] = w[0]; }")
       (canon ~bound:[ "n12_x"; "w" ] "void k(float *n12_x, float *w) { n12_x[0] = w[0]; }"));
  p "two layers' kernels over differently labelled nodes render alike"
    (String.equal
       (canon ~bound:[ "l1_w"; "l1_b" ] "void k(float *l1_w, float *l1_b) { l1_w[0] += l1_b[0]; }")
       (canon ~bound:[ "l2_w"; "l2_b" ] "void k(float *l2_w, float *l2_b) { l2_w[0] += l2_b[0]; }"));
  p "a name outside the bound set is not renamed"
    (String.equal (canon ~bound:[ "x" ] "x[0] = y[0];") "ocannl_k0[0] = y[0];");
  p "two parameters stay apart from one parameter used twice"
    (not
       (String.equal
          (canon ~bound:[ "a"; "b" ] "void k(float *a, float *b) { a[0] = a[0] + b[0]; }")
          (canon ~bound:[ "a"; "b" ] "void k(float *a, float *b) { a[0] = a[0] + a[0]; }")));
  p "swapped operands stay apart"
    (not
       (String.equal
          (canon ~bound:[ "a"; "b" ] "void k(float *a, float *b) { a[0] = a[0] - b[0]; }")
          (canon ~bound:[ "a"; "b" ] "void k(float *a, float *b) { a[0] = b[0] - a[0]; }")));
  p "a member name is never renamed, even when it spells a bound node"
    (String.equal
       (canon ~bound:[ "x" ] "x[threadIdx.x] = p->x + q::x;")
       "ocannl_k0[threadIdx.x] = p->x + q::x;");
  p "so kernels reading different members stay apart"
    (not
       (String.equal
          (canon ~bound:[ "x" ] "x[threadIdx.x] = 0;")
          (canon ~bound:[ "y" ] "y[threadIdx.y] = 0;")));
  p "attributes and preprocessor lines are copied as they are"
    (String.equal
       (canon ~bound:[ "buffer"; "x" ]
          "#define x 1\nkernel void k(device float *x [[buffer(0)]]) {}")
       "#define x 1\nkernel void k(device float *ocannl_k0 [[buffer(0)]]) {}");
  p "a bound name that is also a free name disables the node renaming"
    (String.equal
       (canon ~bound:[ "x"; "expf" ] ~free:[ "expf" ] "x[0] = expf(x[0]);")
       "x[0] = expf(x[0]);");
  (* Every derived local, spelled the way the emitter spells it ([C_syntax.derived_name] over each
     template), over a node or a symbol base: the canonical text renames each whole. *)
  let module CS = Ir.C_syntax in
  let d = CS.derived_name in
  let chunk, lo, hi = CS.grid_chunk_derivations in
  let lane_v, lane_p = CS.lane_reduce_derivations in
  let warp_v, warp_p = CS.warp_reduce_derivations in
  let derived_locals =
    [
      d CS.mem_alias_derivation "n12_x";
      d chunk "i3";
      d lo "i3";
      d hi "i3";
      d lane_v "i3";
      d lane_p "i3";
      d warp_v "i4";
      d warp_p "i4";
      d CS.mma_fragment_derivation "812";
    ]
  in
  p "the emitter's derived locals are renamed whole, over a node, a symbol or a uid"
    (String.equal
       (canon ~bound:[ "n12_x" ] (String.concat ~sep:" + " derived_locals))
       (String.concat ~sep:" + "
          (List.mapi derived_locals ~f:(fun i _ -> "ocannl_k" ^ Int.to_string i))));
  p_all "every template the emitter spells through is listed for the canonical text"
    (CS.mem_alias_derivation :: CS.mma_fragment_derivation
    :: [ chunk; lo; hi; lane_v; lane_p; warp_v; warp_p ])
    ~f:(List.mem CS.minted_derivations ~equal:Poly.equal);
  p "a derived spelling over an unbound base is left alone"
    (String.equal
       (canon ~bound:[] (d CS.mem_alias_derivation "y" ^ " + " ^ d warp_v "q"))
       (d CS.mem_alias_derivation "y" ^ " + " ^ d warp_v "q"));
  p "different literals stay apart" (not (same ~bound:[ "x" ] "x[0] = 1.5f;" "x[0] = 2.5f;"));
  (* [resolve]: lookup, dedup and compile of the misses, against a fake compiler. *)
  let rc : (int * string) K.t = K.create ~capacity_bytes:(fun () -> 1000) in
  let lock = Stdlib.Mutex.create () in
  let compiled = ref [] in
  let fail_next = ref false in
  let next_program = ref 0 in
  let compile_group ~group:_ ~n_groups:_ _opts idxs =
    if !fail_next then (
      fail_next := false;
      failwith "compiler refused");
    compiled := idxs :: !compiled;
    Int.incr next_program;
    (!next_program, 10)
  in
  let resolve ?(options = [| [ "-O3" ]; [ "-O3" ]; [ "-O3" ] |]) keys symbols =
    compiled := [];
    K.resolve rc ~lock ~keys ~options ~symbols ~compile_group ()
  in
  let r1 = resolve [| "ka"; "kb"; "ka" |] [| "s0"; "s1"; "s2" |] in
  p "equal keys in one batch compile one representative"
    (List.equal (List.equal Int.equal) !compiled [ [ 0; 1 ] ]);
  p "the duplicate shares the representative's program and symbol"
    (Poly.equal r1.(2) r1.(0) && String.equal (snd r1.(0)) "s0");
  let r2 = resolve [| "kb"; "kc"; "ka" |] [| "t0"; "t1"; "t2" |] in
  p "a later batch compiles only its misses" (List.equal (List.equal Int.equal) !compiled [ [ 1 ] ]);
  p "a hit answers the symbol its code was compiled under, not the current name"
    (String.equal (snd r2.(0)) "s1" && String.equal (snd r2.(2)) "s0");
  let _ =
    resolve
      ~options:[| [ "-O3" ]; [ "-O2" ]; [ "-O3" ] |]
      [| "kd"; "ke"; "kf" |] [| "u0"; "u1"; "u2" |]
  in
  p "misses compile as one program per option list, in batch order"
    (List.equal (List.equal Int.equal) (List.rev !compiled) [ [ 0; 2 ]; [ 1 ] ]);
  fail_next := true;
  let refused =
    match resolve [| "kg"; "kh"; "ka" |] [| "v0"; "v1"; "v2" |] with
    | _ -> false
    | exception Failure _ -> true
  in
  p "a refused compile propagates" refused;
  p "and publishes nothing" (Option.is_none (K.find rc "kg") && Option.is_none (K.find rc "kh"));
  let _ = resolve [| "kg"; "kh"; "ka" |] [| "v0"; "v1"; "v2" |] in
  p "the next compile of those kernels compiles them afresh"
    (List.equal (List.equal Int.equal) !compiled [ [ 0; 1 ] ]);
  (* Two domains missing the same keys: the second finds the first's program under the lock. *)
  let racing : (int * string) K.t = K.create ~capacity_bytes:(fun () -> 1000) in
  let n_compiles = Atomic.make 0 in
  let race () =
    K.resolve racing ~lock ~keys:[| "x"; "y" |] ~options:[| []; [] |] ~symbols:[| "a"; "b" |]
      ~compile_group:(fun ~group:_ ~n_groups:_ _ _ ->
        Atomic.incr n_compiles;
        Unix.sleepf 0.05;
        (Atomic.get n_compiles, 1))
      ()
  in
  let d = Stdlib.Domain.spawn race in
  let mine = race () in
  let theirs = Stdlib.Domain.join d in
  p "two domains missing the same keys compile them once" (Atomic.get n_compiles = 1);
  p "and both get the one program" (Poly.equal mine theirs)
