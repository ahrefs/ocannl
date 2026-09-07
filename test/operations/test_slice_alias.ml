(* Regression test for gh-ocannl-293 subtask 293a: [Fetch.Slice] ([@|]) is lowered as a zero-copy
   alias *view* of its parent for unpadded leading-axis slices, instead of materializing a copy.

   Invariants pinned here (each would break a specific acceptance criterion if it regressed): - AC1
   (no fresh allocation): an alias-eligible slice owns NO buffer -- it is absent from the routine
   context's [ctx_buffers], while its parent is present. If the alias path silently fell back to a
   copy (or allocated a fresh pool), the slice would appear in [ctx_buffers]. - AC2 (mutation
   visibility, both directions), proved with kernels, not host slice access: * read-through (parent
   -> slice): running the slice's consumer with different runtime [batch_idx] values reads different
   parent rows -- proving the slice indexes into the parent's storage at the *runtime* binding value
   (a static byte-offset alias could not do this). * write-through (slice -> parent): a kernel
   assignment THROUGH the slice mutates the parent's backing buffer, observed by reading the parent.
   - R2 host-access contract: direct [Context.get_values]/[set_values] on an alias view raise a
   clear [User_error] (no silent fresh allocation), since the view has no buffer of its own.

   AC3 (ineligible slices still work via the copy fallback) is covered by [check_slice_shapes],
   whose slice is conv-padded -> ineligible -> materialized copy, and which stays byte-for-byte
   green. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX

(* A gradient-free parent tensor of shape [batch; out], row-major from [vals]. *)
let make_parent label vals ~batch ~out =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[ batch ] ~input_dims:[] ~output_dims:[ out ] ()

let arr_to_string a =
  "[" ^ String.concat ~sep:" " (Array.to_list (Array.map a ~f:(Printf.sprintf "%g"))) ^ "]"

(* Counts statement-level writes ([Set] / [Set_from_vec] / [Zero_out]) targeting tensor node [id] in
   a lowered [Low_level.t]. A materializing slice copy loop emits a [Set] to the slice node inside a
   [For_loop]; an aliased slice lowers its [Fetch.Slice] to [Noop] and redirects every access to the
   parent, so the slice node is written exactly zero times. This is the non-vacuous "no copy loop"
   probe: it flips from 0 to >0 the moment [Fetch.Slice] lowers to [loop_over_dims] instead of
   [Noop]. *)
let count_writes_to (llc : Ir.Low_level.t) (id : int) : int =
  let open Ir.Low_level in
  let n = ref 0 in
  let hit (tn : Ir.Tnode.t) = if tn.Ir.Tnode.id = id then Int.incr n in
  let rec go_t : Ir.Low_level.t -> unit = function
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier | Tile_mma _ ->
        ()
    | Seq (a, b) ->
        go_t a;
        go_t b
    | For_loop { body; _ } -> go_t body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> go_scalar c.init);
        go_t body
    | Zero_out tn -> hit tn
    | Set { tn; llsc; _ } ->
        hit tn;
        go_scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        hit tn;
        go_scalar v;
        go_scalar llsc
    | Set_from_vec { tn; arg = s, _; _ } ->
        hit tn;
        go_scalar s
    | Set_local (_, s) -> go_scalar s
    | If { cond = c, _; body } ->
        go_scalar c;
        go_t body
  and go_scalar : scalar_t -> unit = function
    | Local_scope { body; _ } -> go_t body
    | Get_dynamic { dyn_value = v, _; _ } -> go_scalar v
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        go_scalar a;
        go_scalar b;
        go_scalar c
    | Binop (_, (a, _), (b, _)) ->
        go_scalar a;
        go_scalar b
    | Unop (_, (a, _)) -> go_scalar a
  in
  go_t llc;
  !n

let () =
  Tensor.unsafe_reinitialize ();
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  (* Parent rows: row0 = [1 2 3], row1 = [4 5 6]. *)
  let images = make_parent "images" [| 1.; 2.; 3.; 4.; 5.; 6. |] ~batch:2 ~out:3 in
  let%op bv = images @| batch_n in
  let%op out = bv *. 2 in

  let ctx = Context.auto () in
  let ctx, routine = Train.to_routine ctx bindings (Train.forward out) in
  let bref = IDX.find_exn routine.Context.bindings batch_n in

  (* --- AC1: the alias view owns no buffer; the parent does. --- *)
  Verdict.p "slice is an alias view" (Ir.Tnode.is_alias bv.value);
  Verdict.p "slice absent from ctx_buffers (no fresh alloc)" (not (Context.mem ctx bv.value));
  Verdict.p "parent present in ctx_buffers" (Context.mem ctx images.value);

  (* --- AC2 read-through: the slice reads the parent row selected by the runtime batch index.
     --- *)
  bref := 0;
  Train.run ctx routine;
  let out0 = Context.get_values ctx out.value in
  bref := 1;
  Train.run ctx routine;
  let out1 = Context.get_values ctx out.value in
  Verdict.pf "read-through batch 0: out=%s (expect [2 4 6])" (arr_to_string out0)
    (Array.equal Float.equal out0 [| 2.; 4.; 6. |]);
  Verdict.pf "read-through batch 1: out=%s (expect [8 10 12])" (arr_to_string out1)
    (Array.equal Float.equal out1 [| 8.; 10.; 12. |]);

  (* --- R2: direct host read/write of the alias view is rejected (no silent fresh allocation).
     --- *)
  let raised_get =
    try
      ignore (Context.get_values ctx bv.value : float array);
      false
    with Utils.User_error _ -> true
  in
  let raised_set =
    try
      ignore (Context.set_values ctx bv.value [| 0.; 0.; 0. |] : Context.t);
      false
    with Utils.User_error _ -> true
  in
  Verdict.p "host get_values on alias raises" raised_get;
  Verdict.p "host set_values on alias raises" raised_set;
  Verdict.p "alias still absent from ctx_buffers after rejected host access"
    (not (Context.mem ctx bv.value));

  (* --- AC2 write-through: a kernel write THROUGH the slice mutates the parent's buffer. The
     slice's [alias_of] is already set (persisted on the tnode from the forward lowering above), so
     this write routine redirects to the parent. We write row [batch_n=0] to 99 and observe it via
     the parent. --- *)
  let%cd writer = bv =: !.99.0 in
  let ctx, wroutine = Train.to_routine ctx bindings writer in
  let wref = IDX.find_exn wroutine.Context.bindings batch_n in
  wref := 0;
  Train.run ctx wroutine;
  let parent_after = Context.get_values ctx images.value in
  Verdict.pf "write-through: parent=%s (expect [99 99 99 4 5 6])" (arr_to_string parent_after)
    (Array.equal Float.equal parent_after [| 99.; 99.; 99.; 4.; 5.; 6. |]);

  (* --- AC1 (no copy loop): lower an eligible slice's forward and confirm its [Fetch.Slice] became
     a [Noop] -- the slice node is written ZERO times in the lowered code. Inspecting the lowered
     [Low_level.t] directly (via [Ir.Assignments.to_low_level], pre-optimization) is the non-vacuous
     part: the "absent from ctx_buffers" check above only proves no allocation, not the absence of a
     copy loop. This uses a fresh slice so the forward code is not the consumed one above. --- *)
  let elig_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let parent2 = make_parent "parent2" [| 1.; 2.; 3.; 4.; 5.; 6. |] ~batch:2 ~out:3 in
  let%op elig = parent2 @| elig_n in
  let elig_fwd = Train.forward elig in
  let elig_llc = Ir.Assignments.to_low_level elig_fwd.Ir.Assignments.asgns in
  Verdict.p "eligible slice is aliased" (Ir.Tnode.is_alias elig.value);
  Verdict.p "eligible slice lowered with NO copy loop (0 writes to slice)"
    (count_writes_to elig_llc elig.value.Ir.Tnode.id = 0);

  (* --- AC3 (virtual-parent fallback): a slice of a KNOWN-VIRTUAL parent is ineligible -- it is NOT
     marked as an alias and falls back to the materializing copy loop (>0 writes to the slice). The
     parent is a computed (mode-unspecified) tensor forced [Virtual] before lowering. --- *)
  let virt_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let vbase = make_parent "vbase" [| 1.; 2.; 3.; 4.; 5.; 6. |] ~batch:2 ~out:3 in
  let%op vparent = vbase + vbase in
  Train.set_virtual vparent.value;
  let%op vslice = vparent @| virt_n in
  let virt_fwd = Train.forward vslice in
  let virt_llc = Ir.Assignments.to_low_level virt_fwd.Ir.Assignments.asgns in
  Verdict.p "virtual-parent known_virtual" (Ir.Tnode.known_virtual vparent.value);
  Verdict.p "virtual-parent slice NOT aliased" (not (Ir.Tnode.is_alias vslice.value));
  Verdict.p "virtual-parent slice falls back to copy loop (writes to slice > 0)"
    (count_writes_to virt_llc vslice.value.Ir.Tnode.id > 0);

  (* --- Finding 1 (Codex): a VECTOR store (e.g. `slice =: uniform ()`, a `uint4x32_to_prec_uniform`
     vec-unop) through an eligible slice must redirect to the parent, just like a scalar `set`.
     Otherwise the `Set_from_vec` targets the alias node directly, which owns no buffer and is absent
     from `ctx_buffers`, so the backend cannot link it. Lower and assert the store hits the parent,
     not the slice. --- *)
  (* out dim is a multiple of 4: `uniform` packs 4 single-precision values per uint4x32. *)
  let vec_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let vecp = make_parent "vecp" (Array.init 8 ~f:Float.of_int) ~batch:2 ~out:4 in
  let%op vs = vecp @| vec_n in
  (* Consume the slice's forward (the [Fetch.Slice], which marks the alias) BEFORE building the
     self-referential vector write, so [vs] is still a forward root here. *)
  let vec_fwd = Train.forward vs in
  let%cd vecwrite = vs =: uniform () in
  let vec_llc =
    Ir.Assignments.to_low_level (Ir.Assignments.sequence [ vec_fwd; vecwrite ]).Ir.Assignments.asgns
  in
  Verdict.p "vec-store slice is aliased" (Ir.Tnode.is_alias vs.value);
  Verdict.p "vec-store redirected: 0 vector writes to slice"
    (count_writes_to vec_llc vs.value.Ir.Tnode.id = 0);
  Verdict.p "vec-store redirected: parent receives the vector write (>0)"
    (count_writes_to vec_llc vecp.value.Ir.Tnode.id > 0);

  (* --- Finding 2 (Codex): host write/read of a FRESHLY built slice, before any lowering has marked
     `alias_of`, must still be rejected via the eager `slice_of` marker -- so no detached buffer is
     allocated for the slice (which later alias lowering would orphan). --- *)
  let fresh_n, _ = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let freshp = make_parent "freshp" [| 1.; 2.; 3.; 4.; 5.; 6. |] ~batch:2 ~out:3 in
  let%op fresh_s = freshp @| fresh_n in
  let fresh_ctx = Context.auto () in
  Verdict.p "fresh slice marked is_slice before lowering" (Ir.Tnode.is_slice fresh_s.value);
  Verdict.p "fresh slice has no alias_of yet (pre-lowering)" (not (Ir.Tnode.is_alias fresh_s.value));
  let fresh_set_raises =
    try
      ignore (Context.set_values fresh_ctx fresh_s.value [| 0.; 0.; 0. |] : Context.t);
      false
    with Utils.User_error _ -> true
  in
  let fresh_get_raises =
    try
      ignore (Context.get_values fresh_ctx fresh_s.value : float array);
      false
    with Utils.User_error _ -> true
  in
  Verdict.p "fresh-slice host set_values rejected (no detached buffer)" fresh_set_raises;
  Verdict.p "fresh-slice host get_values rejected" fresh_get_raises;
  Verdict.p "fresh slice never allocated (absent from ctx_buffers)"
    (not (Context.mem fresh_ctx fresh_s.value));
  ()
