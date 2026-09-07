(** Schedule IR: loop-nest transforms as values, applied to [Low_level.optimized] at the
    [?lowered_transform] seam of backend [compile]. See docs/proposals/schedule-ir-optops.md — that
    document records the normative pass-ordering contract (§2): the schedule runs after the whole
    [optimize_proc] pipeline, transforms fold their own guards by re-running [simplify_llc] (plus
    CSE/hoisting when a transform duplicated code), and there is no re-virtualization. *)

open Base
module Tn = Tnode

type optop =
  | Split of {
      axis : Indexing.symbol;  (** The loop to split, identified by its index symbol. *)
      factor : int;  (** Extent of the new inner loop. *)
      outer : Low_level.axis_type;  (** Axis type of the new outer loop. *)
      inner : Low_level.axis_type;  (** Axis type of the new inner loop. *)
      outer_index : Indexing.symbol;
      inner_index : Indexing.symbol;
    }
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
  | Unroll of { axis : Indexing.symbol; materialize : bool }
  | Partition of {
      axis : Indexing.symbol;  (** The loop to partition, identified by its index symbol. *)
      breakpoints : int list;
          (** Strictly increasing segment starts, each strictly inside the loop range. *)
      segment_indices : Indexing.symbol list;
          (** Fresh symbols, one per segment ([length breakpoints + 1]); see {!partition}. *)
    }
  | Pad of {
      axis : Indexing.symbol;  (** The loop to pad, identified by its index symbol. *)
      to_multiple_of : int;  (** The padded extent is the least multiple [>=] the loop extent. *)
    }
  | Stage of {
      source : Tn.t;
      tile_loops : Indexing.symbol list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : Low_level.swizzle_kind option;
      pad_stride : int option;
      pipeline_depth : int;
      tile_prec : Ops.prec option;
    }
  | Privatize of { target : Tn.t; over : Indexing.symbol }
  | Expand_zero of { tn : Tn.t; indices : Indexing.symbol list }
  | Tensorize of {
      i : Indexing.symbol;
      j : Indexing.symbol;
      k : Indexing.symbol;
      lane : Indexing.symbol;
      simd_width : int;
      tile : Register_tile.t option;
    }
  | Fuse_epilogue of { target : Tn.t; shared : bool }
  | Split_reduce of {
      axis : Indexing.symbol;  (** The Serial reduction loop to split, by its index symbol. *)
      target : Tn.t;  (** The accumulated node whose reduction over [axis] is split. *)
      num_blocks : int;  (** Number of per-block partials (the new block loop's extent). *)
      block_index : Indexing.symbol;
      inner_index : Indexing.symbol;
      combine_indices : Indexing.symbol list;
          (** Fresh symbols, one per [target] axis, binding the combine nest's loops; see
              {!split_reduce}. *)
    }
[@@deriving sexp_of]

type schedule = optop list [@@deriving sexp_of]

let split ~axis ~factor ~outer ~inner =
  let outer_index = Indexing.get_symbol () and inner_index = Indexing.get_symbol () in
  (Split { axis; factor; outer; inner; outer_index; inner_index }, outer_index, inner_index)

let partition ~axis ~breakpoints =
  let segment_indices =
    List.init (List.length breakpoints + 1) ~f:(fun _ -> Indexing.get_symbol ())
  in
  (Partition { axis; breakpoints; segment_indices }, segment_indices)

let tensorize ?tile ~i ~j ~k ~simd_width () =
  let lane = Indexing.get_symbol () in
  (Tensorize { i; j; k; lane; simd_width; tile }, lane)

let expand_zero ~tn =
  let rank = Array.length (Lazy.force tn.Tn.dims) in
  let indices = List.init rank ~f:(fun _ -> Indexing.get_symbol ()) in
  (Expand_zero { tn; indices }, indices)

let split_reduce ~axis ~target ~num_blocks =
  let block_index = Indexing.get_symbol () and inner_index = Indexing.get_symbol () in
  let rank = Array.length (Lazy.force target.Tn.dims) in
  let combine_indices = List.init rank ~f:(fun _ -> Indexing.get_symbol ()) in
  ( Split_reduce { axis; target; num_blocks; block_index; inner_index; combine_indices },
    block_index,
    inner_index,
    combine_indices )

(** {2 Index substitution}

    A symbol is replaced by an affine combination [Σ terms + offset]: [Split] substitutes
    [i := factor*i_o + i_i], materializing [Unroll] substitutes [i := k]. The two places a loop
    symbol can occur are index vectors ([axis_index array]) and [Embed_index] scalars; the
    traversals below cover both. *)

type affine_subst = { terms : (int * Indexing.symbol) list; offset : int }

let rec add_term acc (c, s) =
  match acc with
  | [] -> [ (c, s) ]
  | (c', s') :: tl when Indexing.equal_symbol s s' -> (c + c', s') :: tl
  | hd :: tl -> hd :: add_term tl (c, s)

(* Merge duplicate symbols, drop zero coefficients, and restore the [Fixed_idx] / [Iterator] /
   [Affine] canonical forms ([Affine] must have >1 term or a coefficient other than 0/1). *)
let normalize_affine ~terms ~offset : Indexing.axis_index =
  let terms = List.fold terms ~init:[] ~f:add_term |> List.filter ~f:(fun (c, _) -> c <> 0) in
  match (terms, offset) with
  | [], _ -> Indexing.Fixed_idx offset
  | [ (1, s) ], 0 -> Indexing.Iterator s
  | _ -> Indexing.Affine { symbols = terms; offset }

let subst_axis_index ~sym ~(by : affine_subst) (idx : Indexing.axis_index) : Indexing.axis_index =
  match idx with
  | Indexing.Fixed_idx _ | Indexing.Sub_axis -> idx
  | Indexing.Iterator s ->
      if Indexing.equal_symbol s sym then normalize_affine ~terms:by.terms ~offset:by.offset
      else idx
  | Indexing.Affine { symbols; offset } ->
      if List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym) then
        let terms, offset =
          List.fold symbols ~init:([], offset) ~f:(fun (acc, off) (c, s) ->
              if Indexing.equal_symbol s sym then
                (acc @ List.map by.terms ~f:(fun (bc, bs) -> (c * bc, bs)), off + (c * by.offset))
              else (acc @ [ (c, s) ], off))
        in
        normalize_affine ~terms ~offset
      else idx
  | Indexing.Concat _ -> idx (* Eliminated during lowering; scheduled loops never carry these. *)

let rec map_code ~fidx (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
      llc
  | Seq (a, b) -> Seq (map_code ~fidx a, map_code ~fidx b)
  | For_loop fc -> For_loop { fc with body = map_code ~fidx fc.body }
  | Scan_loop sc ->
      Scan_loop
        {
          sc with
          carried = List.map sc.carried ~f:(fun c -> { c with init = map_scalar ~fidx c.init });
          body = map_code ~fidx sc.body;
        }
  | Set { tn; idcs; llsc; debug } ->
      Set { tn; idcs = Array.map idcs ~f:fidx; llsc = map_scalar ~fidx llsc; debug }
  | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, p; llsc; debug } ->
      Set_dynamic
        {
          tn;
          idcs = Array.map idcs ~f:fidx;
          dyn_axis;
          dyn_value = (map_scalar ~fidx v, p);
          llsc = map_scalar ~fidx llsc;
          debug;
        }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = a, p; debug } ->
      Set_from_vec
        {
          tn;
          idcs = Array.map idcs ~f:fidx;
          length;
          vec_unop;
          arg = (map_scalar ~fidx a, p);
          debug;
        }
  | Set_local (id, llsc) -> Set_local (id, map_scalar ~fidx llsc)
  | Tile_mma ({ d = d_tn, d_idcs; a = a_tn, a_idcs; b = b_tn, b_idcs; fallback; _ } as tm) ->
      Tile_mma
        {
          tm with
          d = (d_tn, Array.map d_idcs ~f:fidx);
          a = (a_tn, Array.map a_idcs ~f:fidx);
          b = (b_tn, Array.map b_idcs ~f:fidx);
          (* The fallback's loop symbols are fresh and bound inside it; only free (outer) symbols
             are substituted, exactly like the base indices. *)
          fallback = map_code ~fidx fallback;
        }
  | If { cond = c, p; body } -> If { cond = (map_scalar ~fidx c, p); body = map_code ~fidx body }

and map_scalar ~fidx (llsc : Low_level.scalar_t) : Low_level.scalar_t =
  let open Low_level in
  match llsc with
  | Local_scope { id; body; orig_indices; mint } ->
      Local_scope
        { id; body = map_code ~fidx body; orig_indices = Array.map orig_indices ~f:fidx; mint }
  | Get_local _ | Constant _ | Constant_bits _ -> llsc
  | Get (tn, idcs) -> Get (tn, Array.map idcs ~f:fidx)
  | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, p } ->
      Get_dynamic
        { tn; idcs = Array.map idcs ~f:fidx; dyn_axis; dyn_value = (map_scalar ~fidx v, p) }
  | Get_merge_buffer (tn, idcs) -> Get_merge_buffer (tn, Array.map idcs ~f:fidx)
  | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
      Ternop (op, (map_scalar ~fidx a, pa), (map_scalar ~fidx b, pb), (map_scalar ~fidx c, pc))
  | Binop (op, (a, pa), (b, pb)) -> Binop (op, (map_scalar ~fidx a, pa), (map_scalar ~fidx b, pb))
  | Unop (op, (a, pa)) -> Unop (op, (map_scalar ~fidx a, pa))
  | Embed_index idx -> Embed_index (fidx idx)

(** {2 The transforms} *)

type floop = {
  index : Indexing.symbol;
  from_ : int;
  to_ : int;
  body : Low_level.t;
  axis : Low_level.axis_type;
}
(* A copy of [For_loop]'s inlined record (which cannot escape its match). *)

let for_loop { index; from_; to_; body; axis } =
  Low_level.For_loop { index; from_; to_; body; axis }

(* EVERY [For_loop] binding [axis] (in preorder), each paired with an environment [~f] threaded over
   its enclosing loops (outermost first) — e.g. the loop ranges {!partition_breakpoints} needs to
   bound the non-axis symbols of a guard. Like [rewrite_loop] this does not descend into a match's
   own body (a loop cannot rebind the symbol it is nested in).

   Locating must cover exactly what rewriting covers, in BOTH directions, or a probe disagrees with
   the op it informs (gh-ocannl-668):

   - As deep. [in_scopes] (the default) descends into [Local_scope] bodies reached from the scalar
   positions of Set / Set_dynamic / Set_local / Set_from_vec, matching [rewrite_loop]'s reach since
   gh-ocannl-639 — the accumulation mints of [Unroll ~materialize:true] and [Partition] wrap
   segment/inner loops in the accumulator's scope, and a statement-only walk reports those absent.
   Pass [~in_scopes:false] only where statement nesting is part of the caller's contract (see
   [apply_split_reduce]). - As wide. [rewrite_loop] rewrites every copy, so a probe that stops at
   the first one speaks for a fraction of what the op does: a materializing [Unroll] leaves one copy
   of the inner loop per unrolled step, each with the outer index substituted by a different
   constant, so copies of one guard flip at different points.

   Neither walk enters [If] conditions or [Tile_mma] fallbacks. *)
let find_loops_env ?(in_scopes = true) axis ~(init : 'a) ~(f : 'a -> floop -> 'a)
    (llc : Low_level.t) : (Low_level.t * 'a) list =
  let open Low_level in
  let found = ref [] in
  let rec go env llc =
    match llc with
    | For_loop { index; _ } when Indexing.equal_symbol index axis -> found := (llc, env) :: !found
    | For_loop { index; from_; to_; body; axis = ty } ->
        go (f env { index; from_; to_; body; axis = ty }) body
    | Seq (a, b) ->
        go env a;
        go env b
    | If { body; _ } -> go env body
    | Set { llsc; _ } | Set_local (_, llsc) -> scalar env llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scalar env v;
        scalar env llsc
    | Set_from_vec { arg = a, _; _ } -> scalar env a
    | _ -> ()
  and scalar env (llsc : scalar_t) =
    if in_scopes then
      match llsc with
      | Local_scope { body; _ } -> go env body
      | Get_dynamic { dyn_value = v, _; _ } -> scalar env v
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          scalar env a;
          scalar env b;
          scalar env c
      | Binop (_, (a, _), (b, _)) ->
          scalar env a;
          scalar env b
      | Unop (_, (a, _)) -> scalar env a
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
  in
  go init llc;
  List.rev !found

let find_loops ?in_scopes axis (llc : Low_level.t) : Low_level.t list =
  List.map ~f:fst (find_loops_env ?in_scopes axis ~init:() ~f:(fun () _ -> ()) llc)

let find_loop ?in_scopes axis (llc : Low_level.t) : Low_level.t option =
  List.hd (find_loops ?in_scopes axis llc)

(* Rewrites every [For_loop] whose index is [sym] (a materializing transform duplicates a loop per
   copy, and a later op must reach all the copies). Loops inside [Local_scope] bodies are IN scope
   since gh-ocannl-639: the accumulation mints of [Unroll ~materialize:true] and [Partition] wrap
   segment/inner loops in the accumulator's scope, and those loops must stay addressable by
   subsequent left-to-right schedule operations ([Schedule.partition] returns the segment symbols
   precisely so callers can target them). Annotated loops inside scopes are still rejected
   downstream by [validate_parallel], loudly. *)
let rewrite_loop ~what ~sym ~(f : floop -> Low_level.t) (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  let found = ref false in
  let rec go llc =
    match llc with
    | For_loop { index; from_; to_; body; axis } when Indexing.equal_symbol index sym ->
        found := true;
        f { index; from_; to_; body; axis }
    | For_loop fc -> For_loop { fc with body = go fc.body }
    | Seq (a, b) -> Seq (go a, go b)
    | If { cond; body } -> If { cond; body = go body }
    | Set s -> Set { s with llsc = go_scalar s.llsc }
    | Set_dynamic ({ dyn_value = v, vp; llsc; _ } as s) ->
        Set_dynamic { s with dyn_value = (go_scalar v, vp); llsc = go_scalar llsc }
    | Set_local (id, llsc) -> Set_local (id, go_scalar llsc)
    | Set_from_vec ({ arg = a, ap; _ } as s) -> Set_from_vec { s with arg = (go_scalar a, ap) }
    | other -> other
  and go_scalar (llsc : scalar_t) : scalar_t =
    match llsc with
    | Local_scope ls -> Local_scope { ls with body = go ls.body }
    | Get_dynamic ({ dyn_value = v, vp; _ } as g) ->
        Get_dynamic { g with dyn_value = (go_scalar v, vp) }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (go_scalar a, pa), (go_scalar b, pb), (go_scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (go_scalar a, pa), (go_scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (go_scalar a, pa))
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
        llsc
  in
  let result = go llc in
  if not !found then
    invalid_arg (what ^ ": no For_loop with index " ^ Indexing.symbol_ident sym ^ " in this routine");
  result

(* Fresh scope ids for scalar locals declared within [llc] (binders: [Declare_local],
   [Local_scope]). Materializing [Unroll] duplicates its body: without refreshing, sibling copies
   would declare the same scope id — rendered as duplicate declarations in one C block — and confuse
   the scope-id-keyed CSE/hoisting passes. References to locals declared outside [llc] are left
   alone. *)
let refresh_scopes (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  let mapping = ref [] in
  let bind id =
    if not (List.Assoc.mem !mapping ~equal:equal_scope_id id) then
      mapping := (id, get_scope id.tn) :: !mapping
  in
  let rec collect llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> ()
    | Tile_mma { fallback; _ } -> collect fallback
    | Declare_local { id; _ } -> bind id
    | Seq (a, b) ->
        collect a;
        collect b
    | For_loop { body; _ } | If { body; _ } -> collect body
    | Scan_loop { carried; body; _ } ->
        (* gh-ocannl-696: the carried pair is a declaration of the construct -- a duplicated scan
           needs fresh state locals like a duplicated [Declare_local]. *)
        List.iter carried ~f:(fun c ->
            bind c.prev;
            bind c.next;
            collect_scalar c.init);
        collect body
    | Set { llsc; _ } | Set_local (_, llsc) -> collect_scalar llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        collect_scalar v;
        collect_scalar llsc
    | Set_from_vec { arg = a, _; _ } -> collect_scalar a
  and collect_scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        bind id;
        collect body
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get_dynamic { dyn_value = v, _; _ } -> collect_scalar v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        collect_scalar a;
        collect_scalar b;
        collect_scalar c
    | Binop (_, (a, _), (b, _)) ->
        collect_scalar a;
        collect_scalar b
    | Unop (_, (a, _)) -> collect_scalar a
  in
  collect llc;
  if List.is_empty !mapping then llc
  else
    let subst id =
      match List.Assoc.find !mapping ~equal:equal_scope_id id with Some id' -> id' | None -> id
    in
    let rec code llc =
      match llc with
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> llc
      | Tile_mma ({ fallback; _ } as tm) -> Tile_mma { tm with fallback = code fallback }
      | Declare_local { id; needs_init } -> Declare_local { id = subst id; needs_init }
      | Seq (a, b) -> Seq (code a, code b)
      | For_loop fc -> For_loop { fc with body = code fc.body }
      | Scan_loop sc ->
          Scan_loop
            {
              sc with
              carried =
                List.map sc.carried ~f:(fun c ->
                    { prev = subst c.prev; next = subst c.next; init = scalar c.init });
              body = code sc.body;
            }
      | If { cond = c, p; body } -> If { cond = (scalar c, p); body = code body }
      | Set ({ llsc; _ } as s) -> Set { s with llsc = scalar llsc }
      | Set_dynamic ({ dyn_value = v, p; llsc; _ } as sd) ->
          Set_dynamic { sd with dyn_value = (scalar v, p); llsc = scalar llsc }
      | Set_local (id, llsc) -> Set_local (subst id, scalar llsc)
      | Set_from_vec ({ arg = a, p; _ } as sv) -> Set_from_vec { sv with arg = (scalar a, p) }
    and scalar (llsc : scalar_t) : scalar_t =
      match llsc with
      | Local_scope { id; body; orig_indices; mint } ->
          Local_scope { id = subst id; body = code body; orig_indices; mint }
      | Get_local id -> Get_local (subst id)
      | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> llsc
      | Get_dynamic ({ dyn_value = v, p; _ } as gd) ->
          Get_dynamic { gd with dyn_value = (scalar v, p) }
      | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
          Ternop (op, (scalar a, pa), (scalar b, pb), (scalar c, pc))
      | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
      | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
    in
    code llc

(* gh-ocannl-485 (PADTO): a pad guard recognized around the micro-kernel's accumulation, reduced to
   the accumulator's coordinates. The guarded index is [pm_terms + 1*role + pm_offset] where [role]
   is the [i] ([pm_row = true]) or [j] ([pm_row = false]) micro symbol; the guard holds iff that
   index is [< pm_bound]. [contract_tensorized_accumulator] re-emits the guard over its fresh
   fragment symbols at the accumulator transfer sites (Where-form 0-fill on the init-load, statement
   [If] on the store-back), so the intrinsic path computes the full padded block into scratch and
   only the valid region round-trips through the accumulator. Reduction-axis ([k]) guards are
   validated (both operands zero-fringe staged tiles, so the padded contributions are exact zeros)
   and dropped, not represented. *)
type pad_mask = {
  pm_row : bool;
  pm_terms : (int * Indexing.symbol) list;
  pm_offset : int;
  pm_bound : int;
}

(* docs/proposals/tensorize-mma.md §3: replace the innermost serial [i × j × k] matmul micro-kernel
   — whose body is a single accumulation [d[...] += a[...] * b[...]] (plain-add or FMA form, as
   [optimize]'s simplify leaves it), possibly under pad/remainder guards (gh-ocannl-485) — with a
   [Tile_mma] block statement wrapped in a fresh extent-[simd_width] [Workgroup] lane loop. The
   statement covers the whole [m×n×k] block, so fragment residency across the reduction is an
   intra-statement codegen concern; the original (guarded) nest becomes the scalar [fallback].
   Divisibility by the backend's intrinsic tile is checked at emission ([mma_syntax] declines per
   call and the fallback runs), since the schedule layer is backend-agnostic. Guards around the
   accumulation are parsed as pad masks (returned for the contraction) or rejected loudly. *)
let tensorize_llc ~(zero_fringe : Tn.t -> bool) ~i ~j ~k ~lane ~simd_width ~tile (llc : Low_level.t)
    : Low_level.t * pad_mask list =
  let open Low_level in
  let out_masks = ref [] in
  let llc =
    rewrite_loop ~what:"Schedule.Tensorize" ~sym:i llc ~f:(fun ifc ->
        if simd_width <= 0 then invalid_arg "Schedule.Tensorize: simd_width must be positive";
        let strip body =
          List.filter (flat_lines [ body ]) ~f:(function Noop | Comment _ -> false | _ -> true)
        in
        (* Pad/remainder guards may wrap the inner loops or the accumulation itself; skim them off,
           collecting the conditions for mask parsing below. *)
        let guard_conds = ref [] in
        let rec skim body =
          match strip body with
          | [ If { cond = c, _; body } ] ->
              guard_conds := c :: !guard_conds;
              skim body
          | stmts -> stmts
        in
        (* What the body actually is, one line per statement — without it a decline says only "not
           perfectly nested", which does not distinguish an intervening loop from a sibling
           statement (a staged tile copy, an inlined cast twin) that landed inside the nest. *)
        let describe stmts =
          String.concat ~sep:"; "
            (List.map stmts ~f:(function
              | For_loop { index; from_; to_; axis; _ } ->
                  Printf.sprintf "For_loop %s[%d..%d] %s" (Indexing.symbol_ident index) from_ to_
                    (Low_level.axis_type_label axis)
              | Set { tn; _ } -> "Set " ^ Tn.debug_name tn
              | Zero_out tn -> "Zero_out " ^ Tn.debug_name tn
              | Tile_mma { d = tn, _; _ } -> "Tile_mma " ^ Tn.debug_name tn
              | other -> List.hd_exn (String.split ~on:' ' (Sexp.to_string (sexp_of_t other)))))
        in
        let nested ~of_ sym body =
          match skim body with
          | [ For_loop { index; from_; to_; body; axis } ] when Indexing.equal_symbol index sym ->
              { index; from_; to_; body; axis }
          | stmts ->
              invalid_arg
                ("Schedule.Tensorize: loop " ^ Indexing.symbol_ident sym
               ^ " must be exactly the body of loop " ^ Indexing.symbol_ident of_
               ^ " (a perfectly nested i x j x k micro-kernel); that body is instead: "
               ^ describe stmts)
        in
        let jfc = nested ~of_:i j ifc.body in
        let kfc = nested ~of_:j k jfc.body in
        List.iter
          [ { ifc with body = Noop }; { jfc with body = Noop }; { kfc with body = Noop } ]
          ~f:(fun fc ->
            if (not (equal_axis_type fc.axis Serial)) || fc.from_ <> 0 then
              invalid_arg
                ("Schedule.Tensorize: loop " ^ Indexing.symbol_ident fc.index
               ^ " must be Serial starting at 0"));
        let d_tn, d_idcs, llsc =
          match skim kfc.body with
          | [ Set { tn; idcs; llsc; _ } ] -> (tn, idcs, llsc)
          | _ ->
              invalid_arg
                "Schedule.Tensorize: the micro-kernel body must be a single accumulation Set"
        in
        let is_d_read (sc : Low_level.scalar_t) =
          match sc with
          | Get (tn, idcs) -> Tn.equal tn d_tn && Array.equal Indexing.equal_axis_index idcs d_idcs
          | _ -> false
        in
        let get_operand (sc : Low_level.scalar_t) =
          match sc with Get (tn, idcs) -> Some (tn, idcs) | _ -> None
        in
        let operands =
          match llsc with
          | Ternop (Ops.FMA, (x, _), (y, _), (acc, _)) when is_d_read acc ->
              Option.both (get_operand x) (get_operand y)
          | Binop (Ops.Add, (acc, _), (Binop (Ops.Mul, (x, _), (y, _)), _)) when is_d_read acc ->
              Option.both (get_operand x) (get_operand y)
          | Binop (Ops.Add, (Binop (Ops.Mul, (x, _), (y, _)), _), (acc, _)) when is_d_read acc ->
              Option.both (get_operand x) (get_operand y)
          | _ -> None
        in
        let x_op, y_op =
          match operands with
          | Some ops -> ops
          | None ->
              invalid_arg
                ("Schedule.Tensorize: the micro-kernel must accumulate a product of reads into "
               ^ Tn.debug_name d_tn ^ " (d[...] += a[...] * b[...], plain-add or FMA form)")
        in
        let mentions sym (idx : Indexing.axis_index) =
          match idx with
          | Indexing.Iterator s -> Indexing.equal_symbol s sym
          | Indexing.Affine { symbols; _ } ->
              List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym)
          | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
        in
        let coeff sym (idx : Indexing.axis_index) =
          match idx with
          | Indexing.Iterator s when Indexing.equal_symbol s sym -> 1
          | Indexing.Affine { symbols; _ } ->
              List.sum
                (module Int)
                symbols
                ~f:(fun (c, s) -> if Indexing.equal_symbol s sym then c else 0)
          | _ -> 0
        in
        (* Index discipline: the tile is a 2-D slice of the operand — [col] appears with coefficient
           1 exactly in the minor component [rank-1] (elements of a tile line are contiguous), [row]
           with coefficient 1 exactly in one earlier component (the tnode's second-to-last axis in
           the plain case; further out when interior batch axes sit between the roles,
           gh-ocannl-528), and the third symbol not at all. Outer-loop terms (the block base, batch
           coordinates included) may appear anywhere. Returns the major-axis leading-dimension
           stride in elements — the fragment loads' [ldm]. *)
        let role_ld (tn, idcs) ~row ~col : int option =
          let rank = Array.length idcs in
          let owned_axis sym =
            let ps = Array.filter_mapi idcs ~f:(fun p idx -> Option.some_if (mentions sym idx) p) in
            match Array.to_list ps with [ p ] when coeff sym idcs.(p) = 1 -> Some p | _ -> None
          in
          if rank < 2 then None
          else
            match (owned_axis row, owned_axis col) with
            | Some pr, Some pc
              when pc = rank - 1
                   && pr < pc
                   && List.for_all [ i; j; k ] ~f:(fun s ->
                       Indexing.equal_symbol s row || Indexing.equal_symbol s col
                       || not (Array.exists idcs ~f:(mentions s))) ->
                let dims = Lazy.force tn.Tn.dims in
                let ld = ref 1 in
                for x = pr + 1 to rank - 1 do
                  ld := !ld * dims.(x)
                done;
                Some !ld
            | _ -> None
        in
        let d_ld =
          match role_ld (d_tn, d_idcs) ~row:i ~col:j with
          | Some ld -> ld
          | None ->
              invalid_arg
                ("Schedule.Tensorize: accumulator " ^ Tn.debug_name d_tn
               ^ " must be indexed [..., i, ..., j] with [j] on its last axis (unit coefficients)")
        in
        (* Operand roles, including transposed storage: [a] is [..., i, ..., k] ([ta = false]) or
           [..., k, ..., i] ([ta = true]); [b] is [..., k, ..., j] ([tb = false]) or [..., j, ...,
           k] ([tb = true]). An operand matches at most one role and orientation ([role_ld] requires
           both role symbols owned at valid positions and the third absent), so the assignment is
           unambiguous. *)
        let a_role op =
          match role_ld op ~row:i ~col:k with
          | Some ld -> Some (false, ld)
          | None -> Option.map (role_ld op ~row:k ~col:i) ~f:(fun ld -> (true, ld))
        in
        let b_role op =
          match role_ld op ~row:k ~col:j with
          | Some ld -> Some (false, ld)
          | None -> Option.map (role_ld op ~row:j ~col:k) ~f:(fun ld -> (true, ld))
        in
        let a_op, ta, a_ld, b_op, tb, b_ld =
          match (a_role x_op, b_role y_op) with
          | Some (ta, a_ld), Some (tb, b_ld) -> (x_op, ta, a_ld, y_op, tb, b_ld)
          | _ -> (
              match (a_role y_op, b_role x_op) with
              | Some (ta, a_ld), Some (tb, b_ld) -> (y_op, ta, a_ld, x_op, tb, b_ld)
              | _ ->
                  (* Naming the two tnodes is not enough to act on: the discipline is about the
                     INDEX EXPRESSIONS, so report them against the micro-kernel symbols. *)
                  let show (tn, idcs) =
                    let dims = Lazy.force tn.Tn.dims in
                    Tn.debug_name tn ^ ":"
                    ^ Sexp.to_string ([%sexp_of: int array] dims)
                    ^ "["
                    ^ String.concat ~sep:", "
                        (Array.to_list idcs
                        |> List.map ~f:(fun idx -> Sexp.to_string (Indexing.sexp_of_axis_index idx))
                        )
                    ^ "]"
                  in
                  invalid_arg
                    ("Schedule.Tensorize: operands of the product must be indexed [..., i, ..., k] \
                      (or transposed [..., k, ..., i]) and [..., k, ..., j] (or transposed [..., \
                      j, ..., k]) with the second role symbol on the last axis (unit \
                      coefficients); with i=" ^ Indexing.symbol_ident i ^ " j="
                   ^ Indexing.symbol_ident j ^ " k=" ^ Indexing.symbol_ident k
                   ^ " the operands are " ^ show x_op ^ ", " ^ show y_op))
        in
        (* Pad-mask parsing (gh-ocannl-485). Every skimmed guard must be a one-sided comparison
           [affine < bound] whose affine part carries exactly one micro symbol with coefficient 1
           (the shape [Pad] and [Split]'s remainder guard construct); anything else is rejected —
           the guard would silently change meaning if dropped. Masked roles require every operand
           mentioning the role symbol to be a zero-fringe staged tile read at the plain iterator
           with a covering dim: the intrinsic path reads the full padded extent, so out-of-range
           slots must exist and hold the additive identity. *)
        let extent_of s =
          if Indexing.equal_symbol s i then ifc.to_ + 1
          else if Indexing.equal_symbol s j then jfc.to_ + 1
          else kfc.to_ + 1
        in
        let validate_padded_operand s (tn, idcs) =
          if Array.exists idcs ~f:(mentions s) then (
            if not (zero_fringe tn) then
              invalid_arg
                ("Schedule.Tensorize: pad guard on " ^ Indexing.symbol_ident s
               ^ " requires operand " ^ Tn.debug_name tn
               ^ " to be a zero-fringe staged tile (Stage the operand over the padded loops first)"
                );
            let dims = Lazy.force tn.Tn.dims in
            Array.iteri idcs ~f:(fun p idx ->
                if mentions s idx then
                  match idx with
                  | Indexing.Iterator _ when dims.(p) >= extent_of s -> ()
                  | _ ->
                      invalid_arg
                        ("Schedule.Tensorize: pad guard on " ^ Indexing.symbol_ident s
                       ^ " requires operand " ^ Tn.debug_name tn
                       ^ " to read the padded axis as a plain covering iterator")))
        in
        let parse_mask (c : Low_level.scalar_t) =
          let reject () =
            invalid_arg
              "Schedule.Tensorize: unrecognized guard around the micro-kernel (expected a pad or \
               remainder guard [affine(one micro symbol, unit coefficient) < constant])"
          in
          match c with
          | Binop (Ops.Cmplt, (Embed_index idx, _), (Constant bound, _)) when Float.is_integer bound
            -> (
              let terms, offset =
                match idx with
                | Indexing.Iterator s -> ([ (1, s) ], 0)
                | Indexing.Affine { symbols; offset } ->
                    (Indexing.coalesce_affine_terms symbols, offset)
                | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> reject ()
              in
              let is_micro s = List.exists [ i; j; k ] ~f:(Indexing.equal_symbol s) in
              let micro_terms, outer_terms =
                List.partition_tf terms ~f:(fun (_, s) -> is_micro s)
              in
              match micro_terms with
              | [ (1, s) ] ->
                  let bound = Float.to_int bound in
                  validate_padded_operand s a_op;
                  validate_padded_operand s b_op;
                  if Indexing.equal_symbol s k then
                    (* Both operands' padded [k] slots are exact zeros, so the extra accumulated
                       contributions are exact zeros: the guard is discharged, not represented. *)
                    None
                  else
                    Some
                      {
                        pm_row = Indexing.equal_symbol s i;
                        pm_terms = outer_terms;
                        pm_offset = offset;
                        pm_bound = bound;
                      }
              | _ -> reject ())
          | _ -> reject ()
        in
        out_masks := List.filter_map !guard_conds ~f:parse_mask;
        let zero = { terms = []; offset = 0 } in
        let base idcs =
          Array.map idcs ~f:(fun idx ->
              subst_axis_index ~sym:i ~by:zero
                (subst_axis_index ~sym:j ~by:zero (subst_axis_index ~sym:k ~by:zero idx)))
        in
        For_loop
          {
            index = lane;
            from_ = 0;
            to_ = simd_width - 1;
            axis = Workgroup;
            body =
              Tile_mma
                {
                  d = (d_tn, base d_idcs);
                  a = (fst a_op, base (snd a_op));
                  b = (fst b_op, base (snd b_op));
                  ta;
                  tb;
                  m = ifc.to_ + 1;
                  n = jfc.to_ + 1;
                  k = kfc.to_ + 1;
                  ldd = d_ld;
                  lda = a_ld;
                  ldb = b_ld;
                  tile;
                  lane;
                  fallback = for_loop ifc;
                };
          })
  in
  (llc, !out_masks)

(* The accumulation mints below forward [Low_level.loop_bounds llc] to [Low_level.peel_accum_nest],
   so a runtime-extent guard ([i < s], gh-490) does not stop them: [s] is bound outside every loop
   and cannot select among enclosing iterations. Derived here rather than certified by [apply]'s
   caller, because declining is NOT neutral for these mints the way it is in codegen -- a refused
   mint makes [Unroll ~materialize:true] round-trip the accumulator through storage per copy and
   [Partition] turn its segment seams into narrowing points, so on narrow storage the candidate
   stops agreeing with the serial baseline, which is the very invariant they exist to keep
   (gh-ocannl-693 review rounds 6-7). *)
let apply_op (llc : Low_level.t) (op : optop) : Low_level.t =
  let loop_ranges = lazy (Low_level.loop_bounds llc) in
  let open Low_level in
  match op with
  | Stage _ | Privatize _ | Fuse_epilogue _ | Tensorize _ | Split_reduce _ ->
      assert false (* Handled by [apply_opt_op]: they need the whole [optimized]. *)
  | Split { axis; factor; outer; inner; outer_index; inner_index } ->
      rewrite_loop ~what:"Schedule.Split" ~sym:axis llc ~f:(fun fc ->
          if factor <= 0 then invalid_arg "Schedule.Split: factor must be positive";
          if fc.from_ <> 0 then
            invalid_arg
              ("Schedule.Split: loop " ^ Indexing.symbol_ident axis
             ^ " must start at 0 (lowering guarantees this)");
          let n = fc.to_ + 1 in
          let terms = [ (factor, outer_index); (1, inner_index) ] in
          let by = { terms; offset = 0 } in
          let body = map_code ~fidx:(subst_axis_index ~sym:axis ~by) fc.body in
          let body =
            if n % factor = 0 then body
            else
              (* Remainder guard, construct-then-fold: [apply]'s trailing [simplify_llc] erases it
                 when the loop extents prove it (i.e. exactly when [factor] divides [n]; here it
                 does not, so the guard survives interval folding — but a later transform of an
                 enclosing loop can still change the environment, so we keep the uniform discipline
                 of always folding rather than special-casing). *)
              let iprec = Ops.index_prec () in
              let cond =
                Binop
                  ( Ops.Cmplt,
                    (Embed_index (Indexing.Affine { symbols = terms; offset = 0 }), iprec),
                    (Constant (Float.of_int n), iprec) )
              in
              If { cond = (cond, iprec); body }
          in
          For_loop
            {
              index = outer_index;
              from_ = 0;
              to_ = ((n + factor - 1) / factor) - 1;
              axis = outer;
              body =
                For_loop { index = inner_index; from_ = 0; to_ = factor - 1; axis = inner; body };
            })
  | Swap { outer; inner } ->
      rewrite_loop ~what:"Schedule.Swap" ~sym:outer llc ~f:(fun ofc ->
          match ofc.body with
          | For_loop { index; from_; to_; body; axis } when Indexing.equal_symbol index inner ->
              for_loop { index; from_; to_; axis; body = for_loop { ofc with body } }
          | _ ->
              invalid_arg
                ("Schedule.Swap: loops " ^ Indexing.symbol_ident outer ^ " and "
               ^ Indexing.symbol_ident inner
               ^ " are not perfectly nested (the outer body must be exactly the inner loop)"))
  | Retype { axis; ty } ->
      rewrite_loop ~what:"Schedule.Retype" ~sym:axis llc ~f:(fun fc ->
          (match ty with
          | Grid | Workgroup | Workgroup_reduce ->
              if fc.from_ <> 0 then
                invalid_arg
                  ("Schedule.Retype: hardware-annotated loop " ^ Indexing.symbol_ident axis
                 ^ " must start at 0")
          | Serial | Unrolled | Vectorized -> ());
          for_loop { fc with axis = ty })
  | Unroll { axis; materialize = false } ->
      rewrite_loop ~what:"Schedule.Unroll" ~sym:axis llc ~f:(fun fc ->
          for_loop { fc with axis = Unrolled })
  | Unroll { axis; materialize = true } ->
      rewrite_loop ~what:"Schedule.Unroll" ~sym:axis llc ~f:(fun fc ->
          let copy k body =
            refresh_scopes
            @@ map_code
                 ~fidx:(subst_axis_index ~sym:axis ~by:{ terms = []; offset = fc.from_ + k })
                 body
          in
          (* gh-ocannl-639: when the unrolled loop is (a nest ending in) a single accumulation
             statement, unroll into the scope-local form virtualization gives virtual accumulators —
             one [Local_scope] holding the running value, one copy of the update per iteration, one
             store — instead of one read-modify-write [Set] per copy. This is where the provenance
             lives: only genuine unrolled copies of ONE source assignment may share an accumulator
             residency (a codegen pass looking at adjacent same-cell [Set]s cannot tell them apart
             from two user-authored assignments, whose separate stores are their semantics). The
             rewrite is numerics-neutral relative to the not-unrolled loop under every
             compute-precision policy: the scope renders at the same precision the policy gives a
             serial reduction's accumulator ([C_syntax.scope_prec_of], including the rng-census and
             storage=compute cases where that is the storage precision), so the unrolled candidate
             computes exactly what the serial baseline computes — which is the invariant autotune's
             menu depends on (candidates compete on speed, never numerics). *)
          let mint =
            (* Under routine logging the per-iteration [Set] copies ARE the trace: a [Local_scope]
               body renders with [log_set_locals:false], so minting the scope would silence every
               update. The serial baseline's widening declines under logging for the same reason
               (C_syntax.try_localize_serial_reduce), so within a logged regime the unrolled and
               serial candidates still agree — both per-step. *)
            if Utils.debug_log_from_routines () then None
            else
              Low_level.peel_accum_nest ~loop_bounds:(Lazy.force loop_ranges) ~free_of:[ axis ]
                fc.body
          in
          match mint with
          | Some (tn, idcs, base, debug, rebuild) ->
              let id, update_code =
                match base with
                | `Update llsc ->
                    let id = Low_level.get_scope tn in
                    (id, Low_level.Set_local (id, Low_level.subst_accum_read ~tn ~idcs ~id llsc))
                | `Scope (id, rest) ->
                    (* A previous materializing unroll (of a deeper reduction axis) already minted
                       the scope form: reuse its accumulator across this axis's copies too, or the
                       whole-nest residency is lost — the enclosing loop is about to disappear, so
                       codegen's scope-base hoist could not recover it. *)
                    (id, unflat_lines rest)
              in
              let copies =
                List.init (fc.to_ - fc.from_ + 1) ~f:(fun k -> copy k (rebuild update_code))
              in
              Set
                {
                  tn;
                  idcs;
                  llsc =
                    Local_scope
                      {
                        id;
                        body =
                          unflat_lines (Low_level.Set_local (id, Low_level.Get (tn, idcs)) :: copies);
                        orig_indices = idcs;
                        mint = Low_level.Schedule_minted;
                      };
                  debug;
                }
          | None -> unflat_lines (List.init (fc.to_ - fc.from_ + 1) ~f:(fun k -> copy k fc.body)))
  | Pad { axis; to_multiple_of } ->
      (* gh-ocannl-485 (PADTO): extend the loop extent to the next multiple of [to_multiple_of] and
         guard each effectful leaf statement of the body with [If (axis < N)] — the pad iterations
         are no-ops, so the op is unconditionally semantics-preserving. Guards go on the leaves, not
         around the whole body, so barriers inserted later (shared [Stage]) stay under uniform
         control flow, and downstream [Split]s of the padded loop divide cleanly (no remainder
         guard). [apply]'s trailing simplify interval-folds the guards wherever the narrowed
         environment proves them (e.g. after a [Partition] of a block loop at the last fully valid
         block); [Tensorize] recognizes them as pad masks (see {!optop}). *)
      rewrite_loop ~what:"Schedule.Pad" ~sym:axis llc ~f:(fun fc ->
          if to_multiple_of <= 0 then invalid_arg "Schedule.Pad: to_multiple_of must be positive";
          if not (equal_axis_type fc.axis Serial) then
            invalid_arg ("Schedule.Pad: loop " ^ Indexing.symbol_ident axis ^ " must be Serial");
          if fc.from_ <> 0 then
            invalid_arg
              ("Schedule.Pad: loop " ^ Indexing.symbol_ident axis
             ^ " must start at 0 (lowering guarantees this)");
          let n = fc.to_ + 1 in
          let m = (n + to_multiple_of - 1) / to_multiple_of * to_multiple_of in
          if m = n then for_loop fc
          else
            let iprec = Ops.index_prec () in
            let cond =
              Binop
                ( Ops.Cmplt,
                  (Embed_index (Indexing.Iterator axis), iprec),
                  (Constant (Float.of_int n), iprec) )
            in
            let guard body = If { cond = (cond, iprec); body } in
            let rec mask = function
              | Seq (a, b) -> Seq (mask a, mask b)
              | For_loop fc' -> For_loop { fc' with body = mask fc'.body }
              | If { cond; body } -> If { cond; body = mask body }
              (* gh-ocannl-696: guarded whole -- a padded iteration runs none of the scan, and a
                 guard inside its body would make the carried update conditional, which the
                 construct forbids. *)
              | (Set _ | Set_dynamic _ | Set_from_vec _ | Set_local _ | Zero_out _ | Scan_loop _) as
                stmt ->
                  guard stmt
              | (Noop | Comment _ | Declare_local _) as stmt -> stmt
              | Workgroup_barrier ->
                  (* Uniformly reached with or without the guard (the padded loop is Serial), and
                     must stay so: never guard a barrier. *)
                  Workgroup_barrier
              | Staged_compilation _ ->
                  invalid_arg
                    ("Schedule.Pad: opaque Staged_compilation in the body of "
                   ^ Indexing.symbol_ident axis)
              | Tile_mma _ ->
                  invalid_arg
                    ("Schedule.Pad: apply Pad before Tensorize (Tile_mma in the body of "
                   ^ Indexing.symbol_ident axis ^ ")")
            in
            for_loop { fc with to_ = m - 1; body = mask fc.body })
  | Partition { axis; breakpoints; segment_indices } ->
      (* gh-ocannl-508: index-set splitting. Segment ranges stay absolute (no rebasing to 0), so the
         substitution is a pure rename of the loop symbol and no index arithmetic changes; each
         segment's narrowed range then lets [apply]'s trailing [simplify_llc] interval-fold the
         guards it decides — statement [If]s and scalar [Where] range guards alike — giving
         guard-free specialized segment nests without any specialization logic here. *)
      rewrite_loop ~what:"Schedule.Partition" ~sym:axis llc ~f:(fun fc ->
          if not (equal_axis_type fc.axis Serial) then
            invalid_arg
              ("Schedule.Partition: loop " ^ Indexing.symbol_ident axis ^ " must be Serial");
          if List.is_empty breakpoints then
            invalid_arg "Schedule.Partition: breakpoints must be non-empty";
          if List.length segment_indices <> List.length breakpoints + 1 then
            invalid_arg "Schedule.Partition: needs one segment index per segment (see {!partition})";
          let starts = fc.from_ :: breakpoints in
          let stops = List.map breakpoints ~f:(fun b -> b - 1) @ [ fc.to_ ] in
          if not (List.for_all2_exn starts stops ~f:(fun lo hi -> lo <= hi)) then
            invalid_arg
              (Printf.sprintf
                 "Schedule.Partition: breakpoints must be strictly increasing and strictly inside \
                  the loop range [%d, %d]"
                 fc.from_ fc.to_);
          let segment s lo hi body =
            (* Sibling segments duplicate the body: refresh scalar-local scope ids exactly as
               materializing [Unroll] does, so copies do not redeclare the same scope id. *)
            let body =
              refresh_scopes
              @@ map_code
                   ~fidx:(subst_axis_index ~sym:axis ~by:{ terms = [ (1, s) ]; offset = 0 })
                   body
            in
            For_loop { index = s; from_ = lo; to_ = hi; body; axis = Serial }
          in
          (* gh-ocannl-639: partitioning a recognized accumulation nest carries ONE accumulator
             scope across all the segments — Partition is an index-set specialization, not a
             partial-reduction boundary, so the segment seams must not become narrowing points
             (unlike a Split, whose halves stay one nest the widening covers whole). Same recognizer
             and same declines as [Unroll ~materialize:true] above. *)
          let mint =
            if Utils.debug_log_from_routines () then None
            else
              Low_level.peel_accum_nest ~loop_bounds:(Lazy.force loop_ranges) ~free_of:[ axis ]
                fc.body
          in
          match mint with
          | Some (tn, idcs, base, debug, rebuild) ->
              let id, update_code =
                match base with
                | `Update llsc ->
                    let id = Low_level.get_scope tn in
                    (id, Low_level.Set_local (id, Low_level.subst_accum_read ~tn ~idcs ~id llsc))
                | `Scope (id, rest) -> (id, unflat_lines rest)
              in
              let segments =
                List.map3_exn segment_indices starts stops ~f:(fun s lo hi ->
                    segment s lo hi (rebuild update_code))
              in
              Set
                {
                  tn;
                  idcs;
                  llsc =
                    Local_scope
                      {
                        id;
                        body =
                          unflat_lines
                            (Low_level.Set_local (id, Low_level.Get (tn, idcs)) :: segments);
                        orig_indices = idcs;
                        mint = Low_level.Schedule_minted;
                      };
                  debug;
                }
          | None ->
              unflat_lines
                (List.map3_exn segment_indices starts stops ~f:(fun s lo hi ->
                     segment s lo hi fc.body)))
  | Expand_zero { tn; indices } ->
      (* Whole-node [Zero_out] is never distributed across hardware threads ([validate_parallel]
         rejects it in multi-threaded kernels); expand it into an ordinary loop nest — over the
         caller-supplied symbols, so subsequent ops in the schedule can split and annotate the
         zeroing with the same geometry as the computation. *)
      let dims = Lazy.force tn.Tn.dims in
      if Array.length dims <> List.length indices then
        invalid_arg
          ("Schedule.Expand_zero: "
          ^ Int.to_string (List.length indices)
          ^ " indices for a rank-"
          ^ Int.to_string (Array.length dims)
          ^ " node");
      let idcs = Array.of_list_map indices ~f:(fun s -> Indexing.Iterator s) in
      let nest =
        List.fold_right
          (List.zip_exn indices (Array.to_list dims))
          ~init:(Set { tn; idcs; llsc = Constant 0.; debug = "" })
          ~f:(fun (s, d) body ->
            For_loop { index = s; from_ = 0; to_ = d - 1; body; axis = Serial })
      in
      let found = ref false in
      let rec go llc =
        match llc with
        | Zero_out tn' when Tn.equal tn tn' ->
            if !found then
              invalid_arg
                ("Schedule.Expand_zero: multiple Zero_out statements for " ^ Tn.debug_name tn);
            found := true;
            nest
        | Seq (a, b) -> Seq (go a, go b)
        | For_loop fc -> For_loop { fc with body = go fc.body }
        | If { cond; body } -> If { cond; body = go body }
        | other -> other
      in
      let result = go llc in
      if not !found then invalid_arg ("Schedule.Expand_zero: no Zero_out of " ^ Tn.debug_name tn);
      result

(** {2 [Stage]: tile staging (schedule-ir-optops §5)}

    The one transform that synthesizes code. [Stage { source; tile_loops; shared }] requires all
    reads of [source] to use one index vector (v1). Each source axis's index is decomposed into a
    tile part (terms over [tile_loops], positive coefficients) and an outer part; source axes with a
    nonempty tile part become tile axes, sized by the tile part's range over the tile loops' extents
    — except that a single-term tile part with coefficient > 1 (a strided window, e.g. a stride-2
    conv's implicit-GEMM row) is {e compacted} (gh-ocannl-502): the tile axis is sized by the loop
    extent and stored/read at coefficient 1, while the load nest's source index (and edge guard)
    keeps the stride, so the packed tile is dense and satisfies [Tensorize]'s unit-coefficient index
    discipline. Hoisted staging rejects compaction (v1). The tile's axes follow the {e tile_loops}
    order (the position of each axis's first tile-part symbol in the list; ties keep source order),
    not the source's: a packing Stage over a transposed operand normalizes its layout —
    [tile_loops = [k; j]] on a [j, k]-stored source packs a [k]-major tile — which is what lets
    packed tiles feed [Tensorize]'s register-tiled micro-kernel with [ta = tb = false]
    (gh-ocannl-469). Every in-tree pipeline before this passed [tile_loops] in source order, where
    the two orders coincide. The load nest is inserted at the deepest loop that must stay outside
    the tile — the innermost loop carrying an outer-part symbol or a reused [Workgroup] tile axis —
    by replacing that loop's body with [loads; barrier; body-with-reads-remapped; barrier]
    ([shared]) or [loads; remapped body] (packing). Cooperative loads reuse [Workgroup]-typed tile
    loops as the cooperating thread indices, iterate [Serial] tile loops under fresh symbols, guard
    each tile axis with an [If (index < dim)] edge guard (construct-then-fold), and — for [shared] —
    restrict redundant loading along non-participating workgroup axes with [If (w == 0)] guards. The
    tile is a fresh [Local]-mode node registered in the traced store (and in [workgroup_shared] when
    [shared]). *)

(* Schedule-minted tile/accumulator nodes live in their own reserved namespace, so their session ids
   are independent of tensor-land ids (allocated from 0 by the [Tensor] session counter). *)
let tile_namespace = "tile"

let fresh_tile_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

let terms_of_index (idx : Indexing.axis_index) : ((int * Indexing.symbol) list * int) option =
  match idx with
  | Indexing.Fixed_idx k -> Some ([], k)
  | Indexing.Iterator s -> Some ([ (1, s) ], 0)
  | Indexing.Affine { symbols; offset } -> Some (symbols, offset)
  | Indexing.Sub_axis -> Some ([], 0)
  | Indexing.Concat _ -> None

(* All reads [Get (source, idcs)] with their enclosing statement-level loop stacks (outermost-first;
   [floop.body] is dummied out). Writes to [source] are rejected. *)
let collect_source_accesses ~source (llc : Low_level.t) :
    (Indexing.axis_index array * floop list) list =
  let open Low_level in
  let acc = ref [] in
  let reject_write tn =
    if Tn.equal tn source then
      invalid_arg ("Schedule.Stage: source " ^ Tn.debug_name source ^ " is written in the routine")
  in
  let rec code stack llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
    | Tile_mma _ ->
        (* Cooperative-tile operands are not remappable reads in v1. *)
        invalid_arg "Schedule.Stage: apply Stage before Tensorize"
    | Zero_out tn -> reject_write tn
    | Seq (a, b) ->
        code stack a;
        code stack b
    | For_loop { index; from_; to_; body; axis } ->
        code ({ index; from_; to_; body = Noop; axis } :: stack) body
    | Scan_loop _ ->
        (* gh-ocannl-696: a scan inside the staged region is out of scope for v1 -- refused loudly
           rather than treated as a loop whose iterations the tile could serve. *)
        invalid_arg "Schedule.Stage: a Scan_loop inside the staged region is unsupported"
    | Set { tn; llsc; _ } ->
        reject_write tn;
        scalar stack llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        reject_write tn;
        scalar stack v;
        scalar stack llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        reject_write tn;
        scalar stack a
    | Set_local (_, llsc) -> scalar stack llsc
    | If { cond = c, _; body } ->
        scalar stack c;
        code stack body
  and scalar stack (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> code stack body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get (tn, idcs) -> if Tn.equal tn source then acc := (idcs, List.rev stack) :: !acc
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        if Tn.equal tn source then
          invalid_arg "Schedule.Stage: dynamically indexed source reads are unsupported";
        scalar stack v
    | Get_merge_buffer (_, _) -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar stack a;
        scalar stack b;
        scalar stack c
    | Binop (_, (a, _), (b, _)) ->
        scalar stack a;
        scalar stack b
    | Unop (_, (a, _)) -> scalar stack a
  in
  code [] llc;
  !acc

(* Replaces reads [Get (source, idcs)] — and, when [writes] is set, writes [Set { tn = source; idcs;
   _ }] — with [idcs] equal to [from_idcs] by accesses of [tile] at [tile_idcs]. *)
let remap_reads ?(writes = false) ~source ~from_idcs ~tile ~tile_idcs (llc : Low_level.t) :
    Low_level.t =
  let open Low_level in
  let rec code llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        llc
    (* Unreachable: [collect_source_accesses] / Privatize's scan reject [Tile_mma] first. *)
    | Tile_mma _ -> llc
    | Seq (a, b) -> Seq (code a, code b)
    | For_loop fc -> For_loop { fc with body = code fc.body }
    | Scan_loop sc ->
        Scan_loop
          {
            sc with
            carried = List.map sc.carried ~f:(fun c -> { c with init = scalar c.init });
            body = code sc.body;
          }
    | Set { tn; idcs; llsc; debug }
      when writes && Tn.equal tn source && Array.equal Indexing.equal_axis_index idcs from_idcs ->
        Set { tn = tile; idcs = tile_idcs; llsc = scalar llsc; debug }
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = scalar llsc; debug }
    (* A dynamically-indexed write never matches the exact [from_idcs]; leave it in place. *)
    | Set_dynamic ({ dyn_value = v, p; llsc; _ } as sd) ->
        Set_dynamic { sd with dyn_value = (scalar v, p); llsc = scalar llsc }
    | Set_from_vec ({ arg = a, p; _ } as sv) -> Set_from_vec { sv with arg = (scalar a, p) }
    | Set_local (id, llsc) -> Set_local (id, scalar llsc)
    | If { cond = c, p; body } -> If { cond = (scalar c, p); body = code body }
  and scalar (llsc : scalar_t) : scalar_t =
    match llsc with
    | Get (tn, idcs) when Tn.equal tn source && Array.equal Indexing.equal_axis_index idcs from_idcs
      ->
        Get (tile, tile_idcs)
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
        llsc
    | Local_scope ({ body; _ } as ls) -> Local_scope { ls with body = code body }
    | Get_dynamic ({ dyn_value = v, p; _ } as gd) ->
        Get_dynamic { gd with dyn_value = (scalar v, p) }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (scalar a, pa), (scalar b, pb), (scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
  in
  code llc

(* A constant operand eligible for hoisted (out-of-routine) packing: declared value-constant with
   registered host-init data to pack from (gh-ocannl-470). Shared by the autotune sketch (which
   proposes hoisted candidates only for such operands) and the canonical digest
   ([Schedule_cache.canonicalize] renders it per tnode, so same-shape programs differing in operand
   constancy do not share cached schedules — a hoisted winner must not replay against a
   non-hoistable site, and a non-hoisted winner must not mask the hoisted candidates of a constant
   site; Codex P2 on PR #123). *)
let hoistable_constant tn = Tn.known_host_constant tn && Host_inits.mem tn

(* Host-side packing for hoisted Stage (gh-ocannl-470): materialize the packed layout of a constant
   operand from its host-init data. Forced at link/upload time — through the packed node's own
   [Host_inits] lazy — and uploaded into the per-device constant pool like any other
   host-initialized constant. [src_prog]/[packed_prog] are the per-axis affine index programs:
   [(coefficient, position in sym_extents) array * offset], evaluated over the odometer enumeration
   of the tile and outer loop symbols. *)
let pack_constant_tile ~debug ~(src_nd : Ndarray.t) ~(src_dims : int array) ~(src_prec : Ops.prec)
    ~(prec : Ops.prec) ~(packed_dims : int array) ~(sym_extents : int array)
    ~(src_prog : ((int * int) array * int) array) ~(packed_prog : ((int * int) array * int) array) :
    Ndarray.t =
  if not (Ops.equal_prec (Ndarray.get_prec src_nd) src_prec) then
    invalid_arg ("Schedule.Stage: hoisted staging: host-init precision mismatch for " ^ debug);
  if not (Array.equal Int.equal (Ndarray.dims src_nd) src_dims) then
    invalid_arg ("Schedule.Stage: hoisted staging: host-init dims mismatch for " ^ debug);
  (* Zero-filled at creation: pad slots of edge tiles (never written below) must read as zeros. *)
  let dst = Ndarray.create_array ~debug prec ~dims:packed_dims ~padding:(Some ([||], 0.0)) in
  let n_syms = Array.length sym_extents in
  let vals = Array.create ~len:n_syms 0 in
  let eval (terms, off) = Array.fold terms ~init:off ~f:(fun acc (c, p) -> acc + (c * vals.(p))) in
  let src_idx = Array.create ~len:(Array.length src_dims) 0 in
  let dst_idx = Array.create ~len:(Array.length packed_dims) 0 in
  let pack ~set_elem =
    let rec go d =
      if d = n_syms then (
        Array.iteri src_prog ~f:(fun a pr -> src_idx.(a) <- eval pr);
        (* The edge guard: out-of-range coordinates arise only from edge tiles. *)
        if Array.for_alli src_idx ~f:(fun a i -> i < src_dims.(a)) then (
          Array.iteri packed_prog ~f:(fun a pr -> dst_idx.(a) <- eval pr);
          set_elem ()))
      else
        for v = 0 to sym_extents.(d) - 1 do
          vals.(d) <- v;
          go (d + 1)
        done
    in
    go 0
  in
  if Ops.equal_prec src_prec prec then
    Ndarray.apply2
      {
        f2 =
          (fun src dstb ->
            pack ~set_elem:(fun () ->
                Stdlib.Bigarray.Genarray.set dstb dst_idx (Stdlib.Bigarray.Genarray.get src src_idx)));
      }
      src_nd dst
  else
    (* The widening pack (gh-ocannl-575): [tile_prec] only admits exact widenings (narrow float to
       f32/f64), so the float round-trip is the identity on every element and matches the kernels'
       own conversions bitwise. *)
    pack ~set_elem:(fun () ->
        Ndarray.set_from_float dst dst_idx (Ndarray.get_as_float src_nd src_idx));
  dst

(* "[Tile_mma] is a barrier": the intrinsic's emission contract brackets the block in workgroup
   barriers on every barrier-capable backend ({!optop.Tensorize} docs) — every rendering form (the
   per-statement intrinsic, the fragment-scope update, the register-tiled and lane-0 scalar
   fallbacks) ENDS with one. Only the leading bracket is form-dependent: the fragment-update form
   emits it once, on the scope that wraps the anchor loop, rather than per iteration — so a barrier
   may be elided against what FOLLOWS it only when that is an explicit [Workgroup_barrier], never
   against a [Tile_mma]'s leading bracket. Conservative on every other statement: [If] is
   predicated, and a dead loop executes nothing. *)
let rec ends_with_barrier (llc : Low_level.t) : bool =
  match llc with
  | Low_level.Workgroup_barrier | Low_level.Tile_mma _ -> true
  | Low_level.Seq (_, b) -> ends_with_barrier b
  (* Hardware axes bind rather than iterate; a serial loop ends with its (non-empty) last
     iteration's end. A dead loop executes nothing, so it ends with no barrier. *)
  | Low_level.For_loop { body; from_; to_; _ } -> to_ >= from_ && ends_with_barrier body
  | _ -> false

(* The tensor nodes a statement writes, recursing into loop / guard / local-scope bodies. *)
let written_nodes (llc : Low_level.t) : Set.M(Tn).t =
  let open Low_level in
  let acc = ref (Set.empty (module Tn)) in
  let rec code llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
    | Zero_out tn -> acc := Set.add !acc tn
    | Tile_mma { d = tn, _; _ } -> acc := Set.add !acc tn
    | Seq (a, b) ->
        code a;
        code b
    | For_loop { body; _ } | If { body; _ } -> code body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scalar c.init);
        code body
    | Set { tn; llsc; _ } ->
        acc := Set.add !acc tn;
        scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        acc := Set.add !acc tn;
        scalar v;
        scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        acc := Set.add !acc tn;
        scalar a
    | Set_local (_, llsc) -> scalar llsc
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> code body
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get_dynamic { dyn_value = v, _; _ } | Unop (_, (v, _)) -> scalar v
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
  in
  code llc;
  !acc

let apply_stage ~source ~tile_loops ~shared ~cooperative ~hoisted ~swizzle ~pad_stride
    ~pipeline_depth ~tile_prec (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  if List.is_empty tile_loops then invalid_arg "Schedule.Stage: empty tile_loops";
  let source_prec = Lazy.force source.Tn.storage_prec in
  (* [tile_prec] admits exact widenings only (gh-ocannl-575): a widening is value-preserving and its
     round-trip is the identity, so the transform stays numerics-preserving no matter which
     precision the consumers compute at. *)
  let stage_prec =
    match tile_prec with
    | None -> source_prec
    | Some p when Ops.equal_prec p source_prec -> p
    | (Some (Ops.Single_prec _ as p) | Some (Ops.Double_prec _ as p))
      when Ops.is_narrow_float source_prec ->
        p
    | Some p ->
        invalid_arg
          (Printf.sprintf
             "Schedule.Stage: tile_prec %s is not an exact widening of %s storage for %s"
             (Ops.prec_string p) (Ops.prec_string source_prec) (Tn.debug_name source))
  in
  Option.iter cooperative ~f:(fun w ->
      if not shared then invalid_arg "Schedule.Stage: cooperative staging requires shared = true";
      if w <= 0 then invalid_arg "Schedule.Stage: cooperative simd width must be positive");
  if pipeline_depth < 1 then
    invalid_arg
      (Printf.sprintf "Schedule.Stage: pipeline_depth must be >= 1, got %d" pipeline_depth);
  if pipeline_depth > 1 && Option.is_none cooperative then
    invalid_arg
      "Schedule.Stage: pipeline_depth > 1 requires cooperative staging (the rotation pipelines the \
       cooperative copy against the compute; packing and reused-workgroup stagings have no \
       prefetch to overlap)";
  if pipeline_depth > 2 then
    invalid_arg
      (Printf.sprintf
         "Schedule.Stage: pipeline_depth %d is not implemented in gh-ocannl-487: both the portable \
          form and the cp.async arm have single-step lookahead (one prefetch per iteration, \
          completed behind one barrier resp. one wait-all), keeping at most one block in flight — \
          copies beyond 2 would only consume shared memory; deeper pipelines need \
          commit-group/wait-group bookkeeping in the async arms"
         pipeline_depth);
  if hoisted && shared then
    invalid_arg "Schedule.Stage: hoisted staging requires shared = false (it emits no load nest)";
  if Option.is_some swizzle && not shared then
    invalid_arg
      "Schedule.Stage: swizzle is a shared-memory bank-conflict layout, it requires shared = true";
  let accesses = collect_source_accesses ~source opt.llc in
  let idcs0, stack0 =
    match accesses with
    | [] -> invalid_arg ("Schedule.Stage: no reads of " ^ Tn.debug_name source ^ " in the routine")
    | hd :: _ -> hd
  in
  List.iter accesses ~f:(fun (idcs, _) ->
      if not (Array.equal Indexing.equal_axis_index idcs idcs0) then
        invalid_arg
          ("Schedule.Stage: v1 requires all reads of " ^ Tn.debug_name source
         ^ " to use identical index vectors"));
  let stack0 =
    Array.of_list stack0
    (* outermost-first *)
  in
  let is_tile s = List.mem tile_loops s ~equal:Indexing.equal_symbol in
  let depth_of s = Array.findi stack0 ~f:(fun _ fl -> Indexing.equal_symbol fl.index s) in
  let floop_of_exn s =
    match depth_of s with
    | Some (_, fl) -> fl
    | None ->
        invalid_arg
          ("Schedule.Stage: tile loop " ^ Indexing.symbol_ident s
         ^ " does not enclose the source access")
  in
  (* Per source axis: tile part (terms over tile loops), outer part, offset. *)
  let decomp =
    Array.map idcs0 ~f:(fun idx ->
        match terms_of_index idx with
        | None -> invalid_arg "Schedule.Stage: Concat indices are unsupported"
        | Some (terms, offset) ->
            let tile_part, outer_part = List.partition_tf terms ~f:(fun (_, s) -> is_tile s) in
            (tile_part, outer_part, offset))
  in
  Array.iter decomp ~f:(fun (tp, _, _) ->
      List.iter tp ~f:(fun (c, s) ->
          if c <= 0 then invalid_arg "Schedule.Stage: nonpositive coefficient on a tile loop index";
          let fl = floop_of_exn s in
          if fl.from_ <> 0 then invalid_arg "Schedule.Stage: tile loops must start at 0"));
  List.iter tile_loops ~f:(fun s ->
      if
        not
          (Array.exists decomp ~f:(fun (tp, _, _) ->
               List.exists tp ~f:(fun (_, s') -> Indexing.equal_symbol s s')))
      then
        invalid_arg
          ("Schedule.Stage: tile loop " ^ Indexing.symbol_ident s
         ^ " does not occur in the source access"));
  let extent s =
    let fl = floop_of_exn s in
    fl.to_ - fl.from_ + 1
  in
  (* Compacting Stage (gh-ocannl-502): a single-term tile part with coefficient [c > 1] — a strided
     window, e.g. the implicit-GEMM row of a stride-2 conv, whose source index carries [c*row] — is
     packed densely: the tile axis is sized by the loop extent rather than the strided range, and
     the tile store/read indices use the symbol with coefficient 1, so downstream [Tensorize]'s
     unit-coefficient index discipline accepts the tile. Only the load's source index (and its edge
     guard) keeps the stride. Multi-term tile parts keep the range-sized (dilated) layout: their
     dense remap is not injective in general. *)
  let compacted a = match decomp.(a) with [ (c, s) ], _, _ when c > 1 -> Some s | _ -> None in
  (* Tile axes: source axes with a nonempty tile part; dim = the tile part's range (the loop extent
     when compacted). Ordered by the position in [tile_loops] of each axis's first tile-part symbol
     (stable within source order), so the caller's [tile_loops] order picks the packed layout (see
     the section comment). *)
  let tile_axes =
    Array.filter_mapi decomp ~f:(fun a (tp, _, _) ->
        if List.is_empty tp then None
        else
          match compacted a with
          | Some s -> Some (a, extent s)
          | None -> Some (a, List.fold tp ~init:1 ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))))
  in
  let tile_loop_pos s =
    match List.findi tile_loops ~f:(fun _ s' -> Indexing.equal_symbol s s') with
    | Some (p, _) -> p
    | None -> List.length tile_loops
  in
  let tile_axes =
    Array.sorted_copy tile_axes ~compare:(fun (a1, _) (a2, _) ->
        let key a =
          let tp, _, _ = decomp.(a) in
          List.map tp ~f:(fun (_, s) -> tile_loop_pos s)
          |> List.min_elt ~compare:Int.compare
          |> Option.value ~default:(List.length tile_loops)
        in
        match Int.compare (key a1) (key a2) with 0 -> Int.compare a1 a2 | c -> c)
  in
  (* Classify tile loops: Workgroup-typed loops are reused as cooperating thread indices in the load
     nest; Serial ones are iterated under fresh symbols. *)
  let reused, iterated =
    List.partition_tf tile_loops ~f:(fun s ->
        match (floop_of_exn s).axis with
        | Workgroup | Workgroup_reduce -> true
        | Serial | Vectorized -> false
        | Grid -> invalid_arg "Schedule.Stage: a Grid-typed loop cannot be a tile loop"
        | Unrolled ->
            invalid_arg "Schedule.Stage: apply Stage before (materializing) Unroll of a tile loop")
  in
  if (not shared) && not (List.is_empty reused) then
    invalid_arg "Schedule.Stage: non-shared (packing) staging requires Serial tile loops";
  if Option.is_some cooperative && not (List.is_empty reused) then
    invalid_arg
      "Schedule.Stage: cooperative staging requires Serial tile loops (the fresh lane loop is the \
       cooperating axis; reusing Workgroup tile loops is the non-cooperative mode)";
  if hoisted then (
    (* Hoisted (out-of-routine) packing for a compile-time-constant source (gh-ocannl-470): the
       packed layout covers the whole source — one packed axis per outer coordinate followed by the
       tile axes — so no load nest is emitted; the reads are remapped to a fresh host-initialized
       constant whose buffer is packed on the host at link time and uploaded once per device into
       the constant pool. *)
    Array.iteri decomp ~f:(fun a _ ->
        if Option.is_some (compacted a) then
          invalid_arg
            "Schedule.Stage: hoisted staging does not support compacting a strided tile part (v1) \
             — its blocked outer decomposition assumes the tile part addresses the source densely");
    if not (Host_inits.mem source) then
      invalid_arg
        ("Schedule.Stage: hoisted staging requires registered host-init data for "
       ^ Tn.debug_name source);
    if
      not
        (Tn.known_host_constant source
        || Tn.Placements.known_constant opt.Low_level.optimize_ctx.placements source)
    then
      invalid_arg
        ("Schedule.Stage: hoisted staging requires a known-constant source, got "
       ^ Tn.debug_name source);
    (match Lazy.force source.Tn.padding with
    | Some _ ->
        invalid_arg
          ("Schedule.Stage: hoisted staging does not support a padded source: "
         ^ Tn.debug_name source)
    | None -> ());
    (* Packing runs at link time, outside the routine: every outer-part symbol must be an enclosing
       loop with a known extent (static/dynamic indices have no binding there), and the affine maps
       below need nonnegative source coordinates. *)
    Array.iter decomp ~f:(fun (_, op_, off) ->
        if off < 0 then
          invalid_arg "Schedule.Stage: hoisted staging requires nonnegative index offsets";
        List.iter op_ ~f:(fun (c, s) ->
            if c <= 0 then
              invalid_arg
                "Schedule.Stage: hoisted staging requires positive outer-part coefficients";
            if Option.is_none (depth_of s) then
              invalid_arg
                ("Schedule.Stage: hoisted staging requires outer-part symbol "
               ^ Indexing.symbol_ident s ^ " to be bound by an enclosing loop")));
    let tile_dim a = Array.find_map tile_axes ~f:(fun (a', d) -> Option.some_if (a = a') d) in
    (* Packed outer axes, in source-axis order: on a tiled axis the outer part (plus offset) divided
       by the tile dim is the tile-count coordinate (divisibility required — the standard blocked
       decomposition [k := bk*KT + k_i] satisfies it); an axis without a tile part keeps its outer
       index expression as-is. Raw [(terms, offset, extent)] — the terms drive both the consumer's
       remapped reads and the host-side packing program. *)
    let outer_axes =
      Array.filter_mapi decomp ~f:(fun a (_tp, op_, off) ->
          if List.is_empty op_ && off = 0 then None
          else
            match tile_dim a with
            | None ->
                let ext =
                  List.fold op_ ~init:(off + 1) ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))
                in
                Some (op_, off, ext)
            | Some t_a ->
                List.iter op_ ~f:(fun (c, _) ->
                    if c % t_a <> 0 then
                      invalid_arg
                        "Schedule.Stage: hoisted staging requires outer-part coefficients \
                         divisible by the tile dim of their axis");
                if off % t_a <> 0 then
                  invalid_arg
                    "Schedule.Stage: hoisted staging requires the index offset divisible by the \
                     tile dim of its axis";
                let q = List.map op_ ~f:(fun (c, s) -> (c / t_a, s)) in
                let qoff = off / t_a in
                let ext =
                  List.fold q ~init:(qoff + 1) ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))
                in
                Some (q, qoff, ext))
    in
    let packed_dims =
      Array.append (Array.map outer_axes ~f:(fun (_, _, ext) -> ext)) (Array.map tile_axes ~f:snd)
    in
    let packed_read_idcs =
      Array.append
        (Array.map outer_axes ~f:(fun (terms, off, _) -> normalize_affine ~terms ~offset:off))
        (Array.map tile_axes ~f:(fun (a, _) ->
             let tp, _, _ = decomp.(a) in
             normalize_affine ~terms:tp ~offset:0))
    in
    let prec = stage_prec in
    let tile =
      Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
        ~label:("packed" :: source.Tn.label)
        ~unpadded_dims:(lazy packed_dims)
        ~padding:(lazy None)
        ()
    in
    (* A host-initialized constant: [Effectively_constant] intent, materialized in this lineage —
       [allocate_delta] routes it (read-only, host-init-backed) into the per-device constant pool,
       and [Host_inits.mem] keeps it out of the routine's required inputs. *)
    Tn.update_memory_mode tile Effectively_constant 176;
    Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.On_device 176;
    let traced = get_node opt.traced_store tile in
    traced.read_only <- true;
    (* The packing program: odometer enumeration of every tile and outer symbol, evaluated through
       positional [(coefficient, symbol slot)] compiles of the affine maps. Captured eagerly — the
       lazy below must not reference the pre-remap loop structure. *)
    let syms =
      Array.to_list decomp
      |> List.concat_map ~f:(fun (tp, op_, _) -> List.map (tp @ op_) ~f:snd)
      |> List.dedup_and_sort ~compare:Indexing.Symbol.compare
      |> List.map ~f:(fun s -> (s, extent s))
      |> Array.of_list
    in
    let pos_of s =
      fst (Option.value_exn (Array.findi syms ~f:(fun _ (s', _) -> Indexing.equal_symbol s s')))
    in
    let compile_terms terms = Array.of_list_map terms ~f:(fun (c, s) -> (c, pos_of s)) in
    let src_prog = Array.map decomp ~f:(fun (tp, op_, off) -> (compile_terms (tp @ op_), off)) in
    let packed_prog =
      Array.append
        (Array.map outer_axes ~f:(fun (terms, off, _) -> (compile_terms terms, off)))
        (Array.map tile_axes ~f:(fun (a, _) ->
             let tp, _, _ = decomp.(a) in
             (compile_terms tp, 0)))
    in
    let sym_extents = Array.map syms ~f:snd in
    let src_dims = Lazy.force source.Tn.dims in
    let debug = Tn.debug_name tile in
    Host_inits.register tile
      (lazy
        (let src_nd =
           match Host_inits.find source with
           | Some l -> Lazy.force l
           | None ->
               invalid_arg
                 ("Schedule.Stage: host-init data for " ^ Tn.debug_name source
                ^ " disappeared before link")
         in
         pack_constant_tile ~debug ~src_nd ~src_dims ~src_prec:source_prec ~prec ~packed_dims
           ~sym_extents ~src_prog ~packed_prog));
    let llc = remap_reads ~source ~from_idcs:idcs0 ~tile ~tile_idcs:packed_read_idcs opt.llc in
    (* [pack_constant_tile] zero-fills pad slots of edge tiles, so the packed buffer satisfies the
       [zero_fringe] contract over its whole index space. *)
    { opt with llc; zero_fringe = Set.add opt.zero_fringe tile })
  else
    (* The insertion point L*: the deepest loop that must stay outside the tile — carrying an
       outer-part symbol or a reused workgroup tile axis. *)
    let outer_sym_depths =
      Array.to_list decomp
      |> List.concat_map ~f:(fun (_, op_, _) -> op_)
      |> List.filter_map ~f:(fun (_, s) -> Option.map (depth_of s) ~f:fst)
    in
    let reused_depths = List.map reused ~f:(fun s -> fst (Option.value_exn (depth_of s))) in
    let lstar_depth = List.max_elt (outer_sym_depths @ reused_depths) ~compare:Int.compare in
    (* Serial tile loops may sit above or below L*: the load nest iterates them under fresh symbols,
     so only outer-part symbols and reused workgroup axes pin the staging point. *)
    (* Mint the tile. *)
    let prec = stage_prec in
    let tile_dims = Array.map tile_axes ~f:snd in
    (* [pad_stride] (gh-ocannl-481 item 4): round the tile's MINOR dim up to a multiple, so the
       tile's leading-dimension stride — which is that dim, and which every consumer reads off the
       node — becomes the padded one while the iterated index space stays the unpadded extents. Two
       payoffs, both about the stride rather than the data: shared-memory bank conflicts on a
       strided read of the tile, and layout rules stated on the stride (a fragment load's
       ld-multiple constraint; [Swizzle_b128]'s 16-byte-unit count, hence "pad first, then check" —
       the validation below runs on the padded dims).

       The padded slots hold nothing in the row-major case: no loop reaches them, so they are
       neither written nor read (the {!Low_level.zero_fringe} contract is about the fringe of the
       staged SOURCE region within the iterated space, which is unaffected). Under a swizzle they do
       carry data — the XOR is a bijection of the whole padded row — and reads use the same map, so
       that case is coherent too. *)
    let tile_dims =
      match pad_stride with
      | None -> tile_dims
      | Some p ->
          let n = Array.length tile_dims in
          if p <= 1 then
            invalid_arg (Printf.sprintf "Schedule.Stage: pad_stride must be > 1, got %d" p)
          else if n < 2 then
            invalid_arg
              "Schedule.Stage: pad_stride pads the minor dim against the row stride, so it \
               requires a tile with at least two axes"
          else begin
            let padded = Array.copy tile_dims in
            padded.(n - 1) <- (tile_dims.(n - 1) + p - 1) / p * p;
            padded
          end
    in
    Option.iter swizzle ~f:(fun kind ->
        let n = Array.length tile_dims in
        if n < 2 then
          invalid_arg
            "Schedule.Stage: swizzle requires a tile with at least two axes (the minor axis is \
             XORed against the row prefix)";
        let c = tile_dims.(n - 1) in
        (* Each flavor's XOR must stay inside the row, so the count of units it permutes must be a
           power of two > 1 — elements for [Swizzle_elem], 16-byte groups for [Swizzle_b128]
           (gh-ocannl-481 item 3, D1). The b128 count is NOT implied by the element one: it is
           coarser, so it admits minor extents the element flavor rejects (a 24-element f16 row is 3
           units, rejected; a 12-element f32 row is 3 units, rejected) and rejects extents the
           element flavor accepts (a 2-element f32 row does not fill one unit). Item 4's
           [pad_stride] is the knob for shapes whose natural minor extent misses it: pad first, then
           check — this validation runs on the padded dims. *)
        match kind with
        | Low_level.Swizzle_elem ->
            if c < 2 || c land (c - 1) <> 0 then
              invalid_arg
                (Printf.sprintf
                   "Schedule.Stage: swizzle requires a power-of-two minor tile dim > 1, got %d" c)
        | Low_level.Swizzle_b128 ->
            let row_bytes = c * Ops.prec_in_bytes prec in
            if row_bytes % 16 <> 0 then
              invalid_arg
                (Printf.sprintf
                   "Schedule.Stage: Swizzle_b128 requires the minor tile dim to span a multiple of \
                    16 bytes, got %d elements of %s (%d bytes)"
                   c (Ops.prec_string prec) row_bytes)
            else
              let units = row_bytes / 16 in
              if units < 2 || units land (units - 1) <> 0 then
                invalid_arg
                  (Printf.sprintf
                     "Schedule.Stage: Swizzle_b128 requires a power-of-two count > 1 of 16-byte \
                      units per row, got %d"
                     units));
    let tile =
      Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
        ~label:("tile" :: source.Tn.label)
        ~unpadded_dims:(lazy tile_dims)
        ~padding:(lazy None)
        ()
    in
    Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.Local 175;
    ignore (get_node opt.traced_store tile : traced_array);
    (* The load nest. *)
    let fresh = List.map iterated ~f:(fun s -> (s, Indexing.get_symbol ())) in
    let load_sym s =
      match List.Assoc.find fresh ~equal:Indexing.equal_symbol s with Some s' -> s' | None -> s
    in
    let subst_terms terms = List.map terms ~f:(fun (c, s) -> (c, load_sym s)) in
    let load_src_idcs =
      Array.map decomp ~f:(fun (tp, op_, off) ->
          normalize_affine ~terms:(subst_terms tp @ op_) ~offset:off)
    in
    let tile_store_idcs =
      Array.map tile_axes ~f:(fun (a, _) ->
          let tp, _, _ = decomp.(a) in
          let terms =
            match compacted a with Some s -> [ (1, load_sym s) ] | None -> subst_terms tp
          in
          normalize_affine ~terms ~offset:0)
    in
    let tile_read_idcs =
      Array.map tile_axes ~f:(fun (a, _) ->
          let tp, _, _ = decomp.(a) in
          let terms = match compacted a with Some s -> [ (1, s) ] | None -> tp in
          normalize_affine ~terms ~offset:0)
    in
    let iprec = Ops.index_prec () in
    let src_dims = Lazy.force source.Tn.dims in
    (* Edge guards per tile axis (construct-then-fold: [apply]'s trailing simplify erases the ones
       the loop extents prove, i.e. whenever the tile sizes divide the source extents). The guards
       are [Where]-form rather than statement [If]s: an out-of-range slot stores 0 — the add-reduce
       accumulation identity — instead of staying uninitialized, so edge tiles of a non-dividing or
       padded staging (gh-ocannl-485) are safe to read over their whole index space. The tile is
       recorded in {!Low_level.optimized.zero_fringe} accordingly. *)
    let guarded_get =
      Array.fold tile_axes
        ~init:(Get (source, load_src_idcs))
        ~f:(fun rhs (a, _) ->
          let cond =
            Binop
              ( Ops.Cmplt,
                (Embed_index load_src_idcs.(a), iprec),
                (Constant (Float.of_int src_dims.(a)), iprec) )
          in
          Ternop (Ops.Where, (cond, iprec), (rhs, prec), (Constant 0., prec)))
    in
    let load_stmt = Set { tn = tile; idcs = tile_store_idcs; llsc = guarded_get; debug = "" } in
    (* Lane-aware cooperative staging (docs/proposals/tensorize-mma.md, "Lane-aware Stage"): a fresh
       extent-[w] [Workgroup] lane loop will wrap the load nest, so the loads are partitioned (or
       restricted) along the same hardware slot the tensorized micro-kernel's lane loop binds —
       positional slot assignment aligns the two loops at slot 0, and [validate_parallel]'s
       barrier-strength extent-uniformity rejects a width mismatch, so the only coordination with
       [Tensorize] is passing the same simd width. The partition folds the lane linearly into the
       innermost fresh copy loop (consecutive lanes touch consecutive minor-axis elements):
       division/modulo are not in the affine index algebra, so an extent that neither divides nor is
       bounded by the width falls back to the representative-lane ([w == 0]) discipline. *)
    let lane = Option.map cooperative ~f:(fun w -> (Indexing.get_symbol (), w)) in
    let lane_plan =
      match (lane, List.last fresh) with
      | None, _ | _, None -> `Serial_all
      | Some (w_sym, w), Some (s_orig, s_inner) ->
          let e = extent s_orig in
          if e <= w then `Drop_inner (s_inner, e, w_sym)
          else if e % w = 0 then `Divide_inner (s_inner, e / w, w_sym, w)
          else `Restrict_lane0
    in
    let load_stmt =
      match lane_plan with
      | `Serial_all | `Restrict_lane0 -> load_stmt
      | `Drop_inner (s_inner, e, w_sym) ->
          (* The lane replaces the innermost copy loop; the edge guard folds exactly when the extent
             equals the width (construct-then-fold, as everywhere in this transform). *)
          let stmt =
            map_code
              ~fidx:(subst_axis_index ~sym:s_inner ~by:{ terms = [ (1, w_sym) ]; offset = 0 })
              load_stmt
          in
          let cond =
            Binop
              ( Ops.Cmplt,
                (Embed_index (Indexing.Iterator w_sym), iprec),
                (Constant (Float.of_int e), iprec) )
          in
          If { cond = (cond, iprec); body = stmt }
      | `Divide_inner (s_inner, _, w_sym, w) ->
          (* [s_inner := w * s_inner + w_sym]: per copy step, the [w] lanes cover a contiguous chunk
             of the minor axis. *)
          map_code
            ~fidx:
              (subst_axis_index ~sym:s_inner
                 ~by:{ terms = [ (w, s_inner); (1, w_sym) ]; offset = 0 })
            load_stmt
    in
    let load_nest =
      List.fold (List.rev fresh) ~init:load_stmt ~f:(fun body (s, s') ->
          match lane_plan with
          | `Drop_inner (si, _, _) when Indexing.equal_symbol s' si -> body
          | `Divide_inner (si, ext', _, _) when Indexing.equal_symbol s' si ->
              For_loop { index = s'; from_ = 0; to_ = ext' - 1; body; axis = Serial }
          | _ -> For_loop { index = s'; from_ = 0; to_ = extent s - 1; body; axis = Serial })
    in
    (* The splice target. With an anchor L* (an outer-part symbol or reused workgroup axis), the
       load nest goes at the start of L*'s body. A shared stage with no anchor (e.g. staging a
       broadcast vector indexed only by serial tile loops) must NOT go to the routine root: every
       hardware thread executes top-level statements, and no workgroup index symbol is bound there
       to restrict the loads — so wrap the outermost tile loop instead, where the workgroup axes
       enclosing the consumer also enclose (and can guard) the loads (PR #90 review). Packing
       (shared = false) writes per-thread scratch, so the root is fine. *)
    let outermost_tile_depth =
      List.map tile_loops ~f:(fun s -> fst (Option.value_exn (depth_of s)))
      |> List.min_elt ~compare:Int.compare |> Option.value_exn
    in
    let wrap_outermost_tile = shared && Option.is_none lstar_depth in
    (* Loops enclosing the staging point, outermost-first. *)
    let in_scope =
      match lstar_depth with
      | Some ld -> Array.to_list (Array.sub stack0 ~pos:0 ~len:(ld + 1))
      | None when wrap_outermost_tile ->
          Array.to_list (Array.sub stack0 ~pos:0 ~len:outermost_tile_depth)
      | None -> []
    in
    (* Every workgroup slot active in the kernel's launch must be either reused by this tile's
       cooperative load or bound by an enclosing loop at the staging point (restricted to [w == 0]
       below): hardware threads differing only in an uncovered slot would all execute the loads,
       writing the same shared addresses concurrently — a same-value race is still a race. *)
    (if shared then
       let axes = hardware_axes opt.llc in
       let wg_axes =
         List.filter axes ~f:(fun a -> match a.ha_kind with `Workgroup -> true | `Grid -> false)
       in
       let active_slots =
         List.filter_map wg_axes ~f:(fun a -> if a.ha_extent > 1 then Some a.ha_slot else None)
         |> List.dedup_and_sort ~compare:Int.compare
       in
       let covered =
         List.filter_map in_scope ~f:(fun fl ->
             match fl.axis with
             | Workgroup | Workgroup_reduce ->
                 List.find_map wg_axes ~f:(fun a ->
                     if Indexing.equal_symbol a.ha_index fl.index then Some a.ha_slot else None)
             | Serial | Grid | Unrolled | Vectorized -> None)
       in
       (* Cooperative staging covers slot 0 by construction: the fresh lane loop is the innermost
          Workgroup loop of the load nest, and extent agreement with the kernel's other slot-0 loops
          is enforced downstream by [validate_parallel]'s barrier-strength uniformity. *)
       let covered = if Option.is_some cooperative then 0 :: covered else covered in
       let missing =
         List.filter active_slots ~f:(fun s -> not (List.mem covered s ~equal:Int.equal))
       in
       if not (List.is_empty missing) then
         invalid_arg
           ("Schedule.Stage: workgroup slot(s) "
           ^ String.concat ~sep:", " (List.map missing ~f:Int.to_string)
           ^ " are active in this kernel but no loop binding them encloses the staging point for "
           ^ Tn.debug_name source
           ^ ": their threads would race on the shared tile; restructure the schedule so the \
              workgroup loops enclose the staging point (or reuse them as tile loops)"));
    (* Restrict redundant cooperative loading along in-scope workgroup axes that do not participate
       in this tile: one representative thread ([w == 0]) loads for all. *)
    let load_nest =
      if not shared then load_nest
      else
        List.fold in_scope ~init:load_nest ~f:(fun body fl ->
            match fl.axis with
            | (Workgroup | Workgroup_reduce)
              when not (List.mem reused fl.index ~equal:Indexing.equal_symbol) ->
                If
                  {
                    cond =
                      ( Binop
                          ( Ops.Cmpeq,
                            (Embed_index (Indexing.Iterator fl.index), iprec),
                            (Constant 0., iprec) ),
                        iprec );
                    body;
                  }
            | _ -> body)
    in
    (* The lane loop wraps the (possibly lane-restricted) load nest, including the [w == 0]
       restrictions along other in-scope workgroup axes; the barriers stay its siblings at the
       staging point — hardware [Workgroup] loops bind rather than iterate, so they remain uniformly
       reached. *)
    let load_nest =
      match lane with
      | None -> load_nest
      | Some (w_sym, w) ->
          let body =
            match lane_plan with
            | `Restrict_lane0 ->
                If
                  {
                    cond =
                      ( Binop
                          ( Ops.Cmpeq,
                            (Embed_index (Indexing.Iterator w_sym), iprec),
                            (Constant 0., iprec) ),
                        iprec );
                    body = load_nest;
                  }
            | `Serial_all | `Drop_inner _ | `Divide_inner _ -> load_nest
          in
          For_loop { index = w_sym; from_ = 0; to_ = w - 1; axis = Workgroup; body }
    in
    let remap llc = remap_reads ~source ~from_idcs:idcs0 ~tile ~tile_idcs:tile_read_idcs llc in
    (* Same-anchor load-phase grouping at depth 1 (gh-ocannl-567; the [build_pipelined] arm below is
       the same recognizer one depth up). A previous shared stage anchored at THIS loop left the
       body as [loads; barrier; compute...; barrier]. Wrapping it again would render the k-block as
       [loadA; barrier; loadB; barrier; compute; barrier; barrier]: the mid barrier serializes two
       independent copies (distinct tiles, no dependency between them), and the two trailing ones
       are adjacent with no work between. Group instead — [loadA; loadB; barrier; compute...;
       barrier] — one phase per k-block. Safety: the barrier we drop between the two load nests
       ordered this tile's writes against the previous phase's statements, so the recognizer
       requires that phase to read nothing that the remap turns into a read of THIS tile (the tile
       is freshly minted, so nothing else can reference it); and the trailing barrier we drop is
       redundant with the one the previous stage already appended. The tile's writes stay separated
       from its reads by the surviving mid barrier, and its reads from the next iteration's
       overwrites by the trailing one — the same two phases as before, in one bracket instead of
       two. *)
    let same_anchor_load_phase inner =
      let stmts = flat_lines [ inner ] in
      let rec split acc = function
        | Workgroup_barrier :: rest -> Some (List.rev acc, rest)
        | hd :: tl -> split (hd :: acc) tl
        | [] -> None
      in
      match split [] stmts with
      | Some (loads, rest) when (not (List.is_empty loads)) && not (List.is_empty rest) ->
          let is_staged_load stmt =
            let w = written_nodes stmt in
            (not (Set.is_empty w)) && Set.is_subset w ~of_:opt.workgroup_shared
          in
          if
            List.for_all loads ~f:is_staged_load
            (* No read of [source] in the load phase: otherwise the remap below would place a read
               of this stage's tile before the barrier that orders its writes. *)
            && List.for_all loads ~f:(fun s -> List.is_empty (collect_source_accesses ~source s))
            && ends_with_barrier (unflat_lines rest)
          then Some (loads, rest)
          else None
      | _ -> None
    in
    (* [group]: the splice wraps an anchor loop's body, so an already-present load phase there
       belongs to a stage at the SAME anchor. The unanchored splices (routine root, outermost tile
       loop) wrap a whole loop nest instead and never group. *)
    let build ?(group = false) inner =
      if not shared then unflat_lines [ load_nest; remap inner ]
      else
        match if group then same_anchor_load_phase inner else None with
        | Some (loads, rest) ->
            (* [loads] read no [source], so remapping them is the identity. *)
            unflat_lines ((load_nest :: loads) @ [ Workgroup_barrier; remap (unflat_lines rest) ])
        | None -> unflat_lines [ load_nest; Workgroup_barrier; remap inner; Workgroup_barrier ]
    in
    (* Software pipelining (gh-ocannl-487): with [pipeline_depth > 1], the anchor L* — required
       [Serial], the rotor whose counter selects the buffer copy at rendering — gets the staging
       composition restructured from per-iteration [load(k); barrier; compute(k); barrier] to a
       prologue [load(from_)] before the loop, per-iteration [barrier; If (k < to_) load(k+1);
       compute(k)], and one trailing barrier after the loop. The compute subtree is remapped exactly
       as in the unpipelined form and never otherwise touched, and the loads write the same values
       into rotated copies the reads resolve to, so the rendering is bitwise identical to depth 1 —
       only the prefetch timing changes. Same-phase safety is the renderer's rotation: the in-loop
       copy targets buffer [(k+1) mod d] while the compute reads [k mod d], and the single barrier
       separates each buffer's write from its reads (one iteration apart) and from its next
       overwrite ([d - 1] iterations apart). The load nest contains no [Local_scope], so duplicating
       it needs no scope refresh; the [If] guard interval-folds to [Noop] on 1-trip loops
       (construct-then-fold, as for the edge guards above). *)
    let pipeline_rotor =
      if pipeline_depth = 1 then None
      else
        match lstar_depth with
        | Some ld when match stack0.(ld).axis with Serial -> true | _ -> false ->
            (* The prologue fills copy 0 and the renderer's first-iteration read is copy [from_ mod
               depth], so a nonzero lower bound would read an uninitialized copy (Codex P1 on PR
               #303). Unreachable today — lowering emits 0-based loops and a Partition-segmented
               anchor fails the single-index-vector rule first — so this is a defensive rejection,
               not a supported-case gate. *)
            if stack0.(ld).from_ <> 0 then
              invalid_arg
                ("Schedule.Stage: pipeline_depth > 1 requires the staging anchor loop "
                ^ Indexing.symbol_ident stack0.(ld).index
                ^ " to start at 0 (the prologue fills buffer copy 0)");
            (* A dead anchor ([to_ < from_], supported by the IR) never executes its body, so the
               unconditional prologue would run a load the unpipelined form never runs — an
               unreachable body may even carry accesses that are only valid because they never
               execute (Codex P2 on PR #303). Unreachable from real tensors (a k-block loop has at
               least one block); defensive like the bounds check above. *)
            if stack0.(ld).to_ < stack0.(ld).from_ then
              invalid_arg
                ("Schedule.Stage: pipeline_depth > 1 requires the staging anchor loop "
                ^ Indexing.symbol_ident stack0.(ld).index
                ^ " to have at least one iteration (a dead anchor cannot execute the prologue copy)"
                );
            Some stack0.(ld).index
        | Some ld ->
            invalid_arg
              ("Schedule.Stage: pipeline_depth > 1 requires the staging anchor loop "
              ^ Indexing.symbol_ident stack0.(ld).index
              ^ " to be Serial (the buffer rotation needs a well-defined iteration order)")
        | None ->
            invalid_arg
              ("Schedule.Stage: pipeline_depth > 1 requires an anchored staging point (a serial \
                loop carrying an outer-part symbol) for " ^ Tn.debug_name source
             ^ "; an unanchored (broadcast) staging has no loop to rotate the buffers with")
    in
    let build_pipelined rotor fc =
      let subst by = map_code ~fidx:(subst_axis_index ~sym:rotor ~by) load_nest in
      let prologue = subst { terms = []; offset = fc.from_ } in
      let prefetch = subst { terms = [ (1, rotor) ]; offset = 1 } in
      let guard =
        Binop
          ( Ops.Cmplt,
            (Embed_index (Indexing.Iterator rotor), iprec),
            (Constant (Float.of_int fc.to_), iprec) )
      in
      let guarded_prefetch = If { cond = (guard, iprec); body = prefetch } in
      (* Same-anchor composition (Codex P2 on PR #303): a previous pipelined stage at this rotor
         already restructured the body to [barrier; prefetches...; compute] and appended the
         trailing barrier after the loop. Wrapping it again ([barrier; prefetch_B; barrier;
         prefetch_A; compute]) would place a barrier between the two prefetches — the copy issued
         before it must complete at that barrier, so its load no longer overlaps the compute.
         Instead, group this stage's prefetch into the SAME phase: one barrier per iteration,
         prefetches back to back, then the compute. Safety is unchanged — the prefetches write
         next-iteration copies of distinct tiles, and the single barrier still separates every
         copy's write from its reads (one iteration apart) and from its next overwrite. *)
      let same_anchor =
        Map.exists opt.pipelined ~f:(fun { pt_rotor; _ } -> Indexing.equal_symbol pt_rotor rotor)
      in
      match (same_anchor, flat_lines [ fc.body ]) with
      | true, Workgroup_barrier :: rest ->
          let body =
            unflat_lines [ Workgroup_barrier; guarded_prefetch; remap (unflat_lines rest) ]
          in
          (* The previous same-anchor stage's trailing barrier already follows the loop. *)
          unflat_lines [ prologue; for_loop { fc with body } ]
      | _ ->
          let body = unflat_lines [ Workgroup_barrier; guarded_prefetch; remap fc.body ] in
          unflat_lines [ prologue; for_loop { fc with body }; Workgroup_barrier ]
    in
    let llc =
      match (lstar_depth, pipeline_rotor) with
      | Some ld, Some rotor ->
          rewrite_loop ~what:"Schedule.Stage" ~sym:stack0.(ld).index opt.llc
            ~f:(build_pipelined rotor)
      | Some ld, None ->
          rewrite_loop ~what:"Schedule.Stage" ~sym:stack0.(ld).index opt.llc ~f:(fun fc ->
              for_loop { fc with body = build ~group:true fc.body })
      | None, _ when wrap_outermost_tile ->
          rewrite_loop ~what:"Schedule.Stage" ~sym:stack0.(outermost_tile_depth).index opt.llc
            ~f:(fun fc -> build (for_loop fc))
      | None, _ -> build opt.llc
    in
    {
      opt with
      llc;
      workgroup_shared =
        (if shared then Set.add opt.workgroup_shared tile else opt.workgroup_shared);
      swizzled =
        (match swizzle with
        | Some kind -> Map.set opt.swizzled ~key:tile ~data:kind
        | None -> opt.swizzled);
      pipelined =
        (match pipeline_rotor with
        | Some rotor ->
            Map.set opt.pipelined ~key:tile ~data:{ pt_depth = pipeline_depth; pt_rotor = rotor }
        | None -> opt.pipelined);
      zero_fringe = Set.add opt.zero_fringe tile;
    }

(** {2 [Privatize]: accumulator privatization}

    Virtualization already privatizes accumulators — that is what [Local_scope] is — but it is
    forbidden from touching materialized nodes (they are observable). [Privatize { target; over }]
    recovers the same form inside one kernel for a materialized accumulation: the read-modify-write
    of [target] across the [over] loop's whole subtree is contracted to a per-thread [Local]
    accumulator tile — initialized from [target] before the loop, accumulated in place, stored back
    after — keeping a single final write per element. Because the tile is routine-local scratch
    whose address cannot alias the kernel's device pointers, downstream C/CUDA/MSL compilers can
    register-allocate it without [restrict] (gh-ocannl-164).

    Tile shape: per [target] axis, the index terms over loops nested inside [over] (they must be
    [Serial] — a workgroup-indexed private tile would make the store-back nest write other threads'
    elements) form the tile part sizing that axis; terms over loops outside [over] are the
    per-thread element selection, kept in the init-load and store-back indices. No tile part at all
    yields a scalar accumulator (dims [|1|]). Init/store nests iterate fresh serial symbols with
    per-axis edge guards (construct-then-fold, as in [Stage]). Any [Zero_out] of [target] elsewhere
    in the routine is left in place: the init-load observes its effect, so semantics are preserved
    without a surjectivity analysis (dropping the redundant zeroing is a follow-up). *)

let apply_privatize ~target ~over (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  let iprec = Ops.index_prec () in
  let tgt_dims = Lazy.force target.Tn.dims in
  (* Every loop of the routine by index symbol, with its axis type. Guard classification below needs
     the axis type of symbols bound OUTSIDE [over] too (a lane restriction is the whole point of the
     rule), and the [rewrite_loop] callback only sees the subtree. *)
  let loop_axes =
    let tbl = ref (Map.empty (module Indexing.Symbol)) in
    let rec go llc =
      match llc with
      | For_loop { index; axis; body; _ } ->
          tbl := Map.set !tbl ~key:index ~data:axis;
          go body
      | Scan_loop { carried; body; _ } ->
          List.iter carried ~f:(fun c -> go_scalar c.init);
          go body
      | Seq (a, b) ->
          go a;
          go b
      | If { cond = c, _; body } ->
          go_scalar c;
          go body
      | Set { llsc; _ } | Set_local (_, llsc) -> go_scalar llsc
      | Set_dynamic { dyn_value = v, _; llsc; _ } ->
          go_scalar v;
          go_scalar llsc
      | Set_from_vec { arg = a, _; _ } -> go_scalar a
      | Tile_mma { fallback; _ } -> go fallback
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
        ->
          ()
    and go_scalar (llsc : scalar_t) =
      match llsc with
      | Local_scope { body; _ } -> go body
      | Get_dynamic { dyn_value = v, _; _ } -> go_scalar v
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          go_scalar a;
          go_scalar b;
          go_scalar c
      | Binop (_, (a, _), (b, _)) ->
          go_scalar a;
          go_scalar b
      | Unop (_, (a, _)) -> go_scalar a
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
    in
    go opt.llc;
    !tbl
  in
  rewrite_loop ~what:"Schedule.Privatize" ~sym:over opt.llc ~f:(fun fc ->
      if not (equal_axis_type fc.axis Serial) then
        invalid_arg "Schedule.Privatize: the accumulation loop must be Serial";
      (* Accesses of [target] within the loop's subtree, with their loop stacks and enclosing [If]
         guard chains relative to it. Guards matter (PR #91 review): the remap keeps a guard on the
         update itself, but a thread-identifying guard (e.g. [w == 0] lane restriction) means only
         some threads' private accumulators receive updates — the init-load and store-back must then
         run under the {e same} predicate, or stale lanes clobber the result. *)
      let accesses = ref [] in
      let has_write = ref false in
      let rec scan stack conds llc =
        match llc with
        | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
        | Tile_mma _ -> invalid_arg "Schedule.Privatize: apply Privatize before Tensorize"
        | Zero_out tn ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: Zero_out of the target inside the accumulation loop"
        | Seq (a, b) ->
            scan stack conds a;
            scan stack conds b
        | For_loop { index; from_; to_; body; axis } ->
            scan ({ index; from_; to_; body = Noop; axis } :: stack) conds body
        | Scan_loop _ ->
            invalid_arg "Schedule.Privatize: a Scan_loop inside the privatized loop is unsupported"
        | Set { tn; idcs; llsc; _ } ->
            if Tn.equal tn target then (
              has_write := true;
              accesses := (idcs, stack, conds) :: !accesses);
            scan_scalar stack conds llsc
        | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: dynamically indexed target writes are unsupported";
            scan_scalar stack conds v;
            scan_scalar stack conds llsc
        | Set_from_vec { tn; arg = a, _; _ } ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: vector writes to the target are unsupported";
            scan_scalar stack conds a
        | Set_local (_, llsc) -> scan_scalar stack conds llsc
        | If { cond = (c, _) as cond; body } ->
            scan_scalar stack conds c;
            scan stack (cond :: conds) body
      and scan_scalar stack conds (llsc : scalar_t) =
        match llsc with
        | Local_scope { body; _ } -> scan stack conds body
        | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
        | Get (tn, idcs) -> if Tn.equal tn target then accesses := (idcs, stack, conds) :: !accesses
        | Get_dynamic { tn; dyn_value = v, _; _ } ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: dynamically indexed target accesses are unsupported";
            scan_scalar stack conds v
        | Get_merge_buffer (_, _) -> ()
        | Ternop (_, (a, _), (b, _), (c, _)) ->
            scan_scalar stack conds a;
            scan_scalar stack conds b;
            scan_scalar stack conds c
        | Binop (_, (a, _), (b, _)) ->
            scan_scalar stack conds a;
            scan_scalar stack conds b
        | Unop (_, (a, _)) -> scan_scalar stack conds a
      in
      scan [] [] fc.body;
      let idcs0 =
        match !accesses with
        | [] ->
            invalid_arg
              ("Schedule.Privatize: no accesses of " ^ Tn.debug_name target
             ^ " under the accumulation loop")
        | (idcs, _, _) :: _ -> idcs
      in
      if not !has_write then
        invalid_arg
          ("Schedule.Privatize: " ^ Tn.debug_name target ^ " is not written (no accumulation)");
      List.iter !accesses ~f:(fun (idcs, _, _) ->
          if not (Array.equal Indexing.equal_axis_index idcs idcs0) then
            invalid_arg
              ("Schedule.Privatize: v1 requires all accesses of " ^ Tn.debug_name target
             ^ " under the loop to use identical index vectors"));
      (* Guard chains must agree across accesses; each condition is then classified for what it may
         do to the accumulator's lifecycle (PR #91 review; gh-ocannl-730). Three admissible kinds:

         - {e Invariant}: free of memory reads and of symbols bound by [over] or any loop inside its
         subtree. A thread-identifying guard (a [lane == 0] restriction) is of this kind, and it
         means only some threads update their private accumulator — so the init-load and the
         store-back must run under the {e same} predicate, or stale lanes clobber the result. - {e
         Iteration mask}: hardware-free (every symbol it mentions is bound by a [Serial] or
         [Unrolled] loop, or by no loop at all — a launch-uniform binding), mentioning at least one
         symbol bound inside. It selects which iterations of one thread's own accumulation fire; it
         cannot select threads. It stays on the update alone: a tile slot it never updates is
         init-loaded from [target] and stored back unchanged, and every slot belongs to the one
         thread that transfers it. A reduction-axis pad mask (gh-ocannl-485) is of this kind. - {e
         Output mask}: literally [target]'s own index on some axis, compared against a bound no
         larger than that axis's dimension — the shape a row/column pad mask takes once the block
         and register [Split]s have substituted their hardware symbols into it. The transfers
         already carry the per-axis edge guard [src_idcs.(a) < tgt_dims.(a)] over that same
         expression, so they transfer every slot the mask updates and no update is lost; a slot the
         mask skips round-trips through the accumulator unchanged. It too stays on the update alone.

         Anything else — a data-dependent guard, or one mixing a hardware symbol into a comparison
         that is not [target]'s own index — is rejected: it can restrict which threads accumulate
         while leaving the transfers to write back an accumulator that never received the update. *)
      let conds0 = match !accesses with (_, _, conds) :: _ -> conds | [] -> [] in
      List.iter !accesses ~f:(fun (_, _, conds) ->
          if not (List.equal equal_scalar_arg conds conds0) then
            invalid_arg
              ("Schedule.Privatize: accesses of " ^ Tn.debug_name target
             ^ " sit under differing If guards"));
      let bound_inside =
        let acc = ref [ over ] in
        let rec go llc =
          match llc with
          | For_loop { index; body; _ } ->
              acc := index :: !acc;
              go body
          | Scan_loop { index; carried; body; _ } ->
              acc := index :: !acc;
              List.iter carried ~f:(fun c -> go_scalar c.init);
              go body
          | Seq (a, b) ->
              go a;
              go b
          | If { body; _ } -> go body
          | Set { llsc; _ } | Set_local (_, llsc) -> go_scalar llsc
          | Set_dynamic { dyn_value = v, _; llsc; _ } ->
              go_scalar v;
              go_scalar llsc
          | Set_from_vec { arg = a, _; _ } -> go_scalar a
          | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _
          | Workgroup_barrier | Tile_mma _ ->
              ()
        and go_scalar (llsc : scalar_t) =
          match llsc with
          | Local_scope { body; _ } -> go body
          | Get_dynamic { dyn_value = v, _; _ } -> go_scalar v
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              go_scalar a;
              go_scalar b;
              go_scalar c
          | Binop (_, (a, _), (b, _)) ->
              go_scalar a;
              go_scalar b
          | Unop (_, (a, _)) -> go_scalar a
          | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _
            ->
              ()
        in
        go fc.body;
        !acc
      in
      let sym_free s = not (List.mem bound_inside s ~equal:Indexing.equal_symbol) in
      (* The symbols a condition mentions, or [None] when it reads memory (or a [Concat] index,
         whose components this classification does not decompose). *)
      let idx_syms = function
        | Indexing.Fixed_idx _ | Indexing.Sub_axis -> Some []
        | Indexing.Iterator s -> Some [ s ]
        | Indexing.Affine { symbols; _ } -> Some (List.map symbols ~f:snd)
        | Indexing.Concat _ -> None
      in
      let union a b = match (a, b) with Some x, Some y -> Some (x @ y) | _ -> None in
      let rec cond_syms (llsc : scalar_t) =
        match llsc with
        | Constant _ | Constant_bits _ -> Some []
        | Embed_index idx -> idx_syms idx
        | Ternop (_, (a, _), (b, _), (c, _)) ->
            union (cond_syms a) (union (cond_syms b) (cond_syms c))
        | Binop (_, (a, _), (b, _)) -> union (cond_syms a) (cond_syms b)
        | Unop (_, (a, _)) -> cond_syms a
        | Get _ | Get_local _ | Get_dynamic _ | Get_merge_buffer _ | Local_scope _ -> None
      in
      let hardware_free syms =
        List.for_all syms ~f:(fun s ->
            match Map.find loop_axes s with
            | None | Some Serial | Some Unrolled -> true
            | Some (Grid | Workgroup | Workgroup_reduce | Vectorized) -> false)
      in
      let output_mask (c : scalar_t) =
        match c with
        | Binop (Ops.Cmplt, (Embed_index e, _), (Constant n, _)) ->
            Array.existsi idcs0 ~f:(fun a idx ->
                Indexing.equal_axis_index idx e && Float.(n <= of_int tgt_dims.(a)))
        | _ -> false
      in
      (* Only the invariant conditions gate the transfers; the masks stay where they are. *)
      let transfer_conds =
        List.filter conds0 ~f:(fun (c, _) ->
            match cond_syms c with
            | Some syms when List.for_all syms ~f:sym_free -> true
            | Some syms when hardware_free syms || output_mask c -> false
            | _ ->
                invalid_arg
                  ("Schedule.Privatize: accesses of " ^ Tn.debug_name target
                 ^ " sit under an If guard that is neither invariant across the accumulation nor a \
                    mask on one thread's own iterations (it reads memory, or restricts threads); \
                    cannot gate the init/store-back with it"))
      in
      (* Loops bound inside the subtree, by index symbol (union over access paths). *)
      let inner_loop s =
        List.find_map !accesses ~f:(fun (_, stack, _) ->
            List.find stack ~f:(fun fl -> Indexing.equal_symbol fl.index s))
      in
      (* Per axis: tile part (terms over inner loops) and outer part. The [over] symbol itself must
         not occur — the accumulator is carried across that loop. *)
      let decomp =
        Array.map idcs0 ~f:(fun idx ->
            match terms_of_index idx with
            | None -> invalid_arg "Schedule.Privatize: Concat indices are unsupported"
            | Some (terms, offset) ->
                if List.exists terms ~f:(fun (_, s) -> Indexing.equal_symbol s over) then
                  invalid_arg
                    ("Schedule.Privatize: the target's indices mention the accumulation loop "
                   ^ Indexing.symbol_ident over ^ " — nothing to carry the accumulator across");
                let tile_part, outer_part =
                  List.partition_tf terms ~f:(fun (_, s) -> Option.is_some (inner_loop s))
                in
                List.iter tile_part ~f:(fun (c, s) ->
                    if c <= 0 then
                      invalid_arg "Schedule.Privatize: nonpositive coefficient on an inner index";
                    let fl = Option.value_exn (inner_loop s) in
                    if (not (equal_axis_type fl.axis Serial)) || fl.from_ <> 0 then
                      invalid_arg
                        ("Schedule.Privatize: inner index loop " ^ Indexing.symbol_ident s
                       ^ " must be Serial starting at 0 (a workgroup-indexed private tile would \
                          store back other threads' elements)"));
                (tile_part, outer_part, offset))
      in
      let extent s =
        let fl = Option.value_exn (inner_loop s) in
        fl.to_ - fl.from_ + 1
      in
      let tile_axes =
        Array.filter_mapi decomp ~f:(fun a (tp, _, _) ->
            if List.is_empty tp then None
            else Some (a, List.fold tp ~init:1 ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))))
      in
      let scalar_acc = Array.is_empty tile_axes in
      let tile_dims = if scalar_acc then [| 1 |] else Array.map tile_axes ~f:snd in
      let prec = Lazy.force target.Tn.storage_prec in
      let tile =
        Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
          ~label:("acc" :: target.Tn.label)
          ~unpadded_dims:(lazy tile_dims)
          ~padding:(lazy None)
          ()
      in
      Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.Local 176;
      ignore (get_node opt.traced_store tile : traced_array);
      let tile_read_idcs =
        if scalar_acc then [| Indexing.Fixed_idx 0 |]
        else
          Array.map tile_axes ~f:(fun (a, _) ->
              let tp, _, _ = decomp.(a) in
              normalize_affine ~terms:tp ~offset:0)
      in
      (* Init-load and store-back nests over fresh serial symbols (two independent sets). *)
      let transfer ~into_tile =
        let fresh_syms =
          Array.fold_right tile_axes
            ~init:(Map.empty (module Indexing.Symbol))
            ~f:(fun (a, _) m ->
              let tp, _, _ = decomp.(a) in
              List.fold tp ~init:m ~f:(fun m (_, s) ->
                  if Map.mem m s then m else Map.set m ~key:s ~data:(Indexing.get_symbol ())))
        in
        let load_sym s = Option.value (Map.find fresh_syms s) ~default:s in
        let subst_terms terms = List.map terms ~f:(fun (c, s) -> (c, load_sym s)) in
        let src_idcs =
          Array.map decomp ~f:(fun (tp, op_, off) ->
              normalize_affine ~terms:(subst_terms tp @ op_) ~offset:off)
        in
        let t_idcs =
          if scalar_acc then [| Indexing.Fixed_idx 0 |]
          else
            Array.map tile_axes ~f:(fun (a, _) ->
                let tp, _, _ = decomp.(a) in
                normalize_affine ~terms:(subst_terms tp) ~offset:0)
        in
        let stmt =
          if into_tile then
            Set { tn = tile; idcs = t_idcs; llsc = Get (target, src_idcs); debug = "" }
          else Set { tn = target; idcs = src_idcs; llsc = Get (tile, t_idcs); debug = "" }
        in
        (* Per-axis edge guards (construct-then-fold; they survive only for non-dividing tiles). *)
        let stmt =
          Array.fold tile_axes ~init:stmt ~f:(fun stmt (a, _) ->
              let cond =
                Binop
                  ( Ops.Cmplt,
                    (Embed_index src_idcs.(a), iprec),
                    (Constant (Float.of_int tgt_dims.(a)), iprec) )
              in
              If { cond = (cond, iprec); body = stmt })
        in
        let nest =
          Map.fold fresh_syms ~init:stmt ~f:(fun ~key:s ~data:s' body ->
              For_loop { index = s'; from_ = 0; to_ = extent s - 1; body; axis = Serial })
        in
        (* Carry the invariant part of the accesses' guard chain onto the transfers: only the lanes
           that update their private accumulator may load and store it back (PR #91 review). The
           iteration and output masks are deliberately absent — see the classification above. *)
        List.fold transfer_conds ~init:nest ~f:(fun body cond -> If { cond; body })
      in
      let remapped =
        remap_reads ~writes:true ~source:target ~from_idcs:idcs0 ~tile ~tile_idcs:tile_read_idcs
          fc.body
      in
      unflat_lines
        [
          transfer ~into_tile:true; for_loop { fc with body = remapped }; transfer ~into_tile:false;
        ])
  |> fun llc -> { opt with llc }

(** {2 [Split_reduce]: deterministic two-pass split reduction (gh-ocannl-484)}

    Parallel reductions without atomics. The Serial reduction loop [axis] is split into [num_blocks]
    contiguous chunks; each chunk accumulates into its own row of a fresh scratch node [partials] of
    dims [num_blocks x target-dims] (minted in the tile namespace, placed [On_device], registered in
    the traced store), initialized to the accumulation identity; a synthesized combine statement
    then folds the partials into [target] in a fixed balanced-tree order. Both passes are ordinary
    statements of the routine: the fresh block loop is freely annotatable (its index pins the
    partials row, so parallelizing it is race-free), the [partials] producer/consumer pair is
    exactly the materialized cross-nest edge kernel fission cuts at, and the combine tree is a
    function of the schedule alone — reproducible run to run, unlike atomics. Within each chunk the
    original serial order and rounding are preserved; across chunks the reduction is reassociated
    (the same license as [Swap] of accumulations), so results are deterministic {e per schedule},
    not bitwise-equal to the unsplit serial reduction — schedule identity pins numerics.

    Two accumulation forms are recognized in the [axis] subtree (which must contain no other access
    of [target], and the rest of the statement must not touch [target] either — its combined value
    exists only after the combine statement):

    - Static: a single rmw [Set] — [target[idcs] := target[idcs] ⊕ e] with
      [⊕ ∈ {Add, Max, Min, Mul}] (or FMA), [idcs] free of [axis]. Each loop enclosing [axis] in the
      statement must pin exactly one component of [idcs] (injectively, one symbol per component):
      distinct enclosing iterations then use distinct partial cells — no partial is re-initialized —
      and the combine nest re-iterates exactly the written cells through per-component fresh
      symbols. The per-block init [partials := identity] sits inside the block loop, so no separate
      zeroing pass is needed.
    - Dynamic (the gh-466 embedding-backward scatter): a single [Set_dynamic] add-accumulation of
      its own row ([Get_dynamic] rmw form as built by [rewrite_one_hot_reductions]), possibly under
      guards. The write is redirected to [partials] with the block index prepended (dynamic axis
      shifted by one): within a block, colliding rows stay serial; across blocks, rows live in
      disjoint partials slices — so the block loop parallelizes what the scatter alone cannot
      (gh-484 task 2). Rows are data-dependent, so [partials] is zeroed by a preceding whole-node
      [Zero_out] statement (its own fission segment) and the combine covers all of [target]. *)

exception Split_reduce_inner_cell of Indexing.symbol list
(** Raised by {!apply_split_reduce}'s static-form pinning discipline when the accumulation cell
    mentions symbols bound {e inside} the reduction loop — the one rejection cause a loop
    interchange can remove (gh-ocannl-537). Carries the offending symbols so seeding can build the
    enabling [Swap] chain from the recognizer's own verdict rather than from its message;
    {!apply_opt_op} converts it to the [Invalid_argument] the [apply]/[op_legality] surface
    documents, so this constructor never escapes to schedule application. *)

let apply_split_reduce ~axis ~target ~num_blocks ~block_index ~inner_index ~combine_indices
    (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  if num_blocks < 2 then invalid_arg "Schedule.Split_reduce: num_blocks must be at least 2";
  let tgt_dims = Lazy.force target.Tn.dims in
  let rank = Array.length tgt_dims in
  if List.length combine_indices <> rank then
    invalid_arg
      (Printf.sprintf
         "Schedule.Split_reduce: %d combine indices for a rank-%d target (mint the op with \
          Schedule.split_reduce)"
         (List.length combine_indices) rank);
  let prec = Lazy.force target.Tn.storage_prec in
  let iprec = Ops.index_prec () in
  (* Locate the top-level statement holding the axis loop; the combine statement goes right after it
     (and the scatter form's scratch zeroing right before it). *)
  let stmts = flat_lines [ opt.llc ] in
  let stmt =
    (* [~in_scopes:false]: the combine (and the scatter form's zeroing) are inserted at statement
       level around [stmt], which only sequences correctly if the reduction itself runs there — a
       reduction nested in a [Local_scope] is consumed within its own statement, before the combine
       would run. So this op genuinely wants the statement-level walk, and its enclosing path below
       assumes it. *)
    match List.find stmts ~f:(fun s -> Option.is_some (find_loop ~in_scopes:false axis s)) with
    | Some s -> s
    | None ->
        invalid_arg
          ("Schedule.Split_reduce: no statement-level For_loop with index "
         ^ Indexing.symbol_ident axis ^ " in this routine")
  in
  (* [target] accesses confined to the axis subtree: the rest of the statement (checked here on the
     statement with the subtree excised) must not touch it. *)
  let rec touches llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> false
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; fallback; _ } ->
        Tn.equal d_tn target || Tn.equal a_tn target || Tn.equal b_tn target || touches fallback
    | Zero_out tn -> Tn.equal tn target
    | Seq (a, b) -> touches a || touches b
    | For_loop { body; _ } -> touches body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> touches_scalar c.init) || touches body
    | Set { tn; llsc; _ } -> Tn.equal tn target || touches_scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        Tn.equal tn target || touches_scalar v || touches_scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } -> Tn.equal tn target || touches_scalar a
    | Set_local (_, llsc) -> touches_scalar llsc
    | If { cond = c, _; body } -> touches_scalar c || touches body
  and touches_scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> touches body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ | Get_merge_buffer _ -> false
    | Get (tn, _) -> Tn.equal tn target
    | Get_dynamic { tn; dyn_value = v, _; _ } -> Tn.equal tn target || touches_scalar v
    | Ternop (_, (a, _), (b, _), (c, _)) -> touches_scalar a || touches_scalar b || touches_scalar c
    | Binop (_, (a, _), (b, _)) -> touches_scalar a || touches_scalar b
    | Unop (_, (a, _)) -> touches_scalar a
  in
  if touches (rewrite_loop ~what:"Schedule.Split_reduce" ~sym:axis stmt ~f:(fun _ -> Noop)) then
    invalid_arg
      ("Schedule.Split_reduce: " ^ Tn.debug_name target
     ^ " is accessed in the same statement outside the reduction loop — its combined value only \
        exists after the combine statement");
  (* Loops and If guards enclosing the axis loop within the statement. *)
  let rec enclosing_path path guarded llc =
    match llc with
    | For_loop { index; _ } when Indexing.equal_symbol index axis -> Some (List.rev path, guarded)
    | For_loop { index; from_; to_; body; axis = ty } ->
        enclosing_path ({ index; from_; to_; body = Noop; axis = ty } :: path) guarded body
    | Seq (a, b) -> (
        match enclosing_path path guarded a with
        | Some _ as r -> r
        | None -> enclosing_path path guarded b)
    | If { body; _ } -> enclosing_path path true body
    | _ -> None
  in
  let enclosing, guarded_path =
    match enclosing_path [] false stmt with Some r -> r | None -> assert false
  in
  let block_it = Indexing.Iterator block_index in
  let mint_partials () =
    let partials =
      Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
        ~label:("partials" :: target.Tn.label)
        ~unpadded_dims:(lazy (Array.append [| num_blocks |] tgt_dims))
        ~padding:(lazy None)
        ()
    in
    Tn.Placements.update opt.Low_level.optimize_ctx.placements partials Tn.On_device 184;
    partials
  in
  (* The fixed-order balanced combine tree: a pure function of the schedule ([num_blocks]), so a
     given schedule always computes bitwise-identical results, however the passes are annotated. *)
  let build_combine ~partials ~op ~c_idcs ~loops =
    let rec tree lo hi =
      if hi - lo = 1 then Get (partials, Array.append [| Indexing.Fixed_idx lo |] c_idcs)
      else
        let mid = lo + ((hi - lo) / 2) in
        Binop (op, (tree lo mid, prec), (tree mid hi, prec))
    in
    let set =
      Set
        {
          tn = target;
          idcs = c_idcs;
          llsc = Binop (op, (Get (target, c_idcs), prec), (tree 0 num_blocks, prec));
          debug = "";
        }
    in
    List.fold_right loops ~init:set ~f:(fun (s, lo, hi) body ->
        For_loop { index = s; from_ = lo; to_ = hi; body; axis = Serial })
  in
  let combine_stmt = ref Noop in
  let zero_stmt = ref None in
  let rewritten =
    rewrite_loop ~what:"Schedule.Split_reduce" ~sym:axis stmt ~f:(fun fc ->
        if not (equal_axis_type fc.axis Serial) then
          invalid_arg "Schedule.Split_reduce: the reduction loop must be Serial";
        if fc.from_ <> 0 then
          invalid_arg
            ("Schedule.Split_reduce: loop " ^ Indexing.symbol_ident axis
           ^ " must start at 0 (lowering guarantees this)");
        let n = fc.to_ + 1 in
        (* Scan the subtree for the accumulation of [target]. *)
        let writes = ref [] and dyn_writes = ref [] in
        let reads = ref [] and dyn_reads = ref [] in
        let rec scan llc =
          match llc with
          | Noop | Comment _ | Declare_local _ | Workgroup_barrier -> ()
          | Staged_compilation _ ->
              invalid_arg
                "Schedule.Split_reduce: opaque Staged_compilation under the reduction loop"
          | Tile_mma _ -> invalid_arg "Schedule.Split_reduce: apply Split_reduce before Tensorize"
          | Zero_out tn ->
              if Tn.equal tn target then
                invalid_arg
                  "Schedule.Split_reduce: Zero_out of the target inside the reduction loop"
          | Seq (a, b) ->
              scan a;
              scan b
          | For_loop { body; _ } -> scan body
          | Scan_loop _ ->
              invalid_arg
                "Schedule.Split_reduce: a Scan_loop under the reduction loop is unsupported"
          | Set { tn; idcs; llsc; _ } ->
              if Tn.equal tn target then writes := (idcs, llsc) :: !writes;
              scan_scalar llsc
          | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, _; llsc; _ } ->
              if Tn.equal tn target then dyn_writes := (idcs, dyn_axis, v, llsc) :: !dyn_writes;
              scan_scalar v;
              scan_scalar llsc
          | Set_from_vec { tn; arg = a, _; _ } ->
              if Tn.equal tn target then
                invalid_arg "Schedule.Split_reduce: vector writes to the target are unsupported";
              scan_scalar a
          | Set_local (_, llsc) -> scan_scalar llsc
          | If { cond = c, _; body } ->
              scan_scalar c;
              scan body
        and scan_scalar (llsc : scalar_t) =
          match llsc with
          | Local_scope { body; _ } -> scan body
          | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ | Get_merge_buffer _ -> ()
          | Get (tn, idcs) -> if Tn.equal tn target then reads := idcs :: !reads
          | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, _ } ->
              if Tn.equal tn target then dyn_reads := (idcs, dyn_axis) :: !dyn_reads;
              scan_scalar v
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              scan_scalar a;
              scan_scalar b;
              scan_scalar c
          | Binop (_, (a, _), (b, _)) ->
              scan_scalar a;
              scan_scalar b
          | Unop (_, (a, _)) -> scan_scalar a
        in
        scan fc.body;
        let chunk = (n + num_blocks - 1) / num_blocks in
        let subst body =
          map_code
            ~fidx:
              (subst_axis_index ~sym:axis
                 ~by:{ terms = [ (chunk, block_index); (1, inner_index) ]; offset = 0 })
            body
        in
        (* Remainder guard, construct-then-fold exactly as for [Split]. *)
        let guard body =
          if chunk * num_blocks = n then body
          else
            let cond =
              Binop
                ( Ops.Cmplt,
                  ( Embed_index
                      (Indexing.Affine
                         { symbols = [ (chunk, block_index); (1, inner_index) ]; offset = 0 }),
                    iprec ),
                  (Constant (Float.of_int n), iprec) )
            in
            If { cond = (cond, iprec); body }
        in
        let pass1 ~init body =
          let inner =
            For_loop { index = inner_index; from_ = 0; to_ = chunk - 1; body; axis = Serial }
          in
          For_loop
            {
              index = block_index;
              from_ = 0;
              to_ = num_blocks - 1;
              body = (match init with Some i -> Seq (i, inner) | None -> inner);
              axis = Serial;
            }
        in
        match (!writes, !dyn_writes) with
        | [], [] ->
            invalid_arg
              ("Schedule.Split_reduce: " ^ Tn.debug_name target
             ^ " is not written under the reduction loop")
        | [ (idcs, llsc) ], [] ->
            (* The static form: [target[idcs] := target[idcs] ⊕ e]. *)
            if not (List.is_empty !dyn_reads) then
              invalid_arg
                ("Schedule.Split_reduce: mixed static write and dynamic reads of "
               ^ Tn.debug_name target);
            if guarded_path then
              invalid_arg
                "Schedule.Split_reduce: the reduction loop sits under an If guard (v1 supports \
                 unguarded static accumulations)";
            let acc_is = function
              | Get (g, gi) -> Tn.equal g target && [%equal: Indexing.axis_index array] gi idcs
              | _ -> false
            in
            let op =
              match llsc with
              | Binop (op, (t, _), _) when acc_is t -> op
              | Binop (op, _, (t, _)) when acc_is t -> op
              | Ternop (Ops.FMA, _, _, (t, _)) when acc_is t -> Ops.Add
              | _ ->
                  invalid_arg
                    ("Schedule.Split_reduce: the write of " ^ Tn.debug_name target
                   ^ " is not a read-modify-write accumulation")
            in
            let identity =
              match op with
              | Ops.Add -> 0.
              | Ops.Max -> Float.neg_infinity
              | Ops.Min -> Float.infinity
              | Ops.Mul -> 1.
              | _ ->
                  invalid_arg
                    ("Schedule.Split_reduce: unsupported accumulation operator "
                    ^ Sexp.to_string (Ops.sexp_of_binop op))
            in
            (match !reads with
            | [ ridcs ] when [%equal: Indexing.axis_index array] ridcs idcs -> ()
            | _ ->
                invalid_arg
                  ("Schedule.Split_reduce: v1 requires the only read of " ^ Tn.debug_name target
                 ^ " under the reduction loop to be the accumulation's own running total"));
            (* Injectivity discipline: every enclosing loop pins exactly one component of the
               accumulation cell, each component mentions at most one symbol, and only enclosing
               symbols occur — distinct enclosing iterations then hit distinct partial cells (the
               per-block init runs once per cell) and the combine nest re-iterates exactly the
               written cells. *)
            let enclosing_syms = List.map enclosing ~f:(fun fl -> fl.index) in
            let comp_terms =
              Array.map idcs ~f:(fun idx ->
                  match terms_of_index idx with
                  | None -> invalid_arg "Schedule.Split_reduce: Concat indices are unsupported"
                  | Some (terms, off) -> (terms, off))
            in
            let all_syms =
              Array.to_list comp_terms |> List.concat_map ~f:(fun (t, _) -> List.map t ~f:snd)
            in
            if List.exists all_syms ~f:(Indexing.equal_symbol axis) then
              invalid_arg
                ("Schedule.Split_reduce: the accumulation cell mentions the reduction loop "
               ^ Indexing.symbol_ident axis ^ " — not a reduction over it");
            (* The cause gh-ocannl-537 composes around: symbols bound INSIDE the reduction loop
               (OCANNL lowers conv gradients with the accumulated channel loop innermost). Raised
               structurally with the whole offending set, so seeding can ask for it by
               {!split_reduce_hoist} instead of parsing the message; [apply_opt_op] converts it to
               the [Invalid_argument] every other caller expects. *)
            (match
               List.dedup_and_sort ~compare:Indexing.compare_symbol
                 (List.filter all_syms ~f:(fun s ->
                      not (List.mem enclosing_syms s ~equal:Indexing.equal_symbol)))
             with
            | [] -> ()
            | inner -> raise (Split_reduce_inner_cell inner));
            List.iter enclosing_syms ~f:(fun s ->
                match List.count all_syms ~f:(Indexing.equal_symbol s) with
                | 1 -> ()
                | 0 ->
                    invalid_arg
                      ("Schedule.Split_reduce: enclosing loop " ^ Indexing.symbol_ident s
                     ^ " does not index the accumulation cell — Swap it inside "
                     ^ Indexing.symbol_ident axis ^ " (or split-reduce it) first")
                | _ ->
                    invalid_arg
                      ("Schedule.Split_reduce: enclosing loop " ^ Indexing.symbol_ident s
                     ^ " indexes the accumulation cell more than once"));
            Array.iter comp_terms ~f:(fun (terms, _) ->
                if List.length terms > 1 then
                  invalid_arg
                    "Schedule.Split_reduce: v1 requires each component of the accumulation cell to \
                     mention at most one symbol");
            let partials = mint_partials () in
            ignore (get_node opt.traced_store partials : traced_array);
            let p_idcs = Array.append [| block_it |] idcs in
            let body =
              remap_reads ~writes:true ~source:target ~from_idcs:idcs ~tile:partials
                ~tile_idcs:p_idcs (subst fc.body)
            in
            let init = Set { tn = partials; idcs = p_idcs; llsc = Constant identity; debug = "" } in
            let range_of s =
              List.find_exn enclosing ~f:(fun fl -> Indexing.equal_symbol fl.index s)
            in
            let combine_loops = ref [] in
            let c_idcs =
              Array.mapi comp_terms ~f:(fun a (terms, off) ->
                  let ci = List.nth_exn combine_indices a in
                  match terms with
                  | [] -> Indexing.Fixed_idx off
                  | [ (c, s) ] ->
                      let fl = range_of s in
                      combine_loops := (ci, fl.from_, fl.to_) :: !combine_loops;
                      normalize_affine ~terms:[ (c, ci) ] ~offset:off
                  | _ -> assert false)
            in
            combine_stmt := build_combine ~partials ~op ~c_idcs ~loops:(List.rev !combine_loops);
            pass1 ~init:(Some init) (guard body)
        | [], [ (idcs, dyn_axis, dv_scalar, llsc) ] ->
            (* The dynamic (scatter) form, gh-466: [target[.., e, ..] += g] with [e] data-dependent
               (possibly under guards — a guarded pad iteration merely leaves its partials at 0). *)
            if not (List.is_empty !reads) then
              invalid_arg
                ("Schedule.Split_reduce: mixed dynamic write and static reads of "
               ^ Tn.debug_name target);
            let acc_is = function
              | Get_dynamic { tn = g; idcs = gi; dyn_axis = ga; dyn_value = gv, _ } ->
                  Tn.equal g target && ga = dyn_axis
                  && [%equal: Indexing.axis_index array] gi idcs
                  && equal_scalar_t gv dv_scalar
              | _ -> false
            in
            (match llsc with
            | Binop (Ops.Add, (t, _), _) when acc_is t -> ()
            | Binop (Ops.Add, _, (t, _)) when acc_is t -> ()
            | _ ->
                invalid_arg
                  ("Schedule.Split_reduce: the dynamic write of " ^ Tn.debug_name target
                 ^ " is not an add-accumulation of its own row (the gh-466 scatter form)"));
            (match !dyn_reads with
            | [ (ridcs, ra) ] when ra = dyn_axis && [%equal: Indexing.axis_index array] ridcs idcs
              ->
                ()
            | _ ->
                invalid_arg
                  ("Schedule.Split_reduce: v1 requires the only read of " ^ Tn.debug_name target
                 ^ " under the reduction loop to be the scatter's own row"));
            let partials = mint_partials () in
            let traced = get_node opt.traced_store partials in
            traced.zeroed_out <- true;
            let rec redirect llc =
              match llc with
              | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _
              | Workgroup_barrier | Tile_mma _ ->
                  llc
              | Seq (a, b) -> Seq (redirect a, redirect b)
              | For_loop fc' -> For_loop { fc' with body = redirect fc'.body }
              | Scan_loop sc ->
                  Scan_loop
                    {
                      sc with
                      carried =
                        List.map sc.carried ~f:(fun c -> { c with init = redirect_scalar c.init });
                      body = redirect sc.body;
                    }
              | Set ({ llsc; _ } as s) -> Set { s with llsc = redirect_scalar llsc }
              | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, p; llsc; debug }
                when Tn.equal tn target ->
                  Set_dynamic
                    {
                      tn = partials;
                      idcs = Array.append [| block_it |] idcs;
                      dyn_axis = dyn_axis + 1;
                      dyn_value = (redirect_scalar v, p);
                      llsc = redirect_scalar llsc;
                      debug;
                    }
              | Set_dynamic ({ dyn_value = v, p; llsc; _ } as sd) ->
                  Set_dynamic
                    { sd with dyn_value = (redirect_scalar v, p); llsc = redirect_scalar llsc }
              | Set_from_vec ({ arg = a, p; _ } as sv) ->
                  Set_from_vec { sv with arg = (redirect_scalar a, p) }
              | Set_local (id, llsc) -> Set_local (id, redirect_scalar llsc)
              | If { cond = c, p; body } ->
                  If { cond = (redirect_scalar c, p); body = redirect body }
            and redirect_scalar (llsc : scalar_t) : scalar_t =
              match llsc with
              | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, p } when Tn.equal tn target ->
                  Get_dynamic
                    {
                      tn = partials;
                      idcs = Array.append [| block_it |] idcs;
                      dyn_axis = dyn_axis + 1;
                      dyn_value = (redirect_scalar v, p);
                    }
              | Get_dynamic ({ dyn_value = v, p; _ } as gd) ->
                  Get_dynamic { gd with dyn_value = (redirect_scalar v, p) }
              | Local_scope ({ body; _ } as ls) -> Local_scope { ls with body = redirect body }
              | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _
              | Embed_index _ ->
                  llsc
              | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
                  Ternop
                    (op, (redirect_scalar a, pa), (redirect_scalar b, pb), (redirect_scalar c, pc))
              | Binop (op, (a, pa), (b, pb)) ->
                  Binop (op, (redirect_scalar a, pa), (redirect_scalar b, pb))
              | Unop (op, (a, pa)) -> Unop (op, (redirect_scalar a, pa))
            in
            let body = redirect (subst fc.body) in
            zero_stmt := Some (Zero_out partials);
            let c_idcs = Array.of_list_map combine_indices ~f:(fun s -> Indexing.Iterator s) in
            let loops = List.mapi combine_indices ~f:(fun a s -> (s, 0, tgt_dims.(a) - 1)) in
            combine_stmt := build_combine ~partials ~op:Ops.Add ~c_idcs ~loops;
            pass1 ~init:None (guard body)
        | _ ->
            invalid_arg
              ("Schedule.Split_reduce: multiple writes of " ^ Tn.debug_name target
             ^ " under the reduction loop (v1 supports a single accumulation)"))
  in
  let new_stmts =
    List.concat_map stmts ~f:(fun s ->
        if phys_equal s stmt then
          (match !zero_stmt with Some z -> [ z ] | None -> []) @ [ rewritten; !combine_stmt ]
        else [ s ])
  in
  { opt with llc = unflat_lines new_stmts }

(** After [Tensorize] has replaced the inner micro-kernel, contract the enclosing chain of serial
    loops that carry the operands but not the accumulator — for a multi-window conv the whole
    kernel-window nest [kh; kw], not just the innermost loop (gh-ocannl-501). The search is
    outermost-first: the transfers land around the outermost qualifying loop, so the fragment stays
    resident across the entire chain and the store-back executes exactly once per output tile (which
    is what lets [Fuse_epilogue] relocate an elementwise tail there). The synthesized local tile has
    ordinary scalar semantics: lane 0 initializes it, each per-iteration [Tile_mma] accumulates into
    it, and lane 0 stores it back. Metal recognizes the marked three-part region and maps the tile
    to persistent simdgroup fragments; unsupported backend calls keep the local-array fallback. *)
let contract_tensorized_accumulator ~lane ~(masks : pad_mask list) (opt : Low_level.optimized) :
    Low_level.optimized =
  let open Low_level in
  (* [Tensorize] currently identifies one micro-kernel site per scheduled routine. Keep contraction
     single-shot to avoid conflating independent accumulator lifetimes if multi-site tensorization
     is introduced; that extension should promote and mark each site explicitly. *)
  let promoted = ref None in
  (* Pad masks (gh-ocannl-485) force the contraction: with the accumulator retargeted to a padded
     scratch fragment, only the valid region round-trips through the target — the transfers below
     carry the masks. On a GPU pipeline (recognized by staged workgroup-shared tiles) the masked
     fragment is placed in workgroup-shared memory: the guarded transfers are not recognizable as a
     fragment-resident scope, and per-thread [Local] arrays are not loadable by the simdgroup/wmma
     intrinsics — a threadgroup-resident fragment keeps the intrinsic path firing, at the cost of a
     per-statement load/store round-trip and a barrier pair around the reduction loop. *)
  let masked = not (List.is_empty masks) in
  let shared_frag = masked && not (Set.is_empty opt.workgroup_shared) in
  let iprec = Ops.index_prec () in
  let mentions sym idx =
    match terms_of_index idx with
    | Some (terms, _) -> List.exists terms ~f:(fun (_, s) -> Indexing.equal_symbol s sym)
    | None -> false
  in
  let idcs_mention sym = Array.exists ~f:(mentions sym) in
  let same_lane s = Indexing.equal_symbol s lane in
  let rec matching_tiles acc = function
    | Tile_mma { lane = l; _ } as tm when same_lane l -> tm :: acc
    | Seq (a, b) -> matching_tiles (matching_tiles acc a) b
    | For_loop { body; _ } | If { body; _ } -> matching_tiles acc body
    | _ -> acc
  in
  let rec touches_outside_tile target = function
    | Tile_mma { d = tn, _; lane = l; fallback; _ } ->
        if Tn.equal tn target && same_lane l then false else touches_outside_tile target fallback
    | Zero_out tn -> Tn.equal tn target
    | Set { tn; llsc; _ } -> Tn.equal tn target || scalar_touches target llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        Tn.equal tn target || scalar_touches target v || scalar_touches target llsc
    | Set_from_vec { tn; arg = a, _; _ } -> Tn.equal tn target || scalar_touches target a
    | Set_local (_, llsc) -> scalar_touches target llsc
    | Seq (a, b) -> touches_outside_tile target a || touches_outside_tile target b
    | For_loop { body; _ } | If { body; _ } -> touches_outside_tile target body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> scalar_touches target c.init)
        || touches_outside_tile target body
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> false
  and scalar_touches target = function
    | Get (tn, _) | Get_dynamic { tn; _ } -> Tn.equal tn target
    | Local_scope { body; _ } -> touches_outside_tile target body
    | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar_touches target a || scalar_touches target b || scalar_touches target c
    | Binop (_, (a, _), (b, _)) -> scalar_touches target a || scalar_touches target b
    | Unop (_, (a, _)) -> scalar_touches target a
  in
  let rec lane_extent = function
    | For_loop { index; from_; to_; axis = Workgroup; _ } when same_lane index ->
        Some (to_ - from_ + 1)
    | Seq (a, b) -> Option.first_some (lane_extent a) (lane_extent b)
    | For_loop { body; _ } | If { body; _ } -> lane_extent body
    | _ -> None
  in
  let rec bound_symbols acc = function
    | For_loop { index; body; _ } -> bound_symbols (index :: acc) body
    | Seq (a, b) -> bound_symbols (bound_symbols acc a) b
    | If { body; _ } -> bound_symbols acc body
    | _ -> acc
  in
  let fallback_indices target fallback =
    let rec find = function
      | Set { tn; idcs; _ } when Tn.equal tn target -> Some idcs
      | Seq (a, b) -> Option.first_some (find a) (find b)
      | For_loop { body; _ } | If { body; _ } -> find body
      | _ -> None
    in
    find fallback
  in
  let fallback_axes fallback =
    match fallback with
    | For_loop { index = i; body; _ } -> (
        match
          List.filter (flat_lines [ body ]) ~f:(function Noop | Comment _ -> false | _ -> true)
        with
        | [ For_loop { index = j; _ } ] -> Some (i, j)
        | _ -> None)
    | _ -> None
  in
  let add_symbol idx s =
    match terms_of_index idx with
    | Some (terms, offset) -> normalize_affine ~terms:((1, s) :: terms) ~offset
    | None -> invalid_arg "Schedule.Tensorize: Concat accumulator indices are unsupported"
  in
  let lane0 p body =
    let cond =
      Binop (Ops.Cmpeq, (Embed_index (Indexing.Iterator lane), iprec), (Constant 0., iprec))
    in
    For_loop
      {
        index = lane;
        from_ = 0;
        to_ = p - 1;
        body = If { cond = (cond, iprec); body };
        axis = Workgroup;
      }
  in
  (* The shared core of the two contraction sites: mint the fragment, retarget the [Tile_mma] (and
     its fallback's writes) inside [body] via [replace], and assemble [init-load; (barrier;) reloop
     body'; (barrier;) store-back] — the transfers guarded by the pad masks over the fresh fragment
     symbols: the init-load [Where]s out-of-range slots to 0 and the store-back [If]s them away, so
     only the valid region of the padded block round-trips through [target]. *)
  let contract_around ~target ~d_base ~m ~n ~fallback ~simd_width
      ~(reloop : Low_level.t -> Low_level.t) ~(body : Low_level.t) =
    match (fallback_indices target fallback, fallback_axes fallback) with
    | Some original_d_idcs, Some (i, j) ->
        let prec = Lazy.force target.Tn.storage_prec in
        let fragment =
          Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
            ~label:("fragment" :: target.Tn.label)
            ~unpadded_dims:(lazy [| m; n |])
            ~padding:(lazy None)
            ()
        in
        Tn.Placements.update opt.optimize_ctx.placements fragment Tn.Local 178;
        ignore (get_node opt.traced_store fragment : traced_array);
        let fragment_idcs = [| Indexing.Iterator i; Indexing.Iterator j |] in
        let fragment_base = [| Indexing.Fixed_idx 0; Indexing.Fixed_idx 0 |] in
        let fallback =
          remap_reads ~writes:true ~source:target ~from_idcs:original_d_idcs ~tile:fragment
            ~tile_idcs:fragment_idcs fallback
        in
        let replaced = ref false in
        let rec replace = function
          | Tile_mma ({ lane = l; _ } as tm') when same_lane l ->
              replaced := true;
              (* The fragment is a fresh contiguous [m x n] tile, so its leading dimension is its
                 minor extent regardless of the original accumulator's (possibly batched) layout. *)
              Tile_mma { tm' with d = (fragment, fragment_base); ldd = n; fallback }
          | Seq (a, b) -> Seq (replace a, replace b)
          | For_loop f -> For_loop { f with body = replace f.body }
          | If ({ body; _ } as x) -> If { x with body = replace body }
          | other -> other
        in
        let body = replace body in
        assert !replaced;
        let fi = Indexing.get_symbol () and fj = Indexing.get_symbol () in
        let target_idcs = Array.copy d_base in
        let rank = Array.length target_idcs in
        if rank < 2 then invalid_arg "Schedule.Tensorize: accumulator rank must be at least 2";
        (* The row symbol's axis in the original accumulator indexing — the tnode's second-to-last
           axis in the plain case, further out when interior batch axes sit between the roles
           (gh-ocannl-528); the column is always the minor axis ([Tensorize]'s role discipline). *)
        let row_axis =
          let hits =
            Array.filter_mapi original_d_idcs ~f:(fun p idx -> Option.some_if (mentions i idx) p)
          in
          match Array.to_list hits with [ p ] -> p | _ -> rank - 2
        in
        target_idcs.(row_axis) <- add_symbol target_idcs.(row_axis) fi;
        target_idcs.(rank - 1) <- add_symbol target_idcs.(rank - 1) fj;
        let local_idcs = [| Indexing.Iterator fi; Indexing.Iterator fj |] in
        let mask_cond pm =
          let role = if pm.pm_row then fi else fj in
          let idx = normalize_affine ~terms:((1, role) :: pm.pm_terms) ~offset:pm.pm_offset in
          Binop (Ops.Cmplt, (Embed_index idx, iprec), (Constant (Float.of_int pm.pm_bound), iprec))
        in
        let transfer ~into_fragment =
          let stmt =
            if into_fragment then
              let llsc =
                List.fold masks
                  ~init:(Get (target, target_idcs))
                  ~f:(fun rhs pm ->
                    Ternop (Ops.Where, (mask_cond pm, iprec), (rhs, prec), (Constant 0., prec)))
              in
              Set { tn = fragment; idcs = local_idcs; llsc; debug = "" }
            else
              List.fold masks
                ~init:
                  (Set
                     {
                       tn = target;
                       idcs = target_idcs;
                       llsc = Get (fragment, local_idcs);
                       debug = "";
                     })
                ~f:(fun stmt pm -> If { cond = (mask_cond pm, iprec); body = stmt })
          in
          For_loop
            {
              index = fi;
              from_ = 0;
              to_ = m - 1;
              axis = Serial;
              body = For_loop { index = fj; from_ = 0; to_ = n - 1; axis = Serial; body = stmt };
            }
        in
        promoted := Some fragment;
        Some
          (unflat_lines
             ([ lane0 simd_width (transfer ~into_fragment:true) ]
             @ (if shared_frag then [ Workgroup_barrier ] else [])
             @ [ reloop body ]
             @ (if shared_frag then [ Workgroup_barrier ] else [])
             @ [ lane0 simd_width (transfer ~into_fragment:false) ]))
    | _ -> None
  in
  (* Outermost-first: attempt the contraction at each serial loop before descending, so the
     transfers land around the outermost loop of the qualifying chain (an inner qualifying loop is
     then part of the contracted region and is never visited). A loop that fails the conditions —
     typically because its index appears in the accumulator's base indices, i.e. it is an output
     loop — recurses into its body. *)
  let rec rewrite llc =
    match llc with
    | For_loop { index; from_; to_; body; axis }
      when Option.is_none !promoted && equal_axis_type axis Serial && to_ > from_ -> (
        let fc : floop = { index; from_; to_; body; axis } in
        match try_contract fc with
        | Some replaced -> replaced
        | None -> for_loop { fc with body = rewrite fc.body })
    | Seq (a, b) ->
        let a = rewrite a in
        Seq (a, rewrite b)
    | For_loop fc -> For_loop { fc with body = rewrite fc.body }
    | If ({ body; _ } as i) -> If { i with body = rewrite body }
    | other -> other
  and try_contract (fc : floop) =
    match matching_tiles [] fc.body with
    | [ Tile_mma { d = target, d_base; a = a, _; b = b, _; m; n; fallback; _ } ]
      when (not (idcs_mention fc.index d_base))
           && (not (Tn.equal target a))
           && (not (Tn.equal target b))
           && List.for_all (bound_symbols [] fc.body) ~f:(fun s -> not (idcs_mention s d_base))
           && (not (touches_outside_tile target fc.body))
           && List.for_all masks ~f:(fun pm ->
               (* The mask guards must be expressible at the transfer site: every outer term of a
                  guard index must still be bound there. *)
               List.for_all pm.pm_terms ~f:(fun (_, s) ->
                   (not (Indexing.equal_symbol s fc.index))
                   && not (List.mem (bound_symbols [] fc.body) s ~equal:Indexing.equal_symbol)))
      -> (
        match lane_extent fc.body with
        | Some simd_width ->
            contract_around ~target ~d_base ~m ~n ~fallback ~simd_width
              ~reloop:(fun body -> for_loop { fc with body })
              ~body:fc.body
        | None -> None)
    | _ -> None
  in
  let llc = rewrite opt.llc in
  (* Statement-site fallback for masked tensorization without a qualifying enclosing serial loop
     (e.g. a whole-extent block with no reduction blocking, or an extent-1 block loop): the
     transfers wrap the lane loop of the [Tile_mma] itself. *)
  let llc =
    if masked && Option.is_none !promoted then
      let rec go llc =
        match llc with
        | For_loop ({ index; from_; to_; body; _ } as fc)
          when same_lane index && Option.is_none !promoted -> (
            match matching_tiles [] body with
            | [ Tile_mma { d = target, _; a = a, _; b = b, _; _ } ]
              when Tn.equal target a || Tn.equal target b ->
                invalid_arg
                  ("Schedule.Tensorize: pad-masked accumulator " ^ Tn.debug_name target
                 ^ " coincides with an operand")
            | [ Tile_mma { d = target, d_base; m; n; fallback; _ } ] -> (
                match
                  contract_around ~target ~d_base ~m ~n ~fallback
                    ~simd_width:(to_ - from_ + 1)
                    ~reloop:Fn.id ~body:(For_loop fc)
                with
                | Some replaced -> replaced
                | None -> For_loop fc)
            | _ -> For_loop { fc with body = go body })
        | Seq (a, b) ->
            let a = go a in
            Seq (a, go b)
        | For_loop fc -> For_loop { fc with body = go fc.body }
        | If ({ body; _ } as x) -> If { x with body = go body }
        | other -> other
      in
      go llc
    else llc
  in
  if masked && Option.is_none !promoted then
    invalid_arg
      "Schedule.Tensorize: pad masks require the accumulator contraction, but no contraction site \
       qualified (an outer term of a pad guard may be bound inside the reduction loop)";
  match !promoted with
  | None -> { opt with llc }
  | Some fragment ->
      if masked then
        {
          opt with
          llc;
          workgroup_shared =
            (if shared_frag then Set.add opt.workgroup_shared fragment else opt.workgroup_shared);
        }
      else { opt with llc; simdgroup_fragments = Set.add opt.simdgroup_fragments fragment }

let apply_tensorize op (opt : Low_level.optimized) : Low_level.optimized =
  match op with
  | Tensorize { i; j; k; lane; simd_width; tile } ->
      let llc, masks =
        tensorize_llc
          ~zero_fringe:(Set.mem opt.Low_level.zero_fringe)
          ~i ~j ~k ~lane ~simd_width ~tile opt.Low_level.llc
      in
      let opt = { opt with Low_level.llc } in
      contract_tensorized_accumulator ~lane ~masks opt
  | _ -> assert false

(** Epilogue fusion (gh-ocannl-486): fold the sole-consumer, index-space-compatible elementwise tail
    that re-reads [target] — the typical bias add / activation / residual after a reduction — into
    [target]'s store-back site, so the tail's separate memory pass over the output disappears and
    the fused routine is a single kernel/segment. Three fusion sites are recognized, in order:

    - the lane-0 fragment store-back synthesized by [contract_tensorized_accumulator] (the tail
      becomes a fourth, lane-0-guarded statement of the marked region — the region stays
      structurally recognizable by [C_syntax.try_mma_fragment_scope], which renders the extra
      statements after the backend's intrinsic block);
    - the [Privatize] tile store-back (per-element, right after the final write);
    - the plain accumulation nest (the tail slides inside the parallel/output loops, right after the
      serial reduction loop — the classic loop fusion).

    Elementwise tails never reorder the reduction, so on the C backends the fused values are BITWISE
    equal to the two-kernel form. The store-back of [target] itself is kept (v1): [target] may be
    observable, and eliding it is a separate dead-store concern.

    Preconditions (checked, [Invalid_argument] otherwise): the tail is the first real statement
    after the last statement writing [target]; it is a perfect Serial nest over exactly [target]'s
    dims whose leaf assigns a different node at the identity index tuple; every read of [target] in
    the tail uses that same tuple; the tail is elementwise (no local scopes, dynamic or merge-buffer
    reads); no later statement mentions [target] (sole consumer); the tail's other operands are not
    written by the reduction statement; and the store-back tiles cover [target]'s index space
    bijectively, with no surplus enclosing loop (extent > 1, not indexing the site) re-executing the
    store-back per iteration (so the relocated tail writes each output element exactly once, from
    the completed accumulation). Nodes related by buffer aliasing ([Tnode.alias_of]) are not
    analyzed — sole-consumption is judged by node identity. *)
let apply_fuse_epilogue ~target ~shared (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  let fail msg = invalid_arg ("Schedule.Fuse_epilogue: " ^ msg) in
  (* Earlier ops in the same schedule leave constructed-then-folded guards ([Split] remainders,
     [Stage] edge guards) that [apply]'s trailing simplify has not erased yet; fold them now so the
     recognition below sees the guard-free structure it targets (transforms fold their own guards,
     schedule-ir-optops §2). Static-index extents are not needed: the relevant guards compare
     schedule-minted affine forms against loop extents. *)
  let opt = { opt with llc = simplify_llc [] opt.llc } in
  let dims = Lazy.force target.Tn.dims in
  let rank = Array.length dims in
  let idx_mentions s idx =
    match terms_of_index idx with
    | Some (terms, _) -> List.exists terms ~f:(fun (_, s') -> Indexing.equal_symbol s s')
    | None -> false
  in
  (* --- Generic statement scanners. --- *)
  let rec writes_tn tn = function
    | Set { tn = t; _ } | Zero_out t | Set_dynamic { tn = t; _ } | Set_from_vec { tn = t; _ } ->
        Tn.equal t tn
    | Tile_mma { d = t, _; fallback; _ } -> Tn.equal t tn || writes_tn tn fallback
    | Seq (a, b) -> writes_tn tn a || writes_tn tn b
    | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> writes_tn tn body
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier | Set_local _ ->
        false
  in
  let rec mentions_tn tn = function
    | Zero_out t -> Tn.equal t tn
    | Set { tn = t; llsc; _ } -> Tn.equal t tn || scalar_mentions tn llsc
    | Set_dynamic { tn = t; dyn_value = v, _; llsc; _ } ->
        Tn.equal t tn || scalar_mentions tn v || scalar_mentions tn llsc
    | Set_from_vec { tn = t; arg = a, _; _ } -> Tn.equal t tn || scalar_mentions tn a
    | Set_local (_, llsc) -> scalar_mentions tn llsc
    | Tile_mma { d = d, _; a = a, _; b = b, _; fallback; _ } ->
        Tn.equal d tn || Tn.equal a tn || Tn.equal b tn || mentions_tn tn fallback
    | Seq (a, b) -> mentions_tn tn a || mentions_tn tn b
    | For_loop { body; _ } | If { body; _ } -> mentions_tn tn body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> scalar_mentions tn c.init) || mentions_tn tn body
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> false
  and scalar_mentions tn = function
    | Get (t, _) | Get_dynamic { tn = t; _ } | Get_merge_buffer (t, _) -> Tn.equal t tn
    | Local_scope { body; _ } -> mentions_tn tn body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar_mentions tn a || scalar_mentions tn b || scalar_mentions tn c
    | Binop (_, (a, _), (b, _)) -> scalar_mentions tn a || scalar_mentions tn b
    | Unop (_, (a, _)) -> scalar_mentions tn a
  in
  (* --- Statement-level layout: the last top-level statement writing [target] (the reduction), the
     tail immediately after it. --- *)
  let stmts = flat_lines [ opt.llc ] in
  let is_real = function Noop | Comment _ -> false | _ -> true in
  let writer_idcs =
    List.filter_mapi stmts ~f:(fun i s -> if is_real s && writes_tn target s then Some i else None)
  in
  let r =
    match List.last writer_idcs with
    | None -> fail (Tn.debug_name target ^ " is never written in this routine")
    | Some r -> r
  in
  let red_stmt = List.nth_exn stmts r in
  (match red_stmt with
  | Zero_out _ -> fail ("the last write of " ^ Tn.debug_name target ^ " is a whole-node Zero_out")
  | _ -> ());
  let t_idx =
    match List.findi stmts ~f:(fun i s -> i > r && is_real s) with
    | Some (i, _) -> i
    | None -> fail ("no statement follows the reduction over " ^ Tn.debug_name target)
  in
  let tail_stmt = List.nth_exn stmts t_idx in
  List.iteri stmts ~f:(fun i s ->
      if i > t_idx && mentions_tn target s then
        fail
          ("the tail is not the sole consumer: a later statement mentions " ^ Tn.debug_name target));
  (* --- Parse and vet the tail: a perfect Serial nest over [target]'s dims, leaf assigning [out] at
     the identity tuple, elementwise, all reads of [target] at that same tuple. --- *)
  let rec parse_tail loops = function
    | For_loop { index; from_ = 0; to_; axis = Serial; body; _ } -> (
        match List.filter (flat_lines [ body ]) ~f:is_real with
        | [ single ] -> parse_tail ((index, to_ + 1) :: loops) single
        | _ -> fail "the epilogue tail must be a perfect nest with a single statement per level")
    | Set { tn; idcs; llsc; debug } -> (List.rev loops, tn, idcs, llsc, debug)
    | _ ->
        fail
          "the statement after the reduction is not an elementwise tail (expected a perfect Serial \
           nest ending in a single assignment)"
  in
  let tail_loops, out, tail_idcs, tail_llsc, tail_debug = parse_tail [] tail_stmt in
  if Tn.equal out target then fail "the tail assigns the reduction output itself";
  let tail_syms = Array.of_list_map tail_loops ~f:fst in
  let tail_extents = Array.of_list_map tail_loops ~f:snd in
  if Array.length tail_syms <> rank then
    fail "the tail nest's depth does not match the reduction output's rank";
  if not (Array.equal ( = ) tail_extents dims) then
    fail "the tail nest's extents do not match the reduction output's dims";
  if not (Array.equal ( = ) (Lazy.force out.Tn.dims) dims) then
    fail "the tail output's dims do not match the reduction output's dims";
  Array.iteri tail_idcs ~f:(fun i idx ->
      match idx with
      | Indexing.Iterator s when Indexing.equal_symbol s tail_syms.(i) -> ()
      | _ -> fail "the tail must assign its output at the identity index tuple");
  let saw_target = ref false in
  let operands = ref [] in
  let rec vet_scalar = function
    | Get (tn, g_idcs) ->
        if Tn.equal tn target then (
          if not (Array.equal Indexing.equal_axis_index g_idcs tail_idcs) then
            fail "every tail read of the reduction output must use the tail's own index tuple";
          saw_target := true)
        else operands := tn :: !operands
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Binop (_, (a, _), (b, _)) ->
        vet_scalar a;
        vet_scalar b
    | Unop (_, (a, _)) -> vet_scalar a
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        vet_scalar a;
        vet_scalar b;
        vet_scalar c
    | Local_scope _ | Get_local _ | Get_merge_buffer _ | Get_dynamic _ ->
        fail "the tail must be elementwise (no local scopes, dynamic or merge-buffer reads)"
  in
  vet_scalar tail_llsc;
  if not !saw_target then fail "the tail does not read the reduction output";
  if mentions_tn out red_stmt then fail "the reduction statement already mentions the tail output";
  List.iter !operands ~f:(fun tn ->
      if writes_tn tn red_stmt then
        fail ("tail operand " ^ Tn.debug_name tn ^ " is written by the reduction statement"));
  (* --- The relocated tail: substitute each tail symbol by the store-back site's index for that
     axis. Site indices never mention the (fresh, distinct) tail symbols, so per-symbol substitution
     composes. --- *)
  let subst_tail ~(site_idcs : Indexing.axis_index array) : Low_level.t =
    let stmt = Set { tn = out; idcs = tail_idcs; llsc = tail_llsc; debug = tail_debug } in
    Array.foldi site_idcs ~init:stmt ~f:(fun ax stmt idx ->
        match terms_of_index idx with
        | Some (terms, offset) ->
            map_code ~fidx:(subst_axis_index ~sym:tail_syms.(ax) ~by:{ terms; offset }) stmt
        | None -> fail "the store-back site's indices must be affine")
  in
  (* Does [idcs], with symbols ranging over [env] extents (zero-based loops), cover [target]'s index
     space bijectively over all enclosing iterations? Per axis: offset 0 and the (coefficient,
     extent) pairs form an exact mixed radix — so the relocated tail writes each output element
     exactly once. A [guarded] axis (a pad-mask range guard on the store-back, gh-ocannl-485) may
     over-cover: the exact radix then spans the padded extent and the guard trims it to the
     dimension, still visiting each valid element exactly once; an unguarded axis must cover the
     dimension exactly. *)
  let covers_bijectively ?guarded ~env (idcs : Indexing.axis_index array) : bool =
    let guarded = Option.value guarded ~default:(Array.create ~len:rank false) in
    Array.length idcs = rank
    && Array.for_alli idcs ~f:(fun ax idx ->
        match terms_of_index idx with
        | None -> false
        | Some (terms, offset) -> (
            offset = 0
            &&
            let ces =
              Option.all
                (List.map terms ~f:(fun (c, s) ->
                     if c <= 0 then None
                     else
                       Option.map (List.Assoc.find env ~equal:Indexing.equal_symbol s) ~f:(fun e ->
                           (c, e))))
            in
            match ces with
            | None -> false
            | Some ces ->
                let sorted = List.sort ces ~compare:(fun (c1, _) (c2, _) -> Int.compare c1 c2) in
                let rec radix expected = function
                  | [] -> if guarded.(ax) then expected >= dims.(ax) else expected = dims.(ax)
                  | (c, e) :: tl -> c = expected && radix (c * e) tl
                in
                radix 1 sorted))
  in
  let iprec = Ops.index_prec () in
  let lane0 ~lane ~width body =
    let cond =
      Binop (Ops.Cmpeq, (Embed_index (Indexing.Iterator lane), iprec), (Constant 0., iprec))
    in
    For_loop
      {
        index = lane;
        from_ = 0;
        to_ = width - 1;
        body = If { cond = (cond, iprec); body };
        axis = Workgroup;
      }
  in
  let is_lane0_guard lane = function
    | Binop (Ops.Cmpeq, (Embed_index (Indexing.Iterator s), _), (Constant z, _))
    | Binop (Ops.Cmpeq, (Constant z, _), (Embed_index (Indexing.Iterator s), _)) ->
        Indexing.equal_symbol lane s && Float.equal z 0.
    | _ -> false
  in
  (* --- Site 1: the lane-0 fragment store-back marked by [contract_tensorized_accumulator]. The
     epilogue becomes a sibling lane-0 statement re-iterating the store-back's tile loops (the same
     symbols — sibling transfer nests already share them), so the marked three-part region keeps its
     shape and the fragment recognizer renders the extra statement after the intrinsics. A
     pad-masked store-back (gh-ocannl-485: the contraction's transfers [If] out-of-range slots of
     the padded block away) is recognized through its guards, which are collected and re-imposed on
     the relocated tail — the tail then writes exactly the valid cells the store-back writes. --- *)
  let is_fragment frag =
    (* Masked non-shared fragments (the CPU padded pipelines) are registered in neither the
       [simdgroup_fragments] nor the [workgroup_shared] set; recognize them by the label
       [contract_around] mints. *)
    (not (Tn.equal frag target))
    && (Set.mem opt.simdgroup_fragments frag
       || Set.mem opt.workgroup_shared frag
       || match frag.Tn.label with "fragment" :: _ -> true | _ -> false)
  in
  let match_storeback = function
    | For_loop { index = lane; from_ = 0; to_; axis = Workgroup; body; _ } -> (
        (* The lane-0 guard survives on real lane widths; on the CPU pipelines' width-1 lane the
           simplifier folds the vacuous [if lane == 0], leaving the bare transfer nest. *)
        let guarded =
          match body with
          | If { cond = cond, _; body } when is_lane0_guard lane cond -> Some body
          | body when to_ = 0 -> Some body
          | _ -> None
        in
        let rec descend loops guards = function
          | For_loop { index; from_ = 0; to_; axis = Serial; body; _ } ->
              descend ((index, to_ + 1) :: loops) guards body
          | If { cond = cond, _; body } -> descend loops (cond :: guards) body
          | Set { tn; idcs; llsc = Get (frag, _); _ } when Tn.equal tn target && is_fragment frag ->
              Some (List.rev loops, List.rev guards, idcs)
          | _ -> None
        in
        match Option.bind guarded ~f:(descend [] []) with
        | Some (loops, guards, st_idcs) -> Some (lane, to_ + 1, loops, guards, st_idcs)
        | None -> None)
    | _ -> None
  in
  (* Which axis of the store-back a range guard trims: the pad masks compare the site's own index
     term for one axis against that axis's unpadded dimension ([mask_cond] in
     [contract_tensorized_accumulator] builds exactly this form). Guards in any other shape are not
     understood and keep the fusion rejected. *)
  let guard_axis ~(st_idcs : Indexing.axis_index array) cond : int option =
    match cond with
    | Binop (Ops.Cmplt, (Embed_index idx, _), (Constant bound, _)) ->
        Array.findi st_idcs ~f:(fun ax site_idx ->
            Indexing.equal_axis_index idx site_idx && Float.equal bound (Float.of_int dims.(ax)))
        |> Option.map ~f:fst
    | _ -> None
  in
  (* --- Site 1b: the whole-K [Tile_mma] writing [target] directly (the unstaged tensorized
     pipelines, gh-ocannl-521): there is no store-back statement — the intrinsic block itself
     completes [target]'s m x n tile — so the epilogue becomes a sibling lane-0 nest over the tile
     at the accumulator's base indices, exactly the shape site 1 appends after a fragment
     store-back. --- *)
  let match_wholek = function
    | For_loop { index = lane; from_ = 0; to_; axis = Workgroup; body; _ } -> (
        match body with
        | Tile_mma { d = d, d_base; m; n; lane = l; _ }
          when Tn.equal d target && Indexing.equal_symbol l lane ->
            Some (lane, to_ + 1, d_base, m, n)
        | _ -> None)
    | _ -> None
  in
  let add_symbol idx s =
    match terms_of_index idx with
    | Some (terms, offset) -> normalize_affine ~terms:((1, s) :: terms) ~offset
    | None -> fail "the accumulator's base indices must be affine"
  in
  let fused = ref false in
  let rec fuse_at_fragment env llc =
    if !fused then llc
    else
      match llc with
      | Seq (a, b) ->
          let a' = fuse_at_fragment env a in
          Seq (a', fuse_at_fragment env b)
      | For_loop fc -> (
          match match_storeback (For_loop fc) with
          | Some (lane, width, loops, guards, st_idcs) ->
              let env' = loops @ env in
              (* Exactly-once discipline: an enclosing loop that does not index the store-back
                 re-executes the site per iteration with partial accumulations — the relocated tail
                 must not run there (Codex P1 on the gh-ocannl-493 re-land). Since
                 [contract_tensorized_accumulator] contracts across the whole qualifying chain
                 (gh-ocannl-501), a surplus loop here means a hand-built schedule left the
                 store-back genuinely partial; the check stays as the safety net. *)
              List.iter env' ~f:(fun (s, e) ->
                  if e > 1 && not (Array.exists st_idcs ~f:(idx_mentions s)) then
                    fail
                      ("the fragment store-back executes once per iteration of enclosing loop "
                     ^ Indexing.symbol_ident s
                     ^ " which does not index it — the tail would read partial accumulations"));
              let guarded = Array.create ~len:rank false in
              List.iter guards ~f:(fun cond ->
                  match guard_axis ~st_idcs cond with
                  | Some ax -> guarded.(ax) <- true
                  | None ->
                      fail
                        "the fragment store-back carries a guard that is not a per-axis range mask \
                         of the output");
              if not (covers_bijectively ~guarded ~env:env' st_idcs) then
                fail "the fragment store-back tiles do not cover the output space bijectively";
              fused := true;
              let tail_leaf =
                (* Re-impose the store-back's own range guards: the relocated tail writes exactly
                   the valid cells the (padded) store-back writes. *)
                List.fold guards ~init:(subst_tail ~site_idcs:st_idcs) ~f:(fun body cond ->
                    If { cond = (cond, iprec); body })
              in
              let body =
                List.fold_right loops ~init:tail_leaf ~f:(fun (s, e) body ->
                    For_loop { index = s; from_ = 0; to_ = e - 1; body; axis = Serial })
              in
              Seq (For_loop fc, lane0 ~lane ~width body)
          | None -> (
              match match_wholek (For_loop fc) with
              | Some (lane, width, d_base, m, n) ->
                  if Array.length d_base <> rank || rank < 2 then
                    fail "the whole-K Tile_mma accumulator's rank does not match the output";
                  (* Exactly-once discipline, as for the fragment store-back: a surplus enclosing
                     loop would re-run the intrinsic block (and hence the tail) per iteration. *)
                  List.iter env ~f:(fun (s, e) ->
                      if e > 1 && not (Array.exists d_base ~f:(idx_mentions s)) then
                        fail
                          ("the Tile_mma block executes once per iteration of enclosing loop "
                         ^ Indexing.symbol_ident s
                         ^ " which does not index it — the tail would read partial accumulations"));
                  let fi = Indexing.get_symbol () and fj = Indexing.get_symbol () in
                  let site_idcs = Array.copy d_base in
                  site_idcs.(rank - 2) <- add_symbol site_idcs.(rank - 2) fi;
                  site_idcs.(rank - 1) <- add_symbol site_idcs.(rank - 1) fj;
                  let env' = ((fi, m) :: (fj, n) :: env : (Indexing.symbol * int) list) in
                  if not (covers_bijectively ~env:env' site_idcs) then
                    fail "the Tile_mma tiles do not cover the output space bijectively";
                  fused := true;
                  let body =
                    For_loop
                      {
                        index = fi;
                        from_ = 0;
                        to_ = m - 1;
                        axis = Serial;
                        body =
                          For_loop
                            {
                              index = fj;
                              from_ = 0;
                              to_ = n - 1;
                              axis = Serial;
                              body = subst_tail ~site_idcs;
                            };
                      }
                  in
                  Seq (For_loop fc, lane0 ~lane ~width body)
              | None -> (
                  match fc.axis with
                  | Workgroup | Workgroup_reduce -> For_loop fc
                  | _ when fc.from_ = 0 ->
                      For_loop
                        { fc with body = fuse_at_fragment ((fc.index, fc.to_ + 1) :: env) fc.body }
                  | _ -> For_loop fc)))
      | other -> other
  in
  let red_stmt' = fuse_at_fragment [] red_stmt in
  let red_stmt' =
    if !fused then red_stmt'
    else begin
      (* --- Sites 2 and 3: the unique plain [Set] writing [target] — the [Privatize] tile
         store-back (all path loops appear in the write indices: fuse per-element after the store)
         or the direct accumulation nest (the outermost path loop absent from the write indices is
         the serial reduction loop: fuse right after it, re-binding the inner output loops). --- *)
      let rec find_write path llc : (floop list * Indexing.axis_index array) option =
        match llc with
        | Set { tn; idcs; _ } when Tn.equal tn target -> Some (List.rev path, idcs)
        | (Set_dynamic { tn; _ } | Set_from_vec { tn; _ }) when Tn.equal tn target ->
            fail "dynamic or vector writes of the reduction output are unsupported"
        | Tile_mma { d = d, _; _ } when Tn.equal d target ->
            fail
              "the accumulator is a whole-K Tile_mma target; stage the reduction (split K) so it \
               is contracted to a fragment first"
        | Seq (a, b) -> (
            match (find_write path a, find_write path b) with
            | Some _, Some _ -> fail "multiple write sites of the reduction output"
            | (Some _ as res), None | None, (Some _ as res) -> res
            | None, None -> None)
        | For_loop { index; from_; to_; body; axis } ->
            find_write ({ index; from_; to_; body = Noop; axis } :: path) body
        | If { body; _ } ->
            if writes_tn target body then
              fail "guarded writes of the reduction output are unsupported"
            else None
        | _ -> None
      in
      match find_write [] red_stmt with
      | None -> fail ("no plain write site of " ^ Tn.debug_name target ^ " found")
      | Some (path, w_idcs) -> (
          let needed s = Array.exists w_idcs ~f:(idx_mentions s) in
          let rec split_path above = function
            | [] -> `Leaf (List.rev above)
            | (fc : floop) :: below when needed fc.index ->
                (match fc.axis with
                | Serial | Grid -> ()
                | _ ->
                    fail
                      ("output loop " ^ Indexing.symbol_ident fc.index
                     ^ " on the write path must be Serial or Grid"));
                if fc.from_ <> 0 then fail "write-path loops must start at 0";
                split_path (fc :: above) below
            | (fc : floop) :: below ->
                if not (equal_axis_type fc.axis Serial) then
                  fail ("reduction loop " ^ Indexing.symbol_ident fc.index ^ " must be Serial");
                `After (List.rev above, fc, below)
          in
          let env_of loops =
            List.map loops ~f:(fun (fc : floop) -> (fc.index, fc.to_ - fc.from_ + 1))
          in
          match split_path [] path with
          | `Leaf above ->
              if not (covers_bijectively ~env:(env_of above) w_idcs) then
                fail "the store-back writes do not cover the output space bijectively";
              let epilogue = subst_tail ~site_idcs:w_idcs in
              let seen = ref false in
              let rec at_leaf llc =
                match llc with
                | Set { tn; _ } when Tn.equal tn target && not !seen ->
                    seen := true;
                    Seq (llc, epilogue)
                | Seq (a, b) -> Seq (at_leaf a, at_leaf b)
                | For_loop fc -> For_loop { fc with body = at_leaf fc.body }
                | If ({ body; _ } as i) -> If { i with body = at_leaf body }
                | other -> other
              in
              let res = at_leaf red_stmt in
              assert !seen;
              res
          | `After (above, red_loop, below) ->
              let rebuild = List.filter below ~f:(fun (fc : floop) -> needed fc.index) in
              List.iter rebuild ~f:(fun (fc : floop) ->
                  if not (equal_axis_type fc.axis Serial) then
                    fail
                      ("inner output loop " ^ Indexing.symbol_ident fc.index
                     ^ " below the reduction loop must be Serial");
                  if fc.from_ <> 0 then fail "write-path loops must start at 0");
              let env = env_of above @ env_of rebuild in
              if not (covers_bijectively ~env w_idcs) then
                fail "the accumulation writes do not cover the output space bijectively";
              let epilogue =
                List.fold_right rebuild ~init:(subst_tail ~site_idcs:w_idcs)
                  ~f:(fun (fc : floop) body ->
                    For_loop { index = fc.index; from_ = 0; to_ = fc.to_; body; axis = Serial })
              in
              let seen = ref false in
              let rec after_loop llc =
                match llc with
                | For_loop fc
                  when Indexing.equal_symbol fc.index red_loop.index
                       && writes_tn target fc.body && not !seen ->
                    seen := true;
                    Seq (For_loop fc, epilogue)
                | Seq (a, b) -> Seq (after_loop a, after_loop b)
                | For_loop fc -> For_loop { fc with body = after_loop fc.body }
                | If ({ body; _ } as i) -> If { i with body = after_loop body }
                | other -> other
              in
              let res = after_loop red_stmt in
              assert !seen;
              res)
    end
  in
  let llc =
    unflat_lines
      (List.filter_mapi stmts ~f:(fun i s ->
           if i = t_idx then None else if i = r then Some red_stmt' else Some s))
  in
  let opt = { opt with llc } in
  if not shared then opt
  else if not !fused then
    fail
      "shared accumulator placement requires the fragment store-back site (apply after Tensorize's \
       contraction)"
  else
    (* GPU quality knob: the fused tail is often [target]'s last consumer, so placement makes
       [target] routine-local — a per-thread array the fragment hooks cannot [simdgroup_load] from.
       Place it in workgroup-shared memory instead (like [Stage]'s shared tiles), so the intrinsic
       fragment path fires against threadgroup memory. Nodes already settled on-device are left
       alone (device pointers are loadable as-is). CPU backends reject shared placement, so CPU
       schedules must not set [shared]. *)
    match Tn.Placements.get opt.optimize_ctx.placements target with
    | Some (Tn.On_device, _) -> opt
    | _ ->
        Tn.Placements.update opt.optimize_ctx.placements target Tn.Local 486;
        { opt with workgroup_shared = Set.add opt.workgroup_shared target }

let fuse_epilogue_witness ~target (opt : Low_level.optimized) : string option =
  try
    ignore (apply_fuse_epilogue ~target ~shared:false opt : Low_level.optimized);
    None
  with Invalid_argument msg -> Some msg

let can_fuse_epilogue ~target opt = Option.is_none (fuse_epilogue_witness ~target opt)

let apply_opt_op (opt : Low_level.optimized) (op : optop) : Low_level.optimized =
  match op with
  | Stage
      {
        source;
        tile_loops;
        shared;
        cooperative;
        hoisted;
        swizzle;
        pad_stride;
        pipeline_depth;
        tile_prec;
      } ->
      apply_stage ~source ~tile_loops ~shared ~cooperative ~hoisted ~swizzle ~pad_stride
        ~pipeline_depth ~tile_prec opt
  | Privatize { target; over } -> apply_privatize ~target ~over opt
  | Tensorize _ -> apply_tensorize op opt
  | Fuse_epilogue { target; shared } -> apply_fuse_epilogue ~target ~shared opt
  | Split_reduce { axis; target; num_blocks; block_index; inner_index; combine_indices } -> (
      try
        apply_split_reduce ~axis ~target ~num_blocks ~block_index ~inner_index ~combine_indices opt
      with Split_reduce_inner_cell syms ->
        invalid_arg
          ("Schedule.Split_reduce: the accumulation cell mentions "
          ^ String.concat ~sep:", " (List.map syms ~f:Indexing.symbol_ident)
          ^ ", which is not bound by a loop enclosing the reduction loop in this statement — Swap \
             it outside " ^ Indexing.symbol_ident axis ^ " first"))
  | (Split _ | Swap _ | Retype _ | Unroll _ | Partition _ | Pad _ | Expand_zero _) as op ->
      { opt with llc = apply_op opt.Low_level.llc op }

(* gh-ocannl-537: the recognizer's answer to "which loops would have to enclose the reduction for
   this site to be splittable". Probed hermetically like [op_legality]'s [Split_reduce] arm — the
   recognition is a pure function of the code, and the discipline check precedes any minting, so the
   copy is only belt-and-braces. Empty for every other outcome (legal, or rejected for a cause an
   interchange cannot remove), so a caller can treat a non-empty answer as "this exact obstruction,
   and nothing else, stands in the way". *)
let split_reduce_hoist (opt : Low_level.optimized) (op : optop) : Indexing.symbol list =
  match op with
  | Split_reduce { axis; target; num_blocks; block_index; inner_index; combine_indices } -> (
      let hermetic =
        {
          opt with
          Low_level.traced_store = Hashtbl.copy opt.Low_level.traced_store;
          optimize_ctx = Low_level.copy_optimize_ctx opt.Low_level.optimize_ctx;
        }
      in
      match
        apply_split_reduce ~axis ~target ~num_blocks ~block_index ~inner_index ~combine_indices
          hermetic
      with
      | (_ : Low_level.optimized) -> []
      | exception Split_reduce_inner_cell syms -> syms
      | exception Invalid_argument _ -> [])
  | _ -> []

(* gh-ocannl-487 (Codex P2 on PR #303), widened to depth 1 by gh-ocannl-567: "[Tile_mma] is a
   barrier" — the intrinsic's emission contract ends its block with a workgroup barrier on every
   barrier-capable backend and in every rendering form ({!ends_with_barrier} above). So once
   Tensorize has replaced a staged anchor loop's compute with a Tile_mma, the staging composition's
   own trailing barriers render as consecutive barriers with no work between, and are dropped:

   - the barrier a staged anchor's body ends with, against the intrinsic's own trailing bracket
   (depth 1: the phase closer the load-phase grouping left behind); - a barrier immediately after
   such a loop, against its last iteration's bracket; - a PIPELINED rotor body's LEADING barrier,
   against the previous iteration's trailing bracket — loop-carried, hence still gated on the rotor:
   the first iteration is covered by the bracket that precedes the intrinsic block (per statement,
   or once on the fragment scope wrapping the loop), which is what separates the prologue copies
   from the first compute.

   The dual — eliding a barrier against a FOLLOWING Tile_mma's leading bracket — is NOT sound and is
   not done: the fragment-update form has no per-iteration leading bracket, so at depth 1 the phase
   barrier between a k-block's loads and its compute is load-bearing and stays. Conservative
   otherwise: a scalar (untensorized) staged compute keeps all its barriers. Runs at {!apply}'s
   finalization, after Tensorize (Stage precedes Tensorize by Stage's own precondition, so the
   schedule half cannot see the Tile_mma at splice time). *)
let elide_staged_barriers (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  if Set.is_empty opt.workgroup_shared then opt
  else
    let rotors =
      Map.data opt.pipelined
      |> List.map ~f:(fun { pt_rotor; _ } -> pt_rotor)
      |> List.dedup_and_sort ~compare:Indexing.Symbol.compare
    in
    let is_rotor s = List.mem rotors s ~equal:Indexing.equal_symbol in
    (* A staged anchor: a loop whose subtree writes a staged tile — the load phase (depth 1) or the
       prefetch (pipelined). Over-inclusive (an enclosing loop qualifies too), which is harmless:
       every rule below is a local barrier-adjacency redundancy, sound wherever it fires. *)
    let staged_anchor body =
      not (Set.is_empty (Set.inter (written_nodes body) opt.workgroup_shared))
    in
    let rec stmt (llc : Low_level.t) : Low_level.t =
      match llc with
      | For_loop fc when is_rotor fc.index || staged_anchor fc.body ->
          let body_stmts = List.map (flat_lines [ fc.body ]) ~f:stmt in
          (* The loop is cyclic: what precedes the body's first statement is the previous
             iteration's last one — usable only for a rotor (see the header). *)
          let after_previous_iteration =
            is_rotor fc.index && fc.to_ >= fc.from_
            &&
            match body_stmts with
            | Workgroup_barrier :: rest -> ends_with_barrier (unflat_lines rest)
            | _ -> false
          in
          let rec drop prev_ends = function
            | [] -> []
            | Workgroup_barrier :: tl when prev_ends -> drop prev_ends tl
            | hd :: tl -> hd :: drop (ends_with_barrier hd) tl
          in
          For_loop { fc with body = unflat_lines (drop after_previous_iteration body_stmts) }
      | For_loop fc -> For_loop { fc with body = seq fc.body }
      | If { cond; body } -> If { cond; body = seq body }
      | other -> other
    and seq (llc : Low_level.t) : Low_level.t =
      let rec drop = function
        | (For_loop { index; from_; to_; body; _ } as lp) :: Workgroup_barrier :: tl
          when (is_rotor index || staged_anchor body) && to_ >= from_ && ends_with_barrier body ->
            lp :: drop tl
        | hd :: tl -> hd :: drop tl
        | [] -> []
      in
      unflat_lines (drop (List.map (flat_lines [ llc ]) ~f:stmt))
    in
    { opt with llc = seq opt.llc }

let apply ?(static_indices = []) (sched : schedule) (opt : Low_level.optimized) :
    Low_level.optimized =
  if List.is_empty sched then opt
  else
    let opt = List.fold sched ~init:opt ~f:apply_opt_op in
    let opt = elide_staged_barriers opt in
    let llc = opt.Low_level.llc in
    (* Transforms fold their own guards (schedule-ir-optops §2): the pipeline's simplify already
       ran, so re-run it here; and when a transform duplicated code, re-run CSE + hoisting too. *)
    let llc = Low_level.simplify_llc static_indices llc in
    let llc =
      if
        List.exists sched ~f:(function
          | Unroll { materialize = true; _ } | Partition _ -> true
          | _ -> false)
      then Low_level.hoist_cross_statement_cse @@ Low_level.eliminate_common_subexpressions llc
      else llc
    in
    { opt with llc }

let apply_classified ?static_indices sched opt =
  match apply ?static_indices sched opt with
  | result -> result
  | exception Invalid_argument detail ->
      raise
        (Schedule_outcome.Cause_at
           ( Schedule_outcome.Transform,
             Schedule_outcome.Illegal_schedule { check = "Schedule.apply"; detail } ))

(** {2 The op-legality oracle}

    gh-494 waypoint 3: schedule ops consult the affine engine instead of every transform (and every
    candidate compile) embedding its own analysis. A schedule op is a thread-pairing transform over
    the routine's access relations, so its obligation is a query: does pairing the op's loop
    symbol(s) as thread identity leave every write-involving access pair of every written node
    [Disjoint] or [Same_thread]?

    Verdicts are three-valued and both proven directions are sound: [Op_legal] means the queries
    prove the annotation race-free; [Op_illegal] means a definite violation is proven (e.g. a
    materialized node's unguarded write provably independent of the retyped axis — every iteration
    rewrites the same cells); everything else is [Op_unknown] — the op may still be valid under
    semantics the queries do not model (per-thread copies of scratch, renderer fallbacks), so
    consumers must treat [Op_unknown] as "compile and see", never as a rejection. The autotuner
    prunes [Op_illegal] menu proposals before compiling them. *)

type op_verdict = Op_legal | Op_illegal of string | Op_unknown of string
[@@deriving sexp_of, equal]

(* Worst-of combination: Illegal > Unknown > Legal. *)
let combine_verdicts v1 v2 =
  match (v1, v2) with
  | (Op_illegal _ as v), _ | _, (Op_illegal _ as v) -> v
  | (Op_unknown _ as v), _ | _, (Op_unknown _ as v) -> v
  | Op_legal, Op_legal -> Op_legal

let mentions_axis axis (idx : Indexing.axis_index) =
  match idx with
  | Indexing.Iterator s -> Indexing.equal_symbol s axis
  | Indexing.Affine { symbols; _ } ->
      List.exists (Indexing.coalesce_affine_terms symbols) ~f:(fun (_, s) ->
          Indexing.equal_symbol s axis)
  | Indexing.Concat syms -> List.exists syms ~f:(Indexing.equal_symbol axis)
  | Indexing.Fixed_idx _ | Indexing.Sub_axis -> false

(* gh-ocannl-508: derive the static affine breakpoints of the [axis] loop from the guards already
   present in its body — statement [If] conditions and scalar [Where] conditions (the virtualizer's
   per-component range guards of an inlined concatenation, [Split]'s remainder guard, symbolic
   extent guards with the extent already substituted, gh-504's clamped-window range guards). A
   comparison whose two sides differ by [k*axis + off] with everything else constant flips truth
   value at exactly one point of the axis range; partitioning at the collected points makes every
   such guard interval-decided within each segment, so [apply]'s trailing simplify erases them.
   Non-axis symbols bound by loops inside the [axis] loop (e.g. the window symbol of a clamped
   window guard, gh-504) are bounded by their loop ranges: the comparison then reads [k*axis + off]
   with [off] in an interval, giving an always-true / mixed / always-false trichotomy of the axis
   range — both transition points are recorded, so the mixed (boundary) segments are exactly
   delimited and the decided segments fold their guards. *)
let partition_breakpoints ~axis (llc : Low_level.t) : int list =
  let open Low_level in
  (* Every loop binding [axis] contributes, because [Partition] rewrites every one of them: the
     copies a materializing [Unroll] left behind carry the same guard with different constants
     substituted for the unrolled index, so their flip points differ and the union is what makes
     each copy's guard interval-decided. The ranges of the loops enclosing each occurrence come from
     the same walk that locates it: an inner axis's guards may mention enclosing-loop symbols. *)
  let breakpoints_of ~enclosing_ranges ~from_ ~to_ body =
    let points = ref [] in
    (* Affine view [Some (k, off_lo, off_hi)] of a comparison operand as [k*axis + off] with [off]
       ranging over [off_lo, off_hi]: non-axis symbols with a known enclosing-loop range (from
       [ranges]) contribute their extremes to the offset interval. [None] when the operand mentions
       a symbol of unknown range or is not integer-affine. *)
    let affine_view ~ranges (sc : scalar_t) : (int * int * int) option =
      let of_terms symbols offset =
        let symbols = Indexing.coalesce_affine_terms symbols in
        List.fold symbols
          ~init:(Some (0, offset, offset))
          ~f:(fun acc (c, s) ->
            match acc with
            | None -> None
            | Some (k, lo, hi) -> (
                if Indexing.equal_symbol s axis then Some (k + c, lo, hi)
                else
                  match Map.find ranges s with
                  | Some (s_lo, s_hi) ->
                      if c >= 0 then Some (k, lo + (c * s_lo), hi + (c * s_hi))
                      else Some (k, lo + (c * s_hi), hi + (c * s_lo))
                  | None -> None))
      in
      match sc with
      | Embed_index (Indexing.Iterator s) when Indexing.equal_symbol s axis -> Some (1, 0, 0)
      | Embed_index (Indexing.Iterator s) -> (
          match Map.find ranges s with Some (lo, hi) -> Some (0, lo, hi) | None -> None)
      | Embed_index (Indexing.Fixed_idx i) -> Some (0, i, i)
      | Embed_index (Indexing.Affine { symbols; offset }) -> of_terms symbols offset
      | Constant c when Float.is_integer c -> Some (0, Float.to_int c, Float.to_int c)
      | _ -> None
    in
    (* Division rounding towards -inf resp. +inf; [b > 0]. *)
    let fdiv a b = if a >= 0 then a / b else -((-a + b - 1) / b) in
    let cdiv a b = if a >= 0 then (a + b - 1) / b else -(-a / b) in
    (* Transition points of [k*axis + off < 0] with [off] in [off_lo, off_hi]. *)
    let strict_lt_points ~k ~off_lo ~off_hi =
      if k > 0 then points := cdiv (-off_hi) k :: cdiv (-off_lo) k :: !points
      else if k < 0 then points := (fdiv off_hi (-k) + 1) :: (fdiv off_lo (-k) + 1) :: !points
    in
    let rec cond ~ranges (sc : scalar_t) =
      match sc with
      | Binop ((Ops.And | Ops.Or), (a, _), (b, _)) ->
          cond ~ranges a;
          cond ~ranges b
      (* [a < b] iff [k*axis + off < 0], [off] in [off_lo, off_hi]. Always-true while [k*axis +
         off_hi < 0], always-false once [k*axis + off_lo >= 0]: record both transition points (equal
         when the offset is a single value). *)
      | Binop (Ops.Cmplt, (a, _), (b, _)) -> (
          match Option.both (affine_view ~ranges a) (affine_view ~ranges b) with
          | Some ((ka, la, ha), (kb, lb, hb)) ->
              strict_lt_points ~k:(ka - kb) ~off_lo:(la - hb) ~off_hi:(ha - lb)
          | None -> ())
      (* Everything here is integer-affine, so [a <= b] iff [(a - b) - 1 < 0]: the same transition
         points as the strict case with the offset interval shifted down by one. *)
      | Binop (Ops.Cmple, (a, _), (b, _)) -> (
          match Option.both (affine_view ~ranges a) (affine_view ~ranges b) with
          | Some ((ka, la, ha), (kb, lb, hb)) ->
              strict_lt_points ~k:(ka - kb) ~off_lo:(la - hb - 1) ~off_hi:(ha - lb - 1)
          | None -> ())
      | Binop ((Ops.Cmpeq | Ops.Cmpne), (a, _), (b, _)) -> (
          match Option.both (affine_view ~ranges a) (affine_view ~ranges b) with
          | Some ((ka, la, ha), (kb, lb, hb)) ->
              let k = ka - kb and off_lo = la - hb and off_hi = ha - lb in
              (* Equality is possible while [k*axis] meets [-off_hi, -off_lo]: both edges of the
                 possibly-equal axis range (a single point [q, q+1] when the offset is a single
                 value that [k] divides — no points when it does not). *)
              if k <> 0 then
                let neg_lo, neg_hi = (-off_hi, -off_lo) in
                let q_lo = if k > 0 then cdiv neg_lo k else cdiv neg_hi k
                and q_hi = if k > 0 then fdiv neg_hi k else fdiv neg_lo k in
                if q_lo <= q_hi then points := q_lo :: (q_hi + 1) :: !points
          | None -> ())
      | _ -> ()
    in
    let rec go ~ranges (llc : t) =
      match llc with
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
        ->
          ()
      | Seq (a, b) ->
          go ~ranges a;
          go ~ranges b
      | For_loop { index; from_; to_; body; _ } ->
          go ~ranges:(Map.set ranges ~key:index ~data:(from_, to_)) body
      | Scan_loop { index; from_; to_; carried; body; _ } ->
          List.iter carried ~f:(fun c -> scan_scalar ~ranges c.init);
          go ~ranges:(Map.set ranges ~key:index ~data:(from_, to_)) body
      | If { cond = c, _; body } ->
          cond ~ranges c;
          go ~ranges body
      | Tile_mma { fallback; _ } -> go ~ranges fallback
      | Set { llsc; _ } | Set_local (_, llsc) -> scan_scalar ~ranges llsc
      | Set_dynamic { dyn_value = v, _; llsc; _ } ->
          scan_scalar ~ranges v;
          scan_scalar ~ranges llsc
      | Set_from_vec { arg = a, _; _ } -> scan_scalar ~ranges a
    and scan_scalar ~ranges (sc : scalar_t) =
      match sc with
      | Ternop (Ops.Where, (c, _), (t, _), (e, _)) ->
          cond ~ranges c;
          scan_scalar ~ranges t;
          scan_scalar ~ranges e
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          scan_scalar ~ranges a;
          scan_scalar ~ranges b;
          scan_scalar ~ranges c
      | Binop (_, (a, _), (b, _)) ->
          scan_scalar ~ranges a;
          scan_scalar ~ranges b
      | Unop (_, (a, _)) -> scan_scalar ~ranges a
      | Local_scope { body; _ } -> go ~ranges body
      | Get_dynamic { dyn_value = v, _; _ } -> scan_scalar ~ranges v
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
    in
    go ~ranges:enclosing_ranges body;
    List.filter !points ~f:(fun b -> from_ < b && b <= to_)
  in
  let occurrences =
    find_loops_env axis llc
      ~init:(Map.empty (module Indexing.Symbol))
      ~f:(fun env fc -> Map.set env ~key:fc.index ~data:(fc.from_, fc.to_))
  in
  if List.is_empty occurrences then
    invalid_arg
      ("Schedule.partition_breakpoints: no For_loop with index " ^ Indexing.symbol_ident axis);
  List.concat_map occurrences ~f:(function
    | For_loop { from_; to_; body; _ }, enclosing_ranges ->
        breakpoints_of ~enclosing_ranges ~from_ ~to_ body
    | _ -> assert false (* [find_loops_env] only returns [For_loop]s. *))
  |> List.dedup_and_sort ~compare:Int.compare

let acc_interpretable (a : _ Affine.access) =
  (not a.Affine.a_dynamic) && (not a.a_whole) && (not a.a_vec_last)
  && not
       (Array.exists a.a_map ~f:(function
         | Indexing.Sub_axis | Indexing.Concat _ -> true
         | _ -> false))

(* Node accesses outside a given subtree: uid -> written-outside flag. Hardware annotations
   interleave sibling statements' threads with no grid-wide synchronization, so any node shared
   across the subtree boundary (with a write on either side) carries a cross-nest alignment
   obligation the single-op oracle does not model. *)
let accesses_outside (llc : Low_level.t) ~(skip : Low_level.t) : (int, bool) Hashtbl.t =
  let open Low_level in
  let outside = Hashtbl.create (module Int) in
  let note ~write (tn : Tn.t) =
    Hashtbl.update outside tn.Tn.uid ~f:(function None -> write | Some w -> w || write)
  in
  let rec stmt (llc : Low_level.t) =
    if phys_equal llc skip then ()
    else
      match llc with
      | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> ()
      | Seq (a, b) ->
          stmt a;
          stmt b
      | For_loop { body; _ } -> stmt body
      | Scan_loop { carried; body; _ } ->
          List.iter carried ~f:(fun c -> scalar c.init);
          stmt body
      | If { cond = c, _; body } ->
          scalar c;
          stmt body
      | Zero_out tn -> note ~write:true tn
      | Set { tn; llsc; _ } ->
          note ~write:true tn;
          scalar llsc
      | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
          note ~write:true tn;
          scalar v;
          scalar llsc
      | Set_from_vec { tn; arg = a, _; _ } ->
          note ~write:true tn;
          scalar a
      | Set_local (_, llsc) -> scalar llsc
      | Tile_mma { fallback; _ } -> stmt fallback
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> stmt body
    | Get (tn, _) -> note ~write:false tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        note ~write:false tn;
        scalar v
    | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
  in
  stmt llc;
  outside

(* Iteration independence of the loop bound by [pairs]'s symbols (all paired with themselves:
   same-nest thread identity). [licensed w x] exempts a conflicting pair from the obligation
   (reduction reassociation under an explicit license). With [cross_nest] (hardware annotations,
   which interleave sibling statements' threads), a node shared across the loop's statement boundary
   with a write on either side downgrades the proof to [Op_unknown]: such pairs need the
   aligned-mapping analysis of the default annotator, which the single-op oracle does not model.
   Proven intra-nest violations stay [Op_illegal] regardless. *)
let loops_independent (opt : Low_level.optimized) ~(syms : Indexing.symbol list) ~cross_nest
    ~licensed : op_verdict =
  let open Low_level in
  let plc = opt.optimize_ctx.placements in
  let verdict_of loop =
    let accs = Low_level.affine_accesses loop in
    let env = Low_level.loop_bounds loop in
    let range s = List.Assoc.find env s ~equal:Indexing.equal_symbol in
    let dup s = List.Assoc.mem env s ~equal:Indexing.equal_symbol in
    let pairs = List.map syms ~f:(fun s -> (s, s)) in
    let extent s = match range s with Some (lo, hi) -> hi - lo + 1 | None -> 1 in
    let outside =
      if cross_nest then accesses_outside opt.llc ~skip:loop else Hashtbl.create (module Int)
    in
    let by_tn = Hashtbl.create (module Int) in
    List.iter accs ~f:(fun a -> Hashtbl.add_multi by_tn ~key:a.Affine.a_tn.Tn.uid ~data:a);
    Hashtbl.fold by_tn ~init:Op_legal ~f:(fun ~key:uid ~data:accs v ->
        let tn = (List.hd_exn accs).Affine.a_tn in
        let writes = List.filter accs ~f:(fun a -> a.Affine.a_write) in
        let written_outside = Option.value (Hashtbl.find outside uid) ~default:false in
        if List.is_empty writes then
          if written_outside then
            combine_verdicts v
              (Op_unknown
                 (Tn.debug_name tn
                ^ ": written outside the parallelized statement (cross-nest alignment not modeled)"
                 ))
          else v
        else
          let is_mat = Tn.Placements.is_materialized_peek plc tn in
          (* Proven violation: an unguarded, interpretable write of a materialized node whose map
             provably avoids some parallel symbol of extent > 1 — every iteration of that loop
             rewrites the same cells of genuinely shared memory. *)
          let illegal =
            is_mat
            && List.find writes ~f:(fun w ->
                   acc_interpretable w && (not w.a_guarded)
                   && (not (licensed w w))
                   && List.exists syms ~f:(fun s ->
                       extent s > 1 && not (Array.exists w.a_map ~f:(mentions_axis s))))
               |> Option.is_some
          in
          if illegal then
            combine_verdicts v
              (Op_illegal
                 (Tn.debug_name tn
                ^ ": a write provably independent of a parallelized loop rewrites the same cells \
                   on every iteration"))
          else if not (List.for_all accs ~f:acc_interpretable) then
            combine_verdicts v
              (Op_unknown (Tn.debug_name tn ^ ": statically unknown access under the loop"))
          else
            let node_v =
              List.fold writes ~init:Op_legal ~f:(fun nv w ->
                  List.fold accs ~init:nv ~f:(fun nv x ->
                      if licensed w x then nv
                      else
                        match
                          Affine.pair_conflict ~range ~dup_left:dup ~dup_right:dup ~pairs
                            ~left:w.Affine.a_map ~right:x.Affine.a_map
                        with
                        | Affine.Disjoint | Affine.Same_thread -> nv
                        | Affine.Cross_thread wit ->
                            combine_verdicts nv (Op_unknown (Tn.debug_name tn ^ ": " ^ wit))))
            in
            let node_v =
              if equal_op_verdict node_v Op_legal && Hashtbl.mem outside uid then
                Op_unknown
                  (Tn.debug_name tn
                 ^ ": also accessed outside the parallelized statement (cross-nest alignment not \
                    modeled)")
              else node_v
            in
            combine_verdicts v node_v)
  in
  (* The op annotates or rewrites EVERY loop binding the symbol (a materializing [Unroll] leaves one
     copy per step), so the obligation is the worst of their verdicts, not the first one's. *)
  match find_loops (List.hd_exn syms) opt.llc with
  | [] -> Op_unknown ("no loop binds " ^ Indexing.symbol_ident (List.hd_exn syms))
  | loops -> List.fold loops ~init:Op_legal ~f:(fun v loop -> combine_verdicts v (verdict_of loop))

(* [Stage] legality: apply the op on a hermetic copy — [apply_stage]'s precondition surface (source
   unwritten and statically read through one index vector, tile loops enclosing/occurring with
   positive coefficients, the shared-mode workgroup-slot coverage rule, the
   hoisted/cooperative/swizzle contracts) is deterministic on the code, so a raising probe proves
   the candidate compile's apply raises too: [Op_illegal] with no transcription drift. The minted
   tile and its (key-weak) [Host_inits] entry become unreachable with the discarded copy.

   On success, the op's implicit contract — the staged tile covers the reads it replaces within the
   staging scope — is the containment query: every remapped read of the fresh tile must be covered
   by the load nest's prior writes ([Affine.read_covered_before]; edge-guarded loads included,
   mirroring guards-taken analyses). The loads copy a source the routine never writes (checked by
   apply) through the same per-axis index decomposition the reads use, so a covering write holds
   exactly the source cell the original read fetched — cell coverage implies value correctness here.
   A covered non-shared (packing) stage only inserts a serial per-thread copy nest over a [Local]
   tile: [Op_legal]. Shared staging additionally relies on barrier placement and launch-geometry
   uniformity validated downstream by [Low_level.validate_parallel], and hoisted staging on the
   link-time host-side packing program — neither is modeled by the queries, so those report
   [Op_unknown] (never a rejection). *)
let stage_legality (opt : Low_level.optimized) (op : optop) : op_verdict =
  let shared, hoisted =
    match op with Stage { shared; hoisted; _ } -> (shared, hoisted) | _ -> assert false
  in
  let hermetic =
    {
      opt with
      Low_level.traced_store = Hashtbl.copy opt.Low_level.traced_store;
      optimize_ctx = Low_level.copy_optimize_ctx opt.Low_level.optimize_ctx;
    }
  in
  match apply_opt_op hermetic op with
  | exception Invalid_argument msg -> Op_illegal msg
  | staged -> (
      if hoisted then
        Op_unknown "hoisted packing's link-time host-side program is not modeled by the queries"
      else
        (* The fresh tile is the node apply added to the (copied) traced store. *)
        let tile =
          Hashtbl.keys staged.Low_level.traced_store
          |> List.find ~f:(fun tn -> not (Hashtbl.mem opt.Low_level.traced_store tn))
        in
        match tile with
        | None -> Op_unknown "staged tile not found in the probe (oracle limitation)"
        | Some tile -> (
            let accs = Low_level.affine_accesses staged.Low_level.llc in
            let tile_accs = List.filter accs ~f:(fun a -> Tn.equal a.Affine.a_tn tile) in
            let writes = List.filter tile_accs ~f:(fun a -> a.Affine.a_write) in
            let reads = List.filter tile_accs ~f:(fun a -> not a.Affine.a_write) in
            let uncovered =
              List.find_map reads ~f:(fun read ->
                  match Affine.read_covered_before ~read ~writes () with
                  | `Covered -> None
                  | `Unknown witness -> Some witness)
            in
            match uncovered with
            | Some witness ->
                Op_unknown
                  (Tn.debug_name tile ^ ": a staged read is not proven covered by the loads: "
                 ^ witness)
            | None ->
                if shared then
                  Op_unknown
                    "shared staging's barrier placement and launch-geometry uniformity are \
                     validated downstream (validate_parallel)"
                else Op_legal))

let op_legality (opt : Low_level.optimized) (op : optop) : op_verdict =
  let open Low_level in
  let hardware = function Grid | Workgroup -> true | _ -> false in
  let vectorized = function Vectorized -> true | _ -> false in
  (* Under the reassociation license (Vectorized retypes; Swap's accumulation contract), a
     read-modify-write's conflicts with itself and with its own same-cell read are the reduction
     dependence the license permits. Same-cell means the statement's OWN rhs read at the write's
     position ([Affine.same_statement], gh-561): an [If] condition's read no longer shares the
     guarded body's write path, and a read nested in a [Local_scope] body (which used to share a
     non-[Seq] statement's bare path) is not the rmw carrier either. *)
  let rmw_license (w : _ Affine.access) (x : _ Affine.access) =
    w.Affine.a_rmw
    && (phys_equal w x
       || (not x.Affine.a_write)
          && Affine.same_statement w.a_path x.Affine.a_path
          && [%equal: Indexing.axis_index array] w.a_map x.a_map)
  in
  let none _ _ = false in
  match op with
  | Retype { axis; ty } ->
      if hardware ty then loops_independent opt ~syms:[ axis ] ~cross_nest:true ~licensed:none
      else if vectorized ty then
        (* Vectorized (like Swap below) only reorders within the nest; sibling statements still
           execute in serial program order, so the intra-nest scope is the whole obligation. *)
        loops_independent opt ~syms:[ axis ] ~cross_nest:false ~licensed:rmw_license
      else if equal_axis_type ty Workgroup_reduce then
        Op_unknown "Workgroup_reduce is the tensorization pipeline's domain"
      else Op_legal
  | Split { axis; outer; inner; _ } ->
      if hardware outer || hardware inner then
        loops_independent opt ~syms:[ axis ] ~cross_nest:true ~licensed:none
      else if vectorized outer || vectorized inner then
        loops_independent opt ~syms:[ axis ] ~cross_nest:false ~licensed:rmw_license
      else Op_legal
  | Unroll _ -> Op_legal
  | Partition _ ->
      (* Pure index-set reindexing: the segments run in the original order over the same points,
         with no hardware annotation and no reordering. *)
      Op_legal
  | Pad _ ->
      (* The pad iterations are no-ops (every effectful leaf statement is guarded), so the padded
         loop runs the original iterations in the original order. *)
      Op_legal
  | Swap { outer; inner } -> (
      (* Interchange reorders iterations; the optop contract licenses it for the
         associative-commutative accumulation patterns lowering emits (the rmw self-pairs). Beyond
         that license, prove that no write-involving pair can touch a common cell across different
         (outer, inner) iterations at all — then any order computes the same values. *)
      (* Interchange rewrites every copy of the outer loop, so every one of them must be the
         perfect nest the swap assumes. *)
      match find_loops outer opt.llc with
      | [] -> Op_unknown ("no loop binds " ^ Indexing.symbol_ident outer)
      | loops ->
          if
            List.for_all loops ~f:(function
              | Low_level.For_loop { body = Low_level.For_loop { index; _ }; _ } ->
                  Indexing.equal_symbol index inner
              | _ -> false)
          then loops_independent opt ~syms:[ outer; inner ] ~cross_nest:false ~licensed:rmw_license
          else Op_unknown "loops are not perfectly nested")
  | Tensorize { i; j; k; lane; simd_width; tile } -> (
      (* Role-assignment validity first: the micro-kernel recognition in [tensorize_llc] is a pure
         function of the code (given the routine's zero-fringe tiles), so probing it decides exactly
         whether the candidate compile's apply would raise — a proven [Op_illegal] with no
         transcription drift. This is the valuable pruner: of the role permutations the autotuner
         proposes per serial triple, the ones assigning the reduction loop to [i]/[j] (or an output
         loop to [k]) fail the accumulator/operand index discipline here — the [a_rmw]-carrying
         write must be indexed [..., i, j] with the [k] role absent, i.e. [k] is the only loop
         carrying the reduction dependence. Pad-guarded micro-kernels additionally require
         zero-fringe staged operands (gh-ocannl-485); the masked contraction's own site precondition
         is checked at apply, not probed here. *)
      match
        tensorize_llc
          ~zero_fringe:(Set.mem opt.Low_level.zero_fringe)
          ~i ~j ~k ~lane ~simd_width ~tile opt.llc
      with
      | exception Invalid_argument msg -> Op_illegal msg
      | (_ : Low_level.t * pad_mask list) ->
          (* Structure proven; the legality dimension is the affine query: the [Tile_mma] block
             distributes the [i x j] tile across the fresh hardware lane loop (whatever fragment
             mapping the intrinsic picks) and reassociates the [k] reduction — the same license as
             [Vectorized] retypes, discharged by the accumulator's rmw self-pairs. Pairing [i]/[j]
             as thread identity must leave every write-involving pair Disjoint or Same_thread, and
             the lane is a hardware axis, so cross-statement node sharing downgrades to [Op_unknown]
             exactly as for hardware retypes. *)
          loops_independent opt ~syms:[ i; j ] ~cross_nest:true ~licensed:rmw_license)
  | Stage _ -> stage_legality opt op
  | Split_reduce _ -> (
      (* Probe hermetically like [Stage]: the recognition in [apply_split_reduce] is a pure function
         of the code, so a raising apply is a proven [Op_illegal]. On structural success the op is
         semantics-preserving under the reduction-reassociation license (the same license as [Swap]
         of accumulations): it mints no hardware annotation itself — the block loop's later
         annotation is race-free by construction (its index pins the partials row) and is checked as
         its own op — and the combine order is a fixed function of the schedule. *)
      let hermetic =
        {
          opt with
          Low_level.traced_store = Hashtbl.copy opt.Low_level.traced_store;
          optimize_ctx = Low_level.copy_optimize_ctx opt.Low_level.optimize_ctx;
        }
      in
      match apply_opt_op hermetic op with
      | exception Invalid_argument msg -> Op_illegal msg
      | (_ : Low_level.optimized) -> Op_legal)
  | Privatize _ | Expand_zero _ | Fuse_epilogue _ ->
      Op_unknown "not modeled by the oracle (the op's own preconditions apply)"

(** [schedule_legality opt sched]: per-op verdicts, each against the code with the preceding ops
    applied (on a hermetic copy — checking never mutates [opt]). Stops after a proven-illegal op or
    a failing application; an op that fails to apply reports [Op_illegal] with the exception. *)
let schedule_legality (opt : Low_level.optimized) (sched : schedule) : (optop * op_verdict) list =
  let opt =
    {
      opt with
      Low_level.traced_store = Hashtbl.copy opt.Low_level.traced_store;
      optimize_ctx = Low_level.copy_optimize_ctx opt.Low_level.optimize_ctx;
    }
  in
  let rec go opt acc = function
    | [] -> List.rev acc
    | op :: tl -> (
        let v = op_legality opt op in
        match v with
        | Op_illegal _ -> List.rev ((op, v) :: acc)
        | _ -> (
            match apply_opt_op opt op with
            | opt' -> go opt' ((op, v) :: acc) tl
            | exception exn ->
                List.rev ((op, Op_illegal ("apply failed: " ^ Exn.to_string exn)) :: acc)))
  in
  go opt [] sched

(** {2 The default GPU annotator}

    [default_gpu] computes a schedule that makes elementwise / outer-parallel kernels launch with
    real grid and workgroup dimensions (schedule-ir-optops §6). It is deliberately conservative:
    parallelizing is only proposed when the analysis below proves that annotating cannot introduce a
    race, otherwise the empty schedule is returned and the kernel runs 1×1 as before.

    Thread identity after annotation is the tuple of annotated-loop index values. The safety
    argument, per kernel:

    - A written node's every write vector contains each of its nest's chosen parallel symbols as a
      plain [Iterator] component, so equal components imply the same thread (injectivity) — and
      [Split]'s [factor*i_o + i_i] substitution preserves injectivity because [i_i < factor].
    - All accesses to a written node agree on every component that mentions a parallel symbol, so
      reads only ever hit the reading thread's own elements.
    - Cross-nest pairs over a written node (producer/consumer, WAW, WAR) are races once threads
      interleave — sibling nests execute with no global synchronization between them — unless the
      accesses are {e aligned}: each thread then touches only its own elements in both nests, and a
      thread's writes are visible to its own later reads by program order, so values match the
      serial execution. Nests transitively linked by such pairs form a dependency component;
      alignment of a component means: (a) every member's chain is trimmed to one common prefix with
      pointwise-equal extents — the annotator then emits identical geometry for all members, so a
      hardware thread maps to the same chain-index tuple in every member nest; and (b) per axis
      position of every write/access pair over a shared node, either neither side's component
      mentions its own nest's (trimmed) parallel symbols, or both are plain [Iterator]s of symbols
      at the same chain position. Statically-unknown ([Get_dynamic]) accesses never align. If no
      common trim depth >= 1 aligns every pair of the component, the analysis bails. Bare statements
      (outside every nest, executed unconditionally by every thread) have no parallel symbols, so a
      bare access to a nest-written node bails by the same rule. Non-materialized (routine-local)
      scratch is per-thread, hence exempt when its writes mention no parallel symbols (each thread
      writes its whole private copy); otherwise it needs alignment like a materialized node (each
      thread reads back exactly the slice of its private copy that it wrote).
    - Whole-node [Zero_out] of a materialized node, barriers, opaque [Staged_compilation],
      pre-existing hardware annotations, and materialized writes outside every nest all bail. *)

type access = {
  a_tn : Tn.t;
  a_idcs : Indexing.axis_index array;
  a_write : bool;
  a_dynamic : bool;  (** [Get_dynamic]: the effective index is not statically known. *)
  a_dyn_axis : int option;
      (** The data-dependent component of a dynamic access ([Set_dynamic]/[Get_dynamic]'s
          [dyn_axis]; [a_idcs] holds a placeholder there) — affine queries must treat that component
          as opaque ({!query_map}). *)
  a_vec : bool;
      (** [Set_from_vec]: [a_idcs] is the base of a length-run along the minor axis, not a single
          cell — affine queries must treat the last component as opaque ({!query_map}). *)
  a_val_syms : Indexing.symbol list;
      (** Writes only: loop symbols the written value depends on syntactically (index symbols of rhs
          reads, embedded indices, dynamic-index sub-expressions). Direct dependence only — a chain
          through another scratch node is not tracked. *)
}

exception Bail

(* Collects accesses of tensor nodes (not scalar scope-locals) in [llc]. Raises [Bail] on opaque or
   clearly unschedulable constructs. [depth] counts enclosing [Local_scope] bodies: materialized
   writes there are invisible to [validate_parallel]'s coverage check, so bail. *)
let scan_accesses plc ~local_syms (llc : Low_level.t) : access list =
  let open Low_level in
  let acc = ref [] in
  let add ~depth:_ ~write ~dynamic ?dyn_axis ?(vec = false) ?(val_syms = []) tn idcs =
    acc :=
      {
        a_tn = tn;
        a_idcs = idcs;
        a_write = write;
        a_dynamic = dynamic;
        a_dyn_axis = dyn_axis;
        a_vec = vec;
        a_val_syms = val_syms;
      }
      :: !acc
  in
  (* Symbols the value of a setter depends on, syntactically; scope-locals resolve through the
     whole-kernel [local_syms] (a local may be assigned in another top-level statement). *)
  let scalar_syms = Low_level.scalar_value_syms ~locals:local_syms in
  let rec code ~depth llc =
    match llc with
    | Noop | Comment _ | Declare_local _ -> ()
    | Staged_compilation _ -> raise Bail
    | Workgroup_barrier -> raise Bail
    (* Already-scheduled cooperative code: the default annotator leaves it alone. *)
    | Tile_mma _ -> raise Bail
    | Seq (a, b) ->
        code ~depth a;
        code ~depth b
    | For_loop { axis; body; _ } ->
        if not (equal_axis_type axis Serial) then raise Bail;
        code ~depth body
    | Scan_loop { carried; body; _ } ->
        (* gh-ocannl-696: the scan's own index is never a retype target (it is not a [For_loop]),
           and its body's accesses are registered like any serial loop's, so an ENCLOSING loop whose
           iterations the accesses prove independent keeps its hardware mapping -- the carried state
           is per-iteration scratch of that loop. *)
        List.iter carried ~f:(fun c -> scalar ~depth c.init);
        code ~depth body
    | Zero_out tn ->
        if Tn.Placements.is_materialized_peek plc tn then raise Bail
          (* Zeroing per-thread scratch is safe: each thread zeroes its own copy. *)
    | Set { tn; idcs; llsc; _ } ->
        if depth > 0 && Tn.Placements.is_materialized_peek plc tn then raise Bail;
        add ~depth ~write:true ~dynamic:false ~val_syms:(scalar_syms llsc) tn idcs;
        scalar ~depth llsc
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, _; llsc; _ } ->
        (* gh-466: the scatter's effective write index is not statically known. Registering it
           [~dynamic:true] makes the cross-nest alignment reject it, and the per-nest hazard
           analysis mask the dynamic component from the affine queries ([query_map]) — the
           deterministic no-atomics invariant: loops driving the dynamic index are never forced
           equal across threads, so they stay serial, while statically-pinning components (gh-484
           task 2: the per-block partials row of [Split_reduce], the embedding-dim column) may
           parallelize. *)
        if depth > 0 && Tn.Placements.is_materialized_peek plc tn then raise Bail;
        add ~depth ~write:true ~dynamic:true ~dyn_axis
          ~val_syms:(scalar_syms v @ scalar_syms llsc)
          tn idcs;
        scalar ~depth v;
        scalar ~depth llsc
    | Set_from_vec { tn; idcs; arg = a, _; _ } ->
        if depth > 0 && Tn.Placements.is_materialized_peek plc tn then raise Bail;
        add ~depth ~write:true ~dynamic:false ~vec:true ~val_syms:(scalar_syms a) tn idcs;
        scalar ~depth a
    | Set_local (_, llsc) -> scalar ~depth llsc
    | If { cond = c, _; body } ->
        scalar ~depth c;
        code ~depth body
  and scalar ~depth (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> code ~depth:(depth + 1) body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get (tn, idcs) -> add ~depth ~write:false ~dynamic:false tn idcs
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, _; _ } ->
        add ~depth ~write:false ~dynamic:true ~dyn_axis tn idcs;
        scalar ~depth v
    | Get_merge_buffer (_, _) -> () (* The merge buffer is a separate read-only input buffer. *)
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar ~depth a;
        scalar ~depth b;
        scalar ~depth c
    | Binop (_, (a, _), (b, _)) ->
        scalar ~depth a;
        scalar ~depth b
    | Unop (_, (a, _)) -> scalar ~depth a
  in
  code ~depth:0 llc;
  !acc

type nest_info = {
  n_loops : Low_level.t;  (** The nest statement itself ([For_loop] possibly under [If]). *)
  n_accesses : access list;
}

(* Top-level statements of the kernel: [For_loop] (possibly [If]-wrapped) statements become nests;
   everything else contributes to the "bare" pseudo-nest (executed unconditionally by every thread
   of the launch). *)
let split_nests plc (llc : Low_level.t) : nest_info list * access list =
  let open Low_level in
  let rec is_nest = function For_loop _ -> true | If { body; _ } -> is_nest body | _ -> false in
  let stmts = flat_lines [ llc ] in
  let local_syms = Low_level.scope_value_syms llc in
  let nests, bare =
    List.partition_map stmts ~f:(fun stmt ->
        if is_nest stmt then
          First { n_loops = stmt; n_accesses = scan_accesses plc ~local_syms stmt }
        else Second (scan_accesses plc ~local_syms stmt))
  in
  (nests, List.concat bare)

let mentions_sym = Indexing.axis_index_mentions_any

(* The affine-query view of an access's index map: a vectorized write's last (minor-axis run)
   component is the base of a run, and a dynamic access's [dyn_axis] component is a data-dependent
   row (the vector holds a placeholder there) — both are masked to an opaque [Sub_axis], so the
   engine draws no (possibly wrong) disjointness or confinement conclusion from them. *)
let query_map (a : access) : Indexing.axis_index array =
  if ((not a.a_vec) && Option.is_none a.a_dyn_axis) || Array.is_empty a.a_idcs then a.a_idcs
  else begin
    let m = Array.copy a.a_idcs in
    if a.a_vec then m.(Array.length m - 1) <- Indexing.Sub_axis;
    Option.iter a.a_dyn_axis ~f:(fun ax -> if ax < Array.length m then m.(ax) <- Indexing.Sub_axis);
    m
  end

(* The single-child chain of [For_loop]s from the top of a nest, descending through [If] wrappers
   and comments; stops at the first branching ([Seq] with more than one non-comment statement). *)
let path_loops (nest : Low_level.t) : Low_level.t list =
  let open Low_level in
  let strip stmts = List.filter stmts ~f:(function Noop | Comment _ -> false | _ -> true) in
  let rec go llc acc =
    match llc with
    | For_loop fc -> (
        let acc = llc :: acc in
        match strip (flat_lines [ fc.body ]) with [ single ] -> go single acc | _ -> List.rev acc)
    | If { body; _ } -> go body acc
    | _ -> List.rev acc
  in
  go nest []

(* Shared analysis of the default annotator presets (schedule-ir-optops §6): per top-level nest, the
   parallelizable chain of outermost Serial path loops, validated by the conservative race analysis
   (see {!default_gpu}'s doc). Nests linked by cross-nest dependencies keep a common aligned prefix
   of their chains (see the module comment). Raises [Bail] when any check fails.

   [max_chain] caps the per-nest chain length. The default 2 is the presets' shape — each annotated
   nest carries exactly one Grid and one Workgroup loop. A sketch pipeline supplying its own
   geometry per chain position (gh-ocannl-521 companion coverage, via {!aligned_chains}) is not
   bound by that shape and passes its site's arity: a batched matmul's chain is batch loops plus row
   plus column (gh-ocannl-569 — capping at 2 made every rank-3+ site's companion coverage decline,
   serializing the axis whose spreading the hardware wanted most). The alignment rule is
   arity-independent; a longer chain only asks the same per-position question more times. *)
let analyze_parallel_chains ?(max_chain = 2) (opt : Low_level.optimized) : Low_level.t list list =
  let open Low_level in
  let plc = opt.optimize_ctx.placements in
  let nests, bare = split_nests plc opt.llc in
  (* Bare materialized writes cannot be covered by annotated loops. *)
  if List.exists bare ~f:(fun a -> a.a_write && Tn.Placements.is_materialized_peek plc a.a_tn) then
    raise Bail;
  (* Chains: per nest, the outermost (up to [max_chain]) Serial path loops whose index occurs as a
     plain [Iterator] component in every materialized write vector of the nest. *)
  let chains =
    List.map nests ~f:(fun n ->
        let mat_writes =
          List.filter n.n_accesses ~f:(fun a ->
              a.a_write && Tn.Placements.is_materialized_peek plc a.a_tn)
        in
        let qualifies s =
          (not (List.is_empty mat_writes))
          && List.for_all mat_writes ~f:(fun a ->
              Array.exists a.a_idcs ~f:(fun idx ->
                  Indexing.equal_axis_index idx (Indexing.Iterator s)))
        in
        let chain =
          List.filter (path_loops n.n_loops) ~f:(function
            | For_loop fc -> fc.from_ = 0 && qualifies fc.index
            | _ -> false)
          |> fun l -> List.take l max_chain
        in
        if List.is_empty chain && not (List.is_empty mat_writes) then raise Bail;
        chain)
  in
  let chain_syms chain =
    List.filter_map chain ~f:(function For_loop fc -> Some fc.index | _ -> None)
  in
  (* Access groups: the nests plus the bare pseudo-group (executed unconditionally, no chain). *)
  let groups =
    Array.of_list (List.map2_exn nests chains ~f:(fun n c -> (n.n_accesses, c)) @ [ (bare, []) ])
  in
  let n_groups = Array.length groups in
  let group_tbls =
    Array.map groups ~f:(fun (accs, _) ->
        let tbl = Hashtbl.create (module Int) in
        List.iter accs ~f:(fun a -> Hashtbl.add_multi tbl ~key:a.a_tn.Tn.uid ~data:a);
        tbl)
  in
  let full_syms = Array.map groups ~f:(fun (_, c) -> chain_syms c) in
  let sym_arr = Array.map full_syms ~f:Array.of_list in
  let extent_arr =
    Array.map groups ~f:(fun (_, c) ->
        Array.of_list
          (List.filter_map c ~f:(function For_loop fc -> Some (fc.to_ + 1) | _ -> None)))
  in
  (* Per-group loop-bound environments (the bare pseudo-group binds no loops): the box domains for
     the affine conflict queries below. *)
  let env_arr =
    Array.of_list (List.map nests ~f:(fun n -> Low_level.loop_bounds n.n_loops) @ [ [] ])
  in
  (* Cross-group dependency edges (module comment, aligned cross-nest parallelism): a node written
     in one group and touched in another must have all its write/access pairs aligned — unless it is
     per-thread scratch whose writes never mention the writer's parallel symbols (every thread then
     writes its whole private copy, and cross-nest reads are per-thread coherent regardless of
     geometry). *)
  let edges = ref [] in
  for i = 0 to n_groups - 1 do
    for j = i + 1 to n_groups - 1 do
      Hashtbl.iteri group_tbls.(i) ~f:(fun ~key:uid ~data:accs_i ->
          match Hashtbl.find group_tbls.(j) uid with
          | None -> ()
          | Some accs_j ->
              let writes_mention accs syms =
                (* Cell mention, or unpinned value mention: a per-thread copy written with a value
                   depending on a chain symbol that does not also pin the cell diverges from the
                   serial last-writer (see [value_invariant_ok] below), so such scratch cannot be
                   exempt from the cross-nest edge either. *)
                List.exists accs ~f:(fun a ->
                    a.a_write
                    && (Array.exists a.a_idcs ~f:(mentions_sym syms)
                       || List.exists a.a_val_syms ~f:(fun s ->
                           List.mem syms s ~equal:Indexing.equal_symbol
                           && not (Array.exists a.a_idcs ~f:(mentions_sym [ s ])))))
              in
              if
                (List.exists accs_i ~f:(fun a -> a.a_write)
                || List.exists accs_j ~f:(fun a -> a.a_write))
                && (Tn.Placements.is_materialized_peek plc (List.hd_exn accs_i).a_tn
                   || writes_mention accs_i full_syms.(i)
                   || writes_mention accs_j full_syms.(j))
              then edges := (i, j, uid) :: !edges)
    done
  done;
  (* [trims.(g)]: how many chain loops group [g] keeps; alignment may lower it (uniformly per
     dependency component, so the annotator emits identical geometry for linked nests). *)
  let trims = Array.map groups ~f:(fun (_, c) -> List.length c) in
  if not (List.is_empty !edges) then (
    let parent = Array.init n_groups ~f:Fn.id in
    let rec find x =
      if parent.(x) = x then x
      else (
        parent.(x) <- find parent.(x);
        parent.(x))
    in
    List.iter !edges ~f:(fun (i, j, _) ->
        let ri = find i and rj = find j in
        if ri <> rj then parent.(ri) <- rj);
    (* One write/access pair at trim depth [l]: statically-known indices; per axis position,
       parallel symbols are mentioned on both sides or neither; where both, plain [Iterator]s at the
       same chain position (extents are equal by the [l_max] prefix rule below). For materialized
       nodes this is the legacy (procedural) special case of the affine conflict query and is kept
       for [legality_crosscheck]; for non-materialized (per-thread copy) scratch it IS the rule —
       "reads hit exactly the cells the same thread writes" is an order-sensitive per-thread-copy
       fact, not a shared-memory conflict, so it is not subsumed by [Affine.pair_conflict]. *)
    let pair_aligned_procedural ~l gi gj (a : access) (b : access) =
      (not a.a_dynamic) && (not b.a_dynamic)
      &&
      let syms_i = List.take full_syms.(gi) l and syms_j = List.take full_syms.(gj) l in
      let pos g s =
        Array.findi sym_arr.(g) ~f:(fun k s' -> k < l && Indexing.equal_symbol s s')
        |> Option.map ~f:fst
      in
      let comp idcs p = if p < Array.length idcs then idcs.(p) else Indexing.Fixed_idx 0 in
      let rank = max (Array.length a.a_idcs) (Array.length b.a_idcs) in
      let aligned_at p =
        let ci = comp a.a_idcs p and cj = comp b.a_idcs p in
        match (mentions_sym syms_i ci, mentions_sym syms_j cj) with
        | false, false -> true
        | true, true -> (
            match (ci, cj) with
            | Indexing.Iterator si, Indexing.Iterator sj -> (
                match (pos gi si, pos gj sj) with Some ki, Some kj -> ki = kj | _ -> false)
            (* An [Affine] over a parallel symbol is never a plain aligned slice. *)
            | _ -> false)
        | _ -> false
      in
      List.for_all (List.init rank ~f:Fn.id) ~f:aligned_at
    in
    (* Materialized edges: the affine conflict query decides — conflicts between the two nests'
       accesses must be confined to a single hardware thread (the paired trimmed-chain symbols) or
       be disjoint outright (which also admits pairs the procedural rule could only decline, e.g.
       constant-offset or strided-disjoint slices). *)
    let pair_aligned_query ~l gi gj (a : access) (b : access) =
      (not a.a_dynamic) && (not b.a_dynamic)
      &&
      let range s =
        match List.Assoc.find env_arr.(gi) s ~equal:Indexing.equal_symbol with
        | Some _ as r -> r
        | None -> List.Assoc.find env_arr.(gj) s ~equal:Indexing.equal_symbol
      in
      let dup g s = List.Assoc.mem env_arr.(g) s ~equal:Indexing.equal_symbol in
      let pairs = List.zip_exn (List.take full_syms.(gi) l) (List.take full_syms.(gj) l) in
      let verdict =
        Affine.pair_conflict ~range ~dup_left:(dup gi) ~dup_right:(dup gj) ~pairs
          ~left:(query_map a) ~right:(query_map b)
      in
      let query_safe = match verdict with Affine.Cross_thread _ -> false | _ -> true in
      Affine.crosscheck ~site:"schedule cross-nest alignment" ~context:(Tn.debug_name a.a_tn)
        ~procedural_safe:(fun () -> pair_aligned_procedural ~l gi gj a b)
        ~query_safe
        ~witness:(match verdict with Affine.Cross_thread w -> w | _ -> "");
      query_safe
    in
    let edge_aligned ~l (i, j, uid) =
      let accs_i = Hashtbl.find_multi group_tbls.(i) uid
      and accs_j = Hashtbl.find_multi group_tbls.(j) uid in
      let is_mat = Tn.Placements.is_materialized_peek plc (List.hd_exn accs_i).a_tn in
      let pair_aligned = if is_mat then pair_aligned_query else pair_aligned_procedural in
      (* Per-thread copies of statement-crossing scratch: a consumer thread's copy holds its own
         chunk's last value, while the serial reference holds the last chunk's — they coincide only
         when the written value cannot vary across the chunks writing the same cell. At trim level
         [l], every write's value symbols must avoid the writer's parallel symbols unless they also
         pin the written cell (then only the reader's own thread wrote it). The search over [l]
         serializes the offending chain loop. Syntactic, direct dependence only. *)
      let value_invariant_ok =
        is_mat
        || List.for_all
             [ (accs_i, i); (accs_j, j) ]
             ~f:(fun (accs, g) ->
               let syms = List.take full_syms.(g) l in
               List.for_all accs ~f:(fun a ->
                   (not a.a_write)
                   || List.for_all a.a_val_syms ~f:(fun s ->
                       (not (List.mem syms s ~equal:Indexing.equal_symbol))
                       || Array.exists a.a_idcs ~f:(mentions_sym [ s ]))))
      in
      value_invariant_ok
      && List.for_all accs_i ~f:(fun a ->
          List.for_all accs_j ~f:(fun b ->
              (not (a.a_write || b.a_write)) || pair_aligned ~l i j a b))
    in
    let comps = Hashtbl.create (module Int) in
    Array.iteri parent ~f:(fun g _ -> Hashtbl.add_multi comps ~key:(find g) ~data:g);
    Hashtbl.iteri comps ~f:(fun ~key:root ~data:members ->
        if List.length members > 1 then
          let comp_edges = List.filter !edges ~f:(fun (i, _, _) -> find i = root) in
          (* Longest prefix of the members' chains with pointwise-equal extents: the geometry must
             be identical for the thread->iteration maps to coincide across the component (a 1-loop
             chain splits Grid x Workgroup while a 2-loop chain retypes in place, so even a shared
             axis maps differently across different chain shapes). *)
          let l_max =
            List.fold members ~init:max_chain ~f:(fun m g -> min m (Array.length extent_arr.(g)))
          in
          let ext_eq k =
            match members with
            | [] -> true
            | g0 :: rest -> List.for_all rest ~f:(fun g -> extent_arr.(g).(k) = extent_arr.(g0).(k))
          in
          let rec agree k = if k >= l_max || not (ext_eq k) then k else agree (k + 1) in
          let l_max = agree 0 in
          let rec search l =
            if l < 1 then raise Bail
            else if List.for_all comp_edges ~f:(edge_aligned ~l) then
              List.iter members ~f:(fun g -> trims.(g) <- l)
            else search (l - 1)
          in
          search l_max));
  let chains = List.mapi chains ~f:(fun i c -> List.take c trims.(i)) in
  (* Per-nest hazard analysis over the final (possibly trimmed) parallel symbols (see the module
     comment for the safety argument). *)
  List.iter2_exn nests chains ~f:(fun n chain ->
      let syms = chain_syms chain in
      let env = Low_level.loop_bounds n.n_loops in
      let range s = List.Assoc.find env s ~equal:Indexing.equal_symbol in
      let dup s = List.Assoc.mem env s ~equal:Indexing.equal_symbol in
      let pairs = List.map syms ~f:(fun s -> (s, s)) in
      (* The agreement rule: all accesses agree on every component that mentions a parallel symbol.
         For materialized nodes it is the legacy (procedural) special case of the affine conflict
         query, kept for [legality_crosscheck]; for non-materialized (per-thread copy) scratch it IS
         the rule (see [pair_aligned_procedural]'s note). *)
      let agreement_ok accs =
        let rank = List.fold accs ~init:0 ~f:(fun m a -> max m (Array.length a.a_idcs)) in
        let ok = ref true in
        for p = 0 to rank - 1 do
          let comps =
            List.map accs ~f:(fun a ->
                if p < Array.length a.a_idcs then a.a_idcs.(p) else Indexing.Fixed_idx 0)
          in
          if List.exists comps ~f:(mentions_sym syms) then
            match comps with
            | [] -> ()
            | c0 :: rest ->
                if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then ok := false
        done;
        !ok
      in
      let by_tn = Hashtbl.create (module Int) in
      List.iter n.n_accesses ~f:(fun a -> Hashtbl.add_multi by_tn ~key:a.a_tn.Tn.uid ~data:a);
      Hashtbl.iter by_tn ~f:(fun accs ->
          let written = List.exists accs ~f:(fun a -> a.a_write) in
          if written then
            let is_mat = Tn.Placements.is_materialized_peek plc (List.hd_exn accs).a_tn in
            let chain_relevant =
              List.exists accs ~f:(fun a -> Array.exists a.a_idcs ~f:(mentions_sym syms))
            in
            if is_mat || chain_relevant then (
              let has_dynamic = List.exists accs ~f:(fun a -> a.a_dynamic) in
              (* gh-484 (task 2, unbailing the gh-466 scatter): dynamic accesses of a materialized
                 node no longer bail wholesale. The dynamic component is masked to [Sub_axis] in
                 [query_map], so the conflict query decides from the statically-known components: a
                 chain symbol pinning a same-position plain component of every access confines
                 conflicts to its own thread (the per-block partials row of [Split_reduce], the
                 embedding-dim column of the scatter), while loops driving the dynamic index are
                 never forced equal across threads and stay serial. Per-thread (non-materialized)
                 scratch keeps bailing: its containment rule ("reads hit exactly the cells the same
                 thread wrote") is order-sensitive and unknowable under data-dependent rows. *)
              if has_dynamic && not is_mat then raise Bail;
              if is_mat then (
                (* Materialized: genuine shared memory — the affine conflict query decides. Every
                   pair involving a write must have its conflicts confined to one thread of the
                   chain tuple, or be disjoint outright. *)
                let witness = ref "" in
                let query_safe =
                  List.for_all accs ~f:(fun w ->
                      (not w.a_write)
                      || List.for_all accs ~f:(fun x ->
                          match
                            Affine.pair_conflict ~range ~dup_left:dup ~dup_right:dup ~pairs
                              ~left:(query_map w) ~right:(query_map x)
                          with
                          | Affine.Disjoint | Affine.Same_thread -> true
                          | Affine.Cross_thread wit ->
                              witness := wit;
                              false))
                in
                (* The procedural agreement rule does not model dynamic accesses (it never ran on
                   them — they used to bail first), so the crosscheck compares only static nodes. *)
                if not has_dynamic then
                  Affine.crosscheck ~site:"schedule per-nest hazard"
                    ~context:(Tn.debug_name (List.hd_exn accs).a_tn)
                    ~procedural_safe:(fun () -> agreement_ok accs)
                    ~query_safe ~witness:!witness;
                if not query_safe then raise Bail)
              else if not (agreement_ok accs) then raise Bail)));
  chains

(* gh-494 waypoint 2: the per-thread-copy scratch rule, recomputed as same-thread containment
   queries over a whole kernel's code and compared under [legality_crosscheck]. Each nest's final
   (trimmed) chain symbols are renamed to canonical thread symbols — positional pairing, which is
   the annotator's thread identity across aligned nests — turning "each thread reads only cells it
   wrote itself, earlier in its own serial chunk" into exactly the engine's shared-parameter
   cancellation plus statement-order visibility (an unaligned nest's chain loops keep distinct
   bounds, fail the common-prefix test, and decline as residual thread parameters). Called from the
   default annotators after {!analyze_parallel_chains} accepted the kernel (not from its partial
   per-statement invocations, whose code excludes cross-step writes), so the procedural side is a
   constant accept: a raise means an accepted schedule contains a read of per-thread scratch not
   provably covered by the same thread's earlier writes. Nodes with dynamic accesses are skipped
   (scatter into scratch: statically unknown cells). *)
let crosscheck_scratch_containment (opt : Low_level.optimized) (chains : Low_level.t list list) :
    unit =
  if Lazy.force Affine.crosscheck_enabled then (
    let open Low_level in
    let plc = opt.optimize_ctx.placements in
    let nests, _bare = split_nests plc opt.llc in
    let stmts = flat_lines [ opt.llc ] in
    let chain_syms chain =
      List.filter_map chain ~f:(function For_loop fc -> Some fc.index | _ -> None)
    in
    (* Canonical thread symbols: per chain position, the first nest's symbol (no fresh symbols —
       allocating here would drift symbol numbering when the crosscheck is enabled). Symbols are
       unique per loop construct, so a canonical symbol cannot occur in another nest already. *)
    let max_len = List.fold chains ~init:0 ~f:(fun m c -> max m (List.length c)) in
    let canon = Array.create ~len:max_len None in
    List.iter chains ~f:(fun chain ->
        List.iteri (chain_syms chain) ~f:(fun k s ->
            if Option.is_none canon.(k) then canon.(k) <- Some s));
    let renames =
      List.map2_exn nests chains ~f:(fun n chain ->
          let idx, _ = Option.value_exn (List.findi stmts ~f:(fun _ s -> phys_equal s n.n_loops)) in
          (idx, List.mapi (chain_syms chain) ~f:(fun k s -> (s, Option.value_exn canon.(k)))))
    in
    let rename_sym m s =
      List.Assoc.find m s ~equal:Indexing.equal_symbol |> Option.value ~default:s
    in
    let rename_idx m (idx : Indexing.axis_index) =
      match idx with
      | Indexing.Iterator s -> Indexing.Iterator (rename_sym m s)
      | Indexing.Affine { symbols; offset } ->
          Indexing.Affine
            { symbols = List.map symbols ~f:(fun (c, s) -> (c, rename_sym m s)); offset }
      | Indexing.Fixed_idx _ | Indexing.Sub_axis -> idx
      | Indexing.Concat syms -> Indexing.Concat (List.map syms ~f:(rename_sym m))
    in
    let accs =
      List.map (Low_level.affine_accesses opt.llc) ~f:(fun a ->
          match a.Affine.a_path with
          | Affine.Stmt g :: _ -> (
              match List.Assoc.find renames g ~equal:Int.equal with
              | None -> a
              | Some m ->
                  {
                    a with
                    Affine.a_map = Array.map a.Affine.a_map ~f:(rename_idx m);
                    a_loops = List.map a.a_loops ~f:(fun (s, b) -> (rename_sym m s, b));
                    a_val_syms = List.map a.a_val_syms ~f:(rename_sym m);
                  })
          | _ -> a)
    in
    let thread s =
      Array.exists canon ~f:(function Some c -> Indexing.equal_symbol c s | None -> false)
    in
    let by_tn = Hashtbl.create (module Int) in
    List.iter accs ~f:(fun a -> Hashtbl.add_multi by_tn ~key:a.Affine.a_tn.Tn.uid ~data:a);
    Hashtbl.iter by_tn ~f:(fun accs ->
        let accs = List.rev accs in
        let tn = (List.hd_exn accs).Affine.a_tn in
        if
          (not (Tn.Placements.is_materialized_peek plc tn))
          && List.exists accs ~f:(fun a -> a.Affine.a_write)
          && not (List.exists accs ~f:(fun a -> a.Affine.a_dynamic))
        then
          let writes = List.filter accs ~f:(fun a -> a.Affine.a_write) in
          let head = Affine.stmt_head in
          let witness = ref "" in
          (* The value side of the per-thread-copy semantics (see [read_covered_before]'s doc): a
             write covering reads in other top-level statements must have every thread symbol that
             feeds its value also pin the written cell. *)
          let value_ok =
            List.for_all writes ~f:(fun w ->
                let crossing =
                  List.exists accs ~f:(fun r ->
                      (not r.Affine.a_write) && head r.a_path <> head w.Affine.a_path)
                in
                (not crossing)
                || List.for_all w.Affine.a_val_syms ~f:(fun s ->
                    (not (thread s))
                    || Array.exists w.a_map ~f:(fun idx ->
                        match idx with
                        | Indexing.Iterator s' -> Indexing.equal_symbol s s'
                        | Indexing.Affine { symbols; _ } ->
                            List.exists symbols ~f:(fun (_, s') -> Indexing.equal_symbol s s')
                        | _ -> false))
                ||
                (witness := "thread-variant value in statement-crossing scratch write";
                 false))
          in
          let query_safe =
            value_ok
            && List.for_all accs ~f:(fun r ->
                r.Affine.a_write
                ||
                match Affine.read_covered_before ~thread ~read:r ~writes () with
                | `Covered -> true
                | `Unknown w ->
                    witness := w;
                    false)
          in
          Affine.crosscheck ~site:"schedule per-thread scratch containment"
            ~context:(Tn.debug_name tn)
            ~procedural_safe:(fun () -> true)
            ~query_safe ~witness:!witness))

(* Parallel iterations a chain covers. *)
let chain_size chain =
  List.fold chain ~init:1 ~f:(fun sz -> function
    | Low_level.For_loop fc -> sz * (fc.to_ + 1)
    | _ -> sz)

(* Threshold helper: skip kernels whose largest parallelizable nest is too small to pay for a launch
   (GPU) or a task fan-out (CPU). *)
let max_parallel_size chains = List.fold chains ~init:0 ~f:(fun m chain -> max m (chain_size chain))

(* The default annotators' cross-nest analysis, exposed as data instead of as a preset schedule
   (gh-ocannl-521). A sketch pipeline that annotates one nest by construction needs the SAME facts
   to cover the routine's other nests — which nests exist, and which of their outermost loops may
   carry hardware geometry aligned with the site's — but not the presets' choice of geometry, which
   is Grid/Workgroup-per-nest and cannot express a tensorized nest's slot structure. Returning the
   trimmed chains lets the caller supply its own geometry per chain position while the alignment
   (positional thread identity across linked nests, equal extents within a dependency component)
   stays this module's rule. *)
let aligned_chains ?max_chain ?(expanded_zeros = []) (opt : Low_level.optimized) :
    (Low_level.t * (Indexing.symbol * int) list) list option =
  (* A whole-node [Zero_out] is a bare materialized write, which the analysis rejects outright — but
     a caller that pairs this query with {!Expand_zero} turns it into a per-element nest carrying
     the caller's own geometry before the code is validated, so it is not the bare write the rule is
     about. Drop those statements for the query; the resulting nest writes the whole node under the
     same geometry as the accumulation nest that overwrites it, hence is aligned by construction. *)
  let opt =
    if List.is_empty expanded_zeros then opt
    else
      let rec drop (llc : Low_level.t) =
        match llc with
        | Low_level.Zero_out tn when List.exists expanded_zeros ~f:(Tn.equal tn) -> Low_level.Noop
        | Low_level.Seq (a, b) -> Low_level.Seq (drop a, drop b)
        | _ -> llc
      in
      { opt with Low_level.llc = drop opt.Low_level.llc }
  in
  match
    let chains = analyze_parallel_chains ?max_chain opt in
    crosscheck_scratch_containment opt chains;
    chains
  with
  (* Only [Bail] — the analysis declining. A [legality_crosscheck] divergence must stay loud. *)
  | exception Bail -> None
  | chains ->
      let nests, _bare = split_nests opt.optimize_ctx.placements opt.llc in
      Some
        (List.map2_exn nests chains ~f:(fun n chain ->
             ( n.n_loops,
               List.filter_map chain ~f:(function
                 | Low_level.For_loop fc -> Some (fc.index, fc.to_ + 1)
                 | _ -> None) )))

(* The configured block size is a target; the device's capacity is a hard cap. Two hardware facts,
   not one (gh-ocannl-679): the workgroup's thread PRODUCT ([max_threads_per_workgroup]) and its
   per-dimension bound ([max_workgroup_dims]) — on CUDA the latter's [.z] entry is 16x below the
   former. Both annotators below emit exactly one [Workgroup] loop per nest, so the extent they
   choose lands on [.x] and only that entry can bind; clamping here is what keeps
   [check_hardware_limits_classified] a backstop rather than the first line of defence. *)
let clamp_block_size ~(limits : Backend_intf.hardware_limits) block_size =
  let clamp cap n = Option.value_map cap ~default:n ~f:(min n) in
  clamp limits.max_threads_per_workgroup block_size
  |> clamp (Option.map limits.max_workgroup_dims ~f:(fun (x, _, _) -> x))

let default_gpu ?block_size ?min_parallel ?(limits = Backend_intf.no_hardware_limits)
    (opt : Low_level.optimized) : schedule =
  let open Low_level in
  let block_size =
    Option.value block_size
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
  in
  let block_size = clamp_block_size ~limits block_size in
  (* Default 64 (was 1024): a kernel launches either way, so on GPU any real parallelism beats the
     serial 1x1 fallback — a single GPU thread is 1-2 orders of magnitude slower than a CPU core,
     and the "too small to pay for a launch" reasoning of the CPU preset's fan-out threshold does
     not transfer (there is no cheaper non-launching alternative). The remaining small threshold
     keeps sub-simdgroup-scale programs (tutorials, scalar tails) fully serial: their segments then
     coalesce back to a single kernel and fission's placement promotions are undone, so tiny
     programs keep byte-identical artifacts and context contents. *)
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"64")
  in
  try
    let chains = analyze_parallel_chains opt in
    crosscheck_scratch_containment opt chains;
    if max_parallel_size chains < min_parallel then []
    else
      (* Emit per-nest ops. Every annotated nest contributes exactly one Grid and one Workgroup
         loop, so hardware slots are uniform ([.x] of each kind) across nests and every materialized
         write covers all active dimensions ([validate_parallel]'s requirement). *)
      List.concat_map chains ~f:(fun chain ->
          match chain with
          | [] -> []
          | [ For_loop fc ] ->
              let n0 = fc.to_ + 1 in
              let op, _, _ =
                split ~axis:fc.index ~factor:(min block_size n0) ~outer:Grid ~inner:Workgroup
              in
              [ op ]
          | For_loop fc0 :: For_loop fc1 :: _ ->
              let n1 = fc1.to_ + 1 in
              if n1 <= block_size then
                [
                  Retype { axis = fc0.index; ty = Grid };
                  Retype { axis = fc1.index; ty = Workgroup };
                ]
              else
                let op, _, _ =
                  split ~axis:fc1.index ~factor:block_size ~outer:Serial ~inner:Workgroup
                in
                [ Retype { axis = fc0.index; ty = Grid }; op ]
          | _ -> [])
  with Bail -> []

let default_cpu ?min_parallel (opt : Low_level.optimized) : schedule =
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string
        @@ Utils.get_global_arg ~arg_name:"cpu_schedule_min_parallel" ~default:"16384")
  in
  try
    let chains = analyze_parallel_chains opt in
    crosscheck_scratch_containment opt chains;
    if max_parallel_size chains < min_parallel then []
    else
      (* Retype the outermost chain loop to [Grid], nothing else: pool-backed Grid rendering
         (gh-ocannl-164) partitions that loop into contiguous chunks, and a [Workgroup] split would
         only add loop structure that executes serially inside a chunk. *)
      List.concat_map chains ~f:(function
        | Low_level.For_loop fc :: _ -> [ Retype { axis = fc.index; ty = Low_level.Grid } ]
        | _ -> [])
  with Bail -> []

(** {2 Kernel fission at cross-workgroup edges}

    The default annotator's cross-nest interference check (see {!analyze_parallel_chains}) is a fact
    about hardware threads: sibling nests of one kernel execute with no grid-wide synchronization
    between them, so a producer/consumer (or WAW/WAR) pair of top-level statements over a
    materialized node is a race once threads interleave — and the whole routine used to fall back to
    a 1×1 launch. Fission recovers the synchronization from the stream instead: top-level statements
    are partitioned into {e segments} at those dependency edges, each segment compiles to its own
    kernel, and the routine's task launches them in order on the routine's stream with a device-side
    event chained at each boundary (queue FIFO alone does not order overlapping command buffers over
    Metal's untracked resources; see [Raise_backend.link]). Each segment then gets the default
    schedule independently, on its own launch geometry. Dependency edges the aligned cross-nest rule
    proves race-free do {e not} cut, provided the shared kernel keeps every nest's standalone
    parallelism (see {!aligned_merge}): elementwise chains over one intermediate stay a single
    kernel, while parallelism switches (a batch-parallel producer feeding a reduce-over-batch
    consumer) and alignment-trimming merges still cut.

    Segmentation is conservative and total (no [Bail]): a statement opaque to the analysis
    ([Staged_compilation], barriers, pre-annotated loops) or one the annotator can never cover (bare
    materialized writes, materialized writes inside [Local_scope] bodies, non-injective nests) is
    isolated into its own serial segment rather than poisoning its neighbors' schedules.
    Materialized whole-node [Zero_out]s are likewise isolated and — on GPU — expanded
    ({!optop.Expand_zero}) and annotated with the same geometry policy as ordinary nests.

    Two constructs cross segment boundaries and need repair, because a kernel's locals die at launch
    end:

    - Scalar scope-locals hoisted by [Low_level.hoist_cross_statement_cse] (top-level
      [Declare_local] + defining statements): the defining statements are {e replicated} at the head
      of every consuming segment. This is valid exactly when nothing written between the
      definition's original position and the segment start overlaps the definition's reads (the
      replica then computes the same value); otherwise the offending segments are merged back and
      run serially, preserving today's behavior.
    - [Local]-placed scratch tensor nodes whose accesses end up split across segments are promoted
      to [On_device] ({!Tnode.Placements.promote_local_to_device}): [Local] is always a compiler
      decision premised on single-kernel lifetime, and fission runs before any consumer of the
      decision (codegen parameter lists, context allocation) has read it.

    Adjacent segments that both end up unannotated are coalesced back (fewer launches); if
    everything coalesces, the routine compiles to a single kernel exactly as before. *)

(* Per top-level statement: tensor-node access sets, scope-local references crossing statement
   boundaries, and the statement kind for segmentation. [None] = opaque to the analysis. *)
type stmt_summary = {
  s_reads : Set.M(Tn).t;
  s_writes : Set.M(Tn).t;  (** Includes [Zero_out] targets. *)
  s_top_zero : Tn.t option;
      (** The statement is exactly [Zero_out tn] of a materialized node (expandable). *)
  s_scope_reads : Low_level.scope_id list;
      (** [Get_local]s of scope ids bound outside this statement. *)
  s_scope_writes : Low_level.scope_id list;
      (** [Set_local]s of scope ids bound outside this statement. *)
  s_scope_declares : Low_level.scope_id list;  (** [Declare_local]s within this statement. *)
}

exception Opaque_stmt

let summarize_stmt plc (stmt : Low_level.t) : stmt_summary option =
  let open Low_level in
  let reads = ref (Set.empty (module Tn)) and writes = ref (Set.empty (module Tn)) in
  let top_zero = ref None in
  let scope_reads = ref [] and scope_writes = ref [] in
  let bound = ref [] and declares = ref [] in
  let rec code ~top llc =
    match llc with
    | Noop | Comment _ -> ()
    | Staged_compilation _ | Workgroup_barrier | Tile_mma _ -> raise Opaque_stmt
    | Declare_local { id; _ } -> declares := id :: !declares
    | Seq (a, b) ->
        code ~top a;
        code ~top b
    | For_loop { axis; body; _ } ->
        if not (equal_axis_type axis Serial) then raise Opaque_stmt;
        code ~top:false body
    | Scan_loop { carried; body; _ } ->
        (* The carried locals are bound by the statement itself, like a [Declare_local] within
           it. *)
        List.iter carried ~f:(fun c ->
            bound := c.prev :: c.next :: !bound;
            scalar c.init);
        code ~top:false body
    | Zero_out tn ->
        writes := Set.add !writes tn;
        if top && Tn.Placements.is_materialized_peek plc tn then top_zero := Some tn
    | Set { tn; llsc; _ } ->
        writes := Set.add !writes tn;
        scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        (* The RMW read of [tn] surfaces via the [Get_dynamic] inside [llsc]. *)
        writes := Set.add !writes tn;
        scalar v;
        scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        writes := Set.add !writes tn;
        scalar a
    | Set_local (id, llsc) ->
        scope_writes := id :: !scope_writes;
        scalar llsc
    | If { cond = c, _; body } ->
        scalar c;
        code ~top:false body
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        bound := id :: !bound;
        code ~top:false body
    | Get_local id -> scope_reads := id :: !scope_reads
    | Get (tn, _) -> reads := Set.add !reads tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        reads := Set.add !reads tn;
        scalar v
    | Get_merge_buffer (_, _) -> () (* A separate read-only input buffer: never a hazard. *)
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
  in
  match code ~top:true stmt with
  | exception Opaque_stmt -> None
  | () ->
      let internal = !bound @ !declares in
      let externals ids =
        List.filter ids ~f:(fun id -> not (List.mem internal id ~equal:Low_level.equal_scope_id))
        |> List.dedup_and_sort ~compare:Low_level.compare_scope_id
      in
      Some
        {
          s_reads = !reads;
          s_writes = !writes;
          s_top_zero = !top_zero;
          s_scope_reads = externals !scope_reads;
          s_scope_writes = externals !scope_writes;
          s_scope_declares = List.dedup_and_sort !declares ~compare:Low_level.compare_scope_id;
        }

type funit = {
  f_stmts : Low_level.t list;  (** Leading comments/noops + the statement, original order. *)
  f_index : int;  (** Position among units, for scope-replication validity ranges. *)
  f_sum : stmt_summary option;
  f_kind : [ `Normal | `Zeros | `Solo ];
  f_chains : Low_level.t list list;
      (** The statement's chains analyzed standalone ([[]] for non-[`Normal] units): the
          no-parallelism-loss baseline for aligned segment merging (see {!aligned_merge}). *)
}

(* Units: each real statement with the comments preceding it. Trailing comments attach to the last
   unit. [max_chain] caps the standalone chains; it must match the cap {!aligned_merge}'s merged
   analysis runs under (see there). *)
let collect_units ?max_chain plc (opt : Low_level.optimized) (stmts : Low_level.t list) : funit list
    =
  let open Low_level in
  let is_glue = function Noop | Comment _ -> true | _ -> false in
  let mk index glue stmt =
    let f_sum = summarize_stmt plc stmt in
    let f_kind, f_chains =
      match f_sum with
      | None -> (`Solo, [])
      | Some { s_top_zero = Some _; _ } -> (`Zeros, [])
      | Some _ -> (
          (* Isolate statements the annotator could never share a parallel kernel with: it raises
             [Bail] on them even in isolation (bare materialized writes, materialized writes inside
             [Local_scope] bodies, dynamic accesses or non-injective/uncovered writes). In its own
             segment such a statement runs as a serial 1×1 launch and its neighbors keep their
             parallelism. Read-only statements never bail; their chains are empty. *)
          match analyze_parallel_chains ?max_chain { opt with llc = stmt } with
          | chains -> (`Normal, chains)
          | exception Bail -> (`Solo, []))
    in
    { f_stmts = List.rev (stmt :: glue); f_index = index; f_sum; f_kind; f_chains }
  in
  let rec go index glue acc = function
    | [] -> (
        (* Trailing glue: attach to the last unit. *)
        match (acc, glue) with
        | _, [] | [], _ -> List.rev acc
        | last :: rest, glue ->
            List.rev ({ last with f_stmts = last.f_stmts @ List.rev glue } :: rest))
    | stmt :: tl when is_glue stmt -> go index (stmt :: glue) acc tl
    | stmt :: tl -> go (index + 1) [] (mk index glue stmt :: acc) tl
  in
  go 0 [] [] stmts

type segment = {
  g_units : funit list;
  g_kind : [ `Normal | `Zeros | `Solo ];
  g_reads : Set.M(Tn).t;
  g_writes : Set.M(Tn).t;
  g_chains : Low_level.t list list;
      (** Concatenation of the units' standalone chains, in statement order (the merged code's nest
          order): the baseline for {!aligned_merge}'s no-parallelism-loss guard. *)
}

let seg_of_unit u =
  let reads, writes =
    match u.f_sum with
    | Some s -> (s.s_reads, s.s_writes)
    | None -> (Set.empty (module Tn), Set.empty (module Tn))
  in
  { g_units = [ u ]; g_kind = u.f_kind; g_reads = reads; g_writes = writes; g_chains = u.f_chains }

let merge_segs ~kind a b =
  {
    g_units = a.g_units @ b.g_units;
    g_kind = kind;
    g_reads = Set.union a.g_reads b.g_reads;
    g_writes = Set.union a.g_writes b.g_writes;
    g_chains = a.g_chains @ b.g_chains;
  }

(* The within-kernel interference rule of {!analyze_parallel_chains}, reformulated as a cut
   criterion: a materialized node written on one side and touched on the other must not share a
   kernel — unless the merged statements pass the aligned cross-nest analysis without losing
   parallelism ({!aligned_merge}). Local-scratch pairs deliberately do NOT cut — the per-segment
   annotator retains the finer per-thread-privacy analysis for those (and simply declines to
   annotate when it fails). *)
let mat_conflict plc seg (s : stmt_summary) =
  let mat tn = Tn.Placements.is_materialized_peek plc tn in
  let hits w touched = Set.exists w ~f:(fun tn -> mat tn && Set.mem touched tn) in
  hits seg.g_writes (Set.union s.s_reads s.s_writes)
  || hits s.s_writes (Set.union seg.g_reads seg.g_writes)

(* Whether extending [seg] with [u] keeps one parallelizable kernel: the merged statements pass
   {!analyze_parallel_chains} (its aligned cross-nest rule proves the dependency race-free) AND no
   nest's parallel size shrinks versus its standalone analysis. Alignment may trim linked chains to
   a common prefix; when it would, cutting (a launch boundary, today's behavior) is the better
   default than trading parallelism for one launch — and keeps the fissioned candidates of the
   schedule search at full per-segment parallelism.

   [max_chain] must match the cap the standalone chains ([seg.g_chains], [u.f_chains]) were computed
   under, so the no-loss comparison is arity-for-arity.

   [uniform_shapes] (the [arity_cuts] mode of {!fission_scheduled}): additionally require every
   materialized-writing nest's merged chain to carry the {e same} extent list. The GPU matmul
   sketches emit one geometry at the site's full arity over every such nest of the kernel
   ([companion_geometry], gh-ocannl-521/569), so a nest whose chain cannot follow that arity — a
   reduction over the site's minor axis, an initialization nest of the reduction target — makes
   every sketch seed of the shared segment decline; cutting it apart frees the site (gh-ocannl-574).
   Nests with no materialized writes have empty chains and are exempt, mirroring the coverage
   demand. *)
let aligned_merge ?max_chain ?(uniform_shapes = false) (opt : Low_level.optimized) seg (u : funit) :
    bool =
  let llc =
    Low_level.unflat_lines (List.concat_map (seg.g_units @ [ u ]) ~f:(fun u' -> u'.f_stmts))
  in
  match analyze_parallel_chains ?max_chain { opt with llc } with
  | exception Bail -> false
  | merged -> (
      let no_loss =
        match
          List.for_all2 merged (seg.g_chains @ u.f_chains) ~f:(fun m s ->
              chain_size m >= chain_size s)
        with
        | List.Or_unequal_lengths.Ok ok -> ok
        | Unequal_lengths -> false
      in
      no_loss
      &&
      match uniform_shapes with
      | false -> true
      | true -> (
          let extents chain =
            List.filter_map chain ~f:(function
              | Low_level.For_loop fc -> Some (fc.to_ + 1)
              | _ -> None)
          in
          let nonempty c = if List.is_empty c then None else Some (extents c) in
          match List.filter_map merged ~f:nonempty with
          | [] -> true
          | e0 :: rest -> List.for_all rest ~f:(List.equal ( = ) e0)))

let group_units ?max_chain ?(arity_cuts = false) (opt : Low_level.optimized) (units : funit list) :
    segment list =
  let plc = opt.Low_level.optimize_ctx.placements in
  let close cur acc = match cur with None -> acc | Some seg -> seg :: acc in
  let mergeable seg s u =
    (* In [arity_cuts] mode even a conflict-free extension must pass the uniform-shape rule: an
       unrelated nest of a different arity (the reduction target's initialization) fails the
       sketches' companion coverage just like a trimmed one. *)
    if arity_cuts then aligned_merge ?max_chain ~uniform_shapes:true opt seg u
    else (not (mat_conflict plc seg s)) || aligned_merge ?max_chain opt seg u
  in
  let rec go cur acc = function
    | [] -> List.rev (close cur acc)
    | u :: tl -> (
        match (cur, u.f_kind, u.f_sum) with
        | Some ({ g_kind = `Normal; _ } as seg), `Normal, Some s when mergeable seg s u ->
            go (Some (merge_segs ~kind:`Normal seg (seg_of_unit u))) acc tl
        | Some ({ g_kind = `Zeros; _ } as seg), `Zeros, Some s when not (mat_conflict plc seg s) ->
            go (Some (merge_segs ~kind:`Zeros seg (seg_of_unit u))) acc tl
        | _ -> go (Some (seg_of_unit u)) (close cur acc) tl)
  in
  go None [] units

(** {3 Scope-local replication across segments (option (b) v2)} *)

exception Unfissionable
(* Malformed scope-local shape (a scope id referenced with no defining statement before it, or a
   merge cascade reaching the first segment): the driver falls back to single-kernel compilation.
   Repairable crossings do not raise — they return [None] from [plan_replicas] and merge their
   segment range instead. *)

(* Def units of scope id [id] strictly before unit index [limit]: the declaring unit and every unit
   that [Set_local]s it from outside. *)
let def_units_of (units : funit array) ~limit id =
  Array.to_list units
  |> List.filter ~f:(fun u ->
      u.f_index < limit
      &&
      match u.f_sum with
      | None -> false
      | Some s ->
          List.mem s.s_scope_declares id ~equal:Low_level.equal_scope_id
          || List.mem s.s_scope_writes id ~equal:Low_level.equal_scope_id)

(* The replica set for one segment: transitive closure of def units over the scope ids they read
   themselves. Returns [None] when replication is invalid — a def unit writes tensors or is opaque,
   or a unit between the (earliest) def and the segment start writes a tensor some def reads (the
   replica would compute a different value than the original definition did). *)
let plan_replicas (units : funit array) ~seg_start (ext_ids : Low_level.scope_id list) :
    funit list option =
  let module SId = struct
    let equal = Low_level.equal_scope_id
  end in
  let rec closure needed_ids collected =
    match needed_ids with
    | [] -> collected
    | id :: rest ->
        let defs = def_units_of units ~limit:seg_start id in
        if List.is_empty defs then raise Unfissionable;
        let fresh = List.filter defs ~f:(fun d -> not (List.mem collected d ~equal:phys_equal)) in
        let more_ids =
          List.concat_map fresh ~f:(fun d ->
              match d.f_sum with None -> [] | Some s -> s.s_scope_reads)
          |> List.filter ~f:(fun id' ->
              (not (List.mem rest id' ~equal:SId.equal))
              && not
                   (List.exists collected ~f:(fun d ->
                        match d.f_sum with
                        | None -> false
                        | Some s -> List.mem s.s_scope_declares id' ~equal:SId.equal)))
        in
        closure (rest @ more_ids) (collected @ fresh)
  in
  match closure ext_ids [] with
  | [] -> Some []
  | defs ->
      let defs = List.sort defs ~compare:(fun a b -> Int.compare a.f_index b.f_index) in
      let valid =
        List.for_all defs ~f:(fun d ->
            match d.f_sum with None -> false | Some s -> Set.is_empty s.s_writes)
        &&
        let def_reads =
          List.fold defs
            ~init:(Set.empty (module Tn))
            ~f:(fun acc d -> match d.f_sum with None -> acc | Some s -> Set.union acc s.s_reads)
        in
        let earliest = (List.hd_exn defs).f_index in
        Array.for_all units ~f:(fun u ->
            u.f_index <= earliest || u.f_index >= seg_start
            ||
            match u.f_sum with
            | None -> false
            | Some s -> Set.is_empty (Set.inter s.s_writes def_reads))
      in
      if valid then Some defs else None

(* External scope ids a segment references: read or written by its units but not declared by an
   earlier unit of the same segment (statement order is preserved, so any in-segment declaration
   precedes in-segment uses of well-formed code). *)
let seg_external_ids seg =
  let declared =
    List.concat_map seg.g_units ~f:(fun u ->
        match u.f_sum with None -> [] | Some s -> s.s_scope_declares)
  in
  List.concat_map seg.g_units ~f:(fun u ->
      match u.f_sum with None -> [] | Some s -> s.s_scope_reads @ s.s_scope_writes)
  |> List.filter ~f:(fun id -> not (List.mem declared id ~equal:Low_level.equal_scope_id))
  |> List.dedup_and_sort ~compare:Low_level.compare_scope_id

(* Resolve scope-local crossings: per segment, either a replica plan or a forced merge of the
   def..use segment range (the merged segment runs serially — exactly today's behavior for that
   region). Restarts after each merge; terminates because the segment count strictly decreases. *)
let rec resolve_scope_crossings (units : funit array) (segs : segment list) :
    segment list * funit list list =
  let plans =
    List.map segs ~f:(fun seg ->
        let ext = seg_external_ids seg in
        if List.is_empty ext then `Replicas []
        else
          let seg_start = (List.hd_exn seg.g_units).f_index in
          match plan_replicas units ~seg_start ext with
          | Some defs -> `Replicas defs
          | None -> `Merge_back seg_start)
  in
  match List.findi plans ~f:(fun _ -> function `Merge_back _ -> true | `Replicas _ -> false) with
  | None ->
      (segs, List.map plans ~f:(function `Replicas defs -> defs | `Merge_back _ -> assert false))
  | Some (j, _) ->
      (* Merge from the segment holding the earliest def through segment [j]. We do not know the def
         segment without re-running the failed plan; merging [j] with its predecessor is the minimal
         step that makes progress and re-checks. *)
      if j = 0 then raise Unfissionable
      else
        let before = List.take segs (j - 1) in
        let merged =
          match List.drop segs (j - 1) with
          | a :: b :: rest -> merge_segs ~kind:`Solo a b :: rest
          | _ -> assert false
        in
        resolve_scope_crossings units (before @ merged)

(* Per segment (units + replicas): the (reads, writes) tensor-node footprint. *)
let seg_footprints (segs_with_replicas : (segment * funit list) list) :
    (Set.M(Tn).t * Set.M(Tn).t) list =
  List.map segs_with_replicas ~f:(fun (seg, replicas) ->
      List.fold (replicas @ seg.g_units)
        ~init:(Set.empty (module Tn), Set.empty (module Tn))
        ~f:(fun (r, w) u ->
          match u.f_sum with
          | None -> (r, w)
          | Some s -> (Set.union r s.s_reads, Set.union w s.s_writes)))

let crosses_segments segs_with_replicas tn =
  List.count (seg_footprints segs_with_replicas) ~f:(fun (r, w) -> Set.mem r tn || Set.mem w tn)
  >= 2

(* Promote [Local]-placed scratch whose accesses (including replicas') span segments: kernel-local
   arrays do not survive a launch boundary. Runs before per-segment schedules are computed, so the
   annotator sees the promoted (materialized) status consistently — the converse order would let a
   segment annotate without covering the node's writes. Returns the undo list: coalescing may later
   remove a crossing, and a promotion without a surviving crossing must be restored (it would
   otherwise leak an observable placement change out of an all-serial routine). *)
let promote_crossing plc (segs_with_replicas : (segment * funit list) list) :
    (Tn.t * (Tn.memory_mode * int) option) list =
  let footprints = seg_footprints segs_with_replicas in
  let written =
    List.fold footprints ~init:(Set.empty (module Tn)) ~f:(fun acc (_, w) -> Set.union acc w)
  in
  let touched = List.map footprints ~f:(fun (r, w) -> Set.union r w) in
  Set.fold written ~init:[] ~f:(fun undo tn ->
      if
        (not (Tn.Placements.is_materialized_peek plc tn))
        && List.count touched ~f:(fun t -> Set.mem t tn) >= 2
      then (
        let prior = Tn.Placements.raw_entry plc tn in
        Tn.Placements.promote_local_to_device plc tn 177;
        (tn, prior) :: undo)
      else undo)

(* All tensor nodes a segment's code references — including scope ids' backing tnodes, so the
   filtered traced store retains every entry codegen consults — and whether it reads the merge
   buffer. *)
let code_footprint (llc : Low_level.t) : Set.M(Tn).t * bool =
  let open Low_level in
  let tns = ref (Set.empty (module Tn)) and merge = ref false in
  let add tn = tns := Set.add !tns tn in
  let rec code llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier -> ()
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; _ } ->
        add d_tn;
        add a_tn;
        add b_tn
    | Declare_local { id; _ } -> add id.tn
    | Seq (a, b) ->
        code a;
        code b
    | For_loop { body; _ } -> code body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c ->
            add c.prev.tn;
            scalar c.init);
        code body
    | Zero_out tn -> add tn
    | Set { tn; llsc; _ } ->
        add tn;
        scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        add tn;
        scalar v;
        scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        add tn;
        scalar a
    | Set_local (id, llsc) ->
        add id.tn;
        scalar llsc
    | If { cond = c, _; body } ->
        scalar c;
        code body
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        add id.tn;
        code body
    | Get_local id -> add id.tn
    | Get (tn, _) -> add tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        add tn;
        scalar v
    | Get_merge_buffer (tn, _) ->
        add tn;
        merge := true
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
  in
  code llc;
  (!tns, !merge)

let segment_optimized (full : Low_level.optimized) (llc : Low_level.t) : Low_level.optimized =
  let tns, reads_merge = code_footprint llc in
  {
    Low_level.traced_store =
      Hashtbl.filteri full.Low_level.traced_store ~f:(fun ~key ~data:_ -> Set.mem tns key);
    optimize_ctx = full.Low_level.optimize_ctx;
    llc;
    merge_node = (if reads_merge then full.Low_level.merge_node else None);
    workgroup_shared = Set.filter full.Low_level.workgroup_shared ~f:(Set.mem tns);
    simdgroup_fragments = Set.filter full.Low_level.simdgroup_fragments ~f:(Set.mem tns);
    swizzled = Map.filter_keys full.Low_level.swizzled ~f:(Set.mem tns);
    pipelined = Map.filter_keys full.Low_level.pipelined ~f:(Set.mem tns);
    zero_fringe = Set.filter full.Low_level.zero_fringe ~f:(Set.mem tns);
    flip_candidates = full.Low_level.flip_candidates;
    spliced_rbw = Set.filter full.Low_level.spliced_rbw ~f:(Set.mem tns);
  }

(* Expand-and-annotate schedule for a segment of materialized whole-node [Zero_out]s (GPU): the
   expanded nests get the same geometry policy as {!default_gpu}'s chains. Below [min_parallel]
   (largest node) the zeros stay whole-node — a serial kernel renders them as [memset]. *)
let zero_expansion ?block_size ?min_parallel ~(limits : Backend_intf.hardware_limits)
    (tns : Tn.t list) : schedule =
  let block_size =
    Option.value block_size
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
  in
  let block_size = clamp_block_size ~limits block_size in
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"64")
  in
  let dims tn = Lazy.force tn.Tn.dims in
  let numel tn = Array.fold (dims tn) ~init:1 ~f:( * ) in
  if
    List.exists tns ~f:(fun tn -> Array.is_empty (dims tn))
    (* A rank-0 expansion is a bare write: uncoverable in a multi-threaded kernel. *)
    || List.fold tns ~init:0 ~f:(fun m tn -> max m (numel tn)) < min_parallel
  then []
  else
    List.concat_map tns ~f:(fun tn ->
        let op, syms = expand_zero ~tn in
        let ds = dims tn in
        let annots =
          match syms with
          | [] -> assert false
          | [ s0 ] ->
              let n0 = ds.(0) in
              let sp, _, _ =
                split ~axis:s0 ~factor:(min block_size n0) ~outer:Low_level.Grid
                  ~inner:Low_level.Workgroup
              in
              [ sp ]
          | s0 :: s1 :: _ ->
              let n1 = ds.(1) in
              if n1 <= block_size then
                [
                  Retype { axis = s0; ty = Low_level.Grid };
                  Retype { axis = s1; ty = Low_level.Workgroup };
                ]
              else
                let sp, _, _ =
                  split ~axis:s1 ~factor:block_size ~outer:Low_level.Serial
                    ~inner:Low_level.Workgroup
                in
                [ Retype { axis = s0; ty = Low_level.Grid }; sp ]
        in
        op :: annots)

let seg_llc replicas seg =
  Low_level.unflat_lines (List.concat_map (replicas @ seg.g_units) ~f:(fun u -> u.f_stmts))

(* Statement-crossing [Local] intermediates: a nest whose only writes land in [Local] scratch gets
   no parallel chain — the annotator's coverage property quantifies over {e materialized} writes —
   and Local producer/consumer edges deliberately do not cut, so such producers either drag their
   whole segment down to a serial 1x1 launch (when no materialized-writing nest shares it, e.g. a
   softmax max/denominator pair feeding a scalar loss) or are redundantly re-executed by every
   hardware thread of an annotated kernel (e.g. layer-norm statistics). Small reduction
   intermediates land exactly here: [most_local_materialized_mode] keeps any node under the stack
   threshold [Local]. Promoting the statement-crossing ones to [On_device] up front lets the
   ordinary materialized machinery apply — chains qualify, [mat_conflict] cuts, aligned merges keep
   single kernels — with the fission boundary supplying the synchronization. Within-statement
   scratch is untouched. Returns undo entries; the caller restores promotions that fission did not
   end up needing (see the undo filter and the fallback paths — restoring is sound because schedules
   computed under the stricter materialized view remain valid for a [Local] node, cf.
   {!Tn.Placements.raw_entry}). *)
let promote_statement_crossing_locals plc (stmts : Low_level.t list) :
    (Tn.t * (Tn.memory_mode * int) option) list =
  let summaries = List.map stmts ~f:(summarize_stmt plc) in
  let footprints = List.map stmts ~f:(fun s -> fst (code_footprint s)) in
  let crossing tn i = List.existsi footprints ~f:(fun j fp -> j <> i && Set.mem fp tn) in
  List.foldi summaries ~init:[] ~f:(fun i undo -> function
    | None -> undo
    | Some s ->
        Set.fold s.s_writes ~init:undo ~f:(fun undo tn ->
            let eligible =
              match Tn.Placements.get plc tn with
              | Some ((Virtual | Effectively_constant | On_device), _) -> false
              | Some ((Local | Never_virtual), _) | None ->
                  not (Tn.Placements.is_materialized_peek plc tn)
            in
            if eligible && crossing tn i then (
              let prior = Tn.Placements.raw_entry plc tn in
              Tn.Placements.promote_local_to_device plc tn 178;
              (tn, prior) :: undo)
            else undo))

let fission_scheduled ?(promote_locals = false) ?(arity_cuts = false)
    ~(preset : Low_level.optimized -> schedule) ~(zero_sched : Tn.t list -> schedule)
    ~static_indices (opt : Low_level.optimized) :
    ([ `Normal | `Zeros | `Solo ] * Low_level.optimized * schedule * Low_level.optimized) list =
  let plc = opt.Low_level.optimize_ctx.placements in
  let stmts = Low_level.flat_lines [ opt.Low_level.llc ] in
  let pre_promoted = if promote_locals then promote_statement_crossing_locals plc stmts else [] in
  let fallback () =
    (* Single-kernel compilation, exactly as before fission: no boundary needs the promotions, and
       placement changes must not leak out of an unfissioned routine. *)
    List.iter pre_promoted ~f:(fun (tn, prior) -> Tn.Placements.unsafe_restore plc tn prior);
    let sched = preset opt in
    [ (`Normal, opt, sched, apply ~static_indices sched opt) ]
  in
  (* The [arity_cuts] mode analyzes chains uncapped: the no-loss guard (and the uniform-shape rule)
     must see each nest's full arity, not the default presets' Grid+Workgroup prefix — under the
     cap, a merge that trims a rank-3 site's minor axis reads as lossless (gh-ocannl-574). *)
  let max_chain = if arity_cuts then Some Int.max_value else None in
  let units = collect_units ?max_chain plc opt stmts in
  let segs = group_units ?max_chain ~arity_cuts opt units in
  if List.length segs <= 1 then fallback ()
  else
    match resolve_scope_crossings (Array.of_list units) segs with
    | exception Unfissionable -> fallback ()
    | segs, replicas when List.length segs <= 1 ->
        ignore replicas;
        fallback ()
    | segs, replicas ->
        let segs_with_replicas = List.zip_exn segs replicas in
        let promoted = pre_promoted @ promote_crossing plc segs_with_replicas in
        let undo_promotions which =
          List.iter which ~f:(fun (tn, prior) -> Tn.Placements.unsafe_restore plc tn prior)
        in
        let scheduled =
          List.map segs_with_replicas ~f:(fun (seg, replicas) ->
              let sched =
                match seg.g_kind with
                | `Solo -> []
                | `Zeros ->
                    zero_sched
                      (List.filter_map seg.g_units ~f:(fun u ->
                           Option.bind u.f_sum ~f:(fun s -> s.s_top_zero)))
                | `Normal -> preset (segment_optimized opt (seg_llc replicas seg))
              in
              (seg, replicas, sched))
        in
        (* Coalesce adjacent unannotated segments: consecutive serial kernels gain nothing from a
           launch boundary. Merged segments are rebuilt from original units and their replicas
           recomputed for the new boundary (the def..start gap only shrinks under merging, so
           feasibility is preserved). *)
        let coalesced =
          List.fold scheduled ~init:[] ~f:(fun acc (seg, replicas, sched) ->
              match acc with
              | (pseg, _, []) :: rest when List.is_empty sched ->
                  let merged = merge_segs ~kind:`Solo pseg seg in
                  let replicas =
                    match
                      plan_replicas (Array.of_list units)
                        ~seg_start:(List.hd_exn merged.g_units).f_index (seg_external_ids merged)
                    with
                    | Some defs -> defs
                    | None -> assert false (* Merging only shrinks the validity range. *)
                  in
                  (merged, replicas, []) :: rest
              | _ -> (seg, replicas, sched) :: acc)
          |> List.rev
        in
        if List.length coalesced <= 1 then (
          (* Everything merged back: single kernel, exactly as before fission — including
             placements, so undo every promotion (an all-serial small routine must not leak
             observable placement changes; zero2hero's virtual-neuron printouts pinned this). *)
          undo_promotions promoted;
          fallback ())
        else
          (* Coalescing may have absorbed a crossing: promotions without a surviving crossing are
             restored. Sound in this direction — the segments' schedules were computed under the
             stricter materialized view (see [promote_crossing]). *)
          let final_swr = List.map coalesced ~f:(fun (seg, replicas, _) -> (seg, replicas)) in
          undo_promotions
            (List.filter promoted ~f:(fun (tn, _) -> not (crosses_segments final_swr tn)));
          List.map coalesced ~f:(fun (seg, replicas, sched) ->
              let pre = segment_optimized opt (seg_llc replicas seg) in
              (seg.g_kind, pre, sched, apply_classified ~static_indices sched pre))

let fission_default ?promote_locals ~preset ~zero_sched ~static_indices (opt : Low_level.optimized)
    : Low_level.optimized list =
  List.map (fission_scheduled ?promote_locals ~preset ~zero_sched ~static_indices opt)
    ~f:(fun (_kind, _pre, _sched, post) -> post)

(** {2 Wiring: the implicit transform for GPU and CPU backends} *)

let automatic_gpu_schedule =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"automatic_gpu_schedule")

let automatic_cpu_schedule =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"automatic_cpu_schedule")

let backend_is_gpu name =
  String.is_substring name ~substring:"cuda"
  || String.is_substring name ~substring:"hip"
  || String.is_substring name ~substring:"metal"

let backend_is_cpu name = String.is_substring name ~substring:"cc"
let schedule_fission = lazy (Utils.get_global_flag ~default:true ~arg_name:"schedule_fission")

(* Per-compile launch-geometry trace on stderr; consumed by backend [compile]. *)
let log_launches = lazy (Utils.get_global_flag ~default:false ~arg_name:"schedule_log_launches")

let automatic_schedule_active ~backend_name =
  (not (Utils.debug_log_from_routines ()))
  && ((backend_is_gpu backend_name && Lazy.force automatic_gpu_schedule)
     || (backend_is_cpu backend_name && Lazy.force automatic_cpu_schedule))

let default_pipeline_fissions () = Lazy.force schedule_fission

let default_schedule_fingerprint ~backend_name =
  if not (automatic_schedule_active ~backend_name) then "inactive"
  else
    let fission = Lazy.force schedule_fission in
    if backend_is_gpu backend_name then
      let bs =
        String.strip (Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
      in
      let mp =
        String.strip (Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"64")
      in
      [%string "gpu:fission=%{fission#Bool}:block_size=%{bs}:min_parallel=%{mp}"]
    else
      let mp =
        String.strip (Utils.get_global_arg ~arg_name:"cpu_schedule_min_parallel" ~default:"16384")
      in
      [%string "cpu:fission=%{fission#Bool}:min_parallel=%{mp}"]

let maybe_default_schedule ~backend_name ?(limits = Backend_intf.no_hardware_limits) ~static_indices
    (opt : Low_level.optimized) : Low_level.optimized =
  (* [automatic_schedule_active] keeps logged runs serial: runtime kernel logging is
     line-interleaved under parallel execution, and serial logs stay deterministic and readable. *)
  if not (automatic_schedule_active ~backend_name) then opt
  else if backend_is_gpu backend_name then
    apply_classified ~static_indices (default_gpu ~limits opt) opt
  else apply_classified ~static_indices (default_cpu opt) opt

let maybe_default_schedules ~backend_name ?(limits = Backend_intf.no_hardware_limits)
    ~static_indices (opt : Low_level.optimized) : Low_level.optimized list =
  if not (automatic_schedule_active ~backend_name) then [ opt ]
  else
    let gpu = backend_is_gpu backend_name in
    if not (Lazy.force schedule_fission) then
      [ maybe_default_schedule ~backend_name ~limits ~static_indices opt ]
    else
      let preset o = if gpu then default_gpu ~limits o else default_cpu o in
      (* CPU zero segments stay whole-node: a serial kernel renders them as [memset], which is hard
         to beat below many-megabyte sizes; parallel zero expansion on CPU is a follow-up. *)
      let zero_sched tns = if gpu then zero_expansion ~limits tns else [] in
      (* Statement-crossing [Local]s are promoted on GPU only: a serial (or per-thread redundant)
         producer nest costs little next to CPU cores but is catastrophic next to GPU threads, and
         keeping CPU placements unchanged keeps small-routine codegen stable. *)
      fission_default ~promote_locals:gpu ~preset ~zero_sched ~static_indices opt

(** {2 The launch-geometry predicate (gh-ocannl-709)}

    One static reading of what a device permits of a launch's {e geometry}, so the pre-driver gate
    below and autotune's seeding pre-filter ({!Autotune.Sketch_families}) cannot disagree about a
    cap. Before this the two encoded the caps independently: the gate covered five dimensions and
    the seeder pre-filtered exactly one of them, which is how a search could only learn the other
    four one wasted compile at a time. The caps live here; what a caller supplies is a geometry, and
    the two callers differ only in where their geometry comes from — the gate reads it off the
    lowered code ({!Low_level.launch_dims}), the seeder predicts it from the parameters it is about
    to commit to. *)

type launch_geometry = {
  lg_grid_y : int option;
  lg_grid_z : int option;
  lg_block_x : int option;
  lg_block_y : int option;
  lg_block_z : int option;
}
(** As much of a candidate's launch geometry as its caller knows; [None] is "not predicted here",
    and exempts that dimension rather than refusing on it. Five fields, not six: [grid.(0)] is
    2^31-scale on every backend that binds hardware axes, so no backend has a cap to report for it
    (the same reason the gate's table has five rows). A seeder's prediction is a {e lower bound} —
    it describes the site's own nest, while [Low_level.launch_dims] maxes over the kernel's zeroing
    and companion nests too — so an under-prediction costs a compile the gate then declines, and
    only an over-prediction could withhold a legal candidate. *)

let unknown_launch_geometry =
  { lg_grid_y = None; lg_grid_z = None; lg_block_x = None; lg_block_y = None; lg_block_z = None }

let launch_geometry_of_dims (dims : Low_level.launch_dims) =
  {
    lg_grid_y = Some dims.grid.(1);
    lg_grid_z = Some dims.grid.(2);
    lg_block_x = Some dims.block.(0);
    lg_block_y = Some dims.block.(1);
    lg_block_z = Some dims.block.(2);
  }

type launch_excess = {
  lx_resource : Schedule_outcome.resource;
  lx_requested : int;
  lx_limit : int;
  lx_phrase : string;
}
(** The first dimension of a geometry the device refuses. [lx_phrase] is the verb phrase both
    callers render — "requests a .z workgroup extent of 128, exceeding the device limit of 64" — so
    the gate's [detail] and the seeder's refutation witness say the same thing about the same
    candidate, and a reader comparing a decline log against a refutation log sees one sentence. *)

let launch_geometry_excess ~(limits : Backend_intf.hardware_limits) (geom : launch_geometry) :
    launch_excess option =
  let wg f = Option.map limits.max_workgroup_dims ~f in
  let requests what requested limit =
    [%string "requests a %{what} of %{requested#Int}, exceeding the device limit of %{limit#Int}"]
  in
  (* One row per hardware dimension, enumerated rather than hand-written per bound: an ungated
     dimension is a missing ROW, visible beside its neighbours, instead of an absence. That is how
     [gridDim.y] came to be ungated for a release (gh-ocannl-643 gated the fold,
     lukstafi/ocannl-staging#397 added the row blocks) and how the workgroup's per-dimension caps
     came to be missing entirely (gh-ocannl-679). *)
  let rows =
    [
      (* The workgroup's own dimensions: a separate hardware fact from the thread PRODUCT cap
         ([max_threads_per_workgroup], checked by the gate alone since it is not a geometry
         question), and CUDA's [.z] cap of 64 sits 16x below its product cap, so a legal-product
         workgroup with a deep [.z] passes every other check and dies at the driver. [Workgroup]
         slots are capped at 3, so these three rows are exhaustive. *)
      ( wg (fun (x, _, _) -> x),
        geom.lg_block_x,
        Schedule_outcome.Workgroup_x_extent,
        requests ".x workgroup extent" );
      ( wg (fun (_, y, _) -> y),
        geom.lg_block_y,
        Schedule_outcome.Workgroup_y_extent,
        requests ".y workgroup extent" );
      ( wg (fun (_, _, z) -> z),
        geom.lg_block_z,
        Schedule_outcome.Workgroup_z_extent,
        requests ".z workgroup extent" );
      (* [.y] is the grid slot-1 extent — the row-block count of a blocktiled matmul, which grows
         with the site's m-extent rather than with any fold: at [bm = 16] an m-extent past ~1M rows
         is already over the cap. *)
      (limits.max_grid_yz, geom.lg_grid_y, Schedule_outcome.Grid_y_extent, requests ".y grid extent");
      (* The [.z] grid fold (gh-ocannl-643) multiplies every Grid slot [>= 2] into [grid.(2)]. *)
      ( limits.max_grid_yz,
        geom.lg_grid_z,
        Schedule_outcome.Grid_z_extent,
        fun requested limit ->
          [%string
            "folds grid slots >= 2 to a .z extent of %{requested#Int}, exceeding the device limit \
             of %{limit#Int}"] );
    ]
  in
  List.find_map rows ~f:(fun (cap, requested, resource, phrase) ->
      match (cap, requested) with
      | Some limit, Some requested when requested > limit ->
          Some
            {
              lx_resource = resource;
              lx_requested = requested;
              lx_limit = limit;
              lx_phrase = phrase requested limit;
            }
      | _ -> None)

let check_hardware_limits_classified ~name ~(limits : Backend_intf.hardware_limits)
    (opt : Low_level.optimized) : unit =
  Option.iter limits.max_threads_per_workgroup ~f:(fun max_threads ->
      let block = (Low_level.launch_dims opt.llc).block in
      let block_product = Array.fold block ~init:1 ~f:( * ) in
      if block_product > max_threads then
        let detail =
          [%string
            "Schedule: kernel %{name} requests a workgroup of %{block_product#Int} threads, \
             exceeding the device limit of %{max_threads#Int} threads per workgroup"]
        in
        raise
          (Schedule_outcome.Cause_at
             ( Schedule_outcome.Hardware_limits,
               Schedule_outcome.Resource_exceeded
                 {
                   resource = Schedule_outcome.Workgroup_threads;
                   requested = block_product;
                   limit = Some max_threads;
                   detail;
                 } )));
  Option.iter limits.max_workgroup_memory_bytes ~f:(fun max_bytes ->
      let shared_bytes =
        Set.fold opt.workgroup_shared ~init:0 ~f:(fun acc tn ->
            (* A pipelined tile is allocated as [pt_depth] rotating copies (gh-ocannl-487): the
               IR-level dims stay single-copy, so the accounting multiplies here, matching the
               codegen declaration. *)
            let copies =
              match Map.find opt.Low_level.pipelined tn with
              | Some { Low_level.pt_depth; _ } -> pt_depth
              | None -> 1
            in
            acc + (copies * Lazy.force tn.Tn.size_in_bytes))
      in
      if shared_bytes > max_bytes then
        let detail =
          [%string
            "Schedule: kernel %{name} stages %{shared_bytes#Int} bytes of workgroup-shared tiles, \
             exceeding the device limit of %{max_bytes#Int} bytes"]
        in
        raise
          (Schedule_outcome.Cause_at
             ( Schedule_outcome.Hardware_limits,
               Schedule_outcome.Resource_exceeded
                 {
                   resource = Schedule_outcome.Workgroup_memory;
                   requested = shared_bytes;
                   limit = Some max_bytes;
                   detail;
                 } )));
  (* {3 The launch geometry}

     Delegated to {!launch_geometry_excess} — the one static reading of the device's per-dimension
     caps, which autotune's seeding pre-filter consults on its predicted geometry (gh-ocannl-709) so
     the two cannot disagree about what a device permits. What is local to the gate is only where
     the geometry comes from: the lowered code.

     [validate_parallel] deliberately accepts any launch geometry (it is backend-independent), so
     this is where an over-cap launch is refused before it reaches the driver — covering hand-built
     schedules and future annotators, not only the autotune seeds (which pre-filter against the same
     predicate, and whose block extents the annotators clamp against the same
     [max_workgroup_dims]). *)
  Option.iter
    (launch_geometry_excess ~limits (launch_geometry_of_dims (Low_level.launch_dims opt.llc)))
    ~f:(fun { lx_resource; lx_requested; lx_limit; lx_phrase } ->
      raise
        (Schedule_outcome.Cause_at
           ( Schedule_outcome.Hardware_limits,
             Schedule_outcome.Resource_exceeded
               {
                 resource = lx_resource;
                 requested = lx_requested;
                 limit = Some lx_limit;
                 detail = [%string "Schedule: kernel %{name} %{lx_phrase}"];
               } )))

let check_hardware_limits ~name ~limits opt =
  match check_hardware_limits_classified ~name ~limits opt with
  | () -> ()
  | exception Schedule_outcome.Cause_at (_, cause) -> Schedule_outcome.raise_cause cause
