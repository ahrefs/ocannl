(* gh-ocannl-1603: outlining a large kernel -- its top-level statements packed in order into parts,
   each a function the kernel calls in turn -- for {!C_syntax}, whose host C backend renders it.
   Documented in kernel_outline.mli. *)

open Base

(* The size a part's weight measures a statement by: its statement and scalar-expression nodes, a
   proxy for the length of the C it renders to. Measured on the lowered code rather than on the
   rendered text because rendering is a single stateful pass -- a kernel is outlined or not before
   it is rendered, never re-rendered. *)
let rec ir_weight (llc : Low_level.t) : int =
  match llc with
  | Noop -> 0
  | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> 1
  | Seq (a, b) -> ir_weight a + ir_weight b
  | For_loop { body; _ } -> 1 + ir_weight body
  | Scan_loop { carried; body; _ } ->
      1 + ir_weight body
      + List.sum (module Int) carried ~f:(fun c -> scalar_weight c.Low_level.init)
  | Set { llsc; _ } -> 1 + scalar_weight llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } -> 1 + scalar_weight v + scalar_weight llsc
  | Set_from_vec { arg = v, _; _ } -> 1 + scalar_weight v
  | Set_local (_, v) -> 1 + scalar_weight v
  | If { cond = c, _; body } -> 1 + scalar_weight c + ir_weight body
  | Tile_mma { fallback; _ } -> 1 + ir_weight fallback

and scalar_weight (sc : Low_level.scalar_t) : int =
  match sc with
  | Local_scope { body; _ } -> 1 + ir_weight body
  | Get_dynamic { dyn_value = v, _; _ } -> 1 + scalar_weight v
  | Ternop (_, (a, _), (b, _), (c, _)) -> 1 + scalar_weight a + scalar_weight b + scalar_weight c
  | Binop (_, (a, _), (b, _)) -> 1 + scalar_weight a + scalar_weight b
  | Unop (_, (a, _)) -> 1 + scalar_weight a
  | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> 1

(* The scope locals a statement names anywhere inside it: declared, set, read or carried. *)
let rec stmt_scope_ids acc (llc : Low_level.t) =
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> acc
  | Declare_local { id; _ } -> id.scope_id :: acc
  | Seq (a, b) -> stmt_scope_ids (stmt_scope_ids acc a) b
  | For_loop { body; _ } -> stmt_scope_ids acc body
  | Scan_loop { carried; body; _ } ->
      let acc =
        List.fold carried ~init:acc ~f:(fun acc c ->
            scalar_scope_ids
              (c.Low_level.prev.scope_id :: c.Low_level.next.scope_id :: acc)
              c.Low_level.init)
      in
      stmt_scope_ids acc body
  | Set { llsc; _ } -> scalar_scope_ids acc llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_scope_ids (scalar_scope_ids acc v) llsc
  | Set_from_vec { arg = v, _; _ } -> scalar_scope_ids acc v
  | Set_local (id, v) -> scalar_scope_ids (id.scope_id :: acc) v
  | If { cond = c, _; body } -> stmt_scope_ids (scalar_scope_ids acc c) body
  | Tile_mma { fallback; _ } -> stmt_scope_ids acc fallback

and scalar_scope_ids acc (sc : Low_level.scalar_t) =
  match sc with
  | Local_scope { id; body; _ } -> stmt_scope_ids (id.scope_id :: acc) body
  | Get_local id -> id.scope_id :: acc
  | Get_dynamic { dyn_value = v, _; _ } -> scalar_scope_ids acc v
  | Ternop (_, (a, _), (b, _), (c, _)) ->
      scalar_scope_ids (scalar_scope_ids (scalar_scope_ids acc a) b) c
  | Binop (_, (a, _), (b, _)) -> scalar_scope_ids (scalar_scope_ids acc a) b
  | Unop (_, (a, _)) -> scalar_scope_ids acc a
  | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> acc

(* A kernel is outlined only when it weighs more than this many parts: a few parts' worth compiles
   quickly whole, and a fissioned segment, which a candidate compiles by the hundred, stays one
   function. *)
let outline_factor = 16

(* A kernel's top-level statements packed in order into parts of weight at least [part] each, or
   [None] when the kernel stays whole: it weighs no more than [outline_factor] parts, it would make
   fewer than two, or a top-level statement's text is opaque to this module (a staged callback) or
   synchronizes (a barrier). A part never ends where something spans the boundary: a scope local
   named on both sides of it (declared at the top level and read by a later statement), or a zero
   initializer, which the renderer may forward into the statement after it
   ([localized_zero_seed_candidate]). *)
let groups ~part (stmts : Low_level.t list) : Low_level.t list list option =
  let total = List.sum (module Int) stmts ~f:ir_weight in
  let outlinable = function
    | Low_level.Staged_compilation _ | Workgroup_barrier -> false
    | _ -> true
  in
  if total <= outline_factor * part || not (List.for_all stmts ~f:outlinable) then None
  else
    let n = List.length stmts in
    (* [spanning.(k)]: some scope local is named both at or before statement [k] and after it. *)
    let first = Hashtbl.create (module Int) and last = Hashtbl.create (module Int) in
    List.iteri stmts ~f:(fun k stmt ->
        List.iter (stmt_scope_ids [] stmt) ~f:(fun id ->
            if not (Hashtbl.mem first id) then Hashtbl.set first ~key:id ~data:k;
            Hashtbl.set last ~key:id ~data:k));
    let opened = Array.create ~len:(n + 1) 0 in
    Hashtbl.iteri first ~f:(fun ~key:id ~data:k ->
        let l = Hashtbl.find_exn last id in
        if l > k then (
          opened.(k) <- opened.(k) + 1;
          opened.(l) <- opened.(l) - 1));
    let spanning = Array.create ~len:n false in
    ignore
      (Array.foldi spanning ~init:0 ~f:(fun k depth _ ->
           let depth = depth + opened.(k) in
           spanning.(k) <- depth > 0;
           depth)
        : int);
    let groups, current, _ =
      List.foldi stmts ~init:([], [], 0) ~f:(fun k (groups, current, weight) stmt ->
          let current = stmt :: current and weight = weight + ir_weight stmt in
          if
            weight >= part
            && (not spanning.(k))
            && Option.is_none (Low_level.zero_initializer_target stmt)
          then (List.rev current :: groups, [], 0)
          else (groups, current, weight))
    in
    let groups = List.rev (if List.is_empty current then groups else List.rev current :: groups) in
    if List.length groups < 2 then None else Some groups

let part_name ~kernel k = kernel ^ "__part" ^ Int.to_string k

(* A part is never inlined back into its kernel -- that would rebuild the one large function the
   outlining exists to avoid -- and is visible to the other translation units of its library
   only. *)
let part_attribute = "ARRAYJIT_KERNEL_PART"

(* Guarded so that every unit can carry it; PE/COFF has no hidden visibility. *)
let part_attribute_definition =
  "#ifndef ARRAYJIT_KERNEL_PART\n\
   #if defined(_WIN32)\n\
   #define ARRAYJIT_KERNEL_PART __attribute__((noinline))\n\
   #else\n\
   #define ARRAYJIT_KERNEL_PART __attribute__((noinline, visibility(\"hidden\")))\n\
   #endif\n\
   #endif"

(* The identifier a parameter declaration ([uint32_t *restrict n826]) declares: its last token. *)
let trailing_ident decl =
  let decl = String.rstrip decl in
  let start =
    match String.rfindi decl ~f:(fun _ c -> not (Char.is_alphanum c || Char.equal c '_')) with
    | Some j -> j + 1
    | None -> 0
  in
  String.drop_prefix decl start
