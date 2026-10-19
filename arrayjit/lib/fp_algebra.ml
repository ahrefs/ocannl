(* The licensed floating-point algebra as a member of the raw rewrite tier; the contract is in the
   interface. *)

open Base
module LL = Low_level
module Tn = Tnode

let enabled () = not (LL.Fp_licence.equal (LL.Fp_licence.of_config ()) LL.Fp_licence.exact)

let rewrite (licence : LL.Fp_licence.t) (llc : LL.t) : LL.t =
  let rec scalar ((llsc, prec) : LL.scalar_t * Ops.prec) : LL.scalar_t =
    let llsc : LL.scalar_t =
      match llsc with
      | Constant _ | Constant_bits _ | Get _ | Get_local _ | Get_merge_buffer _ | Embed_index _ ->
          llsc
      | Get_dynamic ({ dyn_value = v, vprec; _ } as gd) ->
          Get_dynamic { gd with dyn_value = (scalar (v, vprec), vprec) }
      | Local_scope ({ body; _ } as scope) -> Local_scope { scope with body = proc body }
      | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar (a, pa), pa), (scalar (b, pb), pb))
      | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
          Ternop (op, (scalar (a, pa), pa), (scalar (b, pb), pb), (scalar (c, pc), pc))
      | Unop (op, (a, pa)) -> Unop (op, (scalar (a, pa), pa))
    in
    (* Floating roots only: the exact canonicalizer owns the integer arms ([simplify_llc]). *)
    if not (Ops.is_float prec) then llsc
    else
      match LL.fp_algebra_step licence (llsc, prec) with
      | Some rewritten -> scalar rewritten
      | None -> llsc
  and proc (llc : LL.t) : LL.t =
    match llc with
    | Noop | Zero_out _ | Declare_local _ | Comment _ | Staged_compilation _ | Workgroup_barrier ->
        llc
    | Seq (c1, c2) -> Seq (proc c1, proc c2)
    | For_loop ({ body; _ } as fc) -> For_loop { fc with body = proc body }
    | Scan_loop ({ carried; body; _ } as sc) ->
        Scan_loop
          {
            sc with
            carried =
              List.map carried ~f:(fun c ->
                  { c with init = scalar (c.init, Lazy.force c.prev.tn.Tn.storage_prec) });
            body = proc body;
          }
    | If { cond = c, cprec; body } -> If { cond = (scalar (c, cprec), cprec); body = proc body }
    | Tile_mma ({ fallback; _ } as tm) -> Tile_mma { tm with fallback = proc fallback }
    | Set ({ tn; llsc; _ } as set) ->
        Set { set with llsc = scalar (llsc, Lazy.force tn.Tn.storage_prec) }
    | Set_dynamic ({ tn; dyn_value = v, vprec; llsc; _ } as set) ->
        Set_dynamic
          {
            set with
            dyn_value = (scalar (v, vprec), vprec);
            llsc = scalar (llsc, Lazy.force tn.Tn.storage_prec);
          }
    | Set_from_vec ({ arg = a, aprec; _ } as set) ->
        Set_from_vec { set with arg = (scalar (a, aprec), aprec) }
    | Set_local (id, llsc) -> Set_local (id, scalar (llsc, Lazy.force id.tn.Tn.storage_prec))
  in
  proc llc
