(* gh-ocannl-1462: the upload-arena ownership invariant that [Backends.finalize] checks under
   [debug_check_upload_arenas]. Every tenant a retained arena lists must be held, at its own pool
   and offset, by a registered owner of the arena's pool ([Backend_intf.holds_tenant]); an arena
   with no tenant or no owner list is a violation too.

   The checker is exercised here on hand-built lineage state, since no public API can corrupt a
   lineage's arenas: a holding owner passes, while an owner holding the node only at ANOTHER
   location (the gh-ocannl-1328 confusion), a pool with no owner list and an empty arena are each
   reported. That [finalize] runs it is exercised by [set_values_pool_coalescing], which runs with
   [debug_check_upload_arenas=true] over every packing, fork and trim scenario of both transfer
   paths. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf

let node = Ll_test.node_factory ~first_id:14620 ~dims:[| 4 |] ()
let loc pool_id offset = { BI.pool_id; offset }

let arena pool_id tenants =
  {
    BI.arena_pool_id = pool_id;
    arena_mode = None;
    capacity = 64;
    tenants =
      List.map tenants ~f:(fun (tenant, tenant_offset) ->
          { BI.tenant; tenant_offset; bump = tenant_offset + 16 });
  }

let lineage arenas owners =
  let u = BI.fresh_upload_arenas () in
  u.arenas <- arenas;
  List.iter owners ~f:(fun (pool_id, maps) ->
      Hashtbl.set u.pool_owners ~key:pool_id ~data:(List.map maps ~f:ref));
  u

let buffers entries =
  Map.of_alist_exn (module Ir.Tnode) (List.map entries ~f:(fun (tn, l) -> (tn, l)))

let () =
  Verdict.case "uasc" @@ fun () ->
  let a = node "uasc_a" and b = node "uasc_b" in
  (* Newest first, as [upload_slot] keeps them: [b] was appended after [a]. *)
  let both = arena 7 [ (b, 16); (a, 0) ] in
  let holder = buffers [ (a, loc 7 0); (b, loc 7 16) ] in
  let violations u = BI.upload_arena_violations u in
  p_empty "uasc: an owner holding every tenant at its location is no violation" ~over:both.tenants
    (violations (lineage [ both ] [ (7, [ holder ]) ]));
  p_empty "uasc: one of several owners holding them suffices" ~over:both.tenants
    (violations (lineage [ both ] [ (7, [ buffers [ (a, loc 7 0) ]; holder ]) ]));
  (* The surviving owner holds [b] -- but at another location, a sibling leaf's upload of it. *)
  let elsewhere = buffers [ (a, loc 7 0); (b, loc 9 0) ] in
  let v = violations (lineage [ both ] [ (7, [ elsewhere ]) ]) in
  pf "uasc: a tenant whose node the owner holds only elsewhere is exactly one violation"
    (List.length v = 1);
  pf "uasc: the violation names the tenant, its pool and its offset"
    (List.exists v ~f:(fun m ->
         String.is_substring m ~substring:"uasc_b"
         && String.is_substring m ~substring:"pool 7 offset 16"));
  pf "uasc: a retained arena with no owner list is a violation"
    (match violations (lineage [ both ] []) with
    | [ m ] -> String.is_substring m ~substring:"no registered owner"
    | _ -> false);
  pf "uasc: a retained arena with no tenant is a violation"
    (match violations (lineage [ arena 8 [] ] [ (8, [ holder ]) ]) with
    | [ m ] -> String.is_substring m ~substring:"retains no tenant"
    | _ -> false)
