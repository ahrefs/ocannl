(* gh-ocannl-2452: [Ll_test.execute_with_census] (and [Ll_test.execute] / [optimize_and_execute]
   built on it) must read back only buffers stored in the context, never recompute a [Virtual] node
   on read, never return an externally seeded buffer for a [Virtual] node the routine never wrote,
   and never return unregistered-buffer [Host_init] data.

   Refusal controls beside positive controls, with discriminating producers: - Virtual
   (Recomputable): raw [Context.get_values] recomputes the exact values, while
   [execute_with_census], [execute], [optimize_and_execute], and mixed [~read:[ out; v ]] refuse. -
   Seeded Virtual: seeding a Virtual node gives [Context.observability] a [Stored] answer backed
   only by the uploaded sentinel; raw [Context.get_values] hands back the sentinel while
   [execute_with_census] and [execute] still refuse. - Host_init: a node with registered host-init
   data and no context buffer is refused instead of handing back a copy of its host-init array. -
   Local: preserves [Context.get_values]'s [User_error] and [Ll_test.refused_as_local]. - Positive
   controls: materialized [out], the same program re-optimized with [~materialized:[ v ]], and an
   empty [~read:[]] all succeed and match exact cell-by-cell expected values. *)

open Base
open Verdict.Claims
module L = Ll_test
module Tn = Ir.Tnode

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

let refusal_message f =
  try
    ignore (f ());
    None
  with Invalid_argument msg -> Some msg

let refused_with ~substrings f =
  (not (List.is_empty substrings))
  &&
  match refusal_message f with
  | None -> false
  | Some msg -> List.for_all substrings ~f:(fun s -> String.is_substring msg ~substring:s)

let user_error_with ~substrings f =
  (not (List.is_empty substrings))
  &&
  match f () with
  | _ -> false
  | exception Utils.User_error msg ->
      List.for_all substrings ~f:(fun s -> String.is_substring msg ~substring:s)

(* Pin heuristic virtualization settings while optimizing fixtures so ambient overrides (such as
   [OCANNL_VIRTUALIZE_MAX_INLINE_REDUCTION]) cannot flip the fixture placements before readback is
   exercised. *)
let with_pinned_virtualize_settings f =
  let s = Ir.Low_level.virtualize_settings in
  let saved_enable = s.enable_device_only
  and saved_visits = s.max_visits
  and saved_red = s.max_inline_reduction
  and saved_fanin = s.max_inline_fanin in
  s.enable_device_only <- true;
  s.max_visits <- 2;
  s.max_inline_reduction <- 16;
  s.max_inline_fanin <- 8;
  Exn.protect ~f ~finally:(fun () ->
      s.enable_device_only <- saved_enable;
      s.max_visits <- saved_visits;
      s.max_inline_reduction <- saved_red;
      s.max_inline_fanin <- saved_fanin)

let n = 4
let node = L.node_factory ~first_id:2452000 ~dims:[| n |] ()
let xv = [| 10.; 20.; 30.; 40. |]
let v_expected = [| 11.; 22.; 33.; 44. |]
let out_expected = [| 22.; 44.; 66.; 88. |]

(* [v[i] = x[i] + (1 + i); out[i] = v[i] * 2]: every cell depends on both [x] and [i], stays clear
   of zero and [L.sentinel], and is exact in f32. *)
let make_program ~suffix =
  let x = node ("les_x_" ^ suffix) in
  let v = node ("les_v_" ^ suffix) in
  let out = node ("les_out_" ^ suffix) in
  L.materialize x;
  L.materialize out;
  let i1 = L.sym () in
  let i2 = L.sym () in
  let llc =
    L.seq
      (L.loop ~upto:(n - 1) i1
         (L.set_at v (L.iter i1) (L.add (L.get x [| L.iter i1 |]) (L.tick i1))))
      (L.loop ~upto:(n - 1) i2
         (L.set_at out (L.iter i2) (L.mul (L.get v [| L.iter i2 |]) (L.c 2.))))
  in
  (x, v, out, llc)

(* A node placed [Local] by the pipeline plus a [Virtual] node reading it: a reduction of extent 20
   exceeds the pinned [max_inline_reduction] (16), forcing [Never_virtual], and unobservable small
   storage resolves to [Local] at backend compile; [v_loc] stays [Virtual] and inherits
   [Unobservable_local] with a two-node dependency path. *)
let make_local_program () =
  let loc = node "les_loc" in
  let v_loc = node "les_v_loc" in
  let out = node "les_loc_out" in
  L.materialize out;
  let i1 = L.sym () in
  let r = L.sym () in
  let i2 = L.sym () in
  let i3 = L.sym () in
  let llc =
    L.seq
      (L.loop ~upto:(n - 1) i1
         (L.seq
            (L.set_at loc (L.iter i1) (L.c 0.))
            (L.loop ~upto:19 r
               (L.set_at loc (L.iter i1)
                  (L.add (L.get loc [| L.iter i1 |]) (L.add (L.tick i1) (L.embed r)))))))
      (L.seq
         (L.loop ~upto:(n - 1) i2
            (L.set_at v_loc (L.iter i2) (L.add (L.get loc [| L.iter i2 |]) (L.c 1.))))
         (L.loop ~upto:(n - 1) i3 (L.set_at out (L.iter i3) (L.get v_loc [| L.iter i3 |]))))
  in
  (loc, v_loc, out, llc)

let () =
  (* --- Virtual (Recomputable) refusal control --- *)
  let x, v, out, llc = make_program ~suffix:"virt" in
  let o = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"les_virt" llc) in
  p "the optimizer leaves v virtual" (L.known_virtual o v);
  let raw_ctx = L.run ~ctx:(Context.auto ()) ~name:"les_virt_raw" o ~seed:[ (x, xv) ] in
  p "raw Context.observability classifies the unseeded virtual node as Recomputable"
    (match Context.observability raw_ctx v with Context.Recomputable -> true | _ -> false);
  p_all2 "raw Context.get_values recomputes the virtual node to its exact values"
    (Context.get_values raw_ctx v) v_expected ~f:Float.equal;
  p "execute_with_census refuses reading a Virtual (Recomputable) node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual"; "recomputable" ]
       (fun () ->
         L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_virt_census" o
           ~seed:[ (x, xv) ]
           ~read:[ v ]));
  p "execute refuses reading a Virtual (Recomputable) node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual" ] (fun () ->
         L.execute ~ctx:(Context.auto ()) ~name:"les_virt_exec" o ~seed:[ (x, xv) ] ~read:[ v ]));
  p "optimize_and_execute refuses reading a Virtual (Recomputable) node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual" ] (fun () ->
         with_pinned_virtualize_settings (fun () ->
             let x2, v2, _out2, llc2 = make_program ~suffix:"opt_exec" in
             L.optimize_and_execute ~ctx:(Context.auto ()) ~name:"les_virt_opt_exec" llc2
               ~seed:[ (x2, xv) ]
               ~read:[ v2 ])));
  p "execute_with_census refuses when any node in ~read is Virtual alongside a stored node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual" ] (fun () ->
         L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_virt_mixed" o
           ~seed:[ (x, xv) ]
           ~read:[ out; v ]));

  (* --- Seeded Virtual refusal control --- *)
  let seeded_ctx =
    L.run ~ctx:(Context.auto ()) ~name:"les_seeded_raw" o ~seed:[ (x, xv); (v, L.blank n) ]
  in
  p "seeding a virtual node makes Context.observability report Stored"
    (match Context.observability seeded_ctx v with Context.Stored -> true | _ -> false);
  p_all2 "raw Context.get_values on a seeded virtual node returns the uploaded sentinel"
    (Context.get_values seeded_ctx v) (L.blank n) ~f:Float.equal;
  p "execute_with_census refuses reading a seeded Virtual node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual" ] (fun () ->
         L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_seeded_census" o
           ~seed:[ (x, xv); (v, L.blank n) ]
           ~read:[ v ]));
  p "execute refuses reading a seeded Virtual node"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "Virtual" ] (fun () ->
         L.execute ~ctx:(Context.auto ()) ~name:"les_seeded_exec" o
           ~seed:[ (x, xv); (v, L.blank n) ]
           ~read:[ v ]));

  (* --- Host_init refusal control --- *)
  let host_only = node "les_host_only" in
  let host_values = [| 101.; 102.; 103.; 104. |] in
  let nd =
    Ir.Ndarray.create_array ~debug:"les_host_only" Ir.Ops.single ~dims:[| n |] ~padding:None
  in
  Ir.Ndarray.set_flat_values nd host_values;
  Ir.Host_inits.register host_only (lazy nd);
  p "raw Context.observability classifies an uncompiled host-init node as Host_init"
    (match Context.observability raw_ctx host_only with Context.Host_init -> true | _ -> false);
  p_all2 "raw Context.get_values on a Host_init node returns its host-init array"
    (Context.get_values raw_ctx host_only)
    host_values ~f:Float.equal;
  p "execute_with_census refuses reading a Host_init node with no stored buffer"
    (refused_with ~substrings:[ "Ll_test.execute_with_census"; "host-init" ] (fun () ->
         L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_host_census" o
           ~seed:[ (x, xv) ]
           ~read:[ host_only ]));

  (* --- Local and unreadable-Virtual refusal controls: keep Context.get_values's User_error --- *)
  let loc, v_loc, _loc_out, loc_llc = make_local_program () in
  let o_loc = with_pinned_virtualize_settings (fun () -> L.optimize ~name:"les_local" loc_llc) in
  let loc_ctx = L.run ~ctx:(Context.auto ()) ~name:"les_local_raw" o_loc ~seed:[] in
  p "the reduction intermediate is finalized Local" (L.known_local o_loc loc);
  p "the virtual reader of the Local intermediate stays Virtual" (L.known_virtual o_loc v_loc);
  p "raw Context.observability classifies the intermediate as Unobservable_local"
    (match Context.observability loc_ctx loc with
    | Context.Unobservable_local _ -> true
    | _ -> false);
  p "raw Context.observability classifies the virtual reader as Unobservable_local with path"
    (match Context.observability loc_ctx v_loc with
    | Context.Unobservable_local [ _; _ ] -> true
    | _ -> false);
  p "execute preserves refused_as_local for a Local node"
    (L.refused_as_local (fun () ->
         L.execute ~ctx:(Context.auto ()) ~name:"les_local_exec" o_loc ~seed:[] ~read:[ loc ]));
  p "execute preserves User_error dependency path for an unreadable Virtual node reading Local"
    (user_error_with ~substrings:[ "les_v_loc ->"; "les_loc"; "placed Local" ] (fun () ->
         L.execute ~ctx:(Context.auto ()) ~name:"les_v_local_exec" o_loc ~seed:[] ~read:[ v_loc ]));

  (* --- Positive controls: stored readbacks succeed --- *)
  let got_out, _census =
    L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_ok_out" o
      ~seed:[ (x, xv) ]
      ~read:[ out ]
  in
  p_all2 "reading the materialized output of the virtualized program succeeds" (List.hd_exn got_out)
    out_expected ~f:Float.equal;
  let xm, vm, outm, llcm = make_program ~suffix:"mat" in
  let o_mat =
    with_pinned_virtualize_settings (fun () ->
        L.optimize ~materialized:[ vm ] ~name:"les_ok_mat" llcm)
  in
  p "re-optimizing with ~materialized:[ v ] leaves v non-virtual" (not (L.known_virtual o_mat vm));
  let got_mat, _census =
    L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_ok_mat" o_mat
      ~seed:[ (xm, xv) ]
      ~read:[ vm; outm ]
  in
  let got_v, got_outm = match got_mat with [ gv; go ] -> (gv, go) | _ -> assert false in
  p_all2 "materialized v reads back its exact computed values" got_v v_expected ~f:Float.equal;
  p_all2 "materialized out reads back its exact computed values" got_outm out_expected
    ~f:Float.equal;
  let got_empty, _census =
    L.execute_with_census ~ctx:(Context.auto ()) ~name:"les_ok_empty" o ~seed:[ (x, xv) ] ~read:[]
  in
  p "an empty ~read list succeeds" (match got_empty with [] -> true | _ :: _ -> false);
  Stdio.eprintf "ll_test_execute_stored: backend %s (not part of the golden)\n%!" backend_name
