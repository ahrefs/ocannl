(* gh-ocannl-1329: a context [Context.release] freed is a dead handle. Compile, run (on the released
   value, or on a live value with a routine compiled into the released one), sync, the transfers and
   the decision recorders all refuse it with a typed [Utils.User_error] naming the release, rather
   than reaching a freed pool. The release flag is shared exactly by the values that share pools: an
   earlier value of the same linear upload chain is refused too, while a compile sibling and the
   parent keep working. *)

open Base
open Verdict.Claims
module LL = Ir.Low_level

let node = Ll_test.node_factory ~first_id:13290 ~dims:[| 4 |] ()

(* Whether [f] raises the typed refusal whose message contains [needle]. Any other outcome --
   success, or a different exception -- is a [false] claim rather than an aborted case, so each row
   reports on its own. *)
let refused ~needle f =
  match f () with
  | _ -> false
  | exception Utils.User_error msg -> String.is_substring msg ~substring:needle
  | exception _ -> false

let released = "was released by Context.release"

(* [dst[i] := src[i] + 1 + i]: values that identify their cell and stay off the zero-init. *)
let shift ~src ~dst =
  let i = Ll_test.sym () in
  Ll_test.loop_n i 4
    (Ll_test.set dst
       [| Ll_test.iter i |]
       (Ll_test.add (Ll_test.get src [| Ll_test.iter i |]) (Ll_test.tick i)))

let link_shift ?lctx ~ctx ~name ~src ~dst () =
  Ll_test.materialize src;
  Ll_test.materialize dst;
  let lctx = Option.value_or_thunk lctx ~default:LL.empty_optimize_ctx in
  let o = Ll_test.optimize_in lctx ~materialized:[ src; dst ] ~name (shift ~src ~dst) in
  Ll_test.link ~ctx ~name o

let seed = [| 10.; 20.; 30.; 40. |]
let shifted = [| 11.; 22.; 33.; 44. |]

let () =
  Verdict.case "chr_released" @@ fun () ->
  let base = Context.auto () in
  let x_a = node "chr_x_a" and out_a = node "chr_out_a" in
  let x_b = node "chr_x_b" and out_b = node "chr_out_b" in
  let c_a, r_a = link_shift ~ctx:base ~name:"chr_a" ~src:x_a ~dst:out_a () in
  let c_b, r_b = link_shift ~ctx:base ~name:"chr_b" ~src:x_b ~dst:out_b () in
  (* An upload into a buffer [c_a] already holds continues its linear chain: [c_a'] is the latest
     value, the one to release, and [c_a] shares its flag. *)
  let c_a' = Context.set_values c_a x_a seed in
  let c_a' = Context.run c_a' r_a in
  pf "chr_released: the routine ran before the release"
    (Poly.equal (Context.get_values c_a' out_a) shifted);
  Context.release c_a';
  let refused_here what f = pf "chr_released: %s is refused" what (refused ~needle:released f) in
  refused_here "run on the released value" (fun () -> Context.run c_a' r_a);
  refused_here "run on an earlier value of its upload chain" (fun () -> Context.run c_a r_a);
  refused_here "run on the live parent with a routine compiled into the released value" (fun () ->
      Context.run base r_a);
  refused_here "compile" (fun () -> link_shift ~ctx:c_a' ~name:"chr_a2" ~src:x_a ~dst:out_a ());
  refused_here "sync" (fun () -> Context.sync c_a');
  refused_here "to_host" (fun () -> Context.to_host c_a' out_a);
  refused_here "get_values" (fun () -> Context.get_values c_a' out_a);
  refused_here "set_values" (fun () -> Context.set_values c_a' x_a seed);
  refused_here "from_host" (fun () -> Context.from_host c_a' x_a (Context.to_host c_b x_b));
  refused_here "copy out of it" (fun () -> Context.copy ~src:c_a' ~dst:c_b x_a);
  refused_here "copy into it" (fun () -> Context.copy ~src:c_b ~dst:c_a' x_b);
  refused_here "decide_materialized" (fun () -> Context.decide_materialized c_a' [ out_a ]);
  refused_here "decide_inline" (fun () -> Context.decide_inline c_a' [ out_a ]);
  refused_here "decide_footprint" (fun () -> Context.decide_footprint c_a' [ out_a ]);
  (* The refusal names the context it refused. *)
  p "chr_released: the refusal names the entry point and the backend"
    (refused ~needle:"Context.run: context" (fun () -> Context.run c_a' r_a)
    && refused ~needle:("backend " ^ Context.backend_name c_a') (fun () -> Context.sync c_a'));
  (* The compile sibling keeps its own flag, and the parent stays live. *)
  let c_b = Context.run (Context.set_values c_b x_b seed) r_b in
  p "chr_released: the compile sibling still runs and reads back"
    (Poly.equal (Context.get_values c_b out_b) shifted);
  let x_c = node "chr_x_c" and out_c = node "chr_out_c" in
  let c_c, r_c = link_shift ~ctx:base ~name:"chr_c" ~src:x_c ~dst:out_c () in
  let c_c = Context.run (Context.set_values c_c x_c seed) r_c in
  p "chr_released: the live parent still compiles and runs"
    (Poly.equal (Context.get_values c_c out_c) shifted);
  Context.release c_b;
  Context.release c_c
