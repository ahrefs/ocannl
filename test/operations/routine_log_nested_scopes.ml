(* gh-ocannl-2111: routine logging traces a scope definition nested in any statement's operand, not
   only in a [Set_local]'s value.

   gh-ocannl-2065 carried the trace of a kept scope's assignments into the [Local_scope] definitions
   its value nests; every other statement still rendered its operands with [C_syntax.pp_scalar]'s
   quiet default, so a definition nested in a materialized store's value was computed in the logged
   routine but missing from its log, the store's own line printing the scope local's value with
   nothing saying where it came from. Each case below nests a scope under one statement arm -- a
   [Set] value, a [Set_dynamic] value and its dynamic row, a [Set_from_vec] argument (a [uint4x32]
   block, logged as four lanes), an [If] condition -- compiles it on the selected backend with
   routine logging on, and executes it.

   The claims, per case: the routine computed its values (checked against the host); the stream log
   holds the nested definition's trace ([# local v<id>_...]), one line for every assignment of it
   the routine executes; and the first such line comes before the first line of the enclosing
   statement's store, which reads the local. The log text is not a golden: its values are the
   backend's. The scoped programs reach the backend through [Ll_test.optimize_scoped], since
   [optimize] would collapse a single-assignment scope into its reader. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()
let get = Ll_test.get
let n = 8

(* Distinct small integers, zero and both signs included, so every value is exact in single
   precision and the [If] case takes both branches. *)
let input = Array.init n ~f:(fun i -> Float.of_int ((i * 5 % 11) - 4))

(* The trace's comment line of an assignment to [id]'s local: [C_syntax.pp_scope_id] spells the
   local [v<scope_id>_<node>]. *)
let local_marker (id : LL.scope_id) = Printf.sprintf "# local v%d_" id.scope_id

(* A store's value line, ["<node>[<offset>]{=<old>} = <new> = <expression>"]. *)
let store_marker tn = Ir.Tnode.label_name tn ^ "["

(* What each case leaves for the log claims, read once every routine has run. *)
type case = {
  what : string;
  scopes : (string * LL.scope_id * int) list;
      (** Where each nested definition sits, its scope, and how many assignments of it run. *)
  store : Ir.Tnode.t;
}

(* Compiles and runs one case's routine, reading [read] back; the sync drains Metal's buffered
   routine logs into the stream file (gh-ocannl-1594). *)
let run ~name o ~seed ~read =
  let ctx = Ll_test.run ~name o ~seed in
  let values = List.map read ~f:(Context.get_values ctx) in
  Context.sync ctx;
  values

(* [x[i] = scope(t = a[i] + 1; t = t * t) * 2]. *)
let check_set ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "nls_set_a" and x = mk "nls_set_x" and t = mk "nls_set_t" in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  Ll_test.virtualize t;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let id = LL.get_scope t in
  let value = Ll_test.add (get a [| at |]) (Ll_test.c 1.) in
  let store v = Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.mul v (Ll_test.c 2.))) in
  (* Two assignments, since codegen collapses a [Set]'s single-assignment scope into the store where
     the collapse is exact ([C_syntax]'s [collapse_exact_scopes]), leaving nothing nested. *)
  let squared =
    Ll_test.scope_of id
      (Ll_test.seq
         (LL.Set_local (id, value))
         (LL.Set_local (id, Ll_test.mul (LL.Get_local id) (LL.Get_local id))))
  in
  let name = "nls_set" in
  let o =
    Ll_test.optimize_scoped ~materialized:[ a; x ] ~name
      ~raw:(store (Ll_test.mul value value))
      (store squared)
  in
  (match run ~name o ~seed:[ (a, input); (x, Ll_test.blank n) ] ~read:[ x ] with
  | [ vx ] ->
      p_all2 "Set: the store writes the nested definition's value" vx
        (Array.map input ~f:(fun v -> (v +. 1.) *. (v +. 1.) *. 2.))
        ~f:Float.equal
  | _ -> assert false);
  { what = "Set"; scopes = [ ("value", id, 2 * n) ]; store = x }

(* [y[rows[i]] = scope(a[i] * 3)], its row read through a scope of its own: [scope(rows[i])]. The
   rows reverse the loop index, so each scatter lands where the index alone does not say. *)
let check_set_dynamic ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "nls_dyn_a" and rows = mk "nls_dyn_rows" and y = mk "nls_dyn_y" in
  let t = mk "nls_dyn_t" and r = mk "nls_dyn_r" in
  List.iter [ a; rows; y ] ~f:Ll_test.materialize;
  List.iter [ t; r ] ~f:Ll_test.virtualize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let id_value = LL.get_scope t and id_row = LL.get_scope r in
  let value = Ll_test.mul (get a [| at |]) (Ll_test.c 3.) in
  let row = get rows [| at |] in
  let scatter ~row v =
    Ll_test.loop_n i n
      (Ll_test.scatter ~tn:y ~idcs:[| Ll_test.fixed 0 |] ~dyn_axis:0 ~dyn_value:(row, Ops.single) v)
  in
  (* The analyses refuse a [Set_dynamic], so the raw twin writes [y] at the loop index and reads the
     rows. *)
  let raw = Ll_test.loop_n i n (Ll_test.set_at y at (Ll_test.add value row)) in
  let name = "nls_dyn" in
  let o =
    Ll_test.optimize_scoped ~materialized:[ a; rows; y ] ~name ~raw
      (scatter ~row:(Ll_test.scope ~id:id_row r row) (Ll_test.scope ~id:id_value t value))
  in
  let reversed = Array.init n ~f:(fun i -> Float.of_int (n - 1 - i)) in
  (match run ~name o ~seed:[ (a, input); (rows, reversed); (y, Ll_test.blank n) ] ~read:[ y ] with
  | [ vy ] ->
      p_all2 "Set_dynamic: the scatter writes the nested definition's value at the nested row" vy
        (Array.init n ~f:(fun r -> input.(n - 1 - r) *. 3.))
        ~f:Float.equal
  | _ -> assert false);
  { what = "Set_dynamic"; scopes = [ ("value", id_value, n); ("row", id_row, n) ]; store = y }

(* [z[0..3] = uniform(scope(threefry(threefry(key, ctr), 0)))]: a packed draw whose [uint4x32] block
   is a nested definition. Drawn twice, as a scoped and a plain statement, the two must agree bit
   for bit. *)
let check_set_from_vec ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| 4 |] () in
  let z = mk "nls_vec_z" and plain = mk "nls_vec_plain" in
  let blk =
    Ll_test.node_factory ~prec:u4 ~first_id:(first_id + 50) ~dims:[| 1 |] () "nls_vec_blk"
  in
  List.iter [ z; plain ] ~f:Ll_test.materialize;
  Ll_test.virtualize blk;
  let id = LL.get_scope blk in
  let threefry a b = LL.Binop (Ops.Threefry4x32_light, a, b) in
  let block =
    threefry
      (threefry (LL.Constant_bits 7L, u4) (LL.Constant_bits 42L, u4), u4)
      (LL.Embed_index (Idx.Fixed_idx 0), iprec)
  in
  let draw tn arg =
    Ll_test.set_from_vec ~length:4 ~vec_unop:Ops.Uint4x32_to_prec_uniform tn [| Idx.Fixed_idx 0 |]
      (arg, u4)
  in
  let raw = Ll_test.seq (draw z block) (draw plain block) in
  let name = "nls_vec" in
  let o =
    Ll_test.optimize_scoped ~materialized:[ z; plain ] ~name ~raw
      (Ll_test.seq (draw z (Ll_test.scope ~id blk block)) (draw plain block))
  in
  (match run ~name o ~seed:[] ~read:[ z; plain ] with
  | [ vz; vp ] ->
      p_all2 "Set_from_vec: the scoped draw equals the plain one bit for bit" vz vp ~f:(fun a b ->
          Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b));
      p_all "Set_from_vec: the draws lie in [0, 1)" (Array.to_list vz) ~f:(fun v ->
          Float.(0. <= v && v < 1.))
  | _ -> assert false);
  { what = "Set_from_vec"; scopes = [ ("argument", id, 1) ]; store = z }

(* [if (scope(0 < a[i])) w[i] = a[i] + 5]. *)
let check_if ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk "nls_if_a" and w = mk "nls_if_w" and t = mk "nls_if_t" in
  List.iter [ a; w ] ~f:Ll_test.materialize;
  Ll_test.virtualize t;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let id = LL.get_scope t in
  let cond = Ll_test.cmp_at Ops.single Ops.Cmplt (Ll_test.c 0.) (get a [| at |]) in
  let guarded c =
    Ll_test.loop_n i n
      (Ll_test.if_at Ops.single c
         (Ll_test.set_at w at (Ll_test.add (get a [| at |]) (Ll_test.c 5.))))
  in
  let name = "nls_if" in
  let o =
    Ll_test.optimize_scoped ~materialized:[ a; w ] ~name ~raw:(guarded cond)
      (guarded (Ll_test.scope ~id t cond))
  in
  (match run ~name o ~seed:[ (a, input); (w, Ll_test.blank n) ] ~read:[ w ] with
  | [ vw ] ->
      p_all2 "If: the guard reads the nested definition's value" vw
        (Array.map input ~f:(fun v -> if Float.(v > 0.) then v +. 5. else Ll_test.sentinel))
        ~f:Float.equal
  | _ -> assert false);
  { what = "If"; scopes = [ ("condition", id, n) ]; store = w }

let stream_log_lines () =
  let dir = Utils.log_files_dir () in
  Stdlib.Sys.readdir dir |> Array.to_list
  |> List.filter ~f:(String.is_suffix ~suffix:".log")
  |> List.sort ~compare:String.compare
  |> List.concat_map ~f:(fun f -> Stdio.In_channel.read_lines (Stdlib.Filename.concat dir f))

let first_index lines ~f = List.findi lines ~f:(fun _ l -> f l) |> Option.map ~f:fst

let () =
  if not (Utils.debug_log_from_routines ()) then
    fail "this test runs under --ocannl_debug_log_from_routines=true (see its dune rule)";
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  (* In sequence: a list literal would evaluate them right to left. *)
  let set = check_set ~first_id:2111000 in
  let set_dynamic = check_set_dynamic ~first_id:2111100 in
  let set_from_vec = check_set_from_vec ~first_id:2111200 in
  let if_ = check_if ~first_id:2111300 in
  let cases = [ set; set_dynamic; set_from_vec; if_ ] in
  let lines = stream_log_lines () in
  List.iter cases ~f:(fun { what; scopes; store } ->
      let first_store = first_index lines ~f:(String.is_prefix ~prefix:(store_marker store)) in
      List.iter scopes ~f:(fun (where, id, times) ->
          let marker = local_marker id in
          let what = what ^ " " ^ where in
          let count = List.count lines ~f:(String.is_substring ~substring:marker) in
          Stdio.eprintf "%s: %d lines of %s... (not part of the golden)\n%!" what count marker;
          p
            (Printf.sprintf
               "%s: the log traces every assignment the nested definition executes (%d)" what times)
            (count = times);
          p
            (what ^ ": the nested definition's trace precedes the enclosing store's")
            (match (first_index lines ~f:(String.is_substring ~substring:marker), first_store) with
            | Some d, Some s -> d < s
            | _ -> false)))
