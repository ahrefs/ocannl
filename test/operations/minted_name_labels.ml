(* gh-ocannl-2103: a user label spelled like a name the compiler mints still compiles, and computes
   what an ordinarily labelled twin computes.

   A node's code name is its label unless the minter ([C_syntax.get_code_ident]) reserves it, in
   which case the node takes the disambiguated [n<id>_<label>] form. Four families of names the
   compiler mints were not reserved, so a node labelled like one met it in the emitted kernel and
   the kernel failed to compile:

   1. Another node's minted name: the disambiguated [n<id>_<label>] a repeated label takes, the bare
   [n<id>] of a node with no label, and a gradient's [<label>_grad].

   2. A loop index, [Indexing.symbol_ident] of the loop's symbol ([i<n>]): inside the loop the index
   shadows the node's buffer.

   3. A pool-chunked [Grid] loop's chunk index and bounds ([C_syntax.grid_chunk_derivations] of the
   loop index: [i<n>_chunk], [i<n>_lo], [i<n>_hi]), declared inside the chunk body.

   4. A renderer's fixed temporaries ([C_syntax.fixed_temporaries]): the routine-logging arms'
   [new_set_v] and [new_set_local_v], whose initializer read the node they shadow, and the
   [Set_from_vec] arm's [vec_result], logged or not.

   Every hostile label is derived from what the compiler actually mints: the victims' names from the
   minter itself over the routine without the hostile nodes, the loop names from the loop's own
   symbol and the derivation templates, the temporaries from the constants the renderers declare
   them by. Each leg builds the routine twice, once with the hostile labels and once with ordinary
   ones, executes both on the selected backend, and claims the two agree with each other and with
   values computed on the host; it also claims the hostile nodes took code names of their own.

   The executable runs twice, as [fixed_name_temporaries] does: without routine logging for the
   first three families and [vec_result], and under [--ocannl_debug_log_from_routines=true
   --ocannl_log_level=2] for the logging arms (logging also keeps a [Grid] loop off the pool).

   GPU backends: every node identifier carries the dialect's [t_] prefix there, so only the first
   family can collide (both sides carry the prefix); the others still compile and execute there, and
   the claim that the [Grid] loop rendered on the pool is the host-pool backends' leg. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode
open Verdict.Claims

(* Admit the small Grid loop of family 3 to the pool: the launch floor is a cost heuristic. *)
let () = Unix.putenv "OCANNL_CC_GRID_MIN_UPDATES" "0"
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)
let logged = Utils.debug_log_from_routines ()
let get = Ll_test.get
let n = 16

(* Distinct small integers per input, so every expected value is exact in single precision. *)
let input k = Array.init n ~f:(fun i -> Float.of_int ((((i * 5) + (k * 3)) % 11) - 4))

let code_name tn =
  match tn.Tn.code_name with
  | Some name -> name
  | None -> failwith "minted_name_labels: a node of a compiled routine has no code name"

(* One twin of a leg: its routine, inputs (each seeded with [input]), outputs, the hostile nodes and
   the host's expectation for the outputs. *)
type twin = {
  name : string;
  o : LL.optimized;
  inputs : Tn.t list;
  outputs : Tn.t list;
  hostile : (Tn.t * string) list;  (** A hostile node and the label it was given. *)
  expected : expected;
}

and expected =
  | Values of (float array -> float array)
      (** Outputs, concatenated, from the inputs' values, concatenated. *)
  | Unit_draws  (** Random draws, each at least 0 and below 1: the reference twin is the oracle. *)

(* [run twin] executes [twin] and returns its outputs, concatenated, and its routine. *)
let run twin =
  let seed = List.mapi twin.inputs ~f:(fun k tn -> (tn, input k)) in
  let ctx, routine = Ll_test.run_routine ~name:twin.name twin.o ~seed in
  (Array.concat (List.map twin.outputs ~f:(Context.get_values ctx)), routine)

let input_values twin = Array.concat (List.mapi twin.inputs ~f:(fun k _ -> input k))

(* The claims of one leg, over its hostile twin and its ordinarily labelled one. *)
let check ?(on_routine = fun _ -> ()) what ~(build : hostile:bool -> twin) =
  case what @@ fun () ->
  let reference = build ~hostile:false in
  let got_reference, _ = run reference in
  let twin = build ~hostile:true in
  p (what ^ ": the leg has hostile nodes") (not (List.is_empty twin.hostile));
  let got, routine = run twin in
  (match reference.expected with
  | Values f ->
      p_all2
        (what ^ ": the ordinarily labelled twin computes the host's values")
        got_reference
        (f (input_values reference))
        ~f:Float.equal
  | Unit_draws ->
      p_all (what ^ ": the ordinarily labelled twin draws in [0, 1)") (Array.to_list got_reference)
        ~f:(fun v -> Float.(0. <= v && v < 1.)));
  p_all2
    (what ^ ": the hostile twin compiles and computes the same values")
    got got_reference ~f:Float.equal;
  let names = List.map (twin.inputs @ twin.outputs) ~f:code_name in
  p_pairwise_distinct
    (what ^ ": every node of the hostile twin has a code name of its own")
    names ~equal:String.equal ~to_string:Fn.id;
  p_none (what ^ ": no hostile node keeps its label as its code name") twin.hostile
    ~f:(fun (tn, label) -> String.equal (code_name tn) label);
  on_routine routine

(* The plain [sum_k (k+1) * in_k] of the inputs at each cell, written to [x]. *)
let weighted_sum inputs ~at =
  List.foldi inputs ~init:(Ll_test.c 0.) ~f:(fun k acc tn ->
      Ll_test.add acc (Ll_test.mul (Ll_test.c (Float.of_int (k + 1))) (get tn [| at |])))

let expected_weighted_sum ~inputs values =
  Array.init n ~f:(fun i ->
      List.foldi (List.init inputs ~f:Fn.id) ~init:0. ~f:(fun k acc _ ->
          acc +. (Float.of_int (k + 1) *. values.((k * n) + i))))

(* {1 Family 1: another node's minted name} *)

(* Two nodes repeating a label (each takes [n<id>_<label>]), one with no label (it takes [n<id>])
   and a gradient ([<label>_grad]); the hostile nodes are labelled with the names the minter gives
   those victims over the routine without them. *)
let node_names ~hostile =
  let first_id = if hostile then 2103000 else 2103100 in
  let name = if hostile then "mnl_nodes" else "mnl_nodes_ref" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let dup1 = mk "mnl_dup" and dup2 = mk "mnl_dup" and unlabelled = mk "" in
  (* A gradient's label leads with ["grad"]; [node_factory] takes one component. *)
  let grad =
    Tn.create (Tn.Specified Ops.single) ~id:(first_id + 60) ~label:[ "grad"; "mnl_g" ]
      ~unpadded_dims:(lazy [| n |])
      ~padding:(lazy None)
      ()
  in
  let x = mk "mnl_nodes_x" in
  let victims = [ dup1; dup2; unlabelled; grad ] in
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let body inputs = Ll_test.loop_n i n (Ll_test.set_at x at (weighted_sum inputs ~at)) in
  let minted =
    let mint = LL.get_ident_within_code ~no_dots:true [| body victims |] in
    List.map victims ~f:mint
  in
  let hostile_labels =
    List.mapi minted ~f:(fun k victim -> if hostile then victim else "mnl_ref_" ^ Int.to_string k)
  in
  let hostile_nodes = List.map hostile_labels ~f:mk in
  let inputs = victims @ hostile_nodes in
  List.iter (x :: inputs) ~f:Ll_test.materialize;
  let o = Ll_test.optimize ~materialized:(x :: inputs) ~name (body inputs) in
  {
    name;
    o;
    inputs;
    outputs = [ x ];
    hostile = (if hostile then List.zip_exn hostile_nodes hostile_labels else []);
    expected = Values (expected_weighted_sum ~inputs:(List.length inputs));
  }

(* {1 Family 2: a loop index} *)

let loop_index ~hostile =
  let first_id = if hostile then 2103200 else 2103300 in
  let name = if hostile then "mnl_loop" else "mnl_loop_ref" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let i = Ll_test.sym () in
  let label = if hostile then Idx.symbol_ident i else "mnl_loop_in" in
  let a = mk label and x = mk "mnl_loop_x" in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let at = Ll_test.iter i in
  let o =
    Ll_test.optimize ~materialized:[ a; x ] ~name
      (Ll_test.loop_n i n (Ll_test.set_at x at (weighted_sum [ a ] ~at)))
  in
  {
    name;
    o;
    inputs = [ a ];
    outputs = [ x ];
    hostile = (if hostile then [ (a, label) ] else []);
    expected = Values (expected_weighted_sum ~inputs:1);
  }

(* {1 Family 3: a pool-chunked Grid loop's chunk names} *)

let grid_chunks ~hostile =
  let first_id = if hostile then 2103400 else 2103500 in
  let name = if hostile then "mnl_grid" else "mnl_grid_ref" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let i = Ll_test.sym () in
  let chunk, lo, hi = Ir.C_syntax.grid_chunk_derivations in
  let labels =
    List.mapi [ chunk; lo; hi ] ~f:(fun k d ->
        if hostile then Ir.C_syntax.derived_name d (Idx.symbol_ident i)
        else "mnl_grid_in" ^ Int.to_string k)
  in
  let inputs = List.map labels ~f:mk and x = mk "mnl_grid_x" in
  List.iter (x :: inputs) ~f:Ll_test.materialize;
  let at = Ll_test.iter i in
  let o =
    Ll_test.optimize ~materialized:(x :: inputs) ~name
      (Ll_test.loop_n ~axis:LL.Grid i n (Ll_test.set_at x at (weighted_sum inputs ~at)))
  in
  {
    name;
    o;
    inputs;
    outputs = [ x ];
    hostile = (if hostile then List.zip_exn inputs labels else []);
    expected = Values (expected_weighted_sum ~inputs:3);
  }

(* {1 Family 4: the renderers' fixed temporaries} *)

(* Under logging, a [Set] declares [new_set_v] from the value it stores: a node of that label read
   in the value is read through the temporary being declared. *)
let logged_set ~hostile =
  let first_id = if hostile then 2103600 else 2103700 in
  let name = if hostile then "mnl_logged_set" else "mnl_logged_set_ref" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let label = if hostile then Ir.C_syntax.log_set_temporary else "mnl_set_in" in
  let a = mk label and b = mk "mnl_set_b" and x = mk "mnl_set_x" in
  List.iter [ a; b; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let o =
    Ll_test.optimize ~materialized:[ a; b; x ] ~name
      (Ll_test.loop_n i n (Ll_test.set_at x at (weighted_sum [ a; b ] ~at)))
  in
  {
    name;
    o;
    inputs = [ a; b ];
    outputs = [ x ];
    hostile = (if hostile then [ (a, label) ] else []);
    expected = Values (expected_weighted_sum ~inputs:2);
  }

(* Under logging, a [Set_local] declares [new_set_local_v] from the value it assigns. Optimization
   would fold the single-assignment local away, so the scoped form reaches the backend as
   written. *)
let logged_local ~hostile =
  let first_id = if hostile then 2103800 else 2103900 in
  let name = if hostile then "mnl_logged_local" else "mnl_logged_local_ref" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let label = if hostile then Ir.C_syntax.log_set_local_temporary else "mnl_local_in" in
  let a = mk label and b = mk "mnl_local_b" and x = mk "mnl_local_x" in
  let loc = mk "mnl_local_v" in
  List.iter [ a; b; x ] ~f:Ll_test.materialize;
  Ll_test.virtualize loc;
  let id = LL.get_scope loc in
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let v = weighted_sum [ a; b ] ~at in
  let raw = Ll_test.loop_n i n (Ll_test.set_at x at v) in
  let scoped =
    Ll_test.loop_n i n
      (LL.unflat_lines (Ll_test.define_local id v @ [ Ll_test.set_at x at (LL.Get_local id) ]))
  in
  let o = Ll_test.optimize_scoped ~materialized:[ a; b; x ] ~name ~raw scoped in
  {
    name;
    o;
    inputs = [ a; b ];
    outputs = [ x ];
    hostile = (if hostile then [ (a, label) ] else []);
    expected = Values (expected_weighted_sum ~inputs:2);
  }

(* A [Set_from_vec] declares [vec_result] and stores its lanes into the destination: a destination
   of that label is written through the temporary. *)
let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()

let packed_draw ~hostile =
  let first_id = if hostile then 2104000 else 2104100 in
  let name =
    (if hostile then "mnl_packed" else "mnl_packed_ref") ^ if logged then "_logged" else ""
  in
  let label = if hostile then Ir.C_syntax.vec_result_temporary else "mnl_packed_x" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| 4 |] () in
  let x = mk label and y = mk "mnl_packed_y" in
  List.iter [ x; y ] ~f:Ll_test.materialize;
  let threefry a b = LL.Binop (Ops.Threefry4x32_light, a, b) in
  (* A draw per destination, each from its own key. *)
  let draw tn key =
    Ll_test.set_from_vec ~length:4 ~vec_unop:Ops.Uint4x32_to_prec_uniform tn [| Idx.Fixed_idx 0 |]
      ( threefry
          (threefry (LL.Constant_bits 7L, u4) (LL.Constant_bits key, u4), u4)
          (LL.Embed_index (Idx.Fixed_idx 0), iprec),
        u4 )
  in
  let o = Ll_test.optimize ~materialized:[ x; y ] ~name (Ll_test.seq (draw x 42L) (draw y 43L)) in
  {
    name;
    o;
    inputs = [];
    outputs = [ x; y ];
    hostile = (if hostile then [ (x, label) ] else []);
    expected = Unit_draws;
  }

let () =
  check "another node's minted name" ~build:node_names;
  if logged then (
    check "a logged Set's temporary" ~build:logged_set;
    check "a logged Set_local's temporary" ~build:logged_local)
  else (
    check "a loop index" ~build:loop_index;
    let pool = Verdict.leg ~when_:on_cpu ~on:backend_name () in
    check "a Grid loop's chunk names" ~build:grid_chunks ~on_routine:(fun routine ->
        in_leg ~leg:pool "a Grid loop's chunk names: the hostile twin's loop renders on the pool"
          (fun label -> p label (routine.Context.grid.Ir.C_syntax.pool_rendered >= 1))));
  check "a Set_from_vec's temporary" ~build:packed_draw
