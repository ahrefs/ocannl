(* gh-ocannl-2266: [Fuse_epilogue]'s write inventory and impure scope bodies.

   The plain-write sites of [Fuse_epilogue] (the [Privatize] store-back and the accumulation nest)
   and its statement-level layout read the accumulator's writes with a scanner that does not descend
   into [Local_scope] bodies. That is complete only because no scope body holds a tensor write when
   the op runs: scope purity (gh-ocannl-584) bans [Set], [Zero_out], [Set_dynamic], [Set_from_vec]
   and [Tile_mma] in a scope body, [Low_level.optimize] checks it on the way in, and every schedule
   op is refused when it leaves an impure body behind. The one input reaching a schedule without the
   entry gate is a [?prelowered] routine, hand-built IR the scope-grammar gate deliberately leaves
   to codegen's exit gate; [Fuse_epilogue] checks purity itself, first, rather than misreading its
   writes.

   The program: [acc] reduced over [k] inside an output loop [i], then, in the same [i] iteration
   after the reduction loop, a statement whose scope body overwrites [acc[i]]; the tail [out = acc +
   bias] follows. The scope body's write is the last write of each [acc] cell, so the tail must read
   it. Values are small dyadics, compared exactly, through [Ll_test.interpret], the IR's reference
   semantics (a scope body runs where its scope is read):

   - the entry gate: [optimize] refuses the program, naming [validate_scope_bodies]; - past
   [optimize] (the [?prelowered] form), [Fuse_epilogue] refuses it, naming scope purity, both probed
   and through [Schedule.apply]. On the base it fused: the accumulation-nest site put the tail right
   after the reduction loop, before the scope body's write; - the premise: that relocation, built by
   hand and interpreted, computes different values from the unfused program; - the control: the same
   program with a pure scope body fuses, values equal. *)

open Base
open Ocannl.Operation.DSL_modules
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule

let node = node_factory ~first_id:2266000 ~dims:[| 2 |] ()
let seq_all = function [] -> LL.Noop | s :: tl -> List.fold tl ~init:s ~f:seq

type case = {
  llc : LL.t;  (** The program under test. *)
  raw : LL.t;
      (** Its scope-free twin, which [optimize] accepts: the record's store and placements. *)
  acc : Tn.t;
  out : Tn.t;
  inputs : (Tn.t * float array) list;
}

(* [impure]: the scope body writes [acc[i]] before assigning its local. [relocated]: the program the
   base's accumulation-nest fusion emitted -- the tail inside [i], right after the reduction loop,
   and no tail nest after. *)
let program ?(relocated = false) tag ~impure =
  let acc = node ("acc" ^ tag) and out = node ("out" ^ tag) and sink = node ("sink" ^ tag) in
  let bias = node ("bias" ^ tag) and x = node ~dims:[| 2; 3 |] ("x" ^ tag) in
  let v = node ("v" ^ tag) in
  virtualize v;
  List.iter [ acc; out; sink; bias; x ] ~f:materialize;
  let i = sym () and k = sym () and t = sym () in
  let id = LL.get_scope v in
  let reduce =
    loop_n k 3 (set acc [| iter i |] (add (get acc [| iter i |]) (get x [| iter i; iter k |])))
  in
  let local = LL.Set_local (id, add (c 1.) (embed i)) in
  let body = if impure then seq (set acc [| iter i |] (add (c 100.) (embed i))) local else local in
  let sunk ~scoped = set sink [| iter i |] (if scoped then scope_of id body else c 1.) in
  let tail_at idx = set out [| idx |] (add (get acc [| idx |]) (get bias [| idx |])) in
  let build ~scoped =
    if relocated then
      seq_all [ zero acc; loop_n i 2 (seq_all [ reduce; tail_at (iter i); sunk ~scoped ]) ]
    else seq_all [ zero acc; loop_n i 2 (seq reduce (sunk ~scoped)); loop_n t 2 (tail_at (iter t)) ]
  in
  {
    llc = build ~scoped:true;
    raw = build ~scoped:false;
    acc;
    out;
    inputs =
      [
        (x, Array.init 6 ~f:(fun n -> Float.of_int (n + 1)));
        (bias, Array.init 2 ~f:(fun n -> 0.25 *. Float.of_int (n + 1)));
      ];
  }

let record ~name c = optimize_scoped ~name ~raw:c.raw c.llc

let interpreted c llc =
  let memory = interp_memory c.inputs in
  interpret memory llc;
  Array.copy (Hashtbl.find_exn memory c.out)

let fuse c o = Sched.apply [ Sched.Fuse_epilogue { target = c.acc; shared = false } ] o

let names_purity ~what msg =
  Stdio.eprintf "%s: %s (not part of the golden)\n%!" what msg;
  String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
  && String.is_substring msg ~substring:"impure scope body"
  && String.is_substring msg ~substring:"Set writes the tensor node"

let () =
  let impure = program "i" ~impure:true in
  (* The entry gate: the ordinary pipeline never hands this program to a schedule. *)
  p "entry gate: optimize refuses a tensor write inside a scope body, naming validate_scope_bodies"
    (match optimize ~name:"ep2266_entry" impure.llc with
    | _ -> false
    | exception Invalid_argument msg ->
        Stdio.eprintf "entry gate: %s (not part of the golden)\n%!" msg;
        String.is_substring msg ~substring:"Low_level.validate_scope_bodies"
        && String.is_substring msg
             ~substring:("Set writes the tensor node " ^ Tn.debug_name impure.acc));
  (* Past the entry gate. *)
  p "prelowered: Fuse_epilogue's probe refuses the impure scope body, naming scope purity"
    (match Sched.fuse_epilogue_witness ~target:impure.acc (record ~name:"ep2266_probe" impure) with
    | Some msg -> names_purity ~what:"probe" msg
    | None -> false);
  p "prelowered: Schedule.apply refuses it too, naming scope purity"
    (match fuse impure (record ~name:"ep2266_apply" impure) with
    | _ -> false
    | exception Invalid_argument msg -> names_purity ~what:"apply" msg);
  (* The premise: the fusion the base performed changes the program's values. *)
  let relocated = program "r" ~relocated:true ~impure:true in
  p_exists "premise: the base's relocation computes different values from the unfused program"
    (List.zip_exn
       (Array.to_list (interpreted relocated relocated.llc))
       (Array.to_list (interpreted impure impure.llc)))
    ~f:(fun (a, b) -> not (Float.equal a b));
  (* The control: a pure scope body in the same position. *)
  let pure = program "p" ~impure:false in
  let fused = fuse pure (record ~name:"ep2266_pure" pure) in
  p "control: the same program with a pure scope body fuses"
    (List.length (LL.flat_lines [ fused.LL.llc ]) < List.length (LL.flat_lines [ pure.llc ]));
  p_all2 "control: fused values equal the unfused program" (interpreted pure fused.LL.llc)
    (interpreted pure pure.llc) ~f:Float.equal;
  Stdio.printf "epilogue_fusion_scope_purity: done\n"
