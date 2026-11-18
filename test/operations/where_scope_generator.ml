(* A property-based generator of [Where]/scope programs (gh-ocannl-2250).

   gh-ocannl-2007's scope-ordering bugs were found by hand, one review round at a time. This test
   enumerates a stated finite grammar of post-optimize rendering units and holds every program to
   three relationships, on the selected backend:

   (a) [Access_fold]'s [Emitted] walk -- under the oracle the routine compiled with and over the
   tree it rendered ({!Ir.C_syntax.scope_order_oracles}, after the exact-scope collapse) -- ends the
   scope bodies in the order the kernel text holds them, every rendered body found, each once.

   (a') In the text, nothing reads a scope's local ahead of its definition's last assignment, and no
   scope of a discarded projection operand is named at all. This reads the text and the IR's scope
   ids only, never [Scope_order]: a plan the renderer and the IR derivation shared wrongly would
   pass (a), since the text follows the plan, and fail this. In the walk's own order, each body also
   ends after every body whose local it reads (the IR's read relation).

   (b) The compiled values equal [Ll_test.interpret ~strict:true] over the program's self-contained
   twin, and both equal [replay]: a walk of the descriptor in OCaml floats sharing nothing with
   [Low_level]. The compiled program is the CSE'd one ([Low_level.eliminate_common_subexpressions]),
   with shared definitions and reads of a local outside its arm or ahead of its definition in
   operand order; the lazy interpreter does not model the order [pp_local_defs] emits definitions
   in, so the twin spells every such read as a copy of the scope it reads, which computes the same
   value.

   The grammar, its bound. One [Set] rendering unit [out[r, i] = e] per program, under a loop [i <
   4], at [single] or [int32]. [e] is a skeleton: one select plus a sibling operand ([sel]), a
   select nested in the then- or else-arm ([nest-then], [nest-else]), two sibling selects
   ([siblings]), a [Relu_gate] over a select whose first operand is [1 - i] or the mask ([gate-idx],
   [gate-data]), or a value projection [Arg1]/[Arg2] keeping a select and discarding a scope
   ([keep1], [keep2]). A select's condition is [i < 1], [i < 3] or the data condition [m[i] < 1]
   (blendable at a float precision); [sel] also takes [i < 1] spelled through an [Arg1] or [Arg2]
   projection with a discarded operand, [i < 1] annotated at [single] whatever the statement's
   precision, and a condition reading a scope. Each operand slot holds a constant, a fresh in-range
   scope, a scope reading [2^26] cells off its node except at the one iteration its path admits
   ([off], only where the path's select index conditions admit exactly one: a gate guards no
   definition), one of two shared alpha-equal tokens [A]/[B] (CSE merges them), or a scope whose
   body reads [A]'s local, as a nested copy ([rA]) or as a direct read of the id of [A]'s first
   occurrence outside any body ([fA]: ahead of [A] in operand order where it precedes it; the CSE
   never lets anything outside a body read a local defined in it). Scope bodies are an initializer
   and a two-step accumulation, so the collapse keeps them. At most two scope-bearing slots by
   default, three under [--widen]; tokens are canonical (the first is [A]); [rA]/[fA] need an [A].
   NOT covered: [If]-condition, [Set_dynamic], [Set_from_vec] and [Scan_loop] units, cross-statement
   hoisting, nesting past two selects, other precisions.

   Two shapes are out of the IR's contract, so the grammar does not produce them; the generator met
   each before it was a rule. (1) An out-of-range read guarded only by a gate: a gate guards no
   definition. [C_syntax.gate_entries] keeps behind a gate's test only the definitions that run
   under a select's condition, and every other definition of its second operand hoists ahead of the
   statement, which [Low_level.speculable_select]'s invariant licenses: lowering emits a read that
   can be out of range only under a select of its own. [relu_gate(1 - i, (m<1 ? k : off))] faulted
   on cc (SIGBUS), its blended select's arm hoisted out of the gate. (2) A read, outside a scope
   body, of a local defined inside it: [Low_level.eliminate_common_subexpressions] restores its
   table after each [Local_scope] body ("inner definitions must not leak to sibling subtrees"), so
   the pipeline never produces one, and the renderer declares such a local inside the body's block
   (cc: "undeclared identifier"). Both are now refused by name at compile (gh-ocannl-2258):
   [Low_level.validate_guarded_reads] refuses a read the loop ranges place outside its node where no
   index guard protects it (a gate guards no definition, a value condition may blend), and
   [Low_level.validate_local_binders] a local read outside the scope bodies enclosing its
   definition; [where_arm_scopes] and [local_binder_gate] pin the refusals.

   Programs are compiled in batches, one row of [out] each; a batch the compile refuses is retried
   one program at a time, and a program the compile refuses is a finding. A finding prints [FINDING
   <descriptor>: <check>] on stderr; [known_findings] lists the ones a filed issue owns. [--only
   <substring>] prints the IR and kernel text of the matching programs and checks nothing;
   [--plant-swap] routes check (a) through the planted group swap below, for showing that it fails,
   and [--blind-reader] does the same for (a') with a reader that flags nothing; [--widen] lifts the
   bound ([@test/operations/slow-where_scope_generator-widen]). The census on stdout is the
   grammar's, backend-uniform; what the backend decided (blends, groups) is on stderr and in the
   legs. *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module AF = LL.Access_fold
module Generated = Test_utils.Generated
module SE = Scope_emission

let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Ir.C_syntax.scope_order_oracles_enabled () := true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Generated.init ~backend_name
let facts = Context.backend_facts_of_name backend_name
let argv = Sys.get_argv ()
let has_arg flag = Array.mem argv flag ~equal:String.equal

let arg flag =
  Array.findi argv ~f:(fun _ a -> String.equal a flag)
  |> Option.bind ~f:(fun (i, _) -> if i + 1 < Array.length argv then Some argv.(i + 1) else None)

let widen = has_arg "--widen"
let plant_swap = has_arg "--plant-swap"
let only = arg "--only"

(* [--blind-reader]: the dependency-order text reader flags nothing, for showing that its controls
   fail without it. *)
let reads_ahead =
  if has_arg "--blind-reader" then fun _ _ -> [] else Scope_emission.reads_ahead_of_definition

(* Disagreements a filed issue owns, by descriptor: excluded from the claims, reported on stderr. *)
let known_findings : string list = []

(* === The grammar === *)

let n = 4
let far = 1 lsl 26

(* Discriminating inputs: distinct per cell, off zero and off the sentinel. *)
let xs = Array.init 8 ~f:(fun k -> Float.of_int ((3 * k) + 1))
let mask = [| 0.; 1.; 0.; 1. |]

type cond = Lo | Hi | Data | Proj1 | Proj2 | Flo | Sc
type gate = Gidx | Gdata
type fill = K | Fresh | Off | Tok of int | Reads | Fwd

type expr =
  | Slot of int * fill
  | Sel of cond * expr * expr
  | Gate of gate * expr
  | Keep of bool * expr  (** [true]: [Arg1] keeps the first operand. *)
  | Add of expr * expr

type program = { skeleton : string; prec : Ops.prec; expr : expr }

let cond_name = function
  | Lo -> "i<1"
  | Hi -> "i<3"
  | Data -> "m<1"
  | Proj1 -> "arg1(i<1,_)"
  | Proj2 -> "arg2(_,i<1)"
  | Flo -> "f32(i<1)"
  | Sc -> "s<12"

let fill_name = function
  | K -> "k"
  | Fresh -> "s"
  | Off -> "off"
  | Tok 0 -> "A"
  | Tok _ -> "B"
  | Reads -> "rA"
  | Fwd -> "fA"

let rec show = function
  | Slot (_, f) -> fill_name f
  | Sel (c, a, b) -> Printf.sprintf "(%s ? %s : %s)" (cond_name c) (show a) (show b)
  | Gate (g, e) ->
      Printf.sprintf "%s(%s)" (match g with Gidx -> "relu(1-i)" | Gdata -> "relu(m)") (show e)
  | Keep (true, e) -> Printf.sprintf "arg1(%s,dead)" (show e)
  | Keep (false, e) -> Printf.sprintf "arg2(dead,%s)" (show e)
  | Add (a, b) -> Printf.sprintf "%s + %s" (show a) (show b)

let prec_name prec = if Ops.is_float prec then "f32" else "i32"

let descriptor prog =
  Printf.sprintf "%s %s: %s" prog.skeleton (prec_name prog.prec) (show prog.expr)

let index_cond = function
  | Lo | Proj1 | Proj2 | Flo -> Some (fun i -> i < 1)
  | Hi -> Some (fun i -> i < 3)
  | Data | Sc -> None

(* The iterations a path's index conditions admit, split by a select's condition. *)
let split live c =
  match index_cond c with
  | Some f -> (List.filter live ~f, List.filter live ~f:(fun i -> not (f i)))
  | None -> (live, live)

(* A gate guards no definition: a scope in its second operand hoists ahead of the statement unless a
   select's condition holds it ([C_syntax]'s [gate_entries]). Lowering emits a read that can be out
   of range only under a select of its own ([Low_level.speculable_select]'s invariant), so only a
   select's index condition admits an [off] scope. *)
let gate_live live (_ : gate) = live
let all_iterations = List.init n ~f:Fn.id

(* An [off] scope is valid only where its path admits exactly one iteration. *)
let rec valid live = function
  | Slot (_, Off) -> List.length live = 1
  | Slot _ -> true
  | Sel (c, a, b) ->
      let la, lb = split live c in
      valid la a && valid lb b
  | Gate (g, e) -> valid (gate_live live g) e
  | Keep (_, e) -> valid live e
  | Add (a, b) -> valid live a && valid live b

let rec fills = function
  | Slot (_, f) -> [ f ]
  | Sel (_, a, b) | Add (a, b) -> fills a @ fills b
  | Gate (_, e) | Keep (_, e) -> fills e

(* === The replay: the descriptor's semantics, sharing nothing with [Low_level] === *)

let fresh_shift slot = slot % 2
let off_base slot = 2 * (slot % 3)
let tok_shift t = 2 + t

(* A sum scope over [x] at [i + shift + k], [k < steps]. *)
let scope_value ~steps i shift =
  List.sum (module Float) (List.init steps ~f:Fn.id) ~f:(fun k -> xs.(i + shift + k))

let reads_value ~steps i = (2. *. scope_value ~steps i (tok_shift 0)) +. xs.(i + 4)
let constant slot = Float.of_int (100 + (10 * slot))

let rec value ~steps i = function
  | Slot (s, K) -> constant s
  | Slot (s, Fresh) -> scope_value ~steps i (fresh_shift s)
  | Slot (s, Off) -> scope_value ~steps 0 (off_base s)
  | Slot (_, Tok t) -> scope_value ~steps i (tok_shift t)
  | Slot (_, (Reads | Fwd)) -> reads_value ~steps i
  | Sel (c, a, b) -> if holds ~steps i c then value ~steps i a else value ~steps i b
  | Gate (Gidx, e) -> if 1 - i > 0 then value ~steps i e else 0.
  | Gate (Gdata, e) -> if Float.(mask.(i) > 0.) then value ~steps i e else 0.
  | Keep (_, e) -> value ~steps i e
  | Add (a, b) -> value ~steps i a +. value ~steps i b

and holds ~steps i = function
  | Lo | Proj1 | Proj2 | Flo -> i < 1
  | Hi -> i < 3
  | Data -> Float.(mask.(i) < 1.)
  | Sc -> Float.(scope_value ~steps i 0 < 12.)

let replay ?(steps = 2) prog = Array.init n ~f:(fun i -> value ~steps i prog.expr)

(* === The enumeration === *)

type skeleton = {
  sk : string;
  conds : cond list list;
  slots : int;
  make : cond list -> expr list -> expr;
}

let main_conds = [ Lo; Hi; Data ]
let pairs = List.concat_map main_conds ~f:(fun a -> List.map main_conds ~f:(fun b -> [ a; b ]))
let singles cs = List.map cs ~f:(fun c -> [ c ])
let bad () = invalid_arg "where_scope_generator: a skeleton's arity"

let skeletons =
  [
    {
      sk = "sel";
      conds = singles [ Lo; Hi; Data; Proj1; Proj2; Flo; Sc ];
      slots = 3;
      make =
        (fun cs ss ->
          match (cs, ss) with [ c ], [ t; e; b ] -> Add (Sel (c, t, e), b) | _ -> bad ());
    };
    {
      sk = "nest-then";
      conds = pairs;
      slots = 4;
      make =
        (fun cs ss ->
          match (cs, ss) with
          | [ c1; c2 ], [ t; e; e1; b ] -> Add (Sel (c1, Sel (c2, t, e), e1), b)
          | _ -> bad ());
    };
    {
      sk = "nest-else";
      conds = pairs;
      slots = 4;
      make =
        (fun cs ss ->
          match (cs, ss) with
          | [ c1; c2 ], [ t1; t; e; b ] -> Add (Sel (c1, t1, Sel (c2, t, e)), b)
          | _ -> bad ());
    };
    {
      sk = "siblings";
      conds = pairs;
      slots = 4;
      make =
        (fun cs ss ->
          match (cs, ss) with
          | [ c1; c2 ], [ t1; e1; t2; e2 ] -> Add (Sel (c1, t1, e1), Sel (c2, t2, e2))
          | _ -> bad ());
    };
  ]
  @ List.map
      [ ("gate-idx", Gidx); ("gate-data", Gdata) ]
      ~f:(fun (sk, g) ->
        {
          sk;
          conds = singles main_conds;
          slots = 3;
          make =
            (fun cs ss ->
              match (cs, ss) with
              | [ c ], [ t; e; b ] -> Add (Gate (g, Sel (c, t, e)), b)
              | _ -> bad ());
        })
  @ List.map
      [ ("keep1", true); ("keep2", false) ]
      ~f:(fun (sk, first) ->
        {
          sk;
          conds = singles main_conds;
          slots = 3;
          make =
            (fun cs ss ->
              match (cs, ss) with
              | [ c ], [ t; e; b ] -> Add (Keep (first, Sel (c, t, e)), b)
              | _ -> bad ());
        })

let slot_fills = [ K; Fresh; Off; Tok 0; Tok 1; Reads; Fwd ]
let max_scoped = if widen then 3 else 2

(* Every filling of [slots] slots: tokens canonical, [rA]/[fA] only beside an [A]. *)
let fillings slots =
  let rec go k ~scoped ~tokens =
    if k = slots then [ [] ]
    else
      List.concat_map slot_fills ~f:(fun f ->
          let scoped' = if Poly.equal f K then scoped else scoped + 1 in
          let ok = scoped' <= max_scoped && match f with Tok t -> t <= tokens | _ -> true in
          if not ok then []
          else
            let tokens = match f with Tok t when t = tokens -> tokens + 1 | _ -> tokens in
            List.map (go (k + 1) ~scoped:scoped' ~tokens) ~f:(fun rest -> f :: rest))
  in
  go 0 ~scoped:0 ~tokens:0
  |> List.filter ~f:(fun fs ->
      List.exists fs ~f:(fun f -> not (Poly.equal f K))
      && ((not (List.exists fs ~f:(function Reads | Fwd -> true | _ -> false)))
         || List.mem fs (Tok 0) ~equal:Poly.equal))

let scoped_count prog = List.count (fills prog.expr) ~f:(fun f -> not (Poly.equal f K))

let programs =
  List.concat_map [ single; Ops.int32 ] ~f:(fun prec ->
      List.concat_map skeletons ~f:(fun sk ->
          let fs = fillings sk.slots in
          List.concat_map sk.conds ~f:(fun cs ->
              List.filter_map fs ~f:(fun fill ->
                  let expr = sk.make cs (List.mapi fill ~f:(fun s f -> Slot (s, f))) in
                  if valid all_iterations expr then Some { skeleton = sk.sk; prec; expr } else None))))
  |> List.stable_sort ~compare:(fun a b -> Int.compare (scoped_count a) (scoped_count b))

(* === Building the IR === *)

let mk = node_factory ~first_id:2_250_000 ~dims:[| 8 |] ()
let mk_int = node_factory ~prec:Ops.int32 ~first_id:2_400_000 ~dims:[| 8 |] ()
let node prec = if Ops.is_float prec then mk else mk_int

type built = {
  scalar : LL.scalar_t;
  dead : int list;  (** The scope ids of discarded projection operands. *)
  slot_ids : (int * int) list;  (** Slot to the scope id built there. *)
}

(* [prog] over [x], [m] and the loop symbol [i]. [~twin]: every read of [A]'s local is a copy of
   [A]'s scope. [~steps]: the accumulation steps of every sum body (the planted fault drops one). *)
let build ~twin ~steps ~x ~m ~i prog =
  let p = prog.prec in
  let bin op a b = LL.Binop (op, (a, p), (b, p)) in
  let local () =
    let v = node p "wsg_v" in
    virtualize v;
    v
  in
  let sum_scope_over ?id v ~at =
    let id = match id with Some id -> id | None -> LL.get_scope v in
    let k = sym () in
    let step = LL.Set_local (id, bin Ops.Add (LL.Get_local id) (get x [| at k |])) in
    (id, scope_of id (seq (LL.Set_local (id, LL.Constant 0.)) (loop_n k steps step)))
  in
  let sum_scope ~at = sum_scope_over (local ()) ~at in
  let in_range shift k = aff [ (1, i); (1, k) ] shift in
  (* A token is one node; each occurrence is an alpha-equal copy with an id of its own, as the
     pipeline's CSE sees them. The first occurrence in operand order takes the id minted up front,
     which [fA] reads: the CSE keeps the first occurrence. *)
  let toks = Hashtbl.create (module Int) in
  let token t =
    Hashtbl.find_or_add toks t ~default:(fun () ->
        let v = local () in
        (v, LL.get_scope v, ref false))
  in
  let tok ?(nested = false) t =
    let v, first, used = token t in
    (* A copy nested in [rA]'s body never takes the first id: the CSE keeps a body's definitions to
       that body, so the pipeline never reads a nested local from outside it. *)
    let id = if !used || nested then None else Some first in
    if not nested then used := true;
    sum_scope_over ?id v ~at:(in_range (tok_shift t))
  in
  let dead = ref [] and slot_ids = ref [] in
  let reads ~direct =
    let id = LL.get_scope (local ()) in
    let a =
      if direct && not twin then
        let _, first, _ = token 0 in
        LL.Get_local first
      else snd (tok ~nested:true 0)
    in
    ( id,
      scope_of id
        (seq
           (LL.Set_local (id, bin Ops.Mul (LL.Constant 2.) a))
           (LL.Set_local (id, bin Ops.Add (LL.Get_local id) (get x [| aff [ (1, i) ] 4 |])))) )
  in
  let iprec = Ops.index_prec () in
  let lo = lt (embed i) (ic 1) in
  let cond = function
    | Lo -> (lo, iprec)
    | Hi -> (lt (embed i) (ic 3), iprec)
    | Data -> (cmp_at p Ops.Cmplt (get m [| iter i |]) (LL.Constant 1.), p)
    | Proj1 -> (LL.Binop (Ops.Arg1, (lo, iprec), (ic 2, iprec)), iprec)
    | Proj2 -> (LL.Binop (Ops.Arg2, (ic 2, iprec), (lo, iprec)), iprec)
    | Flo -> (cmp_at single Ops.Cmplt (embed i) (LL.Constant 1.), single)
    | Sc -> (cmp_at p Ops.Cmplt (snd (sum_scope ~at:(in_range 0))) (LL.Constant 12.), p)
  in
  let slot s (id, scalar) =
    slot_ids := (s, id.LL.scope_id) :: !slot_ids;
    scalar
  in
  let rec go live = function
    | Slot (s, K) -> LL.Constant (constant s)
    | Slot (s, Fresh) -> slot s (sum_scope ~at:(in_range (fresh_shift s)))
    | Slot (s, Off) ->
        let pass = List.hd_exn live in
        slot s (sum_scope ~at:(fun k -> aff [ (far, i); (1, k) ] (off_base s - (far * pass))))
    | Slot (s, Tok t) -> slot s (tok t)
    | Slot (s, Reads) -> slot s (reads ~direct:false)
    | Slot (s, Fwd) -> slot s (reads ~direct:true)
    | Sel (c, a, b) ->
        let la, lb = split live c in
        let c = cond c in
        let a = go la a in
        let b = go lb b in
        LL.Ternop (Ops.Where, c, (a, p), (b, p))
    | Gate (g, e) ->
        let first =
          match g with
          | Gidx -> bin Ops.Sub (LL.Constant 1.) (embed i)
          | Gdata -> get m [| iter i |]
        in
        let e = go (gate_live live g) e in
        LL.Binop (Ops.Relu_gate, (first, p), (e, p))
    | Keep (first, e) ->
        let kept = go live e in
        (* Never in range: [i] never reaches 7. *)
        let id, discarded = sum_scope ~at:(fun k -> aff [ (far, i); (1, k) ] (-far * 7)) in
        dead := id.LL.scope_id :: !dead;
        if first then LL.Binop (Ops.Arg1, (kept, p), (discarded, p))
        else LL.Binop (Ops.Arg2, (discarded, p), (kept, p))
    | Add (a, b) ->
        let a = go live a in
        let b = go live b in
        bin Ops.Add a b
  in
  let scalar = go all_iterations prog.expr in
  { scalar; dead = !dead; slot_ids = !slot_ids }

(* === Running a batch === *)

type outcome = {
  prog : program;
  refused : string option;
  values_ok : bool;
  twin_ok : bool;
  order_ok : bool;  (** (a) *)
  text_deps_ok : bool;  (** (a'), the text *)
  walk_deps_ok : bool;  (** (a'), the walk against the IR's reads *)
  dead_ok : bool;
  moved_caught : bool option;
      (** The live control of (a'): with the last assignment of a local some other code reads moved
          to the end of the text, the reader flags a read ahead of it. [None]: no such local. *)
  rendered_named : bool;  (** Every rendered scope's local is named in the text. *)
  rendered : int;  (** Scope bodies the kernel holds for the program. *)
  shared : bool;  (** Some rendered scope's local is read where it is not defined. *)
  groups : int;  (** Conditional groups of the statement's emission order. *)
  operand_out_of_order : bool;
  swap_caught : bool;
  dependent_groups : bool;  (** A group-1 definition reads a group-0 local. *)
  swap_breaks_deps : bool;
  arm_placement : AF.placement option;  (** Slot 0's placement, for the blend legs. *)
}

let refused_outcome prog msg =
  {
    prog;
    refused = Some msg;
    values_ok = false;
    twin_ok = false;
    order_ok = false;
    text_deps_ok = false;
    walk_deps_ok = false;
    dead_ok = false;
    moved_caught = None;
    rendered_named = false;
    rendered = 0;
    shared = false;
    groups = 0;
    operand_out_of_order = false;
    swap_caught = false;
    dependent_groups = false;
    swap_breaks_deps = false;
    arm_placement = None;
  }

(* The planted fault: the blocks of conditional groups 0 and 1 swapped. *)
let swap_groups (e : AF.emission) : AF.emission =
  {
    order =
      (fun unit roots ->
        let o = e.order unit roots in
        let in_group g (_, pl) =
          match pl with AF.Conditional h -> h = g | AF.Unconditional -> false
        in
        let g0 = List.filter o ~f:(in_group 0) and g1 = List.filter o ~f:(in_group 1) in
        if List.is_empty g0 || List.is_empty g1 then o
        else
          let rest = List.filter o ~f:(fun d -> not (in_group 0 d || in_group 1 d)) in
          let uncond, later =
            List.partition_tf rest ~f:(fun (_, pl) -> Poly.equal pl AF.Unconditional)
          in
          uncond @ g1 @ g0 @ later);
  }

let recorded name = List.Assoc.find !(Ir.C_syntax.scope_order_oracles ()) name ~equal:String.equal

(* The rendered statement writing row [r] of [out]. *)
let row_statement llc out r =
  let found = ref None in
  walk llc ~on_stmt:(function
    | LL.Set { tn; idcs; llsc; _ } as s
      when Tn.equal tn out && Poly.equal idcs.(0) (Ir.Indexing.Fixed_idx r) ->
        found := Some (s, llsc)
    | _ -> ());
  !found

(* Each scope of [llsc] with what its own body reads -- the [Get_local]s of other locals and the
   scopes nested in it, not what those nested bodies read -- and the [Get_local]s of the statement
   outside every scope body. *)
let scope_reads llsc =
  let gets_of ~self body =
    let gets = ref [] and nested = ref [] in
    walk body ~in_scopes:false ~on_scalar:(function
      | LL.Get_local r when r.LL.scope_id <> self -> gets := r.LL.scope_id :: !gets
      | LL.Local_scope { id = r; _ } -> nested := r.LL.scope_id :: !nested
      | _ -> ());
    (!gets, !nested)
  in
  let acc = ref [] in
  walk_scalar llsc ~on_scalar:(function
    | LL.Local_scope { id; body; _ } ->
        acc := (id.LL.scope_id, gets_of ~self:id.LL.scope_id body) :: !acc
    | _ -> ());
  let top = ref [] in
  walk_scalar llsc ~in_scopes:false ~on_scalar:(function
    | LL.Get_local r -> top := r.LL.scope_id :: !top
    | _ -> ());
  (!acc, !top)

(* Every body ends after the bodies whose locals it reads, in [ends]' order. *)
let deps_respected deps ends =
  let pos = List.mapi ends ~f:(fun k (id, _) -> (id, k)) in
  let at id = List.Assoc.find pos id ~equal:Int.equal in
  List.for_all deps ~f:(fun (a, reads) ->
      List.for_all reads ~f:(fun r ->
          match (at a, at r) with Some pa, Some pr -> pr < pa | _ -> true))

let scope_ids llsc ~dead =
  let ids = ref [] in
  walk_scalar llsc ~on_scope:(fun id ->
      if not (List.mem dead id.LL.scope_id ~equal:Int.equal) then ids := id.LL.scope_id :: !ids);
  List.dedup_and_sort !ids ~compare:Int.compare

let same_values a b = Array.equal Float.equal a b

let batch_nodes prec rows =
  let x = node prec "wsg_x" and m = node prec ~dims:[| n |] "wsg_m" in
  let out = node prec ~dims:[| rows; n |] "wsg_out" in
  List.iter [ x; m; out ] ~f:materialize;
  (x, m, out)

(* Compiles [progs] as one routine and returns each row's values, with the rendered tree, its oracle
   and the kernel text. *)
let compile ~name ~steps progs =
  let prec = (List.hd_exn progs).prec in
  let rows = List.length progs in
  let x, m, out = batch_nodes prec rows in
  let i = sym () in
  let built = List.map progs ~f:(build ~twin:false ~steps ~x ~m ~i) in
  let scoped =
    loop_n i n
      (LL.unflat_lines (List.mapi built ~f:(fun r b -> set out [| fixed r; iter i |] b.scalar)))
    |> LL.eliminate_common_subexpressions
  in
  let raw =
    loop_n i n
      (LL.unflat_lines
         (set out [| fixed 0; iter i |] (get m [| iter i |])
         :: List.init rows ~f:(fun r -> set out [| fixed r; iter i |] (get x [| iter i |]))))
  in
  let o = optimize_scoped ~name ~raw scoped in
  let got =
    List.hd_exn
      (execute ~name o ~seed:[ (x, xs); (m, mask); (out, blank (rows * n)) ] ~read:[ out ])
  in
  let values = List.init rows ~f:(fun r -> Array.sub got ~pos:(r * n) ~len:n) in
  (x, m, out, built, values)

let interpret_twin prog =
  let x, m, out = batch_nodes prog.prec 1 in
  let i = sym () in
  let b = build ~twin:true ~steps:2 ~x ~m ~i prog in
  let memory = interp_memory [ (x, xs); (m, mask) ] in
  match interpret ~strict:true memory (loop_n i n (set out [| fixed 0; iter i |] b.scalar)) with
  | () -> Ok (Hashtbl.find_exn memory out)
  | exception Interp_fault msg -> Error msg

let findings = ref []

let finding prog check =
  let d = descriptor prog in
  if not (List.mem known_findings d ~equal:String.equal) then findings := (d, check) :: !findings;
  Stdio.eprintf "FINDING %s: %s\n%!" d check

(* [source] with the line holding the last assignment of scope [id]'s local moved to its end. *)
let definition_moved_last source id =
  match SE.last_assignment source id with
  | None -> source
  | Some k ->
      let lines = String.split_lines source in
      String.concat ~sep:"\n" (List.filteri lines ~f:(fun j _ -> j <> k) @ [ List.nth_exn lines k ])

(* The text readers on synthetic fragments: each rule flags its violation and passes the nearest
   legitimate text. *)
let reader_controls () =
  let src lines = String.concat ~sep:"\n" lines in
  let read_first = src [ "  y = (v7_n1_a * 2.0);"; "  v7_n1_a = 1.0;" ] in
  let def_first = src [ "  v7_n1_a = 1.0;"; "  y = (v7_n1_a * 2.0);" ] in
  let decl_and_step =
    src
      [
        "  float v7_n1_a = (float)(0);";
        "  v7_n1_a = (float)(0.0);";
        "  for (int32_t i2 = 0; i2 <= 1; ++i2) {";
        "    v7_n1_a = (v7_n1_a + x[i2]);";
        "  }";
        "  y = v7_n1_a;";
      ]
  in
  let never_assigned = src [ "  y = v7_n1_a;" ] in
  (* The exact lines flagged: a reader flagging too much fails as well as one flagging nothing. *)
  let flagged s expected = List.equal Int.equal (reads_ahead s 7) expected in
  p "reader control: a read ahead of the last assignment is flagged" (flagged read_first [ 0 ]);
  p "reader control: the same lines definition-first are not" (flagged def_first []);
  p "reader control: a declaration and an accumulation step are not reads"
    (flagged decl_and_step []);
  p "reader control: a read of a local never assigned is flagged" (flagged never_assigned [ 0 ]);
  p "reader control: a fragment spelling the local names it" (SE.names_local read_first 7);
  p "reader control: v17_ is not a mention of scope 7"
    (not (SE.names_local (src [ "  y = v17_n1_a;"; "  v17_n1_a = xv7_n1_a;" ]) 7))

let check_rows ~name progs (_, _, out, built, values) =
  let source = Generated.main_logic (Generated.read name) in
  let oracle, rendered =
    match recorded name with
    | Some r -> r
    | None -> failwith (name ^ ": the compile recorded no scope emission oracle")
  in
  Ir.C_syntax.scope_order_oracles () := [];
  let emission = LL.Scope_order.emission oracle in
  let ends_of e = SE.body_ends e rendered source in
  let emitted = ends_of (Emitted (if plant_swap then swap_groups emission else emission)) in
  let operand = ends_of Visit and swapped = ends_of (Emitted (swap_groups emission)) in
  List.mapi
    (List.zip_exn (List.zip_exn progs built) values)
    ~f:(fun r ((prog, b), got) ->
      match row_statement rendered out r with
      | None -> refused_outcome prog "the rendered tree lost the program's statement"
      | Some (stmt, llsc) ->
          let ids = scope_ids llsc ~dead:b.dead in
          let mine ends = List.filter ends ~f:(fun (id, _) -> List.mem ids id ~equal:Int.equal) in
          let ends = mine emitted in
          let order = emission.order stmt [ llsc ] in
          let placement id = List.Assoc.find order id ~equal:Int.equal in
          let groups =
            List.filter_map order ~f:(fun (_, pl) ->
                match pl with AF.Conditional g -> Some g | AF.Unconditional -> None)
            |> List.dedup_and_sort ~compare:Int.compare
            |> List.length
          in
          let reads, top_gets = scope_reads llsc in
          let reads = List.filter reads ~f:(fun (a, _) -> List.mem ids a ~equal:Int.equal) in
          let deps = List.map reads ~f:(fun (a, (gets, nested)) -> (a, gets @ nested)) in
          let read_ids = top_gets @ List.concat_map reads ~f:(fun (_, (gets, _)) -> gets) in
          let dependent_groups =
            List.exists deps ~f:(fun (a, reads) ->
                Poly.equal (placement a) (Some (AF.Conditional 1))
                && List.exists reads ~f:(fun r ->
                    Poly.equal (placement r) (Some (AF.Conditional 0))))
          in
          let want = replay prog in
          let twin = interpret_twin prog in
          let outcome =
            {
              prog;
              refused = None;
              values_ok = same_values got want;
              twin_ok = (match twin with Ok t -> same_values t want | Error _ -> false);
              order_ok = SE.in_text_order ends && List.length ends = List.length ids;
              text_deps_ok = List.for_all ids ~f:(fun id -> List.is_empty (reads_ahead source id));
              walk_deps_ok = deps_respected deps ends;
              dead_ok = List.for_all b.dead ~f:(fun id -> not (SE.names_local source id));
              moved_caught =
                List.find ids ~f:(fun id -> List.mem read_ids id ~equal:Int.equal)
                |> Option.map ~f:(fun r ->
                    not (List.is_empty (reads_ahead (definition_moved_last source r) r)));
              rendered_named = List.for_all ids ~f:(SE.names_local source);
              rendered = List.length ids;
              shared = List.exists ids ~f:(fun id -> List.mem read_ids id ~equal:Int.equal);
              groups;
              operand_out_of_order = not (SE.in_text_order (mine operand));
              swap_caught = not (SE.in_text_order (mine swapped));
              dependent_groups;
              swap_breaks_deps = not (deps_respected deps (mine swapped));
              arm_placement =
                Option.bind (List.Assoc.find b.slot_ids 0 ~equal:Int.equal) ~f:placement;
            }
          in
          if not outcome.values_ok then
            finding prog
              (Printf.sprintf "compiled %s, replay %s"
                 (Sexp.to_string (Array.sexp_of_t Float.sexp_of_t got))
                 (Sexp.to_string (Array.sexp_of_t Float.sexp_of_t want)));
          (match twin with
          | Error msg -> finding prog ("the strict twin faulted: " ^ msg)
          | Ok _ when not outcome.twin_ok ->
              finding prog "the strict twin's values differ from the replay"
          | Ok _ -> ());
          if not outcome.order_ok then
            finding prog "the Emitted walk is out of the kernel's text order";
          if not outcome.text_deps_ok then
            finding prog "the text reads a local ahead of its definition";
          if not outcome.walk_deps_ok then
            finding prog "the Emitted walk ends a body before one it reads";
          if not outcome.dead_ok then finding prog "the text names a discarded operand's scope";
          outcome)

let batch_size = 64

let rec run_batch ~name progs =
  Stdio.eprintf "batch %s: %d programs from %s\n%!" name (List.length progs)
    (descriptor (List.hd_exn progs));
  match compile ~name ~steps:2 progs with
  | compiled -> check_rows ~name progs compiled
  | exception e when List.length progs > 1 ->
      Stdio.eprintf "batch %s refused (%s); retrying its programs one at a time\n%!" name
        (Exn.to_string e);
      List.concat_mapi progs ~f:(fun k prog ->
          run_batch ~name:(Printf.sprintf "%s_%d" name k) [ prog ])
  | exception e ->
      let msg = Exn.to_string e in
      finding (List.hd_exn progs) ("the compile refused it: " ^ msg);
      [ refused_outcome (List.hd_exn progs) msg ]

(* === The verdict === *)

(* [--brief]: the values alone, each program named on stderr before it compiles, so a run that
   faults names the program that faulted. *)
let brief = has_arg "--brief"

let dump prog =
  let name = "wsg_only" in
  Stdio.eprintf "compiling %s\n%!" (descriptor prog);
  Generated.arm name;
  let _, _, _, built, values = compile ~name ~steps:2 [ prog ] in
  Ir.C_syntax.scope_order_oracles () := [];
  let show_values v = Sexp.to_string (Array.sexp_of_t Float.sexp_of_t v) in
  Stdio.printf "== %s\nreplay %s\ncompiled %s\n" (descriptor prog)
    (show_values (replay prog))
    (show_values (List.hd_exn values));
  if not brief then
    Stdio.printf "IR: %s\n%s\n"
      (Sexp.to_string_hum (LL.sexp_of_scalar_t (List.hd_exn built).scalar))
      (Generated.main_logic (Generated.read name))

let () =
  let started = Mtime_clock.counter () in
  let seconds () = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
  let batch_seconds = ref [] in
  match only with
  | Some sub ->
      List.iter programs ~f:(fun prog ->
          if String.is_substring (descriptor prog) ~substring:sub then dump prog)
  | None ->
      let by_prec prec = List.filter programs ~f:(fun p -> Ops.equal_prec p.prec prec) in
      Stdio.printf "%d programs, at most %d scope-bearing slots\n" (List.length programs) max_scoped;
      List.iter skeletons ~f:(fun sk ->
          let count prec = List.count (by_prec prec) ~f:(fun p -> String.equal p.skeleton sk.sk) in
          Stdio.printf "%s: f32=%d i32=%d\n" sk.sk (count single) (count Ops.int32));
      let outcomes =
        List.concat_map [ single; Ops.int32 ] ~f:(fun prec ->
            List.chunks_of (by_prec prec) ~length:batch_size
            |> List.concat_mapi ~f:(fun k progs ->
                let name = Printf.sprintf "wsg_%s%d" (prec_name prec) k in
                let before = seconds () in
                let outcomes = run_batch ~name progs in
                batch_seconds := (name, seconds () -. before) :: !batch_seconds;
                outcomes))
      in
      let checked =
        List.filter outcomes ~f:(fun o ->
            not (List.mem known_findings (descriptor o.prog) ~equal:String.equal))
      in
      let total = List.length programs in
      let floor = total / 2 in
      let floor_batches = total / batch_size / 2 in
      p_all ~min:floor "no compile refused a generated program" checked ~f:(fun o ->
          Option.is_none o.refused);
      p_all ~min:floor "every program's compiled values are the replay's" checked ~f:(fun o ->
          o.values_ok);
      p_all ~min:floor "every program's strict-interpreted twin computes the replay's values"
        checked ~f:(fun o -> o.twin_ok);
      p_all ~min:floor
        "every program's Emitted walk ends its scope bodies in the kernel's text order" checked
        ~f:(fun o -> o.order_ok);
      p_all ~min:floor "every program's text reads each scope local only after its definition"
        checked ~f:(fun o -> o.text_deps_ok);
      p_all ~min:floor "every program's Emitted walk ends a body after the bodies it reads" checked
        ~f:(fun o -> o.walk_deps_ok);
      p_all ~min:100 "no kernel names the scope of a discarded projection operand"
        (List.filter checked ~f:(fun o -> String.is_prefix o.prog.skeleton ~prefix:"keep"))
        ~f:(fun o -> o.dead_ok);
      p_all ~min:floor "every rendered scope's local is named in the kernel" checked ~f:(fun o ->
          o.rendered_named);
      p_all ~min:1000
        "control: with a read local's definition moved after its readers, the text reader flags \
         the read"
        (List.filter_map checked ~f:(fun o -> o.moved_caught))
        ~f:Fn.id;
      case "control: the text readers on synthetic fragments" reader_controls;
      (* Coverage floors: what the kernels rendered, at counts every backend reaches. *)
      let at_least k label xs ~f = pf "%s (at least %d)" label k (List.count xs ~f >= k) in
      at_least floor "programs whose kernel holds a scope body" checked ~f:(fun o -> o.rendered > 0);
      at_least 3000 "programs with a conditional group of definitions" checked ~f:(fun o ->
          o.groups >= 1);
      at_least 1000 "programs with two conditional groups" checked ~f:(fun o -> o.groups >= 2);
      at_least 1500 "programs reading a scope local where it is not defined" checked ~f:(fun o ->
          o.shared);
      (* The negative controls. *)
      p_all ~min:1000 "control: the planted group swap is out of the kernel's text order"
        (List.filter checked ~f:(fun o -> o.groups >= 2))
        ~f:(fun o -> o.swap_caught);
      p_all ~min:200
        "control: where a group-1 body reads a group-0 local, the swap breaks the read order"
        (List.filter checked ~f:(fun o -> o.dependent_groups))
        ~f:(fun o -> o.swap_breaks_deps);
      at_least 1000 "control: programs whose operand order is not the kernel's order" checked
        ~f:(fun o -> o.operand_out_of_order);
      case "control: a dropped accumulation step" (fun () ->
          List.iter [ single; Ops.int32 ] ~f:(fun prec ->
              let progs = List.take (by_prec prec) batch_size in
              let name = "wsg_drop_" ^ prec_name prec in
              let _, _, _, _, values = compile ~name ~steps:1 progs in
              Ir.C_syntax.scope_order_oracles () := [];
              let changed =
                List.filter_map (List.zip_exn progs values) ~f:(fun (prog, got) ->
                    let want = replay prog in
                    if same_values (replay ~steps:1 prog) want then None else Some (got, want))
              in
              p_all ~min:10
                (prec_name prec
               ^ ": the compiled values differ from the replay wherever it says they change")
                changed
                ~f:(fun (got, want) -> not (same_values got want))));
      (* Where a data select's arm definition renders is the backend's fact. *)
      let arm_cases =
        List.filter checked ~f:(fun o ->
            Ops.is_float o.prog.prec
            &&
            match o.prog.expr with
            (* A fresh scope: a token may be read outside the arm too, which places it ahead. *)
            | Add (Sel (Data, Slot (0, Fresh), _), _) -> true
            | _ -> false)
      in
      let blends = facts.Ir.Backend_intf.blends_value_selects in
      in_leg ~leg:(Verdict.leg ~when_:blends ~on:backend_name ())
        "where the backend blends a data select, its arm definitions render unconditionally"
        (fun label ->
          p_all ~min:5 label arm_cases ~f:(fun o ->
              Poly.equal o.arm_placement (Some AF.Unconditional)));
      in_leg ~leg:(Verdict.leg ~when_:(not blends) ~on:backend_name ())
        "where the backend short-circuits a data select, its arm definitions render under its \
         condition" (fun label ->
          p_all ~min:5 label arm_cases ~f:(fun o ->
              match o.arm_placement with Some (AF.Conditional _) -> true | _ -> false));
      Stdio.eprintf
        "where_scope_generator on %s: %d programs, %d findings, %d with conditional groups, %d \
         with two, %d with dependent groups, %d out of operand order, %d reading a local where it \
         is not defined, %.2f s in all (not part of the golden)\n\
         %!"
        backend_name total (List.length !findings)
        (List.count checked ~f:(fun o -> o.groups >= 1))
        (List.count checked ~f:(fun o -> o.groups >= 2))
        (List.count checked ~f:(fun o -> o.dependent_groups))
        (List.count checked ~f:(fun o -> o.operand_out_of_order))
        (List.count checked ~f:(fun o -> o.shared))
        (seconds ());
      p_empty ~min:floor "every disagreement is a known finding" ~over:checked !findings;
      (* The wall-time claim is relative to this run, so a slow or loaded box cannot fail it: a cost
         blow-up in part of the grammar lands in the batches holding it, and stands out against the
         run's own median batch (the slowest is 1.5 medians on cc, multidev_cc and Metal); so does a
         refused batch's retry, one compile per program, where a compile's fixed cost dominates (cc:
         about 26 medians). A slowdown every batch shares is the box's or the backend's, not this
         generator's to judge: no absolute bound or single-compile unit holds across boxes and
         backends (multidev_cc on minix took 185 s, Metal 20 s; Metal's shader cache makes a lone
         compile 4-13 ms). *)
      let times = List.map !batch_seconds ~f:snd |> List.sort ~compare:Float.compare in
      let median = List.nth_exn times (List.length times / 2) in
      let slowest_name, slowest =
        Option.value_exn
          (List.max_elt !batch_seconds ~compare:(fun (_, a) (_, b) -> Float.compare a b))
      in
      Stdio.eprintf
        "where_scope_generator batch times: median %.3f s, slowest %.3f s (%s) (not part of the \
         golden)\n\
         %!"
        median slowest slowest_name;
      p_all ~min:floor_batches "no batch took more than 10 times the median batch" !batch_seconds
        ~f:(fun (_, t) -> Float.(t <= 10. * median))
