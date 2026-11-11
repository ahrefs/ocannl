(* An exhaustive parity oracle for the inliner over small hand-built programs (gh-ocannl-1964).

   Three workers found inliner holes by reasoning about a sentence of a PR body rather than from any
   row they had (gh-ocannl-1896, gh-ocannl-1898, gh-ocannl-1953). This test turns that reasoning
   into a table: it enumerates a stated finite grammar of programs (shapes up to [2; 3] and [3; 2],
   per-axis, flattened and trailing-[Sub_axis] layouts on either side of a read, fixed, iterator,
   offset, reversed, strided and two-symbol positions, loops over all or part of a coordinate, with
   and without [Zero_out], one or two components, reads within and past the writes) and holds every
   one of them to one semantics.

   Tier A is the enumeration. Each program goes through the real [optimize] in both arms, the
   ordinary one and the materialized one: [~materialized] for a node the program decides, a twin
   over fresh materialized nodes for one that skipped the coverage rule, which re-specializing the
   same node would contradict. [Ll_test.interpret] evaluates what each arm's whole pipeline left,
   and each [out] must equal [replay]'s: a walk of the program's descriptor in integer arithmetic,
   sharing nothing with the optimizer, [Affine] or the interpreter, so a defect both arms share
   cannot agree with itself.

   Materializing is always a right answer and a wrong value never is, so nothing here pins an
   individual placement. The census on stdout is aggregate: a change moving the boundary reads as a
   diff of that table, and the floor claims say the inliner did inline in each family. A node that
   skipped the coverage rule cannot materialize, so it may refuse instead, with a recognized message
   ([recognized_refusals]) and nothing else. An uncovered read owns entry values no replay can
   produce, so such a node reading one MUST refuse; distinct entries are what tell the two apart.
   The virtual arm answers for faults a compiled run would swallow, too: [Interp_fault] is a
   finding. Both arms evaluate under [~strict:true], so a scope the code hoists out of a discarded
   [Where] arm is held to its node's bounds as well (gh-ocannl-2077; [strict_probe_controls] pin the
   probe; [virtual_rejection_boundary]'s [row_guarded_scope_read] pins the inliner's arm guard). The
   [src=reduce] family makes [src] itself a virtual two-term reduction over a materialized input:
   scopes then sit inside the replay's guarded arms throughout the table, including out-of-range
   substitutions the inliner must protect with [guard_hoisted].

   Tier B is the anchor. An interpreter agreeing only with itself would prove nothing, so a sample
   across the outcome buckets also compiles and runs on the selected backend, in both arms, against
   the same replay.

   A program's descriptor is its name. [--only <substring>] prints the raw and optimized code of the
   programs whose descriptor contains it and checks nothing, for a finding printed on stderr;
   [--widen] lifts the pools below, for someone changing the inliner. Both are for a direct run of
   the built executable from [_build/default/test/operations], which finds [ocannl_config] there;
   [--widen] also has an alias of its own, [@test/operations/slow-virtual_inliner_oracle-widen],
   which the daily sweep's [cc-oracle-widen] unit builds (tools/sweep.sh) and [tools/test-run.sh
   affected] names to a change reaching the inliner. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:5_000_000 ~dims:[| 3 |] ()

(* === The grammar === *)

(* One index coordinate over the access's symbol slots: [Lin (terms, offset)] is the sum of [coeff *
   slot], [Launch] the launch parameter, read only. *)
type coord = Fix of int | Lin of (int * int) list * int | Launch of int
type pos = Coord of coord | Sub

(* [ranges] is the loop nest, outermost first: slot and inclusive bounds. *)
type access = {
  layout : string;
  pos : pos array;
  extents : int array;
  ranges : (int * (int * int)) list;
}

type kind = Store | Accumulate | Reduce
type comp = { write : access; kind : kind; complex : bool }
type step = Zero | Write of int | Read of int
type standing = Undecided | Declared | Inherited

type program = {
  dims : int array;
  comps : comp array;
  read_accesses : access array;
  steps : step list;
  standing : standing;
  nested_src : bool;
}

let kind_name = function Store -> "store" | Accumulate -> "accumulate" | Reduce -> "reduce"

let standing_name = function
  | Undecided -> "undecided"
  | Declared -> "declared"
  | Inherited -> "inherited"

let show_coord names = function
  | Fix k -> Int.to_string k
  | Launch k -> Printf.sprintf "launch=%d" k
  | Lin (terms, off) ->
      let term (c, slot) =
        let name = String.make 1 names.[slot] in
        if c = 1 then name else Printf.sprintf "%d%s" c name
      in
      String.concat ~sep:"+" (List.map terms ~f:term)
      ^ if off = 0 then "" else Printf.sprintf "%+d" off

let show_access names a =
  let pos =
    String.concat ~sep:";"
    @@ List.map (Array.to_list a.pos) ~f:(function Sub -> "_" | Coord c -> show_coord names c)
  and ranges =
    String.concat ~sep:","
    @@ List.map a.ranges ~f:(fun (slot, (lo, hi)) -> Printf.sprintf "%c:%d..%d" names.[slot] lo hi)
  in
  Printf.sprintf "%s[%s]{%s}" a.layout pos ranges

let show_step = function
  | Zero -> "Z"
  | Write i -> Printf.sprintf "P%d" i
  | Read i -> Printf.sprintf "R%d" i

let show_program prog =
  let dims = String.concat ~sep:"x" (List.map (Array.to_list prog.dims) ~f:Int.to_string) in
  let comps =
    Array.to_list prog.comps
    |> List.mapi ~f:(fun i c ->
        Printf.sprintf "P%d=%s:%s%s" i (kind_name c.kind) (show_access "st" c.write)
          (if c.complex then ":src" else ""))
  and reads =
    Array.to_list prog.read_accesses
    |> List.mapi ~f:(fun i r -> Printf.sprintf "R%d=%s" i (show_access "uv" r))
  in
  String.concat ~sep:" "
    (standing_name prog.standing :: dims
     :: (if prog.nested_src then "src=reduce" else "src=input")
     :: comps
    @ reads
    @ [ String.concat ~sep:";" (List.map prog.steps ~f:show_step) ])

(* Evaluating a descriptor: integer arithmetic only. *)
let coord_value env = function
  | Fix k | Launch k -> k
  | Lin (terms, off) ->
      List.fold terms ~init:off ~f:(fun acc (coeff, slot) -> acc + (coeff * env.(slot)))

let cell_of ~dims pos env =
  let n = Array.length dims in
  let stride = Array.create ~len:n 1 in
  for ax = n - 2 downto 0 do
    stride.(ax) <- stride.(ax + 1) * dims.(ax + 1)
  done;
  Array.foldi pos ~init:0 ~f:(fun ax acc -> function
    | Sub -> acc
    | Coord c -> acc + (coord_value env c * stride.(ax)))

let slots_of pos =
  Array.to_list pos
  |> List.concat_map ~f:(function
    | Coord (Lin (terms, _)) -> List.map terms ~f:snd
    | Coord (Fix _ | Launch _) | Sub -> [])
  |> List.dedup_and_sort ~compare:Int.compare

let iter_box ranges ~f =
  let env = Array.create ~len:3 0 in
  let rec go = function
    | [] -> f env
    | (slot, (lo, hi)) :: rest ->
        for i = lo to hi do
          env.(slot) <- i;
          go rest
        done
  in
  go ranges

(* Every coordinate of the whole loop box stays inside its extent. A read "past the writes" is past
   them and inside the node: a read outside the node has no meaning to preserve. *)
let in_bounds a =
  let ok = ref true in
  iter_box a.ranges ~f:(fun env ->
      Array.iteri a.pos ~f:(fun i -> function
        | Sub -> ()
        | Coord c ->
            let v = coord_value env c in
            if v < 0 || v >= a.extents.(i) then ok := false));
  !ok

(* What one coordinate of [extent] cells may be, one form per path the inliner takes through a
   position: a fixed cell; an iterator (pass 1); an offset and a reversal (unit solving, both
   signs); a stride (no unit coefficient, so an equality guard over a kept loop); and on a
   coordinate wide enough for one, a two-symbol form (a structural match, a unit solve, the
   store-time injectivity refusal). A form using slot 1 alone is the rename of one using slot 0, so
   {!accesses} drops it. *)
let coord_forms ~launch extent =
  let fixes = List.init extent ~f:(fun k -> Fix k) in
  let one slot =
    [
      Lin ([ (1, slot) ], 0);
      Lin ([ (1, slot) ], 1);
      Lin ([ (-1, slot) ], extent - 1);
      Lin ([ (2, slot) ], 0);
    ]
  in
  let two =
    if extent >= 3 then
      Lin ([ (1, 0); (1, 1) ], 0)
      ::
      (if extent >= 4 then
         [ Lin ([ (2, 0); (1, 1) ], 0); Lin ([ (3, 0); (1, 1) ], 0); Lin ([ (1, 0); (2, 1) ], 0) ]
       else [])
    else []
  in
  fixes @ one 0 @ one 1 @ two @ if launch then List.init extent ~f:(fun k -> Launch k) else []

let layouts dims =
  match dims with
  | [| d |] -> [ ("axes", [| d |], fun cs -> [| Coord (List.hd_exn cs) |]) ]
  | [| d0; d1 |] ->
      [
        ("axes", [| d0; d1 |], fun cs -> Array.of_list_map cs ~f:(fun c -> Coord c));
        ("flat", [| d0 * d1 |], fun cs -> [| Sub; Coord (List.hd_exn cs) |]);
        ("trail", [| d0 |], fun cs -> [| Coord (List.hd_exn cs); Sub |]);
      ]
  | _ -> invalid_arg "layouts: rank"

let loop_ranges = [ (0, 0); (0, 1); (0, 2); (1, 1); (1, 2) ]

let accesses ~launch dims =
  List.concat_map (layouts dims) ~f:(fun (layout, coord_extents, assemble) ->
      let extents =
        match (layout, dims) with
        | "flat", _ -> [| 0; coord_extents.(0) |]
        | "trail", _ -> [| coord_extents.(0); 0 |]
        | _ -> coord_extents
      in
      let rec tuples = function
        | [] -> [ [] ]
        | e :: rest ->
            let tails = tuples rest in
            List.concat_map (coord_forms ~launch e) ~f:(fun c ->
                List.map tails ~f:(fun t -> c :: t))
      in
      List.concat_map
        (tuples (Array.to_list coord_extents))
        ~f:(fun cs ->
          let pos = assemble cs in
          let launches = Array.count pos ~f:(function Coord (Launch _) -> true | _ -> false) in
          match slots_of pos with
          | _ when launches > 1 -> []
          | [ 1 ] -> []
          | slots ->
              let rec nests = function
                | [] -> [ [] ]
                | slot :: rest ->
                    let tails = nests rest in
                    List.concat_map loop_ranges ~f:(fun r ->
                        List.map tails ~f:(fun t -> (slot, r) :: t))
              in
              List.filter_map (nests slots) ~f:(fun ranges ->
                  let a = { layout; pos; extents; ranges } in
                  Option.some_if (in_bounds a) a)))

(* === Instantiating a descriptor === *)

type built = {
  x : Ir.Tnode.t;
  src : Ir.Tnode.t;
  inputs : (Ir.Tnode.t * float array) list;
  outs : Ir.Tnode.t array;
  routines : LL.t list;
  static_indices : Idx.static_symbol list;
  bindings : (unit -> unit) Idx.bindings;
  launch : (Idx.static_symbol * int) list;
}

let src_dims = [| 4; 4 |]
let src_cells = Array.init 16 ~f:(fun i -> 5000. +. (7. *. Float.of_int i))

(* Distinct along all three producer axes, with a two-term reduction exact in f32. *)
let src_input_cells = Array.init 32 ~f:(fun i -> Float.of_int (1 + (3 * i)))

let source_cells prog =
  if prog.nested_src then
    Array.init 16 ~f:(fun i ->
        let sum = ref 0. in
        for k = 0 to 1 do
          sum := !sum +. src_input_cells.((2 * i) + k)
        done;
        !sum)
  else src_cells

let entries prog =
  Array.init (Array.fold prog.dims ~init:1 ~f:( * )) ~f:(fun i -> -.(100. +. Float.of_int i))

let out_len a = List.fold a.ranges ~init:1 ~f:(fun acc (_, (lo, hi)) -> acc * (hi - lo + 1))

(* The output offset of one read iteration, in the nest's own order, and the term the consumer adds
   so that equal producer values stay distinct cells. *)
let out_terms a =
  let _, terms, off =
    List.fold_right a.ranges ~init:(1, [], 0) ~f:(fun (slot, (lo, hi)) (stride, terms, off) ->
        (stride * (hi - lo + 1), (stride, slot) :: terms, off - (stride * lo)))
  in
  (terms, off)

let consumer_base = 20000.

(* Values identify the component, every symbol of its iteration, and stay small integers: exact in
   the single precision Tier B stores them at, off the entries (negative) and the sentinel. *)
let write_value ~src_values ~comp_i env ~k ~used c =
  let base = Float.of_int (1 + (100 * comp_i)) in
  let at slot = if List.mem used slot ~equal:Int.equal then env.(slot) else 0 in
  (if c.complex then src_values.((4 * at 0) + at 1) else Float.of_int ((10 * at 0) + at 1))
  +. base
  +. (1000. *. Float.of_int k)

(* The reference: the program's meaning, by its descriptor. *)
let replay prog =
  let cells = entries prog and src_values = source_cells prog in
  let outs = Array.map prog.read_accesses ~f:(fun a -> blank (out_len a)) in
  let apply_write comp_i c =
    let used = slots_of c.write.pos in
    let at env = cell_of ~dims:prog.dims c.write.pos env in
    iter_box c.write.ranges ~f:(fun env ->
        match c.kind with
        | Store -> cells.(at env) <- write_value ~src_values ~comp_i env ~k:0 ~used c
        | Accumulate ->
            cells.(at env) <- cells.(at env) +. write_value ~src_values ~comp_i env ~k:0 ~used c
        | Reduce ->
            for k = 0 to 1 do
              cells.(at env) <- cells.(at env) +. write_value ~src_values ~comp_i env ~k ~used c
            done)
  in
  let apply_read i a =
    let terms, off = out_terms a in
    iter_box a.ranges ~f:(fun env ->
        let o = coord_value env (Lin (terms, off)) in
        outs.(i).(o) <-
          cells.(cell_of ~dims:prog.dims a.pos env) +. consumer_base +. (300. *. Float.of_int o))
  in
  List.iter prog.steps ~f:(function
    | Zero -> Array.fill cells ~pos:0 ~len:(Array.length cells) 0.
    | Write i -> apply_write i prog.comps.(i)
    | Read i -> apply_read i prog.read_accesses.(i));
  outs

let build ?(materialized_src = false) prog ~declare =
  let x = mk ~dims:prog.dims "x" and src = mk ~dims:src_dims "src" in
  let inputs, source =
    if prog.nested_src then (
      if materialized_src then materialize src else virtualize src;
      let input = mk ~dims:[| 4; 4; 2 |] "src_input" in
      materialize input;
      let i = sym () and j = sym () and k = sym () in
      let idcs = [| iter i; iter j |] in
      ( [ (input, src_input_cells) ],
        loop_n i 4
          (loop_n j 4
             (seq
                (set src idcs (c 0.))
                (loop_n k 2
                   (set src idcs (add (get src idcs) (get input [| iter i; iter j; iter k |]))))))
      ))
    else (
      materialize src;
      ([ (src, src_cells) ], LL.Noop))
  in
  declare x;
  let static = ref None in
  let static_of k =
    match !static with
    | Some (s, _, _) -> s
    | None ->
        (* No [~static_range]: an unknown range is the arm of [needs_range_guard] under test. *)
        let s, bindings = Idx.get_static_symbol Idx.Empty in
        static := Some (s, k, bindings);
        s
  in
  let index syms = function
    | Sub -> Idx.Sub_axis
    | Coord (Fix k) -> fixed k
    | Coord (Launch k) -> iter (static_of k).Idx.static_symbol
    | Coord (Lin ([ (1, slot) ], 0)) -> iter syms.(slot)
    | Coord (Lin (terms, off)) ->
        aff (List.map terms ~f:(fun (coeff, slot) -> (coeff, syms.(slot)))) off
  in
  let nest syms a body =
    List.fold_right a.ranges ~init:body ~f:(fun (slot, (lo, hi)) body ->
        loop ~from_:lo ~upto:hi syms.(slot) body)
  in
  let fresh () = Array.init 3 ~f:(fun _ -> sym ()) in
  let write_stmt comp_i comp =
    let syms = fresh () in
    let used = slots_of comp.write.pos in
    let idcs = Array.map comp.write.pos ~f:(index syms) in
    let at slot = if List.mem used slot ~equal:Int.equal then iter syms.(slot) else fixed 0 in
    let value =
      let of_slots =
        if comp.complex then get src [| at 0; at 1 |]
        else
          let tens = if List.mem used 0 ~equal:Int.equal then mul (c 10.) (embed syms.(0)) else c 0.
          and ones = if List.mem used 1 ~equal:Int.equal then embed syms.(1) else c 0. in
          add tens ones
      in
      add (c (Float.of_int (1 + (100 * comp_i)))) of_slots
    in
    let body =
      match comp.kind with
      | Store -> set x idcs value
      | Accumulate -> set x idcs (add (get x idcs) value)
      | Reduce ->
          let k = syms.(2) in
          loop_n k 2 (set x idcs (add (get x idcs) (add value (mul (c 1000.) (embed k)))))
    in
    nest syms comp.write body
  in
  let outs =
    Array.mapi prog.read_accesses ~f:(fun i a ->
        mk ~dims:[| out_len a |] (Printf.sprintf "out%d" i))
  in
  Array.iter outs ~f:materialize;
  let read_stmt i a =
    let syms = fresh () in
    let terms, off = out_terms a in
    let o = aff (List.map terms ~f:(fun (coeff, slot) -> (coeff, syms.(slot)))) off in
    let o = if List.is_empty terms then fixed 0 else o in
    let value =
      add
        (get x (Array.map a.pos ~f:(index syms)))
        (add (c consumer_base) (mul (c 300.) (embed_idx o)))
    in
    nest syms a (set outs.(i) [| o |] value)
  in
  let stmt = function
    | Zero -> zero x
    | Write i -> write_stmt i prog.comps.(i)
    | Read i -> read_stmt i prog.read_accesses.(i)
  in
  let chain steps = List.reduce_exn (List.map steps ~f:stmt) ~f:seq in
  let routines =
    match prog.standing with
    | Undecided | Declared -> [ chain prog.steps ]
    | Inherited ->
        let is_read = function Read _ -> true | Zero | Write _ -> false in
        let before, after = List.split_while prog.steps ~f:(Fn.non is_read) in
        [ chain before; chain after ]
  in
  let routines =
    match routines with [] -> assert false | first :: rest -> seq source first :: rest
  in
  match !static with
  | None ->
      { x; src; inputs; outs; routines; static_indices = []; bindings = Idx.Empty; launch = [] }
  | Some (s, k, bindings) ->
      { x; src; inputs; outs; routines; static_indices = [ s ]; bindings; launch = [ (s, k) ] }

(* === Running both arms === *)

let optimize_routines ?materialized b =
  let ctx = LL.empty_optimize_ctx () in
  List.map b.routines ~f:(fun llc ->
      optimize_in ?materialized ~static_indices:b.static_indices ctx ~name:"vio" llc)

let evaluate prog b (os : LL.optimized list) =
  let memory =
    interp_memory
      (((b.x, entries prog) :: b.inputs)
      @ (if prog.nested_src then [ (b.src, blank 16) ] else [])
      @ List.mapi (Array.to_list b.outs) ~f:(fun i out ->
          (out, blank (out_len prog.read_accesses.(i)))))
  in
  List.iter os ~f:(fun o -> interpret ~strict:true ~launch:b.launch memory o.LL.llc);
  Array.map b.outs ~f:(Hashtbl.find_exn memory)

let same_outs a b = Array.equal (Array.equal Float.equal) a b
let show_outs outs = Sexp.to_string (Array.sexp_of_t (Array.sexp_of_t Float.sexp_of_t) outs)

let outcome_of b (os : LL.optimized list) =
  let o = List.last_exn os in
  if known_virtual o b.x then
    let guards = List.sum (module Int) os ~f:(fun o -> count_where o) in
    if guards > 0 then "inlined-guarded" else "inlined"
  else
    "materialized:"
    ^ Option.value_map (rejection_code o b.x) ~default:"undecided" ~f:Ir.Tnode.provenance_to_string

(* The known refusals for a node with no materialization fallback: a census label paired with its
   diagnostic fragment. A raw placement collision also names the code, so require the diagnostic's
   rejection phrase for the consumption- and store-time legality refusals. *)
let recognized_refusals =
  [
    ("13:call-site-index-mismatch", "rejection 13:call-site-index-mismatch:");
    ("157:guard-fallback-unproven", "rejection 157:guard-fallback-unproven:");
    ("158:accumulation-without-init", "rejection 158:accumulation-without-init:");
    ("stored a computation to inline", "stored a computation to inline");
    ("10:escaping-index-symbol", "rejection 10:escaping-index-symbol:");
    ("51:affine-not-injective", "rejection 51:affine-not-injective:");
  ]

let census = Hashtbl.create (module String)
let finding_count = ref 0
let programs_run = ref 0
let guarded_scope_programs = Hashtbl.create (module String)
let finding_classes = Hashtbl.create (module String)

let finding prog reason =
  Int.incr finding_count;
  let cls =
    String.prefix reason 24 ^ " " ^ standing_name prog.standing ^ " "
    ^ (String.concat ~sep:","
      @@ Array.to_list (Array.map prog.comps ~f:(fun c -> c.write.layout ^ ":" ^ kind_name c.kind))
      )
    ^ "/" ^ prog.read_accesses.(0).layout
    ^ if prog.nested_src then " src=reduce" else ""
  in
  Hashtbl.incr finding_classes cls;
  if Hashtbl.find_exn finding_classes cls <= 2 then
    Stdio.eprintf "FINDING %s: %s\n%!" (show_program prog) (String.prefix reason 400)

let family prog =
  Printf.sprintf "comps=%d %s %s" (Array.length prog.comps) (kind_name prog.comps.(0).kind)
    (standing_name prog.standing)
  ^ if prog.nested_src then " src=reduce" else ""

(* Per bucket, the first program that fell in it: the candidates for Tier B. A program of a family
   the compiled leg cannot seed (a two-routine one) is not one. *)
let anchors = Hashtbl.create (module String)
let guarded_anchors = Hashtbl.create (module String)

let has_guarded_scope os =
  List.exists os ~f:(fun o ->
      let found = ref false in
      walk o.LL.llc ~on_scalar:(function
        | LL.Ternop (Ops.Where, _, (a, _), (b, _)) ->
            List.iter [ a; b ] ~f:(fun arm -> walk_scalar arm ~on_scope:(fun _ -> found := true))
        | _ -> ());
      !found)

let anchor_key ?outcome prog =
  let layouts = prog.comps.(0).write.layout ^ "/" ^ prog.read_accesses.(0).layout in
  String.concat ~sep:" " ([ family prog ] @ Option.to_list outcome @ [ layouts ])

let sorted_values tbl =
  Hashtbl.to_alist tbl
  |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.map ~f:snd

let check_program prog =
  let want = replay prog in
  let tally outcome =
    Hashtbl.incr census (family prog ^ " | " ^ outcome);
    match prog.standing with
    | Inherited -> ()
    | Undecided | Declared ->
        ignore
          (Hashtbl.add anchors ~key:(anchor_key ~outcome prog) ~data:prog : [ `Ok | `Duplicate ])
  in
  let judge label thunk =
    match thunk () with
    | got ->
        if not (same_outs got want) then
          finding prog
            (Printf.sprintf "%s differs from the replay: %s vs %s" label (show_outs got)
               (show_outs want))
    | exception Interp_fault msg -> finding prog (label ^ " faulted: " ^ msg)
    | exception e -> finding prog (label ^ " raised: " ^ Exn.to_string e)
  in
  let note_guarded_scope os =
    if prog.nested_src && has_guarded_scope os then (
      Hashtbl.incr guarded_scope_programs (family prog);
      match prog.standing with
      | Inherited -> ()
      | Undecided | Declared ->
          ignore
            (Hashtbl.add guarded_anchors ~key:(anchor_key prog) ~data:prog : [ `Ok | `Duplicate ]))
  in
  let no_read b os =
    if prog.nested_src then (
      if not (known_virtual (List.last_exn os) b.src) then
        finding prog "the declared virtual source lost its placement";
      if List.exists os ~f:(fun o -> count_get o b.src > 0) then
        finding prog "the declared virtual source left a read");
    if known_virtual (List.last_exn os) b.x && List.exists os ~f:(fun o -> count_get o b.x > 0) then
      finding prog "a virtual node left a read"
  in
  match prog.standing with
  | Undecided -> (
      let b = build prog ~declare:ignore in
      match optimize_routines b with
      | exception e ->
          tally "raised";
          finding prog ("optimize raised: " ^ Exn.to_string e)
      | os ->
          tally (outcome_of b os);
          no_read b os;
          note_guarded_scope os;
          judge "virtual arm" (fun () -> evaluate prog b os);
          judge "materialized arm" (fun () ->
              if prog.nested_src then
                let twin = build ~materialized_src:true prog ~declare:materialize in
                evaluate prog twin (optimize_routines twin)
              else evaluate prog b (optimize_routines ~materialized:[ b.x ] b)))
  | Declared | Inherited -> (
      let twin = build ~materialized_src:true prog ~declare:materialize in
      judge "materialized twin" (fun () -> evaluate prog twin (optimize_routines twin));
      let b =
        build prog ~declare:(match prog.standing with Declared -> virtualize | _ -> ignore)
      in
      match optimize_routines b with
      | exception Utils.User_error msg -> (
          match
            List.find recognized_refusals ~f:(fun (_, substring) ->
                String.is_substring msg ~substring)
          with
          | Some (label, _) -> tally ("refused:" ^ label)
          | None ->
              tally "raised";
              finding prog ("an unrecognized refusal: " ^ msg))
      | exception e ->
          tally "raised";
          finding prog ("optimize raised: " ^ Exn.to_string e)
      | os ->
          tally (outcome_of b os);
          no_read b os;
          note_guarded_scope os;
          judge "virtual arm" (fun () -> evaluate prog b os))

(* [--only <substring>]: print the raw and optimized code of the programs whose descriptor contains
   it, and check nothing. *)
let only =
  let argv = Sys.get_argv () in
  Array.findi argv ~f:(fun _ a -> String.equal a "--only")
  |> Option.map ~f:(fun (i, _) -> argv.(i + 1))

let dump prog =
  Stdio.eprintf "PROGRAM %s\nreplay %s\n" (show_program prog) (show_outs (replay prog));
  let b = build prog ~declare:(match prog.standing with Declared -> virtualize | _ -> ignore) in
  List.iter b.routines ~f:(fun llc ->
      Stdio.eprintf "raw:\n%s\n" (Sexp.to_string_hum (LL.sexp_of_t llc)));
  let os = optimize_routines b in
  Stdio.eprintf "outcome %s\n" (outcome_of b os);
  List.iter os ~f:(fun o ->
      Stdio.eprintf "optimized:\n%s\n%!" (Sexp.to_string_hum (LL.sexp_of_t o.LL.llc)))

let check prog =
  Int.incr programs_run;
  match only with
  | Some sub -> if String.is_substring (show_program prog) ~substring:sub then dump prog
  | None -> check_program prog

(* === The families ===

   The whole cross of every write with every read is 3 to 4 million programs, minutes of a test run,
   so the default run crosses what it can afford to and states what it does not. On a shape with at
   most 20000 pairs, the full cross; on every other, each write against the core reads and each read
   against the core writes. [--widen] runs the full cross on every shape and unthins the pools, for
   someone changing the inliner. *)

let shapes =
  [
    [| 1 |];
    [| 2 |];
    [| 3 |];
    [| 1; 1 |];
    [| 1; 2 |];
    [| 2; 1 |];
    [| 1; 3 |];
    [| 2; 2 |];
    [| 2; 3 |];
    [| 3; 2 |];
  ]

let numel dims = Array.fold dims ~init:1 ~f:( * )

(* A loop that never runs: the one write that leaves the node with no stored computation. *)
let dead_write dims =
  let pos = Array.mapi dims ~f:(fun i _ -> Coord (if i = 0 then Lin ([ (1, 0) ], 0) else Fix 0)) in
  { layout = "axes"; pos; extents = dims; ranges = [ (0, (0, -1)) ] }

(* An access of the pools: bare iterators, fixed cells, launch parameters and two-symbol forms, the
   positions the rest of the grammar varies. One per layout, position shape and way of sitting in
   the node: all of it, part of it from cell 0, part of it from later on. *)
let core ~dims xs =
  let position = function
    | Sub -> Some "_"
    | Coord (Fix _) -> Some "f"
    | Coord (Launch _) -> Some "l"
    | Coord (Lin ([ (1, _) ], 0)) -> Some "i"
    | Coord (Lin ([ _ ], _)) -> None
    | Coord (Lin (_, _)) -> Some "m"
  in
  let cls a =
    Option.map
      (Option.all (List.map (Array.to_list a.pos) ~f:position))
      ~f:(fun shape ->
        let cells = Hash_set.create (module Int) in
        iter_box a.ranges ~f:(fun env -> Hash_set.add cells (cell_of ~dims a.pos env));
        Printf.sprintf "%s %s %b %b" a.layout (String.concat shape)
          (Hash_set.length cells = numel dims)
          (Hash_set.mem cells 0))
  in
  List.filter_map xs ~f:(fun a -> Option.map (cls a) ~f:(fun c -> (c, a)))
  |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.map ~f:snd

(* Every third, in the classes' order: what keeps the crosses below to a test's run time. *)
let widen = Array.mem (Sys.get_argv ()) "--widen" ~equal:String.equal
let thinned xs = if widen then xs else List.filteri xs ~f:(fun i _ -> i % 3 = 0)

let family_runs ?(nested_src = false) ~dims ~comps ~reads ~steps ~standings () =
  List.iter reads ~f:(fun read_accesses ->
      List.iter steps ~f:(fun steps ->
          List.iter standings ~f:(fun standing ->
              check { dims; comps; read_accesses; steps; standing; nested_src })))

let stores write = { write; kind = Store; complex = false }
let one_read reads = List.map reads ~f:(fun r -> [| r |])
let one_component = [ [ Write 0; Read 0 ]; [ Zero; Write 0; Read 0 ] ]

let enumerate dims =
  let all_standings = [ Undecided; Declared; Inherited ] in
  let writes = dead_write dims :: accesses ~launch:false dims
  and reads = accesses ~launch:true dims in
  let core_writes = dead_write dims :: thinned (core ~dims writes)
  and core_reads = thinned (core ~dims reads) in
  let run ~comps ~reads ~steps ~standings = family_runs ~dims ~comps ~reads ~steps ~standings () in
  (* One stored component. *)
  if widen || List.length writes * List.length reads <= 20_000 then
    List.iter writes ~f:(fun w ->
        run
          ~comps:[| stores w |]
          ~reads:(one_read reads) ~steps:one_component ~standings:all_standings)
  else (
    List.iter writes ~f:(fun w ->
        run
          ~comps:[| stores w |]
          ~reads:(one_read core_reads) ~steps:one_component ~standings:[ Undecided; Declared ]);
    List.iter core_writes ~f:(fun w ->
        run
          ~comps:[| stores w |]
          ~reads:(one_read reads) ~steps:one_component ~standings:[ Undecided; Declared ]));
  (* Every kind of component, and a value reading another node. *)
  List.iter core_writes ~f:(fun write ->
      List.iter
        [
          { write; kind = Store; complex = true };
          { write; kind = Accumulate; complex = false };
          { write; kind = Accumulate; complex = true };
          { write; kind = Reduce; complex = false };
          { write; kind = Reduce; complex = true };
        ]
        ~f:(fun comp ->
          run ~comps:[| comp |] ~reads:(one_read core_reads) ~steps:one_component
            ~standings:all_standings));
  (* Nested virtual reduction: the same position/layout pools, not isolated boundary rows. Replay
     computes [src]; materialized twins compute both producers. Keeping this cross on the core
     writes bounds its cost while all reads retain offsets, reversals and launch indices. *)
  List.iter core_writes ~f:(fun write ->
      family_runs ~nested_src:true ~dims
        ~comps:[| { write; kind = Store; complex = true } |]
        ~reads:(one_read reads) ~steps:one_component ~standings:all_standings ());
  (* Reading twice, ahead of the write, around a later zero. *)
  List.iter core_writes ~f:(fun w ->
      run
        ~comps:[| stores w |]
        ~reads:(List.map core_reads ~f:(fun r -> [| r; r |]))
        ~steps:
          [
            [ Write 0; Read 0; Read 1 ];
            [ Read 0; Write 0; Read 1 ];
            [ Write 0; Zero; Read 0; Read 1 ];
            [ Write 0; Read 0; Zero; Read 1 ];
          ]
        ~standings:[ Undecided; Declared ]);
  (* Two components, in either order, the second a store or an accumulation over the first. *)
  let pair_writes = thinned core_writes and pair_reads = thinned core_reads in
  List.iter pair_writes ~f:(fun w1 ->
      List.iter pair_writes ~f:(fun w2 ->
          List.iter [ Store; Accumulate ] ~f:(fun kind ->
              let comps = [| stores w1; { (stores w2) with kind } |] in
              run ~comps ~reads:(one_read pair_reads)
                ~steps:
                  [
                    [ Write 0; Write 1; Read 0 ];
                    [ Zero; Write 0; Write 1; Read 0 ];
                    [ Write 0; Zero; Write 1; Read 0 ];
                  ]
                ~standings:all_standings;
              run ~comps
                ~reads:(List.map pair_reads ~f:(fun r -> [| r; r |]))
                ~steps:
                  [
                    [ Write 0; Read 0; Write 1; Read 1 ]; [ Zero; Write 0; Read 0; Write 1; Read 1 ];
                  ]
                ~standings:[ Undecided; Declared ])))

(* === Self-controls ===

   The replay must see what the oracle exists to see. A program missing its zero, or running its
   components in the other order, is a different program, whose whole pipeline must disagree with
   the original's replay. *)
let disagrees ~original corrupted =
  let b = build corrupted ~declare:ignore in
  not (same_outs (evaluate corrupted b (optimize_routines b)) (replay original))

let self_controls () =
  let sweep = Coord (Lin ([ (1, 0) ], 0)) in
  let over ranges =
    { layout = "axes"; pos = [| sweep |]; extents = [| 3 |]; ranges = [ (0, ranges) ] }
  in
  let prog ~comps ~steps =
    {
      dims = [| 3 |];
      comps;
      read_accesses = [| over (0, 2) |];
      steps;
      standing = Undecided;
      nested_src = false;
    }
  in
  let zeroed = prog ~comps:[| stores (over (0, 1)) |] ~steps:[ Zero; Write 0; Read 0 ] in
  p "self-control: a dropped zero is seen"
    (disagrees ~original:zeroed { zeroed with steps = [ Write 0; Read 0 ] });
  let overlap =
    prog ~comps:[| stores (over (0, 2)); stores (over (1, 2)) |] ~steps:[ Write 0; Write 1; Read 0 ]
  in
  p "self-control: swapped components are seen"
    (disagrees ~original:overlap { overlap with steps = [ Write 1; Write 0; Read 0 ] })

(* gh-ocannl-2077: the instrument's [~strict] probe. Codegen emits a [Local_scope]'s body ahead of
   its statement, so a scope in a discarded [Where] arm runs on a backend; the probe holds the
   scopes of a discarded arm to their nodes' bounds, and nothing else. Each case names what the lazy
   mode and the probe answer: a plain read of a discarded arm stays unevaluated (the control an
   eager-[Where] mutant fails), as do a scope guarded inside its own body, one in an [If] body never
   entered or a loop that never runs, and anything in a projection's unrendered operand, which
   neither mode evaluates. *)
let fault ?strict entries llc =
  match interpret ?strict (interp_memory entries) llc with
  | () -> None
  | exception Interp_fault msg -> Some msg

let is_hoisted_fault = function
  | Some msg -> String.is_substring msg ~substring:"a hoisted scope of a discarded Where arm"
  | None -> false

let strict_probe_controls () =
  let s = mk "s" and out = mk ~dims:[| 5 |] "out" and v = mk "v" in
  let entries = [ (s, [| 10.; 20.; 30. |]); (out, blank 5) ] in
  (* A scope over [v] whose body reads [s] at [idx]. *)
  let reading_scope ?guard idx =
    let id = LL.get_scope v in
    let read = LL.Set_local (id, get s [| idx |]) in
    let body =
      match guard with None -> read | Some g -> seq (LL.Set_local (id, c 0.)) (if_idx g read)
    in
    scope_of id body
  in
  let each f =
    let j = sym () in
    loop_n j 5 (f j)
  in
  let below3 j = lt (embed j) (ic 3) in
  let case name ~lazy_ ~strict llc =
    let lz = fault entries llc and st = fault ~strict:true entries llc in
    p
      (Printf.sprintf "strict probe: %s: lazy %s, strict %s" name
         (if lazy_ then "passes" else "faults")
         (match strict with `Passes -> "passes" | `Hoisted -> "faults on the hoisted scope"))
      (Bool.equal (Option.is_none lz) lazy_
      && match strict with `Passes -> Option.is_none st | `Hoisted -> is_hoisted_fault st)
  in
  case "scope reading past s in a discarded arm" ~lazy_:true ~strict:`Hoisted
    (each (fun j -> set out [| iter j |] (where_ (below3 j) (reading_scope (iter j)) (c 0.))));
  case "plain read past s in a discarded arm" ~lazy_:true ~strict:`Passes
    (each (fun j -> set out [| iter j |] (where_ (below3 j) (get s [| iter j |]) (c 0.))));
  case "scope guarded inside its own body" ~lazy_:true ~strict:`Passes
    (each (fun j ->
         set out [| iter j |] (where_ (below3 j) (reading_scope ~guard:(below3 j) (iter j)) (c 0.))));
  case "scope in an If body never entered past s" ~lazy_:true ~strict:`Passes
    (each (fun j -> if_idx (below3 j) (set out [| iter j |] (reading_scope (iter j)))));
  case "scope in a loop that never runs" ~lazy_:true ~strict:`Passes
    (let j = sym () in
     loop ~from_:5 ~upto:4 j
       (set out [| fixed 0 |] (where_ (below3 j) (c 0.) (reading_scope (iter j)))));
  case "scope in the discarded arm of an If condition" ~lazy_:true ~strict:`Hoisted
    (each (fun j ->
         if_idx
           (lt (c 0.) (where_ (below3 j) (reading_scope (iter j)) (c 1.)))
           (set out [| iter j |] (c 1.))));
  case "scope in a projection's unrendered operand" ~lazy_:true ~strict:`Passes
    (each (fun j ->
         set out
           [| iter j |]
           (where_ (below3 j)
              (binop Ops.Arg1 (get s [| fixed 0 |]) (reading_scope (aff [ (1, j) ] 3)))
              (c 0.))));
  case "plain read past s in a projection's unrendered operand" ~lazy_:true ~strict:`Passes
    (each (fun j -> set out [| iter j |] (binop Ops.Arg1 (c 1.) (get s [| aff [ (1, j) ] 3 |]))))

(* === Tier B: compiled and run === *)

let anchor_limit = 32

let run_anchor i prog =
  let want = replay prog in
  let seed_outs b =
    List.mapi (Array.to_list b.outs) ~f:(fun j out -> (out, blank (out_len prog.read_accesses.(j))))
  in
  let compiled ~name b o ~seed =
    let values =
      execute ~name ~bindings:b.bindings ~launch:b.launch o
        ~seed:(b.inputs @ seed @ seed_outs b)
        ~read:(Array.to_list b.outs)
    in
    Array.of_list values
  in
  let optimized ?materialized b =
    optimize ?materialized ~static_indices:b.static_indices ~name:"vio" (List.hd_exn b.routines)
  in
  let matches got = Array.for_all2_exn got want ~f:(fun a b -> close a b) in
  let twin =
    let b = build ~materialized_src:true prog ~declare:materialize in
    compiled ~name:(Printf.sprintf "vio_m%d" i) b (optimized b) ~seed:[ (b.x, entries prog) ]
  in
  let refused, virtual_arm =
    let b = build prog ~declare:(match prog.standing with Declared -> virtualize | _ -> ignore) in
    match optimized b with
    | exception Utils.User_error _ -> (true, [||])
    | o ->
        let seed = if read_before_write o b.x then [ (b.x, entries prog) ] else [] in
        (false, compiled ~name:(Printf.sprintf "vio_v%d" i) b o ~seed)
  in
  matches twin && (refused || matches virtual_arm)

let tier_b () =
  let candidates = sorted_values anchors and nested = sorted_values guarded_anchors in
  let ordinary = List.filter candidates ~f:(fun p -> not p.nested_src) in
  let sample limit xs =
    let step = Int.max 1 (List.length xs / limit) in
    List.take (List.filteri xs ~f:(fun i _ -> i % step = 0)) limit
  in
  p "compiled anchors include eight nested producers with guarded-arm scopes"
    (List.length (sample 8 nested) = 8);
  let picked = sample 8 nested @ sample (anchor_limit - 8) ordinary in
  Stdio.eprintf "Tier B: %d of %d buckets, on backend %s (not part of the golden)\n%!"
    (List.length picked) (List.length candidates)
    (Context.backend_name (Lazy.force base_ctx));
  p_alli ~min:24 "a compiled program computes the interpreter's values, in both arms" picked
    ~f:(fun i prog ->
      let holds = run_anchor i prog in
      if not holds then Stdio.eprintf "ANCHOR %s\n%!" (show_program prog);
      holds)

(* === The verdict === *)

let () =
  let t0 = Unix.gettimeofday () in
  List.iter shapes ~f:enumerate;
  if Option.is_none only then (
    let seconds = Unix.gettimeofday () -. t0 in
    Stdio.printf "%d programs\n" !programs_run;
    Stdio.eprintf "%d programs in %.1f s, %d findings (not part of the golden)\n%!" !programs_run
      seconds !finding_count;
    (* One row per family, so that a moved boundary edits rows without removing any:
       [tools/mutation-run.sh] reads a run printing fewer rows than the golden as stopped early. *)
    Hashtbl.to_alist census
    |> List.map ~f:(fun (key, n) ->
        match String.substr_index key ~pattern:" | " with
        | Some at ->
            (String.prefix key at, Printf.sprintf "%s=%d" (String.drop_prefix key (at + 3)) n)
        | None -> (key, Int.to_string n))
    |> List.sort ~compare:(fun (a, b) (c, d) ->
        match String.compare a c with 0 -> String.compare b d | order -> order)
    |> List.group ~break:(fun (a, _) (b, _) -> not (String.equal a b))
    |> List.iter ~f:(fun rows ->
        Stdio.printf "%s: %s\n"
          (fst (List.hd_exn rows))
          (String.concat ~sep:" " (List.map rows ~f:snd)));
    let total prefix substring =
      Hashtbl.fold census ~init:0 ~f:(fun ~key ~data acc ->
          if String.is_prefix key ~prefix:(prefix ^ " | ") && String.is_substring key ~substring
          then acc + data
          else acc)
    in
    List.iter
      [
        "comps=1 store undecided";
        "comps=1 store declared";
        "comps=1 store inherited";
        "comps=2 store undecided";
        "comps=2 store declared";
        "comps=2 store inherited";
        "comps=1 accumulate undecided";
        "comps=1 reduce undecided";
        "comps=1 store undecided src=reduce";
        "comps=1 store declared src=reduce";
        "comps=1 store inherited src=reduce";
      ] ~f:(fun fam ->
        pf "%s: some program inlined" fam (total fam "| inlined" >= 100);
        pf "%s: some program inlined under a guard" fam (total fam "| inlined-guarded" >= 100));
    List.iter [ Undecided; Declared; Inherited ] ~f:(fun standing ->
        let fam = "comps=1 store " ^ standing_name standing ^ " src=reduce" in
        pf "%s: some program has a scope in a guarded arm" fam
          (Option.value (Hashtbl.find guarded_scope_programs fam) ~default:0 >= 100));
    Hashtbl.iteri finding_classes ~f:(fun ~key ~data ->
        Stdio.eprintf "findings %s: %d (not part of the golden)\n%!" key data);
    p "every program kept to the semantics" (!programs_run >= 100_000 && !finding_count = 0);
    case "self-controls" self_controls;
    case "strict probe" strict_probe_controls;
    case "tier B" tier_b)
