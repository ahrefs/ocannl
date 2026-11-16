(* gh-ocannl-2181: a program-level random test of shape inference, judged by an oracle that shares
   no code with the solver.

   WHAT IT GENERATES. 2000 seeded DAGs of tensor expressions (depth 2 to 4), built through the
   constructors the [%op] syntax calls: - leaves: fixed-shape constants ([NTDSL.reshape ~b ~i ~o],
   distinct values), numeric literals ([TDSL.number], one size-1 output axis), draws
   ([TDSL.uniform], [TDSL.uniform1], rarely [TDSL.normal]) and parameters ([TDSL.param], whose
   default initialization is a draw); - operations: pointwise [+] and [*.], unary [relu] and [neg],
   [stretch], compose [*], and three [einsum1] specs: a reduction ["ab => a"], a transpose ["ab =>
   ba"], a fixed index ["...|3 => ...|0"]. A reused node is the same [Tensor.t] (a DAG, not a tree),
   and programs are weighted to branch: a binary operation with a resolve-at-use node under both
   operands, or one with two consumers. Each program picks a ground "world" of distinct extents
   (from 2, 3, 5, 7; at most 4 axes); its constants are suffixes of the world's rows, and most roots
   meet the whole world at a constant, so most programs are well-typed and most draws have a use
   site to fill. One program in ten gives one constant an extent (13) the world does not have.

   WHAT IT NEVER GENERATES: explicit row variables beyond the three specs, concatenation, strides,
   windows or convolutions, symbolic extents, a constant's size-1 axis (an explicit 1 is an ordinary
   axis, not a broadcast top, and does not broadcast), a fixed index right under another, captures,
   [Shape.set_dim] or [Shape.set_equal], binary einsum, more than 2 axes per row, gradients,
   multiple roots, precisions. Every random program is built whole and forced once: registration
   phases, late bindings, captures and product ties are the seed families at the end.

   THE ORACLE works on ground rows (batch, input, output) and calls nothing of [Row] or [Shape]. An
   operation's result is its operands' plain broadcast (pointwise; a missing or size-1 axis
   broadcasts), [lhs.output] and [rhs.input] with [rhs.output] broadcasting into [lhs.input]
   (compose), or the spec's own rule (einsum: the rows a spec elides are shared). Resolve-at-use
   nodes (draws, parameters, [stretch]) take the join of what their consumers have and demand, found
   by a fixpoint started from the empty shape, so a node never supports itself: a consumer demands
   of its operand its own shape (siblings included) and what its consumers demand of it; compose and
   the specs map demands per axis ([Exact], [Bcast], [Min] for a fixed index). A parameter takes no
   batch axes; a draw's axis nothing sizes closes to 1. The solution is re-checked strictly.
   Verdicts. [Solved] gives the shapes every node must take (compared modulo leading size-1 axes).
   [Ill_typed]: fixed shapes conflict, whatever the resolve-at-use nodes take (decided with them
   empty, checking only what nothing can change); it must be refused. [Underspecified]: a parameter
   sized only by a fixed index or an output row nothing sizes, refused at closure by design
   (docs/shape_inference.md); it must be refused. [Inconclusive] makes no claim and is counted on
   stderr by reason: demands that disagree at a resolve-at-use node, a resolve-at-use node nothing
   sizes, a contraction between two open rows, a re-check failure.

   CLAIMS (stdout): no program the oracle solves is refused but for a listed cause; every node's
   shape is the oracle's; every executed draw holds as many distinct values as its oracle shape has
   cells; every ill-typed or underspecified program is refused; nothing raises anything but
   [Row.Shape_error]; the judge flags corrupted observations; floors on the programs that agree
   outright, branching and executed, so that no entry grows to excuse them; and every entry of the
   known list still occurs (a stale entry fails until it is removed: the entry's fix removes it).
   Counts go to stderr.

   THE KNOWN LIST ([known] below). An entry excuses only the oracle's shape with axes collapsed to
   1, its draws filling the shape they closed to, at or under its named cause. [reach] additionally
   covers one pointwise consumer through unary results; the residual fixed-index/stretch role also
   follows a unary suffix after that hop (#557's relu), but adds no second pointwise hop. Refusals
   need one of that entry's observed messages.

   gh-ocannl-2229 fixes a fixed-index input's lower bound behind a chain of results: its
   non-parameter minimum above 1 closes before minimal guesses, then stage 6 propagates the concrete
   size to draws. On the initial base this retired 36 of the literal entry's 54 programs and
   reattributed 10 to existing causes. The 8 remaining cases were 5 stretches adopting a fixed-index
   output's size-1 axis and 3 literals after reductions. gh-ocannl-2247 retires the reduction half
   (#536, #682, #1036), which is removed from the predicate. The 5 stretch cases remain (#107, #557,
   #977, #1423, #1572). [(u + 1) *. c[5|3]] and [x[16] *. normal ()] agree in both shapes and
   executed fill; the judge checks every program node's shape, not just its draws.

   Other remaining causes: one observed normal-draw repeat; 3 fixed-index programs, including
   [(stretch u + stretch (stretch u)) ++ "...|3 => ...|0"]. The fixed-index
   entry used to cover all einsum specs over draws; since gh-ocannl-2247 passes a solved row's role
   to its axes, only fixed indices remain.

   Retired entries retain their agreeing fixed programs: gh-ocannl-2244 fixes a draw under a unary
   operation, under both operands of a pointwise operation, under a compose, and a no-axis constant
   in a branching program. The nested-branching entry also becomes stale with gh-ocannl-2229 and
   gh-ocannl-2247 combined, and is removed rather than reassigned to another cause. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row

(* {2 Programs} *)

type rows = { b : int list; i : int list; o : int list }

let equal_row = List.equal Int.equal
let equal_rows r1 r2 = equal_row r1.b r2.b && equal_row r1.i r2.i && equal_row r1.o r2.o
let empty = { b = []; i = []; o = [] }
let prod = List.fold ~init:1 ~f:( * )
let cells r = prod r.b * prod r.i * prod r.o

(* Shapes compare modulo leading size-1 axes: a row broadcasts as if they were absent. *)
let strip r = List.drop_while r ~f:(fun d -> d = 1)
let show_row r = String.concat ~sep:"," (List.map r ~f:Int.to_string)

let show_rows r =
  show_row r.b ^ "|" ^ (if List.is_empty r.i then "" else show_row r.i ^ "->") ^ show_row r.o

type draw = Uniform | Uniform1 | Normal
type spec = Reduce | Transpose | Fixed_index

let spec_string = function
  | Reduce -> "ab => a"
  | Transpose -> "ab => ba"
  | Fixed_index -> "...|3 => ...|0"

type node =
  | Const of rows
  | Lit
  | Draw of draw
  | Param of bool  (** [true]: initialized by [uniform1], as [{ w = uniform1 () }] *)
  | Stretch of int
  | Add of int * int
  | Mul of int * int
  | Relu of int
  | Neg of int
  | Compose of int * int
  | Einsum of spec * int

(* Topologically ordered; the last node is the root. *)
type program = node array

let resolves_at_use = function Draw _ | Param _ | Stretch _ -> true | _ -> false
let is_draw = function Draw _ | Param _ -> true | _ -> false
let is_param = function Param _ -> true | _ -> false

let show_node k = function
  | Const r -> Printf.sprintf "%%%d = const[%s]" k (show_rows r)
  | Lit -> Printf.sprintf "%%%d = 2." k
  | Draw Uniform -> Printf.sprintf "%%%d = uniform" k
  | Draw Uniform1 -> Printf.sprintf "%%%d = uniform1" k
  | Draw Normal -> Printf.sprintf "%%%d = normal" k
  | Param u1 -> Printf.sprintf "%%%d = {p%d%s}" k k (if u1 then " = uniform1 ()" else "")
  | Stretch a -> Printf.sprintf "%%%d = stretch %%%d" k a
  | Add (x, y) -> Printf.sprintf "%%%d = %%%d + %%%d" k x y
  | Mul (x, y) -> Printf.sprintf "%%%d = %%%d *. %%%d" k x y
  | Relu x -> Printf.sprintf "%%%d = relu %%%d" k x
  | Neg x -> Printf.sprintf "%%%d = neg %%%d" k x
  | Compose (x, y) -> Printf.sprintf "%%%d = %%%d * %%%d" k x y
  | Einsum (s, x) -> Printf.sprintf "%%%d = %%%d ++ %S" k x (spec_string s)

let show_program prog = String.concat ~sep:"; " (Array.to_list (Array.mapi prog ~f:show_node))

(* {2 The oracle} *)

exception Mismatch of string

(* Right-aligned broadcasting of two ground rows. *)
let bcast ~strict r1 r2 =
  let n1 = List.length r1 and n2 = List.length r2 in
  let n = Int.max n1 n2 in
  List.init n ~f:(fun k ->
      let at r nr = if k >= n - nr then Some (List.nth_exn r (k - (n - nr))) else None in
      match (at r1 n1, at r2 n2) with
      | Some d, None | None, Some d -> d
      | Some d1, Some d2 when d1 = d2 -> d1
      | Some (0 | 1), Some d | Some d, Some (0 | 1) -> d
      | Some d1, Some d2 ->
          if strict then raise (Mismatch (Printf.sprintf "broadcast %d against %d" d1 d2)) else d2
      | None, None -> assert false)

let bcast3 ~strict s1 s2 =
  { b = bcast ~strict s1.b s2.b; i = bcast ~strict s1.i s2.i; o = bcast ~strict s1.o s2.o }

let pad2 r =
  match r with [ a; b ] -> (a, b) | [ b ] -> (1, b) | [] -> (1, 1) | a :: b :: _ -> (a, b)

(* Forward shapes from the resolve-at-use nodes' assignment. Lenient forward never raises: it is
   what the fixpoint iterates on; strict forward is the re-check. *)
let forward ~strict (prog : program) (assign : rows option array) =
  let s = Array.create ~len:(Array.length prog) empty in
  Array.iteri prog ~f:(fun k node ->
      let a () = Option.value assign.(k) ~default:empty in
      s.(k) <-
        (match node with
        | Const r -> r
        | Lit -> { empty with o = [ 1 ] }
        | Draw _ | Param _ -> a ()
        | Stretch x ->
            let r = bcast3 ~strict (a ()) s.(x) in
            if strict && not (equal_rows r (a ())) then
              raise (Mismatch "stretch widens its use site");
            r
        | Add (x, y) | Mul (x, y) -> bcast3 ~strict s.(x) s.(y)
        | Relu x | Neg x -> s.(x)
        | Compose (x, y) ->
            ignore (bcast ~strict s.(x).i s.(y).o : int list);
            if strict && List.length s.(y).o > List.length s.(x).i then
              raise (Mismatch "compose: the rhs output has more axes than the lhs input");
            { b = bcast ~strict s.(x).b s.(y).b; i = s.(y).i; o = s.(x).o }
        | Einsum (spec, x) -> (
            let r = s.(x) in
            match spec with
            | Reduce | Transpose ->
                if strict && List.length r.o <> 2 then raise (Mismatch "spec wants 2 output axes");
                {
                  r with
                  o =
                    (match (spec, r.o) with
                    | Reduce, a :: _ :: _ -> [ a ]
                    | Transpose, [ a; b ] -> [ b; a ]
                    | _, o -> if strict then o else []);
                }
            | Fixed_index ->
                if strict && not (match r.o with [ d ] -> d >= 4 | _ -> false) then
                  raise (Mismatch "fixed index 3 wants one output axis of at least 4");
                { r with o = [ 1 ] })));
  s

type c = Exact of int | Bcast of int | Min of int | Free

(* A row demand, right-aligned; [exact_len] when the demander fixes the row's length. *)
type demand = { exact_len : bool; cs : c list }
type demand3 = { db : demand; di : demand; do_ : demand }

let of_row ~exact r =
  { exact_len = exact; cs = List.map r ~f:(fun d -> if exact then Exact d else Bcast d) }

let of_rows3 r =
  { db = of_row ~exact:false r.b; di = of_row ~exact:false r.i; do_ = of_row ~exact:false r.o }

(* The rightmost-but-[k] constraint of a demand. *)
let nth_right d k =
  let n = List.length d.cs in
  if k < n then List.nth_exn d.cs (n - 1 - k) else Free

(* The row a resolve-at-use node takes from its demands. *)
let min_only = "an axis only a fixed index bounds"

(* An axis nothing sizes, closing to 1 at the end; in the meantime it is no demand. *)
let unsized = 0
let hidden = "a parameter axis nothing sizes"

let combine ~param (ds : demand list) : (int list, string) Result.t =
  let exact_lens =
    List.dedup_and_sort ~compare:Int.compare
      (List.filter_map ds ~f:(fun d -> if d.exact_len then Some (List.length d.cs) else None))
  in
  match exact_lens with
  | _ :: _ :: _ -> Error "demands of different exact lengths"
  | _ -> (
      let len =
        match exact_lens with
        | [ l ] -> l
        | _ -> List.fold ds ~init:0 ~f:(fun m d -> Int.max m (List.length d.cs))
      in
      let axis k =
        let cs =
          List.filter_map ds ~f:(fun d ->
              if k < List.length d.cs then Some (nth_right d k) else None)
        in
        let cs = List.filter cs ~f:(function Exact 0 | Bcast 0 -> false | _ -> true) in
        let exacts =
          List.dedup_and_sort ~compare:Int.compare
            (List.filter_map cs ~f:(function Exact e -> Some e | _ -> None))
        in
        let bcasts =
          List.dedup_and_sort ~compare:Int.compare
            (List.filter_map cs ~f:(function Bcast e -> Some e | _ -> None))
        in
        (* What a consumer has is a lower bound to join: 1 is the bottom of that join. *)
        let bcasts = match List.filter bcasts ~f:(fun d -> d <> 1) with [] -> bcasts | l -> l in
        let min = List.fold cs ~init:0 ~f:(fun m -> function Min e -> Int.max m e | _ -> m) in
        match (exacts, bcasts) with
        | _ :: _ :: _, _ -> Error "exact demands disagree"
        | [ e ], _ ->
            if List.for_all bcasts ~f:(fun d -> d = e || e = 1) && e >= min then Ok e
            else Error "an exact demand conflicts with the others"
        | [], [ 1 ] when param && min = 0 ->
            (* A broadcast top sizes no parameter axis. *)
            Error hidden
        | [], [ d ] -> if d >= min then Ok d else Error "a broadcast demand below a fixed index"
        | [], [] ->
            if min > 0 then if param then Error min_only else Ok min
            else if param then Error hidden
            else (* A draw's unsized axis closes to 1, which demands nothing of others. *)
              Ok unsized
        | [], _ -> Error "broadcast demands disagree"
      in
      match Result.all (List.init len ~f:(fun j -> axis (len - 1 - j))) with
      | Ok r -> Ok r
      | Error e -> Error e)

type oracle =
  | Solved of rows array
  | Ill_typed of string  (** must be refused *)
  | Underspecified of int option * string
      (** refused at closure by design (docs/shape_inference.md); the parameter, when known *)
  | Inconclusive of string

(* The demands every node receives, given the current forward shapes. A consumer demands of its
   operand what the consumer has -- its own shape, siblings included -- and what its own consumers
   demand of it. [~use_min:false] drops the fixed index's lower bound. *)
let demands ~use_min (prog : program) (s : rows array) =
  let n = Array.length prog in
  let dem = Array.create ~len:n [] in
  let push x d = dem.(x) <- d :: dem.(x) in
  for k = n - 1 downto 0 do
    let ds = of_rows3 s.(k) :: dem.(k) in
    match prog.(k) with
    | Const _ | Lit | Draw _ | Param _ -> ()
    | Stretch x | Relu x | Neg x -> List.iter ds ~f:(push x)
    | Add (x, y) | Mul (x, y) ->
        List.iter ds ~f:(fun d ->
            push x d;
            push y d)
    | Compose (x, y) ->
        List.iter ds ~f:(fun d ->
            push x { d with di = of_row ~exact:false s.(y).o };
            push y { d with do_ = of_row ~exact:false s.(x).i })
    | Einsum (spec, x) ->
        List.iter ds ~f:(fun d ->
            let o =
              match spec with
              | Reduce -> [ nth_right d.do_ 0; Free ]
              | Transpose -> [ nth_right d.do_ 0; nth_right d.do_ 1 ]
              | Fixed_index -> [ (if use_min then Min 4 else Free) ]
            in
            push x { d with do_ = { exact_len = true; cs = o } })
  done;
  dem

(* Ill-typedness, decided with the resolve-at-use nodes empty and checking only what nothing can
   change: a fixed axis survives every broadcast, and only a constant's or a literal's own row is
   rigid -- an operation result can widen when a spec equates it with a longer row. *)
let ill_typed (prog : program) =
  let n = Array.length prog in
  let s = Array.create ~len:n empty in
  let rigid k = match prog.(k) with Const _ | Lit -> true | _ -> false in
  match
    Array.iteri prog ~f:(fun k node ->
        s.(k) <-
          (match node with
          | Const r -> r
          | Lit -> { empty with o = [ 1 ] }
          | Draw _ | Param _ -> empty
          | Stretch x | Relu x | Neg x -> s.(x)
          | Add (x, y) | Mul (x, y) -> bcast3 ~strict:true s.(x) s.(y)
          | Compose (x, y) ->
              let xi = s.(x).i and yo = s.(y).o in
              ignore (bcast ~strict:true xi yo : int list);
              (* The rhs output broadcasts into the lhs input, never the reverse. *)
              if rigid x && List.length yo > List.length xi then
                raise (Mismatch "the rhs output has more axes than the lhs input");
              { b = bcast ~strict:true s.(x).b s.(y).b; i = s.(y).i; o = s.(x).o }
          | Einsum (spec, x) ->
              let o = s.(x).o in
              (if rigid x then
                 match spec with
                 | Reduce | Transpose ->
                     if List.length o <> 2 then raise (Mismatch "spec wants 2 axes")
                 | Fixed_index -> (
                     match o with
                     | [ d ] when d >= 4 -> ()
                     | _ -> raise (Mismatch "fixed index 3 wants one axis of at least 4")));
              let a, b = pad2 o in
              {
                (s.(x)) with
                o =
                  (match spec with
                  | Reduce -> if List.length o = 2 then [ a ] else []
                  | Transpose -> if List.length o = 2 then [ b; a ] else []
                  | Fixed_index -> []);
              }))
  with
  | () -> None
  | exception Mismatch m -> Some m

let args = function
  | Const _ | Lit | Draw _ | Param _ -> []
  | Stretch x | Relu x | Neg x | Einsum (_, x) -> [ x ]
  | Add (x, y) | Mul (x, y) | Compose (x, y) -> [ x; y ]

(* The nodes at or under a node [is_x] selects: the node itself, and everything under its
   operands. *)
let under (prog : program) ~is_x =
  let n = Array.length prog in
  let marked = Array.init n ~f:(fun k -> is_x k prog.(k)) in
  for k = n - 1 downto 0 do
    if marked.(k) then List.iter (args prog.(k)) ~f:(fun x -> marked.(x) <- true)
  done;
  marked

(* A contraction between two rows only resolve-at-use nodes reach, or a reduced axis no constant
   sizes, is left to the solver's policy: no claim. *)
let underdetermined (prog : program) =
  let fixed = Array.create ~len:(Array.length prog) (false, false, false) in
  Array.iteri prog ~f:(fun k node ->
      let or3 (a, b, c) (d, e, f) = (a || d, b || e, c || f) in
      fixed.(k) <-
        (match node with
        | Const _ | Lit -> (true, true, true)
        | Draw _ | Param _ -> (false, false, false)
        | Stretch x | Relu x | Neg x | Einsum (_, x) -> fixed.(x)
        | Add (x, y) | Mul (x, y) -> or3 fixed.(x) fixed.(y)
        | Compose (x, y) ->
            let xb, _, xo = fixed.(x) and yb, yi, _ = fixed.(y) in
            (xb || yb, yi, xo)));
  Array.existsi prog ~f:(fun _ node ->
      match node with
      | Compose (x, y) ->
          let _, xi, _ = fixed.(x) and _, _, yo = fixed.(y) in
          (not xi) && not yo
      | _ -> false)

(* The fixpoint: [Ok assignment] or the first node's failure. *)
let fixpoint ~use_min (prog : program) =
  let n = Array.length prog in
  let assign = Array.init n ~f:(fun k -> if resolves_at_use prog.(k) then Some empty else None) in
  let rec iterate round =
    if round > 16 then Error "the fixpoint does not settle"
    else
      let s = forward ~strict:false prog assign in
      let dem = demands ~use_min prog s in
      let changed = ref false in
      let failure = ref None in
      Array.iteri prog ~f:(fun k node ->
          if resolves_at_use node && Option.is_none !failure then
            let get f = List.map dem.(k) ~f in
            let param = is_param node in
            match
              ( (if param then Ok [] else combine ~param (get (fun d -> d.db))),
                combine ~param (get (fun d -> d.di)),
                combine ~param (get (fun d -> d.do_)) )
            with
            | Ok b, Ok i, Ok o ->
                let r = { b; i; o } in
                if not (Option.equal equal_rows (Some r) assign.(k)) then (
                  assign.(k) <- Some r;
                  changed := true)
            | Error e, _, _ | _, Error e, _ | _, _, Error e ->
                failure := Some (Printf.sprintf "%%%d: %s" k e));
      match !failure with
      | Some e -> Error e
      | None -> if !changed then iterate (round + 1) else Ok assign
  in
  iterate 0

(* Both fixed-index-only sizes and forgotten parameter dimensions are refused at closure,
   regardless of initializer (gh-ocannl-2236). *)
(* The node a fixpoint failure names ("%k: ..."). *)
let failing_node e = Stdlib.Scanf.sscanf_opt e "%%%d:" Fn.id
let is_policy e = String.is_suffix e ~suffix:min_only || String.is_suffix e ~suffix:hidden

let solve (prog : program) : oracle =
  match ill_typed prog with
  | Some m -> Ill_typed m
  | None -> (
      match fixpoint ~use_min:true prog with
      | Error e when is_policy e -> Underspecified (failing_node e, e)
      | Error e -> Inconclusive e
      | Ok assign -> (
          match forward ~strict:true prog assign with
          | exception Mismatch m -> Inconclusive ("the re-check fails: " ^ m)
          | s -> (
              let close r = List.map r ~f:(fun d -> if d = unsized then 1 else d) in
              let s = Array.map s ~f:(fun r -> { b = close r.b; i = close r.i; o = close r.o }) in
              if underdetermined prog then Inconclusive "a contraction between two open rows"
              else if
                Array.existsi prog ~f:(fun k node ->
                    resolves_at_use node
                    && List.is_empty (strip s.(k).b)
                    && List.is_empty (strip s.(k).i)
                    && List.is_empty (strip s.(k).o))
              then Inconclusive "a resolve-at-use node nothing sizes"
              else if
                Array.existsi prog ~f:(fun k node -> is_param node && List.is_empty (strip s.(k).o))
              then Underspecified (None, "a parameter whose output row nothing sizes")
              else if
                Array.exists prog ~f:(function
                  | Compose (x, y) ->
                      List.is_empty s.(x).i
                      && Array.existsi prog ~f:(fun k node ->
                          is_param node
                          && ((under prog ~is_x:(fun j _ -> j = x)).(k)
                             || (under prog ~is_x:(fun j _ -> j = y)).(k)))
                  | _ -> false)
              then Inconclusive "a parameter contracted over an empty row"
              else
                (* A parameter sized through a fixed index, however indirectly, is refused by
                   design: without the index's bound it would have an axis nothing sizes. *)
                match fixpoint ~use_min:false prog with
                | Error e when String.is_suffix e ~suffix:hidden ->
                    Underspecified (failing_node e, "sized only by a fixed index (" ^ e ^ ")")
                | _ -> Solved s)))

(* {2 The solver's side} *)

let values n =
  Ir.Ndarray.init_array ~debug:"c" Ir.Ops.single ~dims:[| n |] ~padding:None ~f:(fun idx ->
      Float.of_int (idx.(0) + 1))

let build (prog : program) : Tensor.t array =
  let ts : Tensor.t option array = Array.create ~len:(Array.length prog) None in
  let t x = Option.value_exn ts.(x) in
  Array.iteri prog ~f:(fun k node ->
      let open TDSL.O in
      ts.(k) <-
        Some
          (match node with
          | Const r ->
              NTDSL.reshape ~l:(Printf.sprintf "c%d" k) ~b:r.b ~i:r.i ~o:r.o (values (cells r)) ()
          | Lit -> TDSL.number 2.
          | Draw Uniform -> TDSL.uniform () ()
          | Draw Uniform1 -> TDSL.uniform1 () ()
          | Draw Normal -> TDSL.normal () ()
          | Param false -> TDSL.param (Printf.sprintf "p%d" k) ()
          | Param true -> TDSL.param ~param_init:(TDSL.uniform1 ()) (Printf.sprintf "p%d" k) ()
          | Stretch x -> stretch (t x)
          | Add (x, y) -> t x + t y
          | Mul (x, y) -> t x *. t y
          | Relu x -> relu (t x)
          | Neg x -> neg (t x)
          | Compose (x, y) -> t x * t y
          | Einsum (spec, x) -> TDSL.einsum1 (spec_string spec) (t x) ()));
  Array.map ts ~f:(fun t -> Option.value_exn t)

(* A tensor's rows, read per kind: [to_dims] is batch, output, input. *)
let observe_rows (t : Tensor.t) =
  let dims = Array.to_list (Shape.to_dims t.Tensor.shape) in
  let bb, bi, bo = Shape.to_bases_bio t.Tensor.shape in
  let lb = Array.length bb and li = Array.length bi and lo = Array.length bo in
  let b = List.take dims lb and rest = List.drop dims lb in
  let o = List.take rest lo and i = List.drop rest lo in
  assert (List.length i = li);
  { b; i; o }

type outcome = Accepted of rows array | Refused of string | Crashed of string

let run_shapes (prog : program) : outcome =
  Tensor.unsafe_reinitialize ();
  match build prog with
  | exception Row.Shape_error (msg, _) -> Refused msg
  | exception e -> Crashed (Exn.to_string e)
  | ts -> (
      match
        ignore (Shape.to_dims ts.(Array.length ts - 1).Tensor.shape : int array);
        Array.map ts ~f:observe_rows
      with
      | rows -> Accepted rows
      | exception Row.Shape_error (msg, _) -> Refused msg
      | exception e -> Crashed (Exn.to_string e))

(* {2 Judging} *)

type judgement =
  | Agrees
  | Known of string  (** a listed known repeat or refusal *)
  | Disagrees of string
  | No_claim of string

(* The resolve-at-use nodes at or under each node. *)
let open_under (prog : program) =
  let sets = Array.create ~len:(Array.length prog) (Set.empty (module Int)) in
  Array.iteri prog ~f:(fun k node ->
      sets.(k) <-
        Set.union_list
          (module Int)
          ((if resolves_at_use node then Set.singleton (module Int) k else Set.empty (module Int))
          :: List.map (args node) ~f:(fun x -> sets.(x))));
  sets

let known_lit = "gh-ocannl-2180: a fixed-index output axis through stretch"

(* The output row's last axes taken over by one size-1 axis: everything else as the oracle says. *)
let output_collapse ~want ~got =
  let w = strip want.o in
  equal_row (strip want.b) (strip got.b)
  && equal_row (strip want.i) (strip got.i)
  && List.exists
       (List.init (List.length w) ~f:(fun j -> j + 1))
       ~f:(fun j -> equal_row (strip got.o) (strip (List.take w (List.length w - j) @ [ 1 ])))

(* Stretch adopts a fixed-index output's size-1 axis. The reduction half of the former literal
   exception was retired by gh-ocannl-2247. *)
let residual_axis_role (prog : program) =
  let fixed = Array.create ~len:(Array.length prog) false in
  Array.iteri prog ~f:(fun k node ->
      fixed.(k) <-
        (match node with Einsum (Fixed_index, _) -> true | _ -> false)
        || List.exists (args node) ~f:(fun x -> fixed.(x)));
  fun _ -> function Stretch x -> fixed.(x) | _ -> false

(* The known lists. Each entry names the operations its cause stands at ([is_x]); it excuses a node
   at or under one of them -- with [reach], also at or under the pointwise operation consuming one
   (through unary results), where the cause was seen to reach the siblings it is broadcast with --
   and for a shape, only one that is the oracle's with some axes collapsed to 1. A refusal of a
   program holding one is excused only when its message holds one of [refusals], the messages the
   entry was seen to produce. The residual literal/fixed-index role also follows a unary suffix
   after that single pointwise consumer; it never adds another pointwise hop. *)
type known = {
  entry : string;
  is_x : program -> int -> node -> bool;
  reach : bool;
  refusals : string list;
}

let region ?(unary_after_consumer = false) (prog : program) ~is_x ~reach =
  let base = Array.mapi prog ~f:(is_x prog) in
  if reach then (
    Array.iteri prog ~f:(fun k node ->
        match node with Stretch x | Relu x | Neg x -> if base.(x) then base.(k) <- true | _ -> ());
    let ext =
      Array.mapi prog ~f:(fun k node ->
          base.(k) || match node with Add (x, y) | Mul (x, y) -> base.(x) || base.(y) | _ -> false)
    in
    (* A unary suffix preserves the residual axis collapse after the one pointwise hop. Restricted
       to the literal/fixed-index role entry: it must not reinstate the retired generic unary
       exception (gh-ocannl-2244, random #557). *)
    if unary_after_consumer then
      Array.iteri prog ~f:(fun k node ->
          match node with Stretch x | Relu x | Neg x -> if ext.(x) then ext.(k) <- true | _ -> ());
    under prog ~is_x:(fun k _ -> ext.(k)))
  else under prog ~is_x:(fun k _ -> base.(k))

let opens prog = open_under prog
let is_open prog x = not (Set.is_empty (opens prog).(x))
let axes_count = "Mismatching number of axes"
let too_many = "Too many axes in an operand"

let known : known list =
  [
    { entry = known_lit; is_x = residual_axis_role; reach = true; refusals = [] };
    {
      entry = "box_muller: a normal draw";
      is_x = (fun _ _ node -> Poly.equal node (Draw Normal));
      reach = true;
      refusals = [ too_many; axes_count ];
    };
    {
      entry = "proposed: a fixed index over a draw";
      (* Was every einsum spec over a draw; since gh-ocannl-2247 a spec's axes take the draw's role,
         and what is left is the fixed index: [(stretch u + stretch (stretch u)) ++ "...|3 =>
         ...|0"]. *)
      is_x = (fun prog _ -> function Einsum (Fixed_index, x) -> is_open prog x | _ -> false);
      reach = false;
      refusals = [];
    };
  ]

let known_entries = List.map known ~f:(fun k -> k.entry)

let same r1 r2 =
  equal_row (strip r1.b) (strip r2.b)
  && equal_row (strip r1.i) (strip r2.i)
  && equal_row (strip r1.o) (strip r2.o)

(* [got] is [want] with some axes collapsed to 1 (right-aligned, leading ones dropped). *)
let collapse_of ~want ~got =
  let row w g =
    let w = strip w and g = strip g in
    List.length g <= List.length w
    && List.for_all2_exn g
         (List.drop w (List.length w - List.length g))
         ~f:(fun g w -> g = w || g = 1)
  in
  (not (same want got)) && row want.b got.b && row want.i got.i && row want.o got.o

(* [fills.(k)] is the number of distinct values draw [k] held when executed, if it was. *)
let judge (prog : program) (oracle : oracle) (outcome : outcome) ~(fills : int option array) =
  match (oracle, outcome) with
  | _, Crashed e -> Disagrees ("raised " ^ e)
  | Inconclusive why, _ -> No_claim why
  | Ill_typed _, Refused _ -> Agrees
  | Ill_typed why, Accepted _ -> Disagrees ("accepted an ill-typed program: " ^ why)
  | Underspecified _, Refused _ -> Agrees
  | Underspecified (_, why), Accepted _ -> Disagrees ("accepted an underspecified program: " ^ why)
  | Solved _, Refused msg -> (
      match
        List.find known ~f:(fun e ->
            List.exists e.refusals ~f:(fun sub -> String.is_substring msg ~substring:sub)
            && Array.existsi prog ~f:(e.is_x prog))
      with
      | Some e -> Known e.entry
      | None -> Disagrees ("refused a solvable program: " ^ msg))
  | Solved want, Accepted got -> (
      let nodes = List.init (Array.length prog) ~f:Fn.id in
      let mismatched = List.filter nodes ~f:(fun k -> not (same want.(k) got.(k))) in
      (* A known repeat explains a mismatch only in its predicted form. *)
      let holds =
        List.map known ~f:(fun e ->
            ( e.entry,
              region ~unary_after_consumer:(String.equal e.entry known_lit) prog ~is_x:e.is_x
                ~reach:e.reach ))
      in
      let explained k =
        if collapse_of ~want:want.(k) ~got:got.(k) then
          List.find_map holds ~f:(fun (entry, at) ->
              if String.equal entry known_lit then
                (* These residual roles collapse only trailing output axes. *)
                Option.some_if at.(k) entry
                |> Option.filter ~f:(fun _ -> output_collapse ~want:want.(k) ~got:got.(k))
              else if at.(k) then Some entry
              else None)
        else None
      in
      let fill k distinct =
        if distinct = cells want.(k) then None
        else
          match explained k with
          | Some e when distinct = cells got.(k) -> Some e
          | _ ->
              if Poly.equal prog.(k) (Draw Normal) && distinct < cells want.(k) then
                Some (List.nth_exn known_entries 1)
              else
                Some
                  (Printf.sprintf "draw %%%d holds %d distinct values for %d cells" k distinct
                     (cells want.(k)))
      in
      match List.find mismatched ~f:(fun k -> Option.is_none (explained k)) with
      | Some k ->
          Disagrees
            (Printf.sprintf "%%%d closed to %s, the oracle says %s" k
               (show_rows got.(k))
               (show_rows want.(k)))
      | None -> (
          let fill_issues = List.filter_map nodes ~f:(fun k -> Option.bind fills.(k) ~f:(fill k)) in
          match
            List.find fill_issues ~f:(fun e -> not (List.mem known_entries e ~equal:String.equal))
          with
          | Some e -> Disagrees e
          | None -> (
              match List.filter_map mismatched ~f:explained @ fill_issues with
              | e :: _ -> Known e
              | [] -> Agrees)))

(* {2 The generator} *)

let sizes = [ 2; 3; 5; 7 ]

type gen = {
  st : Random.State.t;
  mutable nodes : node list;  (** reversed *)
  mutable count : int;
  mutable made : (rows * int) list;  (** the nodes made for a world, for reuse *)
  mutable mutate : bool;
}

let chance g p = Float.( < ) (Random.State.float g.st 1.) p
let pick g l = List.nth_exn l (Random.State.int g.st (List.length l))

let weighted g l =
  let total = List.fold l ~init:0 ~f:(fun s (w, _) -> s + w) in
  let r = Random.State.int g.st total in
  let rec go acc = function
    | (w, x) :: rest -> if r < acc + w then x else go (acc + w) rest
    | [] -> assert false
  in
  go 0 l

let add g node =
  g.nodes <- node :: g.nodes;
  g.count <- g.count + 1;
  g.count - 1

let world_sizes w = w.b @ w.i @ w.o

let fresh_size g w =
  match List.filter sizes ~f:(fun d -> not (List.mem (world_sizes w) d ~equal:Int.equal)) with
  | [] -> 11
  | l -> pick g l

(* A sub-broadcast of the world's row: a suffix of it (never a size-1 axis). *)
let sub_row g r = List.drop r (Random.State.int g.st (List.length r + 1))

let gen_const g w =
  let r = { b = sub_row g w.b; i = sub_row g w.i; o = sub_row g w.o } in
  if g.mutate && chance g 0.5 && not (List.is_empty r.o) then (
    g.mutate <- false;
    { r with o = 13 :: List.tl_exn r.o })
  else r

let gen_leaf g w =
  add g
    (weighted g
       [
         (4, Const (gen_const g w));
         (1, Lit);
         (2, Draw Uniform);
         (1, Draw Uniform1);
         (1, if chance g 0.2 then Draw Normal else Draw Uniform);
         (1, Param false);
         (1, Param true);
       ])

let rec gen_node g w ~depth =
  match List.Assoc.find g.made w ~equal:equal_rows with
  | Some k when chance g 0.2 -> k
  | _ ->
      let k =
        if depth = 0 || chance g 0.25 then gen_leaf g w
        else
          let options =
            [ (5, `Binary); (1, `Unary); (1, `Stretch) ]
            @ (if cells w <= 105 then [ (1, `Compose) ] else [])
            @ (if (not (List.is_empty w.o)) && cells w <= 105 then [ (1, `Reduce) ] else [])
            @ (match w.o with [ _; _ ] -> [ (1, `Transpose) ] | _ -> [])
            @
            (* A fixed index right under another one would bound that one's result axis, which the
               oracle takes as 1: never generated. *)
            if cells w <= 105 && not (match w.o with [ d ] -> d >= 4 | _ -> false) then
              [ (1, `Fixed) ]
            else []
          in
          let d = depth - 1 in
          match weighted g options with
          | `Binary ->
              let x = gen_node g w ~depth:d in
              let y = gen_node g w ~depth:d in
              add g (if chance g 0.5 then Add (x, y) else Mul (x, y))
          | `Unary ->
              let x = gen_node g w ~depth:d in
              add g (if chance g 0.5 then Relu x else Neg x)
          | `Stretch -> add g (Stretch (gen_node g { w with b = sub_row g w.b } ~depth:d))
          | `Compose ->
              let m = [ fresh_size g w ] in
              let x = gen_node g { w with i = m } ~depth:d in
              let y =
                if chance g 0.5 then add g (Const (gen_const g { w with o = m }))
                else gen_node g { w with o = m } ~depth:d
              in
              add g (Compose (x, y))
          | `Reduce ->
              let a = List.last_exn w.o in
              add g (Einsum (Reduce, gen_node g { w with o = [ a; fresh_size g w ] } ~depth:d))
          | `Transpose ->
              add g (Einsum (Transpose, gen_node g { w with o = List.rev w.o } ~depth:d))
          | `Fixed ->
              let s = pick g [ 4; 5; 7 ] in
              add g (Einsum (Fixed_index, gen_node g { w with o = [ s ] } ~depth:d))
      in
      g.made <- (w, k) :: g.made;
      k

let gen_world g =
  let shuffled = List.permute ~random_state:g.st sizes in
  let total = 1 + Random.State.int g.st 4 in
  let lens = Array.create ~len:3 0 in
  for _ = 1 to total do
    let k = Random.State.int g.st 3 in
    if lens.(k) < 2 then lens.(k) <- lens.(k) + 1
  done;
  let b = List.take shuffled lens.(0) in
  let i = List.take (List.drop shuffled lens.(0)) lens.(1) in
  let o = List.take (List.drop shuffled (lens.(0) + lens.(1))) lens.(2) in
  { b; i; o }

let generate st : program =
  let g =
    { st; nodes = []; count = 0; made = []; mutate = Float.( < ) (Random.State.float st 1.) 0.1 }
  in
  let w = gen_world g in
  let x = gen_node g w ~depth:(2 + Random.State.int st 3) in
  (* Most roots meet the whole world at a constant, so that most draws have a use site to fill. *)
  (if chance g 0.7 then
     let c = add g (Const w) in
     ignore (add g (if chance g 0.5 then Mul (x, c) else Add (c, x)) : int));
  Array.of_list (List.rev g.nodes)

(* A program branches when a binary operation has a resolve-at-use node under both operands, or a
   resolve-at-use node has two consumers: the shapes every executed test had before gh-ocannl-2126
   were unary chains. *)
let branches (prog : program) =
  let n = Array.length prog in
  let under = Array.create ~len:n false in
  let consumers = Array.create ~len:n 0 in
  let branching = ref false in
  Array.iteri prog ~f:(fun k node ->
      let args = args node in
      List.iter args ~f:(fun x -> consumers.(x) <- consumers.(x) + 1);
      (match args with [ x; y ] when under.(x) && under.(y) -> branching := true | _ -> ());
      under.(k) <- resolves_at_use node || List.exists args ~f:(fun x -> under.(x)));
  !branching || Array.existsi consumers ~f:(fun k c -> c > 1 && under.(k))

(* {2 Execution} *)

(* Runs [progs] in one routine: each is built and closed, then their roots are reduced and summed
   into one tensor. Returns, per program, the distinct values of each draw. *)
let execute (progs : program list) : int option array list =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed:2181 ();
  let built =
    List.map progs ~f:(fun prog ->
        let ts = build prog in
        ignore (Shape.to_dims ts.(Array.length ts - 1).Tensor.shape : int array);
        Array.iteri prog ~f:(fun k node ->
            if is_draw node then Train.set_materialized ts.(k).Tensor.value);
        ts)
  in
  let sums =
    List.map built ~f:(fun ts -> NTDSL.einsum1 "...|...->... => |->0" ts.(Array.length ts - 1) ())
  in
  let total = List.reduce_exn sums ~f:(fun x y -> NTDSL.O.(x + y)) in
  let ctx = Train.forward_once (Context.cpu ()) total in
  let unique v = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list v)) in
  let result =
    List.map2_exn progs built ~f:(fun prog ts ->
        Array.mapi prog ~f:(fun k node ->
            if is_draw node then Some (unique (Context.get_values ctx ts.(k).Tensor.value))
            else None))
  in
  Context.release ctx;
  result

(* {2 The fixed programs} *)

let c53 = Const { b = [ 5 ]; i = []; o = [ 3 ] }

let fixed_programs : (string * program) list =
  [
    ("(u + v) *. c[5|3], two draws", [| Draw Uniform; Draw Uniform; Add (0, 1); c53; Mul (2, 3) |]);
    ( "[1;2] + (({w} + {v}) * [3;4;5])",
      [|
        Const { empty with o = [ 2 ] };
        Param true;
        Param true;
        Add (1, 2);
        Const { empty with o = [ 3 ] };
        Compose (3, 4);
        Add (0, 5);
      |] );
    ( "((u + 1) + (u *. 2)) *. c[5|3], one draw with two consumers",
      [| Draw Uniform; Lit; Add (0, 1); Lit; Mul (0, 3); Add (2, 4); c53; Mul (5, 6) |] );
    ("(u + 1) *. c[5|3]", [| Draw Uniform; Lit; Add (0, 1); c53; Mul (2, 3) |]);
    ( "(u + z) *. c[5|3], z a constant with no axes",
      [| Draw Uniform; Const empty; Add (0, 1); c53; Mul (2, 3) |] );
    ( "(u + z) + (u *. z), z with no axes, *. c[5|3]",
      [| Draw Uniform; Const empty; Add (0, 1); Mul (0, 1); Add (2, 3); c53; Mul (4, 5) |] );
    ("x[16] *. normal ()", [| Const { empty with o = [ 16 ] }; Draw Normal; Mul (0, 1) |]);
    ("c[|2,5] + u", [| Const { empty with o = [ 2; 5 ] }; Draw Uniform; Add (0, 1) |]);
    ("u + c[|2,5]", [| Draw Uniform; Const { empty with o = [ 2; 5 ] }; Add (0, 1) |]);
    ("c[|2,5] + relu u", [| Const { empty with o = [ 2; 5 ] }; Draw Uniform; Relu 1; Add (0, 2) |]);
    ("relu u + c[|2,5]", [| Draw Uniform; Relu 0; Const { empty with o = [ 2; 5 ] }; Add (1, 2) |]);
    ("c[|2,5] *. relu u", [| Const { empty with o = [ 2; 5 ] }; Draw Uniform; Relu 1; Mul (0, 2) |]);
    ("relu u *. c[|2,5]", [| Draw Uniform; Relu 0; Const { empty with o = [ 2; 5 ] }; Mul (1, 2) |]);
    ("(u *. u) + c[|2]", [| Draw Uniform; Mul (0, 0); Const { empty with o = [ 2 ] }; Add (1, 2) |]);
    ( "c[|3->2] * (u ++ \"ab => a\")",
      [|
        Const { empty with i = [ 3 ]; o = [ 2 ] }; Draw Uniform; Einsum (Reduce, 1); Compose (0, 2);
      |] );
    (* gh-ocannl-2247: the draw takes the spec's axes, which resolve at their use sites with it. *)
    ( "(u ++ \"ab => ba\") *. c[|5,3]",
      [| Draw Uniform; Einsum (Transpose, 0); Const { empty with o = [ 5; 3 ] }; Mul (1, 2) |] );
    (* The draw's sibling under a unary result: fixed by gh-ocannl-2244, the draw's spec axes by
       gh-ocannl-2247. *)
    ( "(neg (u ++ \"ab => a\") *. v) *. c[|5,2]",
      [|
        Draw Uniform;
        Einsum (Reduce, 0);
        Neg 1;
        Draw Uniform;
        Mul (2, 3);
        Const { empty with o = [ 5; 2 ] };
        Mul (4, 5);
      |] );
    ( "(stretch u + stretch (stretch u)) ++ \"...|3 => ...|0\"",
      [| Draw Uniform; Stretch 0; Stretch 1; Add (1, 2); Einsum (Fixed_index, 3) |] );
    ( "(c[|3,2] *. v) *. (u * c[|5])",
      [|
        Const { empty with o = [ 3; 2 ] };
        Draw Uniform;
        Mul (0, 1);
        Draw Uniform;
        Const { empty with o = [ 5 ] };
        Compose (3, 4);
        Mul (2, 5);
      |] );
    ( "c + (u + (v *. w))",
      [|
        Const { empty with o = [ 2; 5 ] };
        Draw Uniform;
        Draw Uniform;
        Draw Uniform;
        Mul (2, 3);
        Add (1, 4);
        Add (0, 5);
      |] );
    ( "c + ((u + v) + w)",
      [|
        Const { empty with o = [ 2; 5 ] };
        Draw Uniform;
        Draw Uniform;
        Add (1, 2);
        Draw Uniform;
        Add (3, 4);
        Add (0, 5);
      |] );
    ( "(u + (v + w)) *. c[5|3]",
      [| Draw Uniform; Draw Uniform; Draw Uniform; Add (1, 2); Add (0, 3); c53; Mul (4, 5) |] );
    ( "((u + v) *. c[5|3]) + w",
      [| Draw Uniform; Draw Uniform; Add (0, 1); c53; Mul (2, 3); Draw Uniform; Add (4, 5) |] );
    ( "((u + v) + z) *. c[|3]",
      [|
        Draw Uniform;
        Draw Uniform;
        Add (0, 1);
        Const empty;
        Add (2, 3);
        Const { empty with o = [ 3 ] };
        Mul (4, 5);
      |] );
    ( "(u + v) *. (w + x) *. c[|3]",
      [|
        Draw Uniform;
        Draw Uniform;
        Add (0, 1);
        Draw Uniform;
        Draw Uniform;
        Add (3, 4);
        Mul (2, 5);
        Const { empty with o = [ 3 ] };
        Mul (6, 7);
      |] );
  ]

(* {2 The sweep} *)

type case = { name : string; prog : program; oracle : oracle; outcome : outcome }

let num_programs = 2000
let num_executed = 120

(* Floors on the programs that agree outright (set a little under the counts at landing, which the
   stderr prints): a known entry grown to excuse them would fail here. *)
let strict_floor_branching = 650
let strict_floor_executed = 90

let () =
  let start = Unix.gettimeofday () in
  let st = Random.State.make [| 2181 |] in
  let rec branching_program () =
    let prog = generate st in
    if branches prog || Float.( < ) (Random.State.float st 1.) 0.2 then prog
    else branching_program ()
  in
  let random =
    List.init num_programs ~f:(fun k -> (Printf.sprintf "random #%d" k, branching_program ()))
  in
  let cases =
    List.map (fixed_programs @ random) ~f:(fun (name, prog) ->
        { name; prog; oracle = solve prog; outcome = run_shapes prog })
  in
  let has_draw c = Array.exists c.prog ~f:is_draw in
  let executable c =
    match (c.oracle, c.outcome) with Solved _, Accepted _ -> has_draw c | _ -> false
  in
  let fixed, random = List.split_n cases (List.length fixed_programs) in
  let exec_random =
    let ok = List.filter random ~f:executable in
    let stride = Int.max 1 (List.length ok / num_executed) in
    List.filteri ok ~f:(fun k _ -> k % stride = 0)
  in
  let to_execute = List.filter fixed ~f:executable @ exec_random in
  let fills = Hashtbl.Poly.create () in
  (* A batch that raises is re-run one program at a time, to name the one that does. *)
  let exec_failed = Hashtbl.Poly.create () in
  List.iter (List.chunks_of to_execute ~length:30) ~f:(fun batch ->
      match execute (List.map batch ~f:(fun c -> c.prog)) with
      | results ->
          List.iter2_exn batch results ~f:(fun c f -> Hashtbl.set fills ~key:c.name ~data:f)
      | exception _ ->
          List.iter batch ~f:(fun c ->
              match execute [ c.prog ] with
              | [ f ] -> Hashtbl.set fills ~key:c.name ~data:f
              | _ -> assert false
              | exception e -> Hashtbl.set exec_failed ~key:c.name ~data:(Exn.to_string e)));
  let judged =
    List.map cases ~f:(fun c ->
        let f =
          Option.value (Hashtbl.find fills c.name) ~default:(Array.map c.prog ~f:(fun _ -> None))
        in
        let j =
          match Hashtbl.find exec_failed c.name with
          | Some e -> Disagrees ("execution raised " ^ e)
          | None -> judge c.prog c.oracle c.outcome ~fills:f
        in
        (c, j, Hashtbl.mem fills c.name || Hashtbl.mem exec_failed c.name))
  in
  let count f = List.count judged ~f in
  Stdio.eprintf
    "(not part of the golden) %d programs (%d branching): %d solved by the oracle, %d ill-typed, \
     %d inconclusive; the solver accepted %d, refused %d; %d executed; %d agree, %d known, %d \
     disagree, %d without a claim; %.1fs\n\
     %!"
    (List.length cases)
    (count (fun (c, _, _) -> branches c.prog))
    (count (fun (c, _, _) -> match c.oracle with Solved _ -> true | _ -> false))
    (count (fun (c, _, _) -> match c.oracle with Ill_typed _ -> true | _ -> false))
    (count (fun (c, _, _) -> match c.oracle with Inconclusive _ -> true | _ -> false))
    (count (fun (c, _, _) -> match c.outcome with Accepted _ -> true | _ -> false))
    (count (fun (c, _, _) -> match c.outcome with Refused _ -> true | _ -> false))
    (count (fun (_, _, e) -> e))
    (count (fun (_, j, _) -> Poly.equal j Agrees))
    (count (fun (_, j, _) -> match j with Known _ -> true | _ -> false))
    (count (fun (_, j, _) -> match j with Disagrees _ -> true | _ -> false))
    (count (fun (_, j, _) -> match j with No_claim _ -> true | _ -> false))
    (Unix.gettimeofday () -. start);
  List.iter judged ~f:(fun (c, j, _) ->
      match j with
      | Disagrees why ->
          Stdio.eprintf "(not part of the golden) DISAGREES %s: %s\n  %s\n%!" c.name why
            (show_program c.prog)
      | _ -> ());
  (* Why the oracle made no claim, by reason (the node number dropped). *)
  List.iter
    (List.sort_and_group ~compare:String.compare
       (List.filter_map judged ~f:(fun (_, j, _) ->
            match j with
            | No_claim e -> Some (String.strip (List.last_exn (String.split e ~on:':')))
            | _ -> None)))
    ~f:(fun group ->
      Stdio.eprintf "(not part of the golden) no claim, %s: %d\n%!" (List.hd_exn group)
        (List.length group));
  List.iter known_entries ~f:(fun e ->
      Stdio.eprintf "(not part of the golden) known %s: %d\n%!" e
        (count (fun (_, j, _) -> Poly.equal j (Known e))));
  (* These thirteen seeded programs were refused by the broad leaf-waits-for-any-bound prototype
     (gh-ocannl-2141). Keep their acceptance independent of the known list. *)
  let fixed_index_regressions =
    [
      ( 45,
        "%0 = uniform; %1 = uniform1; %2 = %0 *. %1; %3 = neg %2; %4 = %3 ++ \"...|3 => ...|0\"; \
         %5 = const[2,5|3->]; %6 = %5 + %4" );
      ( 411,
        "%0 = uniform1; %1 = 2.; %2 = %0 * %1; %3 = stretch %2; %4 = %3 ++ \"...|3 => ...|0\"; %5 \
         = const[|3]; %6 = %4 *. %5" );
      ( 456,
        "%0 = uniform1; %1 = %0 ++ \"...|3 => ...|0\"; %2 = const[|2->]; %3 = uniform; %4 = %2 + \
         %3; %5 = uniform; %6 = stretch %5; %7 = %4 + %6; %8 = const[7|2]; %9 = %7 * %8; %10 = %1 \
         *. %9; %11 = const[7|5,3->]; %12 = %11 + %10" );
      (747, "%0 = uniform1; %1 = neg %0; %2 = const[|]; %3 = %1 * %2; %4 = %3 ++ \"...|3 => ...|0\"");
      ( 762,
        "%0 = uniform1; %1 = %0 ++ \"...|3 => ...|0\"; %2 = uniform1; %3 = stretch %2; %4 = %1 *. \
         %3; %5 = const[3,2|5->7]; %6 = %4 * %5; %7 = const[3,2|5->]; %8 = %6 *. %7" );
      ( 823,
        "%0 = const[|]; %1 = uniform1; %2 = %1 *. %1; %3 = %0 + %2; %4 = %3 ++ \"...|3 => ...|0\""
      );
      ( 845,
        "%0 = uniform; %1 = %0 ++ \"...|3 => ...|0\"; %2 = uniform; %3 = stretch %2; %4 = %1 + %3; \
         %5 = const[|5->]; %6 = relu %5; %7 = relu %6; %8 = %4 *. %7; %9 = relu %8; %10 = relu %9; \
         %11 = %8 + %10; %12 = const[|5->]; %13 = %11 *. %12" );
      ( 1237,
        "%0 = uniform; %1 = stretch %0; %2 = stretch %1; %3 = %1 + %2; %4 = %3 ++ \"...|3 => \
         ...|0\"" );
      ( 1402,
        "%0 = uniform; %1 = 2.; %2 = %0 * %1; %3 = uniform1; %4 = neg %3; %5 = %2 *. %4; %6 = \
         stretch %5; %7 = uniform; %8 = stretch %7; %9 = %8 ++ \"...|3 => ...|0\"; %10 = %9 + %9; \
         %11 = %6 *. %10; %12 = const[2,5|3->]; %13 = %12 + %11" );
      ( 1615,
        "%0 = uniform; %1 = %0 ++ \"...|3 => ...|0\"; %2 = uniform; %3 = stretch %2; %4 = %1 *. \
         %3; %5 = 2.; %6 = const[5|]; %7 = %5 * %6; %8 = %4 *. %7; %9 = const[5|]; %10 = %8 *. %9"
      );
      ( 1703,
        "%0 = const[|2->4]; %1 = %0 ++ \"...|3 => ...|0\"; %2 = uniform1; %3 = const[|]; %4 = %2 + \
         %3; %5 = %1 + %4; %6 = stretch %5; %7 = const[|2->]; %8 = %7 + %6" );
      ( 1783,
        "%0 = uniform1; %1 = const[|7->]; %2 = %0 *. %1; %3 = %2 + %2; %4 = uniform1; %5 = %4 *. \
         %4; %6 = %3 *. %5; %7 = %6 ++ \"...|3 => ...|0\"" );
      ( 1795,
        "%0 = const[7|4]; %1 = %0 ++ \"...|3 => ...|0\"; %2 = uniform; %3 = uniform1; %4 = %2 *. \
         %3; %5 = %1 + %4; %6 = const[7|]; %7 = %6 + %5" );
    ]
    |> List.map ~f:(fun (k, text) -> (Printf.sprintf "random #%d" k, text))
  in
  let fixed_index_cases =
    List.filter cases ~f:(fun c ->
        List.Assoc.mem fixed_index_regressions c.name ~equal:String.equal)
  in
  p_all ~min:13 "the thirteen fixed-index regression names still identify the same programs"
    fixed_index_cases ~f:(fun c ->
      String.equal (show_program c.prog)
        (List.Assoc.find_exn fixed_index_regressions c.name ~equal:String.equal));
  p_all ~min:13 "the thirteen fixed-index closing regressions stay accepted" fixed_index_cases
    ~f:(fun c -> match c.outcome with Accepted _ -> true | _ -> false);
  Verdict.case "the fixed programs" (fun () ->
      List.iter
        (List.take judged (List.length fixed_programs))
        ~f:(fun (c, j, executed) ->
          Stdio.printf "%s: %s%s\n" c.name
            (match j with
            | Agrees -> "agrees"
            | Known e -> "known: " ^ e
            | Disagrees _ -> "DISAGREES"
            | No_claim e -> "no claim: " ^ e)
            (if executed then ", executed" else ""));
      p_all ~min:1 "every fixed program agrees with the oracle or is a listed known case"
        (List.take judged (List.length fixed_programs))
        ~f:(fun (_, j, _) -> match j with Agrees | Known _ -> true | _ -> false));
  Verdict.case "the random sweep" (fun () ->
      let random = List.drop judged (List.length fixed_programs) in
      p_all ~min:1000 "the oracle solves programs and the solver accepts them"
        (List.filter random ~f:(fun (c, _, _) ->
             match (c.oracle, c.outcome) with Solved _, Accepted _ -> true | _ -> false))
        ~f:(fun _ -> true);
      p_all ~min:20 "programs whose fixed shapes conflict occur, and every one is refused"
        (List.filter random ~f:(fun (c, _, _) ->
             match c.oracle with Ill_typed _ -> true | _ -> false))
        ~f:(fun (c, _, _) -> match c.outcome with Refused _ -> true | _ -> false);
      p_all ~min:1000 "branching programs occur, and none disagrees with the oracle"
        (List.filter random ~f:(fun (c, _, _) -> branches c.prog))
        ~f:(fun (_, j, _) -> match j with Disagrees _ -> false | _ -> true);
      p_none ~min:num_programs "no program the oracle solves is refused, but for a listed cause"
        random ~f:(fun (c, j, _) ->
          match (c.oracle, c.outcome, j) with
          | Solved _, Refused _, Known _ -> false
          | Solved _, Refused _, _ -> true
          | _ -> false);
      p_none ~min:num_programs "no program raises anything but a shape error" random
        ~f:(fun (c, _, _) -> match c.outcome with Crashed _ -> true | _ -> false);
      p_none ~min:num_programs "no program disagrees with the oracle" random ~f:(fun (_, j, _) ->
          match j with Disagrees _ -> true | _ -> false);
      p_all ~min:50 "executed programs occur, and every draw fills its oracle shape or is listed"
        (List.filter random ~f:(fun (_, _, e) -> e))
        ~f:(fun (_, j, _) -> match j with Agrees | Known _ -> true | _ -> false);
      (* The strict population: the floors above count excused cases too. *)
      let agree = List.filter random ~f:(fun (_, j, _) -> Poly.equal j Agrees) in
      let agree_branching = List.filter agree ~f:(fun (c, _, _) -> branches c.prog) in
      let agree_executed = List.filter agree ~f:(fun (_, _, e) -> e) in
      Stdio.eprintf
        "(not part of the golden) agree outright: %d, of them branching %d, executed %d\n%!"
        (List.length agree) (List.length agree_branching) (List.length agree_executed);
      p_all ~min:strict_floor_branching "branching programs agree outright, no entry excusing them"
        agree_branching ~f:(fun _ -> true);
      p_all ~min:strict_floor_executed
        "executed programs agree outright, every draw filling its oracle shape" agree_executed
        ~f:(fun _ -> true));
  Verdict.case "the known lists" (fun () ->
      p_all ~min:(List.length known) "every known entry still occurs" known_entries ~f:(fun e ->
          List.exists judged ~f:(fun (_, j, _) -> Poly.equal j (Known e))));
  Verdict.case "the judge flags corrupted observations" (fun () ->
      (* Take the agreeing executed fixed program with two draws and corrupt what was observed. *)
      let c, _, _ = List.hd_exn judged in
      let fills = Hashtbl.find_exn fills c.name in
      let got = match c.outcome with Accepted got -> got | _ -> assert false in
      let corrupt k r = Accepted (Array.mapi got ~f:(fun j g -> if j = k then r else g)) in
      let corruptions =
        [
          ("a collapsed draw", corrupt 0 { b = [ 1 ]; i = []; o = [ 1 ] }, fills);
          ("a wrong intermediate shape", corrupt 2 { b = [ 5 ]; i = []; o = [ 1 ] }, fills);
          ("a refusal", Refused "corrupted", fills);
          ( "a repeating draw",
            c.outcome,
            Array.mapi fills ~f:(fun k f -> if k = 0 then Some 1 else f) );
        ]
      in
      p_all ~min:4 "every corruption is judged a disagreement" corruptions ~f:(fun (_, o, fills) ->
          match judge c.prog c.oracle o ~fills with Disagrees _ -> true | _ -> false);
      let ill =
        [|
          Const { b = [ 5 ]; i = []; o = [ 3 ] }; Const { b = [ 4 ]; i = []; o = [ 3 ] }; Add (0, 1);
        |]
      in
      p "an accepted ill-typed program is judged a disagreement"
        (match
           judge ill (solve ill) (Accepted [| empty; empty; empty |]) ~fills:[| None; None; None |]
         with
        | Disagrees _ -> true
        | _ -> false))

(* {2 The seed families (wave-20261111-ocannl's comment on gh-ocannl-2181)}

   Programs the generator's grammar does not reach -- captures, [Shape.set_dim] and
   [Shape.set_equal] at chosen registration phases, binary einsum, concatenation and strides -- each
   with the expectation its own test pins. *)

let dims t = Array.to_list (Shape.to_dims t.Tensor.shape)
let open_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:(Constant 2.) ()
let ranged_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:Range_over_offsets ()

let seed name f =
  Verdict.case name (fun () ->
      Tensor.unsafe_reinitialize ();
      f ())

let refusal f =
  match f () with
  | () -> None
  | exception Row.Shape_error (msg, _) ->
      Tensor.rewind_to_last_closure ~reason:"gh-2181 seed refusal" ();
      Some msg

let () =
  (* gh-ocannl-2127: late bindings. A draw's use site sized by a capture set before, or after, the
     draw's consumer registers; and a size set after the row closed reaching a tied dimension
     (test/operations/product_tie_cancellation.ml). *)
  List.iter
    [ ("before", true); ("after", false) ]
    ~f:(fun (phase, early) ->
      seed
        ("late binding: the use site sized " ^ phase ^ " the draw's consumer")
        (fun () ->
          let d = Shape.get_variable_ref "d" in
          if early then Shape.set_dim d 5;
          let c = NTDSL.einsum1 ~capture_dims:[ d ] "d => d" (open_leaf "c") () in
          let u = TDSL.uniform () () in
          let y = TDSL.O.(u *. c) in
          if not early then Shape.set_dim d 5;
          p
            ("late binding, sized " ^ phase ^ ": the draw fills its use site")
            (List.equal Int.equal (dims y) [ 5 ] && List.equal Int.equal (dims u) [ 5 ])));
  seed "late binding: a size set after the row closed" (fun () ->
      let n = Shape.get_variable_ref "n" and c = Shape.get_variable_ref "c" in
      let y1 = NTDSL.einsum1 ~capture_dims:[ c; n ] "..c.., n => ..c.., n" (open_leaf "q") () in
      let y2 = NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." y1 () in
      let k = Shape.get_variable_ref "k" in
      let y3 = NTDSL.einsum1 ~capture_dims:[ k ] "k => k" y1 () in
      Shape.set_dim k 5;
      p_all ~min:3 "a size set after the row closed reaches the tied dimension" [ y1; y2; y3 ]
        ~f:(fun t -> List.equal Int.equal (dims t) [ 5 ]));
  (* gh-ocannl-2131: capture-reference mutations (test/operations/solved_capture_basis.ml): two open
     captures, one or both sized, made equal in either order, then a named capture. *)
  let capture n x = NTDSL.einsum1 ~capture_dims:[ n ] "n => n" x () in
  let tagged basis = NTDSL.ndarray [| 1.; 2.; 3. |] ~output_axes:[ (basis, 3) ] () in
  let rgb x = Array.mem (Shape.to_bases x.Tensor.shape) "rgb" ~equal:String.equal in
  List.iter
    [ (true, false, false); (false, true, true); (true, true, false) ]
    ~f:(fun (size_left, size_right, swap) ->
      let name =
        Printf.sprintf "capture mutation, sized %b/%b, %s" size_left size_right
          (if swap then "reversed" else "forward")
      in
      let pair () =
        let n = Shape.get_variable_ref "n" and m = Shape.get_variable_ref "n" in
        let _x = capture n (open_leaf "l") and y = capture m (open_leaf "r") in
        if size_left then Shape.set_dim n 3;
        if size_right then Shape.set_dim m 3;
        if swap then Shape.set_equal m n else Shape.set_equal n m;
        ignore (capture n (tagged "rgb") : Tensor.t);
        (m, y)
      in
      seed (name ^ ": propagation") (fun () ->
          let _, y = pair () in
          p
            (name ^ ": the basis propagates across the equality")
            (List.equal Int.equal (dims y) [ 3 ] && rgb y));
      seed (name ^ ": conflict") (fun () ->
          let m, _ = pair () in
          p
            (name ^ ": a different basis is refused")
            (Option.is_some
               (refusal (fun () -> ignore (dims (capture m (tagged "xyz")) : int list))))));
  (* gh-ocannl-2154: the reuse DAG (test/operations/contraction_pointwise_closing.ml), its
     consumer's widths varied; accepted since that fix. *)
  List.iter
    [ [ 3; 4; 5 ]; [ 3; 5; 4 ] ]
    ~f:(fun widths ->
      let name =
        "reuse DAG, consumer widths " ^ String.concat ~sep:"," (List.map widths ~f:Int.to_string)
      in
      seed name (fun () ->
          let st = TDSL.uniform () () in
          let x = TDSL.range_of_shape ~output_dims:widths () in
          let p2 = TDSL.O.(st *. !.2.) in
          let y = TDSL.einsum "...|h,w,..c..; ...|..c.. => ...|h" p2 st () in
          let q = TDSL.O.(p2 *. x) in
          let z = TDSL.O.(y + q) in
          p
            (name ^ ": accepted, the result takes the consumer's widths")
            (match dims z with
            | ds -> List.equal Int.equal ds widths
            | exception Row.Shape_error _ -> false)));
  (* gh-ocannl-2159: product ties over composite numerators
     (test/einsum/product_tie_concat_affine.ml): [n] is made of [a] by a concatenation or a stride,
     with a known row prefix, a downstream consumer, or [a] sized late; an unsized concatenation
     component is not guessed. *)
  let tie ~spec ~sizing =
    let n = Shape.get_variable_ref "n" and a = Shape.get_variable_ref "a" in
    let y1 = NTDSL.einsum1 ~capture_dims:[ n ] "n => n" (ranged_leaf "q") () in
    let y2 = NTDSL.einsum1 ~capture_dims:[ a ] spec y1 () in
    let z =
      match sizing with
      | `Prefix d ->
          let prefix = Shape.get_variable_ref "prefix" in
          let z =
            NTDSL.einsum1 ~capture_dims:[ prefix ] "prefix, ..tail.. => prefix, ..tail.."
              (ranged_leaf "z") ()
          in
          Shape.set_dim prefix d;
          z
      | _ -> ranged_leaf "z"
    in
    let y3 = NTDSL.einsum1 ~capture_dims:[ n ] "..n.. => ..n.." z () in
    let y3 = match sizing with `Consumed -> NTDSL.add y3 y3 () | _ -> y3 in
    (match sizing with `Late -> Shape.set_dim a 2 | _ -> ());
    [ y1; y2; y3 ]
  in
  List.iter
    [
      ("concatenation, a known prefix", "a^2 => a", `Prefix 4, [ [ 4 ]; [ 2 ]; [ 4 ] ]);
      ("stride, a known prefix", "2*a => a", `Prefix 4, [ [ 4 ]; [ 2 ]; [ 4 ] ]);
      ("stride, a downstream consumer", "2*a => a", `Consumed, [ [ 2 ]; [ 1 ]; [ 2 ] ]);
      ("concatenation, the component sized late", "a^2 => a", `Late, [ [ 4 ]; [ 2 ]; [ 4 ] ]);
    ]
    ~f:(fun (name, spec, sizing, want) ->
      seed ("product tie, " ^ name) (fun () ->
          let ts = tie ~spec ~sizing in
          pf_all2 "product tie, %s: the parts take their sizes" name
            (Array.of_list (List.map ts ~f:dims))
            (Array.of_list want) ~f:(List.equal Int.equal)));
  seed "product tie, an unsized concatenation component" (fun () ->
      p "product tie, an unsized concatenation component: refused, not guessed"
        (Option.exists
           (refusal (fun () ->
                List.iter (tie ~spec:"a^2 => a" ~sizing:`Unsized) ~f:(fun t ->
                    ignore (dims t : int list))))
           ~f:(fun msg -> String.is_substring msg ~substring:"closure does not guess one")))
