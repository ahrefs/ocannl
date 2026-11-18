(* gh-ocannl-2194, gh-ocannl-2334. Recognised forms (closed).

   Input: .ml under test/ and arrayjit/test/, excluding test/ppx/*_expected.ml snapshots, one file
   at a time. Census: applications of Ll_test.exact_partials through scoped module aliases,
   opens/includes, and aliases of unapplied helper identifiers. Comments/string contents are not
   programs.

   Support modules (gh-ocannl-2334): the closed list [support_sources] alone is read across files.
   Each listed test/support source is walked once, with the harness and the sources listed before it
   in scope, and its top-level value bindings -- what each denotes right after its own item -- are
   what [M.name] denotes in every scanned file, M the source's module name, through scoped aliases,
   opens (local ones included) and includes of M. Their function bodies are then interpreted with
   the call's arguments bound, in the support source's own scope. A listed source missing from the
   census, failing to parse, or holding a top-level include exports nothing, so its callers confer
   nothing. Sites inside a support source are counted as that source's own.

   Coverage is per syntactic site. p_all's products must come from matmul_products with two
   operand-derived readers, or reduction_cells whose callback returns a checked cell. A checked cell
   is an operand reader applied to coordinates, or a multiplication by the *. operator of checked
   cells and float constants, with at least one checked cell. operand requires b/o/i labels and an
   initializer; its initializer and coordinates are opaque. No raw array read confers coverage.

   Fully supplied nonrecursive local function wrappers, and the support modules' top-level ones, are
   interpreted at most three calls deep (one bound across modules), binding plain, constrained,
   tuple and list patterns and labelled/optional parameters. A wrapper returning operand, or an
   applied operand, therefore preserves checked-reader provenance (seeded). Let bindings, sequences,
   tuples, literal lists/arrays, match/if branches, lazy and Lazy.force carry summaries; every
   returning branch must agree (bare one-argument invalid_arg/failwith/raise and assert-false
   branches do not return). Wrappers may return readers, cells, references/projections or
   comparisons, never products, collections or helper callees. Unknown returns, recursion and deeper
   calls confer no coverage. Function definitions themselves are walked with unknown parameters,
   without following arbitrary callers for claim discovery.

   The same products' lexical builder identities (source and offset) must reach p_all2 label
   reference sums ~f:Float.equal, directly or through a local comparison wrapper (check_products).
   Sums are Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)). References come
   from Ll_test.execute's result list (List.hd_exn or list-pattern extraction), Context.get_values
   of a Context.run result (its node opaque), or Context.get_values of a Train.forward_once or
   Train.update_once result (bare or Ocannl-qualified, both positional arguments supplied) at a node
   that call's compile kept in a buffer: the [.value] or [.Tensor.value] of the very binding passed
   as the call's tensor argument, which run_once materializes; or a node expression -- an
   identifier, field projections and one-argument applications of identifiers -- that a statement
   [Train.set_materialized node], sequenced before the call in its scope, names alike (printed AST),
   each identifier resolving to the same binding there and at the read. run_once materializes only
   its tensor (Asgns.guess_materialized's outputs are to_routine's alone), so any other node a _once
   context is read at -- another tensor, a gradient not materialized first -- may be Virtual, which
   a read recomputes through a throwaway routine (gh-ocannl-777) rather than reading what the run
   wrote: it confers nothing. Local wrappers, tuples/fst/snd, lazy/force preserve that provenance.
   Compile, get_values without run, and host sums alone never establish an executed reference.

   Inline List.iter over a literal collection joins the elements' tuple fields before walking its
   callback. ALL dispatched products must be checked and compared; an unknown arm grants nothing.
   p_all/p_all2 are bare or Verdict.Claims-qualified. List.hd_exn/map/sum/iter and Array.of_list
   also accept Base prefixes; Context.run/get_values, Lazy.force, Float.equal, Fn.id, fst/snd and *.
   use those spellings (fst/snd also Stdlib). Name-based recognition rejects rebound heads; local
   function wrappers can still derive facts from their bodies. Aliases of standard modules and names
   imported by unknown opens are not resolved. Helper module/value aliases ARE scoped. Only the
   listed arguments are read; dimensions, claim labels and execute/run arguments are opaque except
   get_values' context, a _once context's read node, that call's tensor argument and the operands of
   *.

   Mutations after construction/read (beyond the set_materialized statement above, never undone) and
   aliases through mutable storage are not tracked. This is source provenance, not a proof of
   operation identity or reachability. Branches and quotation AST payloads are traversed; functor
   results, first-class module contents, dynamically obtained callees, external/generated code,
   arbitrary collections, other arithmetic and other comparisons confer no coverage. Fully applied
   exact_partials outside p_all remain debt, as do the helper's mathematical unit tests. Unknown
   forms never confer coverage. *)
open Base
open Ppxlib

type value =
  | Unknown
  | Bottom
  | Helper of string
  | Operand
  | Cell
  | Constant
  | Products of string list
  | References
  | Reference
  | Ran
  (* A Train.forward_once/update_once result: the tensor argument's name, when it is one, and the
     scope at the call, which binds that name and holds the set_materialized markers before it. *)
  | Ran_once of string option * env
  (* A set_materialized marker: the binding frame of each identifier of its node expression. *)
  | Materialized of value Map.M(String).t list
  | Sums of string list
  | Compared of string list
  | Tuple of value list
  | Rows of value list
  | Delayed of value
  | Closure of function_param list * expression * env

(* What a module name denotes: the harness, a support source's exports, or nothing modelled. *)
and denotes = Harness | Support of value Map.M(String).t | Opaque
and env = (value, denotes) Lexical_scope.env

type site = { line : int; covered : bool }

(* Module name to [Support] exports, built by {!add_support}. *)
type support = denotes Map.M(String).t

let no_support : support = Map.empty (module String)
let initial support : env = { frames = []; modules = Map.set support ~key:"Ll_test" ~data:Harness }

let helper (env : env) = function
  | Longident.Lident n -> (
      match Lexical_scope.lookup env n with Some (Helper h) -> Some h | _ -> None)
  | Ldot (Lident m, n) -> ( match Map.find env.modules m with Some Harness -> Some n | _ -> None)
  | _ -> None

let exported (env : env) = function
  | Longident.Ldot (Lident m, n) -> (
      match Map.find env.modules m with Some (Support exports) -> Map.find exports n | _ -> None)
  | _ -> None

let rec path_name = function
  | Longident.Lident n -> Some n
  | Ldot (p, n) -> Option.map (path_name p) ~f:(fun p -> p ^ "." ^ n)
  | Lapply _ -> None

let named env e names =
  match e.pexp_desc with
  | Pexp_ident { txt; _ } ->
      let unshadowed =
        match txt with
        | Lident n -> Option.is_none (Lexical_scope.lookup env n)
        | _ -> not (Map.mem env.modules (Lexical_scope.head txt))
      in
      unshadowed
      && Option.value_map (path_name txt) ~default:false ~f:(fun n ->
          List.mem names n ~equal:String.equal)
  | _ -> false

let positional args = List.filter_map args ~f:(function Nolabel, e -> Some e | _ -> None)

let label args n =
  List.find_map args ~f:(function Labelled k, e when String.equal n k -> Some e | _ -> None)

let labels args names = List.for_all names ~f:(fun n -> Option.is_some (label args n))
let offset e = e.pexp_loc.loc_start.pos_cnum

(* A products builder's lexical identity: its source and offset. A support closure's own builders
   are parsed under the support's name, so they never share an identity with a scanned file's. *)
let builder e =
  let start = e.pexp_loc.loc_start in
  start.pos_fname ^ ":" ^ Int.to_string start.pos_cnum

let union a b = List.dedup_and_sort (a @ b) ~compare:String.compare

let once_runs =
  [
    "Train.forward_once";
    "Train.update_once";
    "Ocannl.Train.forward_once";
    "Ocannl.Train.update_once";
  ]

(* The frame binding [n] in scope: two reads of [n] reach one binding iff they reach one frame. *)
let binding_frame (env : env) n = List.find env.frames ~f:(fun frame -> Map.mem frame n)

let same_binding ~at_call ~at_read n =
  match (binding_frame at_call n, binding_frame at_read n) with
  | Some a, Some b -> phys_equal a b
  | _ -> false

(* The identifiers of a node expression, in order: an identifier, field projections and one-argument
   applications of identifiers; [None] outside that grammar. *)
let rec node_names e =
  match e.pexp_desc with
  | Pexp_ident { txt = Lident n; _ } -> Some [ n ]
  | Pexp_field (inner, _) -> node_names inner
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident f; _ }; _ }, [ (Nolabel, arg) ]) ->
      Option.map (node_names arg) ~f:(fun names -> f :: names)
  | _ -> None

let node_frames env e =
  Option.bind (node_names e) ~f:(fun names -> Option.all (List.map names ~f:(binding_frame env)))

(* The space keeps a marker's key apart from every identifier. *)
let marker_key e = "set_materialized " ^ Stdlib.Format.asprintf "%a" Pprintast.expression e

(* The scope after a statement [Train.set_materialized node], with that node's marker in it. *)
let materialize env statement =
  match statement.pexp_desc with
  | Pexp_apply (f, [ (Nolabel, node) ])
    when named env f [ "Train.set_materialized"; "Ocannl.Train.set_materialized" ] -> (
      match node_frames env node with
      | Some frames -> Lexical_scope.bind_values env [ marker_key node ] (Materialized frames)
      | None -> env)
  | _ -> env

(* Whether a read of [node] from a _once context reads what that run wrote: the call's own tensor's
   value, or a node materialized before the call, each identifier the same binding at both. *)
let executed_read ~at_call ~at_read tensor node =
  let own_value =
    match (tensor, node.pexp_desc) with
    | ( Some n,
        Pexp_field
          ( { pexp_desc = Pexp_ident { txt = Lident m; _ }; _ },
            { txt = Lident "value" | Ldot (Lident "Tensor", "value"); _ } ) ) ->
        String.equal n m && same_binding ~at_call ~at_read n
    | _ -> false
  in
  own_value
  ||
  match (Lexical_scope.lookup at_call (marker_key node), node_frames at_read node) with
  | Some (Materialized marked), Some read -> List.equal phys_equal marked read
  | _ -> false

(* The scope after an open (or include) of a module denoting [denotes]: the harness's helpers, a
   support source's exports, or, for a module not modelled, nothing brought in. *)
let opened env = function
  | Some (Support exports) -> Lexical_scope.push_frame env exports
  | Some Harness ->
      Lexical_scope.push_frame env
        (Map.of_alist_exn
           (module String)
           (List.map
              [ "exact_partials"; "operand"; "matmul_products"; "reduction_cells"; "execute" ]
              ~f:(fun n -> (n, Helper n))))
  | Some Opaque | None -> env

let rec join a b =
  match (a, b) with
  | Bottom, v | v, Bottom -> v
  | Cell, Cell -> Cell
  | Constant, Constant -> Constant
  | Operand, Operand -> Operand
  | Reference, Reference -> Reference
  | References, References -> References
  | Ran, Ran -> Ran
  | Ran_once (a, x), Ran_once (b, y) when Option.equal String.equal a b && phys_equal x y ->
      Ran_once (a, x)
  | Products a, Products b -> Products (union a b)
  | Compared a, Compared b -> Compared (union a b)
  | Sums a, Sums b -> Sums (union a b)
  | Delayed a, Delayed b -> Delayed (join a b)
  | Tuple a, Tuple b when List.length a = List.length b -> Tuple (List.map2_exn a b ~f:join)
  | _ -> Unknown

let rec bind (env : env) p v =
  let env =
    {
      env with
      modules =
        List.fold (Lexical_scope.pattern_unpacks [ p ]) ~init:env.modules ~f:(fun modules n ->
            Map.set modules ~key:n ~data:Opaque);
    }
  in
  match (p.ppat_desc, v) with
  | Ppat_var { txt; _ }, _ -> Lexical_scope.bind_values env [ txt ] v
  | Ppat_constraint (p, _), _ -> bind env p v
  | Ppat_tuple ps, Tuple vs when List.length ps = List.length vs ->
      List.fold2_exn ps vs ~init:env ~f:bind
  | ( Ppat_construct ({ txt = Lident "::"; _ }, Some (_, { ppat_desc = Ppat_tuple [ h; t ]; _ })),
      References ) ->
      bind (bind env h Reference) t References
  | ( Ppat_construct ({ txt = Lident "::"; _ }, Some (_, { ppat_desc = Ppat_tuple [ h; t ]; _ })),
      Rows (head :: tail) ) ->
      bind (bind env h head) t (Rows tail)
  | _ -> Lexical_scope.bind_values env (Lexical_scope.pattern_vars [ p ]) Unknown

(* Products carry lexical builder identities. Do not export a builder's identity from a called
   wrapper: two invocations would otherwise look like the same products binding. *)
let rec wrapper_result = function
  | Products _ | Sums _ | Rows _ | Helper _ | Closure _ | Constant | Materialized _ -> Unknown
  | Tuple vs -> Tuple (List.map vs ~f:wrapper_result)
  | Delayed v -> Delayed (wrapper_result v)
  | v -> v

let rec value fuel env e =
  let v = value fuel env in
  match e.pexp_desc with
  | Pexp_ident { txt = Lident n; _ } -> Option.value (Lexical_scope.lookup env n) ~default:Unknown
  | Pexp_ident { txt; _ } -> (
      match exported env txt with
      | Some v -> v
      | None -> Option.value_map (helper env txt) ~default:Unknown ~f:(fun h -> Helper h))
  | Pexp_constant (Pconst_float _) -> Constant
  | Pexp_function (ps, _, Pfunction_body body) -> Closure (ps, body, env)
  | Pexp_let (Nonrecursive, bs, body) ->
      let facts = List.map bs ~f:(fun b -> v b.pvb_expr) in
      let inner =
        List.fold2_exn bs facts ~init:env ~f:(fun env b fact -> bind env b.pvb_pat fact)
      in
      value fuel inner body
  | Pexp_sequence (first, last) -> value fuel (materialize env first) last
  | Pexp_tuple es -> Tuple (List.map es ~f:v)
  | Pexp_array es -> Rows (List.map es ~f:v)
  | Pexp_construct ({ txt = Lident "[]"; _ }, None) -> Rows []
  | Pexp_construct ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ h; t ]; _ }) -> (
      match v t with Rows tail -> Rows (v h :: tail) | _ -> Unknown)
  | Pexp_lazy e -> Delayed (v e)
  (* A local open of the harness or a support module, as the walker's scope reads it; one of a
     module not modelled could shadow any name, so its body confers nothing. *)
  | Pexp_open ({ popen_expr = { pmod_desc = Pmod_ident { txt = Lident m; _ }; _ }; _ }, body)
    when match Map.find env.modules m with Some (Harness | Support _) -> true | _ -> false ->
      value fuel (opened env (Map.find env.modules m)) body
  | Pexp_ifthenelse (_, a, Some b) -> join (v a) (v b)
  | Pexp_match (scrutinee, cases) ->
      let fact = v scrutinee in
      List.fold cases ~init:Bottom ~f:(fun acc c ->
          join acc (value fuel (bind env c.pc_lhs fact) c.pc_rhs))
  | Pexp_assert { pexp_desc = Pexp_construct ({ txt = Lident "false"; _ }, None); _ } -> Bottom
  | Pexp_apply (f, args) -> (
      let vs = List.map (positional args) ~f:v in
      match (v f, vs) with
      | Helper "operand", [ _ ] when labels args [ "b"; "o"; "i" ] -> Operand
      | Helper "operand", [ _; _ ] when labels args [ "b"; "o"; "i" ] -> Cell
      | Operand, [ _ ] -> Cell
      | Helper "matmul_products", [ a; b ]
        when labels args [ "m"; "n"; "k" ] && reader fuel a && reader fuel b ->
          Products [ builder e ]
      | Helper "reduction_cells", [ fn ] when labels args [ "out"; "over" ] -> (
          match apply fuel fn [ (Nolabel, Unknown); (Nolabel, Unknown) ] with
          | Cell -> Products [ builder e ]
          | _ -> Unknown)
      | Helper "execute", _ -> References
      | _, [ _ ] when named env f [ "invalid_arg"; "failwith"; "raise" ] -> Bottom
      | _, _ when named env f [ "Context.run" ] -> Ran
      | _, [ _; _ ] when named env f once_runs ->
          let tensor =
            match positional args with
            | [ _; { pexp_desc = Pexp_ident { txt = Lident n; _ }; _ } ] -> Some n
            | _ -> None
          in
          Ran_once (tensor, env)
      | _, [ Ran; _ ] when named env f [ "Context.get_values" ] -> Reference
      | _, [ Ran_once (tensor, at_call); _ ] when named env f [ "Context.get_values" ] -> (
          match positional args with
          | [ _; node ] when executed_read ~at_call ~at_read:env tensor node -> Reference
          | _ -> Unknown)
      | _, [ References ] when named env f [ "List.hd_exn"; "Base.List.hd_exn" ] -> Reference
      | _, [ Delayed x ] when named env f [ "Lazy.force" ] -> x
      | _, [ Tuple (a :: _) ] when named env f [ "fst"; "Stdlib.fst" ] -> a
      | _, [ Tuple [ _; b ] ] when named env f [ "snd"; "Stdlib.snd" ] -> b
      | (_, [ Cell; Cell ] | _, [ Cell; Constant ] | _, [ Constant; Cell ])
        when named env f [ "*." ] ->
          Cell
      | _, [ Sums ids ] when named env f [ "Array.of_list"; "Base.Array.of_list" ] -> Sums ids
      | _, [ Products ids ] when named env f [ "List.map"; "Base.List.map" ] -> (
          match label args "f" with
          | Some { pexp_desc = Pexp_apply (sum, sa); _ }
            when named env sum [ "List.sum"; "Base.List.sum" ]
                 && (match positional sa with
                   | [
                    {
                      pexp_desc =
                        Pexp_pack { pmod_desc = Pmod_ident { txt = Lident "Float"; _ }; _ };
                      _;
                    };
                   ] ->
                       true
                   | _ -> false)
                 && Option.value_map (label sa "f") ~default:false ~f:(fun f ->
                     named env f [ "Fn.id" ]) ->
              Sums ids
          | _ -> Unknown)
      | _, [ _; Reference; Sums ids ]
        when named env f [ "p_all2"; "Verdict.Claims.p_all2" ]
             && Option.value_map (label args "f") ~default:false ~f:(fun f ->
                 named env f [ "Float.equal" ]) ->
          Compared ids
      | (Closure _ as fn), _ ->
          wrapper_result (apply fuel fn (List.map args ~f:(fun (l, e) -> (l, v e))))
      | _ -> Unknown)
  | _ -> Unknown

and reader fuel = function
  | Operand -> true
  | Closure _ as fn -> (
      match apply fuel fn [ (Nolabel, Unknown) ] with Cell -> true | _ -> false)
  | _ -> false

and apply fuel fn args =
  match fn with
  | Closure (params, body, env) when fuel > 0 ->
      let rec parameters env remaining = function
        | [] ->
            let result = value (fuel - 1) env body in
            if List.is_empty remaining then result else apply (fuel - 1) result remaining
        | p :: ps -> (
            match p.pparam_desc with
            | Pparam_val (lab, default, pat) -> (
                let matches l =
                  match (lab, l) with
                  | Nolabel, Nolabel -> true
                  | (Labelled a | Optional a), (Labelled b | Optional b) -> String.equal a b
                  | _ -> false
                in
                match List.findi remaining ~f:(fun _ (l, _) -> matches l) with
                | Some (i, (_, fact)) ->
                    parameters (bind env pat fact)
                      (List.filteri remaining ~f:(fun j _ -> i <> j))
                      ps
                | None -> (
                    match lab with
                    | Optional _ ->
                        parameters
                          (bind env pat
                             (Option.value_map default ~default:Unknown ~f:(value (fuel - 1) env)))
                          remaining ps
                    | _ -> Unknown))
            | Pparam_newtype _ -> parameters env remaining ps)
      in
      parameters env args params
  | Operand when List.length args = 1 -> Cell
  | _ -> Unknown

(* Walks one source's structure under [support]: its sites, and its top-level value bindings with
   what each denotes right after its own item ([None] once a top-level include makes that
   unknown). *)
let scan support structure =
  let sites = Hashtbl.create (module Int) and candidates = Hashtbl.create (module Int) in
  let compared = Hash_set.create (module String) in
  let exports = ref (Some (Map.empty (module String))) in
  let walker =
    object (self)
      inherit [value, denotes] Lexical_scope.scoped as super
      method local = Unknown

      method module_path env =
        function
        | Lident n -> Some (Option.value (Map.find env.modules n) ~default:Opaque)
        | _ -> Some Opaque

      method! shadowed = Some Opaque

      method! item_scope ~top env item =
        (if top then
           match item.pstr_desc with
           | Pstr_value (_, bindings) ->
               exports :=
                 Option.map !exports ~f:(fun exports ->
                     List.fold
                       (Lexical_scope.pattern_vars (List.map bindings ~f:(fun b -> b.pvb_pat)))
                       ~init:exports
                       ~f:(fun exports n ->
                         Map.set exports ~key:n
                           ~data:(Option.value (Lexical_scope.lookup env n) ~default:Unknown)))
           | Pstr_include _ -> exports := None
           | _ -> ());
        env

      method! binding_denotes env b = value 3 env b.pvb_expr
      method! recursive_denotes ~top:_ _ bindings = List.map bindings ~f:(fun _ -> Unknown)

      method! denotations env flag bindings =
        match flag with
        | Recursive -> List.map bindings ~f:(fun _ -> Unknown)
        | Nonrecursive -> super#denotations env flag bindings

      method! bind_group env bindings facts =
        List.fold2_exn bindings facts ~init:env ~f:(fun env b fact -> bind env b.pvb_pat fact)

      method! opened ~top:_ ~include_:_ env denotes = opened env denotes

      method! expression env e =
        (match e.pexp_desc with
        | Pexp_apply (f, args) -> (
            (match value 3 env f with
            | Helper "exact_partials" ->
                Hashtbl.set sites ~key:(offset e) ~data:e.pexp_loc.loc_start.pos_lnum
            | _ -> ());
            (if named env f [ "p_all"; "Verdict.Claims.p_all" ] then
               match (positional args, label args "f") with
               | [ _; products ], Some ({ pexp_desc = Pexp_apply (pred, _); _ } as app)
                 when match value 3 env pred with Helper "exact_partials" -> true | _ -> false -> (
                   match value 3 env products with
                   | Products ids -> Hashtbl.set candidates ~key:(offset app) ~data:ids
                   | _ -> ())
               | _ -> ());
            (match value 3 env e with
            | Compared ids -> List.iter ids ~f:(Hash_set.add compared)
            | _ -> ());
            if named env f [ "List.iter"; "Base.List.iter" ] then
              match (positional args, label args "f") with
              | ( [ rows ],
                  Some
                    {
                      pexp_desc =
                        Pexp_function
                          ( [ { pparam_desc = Pparam_val (Nolabel, None, pat); _ } ],
                            _,
                            Pfunction_body body );
                      _;
                    } ) -> (
                  match value 3 env rows with
                  | Rows (first :: rest) ->
                      let joined = List.fold rest ~init:first ~f:join in
                      ignore (self#expression (bind env pat joined) body : expression)
                  | _ -> ())
              | _ -> ())
        | _ -> ());
        match e.pexp_desc with
        | Pexp_sequence (first, rest) ->
            ignore (self#expression env first : expression);
            ignore (self#expression (materialize env first) rest : expression);
            e
        | _ -> super#expression env e
    end
  in
  ignore (walker#structure (initial support) structure : structure);
  let sites =
    Hashtbl.to_alist sites
    |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:(fun (key, line) ->
        {
          line;
          covered =
            Option.value_map (Hashtbl.find candidates key) ~default:false ~f:(fun ids ->
                List.for_all ids ~f:(Hash_set.mem compared));
        })
  in
  (sites, !exports)

let parse ?(name = "") source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf name;
  Parse.implementation lexbuf

let read ?(support = no_support) source = fst (scan support (parse source))

(* [support] with the module [name] read from [source]; a top-level include exports nothing. *)
let add_support support ~name source =
  let exports =
    Option.value
      (snd (scan support (parse ~name:("support " ^ name) source)))
      ~default:(Map.empty (module String))
  in
  Map.set support ~key:name ~data:(Support exports)

(* The closed list of support sources (gh-ocannl-2334), in the order each may use the earlier. *)
let support_sources = [ "test/support/exact_sums.ml" ]

let module_name path =
  String.capitalize (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))

(* [support_sources] read through [source_of]; one that is absent or does not parse is left out, so
   its callers confer nothing (its own read reports the parse failure). *)
let support ~source_of =
  List.fold support_sources ~init:no_support ~f:(fun support path ->
      match source_of path with
      | None -> support
      | Some source -> (
          match
            Scan_attempt.run (fun () -> add_support support ~name:(module_name path) source)
          with
          | Scanned support -> support
          | Unparsed _ | Raised _ -> support))

let uncovered sites = List.count sites ~f:(fun s -> not s.covered)

let violations ~debt rows =
  let actual =
    Map.of_alist_exn
      (module String)
      (List.map rows ~f:(fun (path, sites) -> (path, uncovered sites)))
  in
  let growth =
    List.filter_map rows ~f:(fun (path, sites) ->
        let n = uncovered sites
        and allowed = Option.value (List.Assoc.find debt path ~equal:String.equal) ~default:0 in
        if n > allowed then
          Some (Printf.sprintf "%s: unchecked premise count grew: %d > %d" path n allowed)
        else None)
  in
  let stale =
    List.filter_map debt ~f:(fun (path, n) ->
        let got = Option.value (Map.find actual path) ~default:0 in
        if n <= 0 || got < n then
          Some
            (Printf.sprintf "%s: stale premise debt row: %d now, recorded %d; lower or remove it"
               path got n)
        else None)
  in
  growth @ stale

let read_or_refusal ~reader ~source content =
  match Scan_attempt.run (fun () -> reader content) with
  | Scanned sites -> Ok sites
  | Unparsed message -> Error (Printf.sprintf "%s: premise input does not parse: %s" source message)
  | Raised message -> Error (Printf.sprintf "%s: premise scanner raised: %s" source message)
