open Base
(* gh-ocannl-2194. Recognised forms (closed).

   Input: OCaml implementations supplied by the driver, whose corpus is .ml under test/ and
   arrayjit/test/, excluding test/ppx/*_expected.ml generated snapshots.

   Census: every application whose callee is [Ll_test.exact_partials], including partial
   applications used as predicates. Module aliases, local module aliases, opens and aliases of
   unapplied helper identifiers resolve through [Test_scan.Lexical_scope]. Comments and string
   contents are not traversed as programs.

   Coverage is PER SITE, deliberately conservative: the enclosing [p_all] supplies a products
   binding built by [Ll_test.matmul_products] from two [Ll_test.operand] readers. The SAME binding
   must occur in [p_all2 label reference (Array.of_list (List.map products ~f:(List.sum (module
   Float) ~f:Fn.id))) ~f:Float.equal]. The reference is [List.hd_exn (Ll_test.execute ...)], inline
   or let-bound. These claims are recognised under [Verdict.Claims], or bare as after its open.
   Products and reference bindings retain their lexical identities, so another scope's same spelling
   does not cover a site.

   Only plain-variable let bindings and plain identifier aliases propagate summaries. Pattern
   destructuring, wrappers and type constraints confer no coverage. Operand readers may be inline or
   let-bound. Operand calls require b/o/i labels and one positional initializer; matmul_products
   requires m/n/k labels and two positional readers. List/Array/Float/Fn and bare claim names are
   read only when their head is not locally rebound. Aliases of those standard modules do not confer
   coverage; unknown opens are not resolved. Ll_test and value aliases ARE scoped. No claim is made
   that a syntactically executed reference computes the intended operation.

   Every other censused site remains debt, INCLUDING already-correct tests expressed through
   reduction_cells, wrappers, tuple-dispatched products, a direct fully-applied exact_partials, or
   another comparison spelling. This is a migration protocol, not general dataflow analysis.
   Function bodies, branches and quotation AST payloads are traversed, without reachability
   analysis. Functor results and first-class module contents are not resolved as Ll_test;
   dynamically obtained callees and calls hidden in external/generated source are outside the
   census. Unknown forms never confer coverage. The helper's own mathematical unit tests remain
   counted debt too; their rows explain that they are not operand premises. *)

open Ppxlib

type value = Unknown | Helper of string | Operand | Products of int | Reference | Sums of int
type env = (value, bool) Test_scan.Lexical_scope.env
type site = { line : int; covered : bool }

let initial : env =
  { frames = []; modules = Map.of_alist_exn (module String) [ ("Ll_test", true) ] }

let helper env = function
  | Longident.Lident name -> (
      match Test_scan.Lexical_scope.lookup env name with Some (Helper h) -> Some h | _ -> None)
  | Ldot (Lident m, name) when Map.find env.modules m |> Option.value ~default:false -> Some name
  | _ -> None

let rec path_name = function
  | Longident.Lident n -> Some n
  | Ldot (path, n) -> Option.map (path_name path) ~f:(fun prefix -> prefix ^ "." ^ n)
  | Lapply _ -> None

let name e = match e.pexp_desc with Pexp_ident { txt; _ } -> path_name txt | _ -> None

let named env e names =
  let unshadowed =
    match e.pexp_desc with
    | Pexp_ident { txt = Lident n; _ } -> Option.is_none (Test_scan.Lexical_scope.lookup env n)
    | Pexp_ident { txt; _ } -> not (Map.mem env.modules (Test_scan.Lexical_scope.head txt))
    | _ -> false
  in
  unshadowed
  && Option.value_map (name e) ~default:false ~f:(fun n -> List.mem names n ~equal:String.equal)

let positional args = List.filter_map args ~f:(function Nolabel, e -> Some e | _ -> None)

let label args n =
  List.find_map args ~f:(function Labelled k, e when String.equal k n -> Some e | _ -> None)

let labels args names = List.for_all names ~f:(fun n -> Option.is_some (label args n))
let offset e = e.pexp_loc.loc_start.pos_cnum

let rec value env e =
  match e.pexp_desc with
  | Pexp_ident { txt = Lident n; _ } ->
      Option.value (Test_scan.Lexical_scope.lookup env n) ~default:Unknown
  | Pexp_ident { txt; _ } ->
      Option.value_map (helper env txt) ~default:Unknown ~f:(fun h -> Helper h)
  | Pexp_apply (f, args) -> (
      let vs = List.map (positional args) ~f:(value env) in
      match (value env f, vs) with
      | Helper "operand", [ _ ] when labels args [ "b"; "o"; "i" ] -> Operand
      | Helper "matmul_products", [ Operand; Operand ] when labels args [ "m"; "n"; "k" ] ->
          Products (offset e)
      | _, [ run ] when named env f [ "List.hd_exn"; "Base.List.hd_exn" ] -> (
          match run with Reference -> Reference | _ -> Unknown)
      | Helper "execute", _ -> Reference
      | _, [ Sums id ] when named env f [ "Array.of_list"; "Base.Array.of_list" ] -> Sums id
      | _, [ Products id ] when named env f [ "List.map"; "Base.List.map" ] -> (
          match label args "f" with
          | Some { pexp_desc = Pexp_apply (sum, sum_args); _ }
            when named env sum [ "List.sum"; "Base.List.sum" ]
                 && (match positional sum_args with
                   | [
                    {
                      pexp_desc =
                        Pexp_pack { pmod_desc = Pmod_ident { txt = Lident "Float"; _ }; _ };
                      _;
                    };
                   ] ->
                       true
                   | _ -> false)
                 && Option.value_map (label sum_args "f") ~default:false ~f:(fun f ->
                     named env f [ "Fn.id" ]) ->
              Sums id
          | _ -> Unknown)
      | _ -> Unknown)
  | _ -> Unknown

let read source =
  let sites = Hashtbl.create (module Int) in
  let candidates = Hashtbl.create (module Int) in
  let compared = Hash_set.create (module Int) in
  let walker =
    object
      inherit [value, bool] Test_scan.Lexical_scope.scoped as super
      method local = Unknown

      method module_path env =
        function
        | Lident n -> Some (Option.value (Map.find env.modules n) ~default:false)
        | _ -> Some false

      method! shadowed = Some false

      method! binding_denotes env b =
        match b.pvb_pat.ppat_desc with Ppat_var _ -> value env b.pvb_expr | _ -> Unknown

      method! opened ~top:_ ~include_:_ env =
        function
        | Some true ->
            Test_scan.Lexical_scope.push_frame env
              (Map.of_alist_exn
                 (module String)
                 (List.map [ "exact_partials"; "operand"; "matmul_products"; "execute" ]
                    ~f:(fun n -> (n, Helper n))))
        | _ -> env

      method! expression env e =
        (match e.pexp_desc with
        | Pexp_apply (f, args) -> (
            (match value env f with
            | Helper "exact_partials" ->
                Hashtbl.set sites ~key:(offset e) ~data:e.pexp_loc.loc_start.pos_lnum
            | _ -> ());
            (if named env f [ "p_all"; "Verdict.Claims.p_all" ] then
               match (positional args, label args "f") with
               | [ _; products ], Some ({ pexp_desc = Pexp_apply (pred, _); _ } as application)
                 when match value env pred with Helper "exact_partials" -> true | _ -> false -> (
                   match value env products with
                   | Products id -> Hashtbl.set candidates ~key:(offset application) ~data:id
                   | _ -> ())
               | _ -> ());
            if named env f [ "p_all2"; "Verdict.Claims.p_all2" ] then
              match (positional args, label args "f") with
              | [ _; reference; sums ], Some equality when named env equality [ "Float.equal" ] -> (
                  match (value env reference, value env sums) with
                  | Reference, Sums id -> Hash_set.add compared id
                  | _ -> ())
              | _ -> ())
        | _ -> ());
        super#expression env e
    end
  in
  ignore (walker#structure initial (Parse.implementation (Lexing.from_string source)) : structure);
  Hashtbl.to_alist sites
  |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
  |> List.map ~f:(fun (key, line) ->
      {
        line;
        covered =
          Option.value_map (Hashtbl.find candidates key) ~default:false ~f:(Hash_set.mem compared);
      })

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
  match Test_scan.Scan_attempt.run (fun () -> reader content) with
  | Scanned sites -> Ok sites
  | Unparsed message -> Error (Printf.sprintf "%s: premise input does not parse: %s" source message)
  | Raised message -> Error (Printf.sprintf "%s: premise scanner raised: %s" source message)
