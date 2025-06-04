open Base

type symbol = Symbol of int [@@deriving compare, equal, sexp, hash, variants]

let unique_id = ref 1

let get_symbol () =
  let uid = !unique_id in
  Int.incr unique_id;
  Symbol uid

module CompareSymbol = struct
  type t = symbol = Symbol of int [@@deriving compare, equal, sexp, hash]
end

module Symbol = struct
  include CompareSymbol
  include Comparator.Make (CompareSymbol)
end

let symbol_ident (Symbol s) = "i" ^ Int.to_string s

type 'a environment = 'a Map.M(Symbol).t [@@deriving sexp]

let empty_env : 'a environment = Map.empty (module Symbol)

type static_symbol = {
  static_symbol : symbol;
  mutable static_range : int option; [@compare.ignore] [@equal.ignore] [@hash.ignore]
}
[@@deriving compare, equal, sexp, hash]

type 'a bindings = Empty | Bind of static_symbol * (int -> 'a) bindings [@@deriving sexp_of]

let bound_symbols bs =
  let rec loop : 'a. 'a bindings -> static_symbol list =
   fun (type a) (b : a bindings) -> match b with Empty -> [] | Bind (s, bs) -> s :: loop bs
  in
  (* Reverse order to match [lowered_bindings]. *)
  List.rev @@ loop bs

(** Helps lowering the bindings. *)
type ('r, 'idcs, 'p1, 'p2) variadic =
  | Result of 'r
  | Param_idx of int ref * (int -> 'r, int -> 'idcs, 'p1, 'p2) variadic
  | Param_1 of 'p1 option ref * ('p1 -> 'r, 'idcs, 'p1, 'p2) variadic
  | Param_2 of 'p2 option ref * ('p2 -> 'r, 'idcs, 'p1, 'p2) variadic
  | Param_2f :
      ('p2f -> 'p2) * 'p2f option ref * ('p2 -> 'r, 'idcs, 'p1, 'p2) variadic
      -> ('r, 'idcs, 'p1, 'p2) variadic

type unit_bindings = (unit -> unit) bindings [@@deriving sexp_of]
type lowered_bindings = (static_symbol, int ref) List.Assoc.t [@@deriving sexp_of]

(** [apply run_variadic ()] applies the parameters in reverse order to how they appear in the
    [run_variadic] list. *)
let rec apply : 'r 'idcs 'p1 'p2. ('r, 'idcs, 'p1, 'p2) variadic -> 'r =
 fun (type r idcs p1 p2) (f : (r, idcs, p1, p2) variadic) ->
  match f with
  | Result rf -> rf
  | Param_idx (i, more) -> apply more !i
  | Param_1 ({ contents = Some p1 }, more) -> apply more p1
  | Param_2 ({ contents = Some p2 }, more) -> apply more p2
  | Param_2f (pf, { contents = Some p2 }, more) -> apply more @@ pf p2
  | Param_1 ({ contents = None }, _) -> invalid_arg "Indexing.apply: Param_1 missing value"
  | Param_2 ({ contents = None }, _) -> invalid_arg "Indexing.apply: Param_2 missing value"
  | Param_2f (_, { contents = None }, _) -> invalid_arg "Indexing.apply: Param_2 missing value"

let lowered_bindings bs vs =
  let rec loop : 'r 'idcs. 'idcs bindings * ('r, 'idcs, 'p1, 'p2) variadic -> lowered_bindings =
   fun (type r idcs) (f : idcs bindings * (r, idcs, 'p1, 'p2) variadic) ->
    match f with
    | Empty, Result _ -> []
    | Bind (s, bs), Param_idx (i, vs) -> (s, i) :: loop (bs, vs)
    | bs, Param_1 (_, vs) -> loop (bs, vs)
    | bs, Param_2 (_, vs) -> loop (bs, vs)
    | bs, Param_2f (_, _, vs) -> loop (bs, vs)
    | Empty, _ -> assert false
    | Bind _, Result _ -> assert false
  in
  (* Reverse order because [apply] above also reverses the order! *)
  List.rev @@ loop (bs, vs)

let find_exn (bs : lowered_bindings) = List.Assoc.find_exn ~equal:equal_static_symbol bs

let get_static_symbol ?static_range bindings =
  let s = { static_symbol = get_symbol (); static_range } in
  (s, Bind (s, bindings))

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x "
    @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ Int.to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string

type axis_index =
  | Fixed_idx of int  (** A fixed position along an axis *)
  | Iterator of symbol  (** A simple iterator symbol *)
  | Affine of { symbols : (int * symbol) list; offset : int }
      (** An affine combination of symbols with coefficients and an offset. Represents:

          Σ(coeff_i * symbol_i) + offset

          For convolutions: [symbols = [(stride, i1); (dilation, i2)]] and [offset = ~-padding].
          Note: for readability, we use [Fixed_idx] and [Iterator] as separate variants and require
          [Affine] to not be ambiguous: [symbols] should be longer than 1 or have a coefficient
          different from 1 and 0. *)
[@@deriving compare, equal, sexp]

type str_osym_map = (string, symbol option, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_osym_map (map : str_osym_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * symbol option])

type projections_debug = { spec : string; derived_for : Sexp.t; trace : (string * int) list }
[@@deriving sexp]

let unique_debug_id =
  let projections_uid = ref 0 in
  fun () ->
    Int.incr projections_uid;
    !projections_uid

type projections = {
  product_space : int array;
      (** The product space dimensions that an operation should parallelize (map-reduce) over. *)
  lhs_dims : int array;  (** The dimensions of the LHS array. *)
  rhs_dims : int array array;
      (** The dimensions of the RHS arrays, needed for deriving projections from other projections.
      *)
  product_iterators : symbol array;
      (** The product space iterators (concatentation of the relevant batch, output, input axes) for
          iterating over the [product_space] axes, where same axes are at same array indices. *)
  project_lhs : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the
          result of an operation. *)
  project_rhs : axis_index array array;
      (** [project_rhs.(i)] Produces an index into the [i+1]th argument of an operation. *)
  debug_info : (projections_debug[@sexp.ignore] [@compare.ignore] [@equal.ignore]);
}
[@@deriving compare, equal, sexp]
(** All the information relevant for code generation. *)

let iterated dim = dim > 1
let opt_symbol d = if iterated d then Some (get_symbol ()) else None
let opt_iterator = function None -> Fixed_idx 0 | Some sym -> Iterator sym

let is_bijective proj =
  let lhs_symbols =
    Set.of_array (module Symbol)
    @@ Array.concat_map proj.project_lhs ~f:(function
         | Iterator s -> [| s |]
         | Fixed_idx _ -> [||]
         | Affine { symbols; _ } ->
             (* For affine indices, we consider all symbols with coefficient 1 *)
             List.filter_map symbols ~f:(fun (coeff, s) -> if coeff = 1 then Some s else None)
             |> Array.of_list)
  in
  Set.equal lhs_symbols (Set.of_array (module Symbol) proj.product_iterators)

(** Projections for a pointwise unary operator. Provide only one of [debug_info] or [derived_for].
*)
let identity_projections ?debug_info ?derived_for ~lhs_dims () =
  let product_iterators = Array.map lhs_dims ~f:opt_symbol in
  let project_lhs = Array.map product_iterators ~f:opt_iterator in
  let product_space = Array.filter ~f:iterated lhs_dims in
  let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
  let debug_info =
    match (debug_info, derived_for) with
    | Some debug_info, _ ->
        {
          debug_info with
          trace = ("indentity_projections", unique_debug_id ()) :: debug_info.trace;
        }
    | None, Some derived_for ->
        {
          spec = "";
          derived_for = Sexp.Atom derived_for;
          trace = [ ("indentity_projections", unique_debug_id ()) ];
        }
    | None, None ->
        {
          spec = "";
          derived_for = Sexp.Atom "";
          trace = [ ("indentity_projections", unique_debug_id ()) ];
        }
  in
  {
    product_space;
    lhs_dims;
    rhs_dims = [| lhs_dims |];
    product_iterators;
    project_lhs;
    project_rhs = [| project_lhs |];
    debug_info;
  }

let derive_index ~product_syms ~(projection : axis_index array) =
  let sym_to_i =
    Array.mapi product_syms ~f:(fun i s -> (s, i))
    |> Array.to_list
    |> Map.of_alist_exn (module Symbol)
  in
  let positions =
    Array.map projection ~f:(function
      | Iterator s when Map.mem sym_to_i s -> Either.First (Map.find_exn sym_to_i s)
      | Fixed_idx _ as it -> Second it
      | Affine _ as it -> Second it
      | Iterator _ as it -> Second it)
  in
  fun ~product ->
    Array.map positions ~f:(function
      | First p -> product.(p)
      | Second (Fixed_idx i) -> i
      | Second (Iterator s) ->
          (* This shouldn't happen if sym_to_i is complete *)
          failwith ("derive_index: unresolved iterator " ^ symbol_ident s)
      | Second (Affine { symbols; offset }) ->
          List.fold symbols ~init:offset ~f:(fun acc (coeff, s) ->
              match Map.find sym_to_i s with
              | Some idx -> acc + (coeff * product.(idx))
              | None ->
                  failwith ("derive_index: unresolved symbol in affine index " ^ symbol_ident s)))

module Pp_helpers = struct
  open PPrint

  let pp_comma () = comma ^^ space
  let pp_symbol sym = string (symbol_ident sym)

  let pp_static_symbol { static_symbol; static_range } =
    match static_range with
    | None -> pp_symbol static_symbol
    | Some range ->
        infix 4 1 colon (pp_symbol static_symbol) (brackets (string "0.." ^^ OCaml.int (range - 1)))

  let pp_axis_index = function
    | Iterator sym -> pp_symbol sym
    | Fixed_idx i -> OCaml.int i
    | Affine { symbols; offset } -> (
        let terms =
          List.map symbols ~f:(fun (coeff, sym) ->
              if coeff = 1 then pp_symbol sym else OCaml.int coeff ^^ string "*" ^^ pp_symbol sym)
        in
        let all_terms =
          if offset = 0 then terms
          else if offset > 0 then terms @ [ OCaml.int offset ]
          else terms @ [ string "-" ^^ OCaml.int (-offset) ]
        in
        match all_terms with
        | [] -> OCaml.int 0
        | [ t ] -> t
        | t :: ts -> List.fold ts ~init:t ~f:(fun acc t -> acc ^^ string "+" ^^ t))

  let pp_indices idcs = separate (pp_comma ()) (Array.to_list idcs |> List.map ~f:pp_axis_index)
  let print ppf doc = ToFormatter.pretty 1.0 80 ppf doc
end

module Doc_helpers = struct
  let ( ^^ ) = PPrint.( ^^ )
  let ( !^ ) = PPrint.( !^ )
  let int = PPrint.OCaml.int
  let comma_sep = PPrint.(comma ^^ space)
  let pp_comma () = comma_sep
  let pp_symbol sym = PPrint.string @@ symbol_ident sym

  let pp_static_symbol { static_symbol; static_range } =
    match static_range with
    | None -> pp_symbol static_symbol
    | Some range ->
        PPrint.infix 4 1 PPrint.colon (pp_symbol static_symbol)
          (PPrint.brackets (PPrint.string "0.." ^^ int (range - 1)))

  let pp_axis_index idx =
    match idx with
    | Iterator sym -> pp_symbol sym
    | Fixed_idx i -> PPrint.OCaml.int i
    | Affine { symbols; offset } -> (
        let open PPrint in
        let terms =
          List.map symbols ~f:(fun (coeff, sym) ->
              if coeff = 1 then pp_symbol sym else int coeff ^^ string "*" ^^ pp_symbol sym)
        in
        let all_terms =
          if offset = 0 then terms
          else if offset > 0 then terms @ [ int offset ]
          else terms @ [ string "-" ^^ int (-offset) ]
        in
        match all_terms with
        | [] -> int 0
        | [ t ] -> t
        | t :: ts -> List.fold ts ~init:t ~f:(fun acc t -> acc ^^ string "+" ^^ t))

  let pp_indices idcs =
    PPrint.separate (pp_comma ()) (Array.to_list idcs |> List.map ~f:pp_axis_index)
end
