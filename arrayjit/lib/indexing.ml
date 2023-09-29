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
  mutable static_range : int option; [@compare.ignore] [@equal.ignore]
}
[@@deriving compare, equal, sexp, hash]

type 'a bindings = Empty | Bind of static_symbol * int ref * (int -> 'a) bindings

let rec sexp_of_bindings : 'a. 'a bindings -> Sexp.t =
 fun (type a) (b : a bindings) ->
  match b with
  | Empty -> Sexp.Atom "bindings"
  | Bind (s, i, bs) -> Sexp.List [ sexp_of_static_symbol s; sexp_of_int !i; sexp_of_bindings bs ]

let assoc_of_bindings bs =
  let rec loop : 'a. 'a bindings -> (static_symbol, int ref) List.Assoc.t =
   fun (type a) (b : a bindings) -> match b with Empty -> [] | Bind (s, i, bs) -> (s, i) :: loop bs
  in
  List.rev @@ loop bs

(** Helps manipulating the bindings. *)
type 'a variadic = Result of (unit -> 'a) | Param of int ref * (int -> 'a) variadic

let rec apply : 'a. 'a variadic -> 'a =
 fun (type b) (f : b variadic) -> match f with Result rf -> rf () | Param (i, more) -> apply more !i

let get_static_symbol ?static_range bindings =
  let s = { static_symbol = get_symbol (); static_range } in
  let r = ref 0 in
  (s, r, Bind (s, r, bindings))

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x " @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ Int.to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string

type axis_index =
  | Fixed_idx of int  (** The specific position along an axis. *)
  | Iterator of symbol
      (** The given member of the [product_space] corresponding to some [product_iterators]. *)
[@@deriving compare, equal, sexp, variants]

type str_osym_map = (string, symbol option, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_osym_map (map : str_osym_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * symbol option])

type projections_debug = { spec : string; derived_for : Sexp.t } [@@deriving sexp]

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
      (** The dimensions of the RHS arrays, needed for deriving projections from other projections. *)
  product_iterators : symbol array;
      (** The product space iterators (concatentation of the relevant batch, output, input axes)
      for iterating over the [product_space] axes, where same axes are at same array indices. *)
  project_lhs : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the result of
      an operation. *)
  project_rhs : axis_index array array;
      (** [project_rhs.(i)] Produces an index into the [i+1]th argument of an operation. *)
  debug_info : (projections_debug[@sexp.ignore] [@compare.ignore] [@equal.ignore]);
  unique_debug_id : int;
}
[@@deriving compare, equal, sexp]
(** All the information relevant for code generation. *)

let iterated dim = dim > 1
let opt_symbol d = if iterated d then Some (get_symbol ()) else None
let opt_iterator = function None -> Fixed_idx 0 | Some sym -> Iterator sym

(** Projections for a pointwise unary operator. *)
let identity_projections ~debug_info ~lhs_dims =
  let product_iterators = Array.map lhs_dims ~f:opt_symbol in
  let project_lhs = Array.map product_iterators ~f:opt_iterator in
  let product_space = Array.filter ~f:iterated lhs_dims in
  let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
  {
    product_space;
    lhs_dims;
    rhs_dims = [| lhs_dims |];
    product_iterators;
    project_lhs;
    project_rhs = [| project_lhs |];
    debug_info;
    unique_debug_id = unique_debug_id ();
  }

let derive_index ~product_syms ~(projection : axis_index array) =
  let sym_to_i =
    Array.mapi product_syms ~f:(fun i s -> (s, i)) |> Array.to_list |> Map.of_alist_exn (module Symbol)
  in
  let positions =
    Array.map projection ~f:(function
      | Iterator s when Map.mem sym_to_i s -> Either.First (Map.find_exn sym_to_i s)
      | it -> Second it)
  in
  fun ~product -> Array.map positions ~f:(function First p -> product.(p) | Second it -> it)

module IDX = struct
  let empty = Empty
  let get_static_symbol = get_static_symbol
end
