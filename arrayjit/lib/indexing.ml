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

type dedicated_axis = Task_id | Sample_num [@@deriving equal, compare, sexp, variants]

type axis_special =
  | Dim  (** A "randomly accessed" or "frequently reduced" axis -- no optimization hint. *)
  | Dedicated of dedicated_axis  (** An axis whose iteration can have special treatment on some backends. *)
  | Frozen
      (** An axis that should be indexed at a single position during a single `refresh_session`:
          a dynamic index into it will be a [Frozen_recipient]. *)
[@@deriving equal, compare, sexp, variants]

type dim = { special : axis_special; dim : int } [@@deriving equal, compare, sexp]

let dim dim = { special = Dim; dim }
let frozen dim = { special = Frozen; dim }
let parallel dim = { special = Dedicated Task_id; dim }
let minibatch dim = { special = Dedicated Sample_num; dim }
let dim_1 = function { special = Dedicated _; _ } -> false | { dim = 1; _ } -> true | _ -> false

let dim_to_string = function
  | { special = Dim; dim } -> Int.to_string dim
  | { special = Frozen; dim } -> "frozen " ^ Int.to_string dim
  | { special = Dedicated Task_id; dim } -> "parallel " ^ Int.to_string dim
  | { special = Dedicated Sample_num; dim } -> "minibatch " ^ Int.to_string dim

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x " @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ dim_to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:dim_to_string

(** An index into a single axis for doing computations over multiple [Shape]-derived [Code]s. *)
type axis_index =
  | Fixed_idx of int  (** The specific position along an axis. *)
  | Iterator of symbol
      (** The given member of the [product_space] corresponding to some [product_iterators]. *)
[@@deriving compare, equal, sexp, variants]

type str_osym_map = (string, symbol option, Base.String.comparator_witness) Base.Map.t

let sexp_of_str_osym_map (map : str_osym_map) =
  Sexp.List (Map.to_alist map |> List.map ~f:[%sexp_of: string * symbol option])

type projections = {
  product_space : dim array;
      (** The product space dimensions that an operation should parallelize (map-reduce) over. *)
  lhs_dims : dim array;
      (** The dimensions of the LHS tensor. *)
  product_iterators : symbol array;
      (** The product space iterators (concatentation of the relevant batch, output, input axes)
      for iterating over the [product_space] axes, where same axes are at same array indices. *)
  project_lhs : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the result of
      an operation. *)
  project_rhs1 : axis_index array;
      (** A projection that takes an [product_space]-bound index and produces an index into the (first)
      argument of an operation. *)
  project_rhs2 : axis_index array option;
      (** A projection that takes an [product_space]-bound index and produces an index into the second
      argument of a binary operation. *)
}
[@@deriving compare, equal, sexp]
(** All the information relevant for [Code] code generation contained in a completed [update_step]. *)

let task_id_symbols = Hash_set.create (module Symbol)
let sample_num_symbols = Hash_set.create (module Symbol)

let get_sym_for_axis = function
  | Dim -> get_symbol ()
  | Dedicated Task_id ->
      let uid = get_symbol () in
      Hash_set.add task_id_symbols uid;
      uid
  | Dedicated Sample_num ->
      let uid = get_symbol () in
      Hash_set.add sample_num_symbols uid;
      uid
  | Frozen -> get_symbol ()

let task_id_sym = Hash_set.mem task_id_symbols
let sample_num_sym = Hash_set.mem sample_num_symbols
(* let iterate_sample_num = ref true *)

let fresh_symbol sym =
  if task_id_sym sym then get_sym_for_axis (Dedicated Task_id)
  else if sample_num_sym sym then get_sym_for_axis (Dedicated Sample_num)
  else get_symbol ()

let is_dedicated_any sym = task_id_sym sym || sample_num_sym sym
let is_dedicated_kind = function Task_id -> task_id_sym | Sample_num -> sample_num_sym

let iterated = function
  | { special = Dim; dim } when dim > 1 -> true
  (* | { special = Dedicated Sample_num; dim } when !iterate_sample_num && dim > 1 -> true *)
  | { special = Dedicated _; _ } -> true
  | _ -> false

let opt_symbol d = Option.some_if (iterated d) @@ get_sym_for_axis d.special

let opt_iterator = function None -> Fixed_idx 0 | Some sym -> Iterator sym

(** Projections for iterating over a terminal, or for a pointwise unary operator. *)
let identity_projections ~lhs_dims =
  let product_iterators = Array.map lhs_dims ~f:opt_symbol in
  let project_lhs = Array.map product_iterators ~f:opt_iterator in
  let product_space = Array.filter ~f:iterated lhs_dims in
  let product_iterators = Array.filter_map ~f:Fn.id product_iterators in
  { product_space; lhs_dims; product_iterators; project_lhs; project_rhs1 = project_lhs; project_rhs2 = None }

let derive_index ~product_syms ~(projection : axis_index array) =
  let sym_to_i =
    Array.mapi product_syms ~f:(fun i (Symbol s) -> (s, i)) |> Array.to_list |> Map.of_alist_exn (module Int)
  in
  let positions =
    Array.map projection ~f:(function
      | Iterator (Symbol s) when Map.mem sym_to_i s -> Either.First (Map.find_exn sym_to_i s)
      | Fixed_idx _ as it -> Second it
      | Iterator s as it ->
          assert (is_dedicated_any s);
          Second it)
  in
  fun ~product -> Array.map positions ~f:(function First p -> product.(p) | Second it -> it)

