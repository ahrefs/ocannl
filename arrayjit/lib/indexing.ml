open Base

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_INDEXING=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_INDEXING"]

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

type static_symbol = {
  static_symbol : symbol;
  mutable static_range : int option; [@compare.ignore] [@equal.ignore] [@hash.ignore]
  mutable used_as_extent : bool; [@compare.ignore] [@equal.ignore] [@hash.ignore] [@sexp.bool]
      (** The symbol is used as a symbolic dimension (gh-490): its bound value is a size in
          [[0, static_range]] (inclusive -- buffers are sized [static_range]), rather than an index
          in [\[0, static_range)]. Set by [Row.get_sym_dim]. *)
  mutable used_as_slice : bool; [@compare.ignore] [@equal.ignore] [@hash.ignore] [@sexp.bool]
      (** The symbol is used as a batch-slice index ([@|]): its bound value indexes an axis of size
          [static_range], so the strict [\[0, static_range)] validation must apply. Set by the
          [Batch_slice] shape logic; mutually exclusive with [used_as_extent] (a single binding
          cannot be validated both inclusively and strictly). *)
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
  let s =
    { static_symbol = get_symbol (); static_range; used_as_extent = false; used_as_slice = false }
  in
  (s, Bind (s, bindings))

(** Validates a launch-parameter value against its declared range (minimum 0 is implicit) and, when
    [width64] is false, against the int32 index width (docs/proposals/signed-index-precision.md:
    bind-time validation mirroring tinygrad's [bind] assert; one host compare per launch). The
    per-routine node-extent contract is NOT sound for launch parameters -- value-embedded parameters
    (step counters) take runtime values unrelated to any touched node's extent -- so narrowing an
    unbounded parameter requires declaring (and thereby bind-validating) a range. *)
let validate_bound_value ?(width64 = false)
    ({ static_symbol; static_range; used_as_extent; used_as_slice = _ } : static_symbol) (v : int) =
  let ident = symbol_ident static_symbol in
  if v < 0 then
    raise
    @@ Utils.User_error
         (Printf.sprintf "Indexing: bound value %d for static index %s is negative" v ident);
  (* The width check applies regardless of a declared range: a range wider than the emitted
     parameter width does not make an unrepresentable value representable. *)
  if (not width64) && v > 2147483647 then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Indexing: bound value %d for static index %s exceeds the int32 index width; enable \
             large_models for 64-bit indices"
            v ident);
  match static_range with
  | Some range when used_as_extent && v > range ->
      (* An extent (gh-490 symbolic dimension) is a size: the full range is a legal value. *)
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Indexing: bound value %d for symbolic extent %s exceeds its declared maximum %d" v
              ident range)
  | Some range when (not used_as_extent) && v >= range ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Indexing: bound value %d for static index %s exceeds its declared range [0, %d)" v
              ident range)
  | Some _ | None -> ()

let validate_lowered_bindings ?width64 (bs : lowered_bindings) =
  List.iter bs ~f:(fun (s, r) -> validate_bound_value ?width64 s !r)

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x "
    @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ Int.to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string

(** Coalesce an affine [symbols] list into canonical [(coeff, symbol)] terms: repeated symbols are
    summed and zero-coefficient terms dropped. Order is unspecified. *)
let coalesce_affine_terms symbols =
  List.fold symbols
    ~init:(Map.empty (module Symbol))
    ~f:(fun acc (coeff, s) ->
      Map.update acc s ~f:(function None -> coeff | Some c0 -> c0 + coeff))
  |> Map.to_alist
  |> List.filter_map ~f:(fun (s, c) -> if c = 0 then None else Some (c, s))

type affine_index = { symbols : (int * symbol) list; offset : int }
[@@deriving compare, equal, sexp]

type axis_index =
  | Fixed_idx of int  (** A fixed position along an axis *)
  | Iterator of symbol  (** A simple iterator symbol *)
  | Affine of affine_index
      (** An affine combination of symbols with coefficients and an offset. Represents:

          Σ(coeff_i * symbol_i) + offset

          For convolutions: [symbols = [(stride, i1); (dilation, i2)]] and [offset = ~-padding].
          Note: for readability, we use [Fixed_idx] and [Iterator] as separate variants and require
          [Affine] to not be ambiguous: [symbols] should be longer than 1 or have a coefficient
          different from 1 and 0, or a nonzero offset. Construction goes through [affine]. *)
  | Sub_axis  (** This axis belongs to an adjacent multi-axis index. *)
  | Concat of symbol list
      (** This axis is formed by concatenating multiple axes, each represented by an iterator
          symbol. [Concat] indices are eliminated during lowering. *)
[@@deriving compare, equal, sexp]

(** The only construction boundary for affine indices. *)
let affine ~symbols ~offset =
  match coalesce_affine_terms symbols with
  | [] -> Fixed_idx offset
  | [ (1, s) ] when offset = 0 -> Iterator s
  | symbols -> Affine { symbols; offset }

(* Deserialization observes the same invariant as programmatic construction. *)
let sexp_of_axis_index idx =
  match sexp_of_axis_index idx with
  | Sexp.List [ Sexp.Atom "Affine"; Sexp.List fields ] -> Sexp.List (Sexp.Atom "Affine" :: fields)
  | sexp -> sexp

let axis_index_of_sexp sexp =
  (* Preserve the former inline-record wire format when sealing the payload. *)
  let sexp =
    match sexp with
    | Sexp.List (Sexp.Atom "Affine" :: fields) -> Sexp.List [ Sexp.Atom "Affine"; Sexp.List fields ]
    | sexp -> sexp
  in
  match axis_index_of_sexp sexp with
  | Affine { symbols; offset } -> affine ~symbols ~offset
  | idx -> idx

(** Whether the value of [idx] depends on [s]: [s] is the axis's own iterator, appears as a term of
    its affine combination, or is one of the iterators a [Concat] axis is formed from. A
    [Fixed_idx]ed or [Sub_axis] axis mentions nothing. *)
let axis_index_mentions_symbol (s : symbol) (idx : axis_index) : bool =
  match idx with
  | Iterator s' -> equal_symbol s s'
  | Affine { symbols; _ } -> List.exists symbols ~f:(fun (_, s') -> equal_symbol s s')
  | Concat syms -> List.exists syms ~f:(equal_symbol s)
  | Fixed_idx _ | Sub_axis -> false

(** {!axis_index_mentions_symbol} over a set of symbols: whether [idx] depends on any of [syms]. *)
let axis_index_mentions_any (syms : symbol list) (idx : axis_index) : bool =
  List.exists syms ~f:(fun s -> axis_index_mentions_symbol s idx)

type str_osym_map = (string, symbol option, Base.String.comparator_witness) Base.Map.t

type projections_debug = { spec : string; derived_for : Sexp.t; trace : (string * int) list }
[@@deriving sexp]

let unique_debug_id =
  let projections_uid = ref 0 in
  fun () ->
    Int.incr projections_uid;
    !projections_uid

type component = (int * symbol) list [@@deriving compare, equal, sexp]
(** One component of the product space: its [(dimension, iterator)] segments. Singleton except for
    concatenation components, which are a connected component of symbols participating in
    concatenated axes -- their segments are iterated SEQUENTIALLY, one loop each, in list order.
    Pairing the dimension with its iterator makes the same-index invariant structural: there is no
    way to hold a dimension whose iterator is elsewhere (gh-ocannl-775). *)

type projections = {
  components : component array;
      (** The product space that an operation should parallelize (map-reduce) over: one component
          per iterated axis (concatenation of the relevant batch, output, input axes), each carrying
          its segments' dimensions and iterator symbols. Iterators may be shared between operations;
          lowering creates fresh symbols for loop indices. *)
  lhs_dims : int array;  (** The dimensions of the LHS array. *)
  rhs_dims : int array array;
      (** The dimensions of the RHS arrays, needed for deriving projections from other projections.
      *)
  project_lhs : axis_index array;
      (** A projection that takes a [components]-bound index and produces an index into the result
          of an operation. *)
  project_rhs : axis_index array array;
      (** [project_rhs.(i)] Produces an index into the [i+1]th argument of an operation. *)
  extent_syms : (symbol option * static_symbol) list;
      (** gh-490 symbolic extents: associates the product iterator, when there is one, with the
          static symbol whose bound value is the axis's runtime extent. The corresponding segment's
          dimension is the maximum extent (the symbol's declared range); lowering guards the loop
          body with [iterator < value] when the symbol is among the routine's bindings. A
          maximum-one axis has no iterator and is retained as [None], so consumers that cannot honor
          dynamic extents can still reject it explicitly (gh-ocannl-817). *)
  debug_info : (projections_debug[@sexp.ignore] [@compare.ignore] [@equal.ignore]);
}
[@@deriving compare, equal, sexp]
(** All the information relevant for code generation. *)

let iterated dim = dim > 1

(** Every product iterator of [p], across all components and all their segments. *)
let all_iterators (p : projections) : symbol list =
  Array.to_list p.components |> List.concat_map ~f:(List.map ~f:snd)

(** The per-segment dimension of every product iterator of [p]. *)
let iterator_sizes (p : projections) : int Map.M(Symbol).t =
  Array.fold p.components
    ~init:(Map.empty (module Symbol))
    ~f:(fun acc comp ->
      List.fold comp ~init:acc ~f:(fun acc (d, iter) -> Map.set acc ~key:iter ~data:d))

(* gh-133 Stage B: injectivity of an affine LHS map over its non-static symbols.

   [symbol_range s] is the loop width of [s] (>= 1); a width of 1 marks a static / range-1 symbol,
   which is pinned from the start and ignored for injectivity (it contributes a single value).

   A single position [Σ cᵢ·sᵢ + offset] is injective on its remaining (non-pinned, non-static)
   symbols when, sorting them by ascending [abs coeff], every term k >= 2 satisfies the mixed-radix
   criterion [abs c_k >= 1 + Σ_{i<k} abs c_i · (range_i − 1)]. The offset never affects injectivity.
   This is sufficient, not necessary (e.g. [3·a + 4·b] over ranges [(3, 2)] is rejected though
   distinct), matching the proposal's known-incomplete case.

   The whole-LHS criterion is a pinning fixpoint: pin static/range-1 symbols, then repeatedly pin
   the residual symbols of any position that passes the per-position criterion once already-pinned
   terms are removed. The map is injective iff every non-static symbol is eventually pinned. This
   accepts triangular maps such as [(s1, s1 + s2)] (position 1 pins s1, then position 2 pins s2).

   [Concat] axes partition their range across disjoint sub-ranges, so each concat symbol is pinned
   directly (preserving the pre-Stage-B treatment); [Fixed_idx]/[Sub_axis] carry no symbol. *)
let affine_injective ~symbol_range (project_lhs : axis_index array) : bool =
  let dyn s = symbol_range s > 1 in
  (* Every non-static symbol that must be pinned for the map to be injective. *)
  let add_dyn acc ss =
    List.fold ss ~init:acc ~f:(fun acc s -> if dyn s then Set.add acc s else acc)
  in
  let all_syms =
    Array.fold project_lhs
      ~init:(Set.empty (module Symbol))
      ~f:(fun acc idx ->
        match idx with
        | Iterator s -> add_dyn acc [ s ]
        | Affine { symbols; _ } -> add_dyn acc (List.map (coalesce_affine_terms symbols) ~f:snd)
        | Concat syms -> add_dyn acc syms
        | Fixed_idx _ | Sub_axis -> acc)
  in
  (* Symbols newly pinnable by [idx] given the currently [pinned] set; [None] if the position is not
     (yet) injective on its residual symbols. *)
  let position_pins pinned idx =
    match idx with
    | Fixed_idx _ | Sub_axis -> Some []
    | Iterator s -> Some (if dyn s then [ s ] else [])
    | Concat syms -> Some (List.filter syms ~f:dyn)
    | Affine { symbols; _ } ->
        let residual =
          coalesce_affine_terms symbols
          |> List.filter ~f:(fun (_, s) -> dyn s && not (Set.mem pinned s))
        in
        let sorted =
          List.sort residual ~compare:(fun (c1, _) (c2, _) -> Int.compare (abs c1) (abs c2))
        in
        let rec check acc = function
          | [] -> true
          | (c, s) :: rest ->
              if abs c >= 1 + acc then check (acc + (abs c * (symbol_range s - 1))) rest else false
        in
        if check 0 sorted then Some (List.map sorted ~f:snd) else None
  in
  let rec fixpoint pinned =
    let pinned' =
      Array.fold project_lhs ~init:pinned ~f:(fun pinned idx ->
          match position_pins pinned idx with
          | Some syms -> List.fold syms ~init:pinned ~f:Set.add
          | None -> pinned)
    in
    if Set.length pinned' > Set.length pinned then fixpoint pinned' else pinned'
  in
  Set.equal (fixpoint (Set.empty (module Symbol))) all_syms

(** The identity projection of a tensor laid out over the full product space of [p] (e.g. the
    (output x kernel) pair space of a windowed reduction): the tensor's axes are the product
    components, possibly interleaved with non-iterated (dimension-1) axes, which the product space
    omits. [dims] is the tensor's dimensions; each dimension-1 axis maps to [Fixed_idx 0] and every
    other axis consumes the first not-yet-consumed product component of equal extent. Matching by
    extent rather than strictly by position tolerates a proxy-shape layout whose axis order deviates
    from the product order (e.g. a reduced-over row variable placed in the result's output row while
    the result keeps input axes): any pairing is correct as long as it is consistent, and this is a
    pure function of [p] and [dims], so every access to the tensor uses the same pairing. A leftover
    on either side means the product-space proxy shape (gh-512) got out of sync with the derived
    projections, and raises rather than corrupting memory. *)
let prod_project_for (p : projections) ~(dims : int array) : axis_index array =
  let mismatch () =
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Indexing.prod_project_for: product-space tensor dimensions %s are inconsistent with \
             the operation's product space %s (proxy shape vs. projections mismatch)"
            (dims_to_string dims)
            (Sexp.to_string_hum ([%sexp_of: component array] p.components)))
  in
  (* One (extent, iterators) entry per product component, in [components] order. A concatenation
     component's extent is the sum of its segment extents; non-iterated (dimension-1) axes do not
     appear as components at all. *)
  let comps =
    ref
      (Array.to_list p.components
      |> List.map ~f:(fun segments ->
          (List.fold segments ~init:0 ~f:(fun acc (d, _) -> acc + d), List.map segments ~f:snd)))
  in
  let take_first_by_extent d =
    let rec go acc = function
      | [] -> mismatch ()
      | (cd, syms) :: rest when cd = d ->
          comps := List.rev_append acc rest;
          syms
      | c :: rest -> go (c :: acc) rest
    in
    go [] !comps
  in
  let project =
    Array.map dims ~f:(fun d ->
        if d = 1 then Fixed_idx 0
        else match take_first_by_extent d with [ s ] -> Iterator s | syms -> Concat syms)
  in
  if not (List.is_empty !comps) then mismatch ();
  project

(** The row-major flat offset of [projection] into an array of [dims], as one affine index.

    [Concat] is rejected rather than approximated. Its segments partition the axis into disjoint
    sub-ranges, so the axis's contribution is not a single stride times a single symbol and cannot
    be written as one term of an affine sum -- the arm this replaces gave every segment symbol the
    same stride, which is a different index map. The only caller is the [Range_over_offsets] fetch
    in {!Assignments}, whose indices come from [Low_level.loop_over_dims] and are therefore
    [Iterator]s and [Fixed_idx]es; [Concat] axes are in any case eliminated during lowering. *)
let reflect_projection ~(dims : int array) ~(projection : axis_index array) =
  Array.zip_exn dims projection
  |> Array.fold_right ~init:(1, [], 0) ~f:(fun (dim, idx) (stride, symbols, offset) ->
      match idx with
      | Fixed_idx fixed_offset -> (stride * dim, symbols, offset + (fixed_offset * stride))
      | Iterator sym -> (stride * dim, (stride, sym) :: symbols, offset)
      | Affine { symbols = affine_symbols; offset = affine_offset } ->
          let new_symbols =
            List.map affine_symbols ~f:(fun (coeff, sym) -> (coeff * stride, sym))
          in
          (stride * dim, new_symbols @ symbols, offset + (affine_offset * stride))
      | Sub_axis -> (stride * dim, symbols, offset)
      | Concat _ ->
          raise
          @@ Utils.User_error
               "Indexing.reflect_projection: a concatenated axis has no single affine offset; \
                Concat indices are eliminated during lowering and cannot reach this function")
  |> fun (_, symbols, offset) -> affine ~symbols ~offset

type variable_ref = {
  ref_label : string;
  mutable solved_dim : int option;
  mutable solved_sym : static_symbol option;
      (** The dimension is a symbolic extent (gh-490) rather than a concrete integer; where a
          concrete value is required (e.g. [Embed_dim]), it materializes at the symbol's declared
          range (the maximum extent). At most one of [solved_dim], [solved_sym] is set. *)
}
[@@deriving sexp_of, equal]

module Doc_helpers = struct
  let ( ^^ ) = PPrint.( ^^ )
  let ( !^ ) = PPrint.( !^ )
  let int = PPrint.OCaml.int
  let comma_sep = PPrint.(comma ^^ space)
  let pp_comma () = comma_sep
  let pp_symbol sym = PPrint.string @@ symbol_ident sym

  let pp_static_symbol { static_symbol; static_range; used_as_extent = _; used_as_slice = _ } =
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
    | Sub_axis -> PPrint.empty
    | Concat syms ->
        let open PPrint in
        separate (string " ^ ") (List.map syms ~f:pp_symbol)

  let pp_indices idcs =
    PPrint.separate (pp_comma ()) (Array.to_list idcs |> List.map ~f:pp_axis_index)
end
