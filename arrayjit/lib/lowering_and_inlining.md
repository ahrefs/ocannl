# Compilation to Cross-Backend Low-Level Representation, and Backend-Independent Optimizations

Computation in OCANNL is imperative. At the high-level, we store tensor node assignments as `Assignments.t`:

```ocaml
(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Access of Low_level.dedicated_access
  | Slice of { batch_idx : Indexing.static_symbol; sliced : Tnode.t }
  | Embed_symbol of Indexing.static_symbol

and t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
  | Accum_binop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.binop;
      lhs : Tnode.t;
      rhs1 : Tnode.t;
      rhs2 : Tnode.t;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_unop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.unop;
      lhs : Tnode.t;
      rhs : Tnode.t;
      projections : Indexing.projections Lazy.t;
    }
  | Fetch of { array : Tnode.t; fetch_op : fetch_op; dims : int array Lazy.t }
```

The effect of `Accum_binop { initialize_neutral; accum; op; lhs; rhs1; rhs2; projections }` is:

> if `initialize_neutral` then `lhs` := neutral value of `accum`;  
> `lhs` := `lhs` `accum` (`rhs1` `op` `rhs2`)

The `Assignments` module depends on the `Low_level` module and puts the pieces together in the `compile_proc` function. In addition to the assignments, `compile_proc` takes a `Indexing.static_symbol list` of the static indices, currently they are needed for optimization but not remembered in the `Assignments.t` nor `Low_level.t` types.

The low-level representation is a C-like mini-language operating on scalars.

```ocaml
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tnode.t
  | Set of { array : Tnode.t; idcs : Indexing.axis_index array; llv : float_t; mutable debug : string }
  | Set_local of scope_id * float_t

and float_t =
  | Local_scope of {
      id : scope_id;
      prec : Ops.prec;
      body : t;
      orig_indices : Indexing.axis_index array;
    }
  | Get_local of scope_id
  | Access of Low_level.dedicated_access * Indexing.axis_index array option
  | Get of Tnode.t * Indexing.axis_index array
  | Binop of Ops.binop * float_t * float_t
  | Unop of Ops.unop * float_t
  | Constant of float
  | Embed_index of Indexing.axis_index
```

The odd part is the `Staged_compilation` element. Backends can use `Staged_compilation` to embed some emitted code within on-the-fly generated `Low_level.t` code. Currently this works only for `PPrint.document` based backends like `C_syntax` derivatives, but this covers almost all backends.

TODO: flesh out explanation.

## Translation

The translation `Assignments.to_low_level` is straightforward. Commented code blocks are delineated by `Low_level.Comment "end"` statements. Indices into tensor nodes are derived from the `projections` fields by the `Indexing.derive_index` function. We translate `projections.product_space` elements into for loops. `to_low_level` returns all the data that `Low_level` optimizations generated, so that backends can make more informed decisions when jitting, i.e. emitting the backend-specific code.

## Inlining

Inlining is a process where we take the computations pertaining to a tensor node, and inline them at the `Get` access sites on a per-scalar basis.

```ocaml
type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_tracing_dim : int;
}

type visits =
  | Visits of int
  | Recurrent  (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)

type traced_array = {
  nd : Tn.t;
  mutable computations : (Indexing.axis_index array option * t) list;
      (** The computations (of the tensor node) are retrieved for optimization just as they are populated,
          so that the inlined code corresponds precisely to the changes to the arrays that would happen
          up till that point. Within the code blocks paired with an index tuple, all assignments and accesses
          must happen via the index tuple; if this is not the case for some assignment, the node cannot
          be virtual. Currently, we only allow for-loop symbols in assignment indices of virtual nodes. *)
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
      (** For dynamic indexes, we take a value of 0. This leads to an overestimate of visits, which is safe. *)
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;  (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
}
```

- The `visit_llc` function interprets (symbolically executes) the given computation, and fills in `traced_store`: the map of `traced_array`s.
- 


## Rewriting