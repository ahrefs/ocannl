(** The C-tile geometry of the register-tiled [Tile_mma] rendering (gh-ocannl-619).

    The C backends render a whole-K [Tile_mma] tinyBLAS-style: an [rm x rn] grid of [lanes]-wide
    vector registers holds the accumulator across the entire k-loop — per k step, [rn] B-row vector
    loads, [rm] A-element splats and [rm * rn] fused FMAs — and the rows and columns the full
    [rm x (rn * lanes)] passes do not cover are narrower register tiles of the same form
    (gh-ocannl-620; {!coverage}), never scalar code. Until gh-ocannl-619 that geometry was picked
    inside the renderer alone; it is now a value a schedule can carry ({!Schedule.optop.Tensorize}'s
    [tile]), the renderer honours or declines ([C_syntax.try_register_tile]), and the sketch seeding
    proposes alternatives of ({!alternatives}), so the tuner picks by timing. This module is a leaf
    — below [Low_level], which stores the geometry on its [Tile_mma] statement — so the ranking
    model and the fit rules live in one place both the emission and the seeding consult. *)

open Base

type t = { rm : int; rn : int; lanes : int }
(** [rm] accumulator rows, [rn] vector columns, [lanes] elements per vector: an [rm x rn] register
    grid covering [rm] rows by [rn * lanes] columns per pass. *)

val compare : t -> t -> int
val equal : t -> t -> bool
val sexp_of_t : t -> Sexp.t
val t_of_sexp : Sexp.t -> t

val to_string : t -> string
(** ["rm4 rn2 lanes8"], the display form shared by decline diagnostics and family-tree labels. *)

val width : t -> int
(** [rn * lanes], the columns one tile pass covers. *)

val live_registers : t -> int
(** [rm * rn + rm + rn]: the accumulator grid plus the per-k-step A splats and B rows. *)

val simd_lane_ladder : vector_bytes:int -> elt_bytes:int -> int list
(** The vector widths (in lanes) a [vector_bytes]-wide register file renders for [elt_bytes]-wide
    elements, widest first, halving down to the 32-byte floor (never below [vector_bytes] itself):
    64 bytes at f32 gives [16; 8], 32 bytes gives [8], 16 bytes gives [4]. Widths of fewer than two
    lanes are dropped. Re-exported by [Backend_intf.simd_lane_ladder], where the [Vectorized]
    renderings' width choices ([simd_lanes_for] and its accumulating variants) build on it. *)

val budget : vector_bytes:int -> int
(** The live-register budget a requested geometry must fit: the {!live_registers} of the widest
    default tile, i.e. tinyBLAS's [4 x 3] on 16-register (32-byte) files and [4 x 6] on 32-register
    files — 19 and 34. A geometry over it is not wrong, only likely to spill; it is declined rather
    than rendered so a candidate never times a spilling kernel under a label that promised a tile.
*)

val rn_cap : vector_bytes:int -> int
(** The default model's [rn] ceiling: 3 on 32-byte vector files, 6 otherwise. *)

val rm_cap : int
(** The [rm] ceiling — tinyBLAS's four accumulator rows, which {!default} takes wherever the row
    extent affords them. Exposed so that a geometry request naming only [rn] (bin/narrow_gebp_bench
    and bin/schedule_bench's [--rn=]) derives the rows the renderer would have chosen rather than
    restating the constant. *)

type coverage = { m_full : int; n_full : int; tail_widths : int list }
(** How a geometry covers an [m x n] site (gh-ocannl-620): [m_full] rows by [n_full] columns of full
    [rm x (rn * lanes)] passes; the [n - n_full] leftover columns as a column tail of [tail_widths]
    vector columns of [lanes] lanes each — every entry is [lanes] except a last, PARTIAL one whose
    entry is its valid lane count; and the [m - m_full] leftover rows as a band of that many rows
    over the same columns. The renderer emits exactly this decomposition (a partial column loads
    zeros past its width and stores only its width), so nothing is peeled to scalar code and every
    element's k-chain is the same fused serial chain. *)

val coverage : m:int -> n:int -> t -> coverage

val default : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t option
(** The renderer's own choice for an [m x n] site when the schedule carries no geometry:
    [rm = min 4 m], and [(lanes, rn)] ranked over the widths the file renders (the ladder, narrowed
    to those [n] can fill) and [rn <= rn_cap] by a ranking model in vector-issue slots per unit of
    m*k: one fused FMA per vector column plus the B loads (1/rm per FMA) and the A splats (1/rn) —
    for the full passes at [rn], and for the column tail at its own, smaller column count (the tail
    is a narrower tile, gh-ocannl-620; a partial last vector is a whole issue). No fitted constant:
    until gh-ocannl-620 the tail was a scalar peel priced at a measured 10 lane-slots per column
    (gh-ocannl-575), which made the width a divisibility question — now it is a reuse question, and
    the model only has to RANK. Ties go to the wider vector, then to the tail-free tile, then to the
    larger tile. [None] when even the narrowest width exceeds [n] (or [m], [n] < 1). *)

val check : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t -> (unit, string) Result.t
(** Whether the renderer can honour [t] on an [m x n] site: [lanes] is a width the file renders and
    [n] fills, [rm <= m], [rn * lanes <= n], and {!live_registers} within {!budget}. The [Error]
    names the violated rule — it becomes the decline diagnostic. *)

val alternatives : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t list
(** The geometries the sketch seeding proposes beside the renderer's {!default}, for the tuner to
    time: at the widest width [n] fills and the default's [rm], the largest tail-free [rn >= 2] (the
    width divides [n] exactly, so the site is one tile body: what a tail-bearing default's second,
    lower-reuse tile is traded against — the smaller tail-free widths are dominated on the model's
    own terms, equal issues at less reuse, so they are not seeded) plus the register-budget cap when
    its column tail is at most one vector (the most A-reuse the file affords — the register-pressure
    corner gh-ocannl-614 found the model cannot see — but not on sites where the tail is a fat
    second tile the model already prices), minus the default itself and anything {!check} would
    decline. Deliberately small: at most two alternatives per site, one on the common shapes, none
    where the width divides the extent. *)
