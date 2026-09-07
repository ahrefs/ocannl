(** The C-tile geometry of the register-tiled [Tile_mma] rendering (gh-ocannl-619).

    The C backends render a whole-K [Tile_mma] tinyBLAS-style: an [rm x rn] grid of [lanes]-wide
    vector registers holds the accumulator across the entire k-loop — per k step, [rn] B-row vector
    loads, [rm] A-element splats and [rm * rn] fused FMAs — and the columns the width [rn * lanes]
    does not cover are peeled to scalar code. Until gh-ocannl-619 that geometry was picked inside
    the renderer alone; it is now a value a schedule can carry ({!Schedule.optop.Tensorize}'s
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

val default : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t option
(** The renderer's own choice for an [m x n] site when the schedule carries no geometry:
    [rm = min 4 m], and [(lanes, rn)] ranked over the widths the file renders (the ladder, narrowed
    to those [n] can fill) and [rn <= rn_cap] by a ranking model — per unit of m*k, one vector FMA
    per lane-column of the full blocks plus the B loads (1/rm per FMA) and the A splats (1/rn), and
    a constant 10 lane-slots per peeled column. The peel weight does NOT scale with the lane count
    (the peel is the same scalar loop at either width) and the fits agree: ~8 from an 8-lane sweep,
    ~10 from a 4-lane one, ~20 from an n = 2048 pair (gh-ocannl-575). The model only has to RANK: it
    reproduces the measured order at n = 512 within a few percent across rn = 2..6, and where
    several widths divide [n] it lands on the largest affordable one. Erring low on the peel weight
    is what costs choices — weighting a peeled column at [lanes] rather than the fit picked the
    peeling rn = 6 over a peel-free rn = 4 at n = 2048, which measures 1.15x slower (Codex P2 on
    staging PR #357). Ties go to the wider vector, then the larger tile. [None] when even the
    narrowest width exceeds [n] (or [m], [n] < 1). *)

val check : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t -> (unit, string) Result.t
(** Whether the renderer can honour [t] on an [m x n] site: [lanes] is a width the file renders and
    [n] fills, [rm <= m], [rn * lanes <= n], and {!live_registers} within {!budget}. The [Error]
    names the violated rule — it becomes the decline diagnostic. *)

val alternatives : vector_bytes:int -> elt_bytes:int -> m:int -> n:int -> t list
(** The geometries the sketch seeding proposes beside the renderer's {!default}, for the tuner to
    time: at the widest width [n] fills and the default's [rm], every peel-free [rn >= 2] (the width
    divides [n] exactly — a scalar column costs about a vector slot, so these are the candidates the
    model's peel weight decides between) plus the register-budget cap when its peel is at most one
    vector per row (the most A-reuse the file affords — the register-pressure corner gh-ocannl-614
    found the model cannot see — but not on sites where it would peel a fat remainder the model
    already prices with confidence), minus the default itself and anything {!check} would decline.
    Deliberately small: one alternative per site on the common shapes, none at all on many. *)
