(** The vector register file the C backends' explicit-SIMD renderings target: its register width and
    its register count, as ONE value (gh-ocannl-1399).

    One value because neither means anything without the other, and a default for either is wrong:
    with two fields, a limits record or a config module that states the width can inherit a count of
    [0] from its base, and {!Register_tile.rn_cap} of 0 registers is 0 -- the seeding proposes no
    tensorized candidate and the renderer declines every one, with nothing failing to say so. A
    record literal of this type names both. *)

open Base

type t = {
  bytes : int;
      (** The register width in bytes: [lanes = bytes / element size] for the [Vectorized]
          renderings and the register-tiled [Tile_mma] micro-kernel. [0] where the backend does no
          explicit-SIMD rendering. *)
  registers : int;
      (** How many registers of [bytes] width the file has: 32 on AArch64 and on AVX-512 targets, 16
          on other x86 targets (SSE, AVX2), fewer where the width is wider than a native register
          (the compiler splits such a vector across several). {!Register_tile.rn_cap} fits the
          register tile's columns to it. [0] where [bytes] is [0], and where the backend renders no
          register tile (the GPU backends' packed-struct vectors, which only move data). *)
}
[@@deriving sexp, compare, equal]

val none : t
(** [{ bytes = 0; registers = 0 }]: no explicit-SIMD rendering. *)
