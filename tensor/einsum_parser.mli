(** Entry point for the einsum parser library.

    This module provides functions to parse einsum notation specifications using a Menhir-based
    parser. *)

open Base

(* Re-export types from Einsum_types *)
include module type of Einsum_types

exception Parse_error of string
(** Exception raised when parsing fails. *)

val is_multichar : string -> bool
(** Determine if a spec uses multichar mode. Multichar mode is triggered by presence of: ',', '*',
    '+', '^', '&' *)

val axis_labels_of_spec : string -> parsed_axis_labels
(** Parse an axis labels specification.

    Examples:
    - "abc" (single-char mode)
    - "a, b, c" (multichar mode, triggered by comma)
    - "batch|input->output"
    - "...a..b" *)

val einsum_of_spec : string -> parsed_axis_labels * parsed_axis_labels option * parsed_axis_labels
(** Parse an einsum specification.

    Examples:
    - "ij;jk=>ik" (matrix multiplication)
    - "ij=>ji" (transpose/permute)
    - "i,j->2*i+j" (convolution) *)
