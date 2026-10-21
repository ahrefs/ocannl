(** Entry point for the einsum parser library.

    This module provides functions to parse einsum notation specifications using a Menhir-based
    parser. *)

open Base

(* Re-export types from Einsum_types *)
include module type of Einsum_types

exception Parse_error of string
(** Exception raised when parsing fails. *)

val binary_operators_with_generated_specs : string list
val unary_operators_with_generated_specs : string list
val concat_operator_with_generated_specs : string

val operators_with_generated_specs : string list
(** The unqualified DSL operators whose literal specifications the OCANNL PPXs pass through
    identifier substitution. The PPX dispatch tables and source-policy scans share these values so
    adding a syntax form cannot silently update only one consumer. *)

val is_multichar : string -> bool
(** Determine if a spec uses multichar mode. Multichar mode is triggered by presence of: ',', '*',
    '+', '^', '&', '<', or a '=' that does not begin "=>". In multichar mode axes are separated by
    commas; whitespace does not separate them. *)

val axis_labels_of_spec : string -> parsed_axis_labels
(** Parse an axis labels specification.

    Examples:
    - "abc" (single-char mode)
    - "a, b, c" (multichar mode, triggered by comma)
    - "batch|input->output"
    - "...a..b" *)

val einsum_of_spec : string -> parsed_axis_labels list * parsed_axis_labels
(** Parse an einsum specification. Returns a list of RHS specs and the LHS (result) spec.

    Examples:
    - "ij;jk=>ik" (matrix multiplication, 2 RHSes)
    - "ij=>ji" (transpose/permute, 1 RHS)
    - "a;b;c=>d" (3 RHSes for block tensors/concatenation)
    - "i,j=>2*i+j" (convolution) *)
