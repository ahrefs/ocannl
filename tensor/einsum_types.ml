(** Shared types for the einsum parser.

    These types are shared between the einsum parser and shape inference. *)

open Base

type use_padding_spec = [ `True | `False | `Unspecified ] [@@deriving compare, sexp]
(** Use_padding specification for convolutions. *)

type conv_spec = { dilation : string; kernel_label : string; use_padding : use_padding_spec }
[@@deriving compare, sexp]
(** Convolution component for affine axis specifications. Note: [dilation] is a string because it
    can be an identifier at parse time, and is resolved to an int at runtime. *)

(** Specification for individual axes in the einsum notation. Note: [stride] is a string because it
    can be an identifier at parse time, and is resolved to an int at runtime. *)
type axis_spec =
  | Label of string  (** A variable axis label. *)
  | Fixed_index of int  (** A fixed index, used for projection. *)
  | Affine_spec of {
      stride : string;  (** Coefficient for the over dimension (string to allow identifiers). *)
      over_label : string;  (** The output/iteration dimension label. *)
      conv : conv_spec option;  (** Optional convolution: dilation*kernel. *)
      stride_offset : int;  (** Constant offset added after stride*over. *)
    }
      (** Affine axis specification: stride*over + stride_offset [+ dilation*kernel]. Corresponds to
          [Row.Affine] in shape inference. *)
  | Concat_spec of string list
      (** Concatenation of multiple axis labels into a single axis. Corresponds to [Row.Concat] in
          shape inference. *)
[@@deriving compare, sexp]

(** An index pointing to any of a shape's axes, including the kind of the axis ([Batch, Input,
    Output]) and the position (which is counted from the end to facilitate broadcasting).

    Note the following inconsistency due to differing conventions in function notation and matrix
    notation: for label specifications and einsum notation, we write "batch|inputs->outputs", but
    when we convert a shape to an [Ndarray] index we do it in the order [[batch; outputs; inputs].
*)
module AxisKey = struct
  module T = struct
    type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash]

    type t = {
      in_axes : kind;
      pos : int;
          (** Indices start at [1] (note this is axis index, dimension indices are always 0-based),
              counted from the end if [from_end] is true. *)
      from_end : bool;
          (** Axes are indexed from the front (rarely) or from the end (typically), to avoid
              reindexing when broadcasting. *)
    }
    [@@deriving equal, compare, sexp]
  end

  include T
  include Comparator.Make (T)
end

type axis_key = AxisKey.t [@@deriving equal, compare, sexp]
type 'a axis_map = 'a Map.M(AxisKey).t [@@deriving compare, sexp]

type parsed_axis_labels = {
  bcast_batch : string option;
  bcast_input : string option;
  bcast_output : string option;
  given_batch : axis_spec list;
  given_input : axis_spec list;
  given_output : axis_spec list;
  given_beg_batch : axis_spec list;
  given_beg_input : axis_spec list;
  given_beg_output : axis_spec list;
  labels : axis_spec axis_map;
}
[@@deriving compare, sexp, fields]
(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent
    whether additional leading/middle axes are allowed (corresponding to the dot-ellipsis syntax for
    broadcasting). The string can be used to identify a row variable, and defaults to ["batch"],
    ["input"], ["output"] respectively when parsing ["..."]. The [given_] fields are lists of axis
    specs of the corresponding kind in [labels] where [from_end=true], [given_beg_] where
    [from_end=false]. *)

let axis_labels parsed = parsed.labels
