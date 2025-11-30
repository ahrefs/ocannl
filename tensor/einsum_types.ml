(** Shared types for the einsum parser.

    These types are shared between the einsum parser and shape inference. *)

open Base

(** Specification for individual axes in the einsum notation. *)
type axis_spec =
  | Label of string  (** A variable axis label. *)
  | Fixed_index of int  (** A fixed index, used for projection. *)
  | Conv_spec of { stride : int; output_label : string; dilation : int; kernel_label : string }
      (** Convolution-style axis specification: stride*output + dilation*kernel. *)
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
  given_batch : int;
  given_input : int;
  given_output : int;
  given_beg_batch : int;
  given_beg_input : int;
  given_beg_output : int;
  labels : axis_spec axis_map;
}
[@@deriving compare, sexp, fields]
(** The labels are strings assigned to [AxisKey] axes. Moreover the [bcast_] fields represent
    whether additional leading/middle axes are allowed (corresponding to the dot-ellipsis syntax for
    broadcasting). The string can be used to identify a row variable, and defaults to ["batch"],
    ["input"], ["output"] respectively when parsing ["..."]. The [given_] fields count the number of
    specified axes of the corresponding kind in [labels] where [from_end=true], [given_beg_] where
    [from_end=false]. *)

let axis_labels parsed = parsed.labels
