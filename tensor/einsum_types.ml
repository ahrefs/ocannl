(** Shared types for the einsum parser. *)

open Base

(** Specification for individual axes in the einsum notation. *)
type axis_spec =
  | Label of string
  | Fixed_index of int
  | Conv_spec of { stride : int; output_label : string; dilation : int; kernel_label : string }
[@@deriving compare, sexp]

type axis_key_kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp]

type axis_key = {
  in_axes : axis_key_kind;
  pos : int;
  from_end : bool;
}
[@@deriving equal, compare, sexp]

module AxisKey = struct
  module T = struct
    type t = axis_key [@@deriving equal, compare, sexp]
  end
  include T
  include Comparator.Make (T)
end

type 'a axis_map = 'a Map.M(AxisKey).t

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
