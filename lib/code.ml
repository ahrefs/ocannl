(** The code for operating on n-dimensional arrays. *)
open Base

type precision =
  | Half
  | Single
  | Double

type data = Ocannl_runtime.Node.data

type binop =
  | Skip_arg
  | Add
  | Mul
  | Relu_gate
  | Uniform

type unop =
  | Identity
  | Relu

type t =
  | Par of t * t
  | Seq of t * t
  | Accum_binop of {
      zero_out: bool;
      accum: binop; op: binop;
      lhs: data option; rhs1: data option; rhs2: data option;
      projections: unit -> Shape.projections;
      precision: precision }
  | Accum_unop of {
      zero_out: bool;
      accum: binop; op: unop;
      lhs: data option; rhs: data option;
      projections: unit -> Shape.projections;
      precision: precision }
  | Create of {
        tensor: data; precision: precision; dims: unit -> int array;
        init_values: float array;
        (** [init_values] can be empty -- no initialization, single number -- initialize the whole tensor,
            the length of the tensor -- initialize from numbers where the rightmost axis is contiguous. *)
      }
  | Reset of {
        tensor: data; dims: unit -> int array;
        reset_values: float array;
        (** [reset_values] can be empty -- no initialization, single number -- initialize the whole tensor,
            the length of the tensor -- initialize from numbers where the rightmost axis is contiguous. *)
      }
  | Noop