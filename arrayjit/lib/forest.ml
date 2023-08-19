(** Reconstitute the computation expression forest from a high-level computation, for visualization. *)
open Base

let rec last_array = function
  | High_level.Noop -> None
  | Seq (c1, c2) ->
      Option.value_or_thunk ~default:(fun () -> last_array c1) @@ Option.(map ~f:some @@ last_array c2)
  | Block_comment (_, c) | Comment_reference c -> last_array c
  | Accum_binop { lhs; _ } | Accum_unop { lhs; _ } | Fetch { array = lhs; _ } -> Some lhs
