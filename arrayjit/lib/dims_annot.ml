open Base

type dims_annot = { batch : int list; input : int list; output : int list }
type node = dims_annot Ndarray.t

let pp_tensor_inline fmt ?labels_spec n =
  let a = n.Ndarray.annot in
  let num_batch_axes = List.length a.batch in
  let num_output_axes = List.length a.output in
  let num_input_axes = List.length a.input in
  Ndarray.pp_tensor_inline fmt ~num_batch_axes ~num_output_axes ~num_input_axes ?axes_spec n.array

let default_display_indices ~num_batch_axes ~num_output_axes ~num_input_axes ~dims =
  let axes = Array.create ~len:(Array.length dims) 0 in
  let occupied = Array.create ~len:5 false in
  let set_occu prio =
    occupied.(prio + 5) <- true;
    prio
  in
  let occu prio = occupied.(prio + 5) in
  let to_axis_pos ~in_axes ~from_end =
    match in_axes with
    | `Input -> num_batch_axes + num_output_axes + num_input_axes - from_end
    | `Output -> num_batch_axes + num_output_axes - from_end
    | `Batch -> num_batch_axes - from_end
  in
  let remaining =
    Stack.of_list
    @@ List.filter ~f:(( > ) @@ Array.length dims)
    @@ [
         to_axis_pos ~in_axes:`Input ~from_end:1;
         to_axis_pos ~in_axes:`Output ~from_end:1;
         to_axis_pos ~in_axes:`Input ~from_end:2;
         to_axis_pos ~in_axes:`Output ~from_end:2;
         (if num_input_axes > 1 then to_axis_pos ~in_axes:`Batch ~from_end:1
          else to_axis_pos ~in_axes:`Output ~from_end:3);
         to_axis_pos ~in_axes:`Batch ~from_end:1;
         to_axis_pos ~in_axes:`Batch ~from_end:2;
         to_axis_pos ~in_axes:`Input ~from_end:3;
         to_axis_pos ~in_axes:`Output ~from_end:3;
         to_axis_pos ~in_axes:`Input ~from_end:4;
         to_axis_pos ~in_axes:`Output ~from_end:4;
         to_axis_pos ~in_axes:`Input ~from_end:5;
         to_axis_pos ~in_axes:`Output ~from_end:5;
       ]
  in
  let rec loop offset =
    if Stack.is_empty remaining || offset > 5 then axes
    else if Fn.non occu ~-offset then (
      axes.(Stack.pop_exn remaining) <- set_occu ~-offset;
      loop (offset + 1))
    else loop (offset + 1)
  in
  loop 1

let default_value_prec = ref Ndarray.single
let default_grad_prec = ref Ndarray.single
(*
   let create ?(batch=[]) ?(input=[]) ?(output=[]) values =
*)
