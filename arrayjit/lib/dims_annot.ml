open Base

type dims_annot = { batch : int list; input : int list; output : int list }
type node = dims_annot Ndarray.t

let get_shape_string ?(style = `Axis_size) n =
  let n_outputs = List.length n.Ndarray.annot.output in
  let n_batch = List.length n.annot.batch in
  let dims_to_string kind_dims kind =
    String.concat ~sep:","
    @@ List.mapi kind_dims ~f:(fun i d ->
           let num =
             match kind with `Input -> n_batch + n_outputs + i | `Output -> n_batch + i | `Batch -> i
           in
           match style with
           | `Axis_size -> Int.to_string d
           | `Axis_number_and_size -> Int.to_string num ^ ":" ^ Int.to_string d)
  in
  let batch_dims = dims_to_string n.annot.batch `Batch in
  let input_dims = dims_to_string n.annot.input `Input in
  let output_dims = dims_to_string n.annot.output `Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

let pp_tensor_inline fmt n =
  let axes_spec =
    if List.exists ~f:(( = ) 1) n.Ndarray.annot.input then Some (get_shape_string n) else None
  in
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

let pp_tensor ?shape_style ?entries_per_axis fmt n =
  let dims = Ndarray.dims n.Ndarray.array in
  let prefix = get_shape_string ?style:shape_style n in
  let indices =
    default_display_indices ~num_batch_axes:(List.length n.annot.batch)
      ~num_output_axes:(List.length n.annot.output) ~num_input_axes:(List.length n.annot.input) ~dims
  in
  Ndarray.pp_tensor fmt ~prefix ?entries_per_axis ~indices n.array

let default_prec = ref Ndarray.single
(*
   let create ?(batch=[]) ?(input=[]) ?(output=[]) values =
*)
