open Base
module LA = Low_level.Lazy_array

type dims_annot = { batch : int list; input : int list; output : int list }
type node = { shape : dims_annot; array : LA.t }

let get_shape_string ?(style = `Axis_size) n =
  let n_outputs = List.length n.output in
  let n_batch = List.length n.batch in
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
  let batch_dims = dims_to_string n.batch `Batch in
  let input_dims = dims_to_string n.input `Input in
  let output_dims = dims_to_string n.output `Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

let pp_array_inline fmt n =
  let axes_spec = if List.exists ~f:(( = ) 1) n.shape.input then Some (get_shape_string n.shape) else None in
  let num_batch_axes = List.length n.shape.batch in
  let num_output_axes = List.length n.shape.output in
  let num_input_axes = List.length n.shape.input in
  Ndarray.pp_array_inline fmt ~num_batch_axes ~num_output_axes ~num_input_axes ?axes_spec
  @@ LA.get_exn n.array

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

let pp_array ?shape_style ?entries_per_axis fmt n =
  let dims = Lazy.force n.array.dims in
  let prefix = get_shape_string ?style:shape_style n.shape in
  let indices =
    default_display_indices ~num_batch_axes:(List.length n.shape.batch) ~num_output_axes:(List.length n.shape.output)
      ~num_input_axes:(List.length n.shape.input) ~dims
  in
  Ndarray.pp_array fmt ~prefix ?entries_per_axis ~indices @@ LA.get_exn n.array

let default_prec = ref Ndarray.single
(*
   let create ?(batch=[]) ?(input=[]) ?(output=[]) values =
*)
