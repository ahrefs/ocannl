open Base
(** `Node`: the computation type, global state and utils which the `Tensor` staged code uses. *)

module A = Bigarray.Genarray

type 'a bigarray = (float, 'a, Bigarray.c_layout) A.t

let sexp_of_bigarray (arr : 'a bigarray) =
  let dims = A.dims arr in
  Sexp.Atom ("bigarray_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

(* FIXME: Upcoming in OCaml 5.2.0. See:
   https://github.com/ocaml/ocaml/pull/10775/commits/ba6a2c378056c8669fb1bb99bf07b12d69bd4a12 *)
type float16_elt = Bigarray.float32_elt
type float32_elt = Bigarray.float32_elt

let float16 : (float, float16_elt) Bigarray.kind = Bigarray.float32

type half_nd = float16_elt bigarray
type single_nd = Bigarray.float32_elt bigarray
type double_nd = Bigarray.float64_elt bigarray

type 'a precision = Half : half_nd precision | Single : single_nd precision | Double : double_nd precision
[@@deriving sexp_of]

type prec =
  | Void_prec : prec
  | Half_prec : half_nd precision -> prec
  | Single_prec : single_nd precision -> prec
  | Double_prec : double_nd precision -> prec

let half = Half_prec Half
let single = Single_prec Single
let double = Double_prec Double

let sexp_of_prec = function
  | Void_prec -> Sexp.Atom "Void_prec"
  | Half_prec _ -> Sexp.Atom "Half_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Void_prec" -> Void_prec
  | Sexp.Atom "Half_prec" -> half
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

type ndarray =
  | Half_nd of (half_nd[@sexp.opaque])
  | Single_nd of (single_nd[@sexp.opaque])
  | Double_nd of (double_nd[@sexp.opaque])
[@@deriving sexp_of]

type 'a t = {
  array : (ndarray[@compare.ignore] [@equal.ignore]);
  id : int;
  annot : ('a[@compare.ignore] [@equal.ignore]);
}
[@@deriving sexp_of, compare, equal]

type ptr = Ptr of int [@@deriving sexp, compare, equal, hash]

let ptr { id; _ } = Ptr id
let get_name { id; _ } = "#" ^ Int.to_string id

let as_array (type arr_t) (prec : arr_t precision) (arr : arr_t) =
  match prec with Half -> Half_nd arr | Single -> Single_nd arr | Double -> Double_nd arr

module ComparePtr = struct
  type t = ptr = Ptr of int [@@deriving sexp, compare, equal, hash]
end

module Ptr = struct
  include ComparePtr
  include Comparator.Make (ComparePtr)
end

let precision_to_bigarray_kind (type elt_t) (prec : elt_t bigarray precision) : (float, elt_t) Bigarray.kind =
  match prec with Half -> float16 | Single -> Bigarray.Float32 | Double -> Bigarray.Float64

let precision_to_string (type arr_t) (prec : arr_t precision) =
  match prec with Half -> "half" | Single -> "single" | Double -> "double"

let precision_string = function Half_nd _ -> "half" | Single_nd _ -> "single" | Double_nd _ -> "double"
let default_kind = Single
let is_double_prec_t = function Double_nd _ -> true | _ -> false
let is_double (type arr_t) (prec : arr_t precision) = match prec with Double -> true | _ -> false
let is_double_prec = function Double_prec _ -> true | _ -> false

let pack_prec (type a) (prec : a precision) =
  match prec with Half -> half | Single -> single | Double -> double

let get_prec = function Half_nd _ -> half | Single_nd _ -> single | Double_nd _ -> double

type 'r map_as_bigarray = { f : 'a. 'a bigarray -> 'r }

let map { f } = function Half_nd arr -> f arr | Single_nd arr -> f arr | Double_nd arr -> f arr
let dims = map { f = A.dims }

let get_as_float arr idx =
  let f x = A.get x idx in
  map { f } arr

let get_as_int arr idx =
  let f x =
    let v = A.get x idx in
    try Float.to_int v
    with Invalid_argument _ ->
      Stdio.eprintf "\nRuntime error: Ndarray.get_as_int invalid float: %f\n%!" v;
      0
  in
  map { f } arr

let set_from_float arr idx v =
  let f x = A.set x idx in
  map { f } arr v

let fill_from_float arr v = map { f = A.fill } arr v

type 'r map_prec = { f : 'a. 'a bigarray precision -> 'r }

let iter_prec { f } = function
  | Void_prec -> ()
  | Half_prec (Half | Single) -> f Half
  | Single_prec (Single | Half) -> f Single
  | Double_prec Double -> f Double
  | _ -> .

let map_prec ?default { f } = function
  | Void_prec -> Option.value_or_thunk default ~default:(fun () -> invalid_arg "map_prec: Void_prec")
  | Half_prec (Half | Single) -> f Half
  | Single_prec (Single | Half) -> f Single
  | Double_prec Double -> f Double
  | _ -> .

type 'r map_with_prec = { f : 'a. 'a bigarray precision -> 'a bigarray -> 'r }

let map_with_prec { f } = function
  | Half_nd arr -> f Half arr
  | Single_nd arr -> f Single arr
  | Double_nd arr -> f Double arr

(** Initializes or resets a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op =
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous, looping over the provided values. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
[@@deriving sexp]

let create_bigarray_of_prec (type elt_t) (prec : elt_t bigarray precision) dims : elt_t bigarray =
  A.create (precision_to_bigarray_kind prec) Bigarray.C_layout dims

let init_bigarray_of_prec (type elt_t) (prec : elt_t bigarray precision) dims ~(f : int array -> float) :
    elt_t bigarray =
  A.init (precision_to_bigarray_kind prec) Bigarray.C_layout dims f

let indices_to_offset ~dims ~idcs =
  Array.fold2_exn dims idcs ~init:0 ~f:(fun accu dim idx -> (accu * dim) + idx)

let fixed_state_for_init = ref None

let create_bigarray (type elt_t) (prec : elt_t bigarray precision) dims (init_op : init_op) : elt_t bigarray =
  Option.iter !fixed_state_for_init ~f:(fun seed -> Random.init seed);
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      init_bigarray_of_prec prec dims ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets ->
      init_bigarray_of_prec prec dims ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> init_bigarray_of_prec prec dims ~f:(fun _ -> Random.float_range 0.0 1.0)

let create_array prec dims init_op =
  let f prec = as_array prec @@ create_bigarray prec dims init_op in
  map_prec { f } prec

let unique_id = ref 1

let create prec dims init_op annot =
  let id = !unique_id in
  Int.incr unique_id;
  { array = create_array prec dims init_op; id; annot }

let precision_in_bytes = function Half_nd _ -> 2 | Single_nd _ -> 4 | Double_nd _ -> 8

let reset_bigarray arr ~f =
  let dims = A.dims arr in
  let rec cloop idx f col =
    if col = Array.length idx then A.set arr idx (f idx)
    else
      for j = 0 to Int.pred dims.(col) do
        idx.(col) <- j;
        cloop idx f (Int.succ col)
      done
  in
  let len = Array.length dims in
  cloop (Array.create ~len 0) f 0

let fold_bigarray arr ~init ~f =
  let dims = A.dims arr in
  let accu = ref init in
  let rec cloop idx col =
    if col = Array.length idx then accu := f !accu idx @@ A.get arr idx
    else
      for j = 0 to Int.pred dims.(col) do
        idx.(col) <- j;
        cloop idx (Int.succ col)
      done
  in
  let len = Array.length dims in
  cloop (Array.create ~len 0) 0;
  !accu

let empty prec = create_array prec [||] (Constant_fill [| 0.0 |])

let init_bigarray (init_op : init_op) (type b) (arr : b bigarray) =
  let dims = A.dims arr in
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      reset_bigarray arr ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets -> reset_bigarray arr ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> reset_bigarray arr ~f:(fun _ -> Random.float_range 0.0 1.0)

let init init_op arr =
  let f arr = init_bigarray init_op arr in
  map { f } arr

let fold ~init ~f arr =
  let f arr = fold_bigarray ~init ~f arr in
  map { f } arr

let size_in_bytes n =
  (* Cheating here because 1 number Bigarray is same size as empty Bigarray:
     it's more informative to report the cases differently. *)
  let f arr = if Array.is_empty @@ A.dims arr then 0 else A.size_in_bytes arr in
  map { f } n

let retrieve_2d_points ?from_axis ~xdim ~ydim arr =
  let dims = dims arr in
  if Array.is_empty dims then [||]
  else
    let n_axes = Array.length dims in
    let from_axis = Option.value from_axis ~default:(n_axes - 1) in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x =
          idx.(from_axis) <- xdim;
          get_as_float arr idx
        in
        let y =
          idx.(from_axis) <- ydim;
          get_as_float arr idx
        in
        result := (x, y) :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        for p = 0 to dims.(axis) - 1 do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

let retrieve_1d_points ?from_axis ~xdim arr =
  let dims = dims arr in
  if Array.is_empty dims then [||]
  else
    let n_axes = Array.length dims in
    let from_axis = Option.value from_axis ~default:(n_axes - 1) in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x =
          idx.(from_axis) <- xdim;
          get_as_float arr idx
        in
        result := x :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        for p = 0 to dims.(axis) - 1 do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let int_dims_to_string ?(with_axis_numbers = false) dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x " @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ Int.to_string s)
  else String.concat_array ~sep:"x" @@ Array.map dims ~f:Int.to_string

(** When rendering tensors, outputs this many decimal digits. *)
let print_decimals_precision = ref 2

let concise_float ~prec v =
  Printf.sprintf "%.*e" prec v
  |> (* The C99 standard requires at least two digits for the exponent, but the leading zero
        is a waste of space. *)
  String.substr_replace_first ~pattern:"e+0" ~with_:"e+"
  |> String.substr_replace_first ~pattern:"e-0" ~with_:"e-"

(** Prints 0-based [indices] entries out of [arr], where a number between [-5] and [-1] in an axis means
    to print out the axis, and a non-negative number means to print out only the indexed dimension of the axis.
    Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis, possibly with ellipsis
    in the middle. [labels] provides the axis labels for all axes (use [""] or ["_"] for no label).
    The last label corresponds to axis [-1] etc. The printed out axes are arranged as:
    * -1: a horizontal segment in an inner rectangle (i.e. column numbers of the inner rectangle),
    * -2: a sequence of segments in a line of text (i.e. column numbers of an outer rectangle),
    * -3: a vertical segment in an inner rectangle (i.e. row numbers of the inner rectangle),
    * -4: a vertical sequence of segments (i.e. column numbers of an outer rectangle),
    * -5: a sequence of screens of text (i.e. stack numbers of outer rectangles).
    Printing out of axis [-5] is interrupted when a callback called in between each outer rectangle
    returns true. *)
let render_tensor ?(brief = false) ?(prefix = "") ?(entries_per_axis = 4) ?(labels = [||]) ~indices arr =
  let module B = PrintBox in
  let dims = dims arr in
  let has_nan = fold ~init:false ~f:(fun has_nan _ v -> has_nan || Float.is_nan v) arr in
  let has_inf = fold ~init:false ~f:(fun has_inf _ v -> has_inf || Float.(v = infinity)) arr in
  let has_neg_inf =
    fold ~init:false ~f:(fun has_neg_inf _ v -> has_neg_inf || Float.(v = neg_infinity)) arr
  in
  let header =
    prefix
    ^ (if has_nan || has_inf || has_neg_inf then " includes" else "")
    ^ (if has_nan then " NaN" else "")
    ^ (if has_inf then " pos. inf." else "")
    ^ if has_neg_inf then " neg. inf." else ""
  in
  if Array.is_empty dims then B.vlist ~bars:false [ B.text header; B.line "<void>" ]
  else
    let indices = Array.copy indices in
    let entries_per_axis = if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis in
    let var_indices = Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (5 + d, i) else None) in
    let extra_indices =
      [| (0, -1); (1, -1); (2, -1); (3, -1); (4, -1) |]
      |> Array.filter ~f:(Fn.non @@ Array.mem var_indices ~equal:(fun (a, _) (b, _) -> Int.equal a b))
    in
    let var_indices = Array.append extra_indices var_indices in
    Array.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b) var_indices;
    let var_indices = Array.map ~f:snd @@ var_indices in
    let ind0, ind1, ind2, ind3, ind4 =
      match var_indices with
      | [| ind0; ind1; ind2; ind3; ind4 |] -> (ind0, ind1, ind2, ind3, ind4)
      | _ -> invalid_arg "render: indices should contain at most 5 negative numbers"
    in
    let labels = Array.map labels ~f:(fun l -> if String.is_empty l then "" else l ^ "=") in
    let entries_per_axis = (entries_per_axis / 2 * 2) + 1 in
    let size0 = if ind0 = -1 then 1 else min dims.(ind0) entries_per_axis in
    let size1 = if ind1 = -1 then 1 else min dims.(ind1) entries_per_axis in
    let size2 = if ind2 = -1 then 1 else min dims.(ind2) entries_per_axis in
    let size3 = if ind3 = -1 then 1 else min dims.(ind3) entries_per_axis in
    let size4 = if ind4 = -1 then 1 else min dims.(ind4) entries_per_axis in
    let no_label ind = Array.length labels <= ind in
    let label0 = if ind0 = -1 || no_label ind0 then "" else labels.(ind0) in
    let label1 = if ind1 = -1 || no_label ind1 then "" else labels.(ind1) in
    let label2 = if ind2 = -1 || no_label ind2 then "" else labels.(ind2) in
    let label3 = if ind3 = -1 || no_label ind3 then "" else labels.(ind3) in
    let label4 = if ind4 = -1 || no_label ind4 then "" else labels.(ind4) in
    (* FIXME: handle ellipsis. *)
    let halfpoint = (entries_per_axis / 2) + 1 in
    let expand i ~ind =
      if dims.(ind) <= entries_per_axis then i
      else if i < halfpoint then i
      else dims.(ind) - entries_per_axis + i
    in
    let update_indices v i j k l =
      if ind0 <> -1 then indices.(ind0) <- expand v ~ind:ind0;
      if ind1 <> -1 then indices.(ind1) <- expand i ~ind:ind1;
      if ind2 <> -1 then indices.(ind2) <- expand j ~ind:ind2;
      if ind3 <> -1 then indices.(ind3) <- expand k ~ind:ind3;
      if ind4 <> -1 then indices.(ind4) <- expand l ~ind:ind4
    in
    let elide_for i ~ind = ind >= 0 && dims.(ind) > entries_per_axis && i + 1 = halfpoint in
    let is_ellipsis () = Array.existsi indices ~f:(fun ind i -> elide_for i ~ind) in
    let inner_grid v i j =
      B.init_grid ~bars:false ~line:size3 ~col:size4 (fun ~line ~col ->
          update_indices v i j line col;
          try
            B.hpad 1 @@ B.line
            @@
            if is_ellipsis () then "..."
            else concise_float ~prec:!print_decimals_precision (get_as_float arr indices)
          with Invalid_argument _ as error ->
            Stdio.Out_channel.printf "Invalid indices: %s into array: %s\n%!" (int_dims_to_string indices)
              (int_dims_to_string dims);
            raise error)
    in
    let tag ?pos label ind =
      if ind = -1 then ""
      else
        match pos with
        | Some pos when elide_for pos ~ind -> "~~~~~"
        | Some pos when pos >= 0 -> Int.to_string (expand pos ~ind) ^ " @ " ^ label ^ Int.to_string ind
        | _ -> "axis " ^ label ^ Int.to_string ind
    in
    let nlines = if brief then size1 else size1 + 1 in
    let ncols = if brief then size2 else size2 + 1 in
    let outer_grid v =
      (if brief then Fn.id else B.frame)
      @@ B.init_grid ~bars:true ~line:nlines ~col:ncols (fun ~line ~col ->
             if (not brief) && line = 0 && col = 0 then
               B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@ [ tag ~pos:v label0 ind0 ]
             else if (not brief) && line = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(col - 1) label2 ind2; tag label4 ind4 ]
             else if (not brief) && col = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(line - 1) label1 ind1; tag label3 ind3 ]
             else
               let nline = if brief then line else line - 1 in
               let ncol = if brief then col else col - 1 in
               if elide_for ncol ~ind:ind2 || elide_for nline ~ind:ind1 then B.hpad 1 @@ B.line "..."
               else inner_grid v nline ncol)
    in
    let screens =
      B.init_grid ~bars:true ~line:size0 ~col:1 (fun ~line ~col:_ ->
          if elide_for line ~ind:ind0 then B.hpad 1 @@ B.line "..." else outer_grid line)
    in
    (if brief then Fn.id else B.frame) @@ B.vlist ~bars:false [ B.text header; screens ]

let pp_tensor fmt ?prefix ?entries_per_axis ?labels ~indices arr =
  PrintBox_text.pp fmt @@ render_tensor ?prefix ?entries_per_axis ?labels ~indices arr

(** Prints the whole tensor in an inline syntax. *)
let pp_tensor_inline fmt ~num_batch_axes ~num_output_axes ~num_input_axes ?axes_spec arr =
  let dims = dims arr in
  let num_all_axes = num_batch_axes + num_output_axes + num_input_axes in
  let open Caml.Format in
  let ind = Array.copy dims in
  (match axes_spec with None -> () | Some spec -> fprintf fmt "\"%s\" " spec);
  let rec loop axis =
    let sep =
      if axis < num_batch_axes then ";" else if axis < num_batch_axes + num_output_axes then ";" else ","
    in
    let open_delim =
      if axis < num_batch_axes then "[|"
      else if axis < num_batch_axes + num_output_axes then "["
      else if axis = num_batch_axes + num_output_axes then ""
      else "("
    in
    let close_delim =
      if axis < num_batch_axes then "|]"
      else if axis < num_batch_axes + num_output_axes then "]"
      else if axis = num_batch_axes + num_output_axes then ""
      else ")"
    in
    if axis = num_all_axes then fprintf fmt "%.*f" !print_decimals_precision (get_as_float arr ind)
    else (
      fprintf fmt "@[<hov 2>%s@," open_delim;
      for i = 0 to dims.(axis) - 1 do
        ind.(axis) <- i;
        loop (axis + 1);
        if i < dims.(axis) - 1 then fprintf fmt "%s@ " sep
      done;
      fprintf fmt "@,%s@]" close_delim)
  in
  loop 0
