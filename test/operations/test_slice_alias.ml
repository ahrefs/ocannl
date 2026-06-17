(* Regression test for gh-ocannl-293 subtask 293a: [Fetch.Slice] ([@|]) is lowered as a zero-copy
   alias *view* of its parent for unpadded leading-axis slices, instead of materializing a copy.

   Invariants pinned here (each would break a specific acceptance criterion if it regressed):
   - AC1 (no fresh allocation): an alias-eligible slice owns NO buffer -- it is absent from the
     routine context's [ctx_buffers], while its parent is present. If the alias path silently fell
     back to a copy (or allocated a fresh pool), the slice would appear in [ctx_buffers].
   - AC2 (mutation visibility, both directions), proved with kernels, not host slice access:
       * read-through (parent -> slice): running the slice's consumer with different runtime
         [batch_idx] values reads different parent rows -- proving the slice indexes into the parent's
         storage at the *runtime* binding value (a static byte-offset alias could not do this).
       * write-through (slice -> parent): a kernel assignment THROUGH the slice mutates the parent's
         backing buffer, observed by reading the parent.
   - R2 host-access contract: direct [Context.get_values]/[set_values] on an alias view raise a clear
     [User_error] (no silent fresh allocation), since the view has no buffer of its own.

   AC3 (ineligible slices still work via the copy fallback) is covered by [check_slice_shapes], whose
   slice is conv-padded -> ineligible -> materialized copy, and which stays byte-for-byte green. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX

(* A gradient-free parent tensor of shape [batch; out], row-major from [vals]. *)
let make_parent label vals ~batch ~out =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[ batch ] ~input_dims:[] ~output_dims:[ out ] ()

let arr_to_string a =
  "[" ^ String.concat ~sep:" " (Array.to_list (Array.map a ~f:(Printf.sprintf "%g"))) ^ "]"

let () =
  Tensor.unsafe_reinitialize ();
  let batch_n, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  (* Parent rows: row0 = [1 2 3], row1 = [4 5 6]. *)
  let images = make_parent "images" [| 1.; 2.; 3.; 4.; 5.; 6. |] ~batch:2 ~out:3 in
  let%op bv = images @| batch_n in
  let%op out = bv *. 2 in

  let ctx = Context.auto () in
  let routine = Train.to_routine ctx bindings (Train.forward out) in
  let ctx = Context.context routine in
  let bref = IDX.find_exn (Context.bindings routine) batch_n in

  (* --- AC1: the alias view owns no buffer; the parent does. --- *)
  Stdio.printf "slice is an alias view: %b\n" (Ir.Tnode.is_alias bv.value);
  Stdio.printf "slice absent from ctx_buffers (no fresh alloc): %b\n"
    (not (Context.mem ctx bv.value));
  Stdio.printf "parent present in ctx_buffers: %b\n" (Context.mem ctx images.value);

  (* --- AC2 read-through: the slice reads the parent row selected by the runtime batch index. --- *)
  bref := 0;
  Train.run ctx routine;
  let out0 = Context.get_values ctx out.value in
  bref := 1;
  Train.run ctx routine;
  let out1 = Context.get_values ctx out.value in
  Stdio.printf "read-through batch 0: out=%s (expect [2 4 6]): %b\n" (arr_to_string out0)
    (Array.equal Float.equal out0 [| 2.; 4.; 6. |]);
  Stdio.printf "read-through batch 1: out=%s (expect [8 10 12]): %b\n" (arr_to_string out1)
    (Array.equal Float.equal out1 [| 8.; 10.; 12. |]);

  (* --- R2: direct host read/write of the alias view is rejected (no silent fresh allocation). --- *)
  let raised_get =
    try
      ignore (Context.get_values ctx bv.value : float array);
      false
    with Utils.User_error _ -> true
  in
  let raised_set =
    try
      ignore (Context.set_values ctx bv.value [| 0.; 0.; 0. |] : Context.t);
      false
    with Utils.User_error _ -> true
  in
  Stdio.printf "host get_values on alias raises: %b\n" raised_get;
  Stdio.printf "host set_values on alias raises: %b\n" raised_set;
  Stdio.printf "alias still absent from ctx_buffers after rejected host access: %b\n"
    (not (Context.mem ctx bv.value));

  (* --- AC2 write-through: a kernel write THROUGH the slice mutates the parent's buffer. The slice's
     [alias_of] is already set (persisted on the tnode from the forward lowering above), so this write
     routine redirects to the parent. We write row [batch_n=0] to 99 and observe it via the parent. --- *)
  let%cd writer = bv =: !.99.0 in
  let wroutine = Train.to_routine ctx bindings writer in
  let ctx = Context.context wroutine in
  let wref = IDX.find_exn (Context.bindings wroutine) batch_n in
  wref := 0;
  Train.run ctx wroutine;
  let parent_after = Context.get_values ctx images.value in
  Stdio.printf "write-through: parent=%s (expect [99 99 99 4 5 6]): %b\n" (arr_to_string parent_after)
    (Array.equal Float.equal parent_after [| 99.; 99.; 99.; 4.; 5.; 6. |]);
  ()
