(* gh-ocannl-1657: the execute-audit of padded operand windows, offsets included. A padded ([=+])
   window on an operand, [stride*o=+offset+dilation*k], reads the zero-padded operand at position
   [stride*o+offset+dilation*k-left] for each result position [o], over an operand axis of extent
   [stride*|o|]. The margins total the dilated kernel extent [span=dilation*(K-1)+1] and
   [left=span/2] (the smaller half when the split is uneven). A tap outside the operand's extent
   reads the padding value, 0 for the add family; a max-family (tropical) window is clamped to the
   extent instead, so an all-outside window yields the neutral element.

   Each form below runs and is compared, extent included, against [padded_window], computed here in
   plain OCaml from that definition alone; the values are small integers, so every sum is exact in
   single precision on any backend and in any order. The forms: the offset below, at and past the
   stride, with odd, even and dilated kernels; the raw-string, [%op] (identifier coefficients),
   [%cd] with [~logic], captured-kernel (a padded sum pool) and ternary routes; two padded windows
   in one operand; the max family; and the gradients, d/dx and d/dw, of a strided, dilated window
   with an offset, of an offset past the stride, and of a max-family window.

   The audit found the dilated window wrong: its margins were sized by the undilated kernel
   ([left=K/2]), so [o=+2*k] over a kernel of 3 read [o-1], [o+1], [o+3] instead of the centered
   [o-2], [o], [o+2], with or without an offset. It also found the documented [stride > 0] and
   [dilation > 0] invariants unenforced: [o=+0*k] ran, multiplying one cell by every tap, and
   [o<+0*k] died with an [Invalid_argument] in the dimension solver. A coefficient below 1 is now
   refused while the tensor is built ([Shape.refuse_nonpositive_coefficient]), on every route and in
   either slot; the refusals are pinned at the end. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let pg = Tensor.Prohibit_grad

let ndarray ?(grad_spec = pg) output_dims values =
  Tensor.ndarray ~grad_spec values ~batch_dims:[] ~input_dims:[] ~output_dims ()

let ramp len = Array.init len ~f:(fun i -> Float.of_int (i + 1))
let k3 = [| 1.; 2.; 4. |]
let k3b = [| 3.; 5.; 7. |]
let k2 = [| 1.; 3. |]
let k4 = [| 1.; 2.; 4.; 8. |]

(* {2 The oracle} *)

type window = { stride : int; offset : int; dilation : int }

(* The taps of result position [o] over an operand axis of extent [n] and a kernel of [kk]: [(c, k,
   inside)] for each kernel position [k], where [c] is the operand position read. *)
let taps { stride; offset; dilation } ~n ~kk o =
  let left = ((dilation * (kk - 1)) + 1) / 2 in
  List.init kk ~f:(fun k ->
      let c = (stride * o) + offset + (dilation * k) - left in
      (c, k, c >= 0 && c < n))

let padded_window ?(reduce = `Sum) win x w =
  let n = Array.length x and kk = Array.length w in
  Array.init (n / win.stride) ~f:(fun o ->
      let taps = taps win ~n ~kk o in
      match reduce with
      | `Sum ->
          List.fold taps ~init:0. ~f:(fun acc (c, k, inside) ->
              if inside then acc +. (x.(c) *. w.(k)) else acc)
      | `Max ->
          List.fold taps ~init:Float.neg_infinity ~f:(fun acc (c, k, inside) ->
              if inside then Float.max acc (x.(c) +. w.(k)) else acc))

(* {2 Executed forms} *)

let exn_message = function Row.Shape_error (msg, _) -> msg | e -> Stdlib.Printexc.to_string e

let read ctx t =
  (Ir.Tnode.dims_without_padding t.Tensor.value, Context.get_values ctx t.Tensor.value)

let matches label ~dims ~want (got_dims, got) =
  p (label ^ ": extent") (Array.equal Int.equal got_dims dims);
  pf_all2 "%s: values" label got want ~f:Float.equal

(* The operand is computed, so its layout is not committed at creation and the padded window can
   grant it margins (padding_lifecycle). *)
let computed ?(spec = "s, c => s, c") dims values =
  Operation.einsum1 spec ~grad_spec:pg (ndarray dims values) ()

let executes label ~dims ~want build =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      match
        let t = build () in
        read (Train.forward_once (Context.auto ()) t) t
      with
      | exception e ->
          Stdio.eprintf "%s: raised %s\n%!" label (exn_message e);
          pf "%s: executes" label false
      | got -> matches label ~dims ~want got)

(* A binary padded window over a [1; n] operand, from a plain string. *)
let binary ?(reduce = `Sum) label win ~n ~w spec =
  executes label
    ~dims:[| 1; n / win.stride |]
    ~want:(padded_window ~reduce win (ramp n) w)
    (fun () ->
      let x = computed [ 1; n ] (ramp n) in
      let wt = ndarray [ Array.length w ] w in
      match reduce with
      | `Sum -> Operation.einsum spec ~grad_spec:pg x wt ()
      | `Max -> Operation.tropical spec ~grad_spec:pg x wt ())

let run_forms () =
  binary "2*o=+1+k, kernel 3"
    { stride = 2; offset = 1; dilation = 1 }
    ~n:8 ~w:k3 "s, 2*o=+1+k; k => s, o";
  binary "3*o=+2+k, kernel 3"
    { stride = 3; offset = 2; dilation = 1 }
    ~n:9 ~w:k3 "s, 3*o=+2+k; k => s, o";
  binary "3*o=+1+k, kernel 4"
    { stride = 3; offset = 1; dilation = 1 }
    ~n:9 ~w:k4 "s, 3*o=+1+k; k => s, o";
  binary "2*o=+1+k, kernel 2"
    { stride = 2; offset = 1; dilation = 1 }
    ~n:8 ~w:k2 "s, 2*o=+1+k; k => s, o";
  binary "2*o=+1+2*k, kernel 3"
    { stride = 2; offset = 1; dilation = 2 }
    ~n:8 ~w:k3 "s, 2*o=+1+2*k; k => s, o";
  binary "o=+1+k (offset at the stride), kernel 3"
    { stride = 1; offset = 1; dilation = 1 }
    ~n:6 ~w:k3 "s, o=+1+k; k => s, o";
  binary "o=+2+k (offset past the stride), kernel 3"
    { stride = 1; offset = 2; dilation = 1 }
    ~n:6 ~w:k3 "s, o=+2+k; k => s, o";
  binary "2*o=+5+k (offset past the stride), kernel 3"
    { stride = 2; offset = 5; dilation = 1 }
    ~n:8 ~w:k3 "s, 2*o=+5+k; k => s, o";
  binary "o=+2*k (dilated, no offset), kernel 3"
    { stride = 1; offset = 0; dilation = 2 }
    ~n:6 ~w:k3 "s, o=+2*k; k => s, o";
  binary "2*o=+2*k, kernel 3"
    { stride = 2; offset = 0; dilation = 2 }
    ~n:8 ~w:k3 "s, 2*o=+2*k; k => s, o";
  binary "o=+1+3*k, kernel 2"
    { stride = 1; offset = 1; dilation = 3 }
    ~n:6 ~w:k2 "s, o=+1+3*k; k => s, o";
  binary ~reduce:`Max "max 2*o=+1+k, kernel 3"
    { stride = 2; offset = 1; dilation = 1 }
    ~n:8 ~w:k3 "s, 2*o=+1+k; k => s, o";
  binary ~reduce:`Max "max o=+2+k (an all-outside window is the neutral element), kernel 3"
    { stride = 1; offset = 2; dilation = 1 }
    ~n:6 ~w:k3 "s, o=+2+k; k => s, o";
  binary ~reduce:`Max "max o=+1+2*k, kernel 3"
    { stride = 1; offset = 1; dilation = 2 }
    ~n:6 ~w:k3 "s, o=+1+2*k; k => s, o";
  (let win = { stride = 2; offset = 1; dilation = 2 } in
   executes "%op, identifier stride and dilation: stride*o=+1+dilation*k" ~dims:[| 1; 4 |]
     ~want:(padded_window win (ramp 8) k3)
     (fun () ->
       let stride = win.stride and dilation = win.dilation in
       let x = computed [ 1; 8 ] (ramp 8) in
       let w = ndarray [ 3 ] k3 in
       [%op x +* "s, stride*o=+1+dilation*k; k => s, o" w]));
  (* A padded sum pool: the kernel label is captured and set, and the kernel is a stretched constant
     1, the pooling layers' idiom ([Nn_blocks.max_pool2d]). The one-operand route, with no kernel
     tensor, is executed under "One-operand windows" below. *)
  executes "sum pool, captured kernel over stretch 1.0: 2*o=+1+k" ~dims:[| 1; 4 |]
    ~want:(padded_window { stride = 2; offset = 1; dilation = 1 } (ramp 8) (Array.create ~len:3 1.))
    (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      let%op r = x +* "s, 2*o=+1+k; k => s, o" [ "k" ] (stretch 1.0) in
      Shape.set_dim k 3;
      r);
  executes "ternary 2*o=+1+k" ~dims:[| 1; 4 |]
    ~want:
      (padded_window
         { stride = 2; offset = 1; dilation = 1 }
         (ramp 8) (Array.map2_exn k3 k3b ~f:( *. )))
    (fun () ->
      Operation.einsum3 "s, 2*o=+1+k; k; k => s, o" ~grad_spec:pg
        (computed [ 1; 8 ] (ramp 8))
        (ndarray [ 3 ] k3) (ndarray [ 3 ] k3b) ());
  (* Two padded windows in one operand: rows [2*a=+1+k] over a kernel of 3, columns [b=+1+2*l] over
     a kernel of 2 dilated by 2. The oracle crosses the two 1-D tap lists. *)
  (let rows = { stride = 2; offset = 1; dilation = 1 }
   and cols = { stride = 1; offset = 1; dilation = 2 } in
   let n1 = 6 and n2 = 5 in
   let x_vals = Array.init (n1 * n2) ~f:(fun i -> Float.of_int ((10 * (i / n2)) + (i % n2) + 1)) in
   let w_vals = Array.init (3 * 2) ~f:(fun i -> k3.(i / 2) *. k2.(i % 2)) in
   let want =
     Array.init
       (n1 / 2 * n2)
       ~f:(fun i ->
         let a = i / n2 and b = i % n2 in
         List.sum
           (module Float)
           (taps rows ~n:n1 ~kk:3 a)
           ~f:(fun (r, k, r_in) ->
             List.sum
               (module Float)
               (taps cols ~n:n2 ~kk:2 b)
               ~f:(fun (c, l, c_in) ->
                 if r_in && c_in then x_vals.((r * n2) + c) *. w_vals.((k * 2) + l) else 0.)))
   in
   executes "two padded windows 2*a=+1+k, b=+1+2*l"
     ~dims:[| n1 / 2; n2 |]
     ~want
     (fun () ->
       let x = computed ~spec:"a, b => a, b" [ n1; n2 ] x_vals in
       let w = ndarray [ 3; 2 ] w_vals in
       Operation.einsum "2*a=+1+k, b=+1+2*l; k, l => a, b" ~grad_spec:pg x w ()));
  case "%cd ~logic 2*o=+1+2*k" (fun () ->
      Tensor.unsafe_reinitialize ();
      let label = "%cd ~logic 2*o=+1+2*k" in
      match
        let x = computed [ 1; 8 ] (ramp 8) in
        let w = ndarray [ 3 ] k3 in
        let%cd asn = { sm } =:+ x * w ~logic:"s, 2*o=+1+2*k; k => s, o" in
        Train.set_materialized sm.Tensor.value;
        let ctx = Context.auto () in
        let ctx, routine = Context.compile ~name:"cd_padded" ctx asn Ir.Indexing.Empty in
        read (Context.run ctx routine) sm
      with
      | exception e ->
          Stdio.eprintf "%s: raised %s\n%!" label (exn_message e);
          pf "%s: executes" label false
      | got ->
          matches label ~dims:[| 1; 4 |]
            ~want:(padded_window { stride = 2; offset = 1; dilation = 2 } (ramp 8) k3)
            got)

(* {2 Gradients}

   With the loss [sum_o c[o] * r[o]] and distinct weights [c[o] = o + 1], the result's gradient is
   [c]. A sum window has [dx[p] = sum over the taps (o, k) reading p of c[o] * w[k]] and [dw[k] =
   sum_o c[o] * x[the tap (o, k)]], a tap outside the extent contributing nothing; a max window
   sends [c[o]] to its argmax tap alone, in [x] and in [w] (the inputs below have no ties). *)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let oracle_grads ~reduce win x w =
  let n = Array.length x and kk = Array.length w in
  let dx = Array.create ~len:n 0. and dw = Array.create ~len:kk 0. in
  for o = 0 to (n / win.stride) - 1 do
    let cw = Float.of_int (o + 1) in
    let inside = List.filter (taps win ~n ~kk o) ~f:(fun (_, _, inside) -> inside) in
    match reduce with
    | `Sum ->
        List.iter inside ~f:(fun (c, k, _) ->
            dx.(c) <- dx.(c) +. (cw *. w.(k));
            dw.(k) <- dw.(k) +. (cw *. x.(c)))
    | `Max -> (
        match
          List.max_elt inside ~compare:(fun (c1, k1, _) (c2, k2, _) ->
              Float.compare (x.(c1) +. w.(k1)) (x.(c2) +. w.(k2)))
        with
        | Some (c, k, _) ->
            dx.(c) <- dx.(c) +. cw;
            dw.(k) <- dw.(k) +. cw
        | None -> ())
  done;
  (dx, dw)

let gradient ?(reduce = `Sum) label win ~n ~w spec =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let m = n / win.stride in
      let src = ndarray ~grad_spec:Tensor.Require_grad [ 1; n ] (ramp n) in
      let wt = ndarray ~grad_spec:Tensor.Require_grad [ Array.length w ] w in
      let c = ndarray [ 1; m ] (Array.init m ~f:(fun o -> Float.of_int (o + 1))) in
      let%op x = src ++ "s, c => s, c" in
      let r =
        match reduce with
        | `Sum -> Operation.einsum spec x wt ()
        | `Max -> Operation.tropical spec x wt ()
      in
      let%op loss = (r *. c) ++ "s, o => 0" in
      Train.set_materialized (grad_of x);
      Train.set_materialized (grad_of wt);
      let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
      let dx, dw = oracle_grads ~reduce win (ramp n) w in
      pf_all2 "%s: d/dx" label (Context.get_values ctx (grad_of x)) dx ~f:Float.equal;
      pf_all2 "%s: d/dw" label (Context.get_values ctx (grad_of wt)) dw ~f:Float.equal)

let run_gradients () =
  gradient "gradient of 2*o=+1+2*k"
    { stride = 2; offset = 1; dilation = 2 }
    ~n:8 ~w:k3 "s, 2*o=+1+2*k; k => s, o";
  gradient "gradient of o=+2+k"
    { stride = 1; offset = 2; dilation = 1 }
    ~n:6 ~w:k3 "s, o=+2+k; k => s, o";
  gradient ~reduce:`Max "gradient of max 2*o=+1+k"
    { stride = 2; offset = 1; dilation = 1 }
    ~n:8 ~w:k3 "s, 2*o=+1+k; k => s, o"

(* {2 One-operand windows}

   A window needs no kernel tensor: [x ++ "s, 2*o=+1+k => s, o" ["k"]] with [Shape.set_dim k 3] is a
   sum pool, and [x @^^ ...] a max pool, where no operand carries the kernel's extent. Projection
   derivation re-solves each step locally from the closed shapes, which hold every operand axis but
   not the kernel; the captured kernel's solved size reaches that solve as a projection-local sizing
   constraint (gh-ocannl-1656), so the window gets its extent from the capture (gh-ocannl-1691). The
   oracle is the binary one over an all-ones kernel (sum) or an all-zeros one (max). *)

(* The unpadded ([<]) window: result position [o] reads [stride*o + dilation*k], over an extent of
   [(n - span) / stride + 1], with no offset (refused, gh-ocannl-1618). *)
let valid_window ?(reduce = `Sum) ~stride ~dilation x ~kk =
  let n = Array.length x in
  let span = (dilation * (kk - 1)) + 1 in
  Array.init
    (((n - span) / stride) + 1)
    ~f:(fun o ->
      let taps = List.init kk ~f:(fun k -> x.((stride * o) + (dilation * k))) in
      match reduce with
      | `Sum -> List.fold taps ~init:0. ~f:( +. )
      | `Max -> List.fold taps ~init:Float.neg_infinity ~f:Float.max)

let ones kk = Array.create ~len:kk 1.
let zeros kk = Array.create ~len:kk 0.

let run_unary_forms () =
  executes "one-operand padded sum 2*o=+1+k, kernel 3" ~dims:[| 1; 4 |]
    ~want:(padded_window { stride = 2; offset = 1; dilation = 1 } (ramp 8) (ones 3))
    (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      let%op r = x ++ "s, 2*o=+1+k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r);
  executes "one-operand padded sum o=+2*k (dilated), kernel 3" ~dims:[| 1; 6 |]
    ~want:(padded_window { stride = 1; offset = 0; dilation = 2 } (ramp 6) (ones 3))
    (fun () ->
      let x = computed [ 1; 6 ] (ramp 6) in
      let%op r = x ++ "s, o=+2*k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r);
  executes "one-operand padded sum 3*o=+1+k, kernel 4, raw ~capture_dims" ~dims:[| 1; 3 |]
    ~want:(padded_window { stride = 3; offset = 1; dilation = 1 } (ramp 9) (ones 4))
    (fun () ->
      let k = Shape.get_variable_ref "k" in
      let r =
        Operation.einsum1 ~capture_dims:[ k ] "s, 3*o=+1+k => s, o" ~grad_spec:pg
          (computed [ 1; 9 ] (ramp 9))
          ()
      in
      Shape.set_dim k 4;
      r);
  executes "one-operand valid sum 2*o<+k, kernel 3" ~dims:[| 1; 4 |]
    ~want:(valid_window ~stride:2 ~dilation:1 (ramp 9) ~kk:3)
    (fun () ->
      let x = computed [ 1; 9 ] (ramp 9) in
      let%op r = x ++ "s, 2*o<+k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r);
  executes "one-operand valid sum o<+2*k (dilated), kernel 2" ~dims:[| 1; 4 |]
    ~want:(valid_window ~stride:1 ~dilation:2 (ramp 6) ~kk:2)
    (fun () ->
      let x = computed [ 1; 6 ] (ramp 6) in
      let%op r = x ++ "s, o<+2*k => s, o" [ "k" ] in
      Shape.set_dim k 2;
      r);
  executes "one-operand padded max 2*o=+1+k, kernel 3" ~dims:[| 1; 4 |]
    ~want:(padded_window ~reduce:`Max { stride = 2; offset = 1; dilation = 1 } (ramp 8) (zeros 3))
    (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      let%op r = x @^^ "s, 2*o=+1+k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r);
  executes "one-operand valid max 2*o<+k, kernel 2" ~dims:[| 1; 4 |]
    ~want:(valid_window ~reduce:`Max ~stride:2 ~dilation:1 (ramp 8) ~kk:2)
    (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      let%op r = x @^^ "s, 2*o<+k => s, o" [ "k" ] in
      Shape.set_dim k 2;
      r);
  (* [Nn_blocks.max_pool2d ~use_padding:true]'s window without its constant kernel: two padded axes,
     window 3, stride 2. The values are not monotone along either axis, so the argmax moves within
     the window. *)
  let n1 = 4 and n2 = 6 and kk = 3 in
  let win = { stride = 2; offset = 0; dilation = 1 } in
  let x_vals =
    [|
      3.;
      9.;
      1.;
      7.;
      4.;
      2.;
      8.;
      0.;
      6.;
      11.;
      5.;
      10.;
      2.;
      12.;
      7.;
      3.;
      9.;
      1.;
      6.;
      4.;
      13.;
      0.;
      8.;
      5.;
    |]
  in
  let want =
    Array.init
      (n1 / 2 * (n2 / 2))
      ~f:(fun i ->
        let a = i / (n2 / 2) and b = i % (n2 / 2) in
        List.fold (taps win ~n:n1 ~kk a) ~init:Float.neg_infinity ~f:(fun acc (r, _, r_in) ->
            List.fold (taps win ~n:n2 ~kk b) ~init:acc ~f:(fun acc (c, _, c_in) ->
                if r_in && c_in then Float.max acc x_vals.((r * n2) + c) else acc)))
  in
  executes "one-operand padded 2-D max pool 2*a=+k, 2*b=+l, window 3"
    ~dims:[| n1 / 2; n2 / 2 |]
    ~want
    (fun () ->
      let x = computed ~spec:"a, b => a, b" [ n1; n2 ] x_vals in
      let%op r = x @^^ "2*a=+k, 2*b=+l => a, b" [ "k"; "l" ] in
      Shape.set_dim k kk;
      Shape.set_dim l kk;
      r)

(* The gradient of a one-operand window with respect to its operand, against [oracle_grads] over the
   implicit kernel; [build] makes the window over [x] and sets its kernel. *)
let unary_gradient ?(reduce = `Sum) label win ~n ~kk build =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let m = n / win.stride in
      let src = ndarray ~grad_spec:Tensor.Require_grad [ 1; n ] (ramp n) in
      let c = ndarray [ 1; m ] (Array.init m ~f:(fun o -> Float.of_int (o + 1))) in
      let%op x = src ++ "s, c => s, c" in
      let r = build x in
      let%op loss = (r *. c) ++ "s, o => 0" in
      Train.set_materialized (grad_of x);
      let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
      let w = match reduce with `Sum -> ones kk | `Max -> zeros kk in
      let dx, _ = oracle_grads ~reduce win (ramp n) w in
      pf_all2 "%s: d/dx" label (Context.get_values ctx (grad_of x)) dx ~f:Float.equal)

let run_unary_gradients () =
  unary_gradient "gradient of one-operand padded sum 2*o=+1+2*k"
    { stride = 2; offset = 1; dilation = 2 } ~n:8 ~kk:3 (fun x ->
      let%op r = x ++ "s, 2*o=+1+2*k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r);
  (* Stride 2 under a window of 3: the windows overlap, so the product-space gate decides. *)
  unary_gradient ~reduce:`Max "gradient of one-operand padded max 2*o=+1+k"
    { stride = 2; offset = 1; dilation = 1 } ~n:8 ~kk:3 (fun x ->
      let%op r = x @^^ "s, 2*o=+1+k => s, o" [ "k" ] in
      Shape.set_dim k 3;
      r)

(* {2 Refused coefficients} *)

type outcome = Refused_at_construction of string | Refused_later of string | Ran

(* [build] constructs the operation and returns the thunk that compiles and runs it. A failed case
   leaves the session tainted (gh-ocannl-1301); it is rewound, so the next case starts from the last
   closure. *)
let attempt build =
  let outcome =
    match build () with
    | exception Row.Shape_error (msg, _) -> Refused_at_construction msg
    | run -> ( match run () with () -> Ran | exception e -> Refused_later (exn_message e))
  in
  if Option.is_some (Shape.taint_provenance ()) then Tensor.rewind_to_last_closure ();
  outcome

let forward t () = ignore (Train.forward_once (Context.auto ()) t : Context.t)

let refused ?(print_message = false) label ~axis ~spec build =
  case label (fun () ->
      let claim_label =
        Printf.sprintf "%s: refused at construction, naming the affine axis %s" label axis
      in
      match attempt build with
      | Refused_at_construction msg ->
          if print_message then Stdio.printf "%s: %s\n" label msg;
          let names_it =
            String.is_substring msg ~substring:("affine axis `" ^ axis ^ "`")
            && String.is_substring msg ~substring:("einsum spec \"" ^ spec ^ "\"")
            && String.is_substring msg ~substring:"must be at least 1"
          in
          if not names_it then Stdio.eprintf "%s: refused with another message: %s\n" label msg;
          pf "%s" claim_label names_it
      | Refused_later msg ->
          Stdio.eprintf "%s: refused only after construction: %s\n" label msg;
          pf "%s" claim_label false
      | Ran ->
          Stdio.eprintf "%s: was not refused\n" label;
          pf "%s" claim_label false)

let operand () = computed [ 1; 6 ] (ramp 6)
let kernel () = ndarray [ 3 ] k3

(* A one-operand padded window whose kernel nothing sizes -- not captured, or captured and never set
   -- has no extent for the kernel: a padded window's extent equation omits it. It is refused naming
   the kernel and the two remedies, where it used to die with the internal
   [projection_of_solved_dims: still not fully inferred]. (An unpadded window's extent equation does
   determine the kernel, so the unpadded form is not here.) The refusal comes at projection
   derivation, after construction, so the session is rewound by [attempt]. *)
let refused_unsized ?(print_message = false) label build =
  case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let claim_label = label ^ ": refused, naming the unsized kernel and the remedies" in
      match attempt (fun () -> forward (build ())) with
      | Refused_later msg | Refused_at_construction msg ->
          if print_message then Stdio.printf "%s: %s\n" label msg;
          let names_it =
            String.is_substring msg ~substring:"the window kernel k has no size"
            && String.is_substring msg ~substring:"Shape.set_dim"
            && String.is_substring msg ~substring:"kernel operand"
          in
          if not names_it then Stdio.eprintf "%s: refused with another message: %s\n" label msg;
          pf "%s" claim_label names_it
      | Ran ->
          Stdio.eprintf "%s: was not refused\n" label;
          pf "%s" claim_label false)

let run_unsized_kernels () =
  refused_unsized ~print_message:true "one-operand padded sum, kernel not captured" (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      [%op x ++ "s, 2*o=+1+k => s, o"]);
  refused_unsized "one-operand padded sum, kernel captured but not set" (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      let%op r = x ++ "s, 2*o=+1+k => s, o" [ "k" ] in
      ignore (k : Shape.delayed_var_ref);
      r);
  refused_unsized "one-operand padded max, kernel not captured" (fun () ->
      let x = computed [ 1; 8 ] (ramp 8) in
      [%op x @^^ "s, 2*o=+1+k => s, o"])

let run_refusals () =
  Tensor.unsafe_reinitialize ();
  refused ~print_message:true "zero dilation, padded window" ~axis:"o=+0*k"
    ~spec:"s, o=+0*k; k => s, o" (fun () ->
      forward (Operation.einsum "s, o=+0*k; k => s, o" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused "zero dilation, unpadded window" ~axis:"o<+0*k" ~spec:"s, o<+0*k; k => s, o" (fun () ->
      forward (Operation.einsum "s, o<+0*k; k => s, o" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused ~print_message:true "zero stride, padded window" ~axis:"0*o=+k"
    ~spec:"s, 0*o=+k; k => s, o" (fun () ->
      forward (Operation.einsum "s, 0*o=+k; k => s, o" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused "zero stride, unpadded window" ~axis:"0*o<+k" ~spec:"s, 0*o<+k; k => s, o" (fun () ->
      forward (Operation.einsum "s, 0*o<+k; k => s, o" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused "zero stride, operand without a window" ~axis:"0*o" ~spec:"s, 0*o => s, o" (fun () ->
      forward (Operation.einsum1 "s, 0*o => s, o" ~grad_spec:pg (operand ()) ()));
  refused "zero stride, result side" ~axis:"0*i" ~spec:"s, i => s, 0*i" (fun () ->
      forward (Operation.einsum1 "s, i => s, 0*i" ~grad_spec:pg (operand ()) ()));
  refused "zero dilation, result side" ~axis:"i<+0*k" ~spec:"s, i; k => s, i<+0*k" (fun () ->
      forward (Operation.einsum "s, i; k => s, i<+0*k" ~grad_spec:pg (operand ()) (kernel ()) ()));
  refused "zero dilation, %op identifier" ~axis:"o=+0*k" ~spec:"s, o=+0*k; k => s, o" (fun () ->
      let dilation = 0 in
      let x = operand () in
      let w = kernel () in
      let%op r = x +* "s, o=+dilation*k; k => s, o" w in
      forward r);
  refused "zero stride, %op identifier" ~axis:"0*o<+k" ~spec:"s, 0*o<+k; k => s, o" (fun () ->
      let stride = 0 in
      let x = operand () in
      let w = kernel () in
      let%op r = x +* "s, stride*o<+k; k => s, o" w in
      forward r);
  refused "zero dilation, %cd ~logic" ~axis:"o=+0*k" ~spec:"s, o=+0*k; k => s, o" (fun () ->
      let x = operand () in
      let w = kernel () in
      let%cd asn = { sm } =:+ x * w ~logic:"s, o=+0*k; k => s, o" in
      fun () ->
        let ctx = Context.auto () in
        let ctx, routine = Context.compile ~name:"cd_zero" ctx asn Ir.Indexing.Empty in
        ignore (Context.run ctx routine : Context.t);
        ignore (sm : Tensor.t))

let () =
  run_forms ();
  run_gradients ();
  run_unary_forms ();
  run_unary_gradients ();
  run_unsized_kernels ();
  run_refusals ()
