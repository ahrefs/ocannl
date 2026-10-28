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
   [o-2], [o], [o+2], with or without an offset. *)

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
     1, the pooling layers' idiom ([Nn_blocks.max_pool2d]): projection inference needs an operand
     carrying the window's extent, so the unary [x ++ "s, 2*o=+1+k => s, o"] is not the route. *)
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

let () =
  run_forms ();
  run_gradients ()
