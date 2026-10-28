(* gh-ocannl-1656: operand-side concatenation -- slicing a segment out of a concatenated axis ([p^o
   => o], [o^q => o], [p^o^q => o]), the replace forms writing a segment of one ([a => a^b], [b =>
   a^b^c]).

   Before the fix the extract forms died in projection inference: the concatenated axis is an
   OPERAND axis there, every operand axis is iterated, so it got a product iterator of its own and
   the deferred concat-target pass refused the conflict ("Concat target projection ... conflicts").
   A middle extraction also asserted in [derive_projections], whose projection-local re-solve saw
   only the sum of the captured components.

   Every case is executed and checked against a host oracle built from the definitions: the forward
   values, and the gradient of [sum (t *. w)] with distinct prime weights [w], over the WHOLE
   operand -- the cells an extraction skips must get a zero gradient, the cells a replacement does
   not touch must be the result's zero initialization. Operand values are distinct, so a wrong
   segment or a wrong offset changes a value. Each case runs through [++] ([Shape.Permute]) and
   single-operand [++^] ([Shape.Block]), whose lowerings differ. All values are small integers,
   exact in single precision. The [%cd] replace forms assign over a target holding sentinels and
   check that the untouched cells keep them. The refusals: an underdetermined component is a shape
   error naming it, never an assertion failure. *)

open Base
open Ocannl
open Operation.DSL_modules
open Verdict.Claims

let primes =
  [|
    2.;
    3.;
    5.;
    7.;
    11.;
    13.;
    17.;
    19.;
    23.;
    29.;
    31.;
    37.;
    41.;
    43.;
    47.;
    53.;
    59.;
    61.;
    67.;
    71.;
    73.;
    79.;
    83.;
    89.;
  |]

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let capture name n =
  let r = Shape.get_variable_ref name in
  Shape.set_dim r n;
  r

(* The operand and result are laid out [lead; trail] in memory: the concatenated (resp. sliced) axis
   is [trail] for [`Trailing], [lead] for [`Leading], and [rows] is the other one. *)
type layout = Trailing | Leading

let flat ~layout ~rows ~width r c =
  match layout with Trailing -> (r * width) + c | Leading -> (c * rows) + r

(* A leaf holding [11 + 3 * i] at flat position [i]; [`Leading] puts the concatenated axis in the
   output row and [rows] in the input row (the input row is innermost in memory). *)
let leaf ~layout ~rows width =
  let values = Array.init (rows * width) ~f:(fun i -> Float.of_int (11 + (3 * i))) in
  let batch_dims, input_dims, output_dims =
    match (layout, rows) with
    | Trailing, 1 -> ([], [], [ width ])
    | Trailing, _ -> ([], [], [ rows; width ])
    | Leading, _ -> ([], [ rows ], [ width ])
  in
  ( Tensor.ndarray ~grad_spec:Tensor.Require_grad values ~batch_dims ~input_dims ~output_dims (),
    values )

type mode =
  | Extract  (** The result is the operand's segment at [offset]. *)
  | Replace  (** The result's segment at [offset] is the operand; the rest is zero. *)

(* One case: the operand is [rows] x [src], the result [rows] x [dst]; [offset] places the narrower
   axis inside the wider one. [make] builds the result from the operand. *)
let run_case ~label ~layout ~rows ~src ~dst ~offset ~mode make =
  Verdict.case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let x, xs = leaf ~layout ~rows src in
      let t = make x in
      let w_values = Array.sub primes ~pos:0 ~len:(rows * dst) in
      let w =
        let _, input_dims, output_dims =
          match (layout, rows) with
          | Trailing, 1 -> ([], [], [ dst ])
          | Trailing, _ -> ([], [], [ rows; dst ])
          | Leading, _ -> ([], [ rows ], [ dst ])
        in
        Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad w_values ~batch_dims:[] ~input_dims
          ~output_dims ()
      in
      let%op loss = (t *. w) ++ "...|... => |->0" in
      Train.set_materialized t.Tensor.value;
      Train.set_materialized (grad_of x);
      let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
      let want_dims =
        match (layout, rows) with
        | Trailing, 1 -> [ dst ]
        | Trailing, _ -> [ rows; dst ]
        | Leading, _ -> [ dst; rows ]
      in
      p
        (label ^ ": the result has the expected dims")
        (List.equal Int.equal (Array.to_list (Lazy.force t.value.dims)) want_dims);
      let fwd = Array.create ~len:(rows * dst) 0. and grad = Array.create ~len:(rows * src) 0. in
      let narrow = match mode with Extract -> dst | Replace -> src in
      for r = 0 to rows - 1 do
        for j = 0 to narrow - 1 do
          let s_c, d_c =
            match mode with Extract -> (offset + j, j) | Replace -> (j, offset + j)
          in
          let si = flat ~layout ~rows ~width:src r s_c
          and di = flat ~layout ~rows ~width:dst r d_c in
          fwd.(di) <- xs.(si);
          grad.(si) <- w_values.(di)
        done
      done;
      pf_all2 "%s: the forward is the operand's segment, every other result cell zero" label
        (Context.get_values ctx t.value) fwd ~f:Float.equal;
      pf_all2 "%s: the operand's gradient is the segment's weights, zero elsewhere" label
        (Context.get_values ctx (grad_of x))
        grad ~f:Float.equal)

(* Both front ends of one spec: the unary einsum and the single-operand concatenation. *)
let both ~label ~spec ?(captures = fun () -> []) ~layout ~rows ~src ~dst ~offset mode =
  run_case ~label:(label ^ " (++)") ~layout ~rows ~src ~dst ~offset ~mode (fun x ->
      TDSL.einsum1 ~capture_dims:(captures ()) spec x ());
  run_case ~label:(label ^ " (++^)") ~layout ~rows ~src ~dst ~offset ~mode (fun x ->
      TDSL.concat ~capture_dims:(captures ()) spec [| x |] ())

let () =
  (* Extraction with captured, set components. *)
  both ~label:"prefix dropped, p^o => o" ~spec:"p^o => o"
    ~captures:(fun () -> [ capture "p" 2 ])
    ~layout:Trailing ~rows:1 ~src:5 ~dst:3 ~offset:2 Extract;
  both ~label:"suffix dropped, o^q => o" ~spec:"o^q => o"
    ~captures:(fun () -> [ capture "q" 2 ])
    ~layout:Trailing ~rows:1 ~src:5 ~dst:3 ~offset:0 Extract;
  both ~label:"middle, p^o^q => o, unequal margins" ~spec:"p^o^q => o"
    ~captures:(fun () -> [ capture "p" 1; capture "q" 3 ])
    ~layout:Trailing ~rows:1 ~src:7 ~dst:3 ~offset:1 Extract;
  both ~label:"middle, singleton selected segment" ~spec:"p^o^q => o"
    ~captures:(fun () -> [ capture "p" 2; capture "q" 1 ])
    ~layout:Trailing ~rows:1 ~src:4 ~dst:1 ~offset:2 Extract;
  both ~label:"middle, zero-width suffix" ~spec:"p^o^q => o"
    ~captures:(fun () -> [ capture "p" 2; capture "q" 0 ])
    ~layout:Trailing ~rows:1 ~src:5 ~dst:3 ~offset:2 Extract;
  both ~label:"batched rows, s, p^o => s, o" ~spec:"s, p^o => s, o"
    ~captures:(fun () -> [ capture "p" 1 ])
    ~layout:Trailing ~rows:3 ~src:4 ~dst:3 ~offset:1 Extract;
  both ~label:"leading sliced axis, s -> p^o^q => s -> o" ~spec:"s -> p^o^q => s -> o"
    ~captures:(fun () -> [ capture "p" 1; capture "q" 2 ])
    ~layout:Leading ~rows:2 ~src:5 ~dst:2 ~offset:1 Extract;
  (* Replacement: the operand fills one segment of a fresh result, the rest is its zero init. *)
  both ~label:"replace prefix, a => a^b" ~spec:"a => a^b"
    ~captures:(fun () -> [ capture "b" 2 ])
    ~layout:Trailing ~rows:1 ~src:3 ~dst:5 ~offset:0 Replace;
  both ~label:"replace suffix, b => a^b" ~spec:"b => a^b"
    ~captures:(fun () -> [ capture "a" 2 ])
    ~layout:Trailing ~rows:1 ~src:3 ~dst:5 ~offset:2 Replace;
  both ~label:"replace middle, b => a^b^c" ~spec:"b => a^b^c"
    ~captures:(fun () -> [ capture "a" 1; capture "c" 2 ])
    ~layout:Trailing ~rows:1 ~src:3 ~dst:6 ~offset:1 Replace;
  both ~label:"replace middle, singleton source, b => a^b^c" ~spec:"b => a^b^c"
    ~captures:(fun () -> [ capture "a" 2; capture "c" 1 ])
    ~layout:Trailing ~rows:1 ~src:1 ~dst:4 ~offset:2 Replace

(* [%cd] replacement assigns into an existing target: the cells outside the segment keep the
   target's prior values (sentinels), which a fresh [++]/[++^] result instead zero-initializes. *)
let cd_replace ~label ~name ~dst ~offset ~src make =
  Verdict.case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let sentinel i = Float.of_int (-100 - i) and value i = Float.of_int (11 + (3 * i)) in
      let tgt =
        NTDSL.init ~l:"cd_tgt" ~prec:Ir.Ops.single ~o:[ dst ] ~f:(fun i -> sentinel i.(0)) ()
      in
      let srct =
        NTDSL.init ~l:"cd_src" ~prec:Ir.Ops.single ~o:[ src ] ~f:(fun i -> value i.(0)) ()
      in
      Train.set_materialized tgt.Tensor.value;
      let comp = make ~tgt ~srct in
      let ctx = Context.auto () in
      let ctx, routine = Context.compile ~name ctx comp Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let want =
        Array.init dst ~f:(fun i ->
            if i >= offset && i < offset + src then value (i - offset) else sentinel i)
      in
      pf_all2 "%s: the segment holds the source, every other cell its prior value" label
        (Context.get_values ctx tgt.value)
        want ~f:Float.equal)

let () =
  cd_replace ~label:"%cd prefix, a => a^b" ~name:"cdr_prefix" ~dst:5 ~offset:0 ~src:3
    (fun ~tgt ~srct -> [%cd tgt =: id srct ~logic:"a => a^b"]);
  cd_replace ~label:"%cd suffix, b => a^b" ~name:"cdr_suffix" ~dst:5 ~offset:2 ~src:3
    (fun ~tgt ~srct -> [%cd tgt =: id srct ~logic:"b => a^b"])

(* An underdetermined middle extraction is refused at closure with a message naming the components
   and the fix, not an assertion failure; margins wider than the operand are a shape error. *)
let refused ~label ~needle make =
  Verdict.case label (fun () ->
      Tensor.unsafe_reinitialize ();
      let outcome =
        match
          let t = make () in
          Lazy.force t.Tensor.value.dims
        with
        | _ -> `Accepted
        | exception Row.Shape_error (msg, _) -> `Shape_error msg
        | exception e -> `Other (Exn.to_string e)
      in
      (match outcome with
      | `Shape_error msg when String.is_substring msg ~substring:needle -> ()
      | `Shape_error msg -> Stdio.eprintf "%s: the shape error reads: %s\n" label msg
      | `Accepted -> Stdio.eprintf "%s: accepted\n" label
      | `Other e -> Stdio.eprintf "%s: raised %s\n" label e);
      p
        (label ^ ": refused with a shape error naming the fix")
        (match outcome with
        | `Shape_error msg -> String.is_substring msg ~substring:needle
        | `Accepted | `Other _ -> false))

let () =
  let x n = fst (leaf ~layout:Trailing ~rows:1 n) in
  refused ~label:"uncaptured middle, p^o^q => o" ~needle:"Shape.set_dim" (fun () ->
      NTDSL.einsum1 "p^o^q => o" (x 6) ());
  refused ~label:"margins wider than the operand, p^o => o with p = 6" ~needle:"exceed" (fun () ->
      NTDSL.einsum1 ~capture_dims:[ capture "p" 6 ] "p^o => o" (x 5) ())
