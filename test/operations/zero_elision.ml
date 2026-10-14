(* gh-ocannl-1257: a covering zero whose node is next touched by an accumulation that covers every
   cell is dropped, and that accumulation opens from zero itself
   ([Ir.Low_level.elide_covered_zeros], applied by [Backends] to a GPU routine's final kernel list
   under [gpu_elide_covered_zeros]). The motivating case is the composed attention backward: its
   [(b, h, s, s)] gradients were each zeroed by the step's zero kernel and then read back by their
   first accumulation.

   One attention training step ([Train.grad_update]: forward, gradient zeroing and backprop in one
   routine), compiled three ways in one process, each in a fresh lineage with identical parameters:

   - reference: the gate off, the default schedules as they are -- every zero kept; - elided: the
   gate off, and the default schedules' kernel list passed through the pass by an explicit
   transform, so the rewritten accumulations execute on every backend, cc included; - shipped: the
   gate on and no transform -- what [Backends] does by itself, which elides on a GPU backend and
   leaves a CPU backend's zero segments alone.

   Executed parity: the loss and every parameter gradient are bitwise equal across the three, on the
   routine's first run AND its second -- an accumulation that opened from a stale cell instead of
   zero would double on the second run, which the zero-then-accumulate reference cannot. Bitwise
   means the float's bits, so a positive zero turned negative fails too. The values are
   non-saturated (the input ramp is scaled into [0, 1), as in [online_softmax.ml]), so the gradients
   are neither zero nor one-hot.

   Structure: the reference has covering zeros of the [(b, h, s, s)] gradients (the population the
   claims are about); the elided list has none of them, and keeps no covering zero whose next access
   the pass would accept (it is maximal); each of the pass's three accumulation forms occurs in it
   -- the constant opening, the localized scope, the first-iteration guard -- and none occurs in the
   reference. *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Sched = Ir.Schedule

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let batch = 2

(* Distinct from every other extent of the model, so a node with two axes of this extent is one of
   the attention's [seq, seq] intermediates or their gradients. *)
let seq = 7
let d_model = 16
let heads = 2
let d_k = 8

(* Built per model: every run starts from [Tensor.unsafe_reinitialize], so the parameters' ids, and
   with them their initial values, repeat only if each run builds the same tensors in the same
   order. *)
let mask () =
  NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
    ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
    ()

let model () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let%op x = x /. 224. in
  let mask = mask () in
  let block = Nn_blocks.multi_head_attention ~label:[ "attn" ] ~num_heads:heads ~d_k ~d_v:d_k () in
  let%op y = x + block ~train_step:None ~mask x in
  y

type run = {
  loss : float;
  first : (string * float array) list;
  second : (string * float array) list;
  segments : LL.t list;
}

let gate on = Unix.putenv "OCANNL_GPU_ELIDE_COVERED_ZEROS" (if on then "true" else "false")

let elide segs =
  List.filter_map
    (List.zip_exn segs (LL.elide_covered_zeros (List.map segs ~f:(fun s -> s.LL.llc))))
    ~f:(fun (seg, llc) -> Option.map llc ~f:(fun llc -> { seg with LL.llc }))

let step ~gate_on ~transform () =
  Tensor.unsafe_reinitialize ();
  gate gate_on;
  let y = model () in
  let%op loss = (y *. y) ++ "... | ... => 0" in
  let params =
    Set.to_list y.Tensor.params
    |> List.sort ~compare:(fun a b -> Int.compare a.Tensor.value.Tn.id b.Tensor.value.Tn.id)
  in
  List.iter params ~f:(fun p -> Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
  let limits = Context.hardware_limits ctx in
  let lowered_transform =
    Option.map transform ~f:(fun f o ->
        f (Sched.maybe_default_schedules ~backend_name ~limits ~static_indices:[] o))
  in
  let ctx, routine = Context.compile ?lowered_transform ctx update Ir.Indexing.Empty in
  let read ctx =
    List.map params ~f:(fun p ->
        ( Tn.debug_name p.Tensor.value,
          Context.get_values ctx (Option.value_exn p.Tensor.diff).Tensor.grad ))
  in
  let ctx = Context.run ctx routine in
  let first = read ctx
  and loss = Array.fold (Context.get_values ctx loss.Tensor.value) ~init:0. ~f:( +. ) in
  let second = read (Context.run ctx routine) in
  gate true;
  { loss; first; second; segments = List.map routine.Context.segments ~f:(fun s -> s.LL.llc) }

let lines segments = LL.flat_lines segments

(* Per statement of a kernel list: [Some (tn, next)] for a covering zero of [tn], with the statement
   that next touches [tn]; [None] for any other statement. *)
let zero_info segments =
  let lines = Array.of_list (lines segments) in
  Array.to_list lines
  |> List.mapi ~f:(fun i line ->
      Option.map (LL.zero_fill_target line) ~f:(fun tn ->
          let next =
            Array.findi lines ~f:(fun j l ->
                j > i
                && List.exists (LL.affine_accesses l) ~f:(fun a -> Tn.equal a.Ir.Affine.a_tn tn))
          in
          (tn, Option.map next ~f:snd)))

let zeros_with_next segments = List.filter_map (zero_info segments) ~f:Fn.id

let elidable = function
  | Some (tn, Some next) -> Option.is_some (LL.seed_first_write ~tn next)
  | _ -> false

let square tn = Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2
let is_grad tn = List.mem tn.Tn.label "grad" ~equal:String.equal

(* The pass's three accumulation forms, as they appear in a kernel list. *)
let constant_openings segments =
  Ll_test.count_scalar ~in_scopes:false
    ~f:(function
      | LL.Binop (Ir.Ops.Add, (LL.Constant 0., _), _)
      | LL.Ternop (Ir.Ops.FMA, _, _, (LL.Constant 0., _)) ->
          true
      | _ -> false)
    (LL.unflat_lines (lines segments))

let scope_openings segments =
  Ll_test.count_stmt
    ~f:(function
      | LL.Set
          {
            llsc =
              LL.Local_scope
                {
                  mint = LL.Schedule_minted;
                  body = LL.Seq (LL.Set_local (_, LL.Constant 0.), LL.For_loop _);
                  _;
                };
            _;
          } ->
          true
      | _ -> false)
    (LL.unflat_lines (lines segments))

let first_iteration_guards segments =
  Ll_test.count_scalar
    ~f:(function
      | LL.Ternop
          ( Ir.Ops.Where,
            ( LL.Binop
                (Ir.Ops.Cmple, (LL.Embed_index _, _), (LL.Embed_index (Ir.Indexing.Fixed_idx _), _)),
              _ ),
            (LL.Constant 0., _),
            (LL.Get _, _) ) ->
          true
      | _ -> false)
    (LL.unflat_lines (lines segments))

let bits a = Array.map a ~f:Int64.bits_of_float

let same_values ~what (a : run) (b : run) =
  p
    (what ^ ": the loss is bitwise equal")
    (Int64.equal (Int64.bits_of_float a.loss) (Int64.bits_of_float b.loss));
  List.iter2_exn a.first b.first ~f:(fun (name, x) (_, y) ->
      p_all2
        (what ^ ": " ^ name ^ ".grad is bitwise equal on the first run")
        (bits x) (bits y) ~f:Int64.equal);
  List.iter2_exn a.second b.second ~f:(fun (name, x) (_, y) ->
      p_all2
        (what ^ ": " ^ name ^ ".grad is bitwise equal on the second run")
        (bits x) (bits y) ~f:Int64.equal)

let () =
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  let reference = step ~gate_on:false ~transform:None () in
  let elided = step ~gate_on:false ~transform:(Some elide) () in
  let shipped = step ~gate_on:true ~transform:None () in
  printf "--- executed parity ---\n";
  p_all "reference: every parameter gradient is nonzero somewhere" reference.first ~f:(fun (_, g) ->
      Array.exists g ~f:(fun v -> Float.(v <> 0.)));
  p "reference: the second run repeats the first"
    (List.equal
       (fun (_, x) (_, y) -> Array.equal Int64.equal (bits x) (bits y))
       reference.first reference.second);
  same_values ~what:"elided" elided reference;
  same_values ~what:"shipped" shipped reference;
  printf "--- structure ---\n";
  let ref_zeros = zeros_with_next reference.segments in
  let square_grad (tn, _) = square tn && is_grad tn in
  p_exists "reference: the attention's (b, h, s, s) gradients are zeroed" ref_zeros ~f:square_grad;
  p_exists "reference: some covering zero is next touched by an accumulation the pass accepts"
    (zero_info reference.segments) ~f:elidable;
  let el_zeros = zeros_with_next elided.segments in
  p_all "elided: none of the reference's zeroed (b, h, s, s) gradients is zeroed"
    (List.filter ref_zeros ~f:square_grad) ~f:(fun (tn, _) ->
      not (List.exists el_zeros ~f:(fun (tn', _) -> Tn.equal tn tn')));
  p_none "elided: no covering zero is left whose next access the pass accepts"
    (zero_info elided.segments) ~f:elidable;
  p "elided: fewer covering zeros than the reference" (List.length el_zeros < List.length ref_zeros);
  let forms =
    [
      ("constant openings", constant_openings);
      ("localized scopes opening at zero", scope_openings);
      ("first-iteration guards", first_iteration_guards);
    ]
  in
  List.iter forms ~f:(fun (what, count) ->
      p ("reference: no " ^ what) (count reference.segments = 0);
      p ("elided: " ^ what ^ " occur") (count elided.segments > 0));
  printf "--- the gate ---\n";
  (* One claim line on every backend, so the golden is backend-uniform: a GPU backend ships what the
     explicit pass produces, a CPU backend ships its reference zeros. *)
  let expected = if Sched.backend_is_gpu backend_name then elided else reference in
  p "shipped: elides exactly on a GPU backend, as the explicit pass does"
    (List.length (zeros_with_next shipped.segments)
     = List.length (zeros_with_next expected.segments)
    && List.length shipped.segments = List.length expected.segments)
