(* gh-ocannl-2362: a narrow-storage gradient accumulates at the backend's accumulator precision and
   narrows once per nest, executed against a materialized f32 run.

   Backprop lowers a gradient over its forward op's product space, so a dense layer's weight
   gradient [w.grad[o,i] += y.grad[b,s,o] * x[b,s,i]] runs its batch reduction OUTERMOST, the input
   gradient [x.grad[b,s,i] += w[o,i] * y.grad[b,s,o]] its contraction loop between two output loops,
   and the bias gradient [bias.grad[o] += y.grad[b,s,o]] its batch reduction outermost too.
   Read-modify-written at every step, a bf16 cell rounded every step, against [bf16_arithmetic]'s
   f32 residency on every backend; [Low_level.sink_narrow_reduction_loops] now moves each cell's
   loops above its reductions, where code generation holds the accumulator wide.

   One layer, its three gradients stored at bf16 and accumulated onto a NONZERO incoming value (the
   [accum_steps] path strips the zeroing; the gradients are prefilled with bf16-exact values), run
   once, beside an f32-gradient twin. Inputs are distinct multiples of 1/8 varying with every axis,
   small enough that the twin's results equal the host's double-precision sums cell for cell (the
   first claim per gradient), whatever order a backend or schedule adds them in. The claims, per
   gradient and cell for cell, both sides of equality (no tolerance): the bf16 result is the twin's
   narrowed once (round to nearest even); and, host-side, a per-step bf16 accumulation would give
   something else in some cell, so the inputs discriminate. Two schedules that must agree with the
   default: the privatized preset ([Autotune.extend_with_privatize] over the backend's default
   preset, whose tiles are judged against the program it is handed), and, where the untuned CUDA/HIP
   default seeds (the [widens] leg), the seeded default, whose shipped accumulator tiles over the
   bf16 gradients are all minted at the accumulator precision -- the tile that rounded every step in
   [gpt2_mini_train]'s bf16 backward on gfx1151. Digits go to stderr. *)

(* cc-fp-contract: exact-operands -- a product of two multiples of 1/8 is representable in f32, so a
   fused multiply-add rounds where the separate operations do, and the twin's claim against the host
   shows its additions do not round either *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let bf16 = Ir.Ops.bfloat16

let nb = 4
and ns = 64
and ni = 96
and no = 64

let rows = nb * ns
let eighths k m off = Float.of_int ((k % m) - off) /. 8.

(* Row-major over (b, s, axis): [x[r,i]], [c[r,o]], [w[o,i]]. *)
let x_at r i = eighths ((r * 5) + (i * 3)) 13 6
let c_at r o = eighths ((r * 7) + (o * 11)) 17 5
let w_at o i = eighths ((o * 3) + (i * 7)) 11 2

(* The incoming gradient values: multiples of 1/4 below 2 in magnitude, exact in bf16. *)
let incoming n = Array.init n ~f:(fun c -> Float.of_int ((c * 5 % 9) - 4) /. 4.)

type layer = {
  comp : Ir.Assignments.comp;
  grads : (string * Tn.t) list;  (** weight, input, bias, in that order *)
}

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let layer ~tag prec =
  let w =
    Operation.init ~l:("ngr_w_" ^ tag) ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
      ~f:(fun idx -> w_at idx.(0) idx.(1))
      ~grad_spec:Tensor.Require_grad ()
  in
  let bias =
    Operation.init ~l:("ngr_bias_" ^ tag) ~prec:Ir.Ops.single ~o:[ no ]
      ~f:(fun idx -> Float.of_int (idx.(0) % 5) /. 8.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    Operation.init ~l:("ngr_x_" ^ tag) ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ ni ]
      ~f:(fun idx -> x_at ((idx.(0) * ns) + idx.(1)) idx.(2))
      ~grad_spec:Tensor.Require_grad ()
  in
  let c =
    NTDSL.init ~l:("ngr_c_" ^ tag) ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ no ]
      ~f:(fun idx -> c_at ((idx.(0) * ns) + idx.(1)) idx.(2))
      ()
  in
  let%op y = (w * x) + bias in
  let%op loss = (y *. c) ++ "... => 0" in
  let grads = [ ("weight", grad_of w); ("input", grad_of x); ("bias", grad_of bias) ] in
  List.iter grads ~f:(fun (_, g) ->
      Tn.update_prec g prec;
      Train.set_materialized g);
  (* [accum_steps:1] keeps [loss.grad = 1] and strips the gradients' zeroing, so each accumulates
     onto the value it holds. *)
  { comp = Train.grad_update ~accum_steps:1 loss; grads }

let sizes = [ ("weight", no * ni); ("input", rows * ni); ("bias", no) ]

(* A cell's terms in the lowered nest's order: its incoming value, then one contribution per
   reduction step. *)
let terms name =
  let g0 = incoming (List.Assoc.find_exn sizes name ~equal:String.equal) in
  Array.mapi g0 ~f:(fun cell g ->
      g
      ::
      (match name with
      | "weight" ->
          let o = cell / ni and i = cell % ni in
          List.init rows ~f:(fun r -> c_at r o *. x_at r i)
      | "input" ->
          let r = cell / ni and i = cell % ni in
          List.init no ~f:(fun o -> w_at o i *. c_at r o)
      | _ -> List.init rows ~f:(fun r -> c_at r cell)))

(* The sums onto the incoming values, in double. *)
let exact name = Array.map (terms name) ~f:(List.fold ~init:0. ~f:( +. ))

(* Values rounded to [prec] through the library's own codec. *)
let at_prec prec vs =
  Ir.Ndarray.retrieve_flat_values
    (Ir.Ndarray.init_array ~debug:"ngr_at_prec" prec
       ~dims:[| Array.length vs |]
       ~padding:None
       ~f:(fun idx -> vs.(idx.(0))))

(* The per-step narrowing a read-modify-write nest does, in its lowered order: every cell's sum
   rounded to bf16 after each step (each cell has the same number of terms). *)
let per_step name =
  let ts = Array.map (terms name) ~f:Array.of_list in
  let acc = ref (Array.map ts ~f:(fun t -> t.(0))) in
  for k = 1 to Array.length ts.(0) - 1 do
    acc := at_prec bf16 (Array.mapi !acc ~f:(fun c a -> a +. ts.(c).(k)))
  done;
  !acc

let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let run ?(compile = fun ctx comp b -> Context.compile ctx comp b) ~name l =
  let ctx, routine =
    compile (Context.auto ())
      { l.comp with asgns = Ir.Assignments.Block_comment (name, l.comp.asgns) }
      Ir.Indexing.Empty
  in
  let ctx =
    List.fold l.grads ~init:ctx ~f:(fun ctx (gname, g) ->
        Context.set_values ctx g (incoming (List.Assoc.find_exn sizes gname ~equal:String.equal)))
  in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (routine, List.map l.grads ~f:(fun (gname, g) -> (gname, Context.get_values ctx g)))

let digits label values =
  Stdio.eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (List.map (List.take (Array.to_list values) 6) ~f:(fun v -> Printf.sprintf "%h" v)))

let () =
  let _, twin = run ~name:"ngr_twin" (layer ~tag:"f32" Ir.Ops.single) in
  let _, narrow = run ~name:"ngr_default" (layer ~tag:"bf16" bf16) in
  List.iter2_exn twin narrow ~f:(fun (name, t) (_, n) ->
      digits ("f32 twin " ^ name) t;
      digits ("bf16 default " ^ name) n;
      pf_all2 "%s gradient: the f32 twin equals the host's sum onto the incoming value" name t
        (exact name) ~f:bitwise;
      pf_all2
        "%s gradient: the bf16 result is the materialized f32 sum narrowed once, cell for cell" name
        n (at_prec bf16 t) ~f:bitwise;
      p_exists
        (Printf.sprintf "%s gradient: narrowing at every step would differ in some cell (host)" name)
        (List.zip_exn (Array.to_list (per_step name)) (Array.to_list (at_prec bf16 (exact name))))
        ~f:(fun (step, once) -> not (bitwise step once)));
  (* The privatized preset: every [Privatize] it appends judged on the program it was handed. *)
  let privatized = ref 0 in
  let l = layer ~tag:"priv" bf16 in
  let transform (opt : LL.optimized) =
    let facts = Context.backend_facts_of_name backend_name in
    let preset =
      if Ir.Backend_intf.binds_hardware_grid facts then Sched.default_gpu ~min_parallel:1 opt
      else if Ir.Backend_intf.renders_on_host_pool facts then Sched.default_cpu ~min_work:0 opt
      else []
    in
    let extended =
      Autotune.extend_with_privatize ~accum_prec:Ll_test.backend_accum_prec ~static_indices:[]
        preset opt
    in
    privatized :=
      List.count extended ~f:(function
        | Sched.Privatize { target; _ } -> List.exists l.grads ~f:(fun (_, g) -> Tn.equal g target)
        | _ -> false);
    Sched.apply extended opt
  in
  let _, priv =
    run ~name:"ngr_privatized"
      ~compile:(fun ctx comp b ->
        Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp b)
      l
  in
  p "the privatized preset privatizes a bf16 gradient's accumulation" (!privatized > 0);
  List.iter2_exn narrow priv ~f:(fun (name, n) (_, pv) ->
      pf_all2 "%s gradient: the privatized preset computes the default's values" name pv n
        ~f:bitwise);
  (* The untuned CUDA/HIP default seeds the weight gradient's contraction with a register-tiled
     sketch whose [Privatize] tile is the accumulator. *)
  let widens =
    Autotune.seeded_default_backend (Context.backend_facts_of_name backend_name)
    && not
         (Ir.Ops.equal_prec
            ((Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.accum_prec bf16)
            bf16)
  in
  let leg = Verdict.leg ~when_:widens ~on:"backends with no widening seeded default" () in
  let seeded =
    lazy
      (let l = layer ~tag:"seed" bf16 in
       let routine, values =
         run ~name:"ngr_seeded"
           ~compile:(fun ctx comp b -> Autotune.seeded_default ~min_reduction:128 ctx comp b)
           l
       in
       let targets = List.map l.grads ~f:snd in
       (* The nodes the shipped kernels write: a [Privatize] tile is written by its updates. *)
       let tiles = Hash_set.create (module Tn) in
       List.iter routine.Context.segments ~f:(fun seg ->
           Ll_test.walk seg.LL.llc ~on_stmt:(function
             | LL.Set { tn; _ } -> Hash_set.add tiles tn
             | _ -> ()));
       (* A [Privatize] tile is labelled [acc] over its target's label
          ([Schedule.apply_privatize]). *)
       let acc_tiles =
         Hash_set.to_list tiles
         |> List.filter ~f:(fun tn ->
             match tn.Tn.label with
             | "acc" :: rest ->
                 List.exists targets ~f:(fun g -> List.equal String.equal g.Tn.label rest)
             | _ -> false)
       in
       Stdio.eprintf
         "seeded: %d accumulator tiles over the bf16 gradients: %s (not part of the golden)\n%!"
         (List.length acc_tiles)
         (String.concat ~sep:", "
            (List.map acc_tiles ~f:(fun tn ->
                 Printf.sprintf "n%d_%s:%s" tn.Tn.id
                   (String.concat ~sep:"_" tn.Tn.label)
                   (Ir.Ops.prec_string (Lazy.force tn.Tn.storage_prec)))));
       (acc_tiles, values))
  in
  in_leg ~leg "the seeded default ships an accumulator tile over a bf16 gradient" (fun label ->
      p label (not (List.is_empty (fst (Lazy.force seeded)))));
  in_leg ~leg "every accumulator tile the seeded default ships over a bf16 gradient is f32"
    (fun label ->
      p_all label
        (fst (Lazy.force seeded))
        ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single));
  in_leg ~leg "the seeded default computes the default's values, every gradient cell for cell"
    (fun label ->
      p_all2 label
        (Array.concat (List.map (snd (Lazy.force seeded)) ~f:snd))
        (Array.concat (List.map narrow ~f:snd))
        ~f:bitwise)
