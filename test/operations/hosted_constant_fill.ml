(* gh-ocannl-633: a below-threshold constant literal (in-kernel [Constant_fill] init, at most
   [limit_constant_fill_size] = 16 elements by default) whose placement lands on materialized must
   behave exactly like an above-threshold one: its initialization moves to a link-time [Host_inits]
   upload instead of staying in the kernel. This pins the two user-visible faces the in-kernel init
   had:

   - a schedule retyping loops to hardware axes was rejected by [validate_parallel]'s coverage rule,
   because the straight-line init writes to the materialized constant were nested under no annotated
   loop — a legality that depended on the operands' literal size, not on the schedule; - a routine
   compiled in a fresh context after another routine had consumed the constant's fetch raised
   [User_error "The linked context lacks node ..."], because only the first forward embeds the init
   code.

   Both parts rely on the constants being materialized rather than virtual: at 4x4 extents each cell
   of [ma]/[mb] is read 4 times, which exceeds the default [virtualize_max_visits] = 1. The executed
   legs compare against an OCaml-side reference with all-distinct operand values (every cell's
   product mix is unique, so a dropped or misplaced init write cannot cancel out); all values are
   small integers, exact in single precision, compared for exact equality. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let m, n, k = (4, 4, 4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* One thread per output element: Grid x Workgroup splits of the matmul's zeroing and accumulation
   loops (the [parallel] shape of [bin/schedule_bench.ml]). Any active hardware dimension makes an
   in-kernel constant init illegal under [validate_parallel]'s coverage rule, which is what the
   parts below exercise. The i/j/k accumulation nest is the unique 3-deep loop nest of the lowered
   matmul. *)
let grid_workgroup_schedule ~mc opt =
  let paths = Ll_test.nest_paths opt.LL.llc in
  let i, j =
    match List.find paths ~f:(fun p -> List.length p = 3) with
    | Some [ i; j; _k ] -> (i, j)
    | _ -> failwith "hosted_constant_fill: no 3-deep i/j/k nest to schedule"
  in
  let ez, zsyms = Sched.expand_zero ~tn:mc in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:2 ~outer:LL.Grid ~inner:LL.Workgroup in
  let sp_zj, _, _ = Sched.split ~axis:zj ~factor:2 ~outer:LL.Grid ~inner:LL.Workgroup in
  let sp_i, _, _ = Sched.split ~axis:i ~factor:2 ~outer:LL.Grid ~inner:LL.Workgroup in
  let sp_j, _, _ = Sched.split ~axis:j ~factor:2 ~outer:LL.Grid ~inner:LL.Workgroup in
  Sched.apply [ ez; sp_zi; sp_zj; sp_i; sp_j ] opt

let () =
  let mav = Array.init (m * k) ~f:(fun i -> Float.of_int (1 + i)) in
  let mbv = Array.init (k * n) ~f:(fun i -> Float.of_int (2 + (3 * i))) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ k ] () in
  let reference =
    Array.init (m * n) ~f:(fun cell ->
        let i = cell / n and j = cell % n in
        Array.fold (Array.init k ~f:Fn.id) ~init:0. ~f:(fun acc l ->
            acc +. (mav.((i * k) + l) *. mbv.((l * n) + j))))
  in
  (* Part 1: one thread per output element (Grid x Workgroup splits of the zeroing and accumulation
     loops). Before gh-633 this compile raised [Invalid_argument] out of [validate_parallel]: the
     in-kernel init of ma/mb was a write to a materialized node covered by no hardware dimension. *)
  let%op mc = ma * mb in
  let comp = named "mm_hosted" (Train.forward mc) in
  let transform = grid_workgroup_schedule ~mc:mc.Tensor.value in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx mc.Tensor.value in
  Verdict.pass_fail_all2
    "scheduled matmul over small constant operands compiles and matches reference" values reference
    ~f:Float.equal ~detail:(fun () ->
      Printf.sprintf "got [%s]"
        (String.concat ~sep:"; " (Array.to_list (Array.map values ~f:Float.to_string))));
  (* Part 2: ma's fetch is now consumed (embedded in mc's forward above), so a computation reading
     ma compiled into a FRESH context finds no init code and no prior context holding the node.
     Before gh-633 this raised [User_error "The linked context lacks node ma"]; with the values
     registered as host-init data the fresh context self-initializes the node at link time. *)
  let%op md = ma *. ma in
  let comp2 = named "sq_hosted" (Train.forward md) in
  let ctx2 = Context.auto () in
  let ctx2, routine2 = Context.compile ctx2 comp2 Ir.Indexing.Empty in
  let ctx2 = Context.run ctx2 routine2 in
  let values2 = Context.get_values ctx2 md.Tensor.value in
  let reference2 = Array.map mav ~f:(fun v -> v *. v) in
  Verdict.pass_fail_all2
    "fresh context reads the consumed constant via its link-time host-init upload" values2
    reference2 ~f:Float.equal ~detail:(fun () ->
      Printf.sprintf "got [%s]"
        (String.concat ~sep:"; " (Array.to_list (Array.map values2 ~f:Float.to_string))));
  (* Part 3 (gh-633 review round 1, both P2s): a 1-element zero literal broadcast to a matmul
     operand lowers as whole-node [Zero_out] with the default size limit, and
     [Train.set_materialized] flips the node's intent from [Effectively_constant] to [On_device], so
     eligibility must ride the persistent [host_constant] marker. Under the same hardware-annotating
     schedule, the [Zero_out] used to be rejected outright by [validate_parallel]'s
     multi-threaded-kernel rule.

     A zero constant's CONTENT cannot discriminate by value — its expected cells equal the
     allocator/init sentinel by design (review round 2) — so the mechanism is asserted structurally:
     the optimized code handed to the schedule transform must carry no surviving setter of [mz]
     ([Ll_test.count_set] counts [Zero_out] too). The executed leg keeps a fully discriminating
     nonzero reference by ADDING the zero operand ([(ma + mz) * mb] = part 1's reference), so a
     dropped [Zero_out] over a garbage-filled buffer still shows; the value-bearing correctness of
     the upload machinery itself rides parts 1-2's nonzero constants, which share it. *)
  let mz = TDSL.ndarray [| 0. |] ~label:[ "mz" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  Train.set_materialized mz.Tensor.value;
  let%op mc3 = (ma + mz) * mb in
  let comp3 = named "mm_zero_hosted" (Train.forward mc3) in
  let ctx3 = Context.auto () in
  let transform3 opt =
    Verdict.p "the zero constant's in-kernel Zero_out moved to link time"
      (Ll_test.count_set opt mz.Tensor.value = 0);
    grid_workgroup_schedule ~mc:mc3.Tensor.value opt
  in
  let ctx3, routine3 =
    Context.compile ~lowered_transform:(fun o -> [ transform3 o ]) ctx3 comp3 Ir.Indexing.Empty
  in
  let ctx3 = Context.run ctx3 routine3 in
  let values3 = Context.get_values ctx3 mc3.Tensor.value in
  Verdict.pass_fail_all2
    "scheduled matmul adding a materialized broadcast-zero constant matches reference" values3
    reference ~f:Float.equal ~detail:(fun () ->
      Printf.sprintf "got [%s]"
        (String.concat ~sep:"; " (Array.to_list (Array.map values3 ~f:Float.to_string))))
