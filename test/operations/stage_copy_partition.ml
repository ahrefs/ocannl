(* gh-ocannl-1661: a shared Stage partitions its copy across the workgroup axis its tile does not
   use, rather than restricting the copy to one representative thread of that axis.

   The staged matmul of schedule_smem_matmul ([mc = ma * mb], i and j split into Grid x Workgroup by
   [bm], k into Serial x Serial by [bk], both operands staged at k_o), at three geometries, one per
   way [Schedule.apply_stage] folds the unused axis (extent [w = bm]) into the copy's serial k loop
   (extent [e = bk]): - divide: [w] divides [e] (8 into 16): the loop runs [e / w] steps over [w * s
   + t], no guard; - drop: [e <= w] (8 against 16): the axis replaces the loop under [t < e]; -
   restrict: neither (24 against 16): one representative thread ([t == 0]) copies, as before.

   On GPU backends each kernel executes against the serial twin, its accumulator prefilled off the
   result so a dropped zero shows. Everywhere, the scheduled IR is read on the host
   ([Ll_test.captured_lowering]): which guards survive, and whether every tile store names both
   Workgroup loops -- each thread copying cells of its own. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let skipped = Verdict.skipped ~backend:backend_name
let has_shared = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let geometry ~tag ~n ~bm ~bk ~expect_guard ~partitioned =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ tag ^ "_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ tag ^ "_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  (* The serial twin: [i, j] sums [ma[i, k] * mb[k, j]] over k, on the host. *)
  let want =
    Array.init (n * n) ~f:(fun c ->
        let i = c / n and j = c % n in
        List.sum
          (module Float)
          (List.range 0 n)
          ~f:(fun k -> mav.((i * n) + k) *. mbv.((k * n) + j)))
  in
  p_exists (tag ^ ": the serial twin has a nonzero cell") (Array.to_list want) ~f:(fun x ->
      Float.(x <> 0.));
  let%op mc = ma * mb in
  let schedule (opt : LL.optimized) : Sched.schedule =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:mc.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      stage ma.Tensor.value [ i_i; k_i ];
      stage mb.Tensor.value [ k_i; j_i ];
    ]
  in
  let comp = named ("scp_" ^ tag) (Train.forward mc) in
  let transform opt = Sched.apply (schedule opt) opt in
  let parity = tag ^ ": staged matmul matches the serial twin" in
  if has_shared then
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let prefill = Array.init (n * n) ~f:(fun c -> Float.of_int (1000 + c)) in
    let ctx = Context.set_values ctx mc.Tensor.value prefill in
    let ctx = Context.run ctx routine in
    p_close parity (Context.get_values ctx mc.Tensor.value) want ~tol:(fun _ -> 1e-3)
  else skipped parity;
  let post = transform (Ll_test.captured_lowering ~name:("scp_capture_" ^ tag) comp) in
  let guards = ref [] in
  Ll_test.walk post.LL.llc ~on_if:(fun c -> guards := c :: !guards);
  let is_guard op = function LL.Binop (op', _, _) -> Ir.Ops.equal_binop op op' | _ -> false in
  let stores = Ll_test.tile_store_partitions ~tiles:post.LL.workgroup_shared post.LL.llc in
  (match expect_guard with
  | None -> p_empty (tag ^ ": no guard survives over the tile copies") ~over:stores !guards
  | Some (what, op) ->
      p (tag ^ ": exactly the two copies are guarded") (List.length !guards = 2);
      p_all (tag ^ ": each guard is " ^ what) !guards ~f:(is_guard op));
  p_all
    (tag
    ^
    if partitioned then ": every tile store is partitioned across both Workgroup loops"
    else ": every tile store sits under both Workgroup loops, indexed by one")
    stores
    ~f:(fun (n_wg, named_all) -> n_wg = 2 && Bool.equal named_all partitioned)

let () =
  geometry ~tag:"divide" ~n:32 ~bm:8 ~bk:16 ~expect_guard:None ~partitioned:true;
  geometry ~tag:"drop" ~n:32 ~bm:16 ~bk:8
    ~expect_guard:(Some ("the thread bound t < e", Ir.Ops.Cmplt))
    ~partitioned:true;
  geometry ~tag:"restrict" ~n:48 ~bm:16 ~bk:24
    ~expect_guard:(Some ("the representative t == 0", Ir.Ops.Cmpeq))
    ~partitioned:false
