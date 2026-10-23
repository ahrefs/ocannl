(* gh-ocannl-627: the packed GEBP schedule [bin/narrow_gebp_bench] times, built by the bench's own
   builder ([Bench_gebp.packed_schedule], linked here rather than copied), at extents its blocking
   does not divide. The builder pads such an axis to its block before splitting it — the autotuner's
   composition (gh-ocannl-485) — and this test executes what it builds:

   every packed leg (serial, and Grid-parallel with its own zeroing nest) equals the unscheduled
   kernel CELL BY CELL, bitwise, and the two packed legs equal each other. Every [Tile_mma] of every
   packed leg rendered the register tiling, read off the compile's census, so a padded site that
   silently fell back to the scalar loops cannot pass as a measured one. And the same holds with
   narrow (bf16) storage widened into f32 packed panels, the [tile_prec] seam the padded
   [schedule_pad] leg does not cover.

   The sites cover the motivating n = 76 at the bench's default 64 x 256 blocking (rows padded to
   128, a k extent below its block), both tails at once with the micro-kernel's column tail (n =
   77), a row-only and a k-only remainder, several full k blocks before a tail, a site below both
   blocks, a rectangular site that tells the three extents apart, and a dividing control. The
   operands are the bench's own ([Bench_gebp.a_value]/[b_value]): they vary with every index pair,
   and their products are multiples of 1/8 small enough that every leg's reduction is exact whatever
   order it sums in — which is why bitwise is the right claim, at bf16 too at these extents.

   Two host-side claims per padded site say the parity has content where a pad could go wrong: the
   reference's masked rows are not all zero, and the k tail's contribution to the output is not all
   zero — so a schedule that dropped either would fail the cell comparison. The before-pin: the same
   composition WITHOUT its pads is refused at the [Swap] that sinks the k-block loop below a
   non-dividing row split, which is what kept the bench's gate in place. A k remainder alone does
   not need the pad (its guard lands inside [k_i]), so the k-only site is no part of that pin. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The compute precision a packed panel is minted at, as the bench derives it. *)
let compute_prec prec =
  let limits = Context.hardware_limits (Context.auto ()) in
  Ir.Numerics.cpu_compute_prec ~native_fp16_arithmetic:limits.Ir.Backend_intf.native_fp16_arithmetic
    prec

(* [m x k] times [k x n]: [m] rows, [n] columns, reduction depth [k]. *)
type site = { label : string; m : int; n : int; k : int; bm : int; bk : int }

let sites =
  [
    { label = "76^3 at the default 64x256"; m = 76; n = 76; k = 76; bm = 64; bk = 256 };
    { label = "77^3 at 16x32"; m = 77; n = 77; k = 77; bm = 16; bk = 32 };
    { label = "row-only remainder 76x40x64 at 16x32"; m = 76; n = 40; k = 64; bm = 16; bk = 32 };
    { label = "k-only remainder 64x40x76 at 16x32"; m = 64; n = 40; k = 76; bm = 16; bk = 32 };
    {
      label = "three k blocks and a tail 48x40x100 at 16x32";
      m = 48;
      n = 40;
      k = 100;
      bm = 16;
      bk = 32;
    };
    { label = "below both blocks 5x24x7 at 64x256"; m = 5; n = 24; k = 7; bm = 64; bk = 256 };
    { label = "rectangular 37x76x45 at 16x32"; m = 37; n = 76; k = 45; bm = 16; bk = 32 };
    { label = "dividing control 64x40x64 at 16x32"; m = 64; n = 40; k = 64; bm = 16; bk = 32 };
  ]

let operands ~prec s =
  let ma =
    NTDSL.init ~l:"ma" ~prec ~i:[ s.k ] ~o:[ s.m ]
      ~f:(fun idcs -> Bench_gebp.a_value idcs.(0) idcs.(1))
      ()
  in
  let mb =
    NTDSL.init ~l:"mb" ~prec ~i:[ s.n ] ~o:[ s.k ]
      ~f:(fun idcs -> Bench_gebp.b_value idcs.(0) idcs.(1))
      ()
  in
  (ma, mb)

(* One compile and run of [ma * mb] under [schedule] (none: the unscheduled kernel), returning the
   readback, the census, and the site the builder read off the nest. *)
let run ~prec ~name ~ma ~mb ~schedule =
  let%op mc = ma * mb in
  Ir.Tnode.update_prec mc.Tensor.value prec;
  let site = ref None in
  let transform opt =
    match schedule with
    | None -> opt
    | Some s ->
        site := Some (Bench_gebp.find_site opt);
        Sched.apply (s ~c:mc.Tensor.value opt) opt
  in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named name (Train.forward mc))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx mc.Tensor.value, Ll_test.renderings routine.Context.mma, !site)

let packed ~grid ~tile_prec ~bm ~bk ~ma ~mb ~c opt =
  Bench_gebp.packed_schedule ~grid ~tile_prec ~bm ~bk ~a:ma.Tensor.value ~b:mb.Tensor.value ~c opt

(* The k tail's contribution to output cell (row, col): the sum over the reduction indices past the
   last full k block. Host arithmetic over the bench's own operand values. *)
let k_tail_contribution s ~row ~col =
  let first = s.k / s.bk * s.bk in
  List.sum
    (module Float)
    (List.range first s.k)
    ~f:(fun kk -> Bench_gebp.a_value row kk *. Bench_gebp.b_value kk col)

let check_site ~prec s =
  let tag = Ir.Ops.prec_string prec in
  let lbl what = Printf.sprintf "%s, %s storage: %s" s.label tag what in
  let cprec = compute_prec prec in
  let tile_prec = if Ir.Ops.equal_prec cprec prec then None else Some cprec in
  let ma, mb = operands ~prec s in
  let want, _, _ = run ~prec ~name:"gpp_ref" ~ma ~mb ~schedule:None in
  let ser, ser_census, ser_site =
    run ~prec ~name:"gpp_serial" ~ma ~mb
      ~schedule:(Some (packed ~grid:false ~tile_prec ~bm:s.bm ~bk:s.bk ~ma ~mb))
  in
  let par, par_census, _ =
    run ~prec ~name:"gpp_grid" ~ma ~mb
      ~schedule:(Some (packed ~grid:true ~tile_prec ~bm:s.bm ~bk:s.bk ~ma ~mb))
  in
  p
    (lbl "the builder reads the site's m x n x k off the nest")
    (match ser_site with
    | Some { Bench_gebp.ni; nj; nk; _ } -> ni = s.m && nj = s.n && nk = s.k
    | None -> false);
  p (lbl "the reference is not all zeros") (Array.exists want ~f:(fun x -> Float.(x <> 0.)));
  p_all2
    (lbl "the serial packed leg matches the unscheduled kernel bitwise")
    ser want ~f:Float.equal;
  p_all2 (lbl "the Grid packed leg matches the serial packed leg bitwise") par ser ~f:Float.equal;
  p
    (lbl "every serial Tile_mma renders the register tiling")
    (Ll_test.all_rendered ~what:(lbl "serial") Ir.C_syntax.Mma_register_tiled ser_census);
  p
    (lbl "every Grid Tile_mma renders the register tiling")
    (Ll_test.all_rendered ~what:(lbl "grid") Ir.C_syntax.Mma_register_tiled par_census)

(* What the comparisons above can see at the padded edges, from the operand values alone: a padded
   row block's real rows, and the k tail's share of the output. Either all-zero would let a schedule
   that dropped it pass. *)
let edge_content s =
  let lbl what = Printf.sprintf "%s: %s" s.label what in
  if s.m % s.bm <> 0 then
    p_exists
      (lbl "the last (masked) row block's rows carry nonzero k-sums")
      (List.range (s.m / s.bm * s.bm) s.m)
      ~f:(fun row ->
        List.exists (List.range 0 s.n) ~f:(fun col ->
            Float.(
              List.sum
                (module Float)
                (List.range 0 s.k)
                ~f:(fun kk -> Bench_gebp.a_value row kk *. Bench_gebp.b_value kk col)
              <> 0.)));
  if s.k % s.bk <> 0 then
    p_exists (lbl "the k tail contributes to some output cell") (List.range 0 s.m) ~f:(fun row ->
        List.exists (List.range 0 s.n) ~f:(fun col -> Float.(k_tail_contribution s ~row ~col <> 0.)))

let () =
  List.iter sites ~f:(fun s ->
      edge_content s;
      List.iter [ Ir.Ops.single; Ir.Ops.bfloat16 ] ~f:(fun prec ->
          let labels =
            List.map
              [
                "the builder reads the site's m x n x k off the nest";
                "the reference is not all zeros";
                "the serial packed leg matches the unscheduled kernel bitwise";
                "the Grid packed leg matches the serial packed leg bitwise";
                "every serial Tile_mma renders the register tiling";
                "every Grid Tile_mma renders the register tiling";
              ]
              ~f:(Printf.sprintf "%s, %s storage: %s" s.label (Ir.Ops.prec_string prec))
          in
          (* The packed composition is the C backends' register-tiled GEBP ([Tensorize] at a unit
             lane width); a GPU backend has nothing of it to execute. *)
          if on_cpu then
            Verdict.case
              (Printf.sprintf "%s, %s storage" s.label (Ir.Ops.prec_string prec))
              (fun () -> check_site ~prec s)
          else List.iter labels ~f:(fun l -> Verdict.skipped ~backend:backend_name l)))

(* The before-pin: the builder's composition with its pads removed, at a non-dividing extent. The
   [Split] then wraps the inner loop's body in its remainder guard, and the [Swap] that sinks the k
   block loop refuses a loop whose body is a guard rather than the next loop — the refusal the
   issue's measurement ran into, and the reason the pads come first. *)
let () =
  let label = "without its pads, the composition is refused at a Swap across the remainder guard" in
  if on_cpu then
    Verdict.case label (fun () ->
        let s = List.hd_exn sites in
        let ma, mb = operands ~prec:Ir.Ops.single s in
        let unpadded ~c opt =
          List.filter (packed ~grid:false ~tile_prec:None ~bm:s.bm ~bk:s.bk ~ma ~mb ~c opt)
            ~f:(function
            | Sched.Pad _ -> false
            | _ -> true)
        in
        let refused =
          match run ~prec:Ir.Ops.single ~name:"gpp_unpadded" ~ma ~mb ~schedule:(Some unpadded) with
          | _ -> None
          | exception e -> Some (Exn.to_string e)
        in
        p label
          (Option.exists refused ~f:(fun msg ->
               String.is_substring msg ~substring:"Schedule.Swap"
               && String.is_substring msg ~substring:"not perfectly nested")))
  else Verdict.skipped ~backend:backend_name label
