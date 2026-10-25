(* gh-ocannl-627, gh-ocannl-1562: the packed GEBP schedules [bin/narrow_gebp_bench] and
   [bin/schedule_bench]'s CPU branch time, built by the benches' own builder
   ([Bench_gebp.packed_schedule], linked here rather than copied), at extents its blocking does not
   divide. The builder pads such an axis to its block before splitting it — the autotuner's
   composition (gh-ocannl-485) — and this test executes what it builds:

   every packed leg that stages both operands (serial; Grid row blocks outermost, with their own
   zeroing nest; Grid row blocks below the k-block loop; Grid outermost with the B~ panel hoisted to
   the constant pool) equals the unscheduled kernel CELL BY CELL, bitwise, and the legs equal each
   other. The layout that reads A in place ([schedule_bench]'s [pm_hoist]) is refused at every
   padded site, by the builder and — with its refusal bypassed by dropping A's [Stage] — by
   [Tensorize] itself, so the builder's gate is the compiler's; at the dividing control it builds
   and agrees. Every [Tile_mma] of every packed leg rendered the register tiling, read off the
   compile's census, so a padded site that silently fell back to the scalar loops cannot pass as a
   measured one. And the same holds with narrow (bf16) storage widened into f32 packed panels, the
   [tile_prec] seam the padded [schedule_pad] leg does not cover.

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
  (* The hoisted-B~ legs pack mb at link time, which needs a known-constant source. [init] does not
     mark one ([Tensor.ndarray], [schedule_bench]'s operand constructor, does), and mb is written
     only by its initialization, so it is declared constant the way [Tensor.ndarray] declares it. *)
  Ir.Tnode.set_host_constant mb.Tensor.value;
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

let packed ~layout ~hoist_b ~stage_a ~tile_prec ~bm ~bk ~ma ~mb ~c opt =
  Bench_gebp.packed_schedule ~layout ~hoist_b ~stage_a ~tile_prec ~bm ~bk ~a:ma.Tensor.value
    ~b:mb.Tensor.value ~c opt

(* The packed legs both benches time, all staging both operands: [narrow_gebp_bench]'s two
   ([Serial], [Grid_outermost]) and [schedule_bench]'s four (those plus [Grid_below_k], its
   [packmma_par], and the hoisted-B~ [Grid_outermost], its [pm_mixed]) — gh-ocannl-1562. *)
type leg = { leg : string; layout : Bench_gebp.layout; hoist_b : bool; what : string }

let legs =
  [
    { leg = "serial"; layout = Bench_gebp.Serial; hoist_b = false; what = "serial" };
    { leg = "grid"; layout = Bench_gebp.Grid_outermost; hoist_b = false; what = "Grid" };
    { leg = "gridk"; layout = Bench_gebp.Grid_below_k; hoist_b = false; what = "Grid-below-k" };
    { leg = "gridh"; layout = Bench_gebp.Grid_outermost; hoist_b = true; what = "hoisted-B~ Grid" };
  ]

let padded s = s.m % s.bm <> 0 || s.k % s.bk <> 0

let agrees_label l =
  if String.equal l.leg "serial" then "the serial packed leg matches the unscheduled kernel bitwise"
  else Printf.sprintf "the %s packed leg matches the serial packed leg bitwise" l.what

let tiled_label what = Printf.sprintf "every %s Tile_mma renders the register tiling" what

(* [schedule_bench]'s [pm_hoist]: the hoisted-B~ Grid leg reading A in place. Padded, the builder
   refuses it, and the refusal is the compiler's own — the composition with A's [Stage] dropped is
   refused at [Tensorize]; dividing, it builds and agrees. *)
let in_place_labels s =
  if padded s then
    [
      "with A read in place, the builder refuses to pad";
      "with A read in place, Tensorize itself refuses the padded composition";
    ]
  else
    [ "the A-in-place packed leg matches the serial packed leg bitwise"; tiled_label "A-in-place" ]

let site_labels s =
  [ "the builder reads the site's m x n x k off the nest"; "the reference is not all zeros" ]
  @ List.map legs ~f:agrees_label
  @ List.map legs ~f:(fun l -> tiled_label l.what)
  @ in_place_labels s

(* The k tail's contribution to output cell (row, col): the sum over the reduction indices past the
   last full k block. Host arithmetic over the bench's own operand values. *)
let k_tail_contribution s ~row ~col =
  let first = s.k / s.bk * s.bk in
  List.sum
    (module Float)
    (List.range first s.k)
    ~f:(fun kk -> Bench_gebp.a_value row kk *. Bench_gebp.b_value kk col)

let check_site ~idx ~prec s =
  let tag = Ir.Ops.prec_string prec in
  (* One routine name per case: [Context.compile ?name] names the build artifacts, and a name shared
     by every case would leave only the last case's code behind for diagnosing an earlier one. *)
  let name leg = Printf.sprintf "gpp_%d_%s_%s" idx tag leg in
  let lbl what = Printf.sprintf "%s, %s storage: %s" s.label tag what in
  let cprec = compute_prec prec in
  let tile_prec = if Ir.Ops.equal_prec cprec prec then None else Some cprec in
  let ma, mb = operands ~prec s in
  let want, _, _ = run ~prec ~name:(name "ref") ~ma ~mb ~schedule:None in
  let runs =
    List.map legs ~f:(fun l ->
        ( l,
          run ~prec ~name:(name l.leg) ~ma ~mb
            ~schedule:
              (Some
                 (packed ~layout:l.layout ~hoist_b:l.hoist_b ~stage_a:true ~tile_prec ~bm:s.bm
                    ~bk:s.bk ~ma ~mb)) ))
  in
  let ser, _, ser_site = List.Assoc.find_exn runs ~equal:phys_equal (List.hd_exn legs) in
  p
    (lbl "the builder reads the site's m x n x k off the nest")
    (match ser_site with
    | Some { Bench_gebp.ni; nj; nk; _ } -> ni = s.m && nj = s.n && nk = s.k
    | None -> false);
  p (lbl "the reference is not all zeros") (Array.exists want ~f:(fun x -> Float.(x <> 0.)));
  List.iter runs ~f:(fun (l, (got, _, _)) ->
      p_all2
        (lbl (agrees_label l))
        got
        (if String.equal l.leg "serial" then want else ser)
        ~f:Float.equal);
  List.iter runs ~f:(fun (l, (_, census, _)) ->
      p
        (lbl (tiled_label l.what))
        (Ll_test.all_rendered ~what:(lbl l.what) Ir.C_syntax.Mma_register_tiled census));
  let in_place ~stage_a =
    packed ~layout:Bench_gebp.Grid_outermost ~hoist_b:true ~stage_a ~tile_prec ~bm:s.bm ~bk:s.bk ~ma
      ~mb
  in
  let raised schedule =
    match run ~prec ~name:(name "inplace") ~ma ~mb ~schedule:(Some schedule) with
    | _ -> None
    (* The raw message: [Exn.to_string] escapes its non-ASCII bytes. *)
    | exception Invalid_argument msg -> Some msg
    | exception e -> Some (Exn.to_string e)
  in
  if padded s then begin
    let why = Option.value_exn (Bench_gebp.in_place_a_refusal ~bm:s.bm ~bk:s.bk ~ni:s.m ~nk:s.k) in
    p
      (lbl "with A read in place, the builder refuses to pad")
      (Option.exists
         (raised (in_place ~stage_a:false))
         ~f:(fun msg -> String.is_substring msg ~substring:why));
    (* The staged composition with A's [Stage] dropped: what the builder would emit without its
       refusal. *)
    let unstaged_a ~c opt =
      List.filter (in_place ~stage_a:true ~c opt) ~f:(function
        | Sched.Stage { source; _ } -> not (Ir.Tnode.equal source ma.Tensor.value)
        | _ -> true)
    in
    p
      (lbl "with A read in place, Tensorize itself refuses the padded composition")
      (Option.exists (raised unstaged_a) ~f:(fun msg ->
           String.is_substring msg ~substring:"Schedule.Tensorize"
           && String.is_substring msg ~substring:"zero-fringe staged tile"))
  end
  else begin
    let got, census, _ =
      run ~prec ~name:(name "inplace") ~ma ~mb ~schedule:(Some (in_place ~stage_a:false))
    in
    p_all2
      (lbl "the A-in-place packed leg matches the serial packed leg bitwise")
      got ser ~f:Float.equal;
    p
      (lbl (tiled_label "A-in-place"))
      (Ll_test.all_rendered ~what:(lbl "A-in-place") Ir.C_syntax.Mma_register_tiled census)
  end

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
  List.iteri sites ~f:(fun idx s ->
      edge_content s;
      List.iter [ Ir.Ops.single; Ir.Ops.bfloat16 ] ~f:(fun prec ->
          let labels =
            List.map (site_labels s)
              ~f:(Printf.sprintf "%s, %s storage: %s" s.label (Ir.Ops.prec_string prec))
          in
          (* The packed composition is the C backends' register-tiled GEBP ([Tensorize] at a unit
             lane width); a GPU backend has nothing of it to execute. *)
          if on_cpu then
            Verdict.case
              (Printf.sprintf "%s, %s storage" s.label (Ir.Ops.prec_string prec))
              (fun () -> check_site ~idx ~prec s)
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
          List.filter
            (packed ~layout:Bench_gebp.Serial ~hoist_b:false ~stage_a:true ~tile_prec:None ~bm:s.bm
               ~bk:s.bk ~ma ~mb ~c opt) ~f:(function
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
