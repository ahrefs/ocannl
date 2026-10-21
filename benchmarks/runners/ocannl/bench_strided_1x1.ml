(* gh-ocannl-1397: the strided 1x1 conv's GEMM row, measured at the site. A stride-2 1x1 conv
   ([conv2d ~kernel_size:1 ~stride:2 ~use_padding:true], cifar_resnet's [s<i>sc] shortcut and
   [resnet_block]'s downsample) reads its input at [(2*oh, 2*ow)]. Its GEMM row is either the batch
   axis [b], read at unit stride, or the strided minor output axis [ow], which only a compacting
   [Stage] of the input tiles (gh-ocannl-1148). The matmul classifier
   ([Sketch_families.classify_matmul]) chooses between them, and this runner times what the choice
   ships: the forward conv alone, without a fixture or a model around it, so a build on each side of
   a classifier change makes the A/B.

   Usage: bench_strided_1x1.exe SITE... [ocannl configuration flags], each SITE as
   [name=b,h,w,ic,oc]: a [b]-image [h x w x ic] input, [oc] out-channels, at stride 2 (the output is
   [h/2 x w/2]). BENCH_REPEATS is the number of synced runs a shipped routine is timed over, min
   taken (default 50). BENCH_TUNE=0 skips the tuned arm (the census and the untuned arm only).

   Prints, per site, one [s1x1-site:] line -- the classifier's roles (the row's extent and stride,
   the batch loops outside and inside it) and the whole-routine seed count under the device's own
   limits -- then one [s1x1:] line per arm: [default] (the untuned compile,
   [Bench_harness.compile_default], so it honours [model_default_schedule]) and [tuned]
   ([Autotune.tune] from an empty schedule cache, with its crowned label and counters). Each arm's
   [ms] is the min-of-repeats synced time of the shipped routine. Inputs are small integers, so
   every product and sum is exact in f32 whatever the schedule: [exact] is the shipped routine's
   output against a host oracle reading the input at the stride, and the runner exits 1 when an arm
   disagrees. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let parse_site s =
  match String.lsplit2 s ~on:'=' with
  | Some (name, dims) -> (
      match List.map (String.split dims ~on:',') ~f:Int.of_string with
      | [ b; h; w; ic; oc ] when b >= 1 && h >= 2 && w >= 2 && ic >= 1 && oc >= 1 ->
          (name, b, h, w, ic, oc)
      | _ -> failwith ("site dims must be b,h,w,ic,oc (positive, h and w at least 2): " ^ s))
  | None -> failwith ("site must be name=b,h,w,ic,oc: " ^ s)

(* Values 1..7 and 1..5, varying with every index (each weight a unit modulo its modulus), so a
   unit-stride misread of the input changes the result and no output cell is zero. *)
let x_val b y x c = Float.of_int (1 + (((b * 1) + (y * 2) + (x * 3) + (c * 5)) % 7))
let k_val o c = Float.of_int (1 + (((o * 2) + (c * 3)) % 5))

(* [Nn_blocks.conv2d ~kernel_size:1 ~stride:2]'s padded-mode spec. *)
let spec = "... | 2*oh=+kh, 2*ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."

let build (name, b, h, w, ic, oc) =
  (* The padding-aware [reshape] constructor, as an upstream layer's output would arrive: [init]
     commits an unpadded layout, which the padded spec rejects. *)
  let x =
    NTDSL.reshape ~l:(name ^ "_x") ~b:[ b ] ~o:[ h; w; ic ]
      (Ir.Ndarray.init_array ~debug:(name ^ "_x") Ir.Ops.single ~dims:[| b; h; w; ic |]
         ~padding:None ~f:(fun idx -> x_val idx.(0) idx.(1) idx.(2) idx.(3)))
      ()
  in
  let k =
    NTDSL.init ~l:(name ^ "_k") ~prec:Ir.Ops.single ~i:[ 1; 1; ic ] ~o:[ oc ]
      ~f:(fun idx -> k_val idx.(0) idx.(3))
      ()
  in
  let y = NTDSL.O.einsum ~label:[ name ] spec x k in
  ( y,
    { (Train.forward y) with asgns = Ir.Assignments.Block_comment (name, (Train.forward y).asgns) }
  )

let out_extent n = ((n - 1) / 2) + 1

let oracle (_, b, h, w, ic, oc) =
  let oh = out_extent h and ow = out_extent w in
  Array.init
    (b * oh * ow * oc)
    ~f:(fun cell ->
      let o = cell % oc
      and j = cell / oc % ow
      and i = cell / (oc * ow) % oh
      and bi = cell / (oc * ow * oh) in
      List.sum
        (module Float)
        (List.range 0 ic)
        ~f:(fun c -> x_val bi (2 * i) (2 * j) c *. k_val o c))

let time_routine ~repeats ctx routine =
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let best = ref Float.infinity in
  for _ = 1 to repeats do
    let t0 = Unix.gettimeofday () in
    let ctx = Context.run ctx routine in
    Context.sync ctx;
    best := Float.min !best ((Unix.gettimeofday () -. t0) *. 1000.)
  done;
  !best

let bench_site ~failed ~repeats ~tune ((name, b, h, w, ic, oc) as site) =
  let want = oracle site in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let is_gpu = Ir.Schedule.backend_is_gpu backend in
  (* The classifier's view, on a lowering the census compile captures and discards. *)
  let census =
    let _, comp = build site in
    let found = ref "no matmul site" in
    (match
       Context.compile_outcome
         ~lowered_transform:(fun opt ->
           (match Autotune.detect_matmul opt.Ir.Low_level.llc with
           | None -> ()
           | Some m ->
               let loops l =
                 String.concat ~sep:","
                   (List.map l ~f:(fun (s, n) ->
                        Printf.sprintf "%s:%d" (Ir.Indexing.symbol_ident s) n))
               in
               found :=
                 Printf.sprintf
                   "row=%s:%d row_stride=%d nj=%d nk=%d batch_out=[%s] batch_in=[%s] seeds=%d"
                   (Ir.Indexing.symbol_ident m.Autotune.m_i)
                   m.Autotune.m_ni m.Autotune.m_row_stride m.Autotune.m_nj m.Autotune.m_nk
                   (loops m.Autotune.m_bo) (loops m.Autotune.m_bi)
                   (List.length
                      (Autotune.sketch_seed_params ~is_gpu ~is_cpu:(not is_gpu) ~limits opt)));
           [ opt ])
         ~provenance:Ir.Schedule_outcome.User_schedule ctx comp Ir.Indexing.Empty
     with
    | Ok (c, _) -> Context.release c
    | Error _ -> ());
    !found
  in
  Stdio.printf "s1x1-site: backend=%s site=%s b=%d h=%d w=%d ic=%d oc=%d %s\n%!" backend name b h w
    ic oc census;
  let arm label compile =
    let y, comp = build site in
    let report = ref None in
    let t0 = Unix.gettimeofday () in
    let ctx, routine = compile ~report ctx comp in
    let compile_s = Unix.gettimeofday () -. t0 in
    let best = time_routine ~repeats ctx routine in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx y.Tensor.value in
    let exact =
      Array.length got = Array.length want && Array.for_all2_exn got want ~f:Float.equal
    in
    if not exact then failed := true;
    let tuned =
      match !report with
      | None -> ""
      | Some r ->
          Printf.sprintf
            " outcome=%s crowned=%s best_ms=%.4f sketch_candidates=%d timed=%d failed=%d"
            (Autotune.outcome_name r.Autotune.outcome)
            r.Autotune.best_label r.Autotune.best_ms r.Autotune.sketch_candidates
            r.Autotune.candidates_timed r.Autotune.candidates_failed
    in
    Stdio.printf "s1x1: backend=%s site=%s arm=%s compile_s=%.2f ms=%.4f exact=%b%s\n%!" backend
      name label compile_s best exact tuned;
    Context.release ctx
  in
  arm "default" (fun ~report:_ ctx comp -> Bench_harness.compile_default ctx comp Ir.Indexing.Empty);
  if tune then
    arm "tuned" (fun ~report ctx comp ->
        Autotune.tune ~name ~search:true ~cache_dir:""
          ~report:(fun r -> report := Some r)
          ctx comp Ir.Indexing.Empty)

let () =
  let sites =
    List.filter_map
      (List.tl_exn (Array.to_list (Sys.get_argv ())))
      ~f:(fun arg ->
        if not (String.is_prefix arg ~prefix:"-") then Some (parse_site arg)
        else if Utils.cmdline_arg_is_config_key arg then None
        else
          (* A dash argument no configuration key claims is a typo, not a setting. *)
          failwith ("bench_strided_1x1: unknown flag " ^ arg ^ " (only ocannl configuration flags)"))
  in
  if List.is_empty sites then
    failwith "usage: bench_strided_1x1.exe name=b,h,w,ic,oc ... [ocannl configuration flags]";
  let repeats =
    Option.value_map (Stdlib.Sys.getenv_opt "BENCH_REPEATS") ~default:50 ~f:Int.of_string
  in
  let tune = not (Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_TUNE") (Some "0")) in
  let failed = ref false in
  List.iter sites ~f:(bench_site ~failed ~repeats ~tune);
  if !failed then Stdlib.exit 1
