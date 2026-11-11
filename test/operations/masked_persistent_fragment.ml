(* gh-ocannl-1736: a pad-masked accumulator held in a persistent register fragment.

   A staged GPU pipeline whose block does not divide the row or column extent pad-masks the
   accumulator: [Schedule.contract_tensorized_accumulator] retargets it to a padded tile whose
   guarded transfers round-trip only the valid region. Until gh-ocannl-1736 that tile was the
   accumulator itself, in workgroup-shared memory, and every reduction step paid a load/store
   round-trip and a barrier pair. Now the loop-site contraction stages the masked transfers through
   that tile as a SCRATCH tile and wraps the reduction in an unmasked fragment over it, which the
   backend's fragment hook holds in registers across the whole reduction.

   Per graph — an f16 x f16 -> f32 matmul and conv whose extents no block divides — every GPU seed
   is applied, and split by what its contraction did: masked at the loop site (a recognized fragment
   region whose target is the masked scratch tile), masked at the statement site, or unmasked. The
   structural claims run everywhere, on a stand-in f16 matrix unit where the host has none:

   - some seed of each graph is masked at the loop site, and each such seed holds its fragment over
   a workgroup-shared scratch tile, with no fragment-scope violation; - a statement-site masked seed
   marks no persistent fragment; - the seeding's pad verdict ([Autotune.seeding_seed_pads], which
   the census prints) agrees with the applied schedule's [masked_fragments], seed by seed.

   On a GPU whose seeder advertises the f16 -> f32 tile, every loop-site seed is compiled, run twice
   and compared with the unscheduled reference bitwise; where the fragment scope is advertised its
   routine renders persistent-fragment scopes, and its emitted source holds exactly that many
   resident regions ([Ll_test.resident_regions]). A third graph, f16 x f16 -> f16 under the wide
   policy, is the A/B against the non-persistent path: its k-block partial sums leave f16, so the
   per-step narrowing of the former shared round-trip and the once-per-tile narrowing of the
   persistent fragment give different values, and the output must be the once-narrowed one wherever
   the scope rendered. *)

(* cc-fp-contract: not-cc -- executed claims need a GPU advertising an f16 tile; cc skips them *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module BI = Ir.Backend_intf
module Generated = Test_utils.Generated

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_gpu = BI.binds_hardware_grid (Context.backend_facts_of_name backend_name)
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let accum_prec = Ll_test.backend_accum_prec
let half = Ir.Ops.half
let single = Ir.Ops.single
let host_limits = Context.hardware_limits (Context.auto ())

(* A stand-in matrix unit for the structural claims where the host has none: an f16 x f16 -> f32
   8x8x8 tile, at no other hardware limit. *)
let standin_limits =
  {
    BI.no_hardware_limits with
    mma =
      Some
        {
          BI.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_f16_wide_acc_scopes = [ BI.Mma_per_statement; BI.Mma_fragment_scope ];
          mma_format_tiles =
            [
              ((BI.Mma_f16, BI.Mma_f16, BI.Mma_f32), (8, 8, 8));
              ((BI.Mma_f16, BI.Mma_f16, BI.Mma_f16), (8, 8, 8));
            ];
        };
  }

(* The seeder's own judgment that this GPU seeds a tensorized site of these precisions. *)
let gpu_advertises ~d_prec =
  on_gpu
  &&
  match
    Ll_test.tensorized_capability ~is_gpu:true ~is_cpu:false ~limits:host_limits ~a_prec:half
      ~b_prec:half ~d_prec
  with
  | `Advertised -> true
  | `Withheld _ -> false

let seed_limits ~d_prec = if gpu_advertises ~d_prec then host_limits else standin_limits

(* Whether the fragment scope is advertised for the triple: the seeder's scope judgment. *)
let scope_advertised ~d_prec =
  match host_limits.BI.mma with
  | Some mma ->
      Option.is_some
        (Autotune.mma_tile_for_precisions_in_scope mma ~scope:BI.Mma_fragment_scope ~a_prec:half
           ~b_prec:half ~d_prec)
  | None -> false

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Every persistent-fragment region of [opt]: [Low_level.fragment_region] at each statement
   sequence, its fragment marked in [simdgroup_fragments] (the renderer's own recognition). *)
let regions (opt : LL.optimized) =
  let acc = ref [] in
  let rec go (c : LL.t) =
    (match LL.fragment_region c with
    | Some r when Set.mem opt.LL.simdgroup_fragments r.LL.fr_fragment -> acc := r :: !acc
    | Some _ | None -> ());
    match c with
    | LL.Seq (a, b) ->
        go a;
        go b
    | For_loop { body; _ } | If { body; _ } | Scan_loop { body; _ } -> go body
    | _ -> ()
  in
  go opt.LL.llc;
  List.dedup_and_sort (List.rev !acc) ~compare:(fun a b ->
      Ir.Tnode.compare a.LL.fr_fragment b.LL.fr_fragment)

type contraction = Loop_site of LL.fragment_region | Statement_site | Unmasked

let classify (opt : LL.optimized) =
  if Set.is_empty opt.LL.masked_fragments then Unmasked
  else
    match List.find (regions opt) ~f:(fun r -> Set.mem opt.LL.masked_fragments r.LL.fr_target) with
    | Some r -> Loop_site r
    | None -> Statement_site

type seed = {
  label : string;
  sp : Autotune.sketch_params;
  pads : Autotune.seed_pad list option;
  applied : LL.optimized option;  (** [None]: the schedule did not apply. *)
}

let seeds_of ~d_prec (lowering : LL.optimized) =
  let limits = seed_limits ~d_prec in
  let family, sps = Autotune.family_seeding ~is_gpu:true ~is_cpu:false ~limits lowering in
  List.map sps ~f:(fun sp ->
      let label = Autotune.sketch_params_label sp in
      let applied =
        let o = Ll_test.hermetic_copy lowering in
        match Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp o) o with
        | applied -> Some applied
        | exception exn ->
            Stdio.eprintf "%s: did not apply (not part of the golden): %s\n" label
              (Exn.to_string exn);
            None
      in
      { label; sp; pads = Autotune.seeding_seed_pads family sp; applied })

let loop_site s =
  match Option.map s.applied ~f:classify with Some (Loop_site _) -> true | _ -> false

let statement_site s =
  match Option.map s.applied ~f:classify with Some Statement_site -> true | _ -> false

let masks_accumulator pads =
  List.exists pads ~f:(fun (sp : Autotune.seed_pad) -> Autotune.pad_masks_accumulator sp.pad_role)

let structural_claims tag (seeds : seed list) =
  Stdio.eprintf
    "%s: %d seeds, %d loop-site masked, %d statement-site masked (not part of the golden)\n" tag
    (List.length seeds) (List.count seeds ~f:loop_site)
    (List.count seeds ~f:statement_site);
  p_all (tag ^ ": every seed's schedule applies") seeds ~f:(fun s -> Option.is_some s.applied);
  p_exists (tag ^ ": some seed's masked accumulator contracts at the loop site") seeds ~f:loop_site;
  p_all
    (tag ^ ": every loop-site masked seed holds an unshared fragment over a shared scratch tile")
    (List.filter seeds ~f:loop_site) ~f:(fun s ->
      match s.applied with
      | Some opt -> (
          match classify opt with
          | Loop_site r ->
              Set.mem opt.LL.workgroup_shared r.LL.fr_target
              && (not (Set.mem opt.LL.workgroup_shared r.LL.fr_fragment))
              && (not (Set.mem opt.LL.masked_fragments r.LL.fr_fragment))
              && List.is_empty (Sched.fragment_scope_violations opt)
          | Statement_site | Unmasked -> false)
      | None -> false);
  (* A masked seed marks a persistent fragment exactly when its contraction landed at the loop site:
     the statement site keeps its single masked tile. *)
  p_all
    (tag ^ ": a masked seed marks a persistent fragment exactly at the loop site")
    (List.filter seeds ~f:(fun s -> loop_site s || statement_site s))
    ~f:(fun s ->
      Bool.equal (loop_site s)
        (Option.exists s.applied ~f:(fun o -> not (Set.is_empty o.LL.simdgroup_fragments))));
  p_all
    (tag ^ ": the seeding's pad verdict is the applied schedule's mask, seed by seed")
    (List.filter seeds ~f:(fun s -> Option.is_some s.pads && Option.is_some s.applied))
    ~f:(fun s ->
      let census = masks_accumulator (Option.value_exn s.pads) in
      let applied = not (Set.is_empty (Option.value_exn s.applied).LL.masked_fragments) in
      if Bool.(census <> applied) then
        Stdio.eprintf
          "%s: census pad verdict %s against the applied mask %s (not part of the golden)\n" s.label
          (Bool.to_string census) (Bool.to_string applied);
      Bool.equal census applied)

(* Compile [t] under [transform], run twice, read [t] back with the routine's mma census. *)
let run ~name ~transform (t : Tensor.t) =
  let ctx, routine =
    Context.compile ~lowered_transform:transform (Context.auto ())
      (named name (Train.forward t))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx t.Tensor.value in
  Context.release ctx;
  (got, routine.Context.mma)

type executed = {
  ex_seed : seed;
  got : float array;
  census : Ir.C_syntax.mma_summary;
  src : string;
}

let execute ?(select = loop_site) ~tag ~make (seeds : seed list) =
  List.filter_mapi seeds ~f:(fun i s ->
      if not (select s) then None
      else
        let name = Printf.sprintf "%s_s%d" tag i in
        Stdio.eprintf "%s = %s (not part of the golden)\n%!" name s.label;
        match
          run ~name
            ~transform:(fun opt ->
              [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:s.sp opt) opt ])
            (make name)
        with
        | got, census -> Some { ex_seed = s; got; census; src = Generated.read name }
        | exception exn ->
            Stdio.eprintf "%s: failed (not part of the golden): %s\n" name (Exn.to_string exn);
            Some { ex_seed = s; got = [||]; census = Ir.C_syntax.summarize_census []; src = "" })

let markers = Ll_test.residency_markers ~backend_name ()

let executed_claims ~leg ~tag ~d_prec ~want (runs : executed list Lazy.t) =
  in_leg ~leg (tag ^ ": every loop-site seed matches the reference bitwise, run twice")
    (fun label ->
      p_all label (Lazy.force runs) ~f:(fun r ->
          let ok =
            Array.length r.got = Array.length (Lazy.force want)
            && Array.for_all2_exn r.got (Lazy.force want) ~f:Float.equal
          in
          if not ok then Stdio.eprintf "%s: values differ from the reference\n" r.ex_seed.label;
          ok));
  let scope_leg =
    Verdict.sub_leg leg
      ~when_:(lazy (scope_advertised ~d_prec))
      ~on:(backend_name ^ " (no fragment scope)")
  in
  in_leg ~leg:scope_leg
    (tag ^ ": every loop-site seed renders persistent-fragment scopes, each resident in its source")
    (fun label ->
      p_all label (Lazy.force runs) ~f:(fun r ->
          let scopes = r.census.Ir.C_syntax.fragment_scopes in
          let resident = Ll_test.resident_regions r.src markers in
          let ok = scopes >= 1 && Option.equal Int.equal resident (Some scopes) in
          if not ok then
            Stdio.eprintf "%s: %d persistent-fragment scopes, %s resident regions in its source\n"
              r.ex_seed.label scopes
              (Option.value_map resident ~default:"no consistent" ~f:Int.to_string);
          ok))

(* --- The f16 x f16 -> f32 matmul: 40 x 40 x 40, which no 16- or 32-row block divides. The operands
   are small nonnegative integers, exact in f16. --- *)
let n = 40

let ma_f =
  Ll_test.operand ~b:[] ~o:[ n ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 7; 3 |] ~modulus:4 ~offset:1. ~stride:1.)

let mb_f =
  Ll_test.operand ~b:[] ~o:[ n ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 5; 11 |] ~modulus:7 ~offset:0. ~stride:1.)

let make_matmul ~d_prec tag =
  let a = NTDSL.init ~l:(tag ^ "a") ~prec:half ~i:[ n ] ~o:[ n ] ~f:ma_f () in
  let b = NTDSL.init ~l:(tag ^ "b") ~prec:half ~i:[ n ] ~o:[ n ] ~f:mb_f () in
  let%op c = a * b in
  Ir.Tnode.update_prec c.Tensor.value d_prec;
  c

(* --- The f16 x f16 -> f32 conv: 6 x 6 output over 8 in-channels, a 3x3 window, 20 out-channels,
   batch 2 -- neither the 6 rows nor the 20 out-channels a multiple of any backend's tile. --- *)
let rows = 6
let ic = 8
let oc = 20

let cx_f =
  Ll_test.operand ~b:[ 2 ]
    ~o:[ rows + 2; rows + 2; ic ]
    ~i:[]
    (Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:5 ~offset:0. ~stride:1.)

let ck_f =
  Ll_test.operand ~b:[] ~o:[ oc ] ~i:[ 3; 3; ic ]
    (Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:4 ~offset:0. ~stride:1.)

let make_conv tag =
  let x = NTDSL.init ~l:(tag ^ "x") ~prec:half ~b:[ 2 ] ~o:[ rows + 2; rows + 2; ic ] ~f:cx_f () in
  let kern = NTDSL.init ~l:(tag ^ "k") ~prec:half ~i:[ 3; 3; ic ] ~o:[ oc ] ~f:ck_f () in
  let%op pr =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  Ir.Tnode.update_prec pr.Tensor.value single;
  pr

(* --- The narrowing A/B: f16 x f16 -> f16 under the wide policy, whose fragment scope accumulates
   in f32 on every tensor-core backend (CUDA's inline-PTX arm, HIP's and Metal's converted
   boundary). The products are integers whose k-block partial sums pass 2048, where f16 stops
   holding every integer: the former shared round-trip narrowed the accumulator to f16 after every k
   block, the persistent fragment narrows it once per tile. Both references are computed on the host
   for each seed's own k block, exactly (integers below 2^24 in the f64 chain, rounded to f16 by
   integer arithmetic). --- *)
let nb_f =
  Ll_test.operand ~b:[] ~o:[ n ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 13; 5 |] ~modulus:61 ~offset:40. ~stride:1.)

let make_narrow tag =
  let a = NTDSL.init ~l:(tag ^ "a") ~prec:half ~i:[ n ] ~o:[ n ] ~f:ma_f () in
  let b = NTDSL.init ~l:(tag ^ "b") ~prec:half ~i:[ n ] ~o:[ n ] ~f:nb_f () in
  let%op c = a * b in
  Ir.Tnode.update_prec c.Tensor.value half;
  c

(* Round a nonnegative integer below 65504 to the nearest f16, ties to even. *)
let round16 x =
  let xi = Float.to_int x in
  if xi < 2048 then xi
  else
    let _, ex = Float.frexp x in
    let ulp = 1 lsl (ex - 1 - 10) in
    let q = xi / ulp and r = xi % ulp in
    let q = if 2 * r > ulp || (2 * r = ulp && q land 1 = 1) then q + 1 else q in
    q * ulp

(* The two narrowings of cell [(i, j)] at k block [bk]: once, and after every k block. *)
let narrowings ~bk =
  Array.init (n * n) ~f:(fun t ->
      let i = t / n and j = t % n in
      let term kk = Float.to_int (ma_f [| i; kk |] *. nb_f [| kk; j |]) in
      let total = List.sum (module Int) (List.range 0 n) ~f:term in
      let per_step =
        List.fold (List.range ~stride:bk 0 n) ~init:0 ~f:(fun acc k0 ->
            let partial = List.sum (module Int) (List.range k0 (Int.min n (k0 + bk))) ~f:term in
            round16 (Float.of_int (acc + partial)))
      in
      (round16 (Float.of_int total), per_step))

let wide f =
  Test_utils.with_policy (fun p -> { p with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_wide }) f

let narrow_claims () =
  let tag = "nw" in
  let lowering =
    wide (fun () ->
        Ll_test.captured_lowering ~name:(tag ^ "_probe") (Train.forward (make_narrow (tag ^ "p"))))
  in
  let seeds = wide (fun () -> seeds_of ~d_prec:half lowering) in
  structural_claims tag seeds;
  p_all
    (tag
   ^ ": for every loop-site seed's k block the data tell per-step narrowing from once-per-tile")
    (List.filter seeds ~f:loop_site) ~f:(fun s ->
      s.sp.Autotune.sk_bk > 0
      && Array.exists (narrowings ~bk:s.sp.Autotune.sk_bk) ~f:(fun (once, step) -> once <> step));
  let leg =
    Verdict.sub_leg
      (Verdict.leg
         ~when_:(wide (fun () -> gpu_advertises ~d_prec:half))
         ~on:(backend_name ^ " (no f16 tile)") ())
      ~when_:(lazy (wide (fun () -> scope_advertised ~d_prec:half)))
      ~on:(backend_name ^ " (no wide f16 fragment scope)")
  in
  (* Every masked seed, whatever its contraction did: the claim is on the values, so the former
     per-step round-trip fails it by value, not by an empty population. *)
  let masked s = Option.exists s.applied ~f:(fun o -> not (Set.is_empty o.LL.masked_fragments)) in
  in_leg ~leg
    (tag ^ ": every masked seed narrows its f32 accumulator once per tile, not per k block")
    (fun label ->
      let runs = wide (fun () -> execute ~select:masked ~tag ~make:make_narrow seeds) in
      p_all label runs ~f:(fun r ->
          let once = Array.map (narrowings ~bk:r.ex_seed.sp.Autotune.sk_bk) ~f:fst in
          let ok =
            Array.length r.got = Array.length once
            && Array.for_all2_exn r.got once ~f:(fun g o -> Float.equal g (Float.of_int o))
          in
          (if not ok then
             let step = Array.map (narrowings ~bk:r.ex_seed.sp.Autotune.sk_bk) ~f:snd in
             let differ ref_ =
               if Array.length r.got <> Array.length ref_ then "all"
               else
                 Int.to_string
                   (Array.counti r.got ~f:(fun i g -> not (Float.equal g (Float.of_int ref_.(i)))))
             in
             Stdio.eprintf
               "%s: not the once-narrowed values (%s cells differ from them, %s from the per-step \
                narrowing; %d persistent-fragment scopes)\n"
               r.ex_seed.label (differ once) (differ step) r.census.Ir.C_syntax.fragment_scopes);
          ok))

let reference ~make tag =
  lazy (fst (run ~name:(tag ^ "_ref") ~transform:(fun o -> [ o ]) (make (tag ^ "r"))))

let () =
  let f32_leg =
    Verdict.leg ~when_:(gpu_advertises ~d_prec:single) ~on:(backend_name ^ " (no f16->f32 tile)") ()
  in

  (* Every partial sum of both f32 reductions is exact whatever the association, so each correct
     rendering returns the reference's bits. *)
  let mm_products = Ll_test.matmul_products ~m:n ~n ~k:n ma_f mb_f in
  let cv_products =
    Ll_test.reduction_cells ~out:[| 2; rows; rows; oc |] ~over:[| 3; 3; ic |] (fun o r ->
        cx_f [| o.(0); o.(1) + r.(0); o.(2) + r.(1); r.(2) |]
        *. ck_f [| o.(3); r.(0); r.(1); r.(2) |])
  in
  List.iter
    [ ("mm", make_matmul ~d_prec:single, mm_products); ("cv", make_conv, cv_products) ]
    ~f:(fun (tag, make, products) ->
      p_all
        (tag ^ ": every partial sum is exact in f32")
        products
        ~f:(Ll_test.exact_partials ~sig_bits:(Ir.Ops.significand_bits single));
      let lowering =
        Ll_test.captured_lowering ~name:(tag ^ "_probe") (Train.forward (make (tag ^ "p")))
      in
      let seeds = seeds_of ~d_prec:single lowering in
      structural_claims tag seeds;
      let want = reference ~make tag in
      p_all2
        (tag ^ ": the executed reference equals the premise's host sums")
        (Lazy.force want)
        (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
        ~f:Float.equal;
      executed_claims ~leg:f32_leg ~tag ~d_prec:single ~want (lazy (execute ~tag ~make seeds)))

let () = narrow_claims ()
