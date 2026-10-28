(* gh-ocannl-1664: a search that seeds the seeded untuned plan times its per-segment singles inside
   that plan.

   On CUDA and HIP the search's first candidate is the seeded untuned plan ([F_seeded],
   gh-ocannl-1388). Its per-segment singles used to be timed over the search's own segmentation with
   every other segment on its preset: a backdrop several times the plan's time, so a single was
   comparable only with its own key's singles (gh-ocannl-1528). With config
   [autotune_seeded_singles] (the default) each single is instead the plan with one segment code's
   choice swapped for a sketch ([F_swap], {!Autotune.seeded_swap}), over the plan's own segmentation
   ({!Autotune.seeded_single_seeds}), so it times on the plan's basis.

   Two parts. The structural part runs on every backend: it derives a plan for "cuda" on synthetic
   limits from this fixture's lowering, enumerates the seeded singles' menu on it, and claims the
   menu's keys are the plan's own segments (a key's digest is the one the plan decided that segment
   on) and that a swap changes the plan's decision on the key's segments and nowhere else -- the
   split prelude, the segmentation and every other decision stay the plan's.

   The search part runs where the untuned compile seeds ([Autotune.seeded_default_backend]): a
   scripted search over the same fixture, the plan's windows at 10 ms, the first seeded single's at
   9 ms and every other single's slower than the plan. Its claims are the timing basis, read off the
   digests the search keys candidates by (one structural digest per compiled segment): every seeded
   single compiles to the plan's segment count, a single swapped alone differs from the plan in at
   most its key's segments, and the singles over the search's own segmentation are not seeded. Then
   the sweep: the 9 ms single joins the running seeded composite, every single of a later key
   carries it as its staff, and the search crowns it. Elsewhere those claims are skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The seeding gate pinned at threshold 32, as in [autotune_seeded_seed]: the ranked [auto] gate
   seeds nothing on this fixture on HIP, and every claim needs a plan that seeds. *)
let () = Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "32"
let seeds_here = Autotune.seeded_default_backend backend_name

(* [autotune_seeded_seed]'s fixture: one dense layer with a bias, backpropagated, whose weight
   gradient (4096 rows) and forward product (96) both reach the seeding threshold. *)
let nb = 1024
and ns = 4
and no = 64
and ni = 96

let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"sw_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"sw_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x =
  NTDSL.ndarray
    (values ~n:(nb * ns * ni) ~stride:389)
    ~label:[ "sw_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()

let dy =
  NTDSL.ndarray
    (values ~n:(nb * ns * no) ~stride:577)
    ~label:[ "sw_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let () =
  Train.set_materialized (grad_of w);
  Train.set_materialized (grad_of b)

let comp = Train.grad_update loss
let static_indices = Ir.Indexing.bound_symbols Ir.Indexing.Empty

(* === The structural part: the menu and the swap, on a plan derived for "cuda". === *)

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The routine's lowering as a compile's transform receives it. *)
let lowering () =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile ~name:"sw_lowering"
      ~lowered_transform:(fun opt ->
        captured := Some (scratch opt);
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

let limits = Ir.Backend_intf.no_hardware_limits

(* Each segment's first unit in a segmentation, by position. *)
let starts segmentation = List.folding_map segmentation ~init:0 ~f:(fun at (_, n) -> (at + n, at))

let decision_digest = function
  | Autotune.Seeded_sketch { digest; _ } | Autotune.Seeded_sunk { digest; _ } -> Some digest
  | Autotune.Seeded_preset -> None

let () =
  let opt = lowering () in
  let plan =
    Autotune.seeded_plan ~min_reduction:1 ~backend:"cuda" ~limits ~accum_prec:Fn.id ~static_indices
      opt
  in
  match plan with
  | Some ({ Autotune.sp_segments = Some (segmentation, decisions); _ } as plan) ->
      p "a plan derived for cuda has a segmentation of its own" true;
      let seeds =
        Autotune.seeded_single_seeds ~capabilities:Ir.Backend_intf.no_codegen_capabilities
          ~is_gpu:true ~is_cpu:false ~limits ~static_indices plan (scratch opt)
      in
      let at = starts segmentation in
      let decision_at unit =
        List.find_map (List.zip_exn at decisions) ~f:(fun (u, d) ->
            if u = unit then Some d else None)
      in
      (* Every (key, unit) pair of the menu, with the plan's decision on that segment. *)
      let keyed =
        List.concat_map seeds ~f:(fun (key, _, _) ->
            List.map key.Autotune.key_units ~f:(fun u -> (key, u, decision_at u)))
      in
      p_all "every unit of a menu key starts a segment of the plan's segmentation" keyed
        ~f:(fun (_, _, d) -> Option.is_some d);
      p_exists "the menu keys a segment the plan decided on its code" keyed ~f:(fun (_, _, d) ->
          Option.exists d ~f:(fun d -> Option.is_some (decision_digest d)));
      p_all "a menu key's digest is the one the plan decided its segment on" keyed
        ~f:(fun (key, _, d) ->
          match Option.bind d ~f:decision_digest with
          | None -> true
          | Some digest -> String.equal digest key.Autotune.key_digest);
      let swaps =
        List.concat_map seeds ~f:(fun (key, _, params) ->
            List.map params ~f:(fun params -> (key, params)))
      in
      p_all "a swap keeps the plan's split prelude, segmentation and admission" swaps
        ~f:(fun choice ->
          let swapped = Autotune.seeded_swap plan [ choice ] in
          phys_equal swapped.sp_split plan.sp_split
          && Bool.equal swapped.sp_admit_narrow plan.sp_admit_narrow
          &&
          match swapped.sp_segments with
          | Some (s, _) -> phys_equal s segmentation
          | None -> false);
      p_all "a swap decides exactly its key's segments, on the key's digest, and no other" swaps
        ~f:(fun ((key, params) as choice) ->
          match (Autotune.seeded_swap plan [ choice ]).sp_segments with
          | None -> false
          | Some (_, swapped) ->
              List.for_all
                (List.zip_exn at (List.zip_exn decisions swapped))
                ~f:(fun (u, (before, after)) ->
                  if List.mem key.key_units u ~equal:Int.equal then
                    match after with
                    | Autotune.Seeded_sketch { digest; params = q } ->
                        String.equal digest key.key_digest && phys_equal q params
                    | Autotune.Seeded_preset | Autotune.Seeded_sunk _ -> false
                  else phys_equal before after));
      p "an empty swap is the plan itself"
        (String.equal
           (Autotune.seeded_plan_digest (Autotune.seeded_swap plan []))
           (Autotune.seeded_plan_digest plan));
      p_all "a seeded single's label names the plan and does not read as the plan's" swaps
        ~f:(fun choice ->
          let label = Autotune.seeded_swap_label plan [ choice ] in
          String.is_prefix label ~prefix:"F_swap["
          && String.is_substring label
               ~substring:("; plan " ^ String.prefix (Autotune.seeded_plan_digest plan) 8 ^ "]")
          && not (String.is_prefix label ~prefix:"F_seeded"))
  | Some { sp_segments = None; _ } | None ->
      p "a plan derived for cuda has a segmentation of its own" false

(* === The search part: a scripted search where the untuned compile seeds. === *)

let is_seeded label = String.is_prefix label ~prefix:"F_seeded["
let is_swap label = String.is_prefix label ~prefix:"F_swap["
let segments digest = String.split digest ~on:'+'

(* A seeded single's focus -- what precedes its staff or plan -- and its staff, as labelled. *)
let focus_and_staff label =
  let body = String.chop_prefix_exn label ~prefix:"F_swap[" in
  let body = fst (Option.value_exn ~here:[%here] (String.lsplit2 body ~on:';')) in
  match String.substr_index body ~pattern:" | staff " with
  | None -> (body, None)
  | Some i -> (String.prefix body i, Some (String.drop_prefix body (i + String.length " | staff ")))

(* The units of a focus' key tag: [<sketch> @<digest tag>:u3+u7] has two. *)
let focus_units focus =
  match String.rsplit2 focus ~on:':' with
  | Some (_, units) -> List.length (String.split units ~on:'+')
  | None -> 0

let differing a b =
  List.count (List.zip_exn (segments a) (segments b)) ~f:(fun (x, y) -> not (String.equal x y))

let () =
  let first_swap = ref None in
  let measure (w : Scripted_search.window) =
    if is_seeded w.label then 10.
    else if is_swap w.label then (
      if Option.is_none !first_swap then first_swap := Some w.label;
      if Option.exists !first_swap ~f:(String.equal w.label) then 9.
      else 11. +. (Float.of_int w.ordinal /. 1000.))
    else 50. +. (Float.of_int w.ordinal /. 1000.)
  in
  (* Each timing call that returns, under the attempt it timed: a window the host refuses still
     returns one but never reaches the script, so the report's window count is read against
     these. *)
  let latest = ref None and timed = ref [] in
  let search () =
    let report = ref None in
    let _routine, record =
      Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label -> latest := Some label)
      @@ fun () ->
      Autotune.with_timing_call_observer (fun (_ : Autotune.timing_call) ->
          Option.iter !latest ~f:(fun label -> timed := label :: !timed))
      @@ fun () ->
      (* The beam finals off, so every timing call after the base compile is a candidate's own. *)
      Scripted_search.scripted
        ~settings:[ ("autotune_finals_margin", "0") ]
        ~measure
        (fun () ->
          Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
            ~cache_dir:"" ~log:false
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Ir.Indexing.Empty)
    in
    (Option.value_exn ~here:[%here] !report, record.Scripted_search.windows)
  in
  let searched = if seeds_here then Some (search ()) else None in
  let searched_claim label f =
    Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
        match searched with
        | None -> p (label ^ " (no search ran)") false
        | Some (report, windows) ->
            let windows =
              List.map windows ~f:(fun (w, ms) -> (w.Scripted_search.label, w.digest, ms))
            in
            f report windows)
  in
  let swaps windows = List.filter windows ~f:(fun (label, _, _) -> is_swap label) in
  let plan_digest windows =
    List.find_map windows ~f:(fun (label, digest, _) ->
        if is_seeded label then Some digest else None)
  in
  Option.iter searched ~f:(fun ((r : Autotune.report), windows) ->
      Stdio.eprintf
        "scripted search (not part of the golden): %d windows, %d seeded singles proposed, %d \
         timed, %d joined, %d refused in all\n\
         %!"
        (List.length windows) r.seeded_single_candidates r.seeded_single_timed r.seeded_single_joins
        r.timings_contended);
  searched_claim "the search seeds seeded singles, and no single over its own segmentation"
    (fun r _ ->
      p "the search seeds seeded singles, and no single over its own segmentation"
        (r.Autotune.seeded_single_candidates > 0 && r.fiss_sketch_candidates = 0));
  searched_claim "no window is a single over the search's own segmentation" (fun _ windows ->
      p_none "no window is a single over the search's own segmentation" windows
        ~f:(fun (label, _, _) -> String.is_prefix label ~prefix:"F_sketch["));
  searched_claim "the report counts every seeded single's window" (fun r windows ->
      p "the report counts every seeded single's window"
        (r.Autotune.seeded_single_timed = List.count !timed ~f:is_swap
        && List.count !timed ~f:is_swap >= List.length (swaps windows)
        && not (List.is_empty (swaps windows))));
  searched_claim "every seeded single compiles to the plan's segment count" (fun _ windows ->
      match plan_digest windows with
      | None -> p "every seeded single compiles to the plan's segment count (no plan window)" false
      | Some plan ->
          p_all "every seeded single compiles to the plan's segment count" (swaps windows)
            ~f:(fun (_, digest, _) -> List.length (segments digest) = List.length (segments plan)));
  searched_claim "a single swapped alone differs from the plan in at most its key's segments"
    (fun _ windows ->
      match plan_digest windows with
      | None -> p "a single swapped alone differs from the plan (no plan window)" false
      | Some plan ->
          p_all "a single swapped alone differs from the plan in at most its key's segments"
            (List.filter (swaps windows) ~f:(fun (label, _, _) ->
                 Option.is_none (snd (focus_and_staff label))))
            ~f:(fun (label, digest, _) ->
              let d = differing plan digest in
              d >= 1 && d <= focus_units (fst (focus_and_staff label))));
  searched_claim "the faster single joined the running seeded composite and was crowned"
    (fun r windows ->
      p "the faster single joined the running seeded composite and was crowned"
        (r.Autotune.seeded_single_joins = 1
        && Option.exists !first_swap ~f:(String.equal r.best_label)
        && Float.equal r.best_ms 9.
        && Option.exists
             (List.hd (swaps windows))
             ~f:(fun (label, _, _) -> Option.exists !first_swap ~f:(String.equal label))));
  (* The joined key's focus tag: the part after its sketch's [" @"]. *)
  let key_of focus = Option.map (String.rsplit2 focus ~on:'@') ~f:snd in
  searched_claim "every later key's single carries the joined single as its staff" (fun _ windows ->
      let joined = Option.map !first_swap ~f:(fun l -> fst (focus_and_staff l)) in
      let joined_key = Option.bind joined ~f:key_of in
      let after_join =
        List.drop_while (swaps windows) ~f:(fun (label, _, _) ->
            not (Option.exists !first_swap ~f:(String.equal label)))
      in
      let later =
        List.filter after_join ~f:(fun (label, _, _) ->
            not (Option.equal String.equal (key_of (fst (focus_and_staff label))) joined_key))
      in
      (* The fixture keys at least two segments, so some key is swept after the join. *)
      p_all "every later key's single carries the joined single as its staff" later
        ~f:(fun (label, _, _) -> Option.equal String.equal (snd (focus_and_staff label)) joined))
