(* gh-ocannl-1138, commit 4: the fine segmentation keeps a running composite of its own. A GPU
   search also seeds singles over the finer [arity_cuts] segmentation (gh-ocannl-574), which frees a
   GEMM from a companion that cannot follow its full arity -- here an lm_head-shaped GEMM followed
   by a row max, whose [-inf] initialization keeps the two in one coarse segment. With config
   [autotune_staffed_singles] those fine singles are swept against the fine segmentation's own
   composite, whose keys the finer cuts left unchanged carry the coarse composite's choices.

   The workload is two batched GEMMs, [x = x0 * v] and then [z = x * w], with [r] the row max of
   [z]: coarsely [x] alone and [z] with its row max; finely [x], [z] and [r] apart, so [x]'s key is
   shared and [z]'s is new. The ranking is scripted as in [autotune_staffed_singles]: a per-segment
   candidate measures the sum of its scheduled segments' costs plus a preset's cost for each keyed
   segment it leaves alone, every other candidate far slower. The claims: every fine single is
   staffed with the coarse composite's choice for the shared key, which the script decides; the
   crown is the fine composite, the fine sweep's own window; and the crowned routine computes what
   the untuned compile does, bit for bit (dyadic inputs, exact sums, and no tensor-core rounding of
   f32 operands by default).

   The finer segmentation is GPU-only, so on any other backend the search is not run and every claim
   is skipped. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Ir.Schedule.backend_is_gpu backend_name
let staffed_key = "autotune_staffed_singles"

(* A per-segment label's entries ([params @key]), its focus and its staff apart. *)
let parts label =
  let body =
    String.chop_prefix_exn label ~prefix:"F_sketch["
    |> String.chop_suffix_exn ~suffix:"]"
    |> String.chop_prefix_if_exists ~prefix:"fine "
  in
  let entries s = if String.is_empty s then [] else String.split s ~on:',' in
  let sep = " | staff " in
  match String.substr_index body ~pattern:sep with
  | Some i ->
      (entries (String.prefix body i), entries (String.drop_prefix body (i + String.length sep)))
  | None -> (entries body, [])

let is_sketch label = String.is_prefix label ~prefix:"F_sketch["
let is_fine label = String.is_prefix label ~prefix:"F_sketch[fine "
let key_of entry = List.last_exn (String.split entry ~on:'@')
let cost entry = 2. +. (Float.of_int (Stdlib.Hashtbl.hash entry % 1000) /. 100.)

(* Coarsely two keyed segments, finely two too ([r]'s max has no matmul site): an unscheduled keyed
   segment costs a preset's. *)
let routine_ms label =
  if is_sketch label then
    let f, s = parts label in
    let entries = f @ s in
    List.sum (module Float) entries ~f:cost +. (50. *. Float.of_int (2 - List.length entries))
  else 1000.

let b = 4
let n = 32
let k = 16
let m = 64

let constant label ~dims ~modulus ~offset ~stride ~batch_dims ~output_dims =
  TDSL.ndarray
    (Array.init
       (Array.fold dims ~init:1 ~f:( * ))
       ~f:(Ll_test.cycle_flat ~dims ~modulus ~offset ~stride))
    ~label:[ label ] ~batch_dims ~output_dims ()

let workload () =
  let x0 =
    constant "sf_x0" ~dims:[| b; n; k |] ~modulus:7 ~offset:0. ~stride:0.25 ~batch_dims:[ b ]
      ~output_dims:[ n; k ]
  in
  let v =
    constant "sf_v" ~dims:[| k; k |] ~modulus:3 ~offset:(-0.5) ~stride:0.5 ~batch_dims:[]
      ~output_dims:[ k; k ]
  in
  let w =
    constant "sf_w" ~dims:[| k; m |] ~modulus:5 ~offset:0. ~stride:0.125 ~batch_dims:[]
      ~output_dims:[ k; m ]
  in
  let%op x = x0 +* "b|ik;kl=>b|il" v in
  Train.set_materialized x.Tensor.value;
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op r = z @^^ "b|ij => b|i" in
  let comp = Train.forward r in
  (z, r, { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment ("sf_head", comp.asgns) })

let values ctx routine (z, r) =
  let ctx = Context.run ctx routine in
  let got = (Context.get_values ctx z.Tensor.value, Context.get_values ctx r.Tensor.value) in
  Context.release ctx;
  got

let () =
  let measured = ref [] and report = ref None in
  let crowned, untuned =
    if not is_gpu then (([||], [||]), ([||], [||]))
    else
      let z, r, comp = workload () in
      let untuned =
        let ctx, routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
        values ctx routine (z, r)
      in
      let crowned =
        Exn.protect
          ~finally:(fun () -> Hashtbl.remove Utils.config_file_args staffed_key)
          ~f:(fun () ->
            Hashtbl.set Utils.config_file_args ~key:staffed_key ~data:"true";
            Autotune.Seam.with_value Autotune.on_candidate_measured (fun ~label ~digest:_ _ms ->
                measured := label :: !measured;
                routine_ms label)
            @@ fun () ->
            let ctx, routine =
              Autotune.with_uncontended_test_windows (fun () ->
                  Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log:false
                    ~progress:false
                    ~report:(fun r -> report := Some r)
                    (Context.auto ()) comp Ir.Indexing.Empty)
            in
            values ctx routine (z, r))
      in
      (crowned, untuned)
  in
  let measured = List.rev !measured in
  let fine_singles = List.filter measured ~f:is_fine in
  let coarse_singles = List.filter measured ~f:(fun l -> is_sketch l && not (is_fine l)) in
  (* The shared key: the one both segmentations' singles focus on. *)
  let focus_keys singles =
    List.dedup_and_sort ~compare:String.compare
      (List.map singles ~f:(fun l -> key_of (List.hd_exn (fst (parts l)))))
  in
  let shared =
    List.filter (focus_keys coarse_singles) ~f:(fun k ->
        List.exists fine_singles ~f:(fun l ->
            List.exists (snd (parts l)) ~f:(fun e -> String.equal (key_of e) k)))
  in
  (* The coarse composite's choice for the shared key: its fastest single in its coarse sweep. *)
  let coarse_choice =
    List.filter_map coarse_singles ~f:(fun l ->
        match fst (parts l) with
        | [ e ] when List.mem shared (key_of e) ~equal:String.equal -> Some e
        | _ -> None)
    |> List.min_elt ~compare:(fun a b -> Float.compare (cost a) (cost b))
  in
  Option.iter !report ~f:(fun r ->
      Stdio.eprintf
        "staffed fine (not part of the golden): %d coarse and %d fine singles, crown %s %.2f ms\n%!"
        (List.length coarse_singles) (List.length fine_singles) r.Autotune.best_label
        r.Autotune.best_ms);
  gated ~when_:is_gpu ~on:backend_name
    "fine: the finer cuts freed a key the coarse ones share a key with"
    ((not (List.is_empty fine_singles)) && List.length shared = 1);
  gated_all ~when_:is_gpu ~on:backend_name
    "fine: every fine single is staffed with the coarse composite's choice for the shared key"
    fine_singles ~f:(fun l ->
      Option.exists coarse_choice ~f:(fun c -> List.equal String.equal (snd (parts l)) [ c ]));
  gated ~when_:is_gpu ~on:backend_name
    "fine: the crown is a staffed fine single, the fine composite"
    (Option.exists !report ~f:(fun r ->
         is_fine r.Autotune.best_label && not (List.is_empty (snd (parts r.Autotune.best_label)))));
  gated ~when_:is_gpu ~on:backend_name
    "fine: the crowned routine computes the untuned output bit for bit"
    (Array.equal Float.equal (fst crowned) (fst untuned)
    && Array.equal Float.equal (snd crowned) (snd untuned)
    && Array.length (fst crowned) = b * n * m)
