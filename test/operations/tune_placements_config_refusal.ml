(* gh-ocannl-1430 (raised on gh-ocannl-1331): a malformed registered setting reaches
   [Train.tune_placements]' caller typed -- [Utils.Invalid_config_value], naming the key -- rather
   than being absorbed by the containment that keeps one arm's failure from destroying the other's
   result (gh-ocannl-550). Contained, it would ship whichever placement happened not to read the
   key, as though the setting had been honored.

   Two legs. The first states a real key unparsable ([autotune_round_expansion], which every search
   reads before it compiles anything once it has a round to expand), so both arms refuse it; that
   pins the type and the key at the caller, but not the containment's exemption, since with both
   arms failed the call propagates the first failure anyway. The second is the leg that does: arm A
   completes, and the refusal arrives at arm B's first attempt (injected through
   [Autotune.on_candidate_attempt], as [autotune_arm_containment] injects its failures) -- with arm
   A's winner in hand, only the exemption keeps the call from shipping it. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let n = 8
let expansion_key = "autotune_round_expansion"

(* [settings] at the config-file level for the scope of [f] alone, and whether they took effect: the
   commandline and the environment outrank the file, and a claim drawn from a statement that did not
   take effect would claim nothing. *)
let with_settings settings f =
  let saved =
    List.map settings ~f:(fun (key, _) -> (key, Hashtbl.find Utils.config_file_args key))
  in
  Exn.protect
    ~finally:(fun () ->
      List.iter saved ~f:(fun (key, old) ->
          match old with
          | Some data -> Hashtbl.set Utils.config_file_args ~key ~data
          | None -> Hashtbl.remove Utils.config_file_args key))
    ~f:(fun () ->
      List.iter settings ~f:(fun (key, data) -> Hashtbl.set Utils.config_file_args ~key ~data);
      let took_effect =
        List.for_all settings ~f:(fun (key, data) ->
            Option.equal String.equal (Option.map (Utils.profile_key_source key) ~f:fst) (Some data))
      in
      (took_effect, f ()))

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "tpc_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tpc_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  let comp = Train.forward t2 in
  let tune ?report ~rounds () =
    Train.tune_placements ~beam_width:2 ~rounds ~repeats:1 ~cache_dir:"" ~placement_store:false
      ~log:false ~progress:false ?report (Context.cpu ()) t2 comp Ir.Indexing.Empty
  in
  let outcome f =
    match f () with
    | ctx, _routine ->
        Context.release ctx;
        `Shipped
    | exception Utils.Invalid_config_value { key; _ } -> `Refused key
  in
  (* --- Leg 1: a real key, stated unparsable. A search reads it when it has a round to expand,
     before it compiles anything. --- *)
  let took_effect, refused =
    with_settings [ (expansion_key, "widest") ] (fun () -> outcome (fun () -> tune ~rounds:1 ()))
  in
  p "real key: autotune_round_expansion=widest is stated for the call" took_effect;
  p "real key: the caller gets Utils.Invalid_config_value naming autotune_round_expansion"
    (match refused with `Refused key -> String.equal key expansion_key | `Shipped -> false);
  (* --- Leg 2: arm A completes; the refusal arrives at arm B's first attempt. --- *)
  let arms_reported = ref 0 and injected = ref false in
  let refused =
    Autotune.Seam.with_value Autotune.on_candidate_attempt
      (fun _label ->
        (* Arm A reports exactly once, when its search ends, so this fires within arm B only. *)
        if !arms_reported >= 1 && not !injected then (
          injected := true;
          raise
            (Utils.Invalid_config_value
               { key = expansion_key; value = "widest"; expected = "full, head or differing" })))
      (fun () -> outcome (fun () -> tune ~report:(fun _ -> Int.incr arms_reported) ~rounds:0 ()))
  in
  p "arm B: the refusal was raised at its first attempt" !injected;
  p "arm B: arm A had completed and reported before it" (!arms_reported >= 1);
  p "arm B: the caller gets Utils.Invalid_config_value naming the key, not arm A's winner"
    (match refused with `Refused key -> String.equal key expansion_key | `Shipped -> false)
