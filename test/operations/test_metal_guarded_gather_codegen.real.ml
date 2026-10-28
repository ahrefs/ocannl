(* Regression coverage for the Metal side of gh-343: guarded dynamic gather uses [double] as a
   signed guard precision in the shared lowering, but Metal has no native double. The backend must
   keep true double storage unsupported while rendering this scalar guard as float casts. *)

open! Base
open Ocannl
open Operation.DSL_modules

let vocab = 4
let embed = 3

module Generated = Test_utils.Generated

let build_embedding id_values =
  let ids =
    TDSL.ndarray id_values ~label:[ "ids" ]
      ~batch_dims:[ Array.length id_values ]
      ~output_dims:[] ()
  in
  let table =
    TDSL.ndarray
      (Array.init (embed * vocab) ~f:Float.of_int)
      ~label:[ "C" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let classes = TDSL.range vocab in
  let%op one_hot = classes = ids in
  let%op embedded = table * one_hot in
  embedded

let () =
  Tensor.unsafe_reinitialize ();
  Utils.settings.output_debug_files_in_build_directory <- true;
  (* This run's own artifacts only, and a missing kernel a failure (gh-ocannl-1294). *)
  Generated.init ~backend_name:"metal";
  let ctx = Context.metal () in
  let embedded = build_embedding [| 1.; 3.; 0. |] in
  Train.set_materialized embedded.Tensor.value;
  let ctx, routine =
    Context.compile ~name:"mggc_embed" ctx (Train.forward embedded) Ir.Indexing.Empty
  in
  let (_ : Context.t) = Context.run ctx routine in
  let src = Generated.read "mggc_embed" in
  let has substring = String.is_substring src ~substring in
  Verdict.p "guarded gather source casts signed guard to float"
    (has ("? " ^ Generated.node_ident "C" ^ "[")
    && has ": (float)(0.0))"
    && has ("(float)(" ^ Generated.node_ident "ids" ^ "["));
  Verdict.p "guarded gather source contains no double declarations"
    ((not (String.is_empty src)) && not (has "double"))
