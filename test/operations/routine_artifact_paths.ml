(* gh-ocannl-1307: the same long labels appear in the routine and its source, while the artifact
   filenames leave room for the enclosing directory under Windows' MAX_PATH. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Generated = Test_utils.Generated

let () =
  let ctx = Context.auto () in
  Generated.init ~backend_name:(Context.backend_name ctx);
  Utils.settings.output_debug_files_in_build_directory <- true;
  let first = "gh1307_" ^ String.make 300 'k' ^ "_first" in
  let second = "gh1307_" ^ String.make 700 'k' ^ "_second" in
  let names = [ first; second ] in
  let extensions = [ Generated.extension (); ".cd"; ".ll"; "-unoptimized.ll"; "-debug.cd" ] in
  p_all "long labels of different lengths have distinct artifact paths of equal length" extensions
    ~f:(fun ext ->
      let first_path = Generated.path ~ext first and second_path = Generated.path ~ext second in
      let old_path = Stdlib.Filename.concat (Stdlib.Filename.dirname first_path) (first ^ ext) in
      let exceeds_max_path path =
        Option.is_some (Utils.Atomic_file.windows_path_length_note ~win32:true path)
      in
      exceeds_max_path old_path
      && String.length (Stdlib.Filename.basename first_path)
         <= Utils.library_kernel_name_limit + String.length ext
      && String.length first_path = String.length second_path
      && not (String.equal first_path second_path));
  let copied_stem = Stdlib.Filename.basename (Generated.path ~ext:"" first) in
  p "a routine named after a shortened stem has its own lowering path"
    (not (String.equal (Generated.path ~ext:".ll" copied_stem) (Generated.path ~ext:".ll" first)));
  List.iteri names ~f:(fun i name ->
      let scale = Float.of_int (i + 2) in
      let x = TDSL.range 16 in
      let multiplier = TDSL.number scale in
      let%op y = (multiplier *. x) + 1. in
      Train.set_materialized y.value;
      let comp = Train.forward y in
      let comp = { comp with Ir.Assignments.asgns = Block_comment (name, comp.asgns) } in
      let ctx, routine = Train.to_routine ctx ~output_cd_file:true Ir.Indexing.Empty comp in
      let ctx = Context.run ctx routine in
      p "the routine retains its full label" (String.equal routine.Context.name name);
      p_alli "the long-named routine computes its expected values"
        (Array.to_list (Context.get_values ctx y.value))
        ~min:16
        ~f:(fun i value -> Float.equal value ((scale *. Float.of_int i) +. 1.)));
  p_all "both generated sources retain their own full routine symbol" names ~f:(fun name ->
      String.is_substring (Generated.read name) ~substring:("void " ^ name ^ "("));
  p_all "both routines have all their lowering artifacts"
    (List.cartesian_product names [ ".cd"; ".ll"; "-unoptimized.ll" ])
    ~f:(fun (name, ext) -> not (String.is_empty (Generated.read ~ext name)));
  p_all "both optional training dumps retain their full labels" names ~f:(fun name ->
      String.is_substring (Generated.read ~ext:"-debug.cd" name) ~substring:name)
