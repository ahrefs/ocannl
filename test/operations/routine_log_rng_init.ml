(* A parameter's initialization routine compiles and runs with routine logging on, and logs its
   stores.

   A default [{ w }] parameter draws its values from a counter-based RNG: the init routine stores
   [uint4x32] Threefry blocks into a node of its own and converts their lanes into [w]. With
   [debug_log_from_routines] each store logs its value, and the log rendering used to treat every
   value as a float: a [uint4x32] block went to printf as [(int)(block * 1000.0)] (or as a [%g]
   argument), which no C compiler accepts for a struct. A second defect sat in the same statements:
   the logged text of the lane index, [(i % 4)], spliced a bare [%] into the format string, which
   printf reads as a conversion ([% 4)]: ' ' flag, width 4, conversion [')']). The rendering is now
   type-aware -- a [uint4x32] value logs its four lanes as [%u] -- and every piece of text spliced
   into a format string has its [%] doubled. Before the fix this routine did not compile on cc.

   The claims: the routine ran (the parameter holds distinct draws in the default init's range, and
   the forward read them); the stream log records a store of the RNG blocks as four unsigned lanes;
   it records every store into [w]; and a logged line carries the lane index's [%] verbatim. The log
   content itself is not a golden: it holds raw Threefry words. CUDA and HIP deliver routine logs
   through a capture window this test does not exercise, so they run the value claims only. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let dims = [ 3; 5 ]
let n = List.fold dims ~init:1 ~f:( * )

let stream_log_lines () =
  let dir = Utils.log_files_dir () in
  Stdlib.Sys.readdir dir |> Array.to_list
  |> List.filter ~f:(String.is_suffix ~suffix:".log")
  |> List.sort ~compare:String.compare
  |> List.concat_map ~f:(fun f -> Stdio.In_channel.read_lines (Stdlib.Filename.concat dir f))

(* ["<%u,%u,%u,%u>"] as printed: four unsigned decimal lanes. *)
let is_lane_block s =
  String.length s >= 2
  && Char.equal s.[0] '<'
  && Char.equal s.[String.length s - 1] '>'
  &&
  let lanes = String.split (String.sub s ~pos:1 ~len:(String.length s - 2)) ~on:',' in
  List.length lanes = 4
  && List.for_all lanes ~f:(fun l -> (not (String.is_empty l)) && String.for_all l ~f:Char.is_digit)

(* A value line of a store, ["<node>[<offset>]{=<old>} = <new> = <expression>"]: its [<new>]. *)
let stored_value line =
  match String.substr_index line ~pattern:"} = " with
  | None -> None
  | Some i -> (
      let rest = String.drop_prefix line (i + 4) in
      match String.substr_index rest ~pattern:" = " with
      | None -> None
      | Some j -> Some (String.prefix rest j))

let leg =
  Verdict.leg
    ~when_:(not (List.mem [ "cuda"; "hip" ] backend_name ~equal:String.equal))
    ~on:backend_name ()

let () =
  if not (Utils.debug_log_from_routines ()) then
    fail "this test runs under --ocannl_debug_log_from_routines=true (see its dune rule)";
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  Tensor.unsafe_reinitialize ();
  let%op y = { w; o = dims } *. 2.0 in
  let ctx = Train.forward_once (Context.auto ()) y in
  let w = Option.value_exn ~here:[%here] (Set.find y.Tensor.params ~f:(fun _ -> true)) in
  let wv = Context.get_values ctx w.Tensor.value and yv = Context.get_values ctx y.Tensor.value in
  Context.sync ctx;
  p_all "the parameter holds draws in the default init's range [-0.25, 0.25)" (Array.to_list wv)
    ~f:(fun v -> Float.(v >= -0.25 && v < 0.25));
  p "the parameter's draws are not all equal"
    (Array.length wv = n && not (Array.for_all wv ~f:(Float.equal wv.(0))));
  p_all2 "the forward read the initialized parameter" yv
    (Array.map wv ~f:(fun v -> 2. *. v))
    ~f:Float.equal;
  let lines = lazy (stream_log_lines ()) in
  in_leg ~leg "the log records an RNG block store as four unsigned lanes" (fun label ->
      p_exists label (Lazy.force lines) ~f:(fun line ->
          Option.exists (stored_value line) ~f:is_lane_block));
  in_leg ~leg "the log records every store into the parameter" (fun label ->
      let name = Ir.Tnode.label_name w.Tensor.value in
      let stores =
        List.filter (Lazy.force lines) ~f:(fun line ->
            String.is_prefix line ~prefix:(name ^ "[") && Option.is_some (stored_value line))
      in
      Stdio.eprintf "%d stores into %s logged (not part of the golden)\n%!" (List.length stores)
        name;
      p label (List.length stores = n));
  in_leg ~leg "a logged line carries the lane index's % verbatim" (fun label ->
      p_exists label (Lazy.force lines) ~f:(String.is_substring ~substring:" % "))
