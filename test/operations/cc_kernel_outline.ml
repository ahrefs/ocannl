(* gh-ocannl-1603: a large cc kernel is rendered OUTLINED -- its top-level statements packed in
   order into [noinline] parts the kernel calls in turn -- and a large batch is compiled as several
   translation units by concurrent compiler processes, linked into one library. Both exist because a
   C compiler's cost grows faster than linearly in a function's size (the serial baseline of a
   [gpt2_mini_train] search: 127 s whole, 8 s outlined over 8 units). Neither may change what the
   kernel computes.

   One routine of several top-level nests is compiled twice in this process: whole (outlining off,
   one unit) and split ([Cc_backend.with_split_settings]: every statement its own part, three
   units). The split compile must have outlined (its source defines parts) and been compiled as
   several units, and the two runs must agree bit for bit on every output. The routine is built to
   exercise what crosses a part boundary. [ol_scratch] is a [Local] array, a declaration of the
   kernel's own frame written by the first nest and read by later ones, which a part reaches through
   a pointer parameter. [ol_z] is zeroed and then accumulated into: the renderer may forward that
   zero into the accumulation, so no part boundary falls between them. One output is labelled as the
   split routine's third part would be named, so the part must take another name rather than be
   shadowed by that node's parameter. Every producer varies with its index, and the inputs are
   distinct dyadic values, so a part that read the wrong cell, or ran out of order, changes the
   output exactly. *)

open Base
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Generated = Test_utils.Generated
module Cc_backend = Context.Cc_backend

let n = 16
let node = Ll_test.node_factory ~first_id:968_000_000 ~dims:[| n |] ()
let x = node "ol_x"
let scratch = node "ol_scratch"
let y = node "ol_y"
let z = node "ol_z"
let w = node "ol_split__part2"
let outputs = [ y; z; w ]
let materialized = x :: outputs

let () =
  List.iter materialized ~f:Ll_test.materialize;
  Tn.update_memory_mode scratch Tn.Local (Site "99:test-setup")

let llc =
  let open Ll_test in
  let at s = [| iter s |] in
  let nest f =
    let i = sym () in
    loop_n i n (f i)
  in
  let stmts =
    [
      (* scratch[i] = 3 x[i] + i + 1 *)
      nest (fun i -> set scratch (at i) (add (mul (c 3.) (get x (at i))) (tick i)));
      (* y[i] = scratch[i] * scratch[i] + 1 *)
      nest (fun i -> set y (at i) (add (mul (get scratch (at i)) (get scratch (at i))) (c 1.)));
      (* z[i] = sum_j scratch[j] * x[i] *)
      zero z;
      (let i = sym () and j = sym () in
       loop_n i n
         (loop_n j n (set z (at i) (add (get z (at i)) (mul (get scratch (at j)) (get x (at i)))))));
      (* w[i] = y[i] + z[i] * (i + 1) *)
      nest (fun i -> set w (at i) (add (get y (at i)) (mul (get z (at i)) (tick i))));
      (* y[i] = y[i] * 0.5 + scratch[i] *)
      nest (fun i -> set y (at i) (add (mul (get y (at i)) (c 0.5)) (get scratch (at i))));
      (* w[i] = w[i] + y[i] * 0.25 *)
      nest (fun i -> set w (at i) (add (get w (at i)) (mul (get y (at i)) (c 0.25))));
    ]
  in
  LL.unflat_lines stmts

let inputs = Array.init n ~f:(fun i -> Float.of_int ((i * 7 % n) + 1) *. 0.125)

let compile_and_run ~name =
  let o = Ll_test.optimize ~materialized ~name llc in
  let local = Ll_test.known_local o scratch in
  (local, Ll_test.execute ~name o ~seed:[ (x, inputs) ] ~read:outputs)

let count_occurrences text ~pattern =
  List.length (String.substr_index_all text ~may_overlap:false ~pattern)

let () =
  Generated.init ~backend_name:"cc";
  let _, whole =
    Cc_backend.with_split_settings { part_weight = 0; jobs = 1; unit_min_bytes = 0 } (fun () ->
        compile_and_run ~name:"ol_whole")
  in
  let units_before = Cc_backend.translation_units_compiled () in
  let local, split =
    Cc_backend.with_split_settings { part_weight = 1; jobs = 3; unit_min_bytes = 0 } (fun () ->
        compile_and_run ~name:"ol_split")
  in
  let units = Cc_backend.translation_units_compiled () - units_before in
  let split_source = Generated.read "ol_split" and whole_source = Generated.read "ol_whole" in
  let part_definitions source = count_occurrences source ~pattern:"ARRAYJIT_KERNEL_PART void " in
  Stdio.eprintf
    "split: %d part declarations and definitions, %d translation units (not part of the golden)\n%!"
    (part_definitions split_source) units;
  p "the scratch node stays a Local array of the kernel's frame" local;
  p "the whole compile renders no part" (part_definitions whole_source = 0);
  (* Each part is defined once and declared once, ahead of the kernel that calls it. *)
  p "the split compile outlines the kernel into at least three parts"
    (part_definitions split_source >= 6);
  p "the split compile is compiled as more than one translation unit" (units >= 2);
  List.iter2_exn outputs (List.zip_exn whole split) ~f:(fun tn (whole, split) ->
      p_all2 ("split and whole agree bit for bit on " ^ Tn.debug_name tn) split whole ~f:Float.equal);
  (* gh-ocannl-1678: a test reading a kernel's statements out of its source follows its parts. The
     split kernel's main logic only calls them, so the text after its marker -- how such tests read
     before [Generated.main_logic] -- holds none of its statements, which the whole kernel's
     does. *)
  let after_marker source =
    match String.substr_index source ~pattern:Ir.C_syntax.main_logic_marker with
    | Some i -> String.subo source ~pos:i
    | None -> ""
  in
  let split_statements = Generated.main_logic split_source
  and whole_statements = Generated.main_logic whole_source in
  let assignment line = String.mem line '=' in
  p "the split kernel's statements, its parts spliced in at their calls, are the whole kernel's"
    (String.equal
       (Generated.normalize_minted split_statements)
       (Generated.normalize_minted whole_statements));
  p_exists "the split kernel's statements hold its assignments"
    (String.split_lines split_statements)
    ~f:assignment;
  p_exists "the text after the whole kernel's marker holds its assignments"
    (String.split_lines (after_marker whole_source))
    ~f:assignment;
  p_none "the text after the split kernel's own marker holds none of its assignments"
    (String.split_lines (after_marker split_source))
    ~f:assignment
