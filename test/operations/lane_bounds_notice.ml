(* gh-ocannl-1990: [debug_check_lane_bounds] is never silently ignored. Only a dialect stating the
   backend fact [renders_lane_bounds_check] (cc) renders the run-time check ([lane_bounds_check]
   pins it); on CUDA, HIP and Metal the key used to check nothing and say nothing, a silence that
   reads as a clean bill. Now every backend's compile asks [Backends.warn_unchecked_lane_bounds],
   which prints one stderr line naming the key as cc-only, the first time it is owed in the process
   and never again.

   The once-only property is a process's, so each leg runs in a child of this executable and the
   parent counts the notice in its stderr:

   - [direct]: the guard driven with the GPU backends' facts, several times over, on any box: key
   on, exactly one line, naming the first backend asked for; key off, none; - [compile]: three
   compiles on the running backend ([Context.auto], which the parent's environment and flags pick):
   key on, one line exactly where that backend's dialect renders no check -- so on a Metal run, one,
   and on cc, none, its kernels checked as before; key off, none, on every backend.

   The parent also holds the pure half ([Backends.unchecked_lane_bounds_notice]) against the scope
   [cache_key_agreement] flips the key in: the notice is owed exactly on the backends whose cache
   key the key does not move, so a warning that changes no kernel enters no GPU cache tag. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module BI = Ir.Backend_intf

let marker = "debug_check_lane_bounds is cc-only"

let notices stderr =
  List.count (String.split_lines stderr) ~f:(String.is_substring ~substring:marker)

let direct () =
  List.iter
    [
      (BI.metal_facts, "metal");
      (BI.metal_facts, "metal");
      (BI.cuda_facts, "cuda");
      (BI.hip_facts, "hip");
      (BI.cc_facts, "cc");
      (BI.metal_facts, "metal");
    ]
    ~f:(fun (facts, backend) -> Context.Backends.warn_unchecked_lane_bounds ~facts ~backend)

let compile () =
  let base = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name base);
  let data =
    NTDSL.init ~l:"lbn_data" ~prec:Ir.Ops.single ~o:[ 4 ]
      ~f:(function [| j |] -> Float.of_int (j + 1) | _ -> assert false)
      ()
  in
  let ctx =
    List.fold [ 1; 2; 3 ] ~init:base ~f:(fun ctx k ->
        let out = NTDSL.add data (NTDSL.number (Float.of_int k)) () in
        Train.set_materialized out.Tensor.value;
        let ctx, routine =
          Context.compile ~name:(Printf.sprintf "lbn_add%d" k) ctx (Tensor.consume_forward_code out)
            Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        Stdio.printf "values %d: %s\n" k
          (String.concat ~sep:" "
             (Array.to_list
                (Array.map (Context.get_values ctx out.Tensor.value) ~f:(fun v ->
                     Printf.sprintf "%Lx" (Int64.bits_of_float v)))));
        ctx)
  in
  Context.release ctx;
  Context.release base

let parent () =
  (* The parent's own flags reach every child, so a backend chosen on the commandline is the
     children's too; the environment is inherited. The leg's own setting goes first, since the first
     matching flag wins. *)
  let flags =
    List.filter (List.tl_exn (Array.to_list Stdlib.Sys.argv)) ~f:(String.is_prefix ~prefix:"--")
  in
  let run mode ~on =
    let key = "debug_check_lane_bounds=" ^ Bool.to_string on in
    let ((status, stdout, stderr) as result) =
      Fresh_process.run (mode :: ("--ocannl_" ^ key) :: flags)
    in
    Fresh_process.report ~label:(mode ^ ", " ^ key) result;
    (Poly.equal status (Unix.WEXITED 0), stdout, stderr)
  in
  (* The pure half, over every backend's facts. *)
  let all_facts = [ BI.cc_facts; BI.cuda_facts; BI.hip_facts; BI.metal_facts ] in
  let notice ~requested facts =
    Context.Backends.unchecked_lane_bounds_notice ~facts ~requested ~backend:"probe"
  in
  p_all ~min:4
    "key on: the notice is owed exactly on the backends whose cache key the key does not move"
    all_facts ~f:(fun facts ->
      Bool.equal
        (Option.is_some (notice ~requested:true facts))
        (not (BI.renders_on_host_pool facts)));
  p_none ~min:4 "key off: no backend owes the notice" all_facts ~f:(fun facts ->
      Option.is_some (notice ~requested:false facts));
  p_all ~min:1 "the notice names the key, that it is cc-only, and the backend asked about"
    (List.filter_map all_facts ~f:(notice ~requested:true))
    ~f:(fun line ->
      String.is_substring line ~substring:marker
      && String.is_substring line ~substring:"probe backend");
  (* The once-only guard, driven with the GPU facts on any box. *)
  let on_ok, _, on_err = run "direct" ~on:true in
  let off_ok, _, off_err = run "direct" ~on:false in
  p "direct, key on: the child exits normally" on_ok;
  p "direct, key on: six asks print the notice exactly once" (notices on_err = 1);
  p "direct, key on: the one notice names the first backend asked about"
    (String.is_substring on_err ~substring:"the metal backend renders no lane-bounds check");
  p "direct, key off: the child exits normally" off_ok;
  p "direct, key off: no notice" (notices off_err = 0);
  (* Three real compiles on the running backend. *)
  let on_ok, on_out, on_err = run "compile" ~on:true in
  let off_ok, off_out, off_err = run "compile" ~on:false in
  let backend =
    List.find_map (String.split_lines on_out) ~f:(String.chop_prefix ~prefix:"backend: ")
  in
  Stdio.eprintf "lane_bounds_notice: compiled on the %s backend (not part of the golden)\n%!"
    (Option.value backend ~default:"<unknown>");
  let owed =
    Option.value_map backend ~default:true ~f:(fun name ->
        not (Context.backend_facts_of_name name).renders_lane_bounds_check)
  in
  p "compile, key on: the child exits normally and names its backend"
    (on_ok && Option.is_some backend);
  p
    "compile, key on: three compiles print the notice once where the dialect renders no check, \
     else never"
    (notices on_err = if owed then 1 else 0);
  p "compile, key off: the child exits normally" off_ok;
  p "compile, key off: no notice on any backend" (notices off_err = 0);
  p "compile: the key changes no value the routines compute"
    (let values out =
       List.filter (String.split_lines out) ~f:(String.is_prefix ~prefix:"values ")
     in
     List.length (values on_out) = 3 && List.equal String.equal (values on_out) (values off_out))

let () =
  match
    List.filter
      (List.tl_exn (Array.to_list Stdlib.Sys.argv))
      ~f:(fun arg -> not (String.is_prefix arg ~prefix:"--"))
  with
  | [] -> parent ()
  | [ "direct" ] -> direct ()
  | [ "compile" ] -> compile ()
  | _ -> failwith "usage: lane_bounds_notice.exe [direct|compile] [--ocannl_<key>=<value>...]"
