(* The tensorization label of a compiled routine (gh-ocannl-626).

   A [Tile_mma] whose emission preconditions fail renders the lane-0 scalar fallback, which
   compiles, runs and times perfectly well — so a timing reported under an [mma-*] label can be a
   timing of scalar code. The census that tells the two apart was opt-in and re-derived at every
   call site, which made "report a variant name unrelated to what rendered" the default for any new
   timing harness. It is now derived once, where the routine is compiled, and carried on it.

   This test pins the label's definition, which is the whole point of the field:

   - [Not_requested]: codegen emitted no [Tile_mma] statement. Nothing about the routine claims
   tensor cores. This is also the NEGATIVE CONTROL — a routine whose census was never going to say
   anything must not read as tensorized. - [Tensorized]: at least one [Tile_mma] rendered to a
   tensor-core or SIMD-register-tile emission. - [Scalar_fallback]: [Tile_mma] statements were
   emitted and EVERY one of them declined.

   and the two properties that make it trustworthy: the field on the routine is exactly what a
   bracket around that same compile would have collected (it is not fabricated), and the label is a
   function of the counts (it cannot drift from them).

   The honoured/declined pair needs a rendering the backend actually has: on the C backends a
   whole-triple [Tensorize] over a standard layout renders register-tiled while a transposed-B
   layout falls back, which is the pair used here. On a GPU backend those two legs report
   [Verdict.skipped] — the label's definition and its negative control are backend-independent and
   still run there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Cs = Ir.C_syntax
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The register-tiled rendering is the C backends'; "cc" also matches "multidev_cc". *)
let on_cpu = Sched.backend_is_cpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let accum_syms (opt : LL.optimized) =
  let paths = nest_paths opt.LL.llc in
  match List.find_exn paths ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let n = 64

(* Compile under [transform] and return BOTH the routine's own summary and the one a bracket around
   the same compile collects. Their agreement is what says the field reports this compile rather
   than a stale or fabricated census. *)
let compile_twice ~name ~transform comp =
  let bracketed_routine, bracketed =
    Cs.with_census (fun () ->
        let _ctx, routine =
          Context.compile ~name
            ~lowered_transform:(fun o -> [ transform o ])
            (Context.auto ()) comp Ir.Indexing.Empty
        in
        routine)
  in
  (bracketed_routine.Context.mma, bracketed)

let matmul ~tag =
  let av = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let bv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray bv ~label:[ tag ^ "_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  mc

(* Transposed-B: the gradient-GEMM shape, whose operand layout the Tile_mma rendering declines. *)
let matmul_tb ~tag =
  let av = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let bv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ n; n ] () in
  let mb = TDSL.ndarray bv ~label:[ tag ^ "_b" ] ~output_dims:[ n; n ] () in
  let%op mc = ma +* "ik;jk=>ij" mb in
  mc

let tensorize_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, k = accum_syms opt in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:n () in
  [ ez; rz; tz ]

(* === The label's three states === *)

let () =
  (* NEGATIVE CONTROL, and it runs on every backend: an ordinary matmul carries no [Tensorize], so
     its routine emitted no [Tile_mma] and there is no census finding to report. The reading that
     must NOT come out of a routine nobody asked to tensorize is "tensorized". *)
  let plain = matmul ~tag:"mtl_plain" in
  let mma, bracketed =
    compile_twice ~name:"mtl_plain"
      ~transform:(fun opt -> opt)
      (named "mtl_plain" (Train.forward plain))
  in
  p "a routine with no Tensorize is labeled not-requested"
    (Cs.equal_tensorization mma.Cs.tensorization Cs.Not_requested);
  p "a routine with no Tensorize does not read as tensorized"
    (not (String.equal (Cs.tensorization_name mma.Cs.tensorization) "tensorized"));
  p "the routine's summary is what a bracket around the same compile collects"
    (Cs.equal_tensorization mma.Cs.tensorization bracketed.Cs.tensorization
    && mma.Cs.statements = bracketed.Cs.statements
    && mma.Cs.scalar_fallbacks = bracketed.Cs.scalar_fallbacks)

let () =
  if not on_cpu then (
    Verdict.skipped ~backend:backend_name "an honoured Tensorize is labeled tensorized";
    Verdict.skipped ~backend:backend_name "a declined Tensorize is labeled scalar-fallback";
    Verdict.skipped ~backend:backend_name
      "a declined Tensorize does not read as tensorized despite the schedule asking")
  else begin
    let honoured = matmul ~tag:"mtl_ok" in
    let mma_ok, _ =
      compile_twice ~name:"mtl_ok"
        ~transform:(fun opt -> Sched.apply (tensorize_schedule ~out:honoured.Tensor.value opt) opt)
        (named "mtl_ok" (Train.forward honoured))
    in
    p "an honoured Tensorize is labeled tensorized"
      (Cs.equal_tensorization mma_ok.Cs.tensorization Cs.Tensorized
      && mma_ok.Cs.statements > 0 && mma_ok.Cs.scalar_fallbacks = 0);
    let declined = matmul_tb ~tag:"mtl_tb" in
    let mma_tb, _ =
      compile_twice ~name:"mtl_tb"
        ~transform:(fun opt -> Sched.apply (tensorize_schedule ~out:declined.Tensor.value opt) opt)
        (named "mtl_tb" (Train.forward declined))
    in
    p "a declined Tensorize is labeled scalar-fallback"
      (Cs.equal_tensorization mma_tb.Cs.tensorization Cs.Scalar_fallback
      && mma_tb.Cs.statements > 0
      && mma_tb.Cs.scalar_fallbacks = mma_tb.Cs.statements);
    (* The defect in one line: the same schedule move, the same variant name, and the emission went
       two different ways. Only the label separates the timings. *)
    p "a declined Tensorize does not read as tensorized despite the schedule asking"
      (not (Cs.equal_tensorization mma_tb.Cs.tensorization mma_ok.Cs.tensorization))
  end

(* === The label is a function of the counts, on every backend === *)

let () =
  let derived renderings =
    let s = Cs.summarize_census renderings in
    (s.Cs.tensorization, s.Cs.statements, s.Cs.scalar_fallbacks)
  in
  let entry r = ("kernel", r) in
  let fb = entry Cs.Mma_scalar_fallback in
  let rt = entry Cs.Mma_register_tiled in
  let ix = entry Cs.Mma_intrinsics in
  let ld = entry Cs.Mma_intrinsics_ldmatrix in
  p "no statements is not-requested"
    (match derived [] with Cs.Not_requested, 0, 0 -> true | _ -> false);
  p "all statements declined is scalar-fallback"
    (match derived [ fb; fb ] with Cs.Scalar_fallback, 2, 2 -> true | _ -> false);
  p_all "every non-fallback rendering counts as tensorized" [ rt; ix; ld ] ~f:(fun r ->
      match derived [ r ] with Cs.Tensorized, 1, 0 -> true | _ -> false);
  (* Mixed: some honoured, some declined. The label answers "did any tensor-core emission happen",
     the counts answer "how much of what was asked for" — a mixed routine is tensorized AND carries
     a nonzero fallback count, and a reader gets both. *)
  p "a partly declined routine is tensorized with a nonzero fallback count"
    (match derived [ rt; fb ] with Cs.Tensorized, 2, 1 -> true | _ -> false)

(* === Nesting: an enclosing collection still sees the inner compiles === *)

let () =
  (* [Context.compile] brackets the census itself now. A harness that ALSO brackets — around a whole
     sweep of compiles, which is what the benches do — must not be emptied by that, so nesting is
     additive: the inner bracket summarizes its own entries, and the outer one still sees them.
     Entries are pushed directly here rather than compiled, so the contract is pinned on every
     backend and independent of what any schedule happens to render. *)
  let inner_seen = ref 0 in
  let (), outer =
    Cs.with_census (fun () ->
        let (), inner =
          Cs.with_census (fun () ->
              Cs.mma_census := ("nested_kernel", Cs.Mma_register_tiled) :: !Cs.mma_census)
        in
        inner_seen := inner.Cs.statements;
        Cs.mma_census := ("outer_kernel", Cs.Mma_scalar_fallback) :: !Cs.mma_census)
  in
  p "a nested bracket summarizes only its own entries" (!inner_seen = 1);
  p "an enclosing bracket observes the entries of a nested one"
    (outer.Cs.statements = 2 && outer.Cs.scalar_fallbacks = 1);
  (* Over what the outer bracket collected: the global is empty AFTER a bracket that recorded
     something, not one that never had anything to leave behind. *)
  p_empty "a completed bracket leaves the census global as it found it" ~over:outer.Cs.renderings
    !Cs.mma_census

(* === The report-level negative control === *)

let () =
  let r = Autotune.no_search_report ~timing:Autotune.Queued in
  (* A call that searched nothing consulted no census. [None] is a distinct value from
     [Not_requested] precisely so this cannot be defaulted into a finding. *)
  p "a report with no crowned candidate carries no tensorization label"
    (Option.is_none r.Autotune.best_tensorization);
  p "a report with no crowned candidate does not render as tensorized"
    (not
       (String.equal
          (Option.value_map r.Autotune.best_tensorization ~default:"none" ~f:Cs.tensorization_name)
          "tensorized"))

(* stderr, not stdout: the golden has to stay backend-uniform (the skipped legs print the passing
   line for exactly that reason), and a run's backend is confirmed by reading its stderr. *)
let () = Stdio.eprintf "mma_tensorization_label: backend %s\n" backend_name
