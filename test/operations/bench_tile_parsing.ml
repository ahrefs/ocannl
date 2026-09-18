(* gh-ocannl-620 follow-up: the register-tile geometry request the bin/ GEBP benches take from their
   commandline ([Bench_tile]).

   The defect this pins is the one the whole census discipline exists against: a bench that times
   ONE geometry while its header line names another. Every claim below is about which geometry
   reaches [Sched.tensorize] for a given argv -- nothing here compiles or times anything, and
   [machine] is a fixture thunk, so the test links no backend.

   The derivation claims are stated against [Register_tile] itself rather than against the numbers
   it happens to produce on this machine: what [--rn=] promises is "the rows and the width the
   renderer's own model would have taken", so the assertions are equalities with [rm_cap] and with
   the head of the ladder, which stay true when either changes. *)

open Base
open Stdio
open Verdict.Claims
module RT = Ir.Register_tile

(* A 16-byte vector file at 4-byte elements -- the NEON f32 case, ladder [4]. The thunk counts its
   own forcings, because "only [--rn=] needs the machine" is a claim: a bench that acquired a
   backend context to answer a question nobody asked would still produce the right geometry, and the
   cost would be invisible. *)
let forcings = ref 0

let machine () =
  Int.incr forcings;
  (16, 4)

let args argv =
  Bench_args.create ~argv:(Array.of_list ("narrow_gebp_bench" :: argv)) "narrow_gebp_bench"

(* [invalid_arg]'s message, or [None] when the thunk returned. *)
let refused f = match f () with _ -> None | exception Invalid_argument msg -> Some msg

let () =
  let before = !forcings in
  let dflt = Bench_tile.of_args (args [ "f16"; "512"; "--ocannl_backend=cc" ]) ~machine in
  p "no flag requests no geometry, so the renderer picks exactly as it did before the flag existed"
    (Option.is_none dflt);
  p "and the machine is not consulted to decide that" (!forcings = before);
  p "the header says so rather than leaving the geometry unstated"
    (String.equal (Bench_tile.describe dflt) "renderer default")

let () =
  let before = !forcings in
  let t = Bench_tile.of_args (args [ "--tile=4,3,8" ]) ~machine in
  p "--tile= is taken verbatim, in rm,rn,lanes order"
    (Option.equal RT.equal t (Some { RT.rm = 4; rn = 3; lanes = 8 }));
  (* --tile= derives nothing, and still asks: a backend whose file renders no multi-lane vector has
     no register-tiled rendering for the geometry to describe (below). *)
  p "a verbatim geometry consults the machine too, once" (!forcings = before + 1);
  p "the header names the requested geometry and says it was requested"
    (String.equal (Bench_tile.describe t) "rm4 rn3 lanes8 (requested)")

let () =
  let before = !forcings in
  let t = Bench_tile.of_args (args [ "--rn=6" ]) ~machine in
  p "--rn= consults the machine exactly once" (!forcings = before + 1);
  let vector_bytes, elt_bytes = (16, 4) in
  let widest = List.hd (RT.simd_lane_ladder ~vector_bytes ~elt_bytes) in
  p "--rn= keeps the column count it was given, and derives the rest"
    (Option.equal RT.equal t
       (Option.map widest ~f:(fun lanes -> { RT.rm = RT.rm_cap; rn = 6; lanes })));
  (* Not "rm = 4 and lanes = 4": those are this machine's answers today. What --rn= promises is the
     renderer's own choices, so the equalities are with the module that owns them. *)
  p "the rows are the renderer's cap, not a constant restated here"
    (Option.value_map t ~default:false ~f:(fun t -> t.RT.rm = RT.rm_cap));
  p "the width is the widest the file renders at this element size, the head of the ladder"
    (Option.value_map t ~default:false ~f:(fun t -> Option.equal Int.equal (Some t.RT.lanes) widest));
  (* And the derivation lands on a geometry the renderer can actually honour where the site affords
     it -- a derived request that [check] declines would time the scalar fallback under a geometry
     label, which is the failure the census bracket exists to make visible. *)
  p "on a site with the rows and columns to spare, the derived geometry is one check accepts"
    (Option.value_map t ~default:false ~f:(fun t ->
         Result.is_ok (RT.check ~vector_bytes ~elt_bytes ~m:64 ~n:512 t)))

let () =
  (* A request the renderer declines is still PASSED to it: the rules live in [Register_tile.check],
     at the site, where the decline reaches the census and the run's warning. Rejecting it here
     would need a second copy of those rules, and would refuse geometries some other site
     affords. *)
  let t = Bench_tile.of_args (args [ "--tile=4,12,8" ]) ~machine in
  p "a geometry no 512-column site can honour is still handed to the renderer, not pre-rejected"
    (Option.equal RT.equal t (Some { RT.rm = 4; rn = 12; lanes = 8 })
    && Result.is_error
         (RT.check ~vector_bytes:16 ~elt_bytes:4 ~m:64 ~n:512 { RT.rm = 4; rn = 12; lanes = 8 }))

(* A backend reporting no SIMD vector file -- every GPU backend does, [simd_vector_bytes = 0]. Its
   [Tile_mma] renders through the hardware's own intrinsics, which the census reports as a SUCCESS,
   so a geometry taken here would print under "(requested)" with nothing having honoured it: the
   "timed is not tensorized" hazard one level above the bracket. *)
let no_vector_file () = (0, 4)

let () =
  p_all "neither spelling is taken on a backend with no register-tiled rendering to describe"
    [ [ "--tile=4,3,8" ]; [ "--rn=3" ] ] ~f:(fun argv ->
      match refused (fun () -> Bench_tile.of_args (args argv) ~machine:no_vector_file) with
      | Some msg ->
          String.is_substring msg ~substring:"means nothing on this backend"
          && String.is_substring msg ~substring:"0-byte vector file"
      | None -> false);
  (* The tool's own half of the same claim: a geometry the machine could honour, in a run whose
     schedules do not carry it (schedule_bench's GPU branch), is refused rather than printed as
     requested. *)
  p "a geometry that reaches no schedule in this run is refused, naming it and why"
    (match
       refused (fun () ->
           Bench_tile.refuse_unreached (args [])
             (Some { RT.rm = 4; rn = 3; lanes = 8 })
             ~why:"backend metal runs the shared/staged GPU schedules")
     with
    | Some msg ->
        String.is_substring msg ~substring:"rm4 rn3 lanes8 reaches no schedule"
        && String.is_substring msg ~substring:"shared/staged GPU schedules"
    | None -> false);
  p "and a run that requested nothing has nothing to refuse"
    (Option.is_none (refused (fun () -> Bench_tile.refuse_unreached (args []) None ~why:"unused")))

let () =
  (* Two spellings of one request is a contradiction, not a precedence question: ranking them would
     run a geometry the commandline does not unambiguously name. *)
  p "--tile= together with --rn= is refused, naming both"
    (match refused (fun () -> Bench_tile.of_args (args [ "--rn=4"; "--tile=4,6,8" ]) ~machine) with
    | Some msg -> String.is_substring msg ~substring:"--tile= and --rn= are two spellings"
    | None -> false);
  let cases =
    [
      ([ "--tile=4,6" ], "three comma-separated counts, got 2");
      ([ "--tile=4,6,8,2" ], "three comma-separated counts, got 4");
      ([ "--tile=4,0,8" ], "rn must be positive, got 0");
      ([ "--tile=4,x,8" ], "rn must be an integer, got \"x\"");
      ([ "--rn=0" ], "--rn= must be positive, got 0");
      ([ "--rn=wide" ], "--rn= must be an integer, got \"wide\"");
    ]
  in
  p_all "a malformed geometry is refused by name, where it is read" cases
    ~f:(fun (argv, substring) ->
      match refused (fun () -> Bench_tile.of_args (args argv) ~machine) with
      | Some msg -> String.is_substring msg ~substring
      | None -> false)

let () =
  (* Spaces around the counts survive a shell that quoted them. *)
  p "whitespace inside the triple is stripped rather than failing to parse"
    (Option.equal RT.equal
       (Bench_tile.of_args (args [ "--tile= 4 , 3 , 8 " ]) ~machine)
       (Some { RT.rm = 4; rn = 3; lanes = 8 }));
  printf "machine consulted %d time(s) over the whole run\n" !forcings
