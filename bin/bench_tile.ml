(** The register-tile geometry a C-backend GEBP bench takes from its commandline (gh-ocannl-620
    follow-up).

    Since gh-ocannl-619 the [Tensorize] optop carries an optional {!Ir.Register_tile.t}, which the
    renderer honours exactly or declines to the scalar fallback naming the rule it broke. The tuner
    times the alternatives {!Ir.Register_tile.alternatives} seeds; what the benches could not do is
    ask for ONE geometry, so a before/after of two of them meant two builds of the library with the
    default model edited in between -- the slowest possible way to settle a 5% question, and one
    that cannot be re-run from a PR body. This module is that flag, shared by the two tools whose
    packed variants build the same [Sched.tensorize] call, so the spelling and the defaulting rules
    cannot drift apart between them.

    Two spellings, and passing both is refused rather than silently ranked:

    - [--tile=rm,rn,lanes] is the whole geometry, verbatim.
    - [--rn=N] names the column count alone and derives the rest the way the renderer's own model
      would: [rm = Ir.Register_tile.rm_cap] rows, and [lanes] the widest width the machine's vector
      file renders at the compute precision ({!Ir.Register_tile.simd_lane_ladder}, the head of the
      ladder the renderer ranks over). This is the spelling the open question of gh-ocannl-620 wants
      -- the tail-free [rn] against the wider tail-bearing one, at one width on one machine.

    NOTHING is validated against the site here. A geometry the renderer cannot honour is declined
    THERE, by {!Ir.Register_tile.check}, and shows up as [Mma_scalar_fallback] in the census bracket
    every timing line carries -- which is the bench's existing "read the bracket, not the variant
    name" discipline, and the one place where the rules a request must pass are stated. Pre-checking
    here would either duplicate those rules or, worse, reject a request the renderer would have
    taken. *)

open Base
module RT = Ir.Register_tile

(** [--tile=rm,rn,lanes]: three positive integers, checked where they are read (the [Bench_args]
    discipline) so a zero reaches a message that names it rather than a geometry the renderer
    declines for an unrelated-sounding reason. *)
let parse_triple args spec =
  let field name s =
    match Option.try_with (fun () -> Int.of_string (String.strip s)) with
    | Some v when v >= 1 -> v
    | Some v -> Bench_args.bad args "--tile=: %s must be positive, got %d" name v
    | None -> Bench_args.bad args "--tile=: %s must be an integer, got %S" name s
  in
  match String.split spec ~on:',' with
  | [ rm; rn; lanes ] -> { RT.rm = field "rm" rm; rn = field "rn" rn; lanes = field "lanes" lanes }
  | parts ->
      Bench_args.bad args "--tile= takes rm,rn,lanes: three comma-separated counts, got %d in %S"
        (List.length parts) spec

(** [--rn=N]: the column count alone. *)
let parse_rn args rn =
  match Option.try_with (fun () -> Int.of_string (String.strip rn)) with
  | Some v when v >= 1 -> v
  | Some v -> Bench_args.bad args "--rn= must be positive, got %d" v
  | None -> Bench_args.bad args "--rn= must be an integer, got %S" rn

(** [of_args args ~machine] is the requested geometry, or [None] when no flag asked for one — in
    which case the renderer picks, exactly as before the flag existed.

    [machine] is a thunk, forced once by EITHER spelling and not at all without a flag: a bench that
    would otherwise not have created a context by this point should not acquire one to answer a
    question nobody asked. It returns the machine's vector width in bytes
    ([Context.hardware_limits]'s [simd_vector_bytes]) and the COMPUTE precision's element width —
    compute, not storage, because that is what the renderer's [elt_bytes] is: a narrow-storage GEBP
    holds its C-tile at the compute precision.

    [--tile=] needs it for the same reason [--rn=] does, even though it derives nothing: a backend
    whose file renders no multi-lane vector has no register-tiled rendering for a geometry to
    describe, and its [Tile_mma] goes to the hardware's own intrinsics (or declines) whatever the
    schedule carries. Taking the request there would print a geometry under "(requested)" that
    nothing honoured — the "timed is not tensorized" hazard, one level up from the census bracket,
    which reports the intrinsic rendering as a success. GPU backends report [simd_vector_bytes = 0]
    and are exactly this case. *)
let of_args args ~machine =
  let honoured_here t =
    let vector_bytes, elt_bytes = machine () in
    match RT.simd_lane_ladder ~vector_bytes ~elt_bytes with
    | [] ->
        Bench_args.bad args
          "a register-tile geometry means nothing on this backend: a %d-byte vector file renders \
           no multi-lane vector at %d-byte elements, so its Tile_mma is not the register-tiled \
           rendering %s describes"
          vector_bytes elt_bytes
          (Option.value_map t ~default:"a geometry" ~f:RT.to_string)
    | ladder -> (vector_bytes, ladder)
  in
  (* Bounds BEFORE any product is formed. [Register_tile.check] rejects these values already, but
     only after computing [rn * lanes] and [rm * rn + rm + rn] in machine ints: at [rn = 2^61] the
     width wraps to zero, the budget check reads a negative live count, both pass, and [coverage]
     then evaluates [n mod 0]. The bounds are the machine's own, not a second copy of the rules —
     every one of them is IMPLIED by a rule [check] enforces, so nothing it would accept is refused
     here. [live_registers] is [rm * rn + rm + rn], which exceeds each of [rm] and [rn] on its own,
     so a field over the register budget can never fit it; and [lanes] must be a width the file
     renders, none of which exceeds the widest. Values below these bounds are left to the renderer,
     so an ordinary mistake (a [lanes] the ladder does not contain) still reaches the decline
     diagnostic rather than being second-guessed here. *)
  let bounded { RT.rm; rn; lanes } ~vector_bytes ~ladder =
    let budget = RT.budget ~vector_bytes in
    let widest = List.hd_exn ladder in
    let over name v cap rule =
      if v > cap then
        Bench_args.bad args "%s=%d exceeds %s (%d), which no geometry can pass" name v rule cap
    in
    over "rm" rm budget "the live-register budget of this vector file";
    over "rn" rn budget "the live-register budget of this vector file";
    over "lanes" lanes widest "the widest vector this file renders at this element size"
  in
  match (Bench_args.flag_value args ~flag:"tile", Bench_args.flag_value args ~flag:"rn") with
  | None, None -> None
  | Some _, Some _ ->
      Bench_args.bad args
        "--tile= and --rn= are two spellings of one request (--rn= derives rm and lanes); pass one"
  | Some spec, None ->
      let t = parse_triple args spec in
      let vector_bytes, ladder = honoured_here (Some t) in
      bounded t ~vector_bytes ~ladder;
      Some t
  | None, Some rn ->
      let rn = parse_rn args rn in
      (* The widest width the file renders is what the renderer's own model ranks first, so a
         derived request lands where [Register_tile.default] would have started. *)
      let vector_bytes, ladder = honoured_here None in
      let t = { RT.rm = RT.rm_cap; rn; lanes = List.hd_exn ladder } in
      bounded t ~vector_bytes ~ladder;
      Some t

(** [refuse_unreached args tile ~why] refuses a geometry that this run will not put in front of the
    renderer at all, naming it and why. {!of_args} answers the question the MACHINE decides — is
    there a register-tiled rendering here — and this one the question a TOOL decides:
    [schedule_bench] on a GPU backend runs only its shared/staged variants, none of which carry
    [?tile], so a request would reach no schedule while the header still said "(requested)". A
    refusal rather than a quieter header: the flag exists to measure a named geometry, and a run
    that measures something else under its name is the failure the census bracket already guards one
    level down. *)
let refuse_unreached args tile ~why =
  Option.iter tile ~f:(fun t ->
      Bench_args.bad args "the requested geometry %s reaches no schedule in this run: %s"
        (RT.to_string t) why)

(** [unmeasured tile ~statements] is the failure line for a run that requested a geometry and
    rendered no [Tile_mma] at all, or [None] when there is nothing to report.

    This is the a-posteriori half of the same claim {!refuse_unreached} makes a priori, and it is
    the one that needs no enumeration of eligibility paths: it asks the RENDERER how many [Tile_mma]
    statements the run actually produced (the merged [C_syntax.mma_census] both benches already
    collect), so every way a tile-bearing schedule can fail to run is covered by construction — an n
    the packed variants cannot block, a degenerate extent that skips every scheduled variant, and a
    variant added later whose eligibility nobody remembered to list here. The two halves are
    complementary and neither subsumes the other: a GPU branch DOES render a [Tile_mma] (through the
    hardware's intrinsics), so the census count would not catch it, and a skipped variant is
    knowable only after the run. Together with the scalar-fallback warning the benches already print
    — a geometry rendered and declined — they cover the three ways a timing can be recorded under a
    geometry that did not produce it. *)
let unmeasured tile ~statements =
  match tile with
  | Some t when statements < 1 ->
      Some
        (Printf.sprintf
           "NOT MEASURED: the requested geometry %s reached no Tile_mma in this run — every \
            variant that would carry it was skipped (see the lines above for why), so no timing \
            here measured it.\n"
           (RT.to_string t))
  | _ -> None

(** How the header line says what was requested — the geometry, or that the renderer's own model
    chose. Printed unconditionally, beside the blocking: a run whose geometry is invisible is one
    whose numbers cannot be compared against another run's. *)
let describe = function None -> "renderer default" | Some t -> RT.to_string t ^ " (requested)"
