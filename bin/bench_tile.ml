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

(** [of_args args ~machine] is the requested geometry, or [None] when no flag asked for one — in
    which case the renderer picks, exactly as before the flag existed.

    [machine] is a thunk, and is forced only by [--rn=]: the other two paths need nothing from the
    backend, and a bench that would otherwise not have created a context by this point should not
    acquire one to answer a question nobody asked. It returns the machine's vector width in bytes
    ([Context.hardware_limits]'s [simd_vector_bytes]) and the COMPUTE precision's element width —
    compute, not storage, because that is what the renderer's [elt_bytes] is: a narrow-storage GEBP
    holds its C-tile at the compute precision. *)
let of_args args ~machine =
  match (Bench_args.flag_value args ~flag:"tile", Bench_args.flag_value args ~flag:"rn") with
  | None, None -> None
  | Some _, Some _ ->
      Bench_args.bad args
        "--tile= and --rn= are two spellings of one request (--rn= derives rm and lanes); pass one"
  | Some spec, None -> Some (parse_triple args spec)
  | None, Some rn -> (
      let rn =
        match Option.try_with (fun () -> Int.of_string (String.strip rn)) with
        | Some v when v >= 1 -> v
        | Some v -> Bench_args.bad args "--rn= must be positive, got %d" v
        | None -> Bench_args.bad args "--rn= must be an integer, got %S" rn
      in
      let vector_bytes, elt_bytes = machine () in
      match RT.simd_lane_ladder ~vector_bytes ~elt_bytes with
      (* No ladder at all is a fact about the machine (or a GPU backend reporting no SIMD file), not
         a mis-typed flag: say which numbers produced it, since --rn= cannot be answered without one
         and --tile= still can. *)
      | [] ->
          Bench_args.bad args
            "--rn= cannot derive a vector width: a %d-byte vector file renders no multi-lane \
             vector at %d-byte elements; pass --tile=rm,rn,lanes instead"
            vector_bytes elt_bytes
      | widest :: _ -> Some { RT.rm = RT.rm_cap; rn; lanes = widest })

(** How the header line says what was requested — the geometry, or that the renderer's own model
    chose. Printed unconditionally, beside the blocking: a run whose geometry is invisible is one
    whose numbers cannot be compared against another run's. *)
let describe = function None -> "renderer default" | Some t -> RT.to_string t ^ " (requested)"
