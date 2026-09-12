(* The per-routine PEEL CENSUS: which decision produced a kernel, not which form was rendered
   (gh-ocannl-733).

   [reduction_forms] classifies an emitted kernel — localized scope, per-step read-modify-write,
   SIMD grid, warp tree — and pins its site counts. That answers which form was rendered, and for
   the reduction peel it cannot answer which decision produced it, because two different decisions
   produce the SAME form:

   - A nest whose accumulated cell is free of the enclosing index ([tot[0] += x[r,k]] under [If (k <
   s)]) is peeled at the OUTER level: both levels join the peel, and the guard is [Confined_to_peel]
   — every symbol it mentions is peeled or bound outside every loop. - A nest whose cell mentions
   the enclosing index ([out[r] += x[r,k]] under [If (r + k < s)]) cannot be peeled there ([out[r]]
   varies with [r]), so the reduction level peels alone and the guard is admitted as
   [Lane_private_if_separated]: it mentions the enclosing [r], and the hoist is legal only because
   [out[r]] gives each lane its own cell (gh-ocannl-721).

   Both emit one localized scope with one closing store. A form claim passes over either, which is
   how a test can be green over a kernel the code path it is named for never touched. This file pins
   the instrument that tells them apart: [Low_level.peel_accum_nest] reporting its verdicts,
   [C_syntax] accumulating them per routine, and [Context.routine.peel] carrying the summary.

   Hand-built {!Ir.Low_level} (via [ll_test]) so the nest shape is the test's rather than shape
   inference's, and executed as well as compiled — a census over a kernel that computes the wrong
   sum would be an instrument reading a program nobody wants. *)

open Base
open Ocannl.Operation.DSL_modules
module Cs = Ir.C_syntax
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Ops = Ir.Ops
open Verdict.Claims

let rows = 3
let cols = 8
let guard_terms = 5
let prec = Ops.single
let cell r k = Float.of_int (((r * cols) + k) % 7) +. 1.
let x_values = Array.init (rows * cols) ~f:(fun n -> cell (n / cols) (n % cols))
let base_ctx = lazy (Context.auto ())
let next_id = ref 733_000_000

(* {1 The nests}

   One reduction, five accumulator/guard combinations. [tot] is the single-cell accumulator whose
   address is free of BOTH loop symbols; [out] is the per-row one. *)

type shape =
  | Plain  (** [for r: for k: out[r] += x[r,k]] — no guard at all. *)
  | Confined_guard
      (** [for r: for k: tot[0] += x[r,k]] under [If (k < s)]: the guard mentions the peeled [k] and
          a launch-bound extent, and the cell is free of both loop symbols — so the peel starts at
          the OUTER level and takes both. *)
  | Lane_private_guard
      (** [for r: for k: out[r] += x[r,k]] under [If (r + k < s)]: the gh-ocannl-721 escape. *)
  | Enclosing_guard
      (** [for r: for k: out[r] += x[r,k]] under [If (r < s)]: at the reduction level the guard
          mentions no peeled symbol, so its truth is fixed for the whole nest and the peel refuses.
      *)
  | Data_guard
      (** A guard reading a node: not a pure index guard, so the level's body is not a shape the
          peel recognizes at all. *)

let shape_name = function
  | Plain -> "plain"
  | Confined_guard -> "confined guard over a shared cell"
  | Lane_private_guard -> "lane-private guard over a per-row cell"
  | Enclosing_guard -> "guard fixed for the whole nest"
  | Data_guard -> "data-dependent guard"

type prog = {
  llc : LL.t;
  materialized : Tn.t list;
  seed : (Tn.t * float array) list;
  out : Tn.t;  (** The node whose contents the value claim reads. *)
  want : float array;
  bindings : Idx.unit_bindings;
  bind : Idx.lowered_bindings -> unit;
}

(* A launch parameter standing for a symbolic DIMENSION: the guard's bound is then outside every
   loop (which is what makes it a peelable guard symbol rather than an enclosing index) and cannot
   be folded away by the guard simplifier the way a constant bound can. *)
let extent_symbol () =
  let s, bindings = Idx.get_static_symbol ~static_range:cols Idx.Empty in
  s.Idx.used_as_extent <- true;
  (s, bindings)

let make shape : prog =
  let node ?(dims = [| rows; cols |]) label =
    Int.incr next_id;
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  let out = node ~dims:[| rows |] "pcout" in
  let tot = node ~dims:[| 1 |] "pctot" in
  let x = node "pcxs" in
  List.iter [ out; tot; x ] ~f:Ll_test.materialize;
  let r = Ll_test.sym () and k = Ll_test.sym () in
  let iprec = Ops.index_prec () in
  let ri = Idx.Iterator r and ki = Idx.Iterator k in
  let accumulate tn idcs =
    LL.Set
      {
        tn;
        idcs;
        llsc = LL.Binop (Ops.Add, (LL.Get (tn, idcs), prec), (LL.Get (x, [| ri; ki |]), prec));
        debug = "";
      }
  in
  let per_row = accumulate out [| ri |] in
  let shared = accumulate tot [| Idx.Fixed_idx 0 |] in
  let nest body =
    LL.For_loop
      {
        index = r;
        from_ = 0;
        to_ = rows - 1;
        axis = LL.Serial;
        body = LL.For_loop { index = k; from_ = 0; to_ = cols - 1; axis = LL.Serial; body };
      }
  in
  let guard cond body = LL.If { cond = (cond, iprec); body } in
  let lt lhs s =
    LL.Binop
      ( Ops.Cmplt,
        (LL.Embed_index lhs, iprec),
        (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
  in
  let seed_x = [ (x, x_values) ] in
  let row_sums ~terms =
    Array.init rows ~f:(fun r ->
        let n = terms r in
        Array.fold (Array.init n ~f:(fun k -> cell r k)) ~init:0. ~f:( +. ))
  in
  let bound s lowered = Idx.find_exn lowered s := guard_terms in
  match shape with
  | Plain ->
      {
        llc = nest per_row;
        materialized = [ out; x ];
        seed = seed_x;
        out;
        want = row_sums ~terms:(fun _ -> cols);
        bindings = Idx.Empty;
        bind = (fun _ -> ());
      }
  | Confined_guard ->
      let s, bindings = extent_symbol () in
      let total = Array.fold (row_sums ~terms:(fun _ -> guard_terms)) ~init:0. ~f:( +. ) in
      {
        llc = nest (guard (lt ki s) shared);
        materialized = [ tot; x ];
        seed = seed_x;
        out = tot;
        want = [| total |];
        bindings;
        bind = bound s;
      }
  | Lane_private_guard ->
      let s, bindings = extent_symbol () in
      {
        llc = nest (guard (lt (Idx.Affine { symbols = [ (1, r); (1, k) ]; offset = 0 }) s) per_row);
        materialized = [ out; x ];
        seed = seed_x;
        out;
        (* [r + k < s] admits [s - r] terms in row [r]. *)
        want = row_sums ~terms:(fun r -> guard_terms - r);
        bindings;
        bind = bound s;
      }
  | Enclosing_guard ->
      let s, bindings = extent_symbol () in
      {
        llc = nest (guard (lt ri s) per_row);
        materialized = [ out; x ];
        seed = seed_x;
        out;
        (* [r < s] is true of every row here ([s] binds to 5, [rows] is 3): the guard selects among
           the ENCLOSING level's iterations, which is exactly why the peel refuses it, and admits
           the whole reduction in each. *)
        want = row_sums ~terms:(fun _ -> cols);
        bindings;
        bind = bound s;
      }
  | Data_guard ->
      let mask = node ~dims:[| cols |] "pcmask" in
      Ll_test.materialize mask;
      let cond = LL.Binop (Ops.Cmplt, (LL.Get (mask, [| ki |]), prec), (LL.Constant 1.0, prec)) in
      {
        llc = nest (LL.If { cond = (cond, prec); body = per_row });
        materialized = [ out; x; mask ];
        seed = (mask, Array.init cols ~f:(fun k -> if k < guard_terms then 0.0 else 1.0)) :: seed_x;
        out;
        want = row_sums ~terms:(fun _ -> guard_terms);
        bindings = Idx.Empty;
        bind = (fun _ -> ());
      }

(* {1 Compiling one nest and reading its census} *)

type run = {
  summary : Cs.peel_summary;
  volatility : Cs.volatility_summary;
  values : float array;
  want : float array;
  bracketed : Cs.peel_summary;
      (** The census a bracket around the same compile collects. Its agreement with the routine's
          own field is what says the field reports THIS compile rather than a stale or fabricated
          census. *)
}

let compile_and_run ~name shape =
  let prog = make shape in
  let static_indices = Idx.bound_symbols prog.bindings in
  let o = Ll_test.optimize ~materialized:prog.materialized ~static_indices ~name prog.llc in
  let (ctx, routine), bracketed =
    Cs.with_peel_census (fun () ->
        Context.compile ~name ~prelowered:o (Lazy.force base_ctx) Ir.Assignments.empty_comp
          prog.bindings)
  in
  prog.bind routine.Context.bindings;
  let ctx = List.fold prog.seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
  let ctx = Context.run ctx routine in
  {
    summary = routine.Context.peel;
    volatility = routine.Context.volatility;
    values = Context.get_values ctx prog.out;
    want = prog.want;
    bracketed;
  }

let localized_verdicts summary =
  List.filter_map summary.Cs.sites ~f:(fun (_, site) ->
      match site with Cs.Peel_localized v -> Some v | _ -> None)

let refusals summary =
  List.filter_map summary.Cs.sites ~f:(fun (_, site) ->
      match site with Cs.Peel_refused why -> Some why | _ -> None)

let verdict_string { Cs.levels; guards } =
  Printf.sprintf "%d level(s) through [%s]" levels
    (String.concat ~sep:"; "
       (List.map guards ~f:(fun g -> Sexp.to_string (LL.sexp_of_peel_guard_verdict g))))

let runs =
  List.map
    [
      (Plain, "pc_plain");
      (Confined_guard, "pc_confined");
      (Lane_private_guard, "pc_lane_private");
      (Enclosing_guard, "pc_enclosing");
      (Data_guard, "pc_data_guard");
    ]
    ~f:(fun (shape, name) -> (shape, compile_and_run ~name shape))

let run_of shape =
  List.Assoc.find_exn runs shape ~equal:(fun a b -> String.equal (shape_name a) (shape_name b))

(* The census of every nest, on stderr: the summaries name backend kernel names and count sites the
   surrounding structure decides, so they are diagnostics rather than golden lines. *)
let () =
  List.iter runs ~f:(fun (shape, run) ->
      Stdio.eprintf "%s: %s\n" (shape_name shape) (Cs.peel_summary_string run.summary);
      Stdio.eprintf "  volatility: %s\n" (Cs.volatility_summary_string run.volatility);
      List.iter run.summary.Cs.sites ~f:(fun (kernel, site) ->
          Stdio.eprintf "    %-24s %s\n" kernel (Sexp.to_string (Cs.sexp_of_peel_site site)));
      Stdio.eprintf "%!")

(* {1 The claims} *)

let () =
  List.iter runs ~f:(fun (shape, run) ->
      p_all2
        (Printf.sprintf "%s: the nest computes its reference sum" (shape_name shape))
        run.values run.want
        ~f:(fun got want -> Float.(abs (got - want) < 1e-4));
      p
        (Printf.sprintf "%s: the routine's census equals the one a bracket around it collects"
           (shape_name shape))
        (List.equal
           (fun (_, a) (_, b) -> Cs.equal_peel_site a b)
           run.summary.Cs.sites run.bracketed.Cs.sites);
      (* The two censuses are collected by different code for different questions, and over these
         nests they must agree on one number: localizing a site is what mints the accumulator the
         volatility census then classifies (gh-ocannl-782). The claim is backend-uniform — on a
         backend that requests the accumulation workaround those sites use volatile reads, on one
         that does not they stay plain, and either way there is one per localized site. Where the
         peel declined, no accumulator exists to classify; on Metal the device-memory
         read-modify-write it left behind receives the expression-level read cast instead, which is
         the other half of the same picture (the summaries above show it). *)
      p
        (Printf.sprintf "%s: one censused accumulator per localized peel site" (shape_name shape))
        (run.volatility.Cs.volatile_accumulations + run.volatility.Cs.plain_accumulations
        = run.summary.Cs.localized))

(* The two localizing shapes. Each pins the FULL verdict — how many levels were peeled and which
   guard verdict each peeled [If] earned — over every site that localized, so a decision that
   swallowed one more level, or admitted a guard on the other ground, fails here. *)
let () =
  let plain = run_of Plain in
  Verdict.p_all "every localized site of the plain nest peels the reduction level alone, unguarded"
    (localized_verdicts plain.summary) ~f:(fun v -> v.Cs.levels = 1 && List.is_empty v.Cs.guards);
  let confined = run_of Confined_guard in
  Verdict.p_all
    "every localized site of the shared-cell nest peels BOTH levels through a confined guard"
    (localized_verdicts confined.summary) ~f:(fun v ->
      v.Cs.levels = 2 && List.equal LL.equal_peel_guard_verdict v.Cs.guards [ LL.Guard_confined ]);
  let lane_private = run_of Lane_private_guard in
  Verdict.p_all
    "every localized site of the per-row nest peels the reduction level alone through a \
     lane-private guard" (localized_verdicts lane_private.summary) ~f:(fun v ->
      v.Cs.levels = 1
      && List.equal LL.equal_peel_guard_verdict v.Cs.guards [ LL.Guard_lane_private ]);
  (* The point of the instrument, stated as a claim: the two guarded nests localize alike — so no
     classification of the emitted form can separate them — and the census does. *)
  let verdicts run = List.map (localized_verdicts run.summary) ~f:verdict_string in
  let confined_v = verdicts confined and lane_v = verdicts lane_private in
  Stdio.eprintf "confined: [%s]  lane-private: [%s]\n%!"
    (String.concat ~sep:", " confined_v)
    (String.concat ~sep:", " lane_v);
  p "both guarded nests localize, and their peel verdicts differ"
    ((not (List.is_empty confined_v))
    && (not (List.is_empty lane_v))
    && not (List.equal String.equal confined_v lane_v))

(* The declines. Each names the refusal it earns, so a peel that started admitting one of these
   shapes — or refused it on another ground — fails rather than passing on "it did not localize". *)
let () =
  let enclosing = run_of Enclosing_guard in
  Verdict.p_empty "the nest-fixed guard localizes nowhere" ~over:enclosing.summary.Cs.sites
    (localized_verdicts enclosing.summary);
  Verdict.p_exists "the nest-fixed guard is refused as a guard whose truth is fixed for the nest"
    (refusals enclosing.summary) ~f:(function
    | LL.Refused_guard_fixed _ -> true
    | _ -> false);
  (* And the enclosing level of that same nest refuses for the OTHER reason: it is the accumulated
     cell that varies there, not the guard. Two refusal kinds from one kernel is what says the
     report distinguishes them rather than labelling every decline alike. *)
  Verdict.p_exists "its enclosing level is refused because the accumulated cell varies"
    (refusals enclosing.summary) ~f:(fun why -> LL.equal_peel_refusal why LL.Refused_cell_varies);
  let data = run_of Data_guard in
  Verdict.p_empty "the data-dependent guard localizes nowhere" ~over:data.summary.Cs.sites
    (localized_verdicts data.summary);
  Verdict.p_all "every peel site of the data-guarded nest refused" data.summary.Cs.sites
    ~f:(fun (_, site) -> not (Cs.is_localized_peel site))

(* {1 The report a refusal carries}

   The census only ever shows the guards of a site that LOCALIZED, but [~report] is public and a
   consumer sees the refusing reports too. A [Lane_private_if_separated] guard is admitted only once
   the base's cell has been shown to separate the enclosing symbols it mentions — a check that
   happens at the base — so a report that refused before reaching one must not read as though the
   guard had been admitted (Codex P2, round 1). Asked of [peel_accum_nest] directly, over the two
   nests that differ ONLY in the accumulated cell: per-lane, so separation holds; shared, so it does
   not. *)
let () =
  let src =
    Tn.create (Tn.Specified prec) ~id:733_900_001 ~label:[ "pcsrc" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None)
      ()
  in
  let lanes =
    Tn.create (Tn.Specified prec) ~id:733_900_002 ~label:[ "pclanes" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None)
      ()
  in
  let shared =
    Tn.create (Tn.Specified prec) ~id:733_900_003 ~label:[ "pcshared" ]
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  let w = Idx.get_symbol () in
  let report_of ~cell =
    let k = Idx.get_symbol () in
    let tn, idcs =
      match cell with
      | `Per_lane -> (lanes, [| Idx.Iterator w |])
      | `Shared -> (shared, [| Idx.Fixed_idx 0 |])
    in
    let nest =
      LL.For_loop
        {
          index = k;
          from_ = 0;
          to_ = 3;
          axis = LL.Serial;
          body =
            LL.If
              {
                (* [w + k < 1]: mentions the peeled [k] and the enclosing [w], which is exactly the
                   guard [Affine.peel_guard] admits conditionally. *)
                cond =
                  ( LL.Binop
                      ( Ops.Cmplt,
                        ( LL.Embed_index (Idx.Affine { symbols = [ (1, w); (1, k) ]; offset = 0 }),
                          prec ),
                        (LL.Constant 1.0, prec) ),
                    prec );
                body =
                  LL.Set
                    {
                      tn;
                      idcs;
                      llsc =
                        LL.Binop
                          ( Ops.Add,
                            (LL.Get (tn, idcs), prec),
                            (LL.Get (src, [| Idx.Iterator k |]), prec) );
                      debug = "";
                    };
              };
        }
    in
    let report = ref None in
    let peeled =
      LL.peel_accum_nest
        ~report:(fun r -> report := Some r)
        ~loop_bounds:[ (w, (0, 3)) ]
        ~free_of:[] nest
    in
    (Option.is_some peeled, Option.value_exn !report)
  in
  let peeled_lane, lane_report = report_of ~cell:`Per_lane in
  let peeled_shared, shared_report = report_of ~cell:`Shared in
  Stdio.eprintf "per-lane: %s\nshared:   %s\n%!"
    (Sexp.to_string (LL.sexp_of_peel_report lane_report))
    (Sexp.to_string (LL.sexp_of_peel_report shared_report));
  p "a lane-private guard over a per-lane cell peels, and the report admits the guard"
    (peeled_lane
    && Option.is_none lane_report.LL.refusal
    && List.equal LL.equal_peel_guard_verdict lane_report.LL.guards [ LL.Guard_lane_private ]);
  p "the same guard over a shared cell refuses, and the report leaves that guard unresolved"
    ((not peeled_shared)
    && Option.equal LL.equal_peel_refusal shared_report.LL.refusal (Some LL.Refused_cell_shared)
    && List.equal LL.equal_peel_guard_verdict shared_report.LL.guards
         [ LL.Guard_lane_private_unresolved ]);
  Verdict.p_none "no refusing report claims an admitted lane-private guard" [ shared_report ]
    ~f:(fun r ->
      Option.is_some r.LL.refusal
      && List.mem r.LL.guards LL.Guard_lane_private ~equal:LL.equal_peel_guard_verdict)

(* {1 The bracket's own discipline}

   [with_peel_census] is what makes the census a property of the compiled routine rather than of
   whichever caller remembered to collect it, so its nesting and restoration are pinned directly —
   with hand-written entries, so the claims do not depend on what any backend renders. *)
let () =
  let site levels = Cs.Peel_localized { levels; guards = [] } in
  let inner_summary, outer_summary =
    Cs.with_peel_census (fun () ->
        Cs.peel_census := ("outer_kernel", site 1) :: !Cs.peel_census;
        let (), inner =
          Cs.with_peel_census (fun () ->
              Cs.peel_census := ("inner_kernel", site 2) :: !Cs.peel_census)
        in
        inner)
  in
  let entries =
    List.equal (fun (n1, s1) (n2, s2) -> String.equal n1 n2 && Cs.equal_peel_site s1 s2)
  in
  p "an inner bracket summarizes only its own sites"
    (entries inner_summary.Cs.sites [ ("inner_kernel", site 2) ]);
  (* Additively, not shadowing: an enclosing collection still sees what an inner compile censused,
     or wrapping the compile path in the bracket would silently empty every outer one. *)
  p "the enclosing bracket sees both, in emission order"
    (entries outer_summary.Cs.sites [ ("outer_kernel", site 1); ("inner_kernel", site 2) ]);
  (* Over what the outer bracket collected: the global is empty AFTER a bracket that recorded
     something, not one that never had anything to leave behind. *)
  Verdict.p_empty "a completed bracket leaves the census global as it found it"
    ~over:outer_summary.Cs.sites !Cs.peel_census;
  p "collection is off outside every bracket" (not !Cs.peel_census_enabled)

(* The census's recurrence gate distinguishes a disjoint gather from a possible self-read
   (gh-ocannl-960). These are predicate tests: no value transformation or hardware legality decision
   consumes this helper. Unknown static expressions must retain the conservative answer. *)
let () =
  let module L = Ll_test in
  let node = L.node_factory ~first_id:960_000_000 ~dims:[| 4; 4; 4 |] () in
  let a = node "gather_census" and other = node "gather_other" in
  let i = L.sym () and j = L.sym () in
  let fixed = L.fixed in
  let target = [| fixed 0; fixed 2; fixed 0 |] in
  let selector = (L.embed i, L.iprec ()) in
  let census ?(tn = a) ?(dyn_value = selector) ?(written = target) idcs =
    L.set a written (L.gather ~tn ~idcs ~dyn_axis:1 ~dyn_value) |> LL.has_accumulating_cell
  in
  p "a gather from a different fixed first slot is not a cell recurrence"
    (not (census [| fixed 1; fixed 0; fixed 0 |]));
  p "a gather from a different fixed last slot is not a cell recurrence"
    (not (census [| fixed 0; fixed 0; fixed 1 |]));
  p "matching static slots keep a potentially aliasing gather in the census"
    (census [| fixed 0; fixed 0; fixed 0 |]);
  (* Written slot 2 differs from the gather placeholder 0: comparing that placeholder would
     manufacture disjointness even though the runtime selector can equal 2. *)
  p "a one-axis gather cannot prove disjointness from its placeholder"
    (L.set a [| fixed 2 |] (L.gather ~tn:a ~idcs:[| fixed 0 |] ~dyn_axis:0 ~dyn_value:selector)
    |> LL.has_accumulating_cell);
  p "a symbolic static slot may alias a fixed written slot"
    (census [| L.iter i; fixed 0; fixed 0 |]);
  p "different symbolic static slots may alias each other"
    (census ~written:[| L.iter i; fixed 2; fixed 0 |] [| L.iter j; fixed 0; fixed 0 |]);
  p "an affine static slot may alias a differently spelled written slot"
    (census ~written:[| L.iter i; fixed 2; fixed 0 |] [| L.aff [ (1, j) ] 1; fixed 0; fixed 0 |]);
  p "a fixed disjoint slot wins even when another static slot is symbolic"
    (not (census [| L.iter i; fixed 0; fixed 1 |]));
  p "a mismatched index rank remains conservatively recurring" (census [| fixed 1; fixed 0 |]);
  p "a gather from another node is not a cell recurrence"
    (not (census ~tn:other [| fixed 0; fixed 0; fixed 0 |]));
  let dyn_value = (L.get a target, prec) in
  p "a disjoint gather still counts a selector that reads the written cell"
    (census ~dyn_value [| fixed 1; fixed 0; fixed 0 |]);
  p "a gather from another node still counts a selector reading the written cell"
    (census ~tn:other ~dyn_value [| fixed 0; fixed 0; fixed 0 |])
