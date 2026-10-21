(* Every backend's advertised tensor-core formats correspond to its arm table, as a pure query
   (gh-ocannl-1246).

   A backend publishes two tables about its tile-MMA emission. The capability descriptor
   ([Backend_intf.mma_capability]: [mma_format_tiles] and the wide-accumulator scopes) is what
   autotune seeds tensorized candidates from; the arm table ([codegen_capabilities.mma_arm], the
   combination resolver both emitting hooks consult) is what the emission renders. A storage triple
   the descriptor admits with no arm is a candidate timed as the scalar fallback under a tensorized
   label (gh-ocannl-545); an arm the descriptor never admits is a rendering no seed reaches.
   [schedule_mma_matmul]'s format-triple leg ties them by rendering, which needs the matching GPU;
   here both tables are evaluated as functions, with no device and no toolkit, for every backend:
   CUDA's [Ir.Cuda_mma] at a compute capability that meets every arm's floor (the floors' own axis
   is [cuda_mma_cutoffs]'), Metal's [Ir.Metal_mma] and HIP's [Ir.Hip_mma] (the descriptors their
   backends install when the device set qualifies), and cc's live tables, which advertise no format
   and have no arm.

   Nothing below names a format, an arm or a policy. The descriptor is read through the consumer's
   own lookup ([Autotune.mma_tile_for_precisions_in_scope], the policy-resolved format triples and
   the wide-accumulator scope gate the seeds use), over every storage triple, emission scope and
   numerics policy ([Numerics.all]). Dropping an advertised triple, an arm, or a wide scope makes
   the two sides disagree somewhere on that grid; an advertised triple no storage triple resolves to
   is caught by the reachability claim, which the agreement claim cannot see.

   The same grid pins each backend's description of its operand loads (gh-ocannl-1456): the
   [mma_operand_fragments] record the cost model's tie-break counts by
   ([Ir.Cost_model.statement_operand_loads]) is there exactly where an arm is. Where the arm's
   reduction loop is rendered by its pure module -- Metal's and HIP's arms, CUDA's wmma ones (its
   inline-PTX bodies live in the [cudajit]-gated backend) -- that loop is rendered at several
   geometries and orientations and its operand loads counted as each lane executes them: the
   statements that read [__mma_ap] or [__mma_bp], weighted by their enclosing loop extents. The
   count must be the cost model's for that statement under the backend's own description, so a
   renderer that starts reloading a fragment, or a description whose tile, reload or whole-fragment
   flag drifts from the loop, disagrees here with no device. The counter reads line-shaped forms
   only: one loop header ([for (...; __x < N; ...) {]), closing brace or statement per line, as the
   renderers emit them. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module N = Ir.Numerics
module Cuda = Ir.Cuda_mma

type table = {
  name : string;
  mma : BI.mma_capability option;
  arm :
    a_prec:Ir.Ops.prec ->
    b_prec:Ir.Ops.prec ->
    d_prec:Ir.Ops.prec ->
    scope:BI.mma_emission_scope ->
    BI.mma_arm option;
  device_floor : int option;
      (** The device architecture the descriptor describes, where arms check a floor (CUDA). *)
  fragments :
    a_prec:Ir.Ops.prec ->
    b_prec:Ir.Ops.prec ->
    d_prec:Ir.Ops.prec ->
    scope:BI.mma_emission_scope ->
    BI.mma_fragments option;
      (** The backend's [mma_operand_fragments]. *)
  loads :
    a_prec:Ir.Ops.prec ->
    b_prec:Ir.Ops.prec ->
    d_prec:Ir.Ops.prec ->
    scope:BI.mma_emission_scope ->
    ta:bool ->
    tb:bool ->
    a:BI.mma_operand_place ->
    b:BI.mma_operand_place ->
    bool * bool;
      (** The backend's [mma_operand_loads]. *)
  k_loop :
    a_prec:Ir.Ops.prec ->
    b_prec:Ir.Ops.prec ->
    d_prec:Ir.Ops.prec ->
    scope:BI.mma_emission_scope ->
    (ta:bool -> tb:bool -> m:int -> n:int -> k:int -> string list) option;
      (** The reduction loop the arm's hooks render, where its pure module renders it. *)
}

(* The lowest compute capability every CUDA arm emits at: above it the descriptor and the resolver
   both stop changing. *)
let cuda_cc =
  Option.value_exn (List.max_elt (List.map Cuda.all_of_arm ~f:Cuda.arm_floor) ~compare:Int.compare)

(* Leading dimensions no tile constraint cares about: the counter reads loop structure, not
   addresses. *)
let lda = 64
let ldb = 72

(* The loop a backend's pure module renders for an arm, keyed on the arm's presence in that module's
   own table. *)
let rendered_by arm render ~a_prec ~b_prec ~d_prec ~scope =
  Option.map (arm ~a_prec ~b_prec ~d_prec ~scope) ~f:(fun _ -> render)

let tables =
  let cpu = Context.cpu () in
  let cpu_caps = Context.codegen_capabilities cpu in
  [
    {
      name = "cuda";
      mma = Cuda.capability ~cc:cuda_cc;
      arm = Cuda.mma_arm;
      device_floor = Some cuda_cc;
      fragments = Cuda.operand_fragments;
      loads = Cuda.operand_loads;
      k_loop =
        (fun ~a_prec ~b_prec ~d_prec ~scope ->
          match Cuda.resolve_arm ~a_prec ~b_prec ~d_prec ~scope with
          | Some (Cuda.Wmma kind) ->
              Some
                (fun ~ta ~tb ~m ~n ~k ->
                  Cuda.wmma_k_loop_lines kind ~acc:"acc" ~ta ~tb ~lda ~ldb ~m ~n ~k)
          | Some (Cuda.Inline_ptx _) | None -> None);
    };
    {
      name = "metal";
      mma = Some Ir.Metal_mma.capability;
      arm = Ir.Metal_mma.mma_arm;
      device_floor = None;
      fragments = Ir.Metal_mma.operand_fragments;
      loads = Ir.Metal_mma.operand_loads;
      k_loop =
        rendered_by Ir.Metal_mma.mma_arm (fun ~ta ~tb ~m ~n ~k ->
            Ir.Metal_mma.k_loop_lines ~a_frag:"A" ~b_frag:"B" ~acc:"acc" ~ta ~tb ~lda ~ldb ~m ~n ~k);
    };
    {
      name = "hip";
      mma = Some Ir.Hip_mma.capability;
      arm = Ir.Hip_mma.mma_arm;
      device_floor = None;
      fragments = Ir.Hip_mma.operand_fragments;
      loads = Ir.Hip_mma.operand_loads;
      k_loop =
        rendered_by Ir.Hip_mma.mma_arm (fun ~ta ~tb ~m ~n ~k ->
            Ir.Hip_mma.k_loop_lines ~ab_typ:"AB" ~acc:"acc" ~ta ~tb ~lda ~ldb ~m ~n ~k);
    };
    {
      name = "cc";
      mma = (Context.hardware_limits cpu).BI.mma;
      arm = cpu_caps.BI.mma_arm;
      device_floor = None;
      fragments = cpu_caps.BI.mma_operand_fragments;
      loads = cpu_caps.BI.mma_operand_loads;
      k_loop = (fun ~a_prec:_ ~b_prec:_ ~d_prec:_ ~scope:_ -> None);
    };
  ]

let triples =
  List.concat_map Ir.Ops.all_precs ~f:(fun a_prec ->
      List.concat_map Ir.Ops.all_precs ~f:(fun b_prec ->
          List.map Ir.Ops.all_precs ~f:(fun d_prec -> (a_prec, b_prec, d_prec))))

let scopes = BI.all_of_mma_emission_scope

(* Both tables read the current numerics policy (the tf32 gate, the wide-accumulator modes), so each
   evaluation installs its policy first. *)
let under policy f =
  N.set_policy policy;
  f ()

(* The arm the hooks select, if the device meets its floor: a hook that declines its arm on
   capability renders the scalar fallback, never another arm. *)
let arm_admits table ~scope (a_prec, b_prec, d_prec) =
  match table.arm ~a_prec ~b_prec ~d_prec ~scope with
  | None -> false
  | Some { BI.arm_floor; _ } ->
      Option.for_all arm_floor ~f:(fun floor ->
          Option.value_map table.device_floor ~default:false ~f:(fun cc -> cc >= floor))

let descriptor_admits table ~scope (a_prec, b_prec, d_prec) =
  match table.mma with
  | None -> false
  | Some mma ->
      Option.is_some (Autotune.mma_tile_for_precisions_in_scope mma ~scope ~a_prec ~b_prec ~d_prec)

let show_triple (a, b, d) =
  String.concat ~sep:"*" (List.map [ a; b ] ~f:Ir.Ops.prec_string) ^ ">" ^ Ir.Ops.prec_string d

let show_key (a, b, d) =
  String.concat ~sep:","
    (List.map [ a; b; d ] ~f:(fun f -> Sexp.to_string (BI.sexp_of_mma_input_format f)))

type row = {
  policy : N.t;
  scope : BI.mma_emission_scope;
  triple : Ir.Ops.prec * Ir.Ops.prec * Ir.Ops.prec;
  keys : (BI.mma_input_format * BI.mma_input_format * BI.mma_input_format) list;
      (** The format triples the storage triple resolves to under [policy]. *)
  descriptor : bool;
  arm : bool;
  has_arm : bool;  (** The arm table has an arm, whatever its floor. *)
  described : bool;  (** The backend describes the arm's operand loads. *)
  counted : bool list;
      (** Per rendered geometry and orientation: whether the loop's load count is the cost model's
          under the description. Empty where no pure module renders the arm's loop. *)
}

(* The operand loads one lane executes running [lines]: the statements that read the [a] / [b] tile
   pointers, each weighted by the extents of the loops enclosing it. [None] on an unbalanced
   nest. *)
let executed_loads lines =
  let extent line =
    if String.is_prefix line ~prefix:"for (" && String.is_suffix line ~suffix:"{" then
      Option.bind (String.lsplit2 line ~on:'<') ~f:(fun (_, rest) ->
          Option.bind (String.lsplit2 rest ~on:';') ~f:(fun (n, _) ->
              Int.of_string_opt (String.strip n)))
    else None
  in
  let step (stack, a, b) line =
    let line = String.strip line in
    let weight = List.fold stack ~init:1 ~f:( * ) in
    match (extent line, stack) with
    | Some n, _ -> Some (n :: stack, a, b)
    | None, _ when String.is_suffix line ~suffix:"{" -> Some (1 :: stack, a, b)
    | None, _ :: rest when String.equal line "}" -> Some (rest, a, b)
    | None, [] when String.equal line "}" -> None
    | None, _ ->
        let reads ptr = if String.is_substring line ~substring:ptr then weight else 0 in
        Some (stack, a + reads "__mma_ap", b + reads "__mma_bp")
  in
  match
    List.fold lines
      ~init:(Some ([], 0, 0))
      ~f:(fun acc l -> Option.bind acc ~f:(fun acc -> step acc l))
  with
  | Some ([], a, b) -> Some (a, b)
  | Some _ | None -> None

let lanes table = Option.value_map table.mma ~default:1 ~f:(fun mma -> mma.BI.mma_simd_width)

(* Several tile multiples per axis, so a reload across a free axis or a tile drift changes the
   count; both orientations of each operand. *)
let geometries = [ (1, 1, 1); (2, 3, 1); (3, 2, 4); (2, 2, 3) ]

let counted table ~scope (a_prec, b_prec, d_prec) =
  match
    (table.k_loop ~a_prec ~b_prec ~d_prec ~scope, table.fragments ~a_prec ~b_prec ~d_prec ~scope)
  with
  | Some render, Some ({ BI.mf_tile = tm, tn, tk; _ } as fragments) ->
      List.concat_map geometries ~f:(fun (mt, nt, kt) ->
          List.concat_map [ false; true ] ~f:(fun ta ->
              List.map [ false; true ] ~f:(fun tb ->
                  let m = mt * tm and n = nt * tn and k = kt * tk in
                  (* The loop renders an accepted call, whose operands are plain. *)
                  let ldmatrix =
                    table.loads ~a_prec ~b_prec ~d_prec ~scope ~ta ~tb ~a:(`Shared, `Plain)
                      ~b:(`Shared, `Plain)
                  in
                  let model =
                    Ir.Cost_model.statement_operand_loads fragments ~ldmatrix ~a_prec ~b_prec
                      ~lanes:(lanes table) ~m ~n ~k
                  in
                  let got = executed_loads (render ~ta ~tb ~m ~n ~k) in
                  let show = Option.value_map ~f:(fun (a, b) -> Printf.sprintf "a %d b %d" a b) in
                  Option.equal (fun (a, b) (a', b') -> a = a' && b = b') model got
                  && Option.is_some got
                  ||
                  (Stdio.eprintf "%s %s %dx%dx%d ta %b tb %b: rendered loads %s, cost model %s\n"
                     table.name
                     (Sexp.to_string (BI.sexp_of_mma_emission_scope scope))
                     m n k ta tb (show got ~default:"unbalanced") (show model ~default:"none");
                   false))))
  | _ -> []

let rows table =
  List.concat_map N.all ~f:(fun policy ->
      under policy (fun () ->
          List.concat_map scopes ~f:(fun scope ->
              List.map triples ~f:(fun ((a_prec, b_prec, d_prec) as triple) ->
                  {
                    policy;
                    scope;
                    triple;
                    keys = Autotune.mma_format_triples ~a_prec ~b_prec ~d_prec;
                    descriptor = descriptor_admits table ~scope triple;
                    arm = arm_admits table ~scope triple;
                    has_arm = Option.is_some (table.arm ~a_prec ~b_prec ~d_prec ~scope);
                    described = Option.is_some (table.fragments ~a_prec ~b_prec ~d_prec ~scope);
                    counted = counted table ~scope triple;
                  }))))

let () =
  let saved = N.get () in
  let min_rows = List.length N.all * List.length scopes * List.length triples in
  List.iter tables ~f:(fun table ->
      let rows = rows table in
      p_all ~min:min_rows
        (table.name
       ^ ": the descriptor admits a storage triple, scope and policy exactly where the arm table \
          has an arm") rows ~f:(fun row ->
          Bool.equal row.descriptor row.arm
          ||
          (Stdio.eprintf "%s disagreement: %s %s under %s: descriptor %b, arm %b\n" table.name
             (Sexp.to_string (BI.sexp_of_mma_emission_scope row.scope))
             (show_triple row.triple)
             (Sexp.to_string (N.sexp_of_t row.policy))
             row.descriptor row.arm;
           false));
      (match table.mma with
      | Some mma when not (List.is_empty mma.BI.mma_format_tiles) ->
          (* The agreement above is not two empty sides: the backend renders some triple and refuses
             some other. *)
          p_exists (table.name ^ ": some storage triple has an arm") rows ~f:(fun row -> row.arm);
          p_exists (table.name ^ ": some storage triple has no arm") rows ~f:(fun row ->
              not row.arm);
          (* A key no storage triple resolves to under any policy is never looked up, so the
             agreement cannot see it: every advertised key is the resolved format of a storage
             triple that has an arm, in some scope under some policy. *)
          p_all
            (table.name
           ^ ": every advertised format triple is reached by a storage triple with an arm")
            mma.BI.mma_format_tiles ~f:(fun (key, _) ->
              List.exists rows ~f:(fun row ->
                  row.arm && List.mem row.keys key ~equal:BI.equal_mma_format_triple)
              ||
              (Stdio.eprintf "%s: advertised %s is reached by no storage triple with an arm\n"
                 table.name (show_key key);
               false));
          (* Staged twins are seeded per advertised staged layout, and the seeds look up the
             triple's tile first, so a staged layout without a format tile is never reached. *)
          if not (List.is_empty mma.BI.mma_staged_layouts) then
            p_all (table.name ^ ": every staged layout's format triple has a format tile")
              mma.BI.mma_staged_layouts ~f:(fun (key, _) ->
                List.Assoc.mem mma.BI.mma_format_tiles key ~equal:BI.equal_mma_format_triple)
      | Some _ | None -> ());
      (* Operand loads (gh-ocannl-1456): described exactly where the backend has an arm at all (the
         description is the arm's, floors aside), and counted as its loop renders them -- on every
         backend with an arm, so a pure module that stops rendering every one of its arms fails the
         population floor. *)
      p_all ~min:min_rows
        (table.name ^ ": the operand loads are described exactly where the arm table has an arm")
        rows ~f:(fun row ->
          Bool.equal row.described row.has_arm
          ||
          (Stdio.eprintf "%s: %s %s under %s: described %b\n" table.name
             (Sexp.to_string (BI.sexp_of_mma_emission_scope row.scope))
             (show_triple row.triple)
             (Sexp.to_string (N.sexp_of_t row.policy))
             row.described;
           false));
      if List.exists rows ~f:(fun row -> row.has_arm) then
        p_all
          (table.name
         ^ ": every rendered reduction loop loads its operands as the cost model counts them")
          (List.concat_map rows ~f:(fun row -> row.counted))
          ~f:Fn.id);
  N.set_policy saved
