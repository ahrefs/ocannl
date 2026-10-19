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
   is caught by the reachability claim, which the agreement claim cannot see. *)

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
}

(* The lowest compute capability every CUDA arm emits at: above it the descriptor and the resolver
   both stop changing. *)
let cuda_cc =
  Option.value_exn (List.max_elt (List.map Cuda.all_of_arm ~f:Cuda.arm_floor) ~compare:Int.compare)

let tables =
  let cpu = Context.cpu () in
  [
    {
      name = "cuda";
      mma = Cuda.capability ~cc:cuda_cc;
      arm = Cuda.mma_arm;
      device_floor = Some cuda_cc;
    };
    {
      name = "metal";
      mma = Some Ir.Metal_mma.capability;
      arm = Ir.Metal_mma.mma_arm;
      device_floor = None;
    };
    {
      name = "hip";
      mma = Some Ir.Hip_mma.capability;
      arm = Ir.Hip_mma.mma_arm;
      device_floor = None;
    };
    {
      name = "cc";
      mma = (Context.hardware_limits cpu).BI.mma;
      arm = (Context.codegen_capabilities cpu).BI.mma_arm;
      device_floor = None;
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
}

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
      match table.mma with
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
  N.set_policy saved
