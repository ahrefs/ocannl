open Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train

(* Keep graph identities identical: random draws also depend on node ids. The first draw absorbs the
   singleton seed's forward code, but is not a parameter of the second graph. *)
let draw ~seed ~initialize =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed ();
  let rs = Tensor.get_random_seed () in
  let unused = NTDSL.uniform () () in
  let w = TDSL.param "probe_w" ~output_dims:[ 8 ] () in
  let base = Context.auto () in
  let seed_ctx = ref base in
  Exn.protect
    ~finally:(fun () ->
      Context.release !seed_ctx;
      Context.release base)
    ~f:(fun () ->
      if initialize then (
        let c, r = Context.compile ~name:"probe_seed" base rs.Tensor.forward Train.IDX.empty in
        seed_ctx := c;
        seed_ctx := Context.run c r);
      let ctx = Train.init_params !seed_ctx Train.IDX.empty w in
      Exn.protect
        ~finally:(fun () -> Context.release ctx)
        ~f:(fun () ->
          let values = Context.get_values ctx w.Tensor.value in
          let repeated = Train.init_params ~reinit_all:true ctx Train.IDX.empty w in
          Exn.protect
            ~finally:(fun () -> Context.release repeated)
            ~f:(fun () ->
              let again = Context.get_values repeated w.Tensor.value in
              ignore (Stdlib.Sys.opaque_identity unused : Tensor.t);
              (values, again))))

let refused ~node f =
  match f () with
  | _ -> false
  | exception Utils.User_error msg ->
      String.is_substring msg ~substring:"required input nodes not initialized"
      && String.is_substring msg ~substring:node

let () =
  Verdict.case "a stolen library seed is initialized" (fun () ->
      let automatic, repeated = draw ~seed:42 ~initialize:false in
      let seeded, _ = draw ~seed:42 ~initialize:true in
      let zero, _ = draw ~seed:0 ~initialize:true in
      Verdict.p "a stolen seed draws exactly the explicitly seeded reference"
        (Poly.equal automatic seeded);
      Verdict.p "a stolen seed does not draw from allocation zero" (not (Poly.equal automatic zero));
      Verdict.p "reinitializing parameters replays the same seed literal"
        (Poly.equal automatic repeated))

let input_case ~zero_valid =
  Tensor.unsafe_reinitialize ();
  let x =
    Tensor.term ~label:[ "embedded_input" ] ~grad_spec:Tensor.Prohibit_grad ~output_dims:[ 4 ] ()
  in
  if zero_valid then Ir.Tnode.mark_zero_valid x.value;
  let y = NTDSL.pointmul x (NTDSL.number 2.) () in
  Train.set_materialized x.value;
  Train.set_materialized y.value;
  let comp = Tensor.consume_forward_code y in
  let ctx, routine =
    Context.compile ~name:"embedded_input" (Context.auto ()) comp Train.IDX.empty
  in
  Verdict.p
    ((if zero_valid then "zero-valid" else "unmarked") ^ " input is embedded and read-only")
    (Set.mem comp.embedded_nodes x.value && not (Set.mem routine.outputs x.value));
  if zero_valid then (
    let ctx = Context.run ctx routine in
    Verdict.p "declared zero-valid embedded input runs from zero"
      (Poly.equal (Context.get_values ctx y.value) [| 0.; 0.; 0.; 0. |]);
    Context.release ctx)
  else (
    Verdict.p "unwritten embedded read-only input is refused"
      (refused ~node:"embedded_input" (fun () -> Context.run ctx routine));
    let ctx = Context.run (Context.set_values ctx x.value [| 1.; 3.; 7.; 11. |]) routine in
    Verdict.p "uploading the embedded input permits the correct result"
      (Poly.equal (Context.get_values ctx y.value) [| 2.; 6.; 14.; 22. |]);
    Context.release ctx)

let () =
  input_case ~zero_valid:false;
  input_case ~zero_valid:true;
  Tensor.unsafe_reinitialize ();
  (* Larger than the literal-inline limit, hence registered host data. *)
  let values = Array.init 32 ~f:(fun i -> Float.of_int (i + 1)) in
  let x = Tensor.term_init values ~grad_spec:Tensor.Prohibit_grad () in
  let y = NTDSL.pointmul x (NTDSL.number 2.) () in
  Train.set_materialized y.value;
  let comp = Tensor.consume_forward_code y in
  Verdict.p "the literal is embedded and host-initialized"
    (Set.mem comp.embedded_nodes x.value && Ir.Host_inits.mem x.value);
  let ctx, routine =
    Context.compile ~name:"embedded_literal" (Context.auto ()) comp Train.IDX.empty
  in
  let ctx = Context.run ctx routine in
  Verdict.p "embedded host literal initializes at link and runs"
    (Poly.equal (Context.get_values ctx y.value) (Array.map values ~f:(fun x -> 2. *. x)));
  Context.release ctx

let () =
  Tensor.unsafe_reinitialize ();
  (* Register host data directly so this fixture can write the shared buffer through the public API.
     High-level term_init literals are immutable to host writes (gh-ocannl-2021); this is the same
     mutable host-initialized fixture as context_handle_refusals. *)
  let x =
    Tensor.term ~label:[ "embedded_shared" ] ~grad_spec:Tensor.Prohibit_grad ~output_dims:[ 32 ] ()
  in
  let nd =
    Ir.Ndarray.create_array ~debug:"embedded_shared" Ir.Ops.single ~dims:[| 32 |] ~padding:None
  in
  Ir.Ndarray.fill_from_float nd 3.;
  Ir.Host_inits.register x.value (lazy nd);
  let y = NTDSL.pointmul x (NTDSL.number 2.) () in
  Train.set_materialized x.value;
  Train.set_materialized y.value;
  let comp = Tensor.consume_forward_code y in
  let ca, ra = Context.compile ~name:"embedded_shared_a" (Context.auto ()) comp Train.IDX.empty in
  let cb, rb = Context.compile ~name:"embedded_shared_b" (Context.auto ()) comp Train.IDX.empty in
  Verdict.p "embedded host data is a shared read, not a required input"
    (Set.mem comp.embedded_nodes x.value
    && Set.mem ra.shared_reads x.value && Set.mem rb.shared_reads x.value
    && not (Set.mem rb.inputs x.value));
  let cb = Context.run cb rb in
  Verdict.p "the shared reader first sees the host initialization"
    (Poly.equal (Context.get_values cb y.value) (Array.create ~len:32 6.));
  let ca = Context.set_values ca x.value (Array.init 32 ~f:(fun i -> Float.of_int (i + 10))) in
  Verdict.p "another root writing the embedded shared buffer makes the reader stale"
    (match Context.run cb rb with
    | _ -> false
    | exception Utils.User_error msg ->
        String.is_substring msg ~substring:"embedded_shared"
        && String.is_substring msg ~substring:"are host-initialized literals");
  Context.release cb;
  Context.release ca

(* gh-ocannl-2119: a seed captured by a graph built under a scoped (or replaced) singleton, whose
   forward code a discarded draw absorbed, is initialized by the parameters' initialization
   whichever singleton is current when it runs. Each construction below is rebuilt from a fresh
   session, so the node ids -- and with them the draws -- of one variant agree across the ways it is
   initialized. *)
type init_mode =
  | Outside  (** After the scope restored another singleton. *)
  | Caller_seeded
      (** Outside too, with the caller writing the captured seed first: the reference. *)
  | Inside  (** While the captured seed is still the singleton. *)

type scope =
  | Scoped  (** [with_saved_random_seed] returning normally. *)
  | Unwound  (** [with_saved_random_seed] left by an exception. *)
  | Replaced  (** No scope: [set_random_seed] replaces the singleton afterwards. *)

type drawn = {
  values : float array;
  unchanged : float array;  (** After an initialization that skips the initialized parameter. *)
  again : float array;  (** After [~reinit_all:true]. *)
}

let captured ?outer ?(scope = Scoped) ?(nested = false) ?(owner = true) ~inner mode =
  Tensor.unsafe_reinitialize ();
  (* The outer singleton's own code is absorbed elsewhere too: initialization must supply the
     captured seed the parameters read, independently of this unread current singleton. *)
  let outer_draw =
    Option.map outer ~f:(fun seed ->
        Tensor.set_random_seed ~seed ();
        NTDSL.uniform () ())
  in
  let base = Context.auto () in
  let contexts = ref [ base ] in
  let track ctx =
    contexts := ctx :: !contexts;
    ctx
  in
  Exn.protect
    ~finally:(fun () -> List.iter !contexts ~f:Context.release)
    ~f:(fun () ->
      let inside_ctx = ref None in
      let build () =
        Tensor.set_random_seed ~seed:inner ();
        let rs = Tensor.get_random_seed () in
        (* The discarded seed-owning draw: it absorbs the captured seed's forward code. *)
        let draw = if owner then Some (NTDSL.uniform () ()) else None in
        let w = TDSL.param "probe_w" ~output_dims:[ 8 ] () in
        (match mode with
        | Inside -> inside_ctx := Some (track (Train.init_params base Train.IDX.empty w))
        | Outside | Caller_seeded -> ());
        (rs, draw, w)
      in
      let rs, draw, w =
        match scope with
        | Scoped -> Tensor.with_saved_random_seed build
        | Unwound -> (
            let cell = ref None in
            match
              Tensor.with_saved_random_seed (fun () ->
                  cell := Some (build ());
                  failwith "unwind the seed scope")
            with
            | () -> assert false
            | exception Failure _ -> Option.value_exn !cell)
        | Replaced ->
            let built = build () in
            Tensor.set_random_seed ~seed:43 ();
            built
      in
      if nested then
        Tensor.with_saved_random_seed (fun () ->
            Tensor.set_random_seed ~seed:9 ();
            ignore (Stdlib.Sys.opaque_identity (NTDSL.uniform () ()) : Tensor.t));
      let ctx =
        match (mode, !inside_ctx) with
        | Inside, Some ctx -> ctx
        | Outside, _ -> track (Train.init_params base Train.IDX.empty w)
        | Caller_seeded, _ ->
            let c, r =
              Context.compile ~name:"captured_seed" base rs.Tensor.forward Train.IDX.empty
            in
            let c = track (Context.run (track c) r) in
            track (Train.init_params c Train.IDX.empty w)
        | Inside, None -> assert false
      in
      let values = Context.get_values ctx w.Tensor.value in
      (* Already initialized: skipped, whichever singleton is current. *)
      let skipped = track (Train.init_params ctx Train.IDX.empty w) in
      let unchanged = Context.get_values skipped w.Tensor.value in
      let repeated = track (Train.init_params ~reinit_all:true skipped Train.IDX.empty w) in
      let again = Context.get_values repeated w.Tensor.value in
      ignore (Stdlib.Sys.opaque_identity (draw, outer_draw) : Tensor.t option * Tensor.t option);
      { values; unchanged; again })

let () =
  Verdict.case "a seed captured by a discarded draw, initialized after the scope" (fun () ->
      let reference = captured ~outer:42 ~inner:7 Caller_seeded in
      let inside = captured ~outer:42 ~inner:7 Inside in
      let outside = captured ~outer:42 ~inner:7 Outside in
      let other = captured ~outer:42 ~inner:8 Caller_seeded in
      Verdict.p "the reference initialized inside the scope draws as the caller-seeded one"
        (Poly.equal inside.values reference.values);
      Verdict.p "the captured seed draws exactly the seed-owning reference after the scope"
        (Poly.equal outside.values reference.values);
      Verdict.p "another captured seed draws differently"
        (not (Poly.equal reference.values other.values));
      Verdict.p_all "an initialized parameter is skipped unchanged after the scope"
        [ reference; inside; outside ] ~f:(fun d -> Poly.equal d.unchanged d.values);
      Verdict.p "reinitializing outside the scope replays the captured seed"
        (Poly.equal outside.again reference.values && Poly.equal inside.again reference.values))

let () =
  Verdict.case "the captured seed's restore paths" (fun () ->
      let draws ?outer ?scope ?nested () =
        let reference = captured ?outer ?scope ?nested ~inner:7 Caller_seeded in
        let outside = captured ?outer ?scope ?nested ~inner:7 Outside in
        Poly.equal outside.values reference.values
      in
      Verdict.p "with no singleton to restore" (draws ());
      Verdict.p "with an outer singleton of the same integer" (draws ~outer:7 ());
      Verdict.p "after a nested scope restored another singleton" (draws ~outer:42 ~nested:true ());
      Verdict.p "after the scope was left by an exception" (draws ~outer:42 ~scope:Unwound ());
      Verdict.p "after a direct replacement of the singleton" (draws ~scope:Replaced ()))

let () =
  Verdict.case "a parameter graph owning its scoped seed" (fun () ->
      let reference = captured ~outer:42 ~owner:false ~inner:7 Caller_seeded in
      let inside = captured ~outer:42 ~owner:false ~inner:7 Inside in
      let outside = captured ~outer:42 ~owner:false ~inner:7 Outside in
      let other = captured ~outer:42 ~owner:false ~inner:8 Outside in
      Verdict.p "an owned scoped seed draws the same inside and outside the scope"
        (Poly.equal inside.values outside.values && Poly.equal outside.values reference.values);
      Verdict.p "an owned scoped seed of another value draws differently"
        (not (Poly.equal outside.values other.values));
      Verdict.p "reinitializing an owned scoped seed replays it"
        (Poly.equal outside.again outside.values))
