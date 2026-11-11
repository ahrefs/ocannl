open Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module Asgns = Ir.Assignments
open Verdict.Claims

(* gh-ocannl-2157: a forward graph reading a random seed whose code another graph absorbed -- a
   sibling compiled into another routine, an unused draw, under the current singleton or a scoped or
   replaced one -- carries the seed's fetch in the code it is handed, so it links and draws exactly
   as the routine that reads the seed from its owner's lineage does. Every construction below is
   rebuilt from a fresh session, so its node ids -- and with them the draws -- agree across the ways
   it is run. *)

let contexts = ref []

let track ctx =
  contexts := ctx :: !contexts;
  ctx

let release_all () =
  List.iter !contexts ~f:Context.release;
  contexts := []

let x () =
  Tensor.term_init
    (Array.init 16 ~f:(fun i -> Float.of_int (i + 1)))
    ~grad_spec:Tensor.Prohibit_grad ()

let run ?(name = "fsf") ctx comp (y : Tensor.t) =
  Train.set_materialized y.value;
  let c, r = Context.compile ~name ctx comp Train.IDX.empty in
  let c = track (Context.run (track c) r) in
  (c, Context.get_values c y.value)

(* How the seed's reader is built: [owner] embeds the seed's code (it is built first), [reader]
   draws from the same seed without embedding it. *)
type variant =
  | Singleton  (** No scope: the current singleton, no parameters. *)
  | Scoped  (** [Nn_blocks.dropout] twice under [with_saved_random_seed]. *)
  | Unwound  (** The same scope left by an exception. *)
  | Replaced  (** No scope: [set_random_seed] replaces the singleton afterwards. *)

let is_singleton = function Singleton -> true | Scoped | Unwound | Replaced -> false

type built = { rs : Tensor.t; owner : Tensor.t; reader : Tensor.t }

let build ~seed variant =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed:42 ();
  let graphs () =
    Tensor.set_random_seed ~seed ();
    let rs = Tensor.get_random_seed () in
    (* Each application its own step counter: the graphs share nothing but the seed. *)
    let drop x = Ocannl.Nn_blocks.dropout ~rate:0.5 () ~train_step:(Some (NTDSL.number 3.)) x in
    let owner = drop (x ()) in
    let reader = drop (x ()) in
    { rs; owner; reader }
  in
  match variant with
  | Singleton ->
      Tensor.set_random_seed ~seed ();
      let rs = Tensor.get_random_seed () in
      let owner = NTDSL.O.(x () *. NTDSL.uniform () ()) in
      let reader = NTDSL.O.(x () *. NTDSL.uniform () ()) in
      { rs; owner; reader }
  | Scoped -> Tensor.with_saved_random_seed graphs
  | Unwound -> (
      let cell = ref None in
      match
        Tensor.with_saved_random_seed (fun () ->
            cell := Some (graphs ());
            failwith "unwind the seed scope")
      with
      | () -> assert false
      | exception Failure _ -> Option.value_exn !cell)
  | Replaced ->
      let built = graphs () in
      Tensor.set_random_seed ~seed:43 ();
      built

type mode =
  | Reference
      (** The reader's raw code, run in its owner's lineage: what a program could already do. *)
  | Fresh  (** The reader's handout, from a context its owner never ran in. *)
  | To_routine  (** The same through [Train.to_routine]. *)
  | Lineage  (** The handout in the owner's lineage: the fetch re-runs there. *)
  | Unconsumed_owner  (** The owner never consumed: still a root, its code pending. *)
  | Embedded_both  (** One computation embedding both graphs' code, as [%cd] does. *)
  | Checkpointed  (** A checkpoint and rewind around a later construction, then [Fresh]. *)

type drawn = { owner_values : float array option; values : float array }

let drawn ~seed variant mode =
  Exn.protect ~finally:release_all ~f:(fun () ->
      let { rs; owner; reader } = build ~seed variant in
      let owner_routine = ref None in
      let run_owner () =
        Train.set_materialized owner.value;
        let c, r =
          Context.compile ~name:"fsf_owner"
            (track (Context.auto ()))
            (Train.forward owner) Train.IDX.empty
        in
        owner_routine := Some r;
        let c = track (Context.run (track c) r) in
        (c, Some (Context.get_values c owner.value))
      in
      match mode with
      | Reference ->
          let c, owner_values = run_owner () in
          ignore (rs : Tensor.t);
          let _, values = run c reader.Tensor.forward reader in
          { owner_values; values }
      | Fresh ->
          let _, owner_values = run_owner () in
          let _, values = run (track (Context.auto ())) (Train.forward reader) reader in
          { owner_values; values }
      | To_routine ->
          let _, owner_values = run_owner () in
          Train.set_materialized reader.value;
          let c, r =
            Train.to_routine (track (Context.auto ())) Train.IDX.empty (Train.forward reader)
          in
          let c = track (Context.run (track c) r) in
          { owner_values; values = Context.get_values c reader.value }
      | Lineage ->
          let c, _ = run_owner () in
          let c, values = run c (Train.forward reader) reader in
          (* The owner's routine again, after the reader re-wrote the seed: its draws stand. *)
          let c = track (Context.run c (Option.value_exn !owner_routine)) in
          { owner_values = Some (Context.get_values c owner.value); values }
      | Unconsumed_owner ->
          let _, values = run (track (Context.auto ())) (Train.forward reader) reader in
          ignore (Sys.opaque_identity owner : Tensor.t);
          { owner_values = None; values }
      | Embedded_both ->
          Train.set_materialized owner.value;
          let comp =
            Asgns.sequence [ Tensor.take_forward_code owner; Tensor.take_forward_code reader ]
          in
          let c, values = run (track (Context.auto ())) comp reader in
          { owner_values = Some (Context.get_values c owner.value); values }
      | Checkpointed ->
          let _, owner_values = run_owner () in
          let ck = Tensor.checkpoint ~label:"fsf" () in
          ignore (NTDSL.O.(x () *. NTDSL.uniform () ()) : Tensor.t);
          Tensor.rewind ck;
          let _, values = run (track (Context.auto ())) (Train.forward reader) reader in
          { owner_values; values })

let refused msg =
  Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
  None

let attempt_session f =
  match f () with
  | v -> Some v
  | exception Utils.User_error msg -> refused msg
  | exception Tensor.Session_error (msg, _) -> refused msg

let variants =
  [ ("singleton", Singleton); ("scoped", Scoped); ("unwound", Unwound); ("replaced", Replaced) ]

let modes =
  [
    ("a fresh context", Fresh);
    ("Train.to_routine", To_routine);
    ("the owner's lineage", Lineage);
    ("an unconsumed owner", Unconsumed_owner);
    ("one computation embedding both", Embedded_both);
    ("a checkpoint rewound", Checkpointed);
  ]

let () =
  List.iter variants ~f:(fun (vname, variant) ->
      Verdict.case
        ("a " ^ vname ^ " seed read by a forward graph whose code it does not own")
        (fun () ->
          let reference = drawn ~seed:7 variant Reference in
          let control = drawn ~seed:8 variant Reference in
          p "another seed draws a different reference"
            (not (Poly.equal reference.values control.values));
          if not (is_singleton variant) then
            p "the reference mask keeps some entries and drops others"
              (Array.exists reference.values ~f:(fun v -> Float.(v = 0.))
              && Array.exists reference.values ~f:(fun v -> Float.(v <> 0.)));
          List.iter modes ~f:(fun (mname, mode) ->
              match attempt_session (fun () -> drawn ~seed:7 variant mode) with
              | None -> p ("the reader is accepted through " ^ mname) false
              | Some d ->
                  p
                    ("through " ^ mname ^ ", the reader draws exactly the reference")
                    (Poly.equal d.values reference.values);
                  Option.iter d.owner_values ~f:(fun owner_values ->
                      p
                        ("through " ^ mname ^ ", the owner's draws are unchanged")
                        (Poly.equal (Some owner_values) reference.owner_values)))))

(* One forward program reading two registered seeds, each absorbed by its own discarded draw. *)
let two_seeds ~caller_seeded =
  Exn.protect ~finally:release_all ~f:(fun () ->
      Tensor.unsafe_reinitialize ();
      let scoped seed =
        Tensor.with_saved_random_seed (fun () ->
            Tensor.set_random_seed ~seed ();
            let rs = Tensor.get_random_seed () in
            ignore (Sys.opaque_identity (NTDSL.uniform () ()) : Tensor.t);
            (rs, NTDSL.uniform () ()))
      in
      let rs7, u7 = scoped 7 and rs9, u9 = scoped 9 in
      let y = NTDSL.O.(x () *. (u7 + u9)) in
      if caller_seeded then
        let c, _ = run ~name:"fsf_rs7" (track (Context.auto ())) rs7.Tensor.forward rs7 in
        let c, _ = run ~name:"fsf_rs9" c rs9.Tensor.forward rs9 in
        snd (run c y.Tensor.forward y)
      else snd (run (track (Context.auto ())) (Train.forward y) y))

let () =
  Verdict.case "one forward program reading two seeds it does not own" (fun () ->
      let reference = two_seeds ~caller_seeded:true in
      match attempt_session (fun () -> two_seeds ~caller_seeded:false) with
      | None -> p "the program is accepted" false
      | Some values ->
          p "it draws exactly the caller-seeded reference" (Poly.equal values reference))

let count_seed_prepends (comp : Asgns.comp) ~(rest : Asgns.t) ~(seed : Asgns.t) =
  let rec go n (a : Asgns.t) =
    if phys_equal a rest then Some n
    else match a with Seq (s, more) when phys_equal s seed -> go (n + 1) more | _ -> None
  in
  go 0 comp.asgns

let () =
  Verdict.case "the handout's shape" (fun () ->
      Exn.protect ~finally:release_all ~f:(fun () ->
          let { rs; owner; reader } = build ~seed:7 Scoped in
          let plain = NTDSL.O.(x () *. NTDSL.number 2.) in
          let owner_handout = Tensor.consume_forward_code owner in
          let handout = Tensor.consume_forward_code reader in
          p "the reader's own code reads the seed and does not write it"
            (Set.mem (fst (Asgns.collect_nodes_guess_output reader.forward.asgns)) rs.value
            && not (Set.mem (Asgns.collect_written reader.forward.asgns) rs.value));
          p "the owner's handout is its own code: it writes the seed already"
            (phys_equal owner_handout owner.forward);
          p "the reader's handout prepends the seed's fetch exactly once"
            (Option.equal Int.equal
               (count_seed_prepends handout ~rest:reader.forward.asgns ~seed:rs.forward.asgns)
               (Some 1));
          p "the reader's handout allocates the seed"
            (Set.mem handout.embedded_nodes rs.value
            && not (Set.mem reader.forward.embedded_nodes rs.value));
          p "the reader's handout keeps its fragment metadata"
            (Set.equal handout.owned reader.forward.owned
            && Set.equal handout.requires reader.forward.requires);
          p "a repeated handout is the same computation"
            (phys_equal handout (Tensor.consume_forward_code reader));
          p "a graph reading no seed is handed its own code"
            (phys_equal (Tensor.consume_forward_code plain) plain.forward)))
