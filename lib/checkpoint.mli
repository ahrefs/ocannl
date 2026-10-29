(** {1 Resumable training checkpoints (gh-ocannl-96)}

    Stopping a training loop and resuming it so that it continues exactly as the uninterrupted run
    would have takes two kinds of state:

    - the {b device state} the loop's routines carry from one run to the next: the parameters, the
      optimizer's moments (an SGD momentum buffer), accumulators -- see {!training_state};
    - the {b host-side progress} the loop itself keeps: the step and epoch counters, the position in
      the data, the host random generator used to shuffle it, and anything else the loop needs --
      see {!type-progress}.

    {!save} publishes both in one checkpoint file (a {!Persistence} checkpoint whose header carries
    the progress), in one atomic write, so a crash mid-save leaves the previous checkpoint whole and
    never a state from one step with the progress of another. {!restore} puts the device state back
    into a freshly built program's context and returns the progress.

    {2 What the resumed program must do}

    A checkpoint holds values, not the program. The resumed process rebuilds the same model and the
    same routines -- the same construction order, so the tensor nodes get the same ids, which is how
    the checkpoint matches them -- then initializes the parameters as usual ([Train.init_params]),
    restores, and continues the loop from the returned progress: it re-arms the data position, the
    step-dependent inputs (a scheduled learning rate is set from [progress.step], see
    [Train.scheduled_learning_rate]) and the shuffling generator.

    Device-side randomness needs nothing saved: OCANNL's random operations are counter-based
    (threefry over the random seed, the node and a counter), so a resumed program with the same seed
    ([fixed_state_for_init], or {!Ocannl_tensor.Tensor.set_random_seed}) whose counters derive from
    the restored step draws the same numbers as the uninterrupted run.

    {2 A minimal resume}

    {[
      let state = Checkpoint.training_state ~routines:[ sgd_routine ] loss in
      (* ... every [k] steps: *)
      Checkpoint.save ~ctx
        ~progress:{ step; epoch; position; host_rng = Some rng; extra = [] }
        state path;
      (* ... in the resumed process, after rebuilding the program and [Train.init_params]: *)
      let ctx, progress = Checkpoint.restore ~ctx state path in
      let rng = Option.value_exn progress.host_rng in
      (* continue from progress.step / progress.epoch / progress.position *)
    ]} *)

type progress = {
  step : int;  (** Optimizer steps completed. *)
  epoch : int;  (** The epoch in progress. *)
  position : int;  (** Where in the epoch's data the next batch starts, in the loop's own units. *)
  host_rng : Stdlib.Random.State.t option;
      (** The host generator the loop draws from (to shuffle the data, say), captured at {!save}
          through [Stdlib.Random.State.to_binary_string]; {!restore} returns a fresh generator in
          the captured state. To checkpoint the global [Stdlib.Random] generator, pass
          [Stdlib.Random.get_state ()] and [Stdlib.Random.set_state] the restored one. (Base's
          [Random.State.t] is abstract and cannot be captured: draw from a [Stdlib] one.) *)
  extra : (string * string) list;
      (** Anything else the loop keeps on the host -- an early-stopping counter, a best loss -- as
          key-value pairs, returned by {!restore} as saved. *)
}

val training_state :
  ?extra:Ir.Tnode.t list ->
  routines:Context.routine list ->
  Ocannl_tensor.Tensor.t ->
  Ir.Tnode.t_set
(** [training_state ~routines loss] is the device state a resumable checkpoint has to hold: the
    values of [loss.params] (the state set, frozen parameters included), plus what each of the
    loop's [routines] carries from one run to the next ({!carried_state}), plus [?extra].

    The routines are the compiled ones the loop runs, so the optimizer's moments are found however
    the optimizer was written: an update that reads its own state without the clearing [=:] prefix
    declares it persistent (see [docs/syntax_extensions.md]), and that is what is read here. Nothing
    is matched by label. *)

val carried_state : Context.routine -> Ir.Tnode.t_set
(** The nodes a compiled routine carries between runs: those it writes whose value on entry it also
    reads -- in the lowered code, the materialized nodes that are both read before written and
    written (an optimizer moment, a parameter it updates, an accumulator). A node it only reads (a
    dataset, a host-set learning rate) is an input the program re-establishes; a node it writes
    before any read (a gradient zeroed each step) carries nothing. Raises [Invalid_argument] for a
    routine recording no kernel segments. *)

val save : ctx:Context.t -> progress:progress -> Ir.Tnode.t_set -> string -> unit
(** [save ~ctx ~progress state path] writes the [state] nodes' current values from [ctx], with
    [progress], to the checkpoint at [path], replacing any file there. The write is atomic (see
    {!Persistence.save}): a failed save leaves the previous checkpoint intact. The checkpoint also
    records which nodes the state was, for {!restore} to check. Raises [Invalid_argument] on an
    empty state or duplicate [progress.extra] keys. *)

val restore : ctx:Context.t -> Ir.Tnode.t_set -> string -> Context.t * progress
(** [restore ~ctx state path] uploads the checkpoint's values of the [state] nodes into [ctx] and
    returns the updated context with the saved progress.

    Raises if the file is not a training checkpoint (a plain {!Persistence.save} records no
    progress), and -- before touching [ctx] -- if [state] is not the set of nodes the checkpoint was
    saved from: a smaller set would restore the parameters while silently restarting the rest (say
    the optimizer moments) from their initialization, a different run. The message lists the nodes
    on either side. Precision and shape mismatches raise as in {!Persistence.restore}. *)
