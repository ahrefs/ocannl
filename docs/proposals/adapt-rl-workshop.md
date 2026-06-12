# Proposal: Adapt FUN OCaml 2025 RL Workshop to Idiomatic OCANNL

**Issue**: [gh-ocannl-402](https://github.com/ahrefs/ocannl/issues/402)
**Effort**: Large (5-8 days)
**Status**: Proposal draft

## Motivation

The FUN OCaml 2025 RL Workshop teaches reinforcement learning using the Raven/Fehu/Kaun/Rune
stack. Adapting its 3-stage progressive curriculum to OCANNL would:

1. Provide a complete RL example suite -- the first non-supervised-learning examples in the repo
2. Showcase OCANNL's define-then-run model in a domain where eager frameworks dominate
3. Supply material for workshop papers (gh-ocannl-299) demonstrating OCANNL's versatility
4. Exercise existing but untested blocks: `sokoban_cnn`, `softmax`, `uniform_at`

## Scope

Port the core workshop content (Stages 1-2) and Sokoban RL from the raven-sokoban repository.
Stage 3 (CNN debugging variants) is deferred -- it adds breadth but not new OCANNL patterns.

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Grid world environment | `test/training/rl_gridworld_env.ml` | Pure OCaml 5x5 grid world (~80 lines) |
| REINFORCE on grid world | `test/training/rl_reinforce_gridworld.ml` | MLP policy + baseline, full training loop |
| Sokoban environment | `test/training/rl_sokoban_env.ml` | Port of movement/box/win/deadlock logic (~250 lines) |
| Sokoban REINFORCE | `test/training/rl_reinforce_sokoban.ml` | CNN policy via `sokoban_cnn`, actor-critic |
| Updated RL slides | `docs/slides-RL-REINFORCE.md` | Replace pseudocode (lines 250-258) with OCANNL code |
| Workshop exercises | `docs/exercises-RL-OCANNL.md` | 5 progressive exercises adapted to OCANNL |

## Implementation Approach

### Phase 1: Grid World + Basic REINFORCE (Days 1-2)

**Environment** (`rl_gridworld_env.ml`): A standalone OCaml module with no framework
dependency. 5x5 grid, 4 actions (Up/Down/Left/Right), goal at (4,4), -0.1 step penalty,
+1.0 goal reward. Returns flat `float array` observations.

**REINFORCE example** (`rl_reinforce_gridworld.ml`): Follows the `bigram.ml` pattern:
- Policy network via `%op mlp` with `softmax` output over 4 actions
- Forward-only inference routine compiled with `%cd` + `.forward` for episode collection
- Action sampling using `uniform_at` + cumulative probability comparison (bigram.ml lines 84-101)
- Returns computed in pure OCaml after episode ends (discounted cumulative sum, gamma=0.99)
- Loss routine: `-G_t * log pi(a_t|s_t)` using one-hot action mask + einsum reduction
- Gradient update via `Train.grad_update` + `Train.sgd_update`, compiled once with `Train.to_routine`
- Exponential moving average baseline subtracted from returns to reduce variance

**Key design decision**: Process episodes step-by-step (not batched) to keep the first example
simple. Each training iteration: collect one episode, compute returns, feed step-by-step through
loss routine, run one gradient update.

### Phase 2: Sokoban Environment + CNN REINFORCE (Days 3-4)

**Environment** (`rl_sokoban_env.ml`): Port from raven-sokoban `verified.ml`:
- Grid representation as 2D int array (wall/floor/box/target/agent channels)
- Movement + box pushing mechanics with validation
- Win detection (all boxes on targets) and basic deadlock detection
- Level generators: corridor (1 box, linear), small room (1-2 boxes, 5x5-7x7)
- Observation as multi-channel float array suitable for CNN input

**Sokoban REINFORCE** (`rl_reinforce_sokoban.ml`):
```ocaml
let policy_value = sokoban_cnn ~label:["sokoban_agent"] ~num_actions:4 () in
(* policy_value returns (action_logits, value) -- actor-critic architecture *)
```
- Uses `sokoban_cnn` from `nn_blocks.ml` (lines 396-413) which already has action + value heads
- Advantage: `A_t = G_t - V(s_t)` using value head as learned baseline
- Separate losses: policy gradient loss + MSE value loss
- Curriculum: start with corridor levels, advance to rooms when success rate > 50%

### Phase 3: Documentation + Exercises (Days 5-6)

**RL slides update**: Replace generic pseudocode in `slides-RL-REINFORCE.md` (lines 250-258)
with working OCANNL code showing the define-then-run pattern:
```ocaml
(* Policy definition *)
let%op policy x = softmax ~spec:"a" () (mlp ~label:["policy"] ~hid_dims:[32; 16] () x) in
(* Compile inference routine *)
let%cd infer_step = policy_out.forward; { dice } =: uniform_at !@step_counter in
let infer_routine = Train.to_routine ctx bindings infer_step in
(* Compile training routine *)
let update = Train.grad_update policy_loss in
let sgd = Train.sgd_update ~learning_rate policy_loss in
let train_step = Train.to_routine ctx bindings (Asgns.sequence [update; sgd]) in
```

**Exercise document** (`exercises-RL-OCANNL.md`): 5 exercises mapping to workshop stages:
1. Random policy baseline (measure average return on grid world)
2. REINFORCE without baseline (observe high variance)
3. Add exponential moving average baseline
4. Replace with learned value baseline (actor-critic)
5. Apply CNN policy to Sokoban with curriculum learning

### Phase 4: Testing + Polish (Days 7-8)

- Verify examples compile with `dune build` and run showing learning progress
- Add dune stanzas in `test/training/dune` for each new executable
- Ensure no regression in existing `moons_demo`, `bigram`, `mlp_names`, `mlp_bn_names`, `circles_conv` tests
- Print learning curves (episode return vs episode number) to stdout

## Key Technical Decisions

### Episode variable length

OCANNL requires fixed tensor dimensions at compile time. For variable-length episodes:
- Process steps one at a time through the loss routine (simplest, used in Phase 1)
- Pad to max_episode_length for batched processing (optimization for Phase 2)
- The forward-only inference pass handles this naturally -- it is a single-step operation

### Log-softmax for numerical stability

Compute `log(softmax(x))` as `x - log(sum(exp(x)))` rather than separate softmax + log.
This avoids log(0) when action probabilities are tiny. OCANNL has both `log` and `exp`
operations with correct gradients, so this composes naturally.

### RL loop structure in define-then-run

The RL training loop alternates between two compiled routines:
1. **Inference routine** (forward-only): `state -> action_probs` -- used during episode collection
2. **Training routine** (forward + backward): `(state, action_oh, return) -> loss` -- used after episode

Both are compiled once via `Train.to_routine` and reused across all episodes. Data flows
through `Tn.set_values` (set input) and `tensor.@[i]` / `tensor.@{[|i;j|]}` (read output).

### What we do NOT port

- **Fehu environment API**: No generic Gym-like interface. Pure OCaml modules are simpler.
- **DQN / A2C / PPO**: Focus on REINFORCE variants only. Other algorithms can follow later.
- **Stage 3 CNN debugging**: Adds complexity without new OCANNL patterns.
- **Batch episode collection**: Deferred to a stretch goal. Single-episode training is clearer.

## Dependencies and Risks

**Dependencies**:
- None blocking. All required OCANNL primitives exist (`sokoban_cnn`, `softmax`, `uniform_at`,
  `mlp`, `Train.*` infrastructure).
- The Sokoban environment is pure OCaml with no external dependencies.

**Risks**:
1. **REINFORCE convergence**: RL training can be finicky. The grid world should converge
   reliably (simple problem). Sokoban may need tuning (learning rate, baseline, curriculum).
   Mitigation: start with corridor levels where optimal policy is short.
2. **Compilation overhead for step-by-step processing**: Running `Train.run` per step may be
   slow if the routine has high overhead. Mitigation: OCANNL compiles routines ahead of time,
   so per-call overhead should be minimal.
3. **Metal backend buffer limits**: The existing FIXME(#344) in bigram.ml notes buffer argument
   limits. The Sokoban CNN is larger. Mitigation: test on CPU backend first; Metal can follow.

## Acceptance Criteria Summary

- [ ] Grid world environment: reset/step API, deterministic with seed
- [ ] REINFORCE grid world: shows improving returns over 500+ episodes
- [ ] Sokoban environment: movement, box pushing, win detection, 2+ level generators
- [ ] Sokoban REINFORCE: solves corridor levels, shows learning progress on room levels
- [ ] RL slides updated with OCANNL code examples
- [ ] Exercise document with 5 progressive exercises
- [ ] All examples compile and pass (`dune runtest`)
- [ ] No regressions in existing tests
