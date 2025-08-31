# Reinforcement Learning: An Introduction to REINFORCE

Welcome to reinforcement learning! If you're familiar with supervised learning and neural network training, you're about to discover a fundamentally different approach to machine learning.

{pause .block}
This presentation is work-in-progress!

## What is Reinforcement Learning? {#rl-definition}

{.definition title="Reinforcement Learning"}
Instead of learning from labeled examples, an **agent** learns by **acting** in an **environment** and receiving **rewards**.

{pause center #rl-framework}
### The RL Framework

> **Agent**: The learner (your neural network with weights θ)
> 
> **Environment**: The world the agent interacts with  
> 
> **Actions**: What the agent can do
> 
> **States**: What the agent observes about the environment
> 
> **Rewards**: Feedback signal (positive or negative)

{pause up=rl-framework}

{.example title="Concrete Example: Sokoban Puzzle"}
- **Environment**: Grid world with boxes, walls, targets
- **Agent**: Neural network controlling a character
- **States**: Current positions of character, boxes, walls (as pixel grid or feature vector)
- **Actions**: Move up, down, left, right (4 discrete actions)
- **Rewards**: +10 for solving puzzle, -1 per step, -5 for invalid moves

{pause}

Think of it like learning to play a game:
- You (the neural network) don't know the rules initially
- You try actions and see what happens to the environment
- Good moves get rewarded, bad moves get punished
- You gradually learn a strategy by updating your weights

***

{pause center #rl-supervised-differences}
## Key Differences from Supervised Learning {#differences}

| **Aspect** | **Supervised Learning** | **Reinforcement Learning** |
|------------|-------------------------|---------------------------|
| **Data** | Fixed dataset with input-output pairs | Generated through interaction with environment |
| **Objective** | Minimize loss function (e.g., MSE, cross-entropy) | Maximize expected cumulative reward |
| **Learning** | Single training phase on static data | Continuous learning from experience |
| **Feedback** | Direct labels for each input | Delayed, sparse rewards for sequences of actions |
| **Data Generation** | Pre-collected and labeled | **You generate your own dataset** by acting |

{pause up=rl-supervised-differences}

### What "Dynamic Interaction" Means for Your Neural Network

In supervised learning, your network processes: `input → prediction`

In RL, your network (the agent) does: `state → action → new_state + reward`
- **Each forward pass** produces an action that changes the world
- **The world responds** with a new state and reward
- **You experience the consequences** of your network's outputs
- **Your training data** comes from your own actions

{pause}

**No pre-labeled data** - the agent must discover what actions are good through trial and error.

***

{pause center #policy-intro}
## The Policy: Your Agent's Strategy

{.definition title="Policy π(a|s,θ)"}
The probability of taking action **a** in state **s**, parameterized by neural network weights **θ**.

{.definition title="State vs Observation (Precise Definition)"}
- **Environment state**: Complete description of environment (all Sokoban box/wall positions)
- **Observation**: What the agent sees (may be partial, e.g., local 5×5 grid view)
- **Information state**: What your neural network system represents about the environment (e.g., current observation + recurrent hidden state from past observations)

{pause down}
Both environment and information states are **Markovian** - they capture all relevant history for decision-making.

{pause up=policy-intro #prob-policies}
### Why Probabilistic Policies?
**Key insight**: We need to **synthesize our own training dataset** through exploration!

{pause down}
> From Sutton & Barto:
> 
> > "action probabilities change smoothly as a function of the learned parameter, whereas in ε-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values"

{pause up=prob-policies}
**Benefits**:
1. **Smooth changes** = **stable learning**
2. **Natural exploration** - probability spreads across actions
3. **Dataset synthesis** - stochastic policy generates diverse experiences

{pause .example title="Policy Examples in Sokoban"}
- **Neural network output**: 4 action preferences [up, down, left, right]
- **Softmax conversion**: [0.1, 0.6, 0.2, 0.1] probabilities  
- **Action sampling**: Choose "down" with 60% probability
- **Learned parameters**: θ represents all network weights and biases

***

{pause up #episodes}
## Episodes and Returns

{.definition title="Episode"}
A complete sequence of interactions from start to terminal state.

{.definition title="Return $G_t$"}
The total reward from time step t until the end of the episode:
$$G_t = R_{t+1} + R_{t+2} + ... + R_T$$

{.definition title="Value of a State $V^\pi(s)$"}
The expected return when starting from state s and following policy π:
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

{pause down}
> ### Supervised Learning Analogy
> 
> | **Supervised Learning** | **Reinforcement Learning** |
> |--------------------------|----------------------------|
> | **Input**: x | **State**: s |
> | **Target**: y | **Return**: G_t (discovered through interaction) |
> | **Loss**: (prediction - target)² . . | **"Loss"**: -(expected return) |
> | **Gradient**: ∇ loss | **Policy Gradient**: ∇ (expected return) |

{pause center}
### The Goal

**Maximize expected return** by updating network weights θ to improve the policy.

{pause}

But how do we compute gradients when the "target" (return) depends on our own actions?

***

{pause center #reinforce-intro}
## Enter REINFORCE

{.definition title="The REINFORCE Algorithm"}
A **policy gradient** method that directly optimizes the policy parameters (network weights θ) to maximize expected return.

{pause up=reinforce-intro}
### Core Insight

We want to:
1. **Increase** the probability of actions that led to high returns
2. **Decrease** the probability of actions that led to low returns

{pause}

{.example title="Sokoban Example"}
> If pushing a box toward a target (action "up") led to solving the puzzle (G_t = +9):
> - **Increase** probability of choosing "up" in that state
> - **Strengthen** neural network weights that favor "up"

{pause}

From Sutton & Barto:

> "it causes the parameter to move most in the directions that favor actions that yield the highest return"

***

{pause up #gradient-theorem}
## The Policy Gradient Theorem

The gradient of expected return with respect to policy parameters θ:

$$\nabla_\theta J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla_\theta \pi(a|s,\theta)$$

{pause}

This looks complicated, but REINFORCE gives us a simple way to estimate it!

{pause .theorem title="REINFORCE Gradient Estimate"}
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause up=gradient-theorem}
### What This Means

From Sutton & Barto:

> "Each increment is proportional to the product of a return $G_t$ and a vector, the gradient of the probability of taking the action actually taken divided by the probability of taking that action"

***

{pause center #algorithm-reinforce}
## REINFORCE Algorithm Steps

{.block title="REINFORCE Algorithm"}
1. **Initialize** policy parameters θ (neural network weights) randomly
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t in episode:
     - Calculate return: $G_t = \sum_{k=t+1}^T R_k$
     - Update: $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)$

{pause up=algorithm-reinforce}
### Key Properties: High Variance Problem

From Sutton & Barto:

> "REINFORCE uses the complete return from time t, which includes all future rewards up until the end of the episode"

{pause}

This makes it an **unbiased** but **high variance** estimator.

{#impact-high-variance}
**Practical Impact of High Variance**:
- Learning is **slow** and **unstable**
- Need many episodes to see improvement
- Updates can be huge (good episode) or tiny (bad episode)
- **Episode-by-episode learning** amplifies noise

{pause center}
### On-Policy vs Off-Policy

{.definition title="Key Terms"}
- **On-policy**: Using data from the **current** policy π(·|·,θ) 
- **Off-policy**: Using data from a **different** policy (e.g., old θ values)

{pause up=impact-high-variance}

### Batching vs. On-Policy Learning Trade-off

**Batching reduces variance** by averaging over multiple episodes, but:
- **Risk**: Collected episodes become **off-policy** as θ changes during batch collection
- **Why off-policy is bad for REINFORCE**: The gradient estimate ∇ln π(A_t|S_t,θ) assumes actions came from current policy θ, but they came from old θ
- **Solution**: Use smaller batches or update more frequently  
- **Balance**: Variance reduction vs. policy staleness

***

{pause up #implementation-reinforce}
## Implementation in Neural Networks

If your policy network outputs action probabilities, the gradient update becomes:

```ocaml
(* Compute log probability gradient *)
let log_prob_grad =
  compute_gradient_log_prob action_taken state in
(* Scale by return *)
let policy_grad = G_t *. log_prob_grad in
(* Update parameters *)
update_parameters policy_grad learning_rate
```

{pause up=implementation-reinforce}
### In Practice

You'll typically:
1. Use **automatic differentiation** to compute ∇ ln π
2. **Collect episodes** in batches for stability
3. Apply **baseline subtraction** to reduce variance

***

{pause up #baselines}
## Reducing Variance with Baselines

REINFORCE can be **very noisy**. We can subtract a baseline b(s) from returns:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[(G_t - b(S_t)) \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause}

From Sutton & Barto:

> "The baseline can be any function, even a random variable, as long as it does not vary with a; the equation remains valid because the subtracted quantity is zero"

{pause}

> "In some states all actions have high values and we need a high baseline to differentiate the higher valued actions from the less highly valued ones"

{pause}

{.example title="Baseline Options"}
> 
> **Simple Average Baseline**:
> - $b = \frac{1}{N} \sum_{i=1}^N G_{t,i}$ (average return over past N episodes)
> - **Not learned** - just computed from episode history
> - Easy to implement, somewhat effective
> 
> **Learned State-Dependent Baseline**:
> - $b(s,w)$ - separate neural network with weights w
> - **Learned** to predict V(s) using gradient descent
> - More complex but much more effective

***

{pause center #reinforce-baseline}
## REINFORCE with Learned Baseline

{.block title="REINFORCE with Learned Baseline Algorithm"}
1. **Initialize** policy parameters θ and **baseline parameters w**
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t:
     - $G_t = \sum_{k=t+1}^T R_k$
     - $\delta = G_t - b(S_t,w)$ ← **prediction error**
     - $\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \ln \pi(A_t|S_t,\theta)$ ← **policy update**
     - $w \leftarrow w + \alpha_w \delta \nabla_w b(S_t,w)$ ← **baseline update**

{pause up=reinforce-baseline}
The baseline **neural network** is learned to predict expected returns, reducing variance without introducing bias.

**Two networks training simultaneously**:
- **Policy network**: θ parameters, outputs action probabilities
- **Baseline network**: w parameters, outputs state value estimates

***

{pause center}
### Actor-Critic Methods

From Sutton & Barto:

> "Methods that learn approximations to both policy and value functions are often called actor–critic methods"

REINFORCE with baseline is a simple actor-critic method:
- **Actor**: The policy π(a|s,θ)  
- **Critic**: The baseline b(s,w)

***

{pause center}
## Summary {#summary}

{.block title="Key Takeaways"}
> 
> ✓ **RL learns from interaction**, not labeled data
> 
> ✓ **REINFORCE optimizes policies directly** using policy gradients  
> 
> ✓ **Returns weight gradient updates** - high returns → strengthen action probabilities
> 
> ✓ **Baselines reduce variance** without introducing bias
> 
> ✓ **Actor-critic architectures** combine policy and value learning

{pause up=summary}
### Next Steps

- Implement REINFORCE on a simple environment
- Experiment with different baseline functions  
- Explore more advanced policy gradient methods (PPO, A3C)
- Consider trust region methods for more stable updates

{pause down=refs-sutton-barto}

**You now have the foundation to start learning policies through interaction!**

***

## References

{#refs-sutton-barto}
Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.