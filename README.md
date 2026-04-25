# Requirements Document: AI Manager + Worker Multi-Agent RL Environment

## Introduction

This document specifies the functional and non-functional requirements for a reinforcement learning training environment that simulates a workplace with one Manager Agent coordinating multiple Worker Agents to complete complex multi-step tasks. The system injects realistic failure modes (hallucinations, off-task behavior, incomplete work, stuck loops) to train the Manager to detect and correct errors under a limited token budget. The environment is built on the OpenEnv framework, trained with PPO, and deployed with a FastAPI backend and React frontend for visualization.
# Complete Build Instructions: AI Manager + Worker Multi-Agent RL Environment

This document is a complete specification for building the project. Every section is written so that another AI (or developer) can pick it up and implement it without ambiguity. Read the entire document before starting any section, because later decisions depend on earlier ones.

---

## What You Are Building — The Big Picture

You are building a **Reinforcement Learning training environment** using the OpenEnv framework. The environment simulates a workplace where one Manager Agent coordinates multiple Worker Agents to complete complex multi-step tasks. The twist that makes this a serious research artifact rather than a demo is that workers can **fail in realistic ways** — they can hallucinate outputs that look correct but are subtly wrong, go completely off-task, produce incomplete work, or get stuck in loops. The Manager Agent must learn, through trial and error with a reward signal, how to detect these failures and correct them under a limited token budget.

The project has four distinct parts that must all work together: the OpenEnv environment itself (the "world"), the training pipeline (the "school" where agents learn), the FastAPI backend (the "bridge" between the environment and the frontend), and the React dashboard (the "control room" where judges can watch everything happen in real time). Think of it like building a flight simulator — the environment is the physics engine, training is the pilot learning to fly, the backend is the cockpit's computer system, and the dashboard is the cockpit display.

---

## Part 1: The OpenEnv Environment

### 1.1 File and Folder Structure

Create a root folder called `manager-worker-env`. Inside it, create five subfolders: `env`, `agents`, `training`, `backend`, and `frontend`. Everything that defines how the world works lives in `env`. Everything about how agents are implemented lives in `agents`. Training scripts live in `training`. The API server lives in `backend`. The React app lives in `frontend`. Also create a `README.md` at the root — this will double as your HuggingFace blog post.

Inside the `env` folder, you need exactly four files. The first is `manager_worker_env.py`, which is the main environment class. The second is `hallucination_engine.py`, which handles injecting failures into worker outputs. The third is `task_library.py`, which stores the templates for tasks the environment can throw at agents. The fourth is `reward_calculator.py`, which contains all reward logic in isolation so it can be tuned independently of the rest.

### 1.2 The Environment Class — What It Must Do

The main environment class must inherit from `openenv.Env` and implement four methods: `__init__`, `reset`, `step`, and `render`. Think of `__init__` as the blueprint, `reset` as starting a new game, `step` as one turn of that game, and `render` as taking a screenshot of the current state.

In `__init__`, the environment must accept a configuration dictionary that specifies the maximum number of workers (use 4 as default), the maximum steps per episode (use 50 as default), the total token budget per episode (use 1000 as default), and a reference to the task library. It must define two spaces: the observation space and the action space, both described in detail below.

The `reset` method must select a random task from the task library, spawn fresh worker agents with randomly assigned skill levels between 0.3 and 1.0, reset the budget to full, and return the initial observation. It should also reset an internal step counter and an episode log (a list of events that happened this episode — you'll use this for the dashboard).

The `step` method receives an action integer from the Manager Agent. It must first validate the action, then execute it (which changes the state of workers and the task), then call the reward calculator, then check if the episode is over (either task completed, budget exhausted, or max steps reached), then return the new observation, the reward, the done flag, and an info dictionary. The info dictionary should always contain the episode log, the current hallucination catch rate, and the budget remaining — the dashboard needs these.

### 1.3 The Observation Space — What the Manager Sees

The Manager's observation is a dictionary with five keys. Understanding why each key exists matters, because this shapes what the agent can learn.

The first key is `task_embedding`, a 64-dimensional float array between -1 and 1. This represents the current task encoded as a vector. In implementation, you can use a small lookup table that maps each task type to a fixed 64-dimensional vector — you don't need a real embedding model for the hackathon, just consistent vectors per task type.

The second key is `worker_states`, a 2D float array of shape (4, 5), meaning 4 workers each described by 5 numbers. Those five numbers per worker are: is_active (0 or 1), progress as a fraction (0.0 to 1.0), hallucination_risk_score (a hidden ground truth the Manager cannot directly read — it must infer this from checking outputs), output_quality_if_checked (0.0 to 1.0, only meaningful if the Manager spent budget to check this worker), and tokens_consumed_by_worker (as a fraction of total budget).

The third key is `subtask_status`, a binary array of length 4 where 1 means that worker's subtask is complete and approved, and 0 means it's still in progress.

The fourth key is `budget_remaining`, a single float from 0.0 to 1.0.

The fifth key is `steps_remaining`, a single float from 0.0 to 1.0.

The reason the Manager cannot directly read hallucination_risk_score is critical — if it could, the problem becomes trivial. The Manager must spend budget to call `check_worker_output`, which reveals output quality and sets a flag in its observation. This budget-for-information tradeoff is the core learning challenge.

### 1.4 The Action Space — What the Manager Can Do

The Manager has exactly 7 discrete actions. Action 0 is `assign_subtask`, which takes the next unassigned subtask from the queue and gives it to an idle worker. This costs 10 tokens. Action 1 is `check_worker_output`, which reviews the currently active worker's output — this is how the Manager spends budget to get information. It costs 50 tokens but reveals the true output quality. Action 2 is `correct_worker`, which sends corrective instructions to a worker. This is the right action when the Manager has detected a problem. It costs 30 tokens and improves the worker's output quality by a random amount between 0.2 and 0.5. Action 3 is `reassign_task`, which takes a subtask from one worker and gives it to another idle worker, discarding all prior work. It costs 40 tokens. Action 4 is `fire_and_replace`, which removes a worker entirely and spawns a fresh one with a random new skill level. This is the nuclear option — it costs 100 tokens. Action 5 is `approve_output`, which accepts a worker's output and marks that subtask as done. This costs 5 tokens. Action 6 is `request_clarification`, which gives the Manager slightly more information about the task structure. It costs 20 tokens and increases the task_embedding specificity.

Workers also have their own action space, but workers are not trained with RL — they are implemented as rule-based or LLM-based agents. Each worker chooses between three actions each step: `work_on_task` (which advances progress and may produce failures), `ask_manager` (which pauses work and sends a flag to the Manager's observation), and `submit_output` (which marks the output as ready for Manager review).

### 1.5 The Task Library

Create at least 15 task templates. A task template is a dictionary containing a task type name, a description string, a list of required subtasks (between 2 and 5 subtasks per task), a difficulty rating from 1 to 5 (which influences how often workers fail), and a quality evaluation function that scores a completed set of subtask outputs on a 0.0 to 1.0 scale.

Good example tasks include "Build a landing page" (subtasks: design mockup, write HTML, write CSS, write JavaScript, deploy), "Conduct market research" (subtasks: identify sources, collect data, analyze data, write report), "Debug a codebase" (subtasks: reproduce bug, identify root cause, write fix, write tests), "Plan a product launch" (subtasks: define audience, write copy, plan distribution, set metrics), and "Write a research paper" (subtasks: literature review, hypothesis formation, methodology design, write draft, edit). Each of these should have a different difficulty level so training sees variety.

---

## Part 2: The Hallucination Engine

This is the most conceptually important file in the entire project. Build it as a standalone class that receives a subtask definition and a worker skill level, and returns an output object. The output object must always contain the actual content of the output, a hidden failure_type field (which the environment knows but the Manager Agent cannot directly observe), a detectable boolean (whether the failure is checkable), and a detection_cost integer (how many tokens it costs to check).

Implement four failure modes. The first is hallucination — the output looks correct and complete but contains subtle errors. For example, the worker claims to have written a CSS file but the styles reference class names that don't exist in the HTML. In your simulation, represent this as an output with high surface_plausibility (0.8 to 0.95) but low actual_quality (0.1 to 0.3). The Manager's `check_worker_output` action reveals the actual_quality, not the surface_plausibility. Without checking, the Manager only sees surface_plausibility, which makes hallucinations dangerous.

The second failure mode is off_task — the worker worked hard on the wrong problem. Surface plausibility is medium (0.4 to 0.6) because the output is clearly work, just misaligned. Actual quality is near zero. Detection cost should be low (around 20 tokens) because off-task outputs are easier to spot with a quick review.

The third failure mode is incomplete — the worker did partial work and submitted too early. Surface plausibility is moderate (0.5) and actual quality reflects completion percentage (0.2 to 0.5). This is the most forgiving failure because `correct_worker` can often finish the job.

The fourth failure mode is stuck — the worker is looping and never submitting. This manifests in the observation as a worker that has been active for many steps with no submit action. The Manager must detect this through patience exhaustion rather than output inspection. The right response is `reassign_task` or `fire_and_replace`.

The probability of each failure mode depends on the worker's skill level and the task's difficulty rating. A skill 1.0 worker on a difficulty 1 task should succeed roughly 95% of the time. A skill 0.3 worker on a difficulty 5 task should fail roughly 80% of the time. The exact formula is: failure_probability = (1 - worker_skill) * (task_difficulty / 5) * 0.9. Use this to gate which output path the engine takes.

---

## Part 3: The Reward Calculator

Build this as a pure function that takes the previous state, the current state, and the action taken, and returns a single float. Keep all reward logic here and nowhere else — this makes tuning straightforward.

The reward has five components that are summed together. First, task completion quality: if the episode just ended with the task complete, add 50.0 multiplied by the final output quality score. This is the primary signal. Second, time efficiency bonus: calculate how many steps were used relative to the maximum, and award up to 10.0 extra points for finishing early. Use the formula: 10.0 times (1 minus steps_used divided by max_steps). Third, budget efficiency: if more than 80% of the budget was used, apply a penalty scaled to how far over 80% you went. Fourth, hallucination detection: if the Manager just called `approve_output` on a hallucinated output, apply a -20.0 penalty. If the Manager called `correct_worker` or `reassign_task` on a hallucinated output (correctly intervening), apply a +15.0 reward. If the Manager wasted an action on a good output (false positive), apply a -3.0 penalty. Fifth, a small step cost of -0.5 per step to discourage passive behavior and reward decisiveness.

---

## Part 4: The Training Pipeline

### 4.1 Main Training Script

Create `training/train_manager.py`. This script must do five things in order. First, instantiate the environment wrapped in a vectorized environment wrapper (use DummyVecEnv from stable-baselines3 for simplicity). Second, create a PPO agent with MultiInputPolicy (this policy handles dictionary observation spaces). Third, set up logging to both TensorBoard and WandB so reward curves are captured automatically. Fourth, call model.learn() with at least 500,000 total timesteps — less than this and you won't see meaningful learning curves. Fifth, save the model and push it to HuggingFace Hub.

The key hyperparameters to set are: learning_rate at 3e-4, n_steps at 2048 (steps collected before each gradient update), batch_size at 64, n_epochs at 10, and gamma at 0.99 (how much the agent values future rewards). Do not change these until you have a baseline run — tune only after you see the first reward curves.

### 4.2 The Colab Notebook

This is mandatory for judging. Create `training/colab_notebook.ipynb`. It must be runnable top-to-bottom on a free Colab instance. Include cells that install dependencies (openenv, stable-baselines3, huggingface-hub, wandb), import the environment, run a short training session of 50,000 steps to demonstrate the learning curve, plot the reward curve using matplotlib, and show a before/after comparison of Manager behavior. The before state is episode 1 (Manager acts randomly). The after state is episode 1000 (Manager correctly identifies and handles hallucinations). This visual comparison is your strongest evidence of learning and covers your 20% judging score on reward improvement.

---

## Part 5: The FastAPI Backend

Create `backend/main.py`. The backend has two jobs: serve the environment state via REST API for the dashboard's initial load, and stream real-time updates via WebSocket so the dashboard reflects what's happening step-by-step during a live demo.

Create four REST endpoints. GET `/episode/start` initializes a new episode and returns the full initial state. GET `/episode/state` returns the current environment state at any point. POST `/episode/step` accepts an action integer and advances the environment one step, returning the new state. GET `/episode/history` returns the full episode log for the current session.

Create one WebSocket endpoint at `/ws/live`. When a client connects, it receives a JSON message every time the environment steps — including the current worker states, the Manager's last action, the reward received, the budget remaining, and any hallucination events that occurred. This is what drives the live pulsing cards in the React dashboard.

The backend must also expose a GET `/training/metrics` endpoint that reads the latest WandB or TensorBoard logs and returns reward curve data as a JSON array. The dashboard's reward chart component polls this every 5 seconds during training visualization.

---

## Part 6: The React Dashboard

### 6.1 Component Architecture

The dashboard is a single-page React application. It has one parent component called `App` and six child components. `TaskPanel` shows the current task and its subtasks with completion status. `WorkerGrid` shows four `WorkerCard` components in a 2x2 grid. `WorkerCard` is the most important visual element — it shows the worker's status, current subtask, progress bar, tokens consumed, and a red pulsing border when a hallucination event is detected. `ManagerLog` shows a scrolling feed of the Manager's decisions in plain English (e.g., "Step 23: Manager checked Worker 2's output and detected a hallucination. Intervening with correction."). `BudgetMeter` shows a horizontal progress bar that drains left-to-right in real time. `RewardChart` shows a live line chart of cumulative reward across the current episode using Recharts.

### 6.2 The WebSocket Connection

In `App.jsx`, establish a WebSocket connection to `/ws/live` on component mount. Every message received updates the application state using useState. The most important state update is the hallucination event handler — when the incoming message contains a hallucination flag on any worker, set that worker's `flagged` property to true for exactly 3 seconds before resetting it. This creates the pulsing red alert effect that will grab judges' attention during the demo.

### 6.3 Visual Design Priorities

Use a dark theme with a charcoal background. Worker cards should have a subtle green glow when healthy, yellow when idle, and a bright red pulse animation when a hallucination is detected. The Manager Log should scroll automatically to the latest entry. The budget meter should turn from green to yellow at 50% remaining and red at 20% remaining. The reward chart should show both a raw reward line and a smoothed moving average line — the smoothed line makes the upward trend clearer and is what you want judges looking at.

---

## Part 7: Deployment to HuggingFace Spaces

The final deployed artifact is a HuggingFace Space that runs a Gradio interface wrapping the React dashboard and the backend. Alternatively, you can deploy the React build as a static Space and point it at a backend hosted separately. The simpler path for the hackathon is to build a self-contained Gradio app that embeds an iframe of the React dashboard. This way you only manage one deployment.

The Space must include a README section at the top of the HuggingFace page that serves as your mini-blog. Write 400-600 words describing the problem you're solving, how the environment works, what failure modes you injected, and what the Manager Agent learned. Include a reward curve screenshot. This written content is what HuggingFace community members see and is your permanent record of the submission.

---

## Part 8: The 3-Minute Pitch Structure

You have 180 seconds. Spend the first 30 seconds on the hook: "Every multi-agent AI system in production today assumes workers do what they're told. CrewAI, LangGraph, AutoGen — they all assume agents complete tasks correctly. Ours doesn't. We built the environment that trains Manager Agents for the real world where workers fail, hallucinate, and go off-task." Spend the next 60 seconds on the live demo — show a Manager Agent in action on the dashboard, let the judges watch a hallucination get detected and corrected in real time with the red pulsing card. Spend the next 30 seconds showing the reward curves — specifically the hallucination catch rate rising from 30% to 85% across training. Spend the final 20 seconds on the production angle: "This isn't a demo. This is the training environment Meta needs before deploying multi-agent systems at the scale of millions of decisions per hour. We built the gym. The athletes are up to you."

---

## Common Mistakes to Avoid

The single biggest mistake would be making the reward function too dense — if the agent gets a reward signal for every tiny action, it learns to game individual actions rather than complete tasks. Keep the primary reward signal sparse (only on episode completion) and let the secondary signals be small nudges.

The second biggest mistake is making the hallucination injection too predictable. If low-skill workers always hallucinate and high-skill workers never do, the Manager will simply learn to fire all low-skill workers immediately. Add noise — even a skill 0.9 worker should hallucinate 5% of the time, and even a skill 0.3 worker should succeed 15% of the time. This forces the Manager to actually check outputs rather than profile workers by skill.

The third mistake is building the dashboard before the environment is stable. Spend your first two days entirely on the environment and get a random agent running in it before touching any frontend code. A broken environment with a beautiful dashboard is a broken project.

---

This document contains everything needed to build the project. Hand each part to a separate AI or developer if working in parallel — Parts 1 and 2 can be built simultaneously since they're independent, Part 3 depends on Part 1, Part 4 depends on Parts 1-3, Part 5 depends on Part 1, and Part 6 depends on Part 5. Good luck — this is a genuinely strong submission.
## Glossary

- **Manager_Agent**: The reinforcement learning agent that learns to coordinate workers and detect failures
- **Worker_Agent**: Simulated agents that execute subtasks and may exhibit failure modes
- **Hallucination**: A failure mode where output appears correct but contains subtle errors (high plausibility, low actual quality)
- **Off_Task**: A failure mode where worker produces output unrelated to the assigned subtask
- **Incomplete**: A failure mode where output is partial or missing key components
- **Stuck**: A failure mode where worker loops without progress
- **Token_Budget**: Hard constraint on total tokens available per episode (default 1000)
- **Subtask**: A discrete unit of work within a larger task
- **Task_Library**: Collection of 15+ diverse task templates for training
- **Observation**: Dictionary containing task embedding, worker states, budget, and progress
- **Action**: Discrete choice (0-6) representing manager intervention (assign, check, correct, reassign, fire, approve, clarify)
- **Reward**: Cumulative score based on quality, time, budget, hallucination detection, and step cost
- **Episode**: One complete training iteration from reset to termination
- **PPO**: Proximal Policy Optimization algorithm for training
- **MultiInputPolicy**: Stable-baselines3 policy that handles dictionary observations
- **DummyVecEnv**: Vectorized environment wrapper for parallel training
- **WebSocket**: Real-time bidirectional communication protocol for live updates
- **HuggingFace_Hub**: Model repository for saving and loading checkpoints
- **HuggingFace_Spaces**: Free deployment platform for Gradio/Streamlit applications

## Requirements

### Requirement 1: Environment Core Functionality

**User Story:** As a reinforcement learning researcher, I want a multi-agent environment that simulates realistic workplace coordination challenges, so that I can train an agent to detect and correct worker failures under resource constraints.

#### Acceptance Criteria

1. WHEN the environment is initialized with valid configuration, THE ManagerWorkerEnv SHALL accept parameters for max_workers (1-4), max_steps (10-100), token_budget (500-5000), task_difficulty (1-5), and failure_injection_rate (0.0-1.0)

2. WHEN env.reset() is called, THE ManagerWorkerEnv SHALL return a dictionary observation with keys: task_embedding (shape 64,), worker_states (shape 4x5), subtask_status (shape 4,), budget_remaining (float 0-1), steps_remaining (float 0-1)

3. WHEN env.step(action) is called with valid action in range [0, 6], THE ManagerWorkerEnv SHALL execute the action, update worker states, deduct tokens, and return (observation, reward, done, info)

4. WHEN an episode terminates (max_steps exceeded or task complete), THE ManagerWorkerEnv SHALL compute final_quality from completed subtasks and return done=True

5. WHEN the token_budget is exceeded, THE ManagerWorkerEnv SHALL reject further actions and terminate the episode with appropriate penalty

6. WHEN env.render() is called, THE ManagerWorkerEnv SHALL return a human-readable string representation of current state or rgb_array image

### Requirement 2: Manager Actions

**User Story:** As a manager agent, I want to take 7 distinct actions to coordinate workers, so that I can learn optimal strategies for task completion and failure detection.

#### Acceptance Criteria

1. WHEN action=0 (assign_subtask) is executed, THE ManagerWorkerEnv SHALL assign a subtask to a worker and deduct 10 tokens from budget

2. WHEN action=1 (check_worker_output) is executed, THE ManagerWorkerEnv SHALL inspect worker output quality and deduct 50 tokens from budget

3. WHEN action=2 (correct_worker) is executed, THE ManagerWorkerEnv SHALL provide correction feedback to worker and deduct 30 tokens from budget

4. WHEN action=3 (reassign_task) is executed, THE ManagerWorkerEnv SHALL reassign current subtask to different worker and deduct 40 tokens from budget

5. WHEN action=4 (fire_and_replace) is executed, THE ManagerWorkerEnv SHALL replace worker with new instance and deduct 100 tokens from budget

6. WHEN action=5 (approve_output) is executed, THE ManagerWorkerEnv SHALL mark subtask complete and deduct 5 tokens from budget

7. WHEN action=6 (request_clarification) is executed, THE ManagerWorkerEnv SHALL request clarification from worker and deduct 20 tokens from budget

### Requirement 3: Worker Failure Modes

**User Story:** As an environment designer, I want to inject realistic failure modes into worker behavior, so that the manager learns to detect and correct common workplace errors.

#### Acceptance Criteria

1. WHEN a worker executes a subtask with failure_injection_rate > 0, THE HallucinationEngine SHALL randomly inject one of four failure modes: hallucination, off_task, incomplete, or stuck

2. WHEN hallucination failure is injected, THE worker output SHALL have surface_plausibility in [0.8, 0.95] but actual_quality in [0.1, 0.3], and detection SHALL cost 50 tokens

3. WHEN off_task failure is injected, THE worker output SHALL have surface_plausibility in [0.4, 0.6] and actual_quality in [0.0, 0.1], and detection SHALL cost 20 tokens

4. WHEN incomplete failure is injected, THE worker output SHALL have surface_plausibility in [0.5, 0.6], actual_quality in [0.2, 0.5], and progress in [0.3, 0.7], and detection SHALL cost 50 tokens

5. WHEN stuck failure is injected, THE worker SHALL loop without progress and patience_counter SHALL increment each step until threshold is reached, and detection SHALL cost 20 tokens

6. WHEN a failure mode is detected via check_worker_output, THE observation SHALL reflect the true quality (not surface plausibility) in worker_states

7. FOR ALL failure modes, the failure_mode field in WorkerState SHALL be set to the injected mode name or None if no failure

### Requirement 4: Token Budget Management

**User Story:** As a system designer, I want to enforce a hard token budget constraint, so that the manager learns to make efficient use of limited resources.

#### Acceptance Criteria

1. WHEN an episode begins, THE ManagerWorkerEnv SHALL initialize budget_remaining to token_budget (default 1000)

2. WHEN any action is executed, THE ManagerWorkerEnv SHALL deduct the action's token cost from budget_remaining

3. WHEN budget_remaining would become negative, THE ManagerWorkerEnv SHALL reject the action and return action_valid=False in info dict

4. WHEN budget_remaining reaches zero, THE ManagerWorkerEnv SHALL terminate the episode and compute final_quality from current state

5. WHEN an episode ends, THE observation SHALL include budget_remaining normalized to [0, 1] range

6. THE token costs for each action SHALL be: assign=10, check=50, correct=30, reassign=40, fire=100, approve=5, clarify=20

### Requirement 5: Episode Configuration and Termination

**User Story:** As a trainer, I want configurable episode parameters and clear termination conditions, so that I can adjust training difficulty and ensure reproducible episodes.

#### Acceptance Criteria

1. WHEN the environment is initialized, THE ManagerWorkerEnv SHALL accept max_steps parameter (default 50, range 10-100)

2. WHEN max_steps is reached, THE ManagerWorkerEnv SHALL set done=True and terminate the episode

3. WHEN all subtasks are completed with quality >= quality_threshold, THE ManagerWorkerEnv SHALL set done=True and compute final_quality as average of all subtask qualities

4. WHEN budget_remaining <= 0, THE ManagerWorkerEnv SHALL set done=True and terminate the episode

5. WHEN a worker enters stuck state (patience_counter > threshold), THE ManagerWorkerEnv SHALL mark worker inactive and reduce final_quality accordingly

6. WHEN an episode terminates, THE info dict SHALL include: task_quality, tokens_used, steps_used, hallucinations_detected, false_positives, incomplete_subtasks

### Requirement 6: Task Library

**User Story:** As a researcher, I want a diverse task library with at least 15 templates, so that the manager learns generalizable strategies across different task types.

#### Acceptance Criteria

1. THE TaskLibrary SHALL contain at least 15 distinct task templates covering domains: web_development, research, software_engineering, product_management, academic_writing, and others

2. WHEN a task is sampled from TaskLibrary, THE task SHALL include: task_id, task_type, description, list of subtasks (2-5 per task), difficulty (1-5), quality_eval_fn, estimated_tokens

3. WHEN a task is sampled, THE task difficulty SHALL be randomly selected or match requested difficulty parameter

4. FOR EACH task, the quality_eval_fn SHALL be a callable that takes output string and returns quality score in [0, 1]

5. FOR EACH subtask, the subtask SHALL include: subtask_id, description, expected_output_format, quality_threshold (0.6-0.9)

6. WHEN a task is completed, THE final_quality SHALL be computed as weighted average of all subtask qualities

### Requirement 7: Observation Space

**User Story:** As a policy network designer, I want a well-structured observation space with normalized values, so that the policy can learn effectively from diverse state representations.

#### Acceptance Criteria

1. WHEN an observation is generated, THE observation dict SHALL contain exactly 5 keys: task_embedding, worker_states, subtask_status, budget_remaining, steps_remaining

2. WHEN task_embedding is generated, THE embedding SHALL be shape (64,) with values normalized to [-1, 1] range

3. WHEN worker_states is generated, THE array SHALL be shape (4, 5) with columns: [is_active, progress, hallucination_risk, output_quality, tokens_consumed_ratio], all values in [0, 1]

4. WHEN subtask_status is generated, THE array SHALL be shape (4,) with binary values: 1 if subtask complete, 0 otherwise

5. WHEN budget_remaining is generated, THE value SHALL be normalized to [0, 1] as (tokens_remaining / token_budget)

6. WHEN steps_remaining is generated, THE value SHALL be normalized to [0, 1] as (steps_left / max_steps)

7. FOR ALL observations, the generation SHALL be deterministic given the same environment state (no randomness in observation generation itself)

### Requirement 8: Reward Function

**User Story:** As a trainer, I want a multi-component reward function that incentivizes quality, efficiency, and hallucination detection, so that the manager learns balanced strategies.

#### Acceptance Criteria

1. WHEN an episode ends, THE RewardCalculator SHALL compute reward as sum of 5 components: quality, time, budget, hallucination_detection, step_cost

2. WHEN quality component is computed, THE reward SHALL be 50.0 × final_quality (range 0-50)

3. WHEN time component is computed, THE reward SHALL be 10.0 × (1 - steps_used/max_steps) (range 0-10)

4. WHEN budget component is computed, THE reward SHALL be -5.0 if tokens_used/token_budget > 0.8, else 0.0

5. WHEN hallucination_detection component is computed, THE reward SHALL be: +15.0 per correct intervention, -20.0 per hallucination approval, -3.0 per false positive

6. WHEN step_cost component is computed, THE reward SHALL be -0.5 × steps_used

7. WHEN final_quality increases (all else equal), THE total_reward SHALL increase

8. WHEN steps_used decreases (all else equal), THE total_reward SHALL increase

9. WHEN tokens_used decreases (all else equal), THE total_reward SHALL increase

### Requirement 9: Training Pipeline

**User Story:** As a researcher, I want to train the manager agent using PPO with support for dictionary observations, so that I can leverage modern deep RL algorithms.

#### Acceptance Criteria

1. WHEN training begins, THE training pipeline SHALL use stable-baselines3 PPO algorithm with MultiInputPolicy

2. WHEN the environment is wrapped for training, THE pipeline SHALL use DummyVecEnv to vectorize multiple environment instances

3. WHEN training runs, THE pipeline SHALL support at least 500,000 timesteps without memory issues

4. WHEN training progresses, THE pipeline SHALL log metrics to TensorBoard at configurable intervals (default every 1000 steps)

5. WHEN training progresses, THE pipeline SHALL log metrics to Weights & Biases (WandB) for experiment tracking

6. WHEN training completes, THE trained model SHALL be saved to HuggingFace Hub with model card and training metadata

7. WHEN training is interrupted, THE pipeline SHALL support resuming from latest checkpoint

8. WHEN training runs, THE learning curves SHALL show clear improvement in cumulative reward over time

### Requirement 10: Backend API - Episode Management

**User Story:** As a frontend developer, I want REST endpoints to manage episodes, so that I can control training and retrieve episode data.

#### Acceptance Criteria

1. WHEN POST /episode/start is called, THE backend SHALL initialize a new episode and return episode_id, initial_observation, and metadata

2. WHEN POST /episode/{episode_id}/step is called with action, THE backend SHALL execute the action and return observation, reward, done, info

3. WHEN GET /episode/{episode_id}/state is called, THE backend SHALL return current episode state including all worker states and task progress

4. WHEN GET /episode/{episode_id}/history is called, THE backend SHALL return list of all steps taken in episode with actions, rewards, and observations

5. WHEN POST /episode/{episode_id}/reset is called, THE backend SHALL reset the episode to initial state

6. WHEN GET /episode/list is called, THE backend SHALL return list of all active episodes with metadata

7. WHEN POST /episode/{episode_id}/end is called, THE backend SHALL terminate the episode and return final statistics

### Requirement 11: Backend API - Metrics and Training

**User Story:** As a dashboard developer, I want endpoints to retrieve training metrics, so that I can display learning progress and performance statistics.

#### Acceptance Criteria

1. WHEN GET /training/metrics is called, THE backend SHALL return current training metrics: total_timesteps, mean_reward, episode_count, learning_rate

2. WHEN GET /training/metrics/history is called, THE backend SHALL return historical metrics as time series data for plotting

3. WHEN GET /training/model/info is called, THE backend SHALL return model metadata: architecture, parameters, training_config, checkpoint_path

4. WHEN GET /training/model/checkpoint is called, THE backend SHALL return path to latest model checkpoint

5. WHEN POST /training/model/save is called, THE backend SHALL save current model to HuggingFace Hub and return save_status

### Requirement 12: Backend API - WebSocket Real-Time Updates

**User Story:** As a frontend developer, I want real-time WebSocket updates during episodes, so that the dashboard can display live progress without polling.

#### Acceptance Criteria

1. WHEN a WebSocket client connects to /ws/live, THE backend SHALL accept the connection and maintain it for the duration of the episode

2. WHEN an episode step completes, THE backend SHALL broadcast update message containing: observation, reward, done, step_number, timestamp

3. WHEN a worker state changes, THE backend SHALL broadcast worker_update message containing: worker_id, new_state, failure_mode, quality_score

4. WHEN budget changes, THE backend SHALL broadcast budget_update message containing: budget_remaining, tokens_used, budget_ratio

5. WHEN an episode ends, THE backend SHALL broadcast episode_end message containing: final_reward, final_quality, episode_statistics

6. WHEN a WebSocket connection is closed, THE backend SHALL gracefully handle disconnection and clean up resources

7. WHEN multiple clients are connected, THE backend SHALL broadcast updates to all connected clients simultaneously

### Requirement 13: Frontend Dashboard - Core Visualization

**User Story:** As a user, I want a real-time dashboard that visualizes the multi-agent environment, so that I can monitor training progress and understand manager behavior.

#### Acceptance Criteria

1. WHEN the dashboard loads, THE React frontend SHALL display 5 main panels: TaskPanel, WorkerGrid, ManagerLog, BudgetMeter, RewardChart

2. WHEN an episode is running, THE TaskPanel SHALL display: current task description, list of subtasks with completion status, estimated tokens, difficulty level

3. WHEN an episode is running, THE WorkerGrid SHALL display 4 worker cards showing: worker_id, is_active status, current_progress (0-100%), hallucination_risk (0-100%), output_quality (0-100%), tokens_consumed

4. WHEN an episode is running, THE ManagerLog SHALL display chronological list of manager actions taken: action_name, worker_id, tokens_cost, timestamp

5. WHEN an episode is running, THE BudgetMeter SHALL display: budget_remaining as progress bar, tokens_used as counter, budget_ratio as percentage

6. WHEN an episode is running, THE RewardChart SHALL display cumulative reward over time as line chart updating in real-time

7. WHEN the dashboard receives WebSocket updates, THE UI SHALL update all panels within 100ms without lag or flickering

### Requirement 14: Frontend Dashboard - Interactivity

**User Story:** As a researcher, I want to interact with the dashboard to control episodes and view detailed information, so that I can experiment with different scenarios.

#### Acceptance Criteria

1. WHEN the user clicks "Start Episode" button, THE dashboard SHALL call POST /episode/start and initialize new episode

2. WHEN the user clicks "Pause Episode" button, THE dashboard SHALL pause the current episode and allow inspection

3. WHEN the user clicks "Resume Episode" button, THE dashboard SHALL resume the paused episode

4. WHEN the user clicks "End Episode" button, THE dashboard SHALL terminate the episode and display final statistics

5. WHEN the user hovers over a worker card, THE dashboard SHALL display detailed worker information: current_subtask, failure_mode, output_buffer (first 100 chars), patience_counter

6. WHEN the user clicks on a step in ManagerLog, THE dashboard SHALL highlight the corresponding worker and action details

7. WHEN the user clicks "Export Data" button, THE dashboard SHALL download episode data as JSON file

### Requirement 15: Frontend Dashboard - Real-Time Updates

**User Story:** As a user, I want the dashboard to update in real-time as the episode progresses, so that I can see live changes without manual refresh.

#### Acceptance Criteria

1. WHEN a WebSocket message is received, THE dashboard SHALL update the corresponding panel within 50ms

2. WHEN worker_states change, THE WorkerGrid SHALL re-render with new values and smooth transitions

3. WHEN budget changes, THE BudgetMeter SHALL animate the progress bar and update token counter

4. WHEN reward changes, THE RewardChart SHALL add new data point and scroll to show latest values

5. WHEN an action is logged, THE ManagerLog SHALL prepend new entry to the top of the list

6. WHEN an episode ends, THE dashboard SHALL display final statistics panel with summary metrics

7. WHEN the WebSocket connection is lost, THE dashboard SHALL display connection error and attempt to reconnect

### Requirement 16: Vectorization and Parallelization

**User Story:** As a trainer, I want the environment to support vectorization for parallel training, so that I can efficiently use multiple CPU cores.

#### Acceptance Criteria

1. WHEN DummyVecEnv wraps multiple ManagerWorkerEnv instances, THE vectorized environment SHALL execute all instances in parallel

2. WHEN env.reset() is called on vectorized environment, THE method SHALL return stacked observations from all instances

3. WHEN env.step(actions) is called on vectorized environment with action array, THE method SHALL execute actions in parallel and return stacked results

4. WHEN an episode terminates in one instance, THE vectorized environment SHALL automatically reset that instance and continue others

5. WHEN vectorized environment runs, THE memory usage SHALL scale linearly with number of instances (no exponential growth)

### Requirement 17: Determinism and Reproducibility

**User Story:** As a researcher, I want reproducible training runs, so that I can verify results and compare different approaches.

#### Acceptance Criteria

1. WHEN a random seed is set before environment initialization, THE environment SHALL produce identical observations and failure modes for identical action sequences

2. WHEN observation generation is called with same state, THE observation values SHALL be identical (no randomness in observation generation)

3. WHEN reward calculation is called with same inputs, THE reward value SHALL be identical

4. WHEN task sampling is called with same seed, THE task sequence SHALL be identical

5. WHEN failure injection is called with same seed, THE failure mode sequence SHALL be identical

### Requirement 18: Deployment to HuggingFace Spaces

**User Story:** As a researcher, I want to deploy the trained model and dashboard to HuggingFace Spaces, so that others can interact with the system without local setup.

#### Acceptance Criteria

1. WHEN the application is deployed to HuggingFace Spaces, THE deployment SHALL work on free tier (no GPU required for inference)

2. WHEN the dashboard loads on HuggingFace Spaces, THE frontend SHALL connect to backend API running on same Space

3. WHEN the user interacts with dashboard on HuggingFace Spaces, THE episode execution SHALL complete within 30 seconds per step

4. WHEN the model is loaded from HuggingFace Hub, THE loading time SHALL be < 5 seconds

5. WHEN the Space is accessed by multiple users, THE backend SHALL handle concurrent requests without errors

6. WHEN the Space is idle, THE backend SHALL not consume resources (auto-sleep on free tier)

### Requirement 19: Colab Notebook Compatibility

**User Story:** As a user, I want to run the training pipeline in a free Colab notebook, so that I can experiment without local GPU.

#### Acceptance Criteria

1. WHEN the training script runs in Colab, THE script SHALL complete 500,000 timesteps within 12-hour session limit

2. WHEN the training script runs in Colab, THE memory usage SHALL not exceed 12GB (Colab free tier limit)

3. WHEN the training script runs in Colab, THE script SHALL use GPU acceleration if available (T4 GPU)

4. WHEN the training script runs in Colab, THE script SHALL save checkpoints to Google Drive for persistence

5. WHEN the training script runs in Colab, THE script SHALL log metrics to WandB for visualization

6. WHEN the training script completes, THE script SHALL upload final model to HuggingFace Hub

### Requirement 20: Model Checkpointing and Persistence

**User Story:** As a trainer, I want to save and load model checkpoints, so that I can resume training and deploy trained models.

#### Acceptance Criteria

1. WHEN training completes, THE pipeline SHALL save model checkpoint to HuggingFace Hub with model card

2. WHEN a checkpoint is saved, THE checkpoint SHALL include: model weights, training config, hyperparameters, training_timesteps

3. WHEN a checkpoint is loaded, THE model SHALL be identical to saved state and ready for inference

4. WHEN a checkpoint is loaded, THE training can resume from saved timestep count

5. WHEN multiple checkpoints are saved, THE pipeline SHALL maintain version history on HuggingFace Hub

### Requirement 21: Logging and Monitoring

**User Story:** As a researcher, I want comprehensive logging and monitoring, so that I can debug issues and analyze training behavior.

#### Acceptance Criteria

1. WHEN training runs, THE pipeline SHALL log to TensorBoard: cumulative_reward, episode_length, task_quality, tokens_used, hallucination_detection_rate

2. WHEN training runs, THE pipeline SHALL log to WandB: same metrics plus learning_rate, policy_loss, value_loss, entropy

3. WHEN an error occurs, THE system SHALL log error message with full stack trace to log file

4. WHEN an episode completes, THE system SHALL log episode summary: episode_id, total_reward, final_quality, steps_used, tokens_used

5. WHEN the backend API receives request, THE system SHALL log: endpoint, method, status_code, response_time

### Requirement 22: Error Handling and Recovery

**User Story:** As a system operator, I want robust error handling, so that the system recovers gracefully from failures.

#### Acceptance Criteria

1. WHEN an invalid action is provided, THE environment SHALL reject the action and return action_valid=False without state change

2. WHEN a worker enters stuck state, THE environment SHALL mark worker inactive and allow manager to fire_and_replace

3. WHEN budget is exhausted, THE environment SHALL terminate episode gracefully with final_quality computed from current state

4. WHEN a WebSocket connection is lost, THE backend SHALL close connection gracefully and log disconnection

5. WHEN the backend API receives malformed request, THE API SHALL return 400 Bad Request with error message

6. WHEN the model checkpoint is corrupted, THE system SHALL log error and attempt to load previous checkpoint

### Requirement 23: Performance and Scalability

**User Story:** As a researcher, I want the system to perform efficiently at scale, so that I can train on large datasets without bottlenecks.

#### Acceptance Criteria

1. WHEN the environment executes one step, THE step execution time SHALL be < 100ms

2. WHEN observation generation occurs, THE generation time SHALL be < 50ms

3. WHEN reward calculation occurs, THE calculation time SHALL be < 10ms

4. WHEN vectorized environment runs with 8 instances, THE throughput SHALL be >= 80 steps/second

5. WHEN the backend API handles 10 concurrent requests, THE response time SHALL be < 500ms per request

6. WHEN the dashboard renders with live updates, THE frame rate SHALL be >= 30 FPS

### Requirement 24: Security and Input Validation

**User Story:** As a system administrator, I want secure input validation, so that the system is protected from malicious inputs.

#### Acceptance Criteria

1. WHEN the environment receives configuration, THE environment SHALL validate all parameters are within specified ranges

2. WHEN the backend API receives action, THE API SHALL validate action is in range [0, 6]

3. WHEN the backend API receives episode_id, THE API SHALL validate episode_id exists and belongs to current session

4. WHEN the WebSocket receives message, THE backend SHALL validate message format before processing

5. WHEN the frontend sends request, THE frontend SHALL include authentication token in request headers

### Requirement 25: Documentation and Examples

**User Story:** As a developer, I want comprehensive documentation and examples, so that I can understand and extend the system.

#### Acceptance Criteria

1. THE system SHALL include README.md with overview, installation, and quick-start guide

2. THE system SHALL include example_training.py showing how to train the manager agent

3. THE system SHALL include example_inference.py showing how to run trained model

4. THE system SHALL include API documentation with endpoint descriptions and example requests

5. THE system SHALL include architecture documentation explaining design decisions

6. THE system SHALL include troubleshooting guide for common issues


# Setting up on your local device

# Project Setup Guide

## Environment Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install openenv-core gymnasium stable-baselines3 torch numpy pandas matplotlib wandb tensorboard fastapi uvicorn websockets pydantic requests huggingface-hub python-dotenv
```

### 3. Verify Installation
```bash
python3 test_environment.py
```

## Project Structure

```
manager-worker-env/
├── env/
│   ├── __init__.py
│   ├── manager_worker_env.py      # Core environment class
│   ├── task_library.py             # 15+ task templates
│   ├── hallucination_engine.py     # Failure mode injection
│   └── reward_calculator.py        # 5-component reward
├── agents/                         # (To be implemented)
├── training/                       # (To be implemented)
├── backend/                        # (To be implemented)
├── frontend/                       # (To be implemented)
├── test_environment.py             # Quick test script
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
└── SETUP.md                        # This file
```

## Environment Features

### ManagerWorkerEnv
- **Inherits from**: `gym.Env` (compatible with stable-baselines3)
- **Observation Space**: Dict with 5 keys
  - `task_embedding`: 64-dim vector [-1, 1]
  - `worker_states`: 4x5 array (4 workers, 5 features each)
  - `subtask_status`: Binary array of length 4
  - `budget_remaining`: Float [0, 1]
  - `steps_remaining`: Float [0, 1]
- **Action Space**: Discrete(7)
  - 0: assign_subtask (10 tokens)
  - 1: check_worker_output (50 tokens)
  - 2: correct_worker (30 tokens)
  - 3: reassign_task (40 tokens)
  - 4: fire_and_replace (100 tokens)
  - 5: approve_output (5 tokens)
  - 6: request_clarification (20 tokens)

### Task Library
- 15 diverse task templates across 5 domains:
  - Web Development (3 tasks)
  - Research (4 tasks)
  - Software Engineering (4 tasks)
  - Product Management (2 tasks)
  - Academic Writing (2 tasks)
- Each task has 2-5 subtasks and difficulty 1-5

### Failure Modes
- **Hallucination**: High plausibility (0.8-0.95), low quality (0.1-0.3)
- **Off-task**: Medium plausibility (0.4-0.6), near-zero quality (0.0-0.1)
- **Incomplete**: Moderate plausibility (0.5-0.6), partial quality (0.2-0.5)
- **Stuck**: Looping behavior, detected via patience counter

### Reward Function (5 components)
1. **Quality Reward**: 50.0 × final_quality
2. **Time Efficiency**: 10.0 × (1 - steps_used/max_steps)
3. **Budget Efficiency**: -5.0 if tokens_used/budget > 80%
4. **Hallucination Detection**: +15.0 per intervention, -20.0 per approval, -3.0 per false positive
5. **Step Cost**: -0.5 per step

## Quick Start

### Test the Environment
```bash
source venv/bin/activate
python3 test_environment.py
```

### Use in Your Code
```python
from env import ManagerWorkerEnv

config = {
    'max_workers': 4,
    'max_steps': 50,
    'token_budget': 1000,
    'task_difficulty': 3,
    'failure_injection_rate': 0.6,
}

env = ManagerWorkerEnv(config)
obs = env.reset()

for step in range(50):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Next Steps

1. **Part 1.3-1.7**: Implement remaining environment methods
2. **Part 2**: Create training pipeline with PPO
3. **Part 3**: Build FastAPI backend
4. **Part 4**: Create React dashboard
5. **Part 5**: Deploy to HuggingFace Spaces

## Notes

- Training will be on HuggingFace (Colab for development, HF Spaces for deployment)
- All code follows the spec documents in `.kiro/specs/manager-worker-rl-env/`
- Property-based testing with hypothesis for correctness validation
- Use `source venv/bin/activate` to activate the virtual environment before running any commands
