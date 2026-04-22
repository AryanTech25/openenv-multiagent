# Implementation Plan: AI Manager + Worker Multi-Agent RL Environment

## Overview

This is a reinforcement learning training environment built on OpenEnv that simulates a workplace with one Manager Agent coordinating multiple Worker Agents to complete complex multi-step tasks under a limited token budget. The implementation is organized into 5 parts with clear dependencies.

## Part 1: Environment Core (env/ folder) - Foundation Tasks

- [x] 1.1 Create project structure and base files
  - Create root folder manager-worker-env with subfolders: env, agents, training, backend, frontend
  - Create env/__init__.py, env/manager_worker_env.py, env/hallucination_engine.py, env/task_library.py, env/reward_calculator.py
  - _Requirements: 1.1, 1.2_

- [x] 1.2 Implement ManagerWorkerEnv class - Core environment
  - Implement __init__ accepting config dict with max_workers, max_steps, token_budget, task_difficulty, failure_injection_rate
  - Define observation_space and action_space using gym.spaces
  - Initialize internal state: workers, task, budget, step_counter, episode_log
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ]* 1.2.1 Write property test for observation space validity
  - **Property 3: Observation Space Validity**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**

- [ ] 1.3 Implement reset() method
  - Select random task from task library
  - Spawn 4 fresh worker agents with random skill levels [0.3, 1.0]
  - Reset budget to full, step counter to 0, episode log to empty
  - Return initial observation dictionary with all 5 keys
  - _Requirements: 1.2, 1.3_

- [ ]* 1.3.1 Write property test for observation determinism
  - **Property 4: Observation Determinism**
  - **Validates: Requirement 7.7, 17.2**

- [ ] 1.4 Implement step(action) method
  - Validate action in range [0, 6]
  - Execute action and update worker states
  - Deduct tokens from budget
  - Call reward calculator
  - Check episode termination conditions
  - Return (observation, reward, done, info)
  - _Requirements: 1.3, 1.4, 2.1-2.7, 4.1-4.6_

- [ ]* 1.4.1 Write property test for budget constraint enforcement
  - **Property 1: Budget Constraint Enforcement**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [ ]* 1.4.2 Write property test for action token cost consistency
  - **Property 10: Action Token Cost Consistency**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 4.6**

- [ ] 1.5 Implement render() method
  - Return human-readable string representation of current state
  - Support mode='human' for console output
  - _Requirements: 1.4_

- [ ] 1.6 Implement observation space generation
  - Generate task_embedding (64-dim, normalized to [-1, 1])
  - Generate worker_states (4x5 array with [is_active, progress, hallucination_risk, output_quality, tokens_consumed_ratio])
  - Generate subtask_status (binary array of length 4)
  - Generate budget_remaining and steps_remaining (normalized to [0, 1])
  - _Requirements: 7.1-7.7_

- [ ]* 1.6.1 Write property test for observation update consistency
  - **Property 12: Observation Update Consistency**
  - **Validates: Requirements 1.3, 1.4**

- [ ] 1.7 Implement action space and action validation
  - Define 7 discrete actions: assign_subtask(10), check_worker_output(50), correct_worker(30), reassign_task(40), fire_and_replace(100), approve_output(5), request_clarification(20)
  - Implement action validation and token deduction
  - _Requirements: 2.1-2.7, 4.6_

- [ ] 1.8 Implement task_library.py with 15+ task templates
  - Create Task and Subtask dataclasses
  - Implement 15+ diverse task templates: build_landing_page, market_research, debug_codebase, plan_product_launch, write_research_paper, etc.
  - Each task includes: task_id, task_type, description, subtasks (2-5), difficulty (1-5), quality_eval_fn, estimated_tokens
  - _Requirements: 6.1-6.6_

- [ ]* 1.8.1 Write property test for task completion quality calculation
  - **Property 13: Task Completion Quality Calculation**
  - **Validates: Requirement 6.6**

- [ ] 1.9 Implement hallucination_engine.py with 4 failure modes
  - Implement FailureMode dataclass
  - Implement 4 failure modes: hallucination (plausibility 0.8-0.95, quality 0.1-0.3), off_task (0.4-0.6, 0.0-0.1), incomplete (0.5-0.6, 0.2-0.5), stuck (looping behavior)
  - Implement failure injection logic based on worker skill and task difficulty
  - _Requirements: 3.1-3.5_

- [ ]* 1.9.1 Write property test for failure mode distribution
  - **Property 8: Failure Mode Distribution**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

- [ ]* 1.9.2 Write property test for failure mode quality characteristics
  - **Property 9: Failure Mode Quality Characteristics**
  - **Validates: Requirements 3.2, 3.3, 3.4**

- [ ] 1.10 Implement reward_calculator.py with 5-component reward
  - Implement calculate_reward() function with 5 components:
    - Quality reward: 50.0 × final_quality
    - Time efficiency: 10.0 × (1 - steps_used/max_steps)
    - Budget efficiency: -5.0 if tokens_used/token_budget > 0.8
    - Hallucination detection: +15.0 per correct intervention, -20.0 per approval, -3.0 per false positive
    - Step cost: -0.5 × steps_used
  - _Requirements: 8.1-8.9_

- [ ]* 1.10.1 Write property test for reward monotonicity - quality
  - **Property 5: Reward Monotonicity - Quality**
  - **Validates: Requirements 8.7, 8.8, 8.9**

- [ ]* 1.10.2 Write property test for reward monotonicity - time efficiency
  - **Property 6: Reward Monotonicity - Time Efficiency**
  - **Validates: Requirements 8.7, 8.8, 8.9**

- [ ]* 1.10.3 Write property test for reward monotonicity - budget efficiency
  - **Property 7: Reward Monotonicity - Budget Efficiency**
  - **Validates: Requirements 8.7, 8.8, 8.9**

- [ ] 1.11 Implement worker state consistency
  - Implement WorkerState dataclass with all required fields
  - Ensure all worker state values remain in valid ranges
  - _Requirements: 3.7, 5.6_

- [ ]* 1.11.1 Write property test for worker state consistency
  - **Property 2: Worker State Consistency**
  - **Validates: Requirements 3.7, 5.6**

- [ ] 1.12 Implement episode termination conditions
  - Terminate when max_steps reached
  - Terminate when all subtasks completed with quality >= threshold
  - Terminate when budget_remaining <= 0
  - Terminate when worker enters stuck state
  - Compute final_quality from completed subtasks
  - _Requirements: 5.2-5.6_

- [ ]* 1.12.1 Write property test for episode termination conditions
  - **Property 11: Episode Termination Conditions**
  - **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6**

- [ ] 1.13 Test environment with random agent (sanity check)
  - Create simple random agent that selects actions uniformly
  - Run 10 episodes and verify environment doesn't crash
  - Verify observations have correct shapes and ranges
  - Verify rewards are computed correctly
  - Log episode statistics (total_reward, final_quality, steps_used, tokens_used)
  - _Requirements: 1.1-1.4_

- [ ] 1.14 Checkpoint - Ensure all environment tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Part 2: Training Pipeline (training/ folder) - Depends on Part 1

- [ ] 2.1 Create train_manager.py with PPO + DummyVecEnv setup
  - Instantiate environment wrapped in DummyVecEnv
  - Create PPO agent with MultiInputPolicy
  - Set hyperparameters: learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99
  - _Requirements: 9.1-9.7_

- [ ] 2.2 Set up TensorBoard logging
  - Configure TensorBoard callback to log: cumulative_reward, episode_length, task_quality, tokens_used, hallucination_detection_rate
  - Log metrics every 1000 steps
  - _Requirements: 21.1_

- [ ] 2.3 Set up WandB logging
  - Configure WandB callback to log: same metrics plus learning_rate, policy_loss, value_loss, entropy
  - Create WandB project and run
  - _Requirements: 21.2_

- [ ] 2.4 Implement model saving to HuggingFace Hub
  - Save model checkpoint with model card and training metadata
  - Include hyperparameters and training_timesteps in checkpoint
  - Push to HuggingFace Hub
  - _Requirements: 20.1-20.5_

- [ ] 2.5 Create colab_notebook.ipynb for free Colab training
  - Install dependencies: openenv, stable-baselines3, huggingface-hub, wandb
  - Import environment and training utilities
  - Run 50,000 step training session
  - Plot reward curve using matplotlib
  - Show before/after Manager behavior comparison
  - _Requirements: 19.1-19.6_

- [ ] 2.6 Run baseline training (50,000 steps) and verify learning curves
  - Execute training script
  - Verify reward curves show improvement over time
  - Verify hallucination detection rate increases
  - Save trained model
  - _Requirements: 9.8_

- [ ] 2.7 Checkpoint - Ensure training completes successfully
  - Ensure all tests pass, ask the user if questions arise.

## Part 3: Backend API (backend/ folder) - Depends on Part 1

- [ ] 3.1 Create FastAPI server structure
  - Create backend/main.py with FastAPI app
  - Set up CORS middleware
  - Initialize environment instance
  - _Requirements: 10.1-10.7, 12.1-12.7_

- [ ] 3.2 Implement REST endpoints (/episode/*)
  - POST /episode/start: Initialize new episode, return episode_id and initial_observation
  - POST /episode/{episode_id}/step: Execute action, return observation, reward, done, info
  - GET /episode/{episode_id}/state: Return current episode state
  - GET /episode/{episode_id}/history: Return full episode log
  - POST /episode/{episode_id}/reset: Reset episode to initial state
  - GET /episode/list: Return list of active episodes
  - POST /episode/{episode_id}/end: Terminate episode and return final statistics
  - _Requirements: 10.1-10.7_

- [ ] 3.3 Implement WebSocket endpoint (/ws/live) for real-time updates
  - Accept WebSocket connections
  - Broadcast update messages on each step: observation, reward, done, step_number, timestamp
  - Broadcast worker_update messages: worker_id, new_state, failure_mode, quality_score
  - Broadcast budget_update messages: budget_remaining, tokens_used, budget_ratio
  - Broadcast episode_end messages: final_reward, final_quality, episode_statistics
  - Handle disconnections gracefully
  - _Requirements: 12.1-12.7_

- [ ] 3.4 Implement /training/metrics endpoint
  - GET /training/metrics: Return current training metrics (total_timesteps, mean_reward, episode_count, learning_rate)
  - GET /training/metrics/history: Return historical metrics as time series
  - GET /training/model/info: Return model metadata
  - GET /training/model/checkpoint: Return path to latest checkpoint
  - POST /training/model/save: Save model to HuggingFace Hub
  - _Requirements: 11.1-11.5_

- [ ] 3.5 Add CORS and error handling
  - Configure CORS to allow frontend requests
  - Implement error handlers for invalid actions, malformed requests
  - Return appropriate HTTP status codes and error messages
  - _Requirements: 22.1-22.6, 24.1-24.5_

- [ ] 3.6 Test backend endpoints with curl/Postman
  - Test all REST endpoints with sample requests
  - Test WebSocket connection and message broadcasting
  - Verify error handling for invalid inputs
  - _Requirements: 10.1-10.7, 12.1-12.7_

- [ ] 3.7 Checkpoint - Ensure all backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Part 4: React Dashboard (frontend/ folder) - Depends on Part 3

- [ ] 4.1 Create React project structure
  - Create frontend/ folder with React app
  - Set up component structure: App.jsx, TaskPanel.jsx, WorkerGrid.jsx, WorkerCard.jsx, ManagerLog.jsx, BudgetMeter.jsx, RewardChart.jsx
  - Set up styling with dark theme
  - _Requirements: 13.1-13.7_

- [ ] 4.2 Implement App.jsx with WebSocket connection
  - Establish WebSocket connection to /ws/live on component mount
  - Handle incoming messages and update application state
  - Manage episode lifecycle (start, pause, resume, end)
  - _Requirements: 13.1, 15.1-15.7_

- [ ] 4.3 Implement TaskPanel component
  - Display current task description
  - Display list of subtasks with completion status
  - Display estimated tokens and difficulty level
  - _Requirements: 13.2_

- [ ] 4.4 Implement WorkerGrid and WorkerCard components
  - Display 4 worker cards in 2x2 grid
  - Show worker_id, is_active status, progress (0-100%), hallucination_risk (0-100%), output_quality (0-100%), tokens_consumed
  - Implement red pulsing animation when hallucination detected (3 second duration)
  - Show green glow when healthy, yellow when idle, red pulse when flagged
  - _Requirements: 13.3, 14.5, 15.1-15.7_

- [ ] 4.5 Implement ManagerLog component
  - Display chronological list of manager actions
  - Show action_name, worker_id, tokens_cost, timestamp
  - Auto-scroll to latest entry
  - _Requirements: 13.4_

- [ ] 4.6 Implement BudgetMeter component
  - Display budget_remaining as progress bar
  - Show tokens_used as counter
  - Show budget_ratio as percentage
  - Color: green at 100%, yellow at 50%, red at 20%
  - _Requirements: 13.5, 15.1-15.7_

- [ ] 4.7 Implement RewardChart component with Recharts
  - Display cumulative reward over time as line chart
  - Show both raw reward line and smoothed moving average line
  - Update in real-time with WebSocket messages
  - _Requirements: 13.6, 15.1-15.7_

- [ ] 4.8 Add dark theme styling and animations
  - Implement dark theme with charcoal background
  - Add smooth transitions and animations
  - Implement pulsing red alert animation for hallucinations
  - _Requirements: 13.1, 14.1-14.7_

- [ ] 4.9 Implement interactivity features
  - Implement "Start Episode" button
  - Implement "Pause Episode" button
  - Implement "Resume Episode" button
  - Implement "End Episode" button
  - Implement hover tooltips for worker details
  - Implement "Export Data" button to download episode JSON
  - _Requirements: 14.1-14.7_

- [ ] 4.10 Test dashboard with live backend
  - Start backend server
  - Start React development server
  - Run full episode and verify all panels update correctly
  - Verify WebSocket updates arrive within 100ms
  - Verify no lag or flickering
  - _Requirements: 13.1-13.7, 15.1-15.7_

- [ ] 4.11 Checkpoint - Ensure all dashboard tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Part 5: Integration and Deployment - Depends on Parts 1-4

- [ ] 5.1 Create root README.md with project overview
  - Write 400-600 word overview of the project
  - Explain the problem being solved
  - Describe how the environment works
  - Explain failure modes and what the Manager learns
  - Include reward curve screenshot
  - _Requirements: 25.1_

- [ ] 5.2 Create HuggingFace Space deployment configuration
  - Create app.py for Gradio interface
  - Embed React dashboard in Gradio Space
  - Configure Space to run backend and frontend
  - _Requirements: 18.1-18.6_

- [ ] 5.3 Deploy to HuggingFace Spaces
  - Push code to HuggingFace Spaces repository
  - Verify deployment works on free tier
  - Test all endpoints and WebSocket connections
  - Verify response times < 30 seconds per step
  - _Requirements: 18.1-18.6_

- [ ] 5.4 Create comprehensive documentation
  - Create API documentation with endpoint descriptions and examples
  - Create architecture documentation explaining design decisions
  - Create troubleshooting guide for common issues
  - Create example_training.py showing how to train the manager
  - Create example_inference.py showing how to run trained model
  - _Requirements: 25.2-25.6_

- [ ] 5.5 Prepare 3-minute pitch materials
  - Write 30-second hook about the problem
  - Prepare live demo script (60 seconds)
  - Prepare reward curve visualization (30 seconds)
  - Prepare production angle closing (20 seconds)
  - _Requirements: 18.1-18.6_

- [ ] 5.6 Final checkpoint - Ensure all integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Start with Part 1 (environment) - this is the foundation
- Test Part 1 thoroughly with random agent before moving to training
- Parts 2 and 3 can be developed in parallel after Part 1
- Part 4 (frontend) should start after Part 3 (backend) is working
- Part 5 (deployment) is final integration
- Training must be on HuggingFace (Colab for development, HF Spaces for deployment)
- Use property-based testing with hypothesis for correctness validation
