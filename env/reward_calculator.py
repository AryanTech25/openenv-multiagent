"""
Reward Calculator: Computes multi-component reward signal.

The reward has 5 components:
1. Task completion quality (0-50 points)
2. Time efficiency (0-10 points)
3. Budget efficiency (penalty if >80% used)
4. Hallucination detection (±15, -20, -3 points)
5. Step cost (-0.5 per step)
"""

from typing import Dict, Any


class RewardCalculator:
    """Calculates multi-component reward for environment."""
    
    # Reward component weights
    QUALITY_WEIGHT = 50.0
    TIME_WEIGHT = 10.0
    BUDGET_PENALTY = -5.0
    HALLUCINATION_CORRECT = 15.0
    HALLUCINATION_APPROVAL = -20.0
    FALSE_POSITIVE = -3.0
    STEP_COST = -0.5
    
    def __init__(self):
        """Initialize reward calculator."""
        pass
    
    def calculate_reward(
        self,
        final_quality: float,
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        token_budget: int,
        hallucination_interventions: int,
        hallucination_approvals: int,
        false_positives: int,
    ) -> float:
        """
        Calculate cumulative reward for episode.
        
        Preconditions:
            - 0 <= final_quality <= 1.0
            - 0 <= steps_used <= max_steps
            - 0 <= tokens_used <= token_budget
            - All intervention counts >= 0
        
        Postconditions:
            - Returns float reward value
            - Reward components are additive
            - Higher quality and efficiency yield higher rewards
            - Hallucination detection is incentivized
        
        Args:
            final_quality: Quality score of completed task (0-1)
            steps_used: Number of steps taken
            max_steps: Maximum steps allowed
            tokens_used: Number of tokens consumed
            token_budget: Total token budget
            hallucination_interventions: Number of correct hallucination detections
            hallucination_approvals: Number of hallucinations approved (bad)
            false_positives: Number of false positive interventions
        
        Returns:
            float: Total reward value
        """
        # Component 1: Task completion quality (0-50 points)
        quality_reward = self.QUALITY_WEIGHT * final_quality
        
        # Component 2: Time efficiency bonus (0-10 points)
        time_efficiency = 1.0 - (steps_used / max_steps) if max_steps > 0 else 0.0
        time_reward = self.TIME_WEIGHT * time_efficiency
        
        # Component 3: Budget efficiency penalty
        budget_ratio = tokens_used / token_budget if token_budget > 0 else 0.0
        budget_reward = self.BUDGET_PENALTY if budget_ratio > 0.8 else 0.0
        
        # Component 4: Hallucination detection
        hallucination_reward = (
            self.HALLUCINATION_CORRECT * hallucination_interventions +
            self.HALLUCINATION_APPROVAL * hallucination_approvals +
            self.FALSE_POSITIVE * false_positives
        )
        
        # Component 5: Step cost
        step_cost = self.STEP_COST * steps_used
        
        # Total reward
        total_reward = (
            quality_reward +
            time_reward +
            budget_reward +
            hallucination_reward +
            step_cost
        )
        
        return total_reward
    
    def get_reward_breakdown(
        self,
        final_quality: float,
        steps_used: int,
        max_steps: int,
        tokens_used: int,
        token_budget: int,
        hallucination_interventions: int,
        hallucination_approvals: int,
        false_positives: int,
    ) -> Dict[str, float]:
        """
        Get breakdown of reward components for analysis.
        
        Returns:
            Dict with individual component values and total
        """
        # Component 1: Task completion quality
        quality_reward = self.QUALITY_WEIGHT * final_quality
        
        # Component 2: Time efficiency
        time_efficiency = 1.0 - (steps_used / max_steps) if max_steps > 0 else 0.0
        time_reward = self.TIME_WEIGHT * time_efficiency
        
        # Component 3: Budget efficiency
        budget_ratio = tokens_used / token_budget if token_budget > 0 else 0.0
        budget_reward = self.BUDGET_PENALTY if budget_ratio > 0.8 else 0.0
        
        # Component 4: Hallucination detection
        hallucination_reward = (
            self.HALLUCINATION_CORRECT * hallucination_interventions +
            self.HALLUCINATION_APPROVAL * hallucination_approvals +
            self.FALSE_POSITIVE * false_positives
        )
        
        # Component 5: Step cost
        step_cost = self.STEP_COST * steps_used
        
        # Total
        total_reward = (
            quality_reward +
            time_reward +
            budget_reward +
            hallucination_reward +
            step_cost
        )
        
        return {
            'quality_reward': quality_reward,
            'time_reward': time_reward,
            'budget_reward': budget_reward,
            'hallucination_reward': hallucination_reward,
            'step_cost': step_cost,
            'total_reward': total_reward,
            'budget_ratio': budget_ratio,
            'time_efficiency': time_efficiency,
        }
