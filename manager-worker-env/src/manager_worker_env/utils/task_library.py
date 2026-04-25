"""
Task Library: Collection of task templates for the environment.

Contains 15+ diverse task templates covering different domains and difficulty levels.
Each task includes subtasks, difficulty rating, and quality evaluation function.
"""

from dataclasses import dataclass
from typing import List, Callable, Optional
import numpy as np


@dataclass
class Subtask:
    """Represents a subtask within a larger task."""
    
    subtask_id: int
    description: str
    expected_output_format: str
    quality_threshold: float  # Minimum acceptable quality (0.6-0.9)


@dataclass
class Task:
    """Represents a complete task to be executed."""
    
    task_id: str
    task_type: str  # e.g., 'web_development', 'research'
    description: str
    subtasks: List[Subtask]
    difficulty: int  # 1-5
    quality_eval_fn: Callable[[str], float]  # Returns quality score 0-1
    estimated_tokens: int


class TaskLibrary:
    """Collection of task templates."""
    
    def __init__(self):
        """Initialize task library with 15+ templates."""
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[Task]:
        """Create all task templates."""
        tasks = []
        
        # Task 1: Build Landing Page
        tasks.append(Task(
            task_id='build_landing_page',
            task_type='web_development',
            description='Build a responsive landing page with hero section, features, and CTA',
            subtasks=[
                Subtask(0, 'Design layout', 'HTML structure', 0.7),
                Subtask(1, 'Style with CSS', 'CSS file', 0.7),
                Subtask(2, 'Add interactivity', 'JavaScript', 0.6),
                Subtask(3, 'Test responsiveness', 'Test report', 0.8),
            ],
            difficulty=2,
            quality_eval_fn=lambda output: self._eval_landing_page(output),
            estimated_tokens=800
        ))
        
        # Task 2: Conduct Market Research
        tasks.append(Task(
            task_id='market_research',
            task_type='research',
            description='Conduct market research for a new product category',
            subtasks=[
                Subtask(0, 'Identify competitors', 'Competitor list', 0.7),
                Subtask(1, 'Analyze market size', 'Market analysis', 0.8),
                Subtask(2, 'Survey target users', 'Survey results', 0.7),
                Subtask(3, 'Compile report', 'Final report', 0.8),
                Subtask(4, 'Present findings', 'Presentation', 0.7),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_research(output),
            estimated_tokens=1200
        ))
        
        # Task 3: Debug Codebase
        tasks.append(Task(
            task_id='debug_codebase',
            task_type='software_engineering',
            description='Debug and fix issues in a provided codebase',
            subtasks=[
                Subtask(0, 'Identify bugs', 'Bug list', 0.8),
                Subtask(1, 'Write test cases', 'Test suite', 0.8),
                Subtask(2, 'Fix bugs', 'Fixed code', 0.9),
                Subtask(3, 'Verify fixes', 'Test results', 0.9),
            ],
            difficulty=4,
            quality_eval_fn=lambda output: self._eval_debugging(output),
            estimated_tokens=1500
        ))
        
        # Task 4: Plan Product Launch
        tasks.append(Task(
            task_id='plan_product_launch',
            task_type='product_management',
            description='Plan a comprehensive product launch strategy',
            subtasks=[
                Subtask(0, 'Define launch goals', 'Goals document', 0.7),
                Subtask(1, 'Create timeline', 'Timeline', 0.8),
                Subtask(2, 'Plan marketing', 'Marketing plan', 0.7),
                Subtask(3, 'Identify risks', 'Risk assessment', 0.8),
                Subtask(4, 'Create launch checklist', 'Checklist', 0.8),
            ],
            difficulty=4,
            quality_eval_fn=lambda output: self._eval_launch_plan(output),
            estimated_tokens=1400
        ))
        
        # Task 5: Write Research Paper
        tasks.append(Task(
            task_id='write_research_paper',
            task_type='academic_writing',
            description='Write a research paper on a technical topic',
            subtasks=[
                Subtask(0, 'Literature review', 'Review document', 0.8),
                Subtask(1, 'Outline paper', 'Outline', 0.7),
                Subtask(2, 'Write sections', 'Draft', 0.8),
                Subtask(3, 'Add citations', 'Cited draft', 0.9),
                Subtask(4, 'Proofread', 'Final paper', 0.9),
            ],
            difficulty=5,
            quality_eval_fn=lambda output: self._eval_paper(output),
            estimated_tokens=2000
        ))
        
        # Task 6: Create Mobile App UI
        tasks.append(Task(
            task_id='create_mobile_app_ui',
            task_type='web_development',
            description='Design and implement mobile app UI with multiple screens',
            subtasks=[
                Subtask(0, 'Design wireframes', 'Wireframe sketches', 0.7),
                Subtask(1, 'Create mockups', 'Visual mockups', 0.8),
                Subtask(2, 'Implement screens', 'React components', 0.8),
                Subtask(3, 'Add animations', 'Animated transitions', 0.7),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_ui(output),
            estimated_tokens=1100
        ))
        
        # Task 7: Analyze Financial Data
        tasks.append(Task(
            task_id='analyze_financial_data',
            task_type='research',
            description='Analyze financial data and create investment recommendations',
            subtasks=[
                Subtask(0, 'Collect data', 'Data CSV', 0.8),
                Subtask(1, 'Clean data', 'Cleaned dataset', 0.8),
                Subtask(2, 'Perform analysis', 'Analysis report', 0.8),
                Subtask(3, 'Create visualizations', 'Charts and graphs', 0.7),
                Subtask(4, 'Write recommendations', 'Investment memo', 0.8),
            ],
            difficulty=4,
            quality_eval_fn=lambda output: self._eval_financial(output),
            estimated_tokens=1600
        ))
        
        # Task 8: Refactor Legacy Code
        tasks.append(Task(
            task_id='refactor_legacy_code',
            task_type='software_engineering',
            description='Refactor legacy code to improve maintainability and performance',
            subtasks=[
                Subtask(0, 'Analyze code', 'Analysis document', 0.7),
                Subtask(1, 'Plan refactoring', 'Refactoring plan', 0.8),
                Subtask(2, 'Implement changes', 'Refactored code', 0.9),
                Subtask(3, 'Write tests', 'Test suite', 0.8),
                Subtask(4, 'Document changes', 'Documentation', 0.7),
            ],
            difficulty=4,
            quality_eval_fn=lambda output: self._eval_refactoring(output),
            estimated_tokens=1700
        ))
        
        # Task 9: Create Marketing Campaign
        tasks.append(Task(
            task_id='create_marketing_campaign',
            task_type='product_management',
            description='Create comprehensive marketing campaign for product launch',
            subtasks=[
                Subtask(0, 'Define target audience', 'Audience profile', 0.7),
                Subtask(1, 'Create messaging', 'Campaign messaging', 0.8),
                Subtask(2, 'Design assets', 'Creative assets', 0.7),
                Subtask(3, 'Plan distribution', 'Distribution strategy', 0.8),
                Subtask(4, 'Set metrics', 'Success metrics', 0.7),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_marketing(output),
            estimated_tokens=1300
        ))
        
        # Task 10: Write Technical Documentation
        tasks.append(Task(
            task_id='write_technical_docs',
            task_type='academic_writing',
            description='Write comprehensive technical documentation for API',
            subtasks=[
                Subtask(0, 'Document endpoints', 'Endpoint docs', 0.8),
                Subtask(1, 'Write examples', 'Code examples', 0.8),
                Subtask(2, 'Create diagrams', 'Architecture diagrams', 0.7),
                Subtask(3, 'Write guides', 'User guides', 0.8),
                Subtask(4, 'Proofread', 'Final documentation', 0.9),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_documentation(output),
            estimated_tokens=1200
        ))
        
        # Task 11: Optimize Database
        tasks.append(Task(
            task_id='optimize_database',
            task_type='software_engineering',
            description='Optimize database queries and schema for performance',
            subtasks=[
                Subtask(0, 'Profile queries', 'Performance report', 0.8),
                Subtask(1, 'Identify bottlenecks', 'Bottleneck analysis', 0.8),
                Subtask(2, 'Optimize schema', 'Optimized schema', 0.9),
                Subtask(3, 'Test performance', 'Performance tests', 0.9),
                Subtask(4, 'Document changes', 'Change documentation', 0.7),
            ],
            difficulty=5,
            quality_eval_fn=lambda output: self._eval_optimization(output),
            estimated_tokens=1800
        ))
        
        # Task 12: Conduct User Research
        tasks.append(Task(
            task_id='conduct_user_research',
            task_type='research',
            description='Conduct user research to understand customer needs',
            subtasks=[
                Subtask(0, 'Design survey', 'Survey questions', 0.7),
                Subtask(1, 'Recruit participants', 'Participant list', 0.8),
                Subtask(2, 'Conduct interviews', 'Interview notes', 0.8),
                Subtask(3, 'Analyze results', 'Analysis report', 0.8),
                Subtask(4, 'Create personas', 'User personas', 0.7),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_user_research(output),
            estimated_tokens=1400
        ))
        
        # Task 13: Build REST API
        tasks.append(Task(
            task_id='build_rest_api',
            task_type='software_engineering',
            description='Build REST API with authentication and validation',
            subtasks=[
                Subtask(0, 'Design schema', 'API schema', 0.8),
                Subtask(1, 'Implement endpoints', 'API endpoints', 0.9),
                Subtask(2, 'Add authentication', 'Auth system', 0.9),
                Subtask(3, 'Write tests', 'Test suite', 0.8),
                Subtask(4, 'Deploy API', 'Deployment guide', 0.8),
            ],
            difficulty=4,
            quality_eval_fn=lambda output: self._eval_api(output),
            estimated_tokens=1600
        ))
        
        # Task 14: Create Data Visualization
        tasks.append(Task(
            task_id='create_data_visualization',
            task_type='research',
            description='Create interactive data visualizations for business intelligence',
            subtasks=[
                Subtask(0, 'Prepare data', 'Clean dataset', 0.8),
                Subtask(1, 'Design visualizations', 'Visualization designs', 0.8),
                Subtask(2, 'Implement dashboards', 'Interactive dashboards', 0.8),
                Subtask(3, 'Add interactivity', 'Interactive features', 0.7),
                Subtask(4, 'Document insights', 'Insights document', 0.7),
            ],
            difficulty=3,
            quality_eval_fn=lambda output: self._eval_visualization(output),
            estimated_tokens=1300
        ))
        
        # Task 15: Write Blog Series
        tasks.append(Task(
            task_id='write_blog_series',
            task_type='academic_writing',
            description='Write 5-part blog series on technical topic',
            subtasks=[
                Subtask(0, 'Plan series', 'Series outline', 0.7),
                Subtask(1, 'Write part 1', 'Blog post 1', 0.8),
                Subtask(2, 'Write part 2', 'Blog post 2', 0.8),
                Subtask(3, 'Write part 3', 'Blog post 3', 0.8),
                Subtask(4, 'Edit and publish', 'Published posts', 0.8),
            ],
            difficulty=2,
            quality_eval_fn=lambda output: self._eval_blog(output),
            estimated_tokens=1000
        ))
        
        return tasks
    
    def sample_task(self, difficulty: Optional[int] = None) -> Task:
        """Sample a random task, optionally filtered by difficulty."""
        if difficulty is not None:
            filtered_tasks = [t for t in self.tasks if t.difficulty == difficulty]
            if filtered_tasks:
                return filtered_tasks[np.random.randint(len(filtered_tasks))]
        
        return self.tasks[np.random.randint(len(self.tasks))]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    # Quality evaluation functions
    @staticmethod
    def _eval_landing_page(output: str) -> float:
        """Evaluate landing page quality."""
        score = 0.5
        if 'html' in output.lower():
            score += 0.2
        if 'css' in output.lower():
            score += 0.15
        if 'responsive' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_research(output: str) -> float:
        """Evaluate research quality."""
        score = 0.5
        if 'analysis' in output.lower():
            score += 0.2
        if 'data' in output.lower():
            score += 0.15
        if 'conclusion' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_debugging(output: str) -> float:
        """Evaluate debugging quality."""
        score = 0.5
        if 'bug' in output.lower():
            score += 0.2
        if 'fix' in output.lower():
            score += 0.15
        if 'test' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_launch_plan(output: str) -> float:
        """Evaluate launch plan quality."""
        score = 0.5
        if 'timeline' in output.lower():
            score += 0.2
        if 'strategy' in output.lower():
            score += 0.15
        if 'metrics' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_paper(output: str) -> float:
        """Evaluate paper quality."""
        score = 0.5
        if 'abstract' in output.lower():
            score += 0.15
        if 'methodology' in output.lower():
            score += 0.15
        if 'conclusion' in output.lower():
            score += 0.2
        return min(1.0, score)
    
    @staticmethod
    def _eval_ui(output: str) -> float:
        """Evaluate UI quality."""
        score = 0.5
        if 'design' in output.lower():
            score += 0.2
        if 'component' in output.lower():
            score += 0.15
        if 'animation' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_financial(output: str) -> float:
        """Evaluate financial analysis quality."""
        score = 0.5
        if 'analysis' in output.lower():
            score += 0.2
        if 'recommendation' in output.lower():
            score += 0.15
        if 'risk' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_refactoring(output: str) -> float:
        """Evaluate refactoring quality."""
        score = 0.5
        if 'performance' in output.lower():
            score += 0.2
        if 'maintainability' in output.lower():
            score += 0.15
        if 'test' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_marketing(output: str) -> float:
        """Evaluate marketing campaign quality."""
        score = 0.5
        if 'audience' in output.lower():
            score += 0.2
        if 'message' in output.lower():
            score += 0.15
        if 'metric' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_documentation(output: str) -> float:
        """Evaluate documentation quality."""
        score = 0.5
        if 'example' in output.lower():
            score += 0.2
        if 'guide' in output.lower():
            score += 0.15
        if 'diagram' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_optimization(output: str) -> float:
        """Evaluate optimization quality."""
        score = 0.5
        if 'performance' in output.lower():
            score += 0.2
        if 'query' in output.lower():
            score += 0.15
        if 'benchmark' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_user_research(output: str) -> float:
        """Evaluate user research quality."""
        score = 0.5
        if 'interview' in output.lower():
            score += 0.2
        if 'persona' in output.lower():
            score += 0.15
        if 'insight' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_api(output: str) -> float:
        """Evaluate API quality."""
        score = 0.5
        if 'endpoint' in output.lower():
            score += 0.2
        if 'authentication' in output.lower():
            score += 0.15
        if 'test' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_visualization(output: str) -> float:
        """Evaluate visualization quality."""
        score = 0.5
        if 'dashboard' in output.lower():
            score += 0.2
        if 'interactive' in output.lower():
            score += 0.15
        if 'insight' in output.lower():
            score += 0.15
        return min(1.0, score)
    
    @staticmethod
    def _eval_blog(output: str) -> float:
        """Evaluate blog quality."""
        score = 0.5
        if 'post' in output.lower():
            score += 0.2
        if 'topic' in output.lower():
            score += 0.15
        if 'publish' in output.lower():
            score += 0.15
        return min(1.0, score)
