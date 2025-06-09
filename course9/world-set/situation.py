from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from enum import Enum

@dataclass
class Observation:
    """What the AI can perceive about the current state"""
    name: str
    value: Any
    confidence: float  # 0.0 to 1.0
    source: str  # sensor, calculation, etc.

@dataclass
class Condition:
    """A logical condition that can be evaluated"""
    name: str
    expression: str  # e.g., "traffic_light == 'red'"
    priority: int  # higher = more important

class ActionType(Enum):
    IMMEDIATE = "immediate"  # Do right now
    CONDITIONAL = "conditional"  # Do if condition met
    SEQUENTIAL = "sequential"  # Do in order
    CONTINUOUS = "continuous"  # Keep doing

@dataclass
class Action:
    """An action the AI can take"""
    name: str
    action_type: ActionType
    preconditions: List[Condition]
    execution_function: str  # Reference to actual function
    risk_level: float  # 0.0 (safe) to 1.0 (dangerous)
    expected_outcome: str

@dataclass
class Situation:
    """Complete situational context"""
    name: str
    description: str
    observations: List[Observation]
    conditions: List[Condition]
    available_actions: List[Action]
    goal: str
    constraints: List[str]  # Hard limits
    preferences: List[str]  # Soft preferences

# Example: Road Crossing Situation
def create_road_crossing_situation():
    observations = [
        Observation("traffic_light", "red", 0.95, "visual_sensor"),
        Observation("vehicle_distance", 50.0, 0.85, "depth_sensor"),
        Observation("vehicle_speed", 25.0, 0.70, "radar"),
        Observation("pedestrian_crossing_available", True, 0.99, "map_data"),
        Observation("weather", "clear", 0.90, "weather_api"),
    ]
    
    conditions = [
        Condition("safe_to_cross", "traffic_light == 'green' AND vehicle_distance > 100", 10),
        Condition("emergency_situation", "vehicle_distance < 10 AND vehicle_speed > 30", 9),
        Condition("weather_clear", "weather in ['clear', 'partly_cloudy']", 3),
        Condition("crosswalk_available", "pedestrian_crossing_available == True", 7),
    ]
    
    actions = [
        Action(
            name="wait_at_curb",
            action_type=ActionType.CONTINUOUS,
            preconditions=[Condition("not_safe", "traffic_light != 'green'", 8)],
            execution_function="maintain_position",
            risk_level=0.1,
            expected_outcome="Stay safe while monitoring conditions"
        ),
        Action(
            name="cross_street",
            action_type=ActionType.SEQUENTIAL,
            preconditions=[Condition("safe_crossing", "traffic_light == 'green' AND vehicle_distance > 50", 10)],
            execution_function="execute_crossing_sequence",
            risk_level=0.3,
            expected_outcome="Successfully reach other side"
        ),
        Action(
            name="find_alternative_route",
            action_type=ActionType.IMMEDIATE,
            preconditions=[Condition("no_safe_crossing", "traffic_light == 'broken' OR NOT crosswalk_available", 6)],
            execution_function="search_alternate_path",
            risk_level=0.2,
            expected_outcome="Find safer crossing point"
        ),
        Action(
            name="emergency_retreat",
            action_type=ActionType.IMMEDIATE,
            preconditions=[Condition("immediate_danger", "vehicle_distance < 15 AND vehicle_speed > 20", 10)],
            execution_function="rapid_retreat_to_safety",
            risk_level=0.4,
            expected_outcome="Avoid collision"
        )
    ]
    
    return Situation(
        name="road_crossing",
        description="AI agent needs to safely cross a street",
        observations=observations,
        conditions=conditions,
        available_actions=actions,
        goal="Cross street safely and efficiently",
        constraints=[
            "Must not enter road when vehicles approaching",
            "Must use designated crossing when available",
            "Must obey traffic signals"
        ],
        preferences=[
            "Minimize crossing time",
            "Use most direct route",
            "Avoid busy intersections if possible"
        ]
    )

# Decision Engine
class SituationDecisionEngine:
    def __init__(self):
        self.situation_library = {}
    
    def register_situation(self, situation: Situation):
        """Add a situation template to the library"""
        self.situation_library[situation.name] = situation
    
    def evaluate_conditions(self, situation: Situation) -> Dict[str, bool]:
        """Evaluate all conditions in current context"""
        results = {}
        for condition in situation.conditions:
            # In real implementation, this would evaluate the expression
            # against current observation values
            results[condition.name] = self._evaluate_expression(
                condition.expression, situation.observations
            )
        return results
    
    def select_action(self, situation: Situation) -> Action:
        """Choose best action based on current conditions"""
        condition_results = self.evaluate_conditions(situation)
        
        # Filter actions by satisfied preconditions
        viable_actions = []
        for action in situation.available_actions:
            if all(condition_results.get(pc.name, False) for pc in action.preconditions):
                viable_actions.append(action)
        
        if not viable_actions:
            # Fallback to safest action
            return min(situation.available_actions, key=lambda a: a.risk_level)
        
        # Select based on priority and risk
        return min(viable_actions, key=lambda a: a.risk_level)
    
    def _evaluate_expression(self, expression: str, observations: List[Observation]) -> bool:
        """Safely evaluate condition expression"""
        # Create context from observations
        context = {obs.name: obs.value for obs in observations}
        
        # In production, use a safe expression evaluator
        # This is simplified for demonstration
        try:
            return eval(expression, {"__builtins__": {}}, context)
        except:
            return False

# Usage Example
if __name__ == "__main__":
    # Create the situation
    road_situation = create_road_crossing_situation()
    
    # Initialize decision engine
    engine = SituationDecisionEngine()
    engine.register_situation(road_situation)
    
    # Make decision
    chosen_action = engine.select_action(road_situation)
    print(f"Recommended action: {chosen_action.name}")
    print(f"Expected outcome: {chosen_action.expected_outcome}")
    
    # Show situation analysis
    conditions = engine.evaluate_conditions(road_situation)
    print("\nCondition Analysis:")
    for name, result in conditions.items():
        print(f"  {name}: {result}")

