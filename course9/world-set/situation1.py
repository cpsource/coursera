# Situation-Action Meta-Language (SAML)
# Think of this like a recipe book where each "recipe" describes how to handle a specific situation

from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from enum import Enum

class ConditionType(Enum):
    SENSORY = "sensory"      # What the AI can perceive
    STATE = "state"          # Internal state or memory
    TEMPORAL = "temporal"    # Time-based conditions
    CONTEXTUAL = "contextual" # Environmental context

class ActionType(Enum):
    PHYSICAL = "physical"    # Real-world actions
    COGNITIVE = "cognitive"  # Mental processes
    COMMUNICATIVE = "comm"   # Interaction with others
    OBSERVATIONAL = "observe" # Gathering more information

@dataclass
class Condition:
    """Like an 'if' statement in code, but for real-world situations"""
    name: str
    condition_type: ConditionType
    check_function: Callable  # Function that returns True/False
    parameters: Dict[str, Any] = None
    weight: float = 1.0  # How important this condition is
    
    def evaluate(self, world_state: Dict) -> bool:
        return self.check_function(world_state, self.parameters or {})

@dataclass
class Action:
    """Like a function call, but for real-world actions"""
    name: str
    action_type: ActionType
    execute_function: Callable
    parameters: Dict[str, Any] = None
    cost: float = 1.0  # How expensive/risky this action is
    duration: float = 1.0  # How long it takes
    
    def execute(self, world_state: Dict) -> Dict:
        return self.execute_function(world_state, self.parameters or {})

@dataclass
class Situation:
    """Like a class definition - describes a whole scenario"""
    name: str
    description: str
    preconditions: List[Condition]  # Must be true to enter this situation
    active_conditions: List[Condition]  # Continuously evaluated while in situation
    available_actions: List[Action]
    success_conditions: List[Condition]  # When situation is resolved successfully
    failure_conditions: List[Condition]  # When situation fails
    
class SituationMatcher:
    """Like a router - matches current world state to appropriate situation"""
    
    def __init__(self):
        self.situations: List[Situation] = []
        
    def add_situation(self, situation: Situation):
        self.situations.append(situation)
        
    def match_situation(self, world_state: Dict) -> List[Situation]:
        """Find all situations that match current world state"""
        matching = []
        for situation in self.situations:
            if all(cond.evaluate(world_state) for cond in situation.preconditions):
                matching.append(situation)
        return matching
    
    def select_best_action(self, situation: Situation, world_state: Dict) -> Action:
        """Choose the best action given current conditions"""
        # This is where your AI's decision-making logic goes
        valid_actions = []
        
        for action in situation.available_actions:
            # Check if conditions favor this action
            relevance_score = 0
            for condition in situation.active_conditions:
                if condition.evaluate(world_state):
                    relevance_score += condition.weight
            
            # Factor in cost and other considerations
            action_score = relevance_score / (action.cost + 0.1)
            valid_actions.append((action, action_score))
        
        # Return highest scoring action
        return max(valid_actions, key=lambda x: x[1])[0] if valid_actions else None

# Example: Road Crossing Situation
# This is like defining a specific "recipe" for crossing roads

def check_traffic_light(world_state: Dict, params: Dict) -> bool:
    return world_state.get('traffic_light') == params.get('required_state', 'green')

def check_vehicle_distance(world_state: Dict, params: Dict) -> bool:
    min_distance = params.get('min_safe_distance', 50)
    return all(vehicle['distance'] > min_distance 
              for vehicle in world_state.get('vehicles', []))

def check_at_crosswalk(world_state: Dict, params: Dict) -> bool:
    return world_state.get('at_crosswalk', False)

def action_wait(world_state: Dict, params: Dict) -> Dict:
    return {'action_taken': 'wait', 'duration': params.get('wait_time', 1)}

def action_cross(world_state: Dict, params: Dict) -> Dict:
    return {'action_taken': 'cross', 'speed': params.get('crossing_speed', 'normal')}

def action_look_both_ways(world_state: Dict, params: Dict) -> Dict:
    # Simulate gathering more information
    new_state = world_state.copy()
    new_state['visibility_checked'] = True
    return {'action_taken': 'observe', 'new_world_state': new_state}

# Define the road crossing situation
road_crossing = Situation(
    name="road_crossing",
    description="AI agent wants to cross a road safely",
    
    preconditions=[
        Condition("at_road_edge", ConditionType.SENSORY, 
                 lambda ws, p: ws.get('position') == 'road_edge', weight=1.0),
        Condition("destination_across_road", ConditionType.STATE,
                 lambda ws, p: ws.get('goal_location') == 'across_road', weight=1.0)
    ],
    
    active_conditions=[
        Condition("light_is_green", ConditionType.SENSORY, check_traffic_light,
                 parameters={'required_state': 'green'}, weight=3.0),
        Condition("no_close_vehicles", ConditionType.SENSORY, check_vehicle_distance,
                 parameters={'min_safe_distance': 30}, weight=5.0),
        Condition("at_crosswalk", ConditionType.SENSORY, check_at_crosswalk, weight=2.0),
        Condition("good_visibility", ConditionType.SENSORY,
                 lambda ws, p: ws.get('visibility') == 'clear', weight=2.0)
    ],
    
    available_actions=[
        Action("wait", ActionType.PHYSICAL, action_wait, 
               parameters={'wait_time': 2}, cost=0.1, duration=2.0),
        Action("cross_road", ActionType.PHYSICAL, action_cross,
               parameters={'crossing_speed': 'quick'}, cost=1.0, duration=5.0),
        Action("look_around", ActionType.OBSERVATIONAL, action_look_both_ways,
               cost=0.05, duration=1.0)
    ],
    
    success_conditions=[
        Condition("reached_other_side", ConditionType.STATE,
                 lambda ws, p: ws.get('position') == 'across_road', weight=1.0)
    ],
    
    failure_conditions=[
        Condition("collision_occurred", ConditionType.STATE,
                 lambda ws, p: ws.get('collision', False), weight=1.0)
    ]
)

# Example usage
if __name__ == "__main__":
    # Create the situation matcher
    matcher = SituationMatcher()
    matcher.add_situation(road_crossing)
    
    # Example world state - like the current "variables" in your program
    current_world = {
        'position': 'road_edge',
        'goal_location': 'across_road',
        'traffic_light': 'green',
        'vehicles': [{'distance': 100, 'speed': 30}],
        'at_crosswalk': True,
        'visibility': 'clear'
    }
    
    # Match and decide
    matching_situations = matcher.match_situation(current_world)
    if matching_situations:
        situation = matching_situations[0]
        best_action = matcher.select_best_action(situation, current_world)
        print(f"Situation: {situation.name}")
        print(f"Recommended action: {best_action.name}")
        
        # Execute the action
        result = best_action.execute(current_world)
        print(f"Action result: {result}")

