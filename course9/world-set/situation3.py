import json
from typing import Dict, Any, List
from dataclasses import dataclass

class SituationProcessor:
    """Think of this like a JSON interpreter for real-world situations"""
    
    def __init__(self, situation_json: Dict):
        self.situation = situation_json['situation']
        self.world_schema = situation_json.get('world_state_schema', {})
        self.strategy = situation_json.get('decision_strategy', {})
    
    def evaluate_condition(self, condition: Dict, world_state: Dict) -> bool:
        """Like eval() but safer - evaluates condition expressions"""
        check_expr = condition['check']
        
        # Simple expression evaluator (in real use, you'd want something more robust)
        try:
            # Replace world_state references with actual values
            expr = self._substitute_world_state(check_expr, world_state)
            return eval(expr)
        except:
            return False
    
    def _substitute_world_state(self, expression: str, world_state: Dict) -> str:
        """Replace world_state.field references with actual values"""
        result = expression
        
        # Handle simple field access
        for key, value in world_state.items():
            if isinstance(value, str):
                result = result.replace(f'world_state.{key}', f"'{value}'")
            elif isinstance(value, bool):
                result = result.replace(f'world_state.{key}', str(value))
            elif isinstance(value, (int, float)):
                result = result.replace(f'world_state.{key}', str(value))
            elif isinstance(value, list):
                # Handle list expressions like min([v.distance for v in vehicles])
                if 'world_state.vehicles' in result and key == 'vehicles':
                    # Extract distances from vehicle objects
                    distances = [v.get('distance', float('inf')) for v in value]
                    result = result.replace('min([v.distance for v in world_state.vehicles])', 
                                          str(min(distances) if distances else float('inf')))
        
        return result
    
    def can_enter_situation(self, world_state: Dict) -> bool:
        """Check if preconditions are met - like __init__ for situations"""
        return all(
            self.evaluate_condition(cond, world_state) 
            for cond in self.situation['preconditions']
        )
    
    def get_best_action(self, world_state: Dict) -> Dict:
        """Choose the best action based on current conditions"""
        if not self.can_enter_situation(world_state):
            return None
        
        action_scores = []
        
        for action in self.situation['available_actions']:
            # Calculate condition score
            condition_score = 0
            for condition in self.situation['active_conditions']:
                if self.evaluate_condition(condition, world_state):
                    condition_score += condition['weight']
            
            # Apply cost penalty
            final_score = condition_score / (action['cost'] + 0.1)
            
            action_scores.append({
                'action': action,
                'score': final_score,
                'condition_score': condition_score
            })
        
        # Return highest scoring action
        if action_scores:
            best = max(action_scores, key=lambda x: x['score'])
            return best['action']
        
        return None
    
    def is_situation_complete(self, world_state: Dict) -> tuple[bool, str]:
        """Check if situation succeeded or failed"""
        # Check success conditions
        for condition in self.situation['success_conditions']:
            if self.evaluate_condition(condition, world_state):
                return True, 'success'
        
        # Check failure conditions  
        for condition in self.situation['failure_conditions']:
            if self.evaluate_condition(condition, world_state):
                return True, 'failure'
        
        return False, 'ongoing'

# Example usage - like a unit test
def main():
    # Load the situation (normally from a file)
    situation_data = {
        "situation": {
            "name": "road_crossing",
            "description": "AI agent wants to cross a road safely",
            "preconditions": [
                {
                    "name": "at_road_edge",
                    "type": "sensory",
                    "check": "world_state.position == 'road_edge'",
                    "weight": 1.0
                }
            ],
            "active_conditions": [
                {
                    "name": "light_is_green",
                    "type": "sensory",
                    "check": "world_state.traffic_light == 'green'",
                    "weight": 3.0
                },
                {
                    "name": "no_close_vehicles",
                    "type": "sensory", 
                    "check": "min([v.distance for v in world_state.vehicles]) > 30",
                    "weight": 5.0
                }
            ],
            "available_actions": [
                {
                    "name": "wait",
                    "type": "physical",
                    "cost": 0.1,
                    "duration": 2.0
                },
                {
                    "name": "cross_road", 
                    "type": "physical",
                    "cost": 1.0,
                    "duration": 5.0
                }
            ],
            "success_conditions": [
                {
                    "name": "reached_other_side",
                    "type": "state",
                    "check": "world_state.position == 'across_road'",
                    "weight": 1.0
                }
            ],
            "failure_conditions": []
        }
    }
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Safe to cross",
            "world_state": {
                "position": "road_edge",
                "traffic_light": "green", 
                "vehicles": [{"distance": 100, "speed": 30}]
            }
        },
        {
            "name": "Dangerous - car too close",
            "world_state": {
                "position": "road_edge",
                "traffic_light": "green",
                "vehicles": [{"distance": 20, "speed": 50}]
            }
        },
        {
            "name": "Red light",
            "world_state": {
                "position": "road_edge", 
                "traffic_light": "red",
                "vehicles": [{"distance": 100, "speed": 30}]
            }
        }
    ]
    
    processor = SituationProcessor(situation_data)
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        world_state = scenario['world_state']
        
        if processor.can_enter_situation(world_state):
            best_action = processor.get_best_action(world_state)
            print(f"Can enter situation: Yes")
            print(f"Best action: {best_action['name'] if best_action else 'None'}")
        else:
            print(f"Can enter situation: No")

if __name__ == "__main__":
    main()

# You can also create a situation library
SITUATION_LIBRARY = {
    "road_crossing": "path/to/road_crossing.json",
    "grocery_shopping": "path/to/grocery_shopping.json", 
    "job_interview": "path/to/job_interview.json"
}

def load_situation(situation_name: str) -> SituationProcessor:
    """Factory function to load situations from files"""
    if situation_name in SITUATION_LIBRARY:
        with open(SITUATION_LIBRARY[situation_name], 'r') as f:
            return SituationProcessor(json.load(f))
    else:
        raise ValueError(f"Unknown situation: {situation_name}")

