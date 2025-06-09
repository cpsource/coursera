import json
import re
from typing import Dict, Any, List, Union

class CompactSituationProcessor:
    """Like a mini programming language interpreter for real-world logic"""
    
    def __init__(self, situations_json: Dict):
        self.situations = situations_json
        self.current_situation = None
        
    def load_situation(self, situation_name: str):
        """Like importing a module - loads a specific situation"""
        if situation_name in self.situations:
            self.current_situation = self.situations[situation_name]
            return True
        return False
    
    def evaluate_expression(self, expr: str, world_state: Dict) -> bool:
        """Like eval() but for world conditions - handles comparisons safely"""
        # Handle different comparison operators
        operators = ['==', '!=', '<=', '>=', '<', '>']
        
        for op in operators:
            if op in expr:
                left, right = expr.split(op, 1)
                left = left.strip()
                right = right.strip()
                
                # Get left value from world state
                left_val = self._get_world_value(left, world_state)
                
                # Parse right value (could be number, string, or world state ref)
                right_val = self._parse_value(right, world_state)
                
                # Perform comparison
                return self._compare_values(left_val, right_val, op)
        
        # If no operator, treat as boolean check
        return self._get_world_value(expr, world_state)
    
    def _get_world_value(self, key: str, world_state: Dict) -> Any:
        """Extract value from world state - like dictionary lookup with defaults"""
        # Handle special functions like min_vehicle_dist
        if key == 'min_vehicle_dist':
            vehicles = world_state.get('vehicles', [])
            if vehicles:
                return min(v.get('distance', float('inf')) for v in vehicles)
            return float('inf')
        
        # Regular world state lookup
        return world_state.get(key, None)
    
    def _parse_value(self, value_str: str, world_state: Dict) -> Any:
        """Parse a value that could be number, string, or world reference"""
        value_str = value_str.strip()
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # Check if it's a world state reference
        if value_str in world_state:
            return world_state[value_str]
        
        # Return as string (remove quotes if present)
        return value_str.strip('"\'')
    
    def _compare_values(self, left: Any, right: Any, operator: str) -> bool:
        """Safely compare two values - like Python's comparison operators"""
        if left is None or right is None:
            return False
            
        try:
            if operator == '==':
                return left == right
            elif operator == '!=':
                return left != right
            elif operator == '<':
                return left < right
            elif operator == '<=':
                return left <= right
            elif operator == '>':
                return left > right
            elif operator == '>=':
                return left >= right
        except TypeError:
            # Can't compare different types
            return False
        
        return False
    
    def evaluate_condition_group(self, condition_def: Union[Dict, str], world_state: Dict) -> tuple[bool, float]:
        """Evaluate nested AND/OR logic - like recursive function evaluation"""
        
        # Simple string condition
        if isinstance(condition_def, str):
            result = self.evaluate_expression(condition_def, world_state)
            return result, 1.0 if result else 0.0
        
        # Dictionary with logic and rules
        if isinstance(condition_def, dict):
            if 'logic' in condition_def:
                return self._evaluate_logical_group(condition_def, world_state)
            else:
                # Single rule with weight
                for expr, weight in condition_def.items():
                    result = self.evaluate_expression(expr, world_state)
                    return result, weight if result else 0.0
        
        return False, 0.0
    
    def _evaluate_logical_group(self, group: Dict, world_state: Dict) -> tuple[bool, float]:
        """Handle AND/OR logic groups - like evaluating boolean expressions"""
        logic_type = group['logic'].upper()
        rules = group['rules']
        
        results = []
        total_weight = 0.0
        
        for rule in rules:
            if isinstance(rule, dict) and len(rule) == 1:
                # Single condition with weight
                expr, weight = next(iter(rule.items()))
                if isinstance(weight, (int, float)):
                    # Simple weighted condition
                    result = self.evaluate_expression(expr, world_state)
                    results.append(result)
                    if result:
                        total_weight += weight
                else:
                    # Nested logic group
                    result, weight = self.evaluate_condition_group(rule, world_state)
                    results.append(result)
                    total_weight += weight
            else:
                # Nested condition group
                result, weight = self.evaluate_condition_group(rule, world_state)
                results.append(result)
                total_weight += weight
        
        if logic_type == 'AND':
            final_result = all(results)
        elif logic_type == 'OR':
            final_result = any(results)
        else:
            final_result = False
        
        return final_result, total_weight if final_result else 0.0
    
    def can_enter_situation(self, world_state: Dict) -> bool:
        """Check preconditions - like validating function parameters"""
        if not self.current_situation:
            return False
        
        preconditions = self.current_situation.get('pre', [])
        return all(self.evaluate_expression(expr, world_state) for expr in preconditions)
    
    def get_best_action(self, world_state: Dict) -> Dict:
        """Choose best action using compact scoring - like finding max in a list"""
        if not self.can_enter_situation(world_state):
            return None
        
        conditions = self.current_situation.get('cond', {})
        actions = self.current_situation.get('act', {})
        
        # Evaluate all condition groups
        condition_scores = {}
        for cond_name, cond_def in conditions.items():
            result, weight = self.evaluate_condition_group(cond_def, world_state)
            if result:
                condition_scores[cond_name] = weight
        
        # Score each action
        action_scores = []
        for action_name, action_def in actions.items():
            base_score = 1.0
            cost = action_def.get('cost', 1.0)
            
            # Check if action has requirements
            required_condition = action_def.get('req')
            if required_condition:
                if required_condition not in condition_scores:
                    continue  # Can't perform this action
                base_score = condition_scores[required_condition]
            else:
                # Use sum of all satisfied conditions
                base_score = sum(condition_scores.values())
            
            final_score = base_score / (cost + 0.1)
            action_scores.append({
                'name': action_name,
                'score': final_score,
                'action_def': action_def
            })
        
        # Return best action
        if action_scores:
            best = max(action_scores, key=lambda x: x['score'])
            return {
                'name': best['name'],
                'score': best['score'],
                **best['action_def']
            }
        
        return None
    
    def is_complete(self, world_state: Dict) -> tuple[bool, str]:
        """Check win/lose conditions - like checking game over states"""
        if not self.current_situation:
            return False, 'no_situation'
        
        # Check win conditions
        win_conditions = self.current_situation.get('win', [])
        for condition in win_conditions:
            if self.evaluate_expression(condition, world_state):
                return True, 'success'
        
        # Check lose conditions
        lose_conditions = self.current_situation.get('lose', [])
        for condition in lose_conditions:
            if self.evaluate_expression(condition, world_state):
                return True, 'failure'
        
        return False, 'ongoing'

# Example usage and testing
def test_compact_processor():
    """Like unit tests - verify the system works correctly"""
    
    # Sample compact situations
    situations = {
        "road_crossing": {
            "desc": "Cross road safely",
            "pre": ["pos==road_edge", "goal==across"],
            "cond": {
                "safe_to_cross": {
                    "logic": "AND",
                    "rules": [
                        {"light==green": 3},
                        {"min_vehicle_dist>30": 5},
                        {
                            "logic": "OR",
                            "rules": [
                                {"visibility==clear": 2},
                                {"time_of_day==day": 1}
                            ]
                        }
                    ]
                }
            },
            "act": {
                "wait": {"cost": 0.1, "dur": 2},
                "cross": {"cost": 1.0, "dur": 5, "req": "safe_to_cross"},
                "look": {"cost": 0.05, "dur": 1}
            },
            "win": ["pos==across"],
            "lose": ["collision==true"]
        }
    }
    
    processor = CompactSituationProcessor(situations)
    processor.load_situation("road_crossing")
    
    # Test scenarios - like different input cases
    test_cases = [
        {
            "name": "Perfect conditions",
            "world": {
                "pos": "road_edge",
                "goal": "across", 
                "light": "green",
                "vehicles": [{"distance": 100}],
                "visibility": "clear",
                "time_of_day": "day"
            },
            "expected_action": "cross"
        },
        {
            "name": "Dangerous - car too close",
            "world": {
                "pos": "road_edge",
                "goal": "across",
                "light": "green", 
                "vehicles": [{"distance": 20}],
                "visibility": "clear"
            },
            "expected_action": "wait" or "look"
        },
        {
            "name": "Night but clear visibility",
            "world": {
                "pos": "road_edge",
                "goal": "across",
                "light": "green",
                "vehicles": [{"distance": 50}], 
                "visibility": "clear",
                "time_of_day": "night"
            },
            "expected_action": "cross"
        }
    ]
    
    print("=== Compact Situation Processor Tests ===\n")
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        world_state = test['world']
        
        if processor.can_enter_situation(world_state):
            action = processor.get_best_action(world_state)
            print(f"  Can enter: Yes")
            print(f"  Best action: {action['name'] if action else 'None'}")
            if action:
                print(f"  Action score: {action['score']:.2f}")
                print(f"  Action cost: {action.get('cost', 'N/A')}")
        else:
            print(f"  Can enter: No")
        
        is_done, status = processor.is_complete(world_state)
        print(f"  Status: {status}")
        print()

if __name__ == "__main__":
    test_compact_processor()

# Bonus: Situation builder helper
def build_situation(name: str, description: str) -> Dict:
    """Helper function to build situations programmatically - like a constructor"""
    return {
        name: {
            "desc": description,
            "pre": [],
            "cond": {},
            "act": {},
            "win": [],
            "lose": []
        }
    }

# Example of building a situation step by step
new_situation = build_situation("coffee_shop", "Order coffee efficiently")
new_situation["coffee_shop"]["pre"] = ["at_counter", "has_money"]
new_situation["coffee_shop"]["cond"] = {
    "quick_order": {
        "logic": "AND", 
        "rules": [
            {"line_length<3": 2},
            {"knows_order": 3}
        ]
    }
}
print("Built situation:", json.dumps(new_situation, indent=2))

