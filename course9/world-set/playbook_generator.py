import json
import re
from typing import Dict, List, Any

class PlaybookGenerator:
    """Like a factory that builds other factories - creates playbooks for any situation"""
    
    def __init__(self, creator_playbook_path: str = None):
        # Load the meta-playbook (playbook for creating playbooks)
        self.creator_playbook = {
            "playbook_creation": {
                "desc": "Create effective playbooks for any real-world situation",
                "pre": ["new_situation_identified", "has_ai_access"],
                "cond": {
                    "ready_for_analysis": {
                        "logic": "AND",
                        "rules": [
                            {"situation_complexity<8": 2},
                            {"examples_available>0": 3},
                            {"domain_knowledge>5": 2}
                        ]
                    },
                    "needs_research": {
                        "logic": "OR", 
                        "rules": [
                            {"situation_complexity>7": 4},
                            {"unfamiliar_domain": 3},
                            {"safety_critical": 5}
                        ]
                    }
                },
                "act": {
                    "research_situation": {
                        "cost": 2.0,
                        "ai_prompt": "Analyze the situation '{situation_name}' for an AI agent. What are the key decisions, conditions to check, possible actions, success/failure states, and common edge cases? Focus on observable conditions and measurable outcomes."
                    },
                    "build_json_structure": {
                        "cost": 3.0,
                        "ai_prompt": "Convert the '{situation_name}' analysis into our compact JSON playbook format. Use: 'pre' for entry conditions, 'cond' for decision logic with AND/OR nesting, 'act' for actions with costs/requirements, 'win'/'lose' for outcomes. Make expressions like 'field==value' and 'field>number'."
                    }
                }
            }
        }
        
        self.processor = CompactSituationProcessor(self.creator_playbook)
        self.processor.load_situation("playbook_creation")
        
    def generate_ai_prompts(self, situation_name: str, complexity: int = 5) -> List[Dict]:
        """Generate the sequence of AI prompts needed to create a playbook"""
        
        # Set up world state for playbook creation
        world_state = {
            "new_situation_identified": True,
            "has_ai_access": True,
            "situation_name": situation_name,
            "situation_complexity": complexity,
            "examples_available": 0,
            "domain_knowledge": 3,  # Start conservative
            "unfamiliar_domain": complexity > 6,
            "safety_critical": self._is_safety_critical(situation_name)
        }
        
        # Generate sequence of prompts
        prompts = []
        step = 1
        
        while True:
            if not self.processor.can_enter_situation(world_state):
                break
                
            action = self.processor.get_best_action(world_state)
            if not action or 'ai_prompt' not in action:
                break
            
            # Format the AI prompt with situation name
            formatted_prompt = action['ai_prompt'].format(
                situation_name=situation_name,
                situation=situation_name
            )
            
            prompts.append({
                'step': step,
                'action': action['name'],
                'prompt': formatted_prompt,
                'cost': action.get('cost', 1.0),
                'expected_output': self._get_expected_output(action['name'])
            })
            
            # Update world state based on action
            world_state = self._simulate_action_result(action['name'], world_state)
            step += 1
            
            if step > 10:  # Safety break
                break
        
        return prompts
    
    def _is_safety_critical(self, situation_name: str) -> bool:
        """Detect if a situation involves safety concerns"""
        safety_keywords = [
            'driving', 'crossing', 'medical', 'emergency', 'fire', 'accident',
            'surgery', 'flying', 'climbing', 'diving', 'chemical', 'electrical'
        ]
        return any(keyword in situation_name.lower() for keyword in safety_keywords)
    
    def _get_expected_output(self, action_name: str) -> str:
        """Describe what we expect the AI to return for each action"""
        expectations = {
            'research_situation': 'Detailed analysis of the situation including key decision points, observable conditions, possible actions, and edge cases',
            'gather_examples': 'List of 3-5 concrete scenarios with varying complexity levels',
            'identify_conditions': 'Categorized list of all conditions the AI should check (sensory, state, temporal, contextual)',
            'define_actions': 'List of possible actions with categories, estimated costs, and durations',
            'map_logic_flows': 'Decision tree showing when to choose each action and how conditions combine',
            'build_json_structure': 'Complete JSON playbook in our compact format',
            'validate_playbook': 'List of issues found and specific improvement suggestions',
            'test_scenarios': '5 test cases with world_state inputs and expected outputs'
        }
        return expectations.get(action_name, 'Relevant analysis for the current step')
    
    def _simulate_action_result(self, action_name: str, world_state: Dict) -> Dict:
        """Simulate the effect of completing an action"""
        new_state = world_state.copy()
        
        if action_name == 'research_situation':
            new_state['domain_knowledge'] = 8
            new_state['research_complete'] = True
            
        elif action_name == 'gather_examples':
            new_state['examples_available'] = 5
            new_state['examples_collected'] = 5
            
        elif action_name == 'identify_conditions':
            new_state['conditions_identified'] = True
            
        elif action_name == 'define_actions':
            new_state['actions_defined'] = True
            
        elif action_name == 'map_logic_flows':
            new_state['logic_mapped'] = True
            
        elif action_name == 'build_json_structure':
            new_state['first_draft_complete'] = True
            new_state['edge_cases_identified'] = True
            
        elif action_name == 'validate_playbook':
            new_state['validation_complete'] = True
            
        elif action_name == 'test_scenarios':
            new_state['test_scenarios_pass'] = True
        
        return new_state
    
    def create_prompt_sequence(self, situation_name: str, complexity: int = 5) -> str:
        """Generate a formatted sequence of prompts for creating a playbook"""
        prompts = self.generate_ai_prompts(situation_name, complexity)
        
        output = f"# Playbook Creation Sequence for: {situation_name}\n\n"
        output += f"**Estimated Complexity**: {complexity}/10\n"
        output += f"**Total Steps**: {len(prompts)}\n"
        output += f"**Safety Critical**: {'Yes' if self._is_safety_critical(situation_name) else 'No'}\n\n"
        
        for prompt_info in prompts:
            output += f"## Step {prompt_info['step']}: {prompt_info['action'].title().replace('_', ' ')}\n\n"
            output += f"**Cost/Effort**: {prompt_info['cost']}/10\n\n"
            output += f"**AI Prompt**:\n```\n{prompt_info['prompt']}\n```\n\n"
            output += f"**Expected Output**: {prompt_info['expected_output']}\n\n"
            output += "---\n\n"
        
        # Add quality checklist
        output += "## Final Quality Checklist\n\n"
        checklist = [
            "All conditions are observable/measurable",
            "Actions have realistic costs and durations", 
            "Success/failure states are clearly defined",
            "Edge cases and safety scenarios covered",
            "No logical contradictions or infinite loops",
            "Weights reflect real-world importance",
            "JSON syntax is valid and compact"
        ]
        
        for item in checklist:
            output += f"- [ ] {item}\n"
        
        return output

# Example usage
def demo_playbook_generation():
    """Demonstrate how to generate prompts for different situations"""
    
    generator = PlaybookGenerator()
    
    # Test with different types of situations
    test_situations = [
        ("buying_groceries", 4),
        ("job_interview", 6), 
        ("emergency_evacuation", 9),
        ("online_dating", 5),
        ("negotiating_salary", 7)
    ]
    
    for situation, complexity in test_situations:
        print(f"\n{'='*60}")
        print(f"GENERATING PLAYBOOK CREATION PLAN")
        print(f"{'='*60}")
        
        sequence = generator.create_prompt_sequence(situation, complexity)
        print(sequence)
        
        # Show just the first prompt for brevity
        prompts = generator.generate_ai_prompts(situation, complexity)
        if prompts:
            print(f"\n🤖 **FIRST AI CALL FOR {situation.upper()}:**")
            print(f"```\n{prompts[0]['prompt']}\n```")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    demo_playbook_generation()

# Utility function to save prompt sequences
def save_creation_plan(situation_name: str, complexity: int = 5, filename: str = None):
    """Save the playbook creation plan to a file"""
    generator = PlaybookGenerator()
    sequence = generator.create_prompt_sequence(situation_name, complexity)
    
    if not filename:
        filename = f"{situation_name}_creation_plan.md"
    
    with open(filename, 'w') as f:
        f.write(sequence)
    
    print(f"Creation plan saved to: {filename}")

# Integration with web AI calls
class WebAIPlaybookCreator:
    """Orchestrates the actual creation process using web AI calls"""
    
    def __init__(self, ai_call_function):
        self.ai_call = ai_call_function  # Function that calls web AI
        self.generator = PlaybookGenerator()
    
    async def create_playbook(self, situation_name: str, complexity: int = 5) -> Dict:
        """Execute the full playbook creation process"""
        prompts = self.generator.generate_ai_prompts(situation_name, complexity)
        results = {}
        
        for prompt_info in prompts:
            print(f"Executing step {prompt_info['step']}: {prompt_info['action']}")
            
            # Make the AI call
            response = await self.ai_call(prompt_info['prompt'])
            results[prompt_info['action']] = response
            
            # If this was the JSON building step, we have our playbook
            if prompt_info['action'] == 'build_json_structure':
                try:
                    # Extract JSON from response
                    playbook_json = self._extract_json_from_response(response)
                    results['final_playbook'] = playbook_json
                except Exception as e:
                    print(f"Error extracting JSON: {e}")
        
        return results
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON playbook from AI response text"""
        # Look for JSON blocks in the response
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            return json.loads(matches[0])
        
        # Try to find JSON without code blocks
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        raise ValueError("No valid JSON found in AI response")


