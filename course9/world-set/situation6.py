import json
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PlaybookLevel(Enum):
    FUNDAMENTAL = "fundamental"    # Basic building blocks
    COMPOSITE = "composite"        # Combines fundamentals  
    ORCHESTRATOR = "orchestrator"  # Manages composites

@dataclass
class CallFrame:
    """Like a function call frame - tracks context during execution"""
    playbook_name: str
    level: PlaybookLevel
    inputs: Dict[str, Any]
    local_variables: Dict[str, Any]
    current_step: int
    parent_frame: Optional['CallFrame']

@dataclass 
class ExecutionResult:
    """Return value from playbook execution"""
    success: bool
    outputs: Dict[str, Any]
    final_action: str
    execution_time: float
    call_depth: int

class HierarchicalPlaybookExecutor:
    """Like a programming language interpreter - executes hierarchical playbooks"""
    
    def __init__(self, playbook_library: Dict):
        self.playbooks = playbook_library
        self.call_stack: List[CallFrame] = []
        self.max_call_depth = 10
        self.execution_log: List[Dict] = []
        
    def execute_playbook(self, playbook_name: str, inputs: Dict[str, Any], world_state: Dict[str, Any]) -> ExecutionResult:
        """Main execution entry point - like calling a function"""
        
        if len(self.call_stack) >= self.max_call_depth:
            raise RuntimeError(f"Maximum call depth {self.max_call_depth} exceeded")
        
        # Find the playbook
        playbook = self._find_playbook(playbook_name)
        if not playbook:
            raise ValueError(f"Playbook '{playbook_name}' not found")
        
        # Create call frame
        frame = CallFrame(
            playbook_name=playbook_name,
            level=PlaybookLevel(playbook.get('level', 'fundamental')),
            inputs=inputs,
            local_variables={},
            current_step=0,
            parent_frame=self.call_stack[-1] if self.call_stack else None
        )
        
        self.call_stack.append(frame)
        self._log_execution(f"CALL: {playbook_name}", inputs)
        
        try:
            result = self._execute_playbook_internal(playbook, world_state, frame)
            self._log_execution(f"RETURN: {playbook_name}", result.outputs)
            return result
            
        finally:
            self.call_stack.pop()
    
    def _find_playbook(self, name: str) -> Optional[Dict]:
        """Find playbook in the library hierarchy"""
        for level in ['fundamental_playbooks', 'composite_playbooks', 'orchestrator_playbooks']:
            if level in self.playbooks and name in self.playbooks[level]:
                playbook = self.playbooks[level][name].copy()
                playbook['level'] = level.split('_')[0]  # Extract level name
                return playbook
        return None
    
    def _execute_playbook_internal(self, playbook: Dict, world_state: Dict, frame: CallFrame) -> ExecutionResult:
        """Execute a single playbook - handles different levels differently"""
        
        level = PlaybookLevel(playbook['level'])
        
        if level == PlaybookLevel.FUNDAMENTAL:
            return self._execute_fundamental(playbook, world_state, frame)
        elif level == PlaybookLevel.COMPOSITE:
            return self._execute_composite(playbook, world_state, frame)
        elif level == PlaybookLevel.ORCHESTRATOR:
            return self._execute_orchestrator(playbook, world_state, frame)
        else:
            raise ValueError(f"Unknown playbook level: {level}")
    
    def _execute_fundamental(self, playbook: Dict, world_state: Dict, frame: CallFrame) -> ExecutionResult:
        """Execute fundamental playbook - like a basic function call"""
        
        # Check preconditions with input binding
        bound_world_state = self._bind_inputs(world_state, frame.inputs, frame)
        
        if not self._check_preconditions(playbook.get('pre', []), bound_world_state):
            return ExecutionResult(False, {}, "preconditions_failed", 0.0, len(self.call_stack))
        
        # Evaluate conditions and choose action
        best_action = self._choose_best_action(playbook, bound_world_state)
        if not best_action:
            return ExecutionResult(False, {}, "no_action_available", 0.0, len(self.call_stack))
        
        # Execute action and capture outputs
        action_outputs = best_action.get('output', {})
        
        # Bind outputs to expected output schema
        expected_outputs = playbook.get('outputs', [])
        final_outputs = {}
        
        for output_key in expected_outputs:
            if output_key in action_outputs:
                final_outputs[output_key] = action_outputs[output_key]
            else:
                # Provide default values based on action taken
                final_outputs[output_key] = self._generate_default_output(output_key, best_action['name'])
        
        return ExecutionResult(True, final_outputs, best_action['name'], 1.0, len(self.call_stack))
    
    def _execute_composite(self, playbook: Dict, world_state: Dict, frame: CallFrame) -> ExecutionResult:
        """Execute composite playbook - orchestrates multiple sub-calls"""
        
        workflow = playbook.get('workflow', [])
        accumulated_results = {}
        
        # Execute workflow steps
        for step_info in workflow:
            # Check if step should execute (conditional execution)
            if 'condition' in step_info:
                condition_met = self._evaluate_condition_expression(
                    step_info['condition'], world_state, frame
                )
                if not condition_met:
                    continue
            
            # Prepare inputs for sub-playbook call
            sub_inputs = self._resolve_input_references(step_info['inputs'], frame, accumulated_results)
            
            # Make the sub-call
            sub_result = self.execute_playbook(
                step_info['call_playbook'], 
                sub_inputs, 
                world_state
            )
            
            # Store results for later reference
            result_key = step_info['store_results_as']
            accumulated_results[result_key] = sub_result.outputs
            frame.local_variables[result_key] = sub_result.outputs
            
            # Update world state with any changes
            world_state.update(sub_result.outputs)
        
        # Evaluate final conditions with all accumulated results
        final_world_state = {**world_state, **accumulated_results}
        best_action = self._choose_best_action(playbook, final_world_state, frame)
        
        if not best_action:
            return ExecutionResult(False, accumulated_results, "workflow_incomplete", 2.0, len(self.call_stack))
        
        return ExecutionResult(True, accumulated_results, best_action['name'], 2.0, len(self.call_stack))
    
    def _execute_orchestrator(self, playbook: Dict, world_state: Dict, frame: CallFrame) -> ExecutionResult:
        """Execute orchestrator playbook - manages high-level strategy"""
        
        # Similar to composite but with more sophisticated control flow
        workflow = playbook.get('workflow', [])
        strategic_results = {}
        
        # Orchestrators can have parallel execution, conditional branching, etc.
        for step_info in workflow:
            # Check execution conditions
            if 'condition' in step_info and not self._evaluate_condition_expression(
                step_info['condition'], world_state, frame
            ):
                continue
            
            # Execute sub-playbook (could be composite or other orchestrator)
            sub_inputs = self._resolve_input_references(step_info['inputs'], frame, strategic_results)
            
            sub_result = self.execute_playbook(
                step_info['call_playbook'],
                sub_inputs,
                world_state
            )
            
            # Orchestrators make strategic decisions based on sub-results
            result_key = step_info['store_results_as']
            strategic_results[result_key] = {
                **sub_result.outputs,
                '_execution_success': sub_result.success,
                '_action_taken': sub_result.final_action
            }
            
            frame.local_variables[result_key] = strategic_results[result_key]
        
        # Make high-level strategic decision
        strategic_world_state = {**world_state, **strategic_results}
        best_action = self._choose_best_action(playbook, strategic_world_state, frame)
        
        return ExecutionResult(
            True, 
            strategic_results, 
            best_action['name'] if best_action else "strategy_complete",
            5.0,
            len(self.call_stack)
        )
    
    def _bind_inputs(self, world_state: Dict, inputs: Dict, frame: CallFrame) -> Dict:
        """Bind input parameters to world state - like parameter passing"""
        bound_state = world_state.copy()
        
        # Add inputs as world state variables
        for key, value in inputs.items():
            bound_state[key] = value
        
        return bound_state
    
    def _resolve_input_references(self, inputs: Dict, frame: CallFrame, accumulated_results: Dict) -> Dict:
        """Resolve @ references in inputs - like variable substitution"""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith('@'):
                # Reference to previous result
                ref_path = value[1:]  # Remove @
                
                if '.' in ref_path:
                    # Nested reference like @research_data.information_quality
                    result_key, field = ref_path.split('.', 1)
                    if result_key in accumulated_results and field in accumulated_results[result_key]:
                        resolved[key] = accumulated_results[result_key][field]
                    else:
                        resolved[key] = None
                else:
                    # Direct reference like @research_data
                    if ref_path in accumulated_results:
                        resolved[key] = accumulated_results[ref_path]
                    else:
                        resolved[key] = None
            else:
                resolved[key] = value
        
        return resolved
    
    def _check_preconditions(self, preconditions: List[str], world_state: Dict) -> bool:
        """Check if preconditions are met"""
        for condition in preconditions:
            if not self._evaluate_simple_condition(condition, world_state):
                return False
        return True
    
    def _evaluate_condition_expression(self, condition: str, world_state: Dict, frame: CallFrame) -> bool:
        """Evaluate a condition expression"""
        # Simple implementation - could be more sophisticated
        return world_state.get(condition, False)
    
    def _choose_best_action(self, playbook: Dict, world_state: Dict, frame: CallFrame = None) -> Optional[Dict]:
        """Choose the best action from available options"""
        actions = playbook.get('act', {})
        conditions = playbook.get('cond', {})
        
        # Evaluate conditions and score actions
        action_scores = []
        
        for action_name, action_def in actions.items():
            score = 1.0  # Base score
            
            # Check if action has requirements
            if 'req' in action_def:
                required_condition = action_def['req']
                if required_condition in conditions:
                    condition_met, condition_score = self._evaluate_condition_group(
                        conditions[required_condition], world_state
                    )
                    if not condition_met:
                        continue  # Skip this action
                    score = condition_score
            
            # Factor in cost
            cost = action_def.get('cost', 1.0)
            final_score = score / (cost + 0.1)
            
            action_scores.append((action_name, action_def, final_score))
        
        # Return best action
        if action_scores:
            best = max(action_scores, key=lambda x: x[2])
            return {'name': best[0], **best[1]}
        
        return None
    
    def _evaluate_condition_group(self, condition_def: Dict, world_state: Dict) -> Tuple[bool, float]:
        """Evaluate complex condition groups"""
        # Simplified version - would need full implementation from previous code
        if isinstance(condition_def, dict) and 'logic' in condition_def:
            logic_type = condition_def['logic'].upper()
            rules = condition_def['rules']
            
            results = []
            total_weight = 0.0
            
            for rule in rules:
                if isinstance(rule, dict) and len(rule) == 1:
                    expr, weight = next(iter(rule.items()))
                    if isinstance(weight, (int, float)):
                        result = self._evaluate_simple_condition(expr, world_state)
                        results.append(result)
                        if result:
                            total_weight += weight
            
            if logic_type == 'AND':
                return all(results), total_weight if all(results) else 0.0
            elif logic_type == 'OR':
                return any(results), total_weight if any(results) else 0.0
        
        return False, 0.0
    
    def _evaluate_simple_condition(self, condition: str, world_state: Dict) -> bool:
        """Evaluate simple conditions like 'field==value' or 'field>number'"""
        # Handle @ references in conditions
        if '@' in condition:
            # Resolve references first
            for key, value in world_state.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        condition = condition.replace(f'@{key}.{subkey}', str(subvalue))
        
        # Simple evaluation - could be more robust
        operators = ['==', '!=', '<=', '>=', '<', '>']
        for op in operators:
            if op in condition:
                left, right = condition.split(op, 1)
                left_val = world_state.get(left.strip(), None)
                right_val = right.strip().strip('"\'')
                
                try:
                    if right_val.replace('.', '').isdigit():
                        right_val = float(right_val)
                    
                    if op == '==': return left_val == right_val
                    elif op == '!=': return left_val != right_val
                    elif op == '<': return left_val < right_val
                    elif op == '<=': return left_val <= right_val
                    elif op == '>': return left_val > right_val
                    elif op == '>=': return left_val >= right_val
                except:
                    return False
        
        # Boolean check
        return world_state.get(condition, False)
    
    def _generate_default_output(self, output_key: str, action_name: str) -> Any:
        """Generate reasonable default outputs"""
        defaults = {
            'risk_level': 'unknown',
            'confidence_level': 5,
            'information_quality': 'partial',
            'chosen_option': action_name,
            'recommended_caution': 'normal'
        }
        return defaults.get(output_key, f"result_of_{action_name}")
    
    def _log_execution(self, event: str, data: Any):
        """Log execution for debugging"""
        self.execution_log.append({
            'event': event,
            'data': data,
            'call_depth': len(self.call_stack),
            'timestamp': len(self.execution_log)
        })
    
    def get_execution_trace(self) -> str:
        """Get a readable execution trace - like stack trace"""
        trace = "EXECUTION TRACE:\n" + "="*50 + "\n"
        
        for entry in self.execution_log:
            indent = "  " * entry['call_depth']
            trace += f"{indent}{entry['event']}: {entry['data']}\n"
        
        return trace

# Demo and testing
def demo_hierarchical_execution():
    """Demonstrate the hierarchical playbook system"""
    
    # Sample hierarchical playbook library
    playbook_library = {
        "fundamental_playbooks": {
            "assess_risk": {
                "desc": "Basic risk assessment",
                "level": "fundamental",
                "inputs": ["threat_source", "potential_impact", "likelihood"],
                "outputs": ["risk_level", "recommended_caution"],
                "pre": ["threat_identified"],
                "cond": {
                    "high_risk": {
                        "logic": "OR",
                        "rules": [{"potential_impact>7": 5}, {"likelihood>8": 4}]
                    }
                },
                "act": {
                    "assess_severe": {"cost": 0.5, "req": "high_risk", "output": {"risk_level": "high", "recommended_caution": "extreme"}},
                    "assess_low": {"cost": 0.1, "output": {"risk_level": "low", "recommended_caution": "minimal"}}
                },
                "win": ["risk_level_determined"]
            },
            "gather_information": {
                "desc": "Information collection",
                "level": "fundamental",
                "inputs": ["information_type", "urgency_level"],
                "outputs": ["information_quality", "confidence_level"],
                "pre": ["information_needed"],
                "act": {
                    "quick_search": {"cost": 0.2, "output": {"information_quality": "basic", "confidence_level": 6}},
                    "thorough_research": {"cost": 1.0, "output": {"information_quality": "comprehensive", "confidence_level": 9}}
                },
                "win": ["information_obtained"]
            },
            "make_decision": {
                "desc": "Decision making framework",
                "level": "fundamental", 
                "inputs": ["options_available", "decision_criteria"],
                "outputs": ["chosen_option", "confidence_level"],
                "pre": ["decision_required"],
                "act": {
                    "choose_best": {"cost": 0.3, "output": {"chosen_option": "optimal_choice", "confidence_level": 8}},
                    "deliberate": {"cost": 0.8, "output": {"chosen_option": "careful_choice", "confidence_level": 7}}
                },
                "win": ["decision_made"]
            }
        },
        "composite_playbooks": {
            "job_interview": {
                "desc": "Interview management using sub-playbooks",
                "level": "composite",
                "pre": ["interview_scheduled"],
                "workflow": [
                    {
                        "step": "research_company",
                        "call_playbook": "gather_information",
                        "inputs": {
                            "information_type": "company_data",
                            "urgency_level": 8
                        },
                        "store_results_as": "research_data"
                    },
                    {
                        "step": "assess_difficulty",
                        "call_playbook": "assess_risk", 
                        "inputs": {
                            "threat_source": "difficult_questions",
                            "potential_impact": 8,
                            "likelihood": 7
                        },
                        "store_results_as": "difficulty_assessment"
                    },
                    {
                        "step": "choose_strategy",
                        "call_playbook": "make_decision",
                        "inputs": {
                            "options_available": ["confident_approach", "humble_approach"],
                            "decision_criteria": "@difficulty_assessment.risk_level"
                        },
                        "store_results_as": "interview_strategy"
                    }
                ],
                "cond": {
                    "ready_for_interview": {
                        "logic": "AND",
                        "rules": [
                            {"@research_data.information_quality!=null": 3},
                            {"@interview_strategy.chosen_option!=null": 4}
                        ]
                    }
                },
                "act": {
                    "proceed_with_interview": {"cost": 2.0, "req": "ready_for_interview"},
                    "request_reschedule": {"cost": 1.5}
                },
                "win": ["interview_completed"]
            }
        }
    }
    
    # Create executor and test
    executor = HierarchicalPlaybookExecutor(playbook_library)
    
    # Test fundamental playbook
    print("=== Testing Fundamental Playbook ===")
    world_state = {"threat_identified": True}
    inputs = {"threat_source": "job_interview", "potential_impact": 8, "likelihood": 7}
    
    result = executor.execute_playbook("assess_risk", inputs, world_state)
    print(f"Risk Assessment Result: {result}")
    print()
    
    # Test composite playbook  
    print("=== Testing Composite Playbook ===")
    world_state = {
        "interview_scheduled": True,
        "information_needed": True,
        "threat_identified": True,
        "decision_required": True
    }
    inputs = {}
    
    result = executor.execute_playbook("job_interview", inputs, world_state)
    print(f"Job Interview Result: {result}")
    print()
    
    # Show execution trace
    print("=== Execution Trace ===")
    print(executor.get_execution_trace())

if __name__ == "__main__":
    demo_hierarchical_execution()

# Advanced features for production use
class PlaybookRegistry:
    """Manages playbook versions and dependencies - like a package manager"""
    
    def __init__(self):
        self.playbooks = {}
        self.versions = {}
        self.dependencies = {}
    
    def register_playbook(self, name: str, playbook: Dict, version: str = "1.0.0"):
        """Register a new playbook version"""
        if name not in self.playbooks:
            self.playbooks[name] = {}
            self.versions[name] = []
        
        self.playbooks[name][version] = playbook
        self.versions[name].append(version)
        
        # Extract dependencies
        if 'dependencies' in playbook:
            self.dependencies[f"{name}:{version}"] = playbook['dependencies']
    
    def resolve_dependencies(self, playbook_name: str, version: str = "latest") -> List[str]:
        """Resolve dependency tree - like npm install"""
        if version == "latest":
            version = max(self.versions.get(playbook_name, ["1.0.0"]))
        
        key = f"{playbook_name}:{version}"
        deps = self.dependencies.get(key, [])
        
        resolved = []
        for dep in deps:
            resolved.append(dep)
            # Recursively resolve dependencies
            resolved.extend(self.resolve_dependencies(dep))
        
        return list(set(resolved))  # Remove duplicates

class DistributedPlaybookExecutor(HierarchicalPlaybookExecutor):
    """Extends executor for distributed/parallel execution"""
    
    def __init__(self, playbook_library: Dict, worker_pool=None):
        super().__init__(playbook_library)
        self.worker_pool = worker_pool
        
    async def execute_parallel_workflow(self, workflow: List[Dict], world_state: Dict, frame: CallFrame) -> Dict:
        """Execute workflow steps in parallel where possible"""
        
        # Analyze dependencies between steps
        dependency_graph = self._build_dependency_graph(workflow)
        
        # Execute in topological order with parallelization
        results = {}
        executed = set()
        
        while len(executed) < len(workflow):
            # Find steps ready to execute (dependencies satisfied)
            ready_steps = [
                step for step in workflow 
                if step['step'] not in executed and 
                all(dep in executed for dep in dependency_graph.get(step['step'], []))
            ]
            
            if not ready_steps:
                break  # Circular dependency or error
            
            # Execute ready steps in parallel
            if self.worker_pool:
                # Use worker pool for parallel execution
                tasks = []
                for step in ready_steps:
                    task = self._submit_step_to_worker(step, world_state, results)
                    tasks.append((step['step'], task))
                
                # Wait for completion and collect results
                for step_name, task in tasks:
                    step_result = await task
                    results[step_name] = step_result
                    executed.add(step_name)
            else:
                # Sequential execution of ready steps
                for step in ready_steps:
                    step_result = self._execute_workflow_step(step, world_state, results)
                    results[step['store_results_as']] = step_result
                    executed.add(step['step'])
        
        return results
    
    def _build_dependency_graph(self, workflow: List[Dict]) -> Dict[str, List[str]]:
        """Build dependency graph from @ references in inputs"""
        dependencies = {}
        
        for step in workflow:
            step_name = step['step']
            deps = []
            
            # Check inputs for @ references
            for input_value in step.get('inputs', {}).values():
                if isinstance(input_value, str) and input_value.startswith('@'):
                    ref = input_value[1:].split('.')[0]  # Get base reference
                    # Find which step produces this reference
                    for other_step in workflow:
                        if other_step.get('store_results_as') == ref:
                            deps.append(other_step['step'])
            
            dependencies[step_name] = deps
        
        return dependencies

# Usage example showing the power of hierarchical playbooks
def real_world_example():
    """Show how hierarchical playbooks work for complex real-world scenarios"""
    
    print("""
    HIERARCHICAL PLAYBOOK EXAMPLE: Career Decision
    
    This shows how a complex life decision can be broken down into
    reusable, composable playbooks:
    
    1. ORCHESTRATOR LEVEL: 'career_advancement'
       - Manages overall career strategy
       - Calls multiple composite playbooks
       - Makes high-level strategic decisions
    
    2. COMPOSITE LEVEL: 'job_interview' and 'startup_business_plan'  
       - Each handles a specific career path option
       - Orchestrates multiple fundamental operations
       - Provides structured analysis for each option
    
    3. FUNDAMENTAL LEVEL: 'assess_risk', 'gather_information', 'make_decision'
       - Reusable building blocks
       - Handle basic cognitive operations
       - Can be used by any higher-level playbook
    
    BENEFITS:
    - REUSABILITY: 'assess_risk' works for interviews, startups, investments, etc.
    - MODULARITY: Change interview strategy without affecting startup analysis
    - TESTABILITY: Each level can be tested independently  
    - MAINTAINABILITY: Update risk assessment logic in one place
    - COMPOSABILITY: Build complex behaviors from simple parts
    
    This is like having a LIBRARY OF LIFE SKILLS that can be combined
    to handle any complex situation.
    """)

if __name__ == "__main__":
    demo_hierarchical_execution()
    real_world_example()


