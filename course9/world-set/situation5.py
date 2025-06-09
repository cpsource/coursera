import json
import re
from typing import Dict, List, Any, Tuple

class DomainAwarePlaybookGenerator:
    """Like having specialized experts for different types of situations"""
    
    def __init__(self):
        # Load domain templates (in real use, this would be from a file)
        self.domain_templates = self._load_domain_templates()
        self.selection_rules = self._load_selection_rules()
        
    def _load_domain_templates(self) -> Dict:
        """Load the domain-specific templates"""
        # This would normally load from the JSON file above
        return {
            "social": {
                "focus_areas": ["emotional_intelligence", "cultural_context", "relationship_dynamics"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a social interaction. Focus on: emotional dynamics, cultural considerations, power structures, communication patterns, relationship goals, and potential misunderstandings.",
                    "conditions": "For '{situation_name}', identify social conditions like: emotional states, formality level, cultural context, group size, relationship dynamics, time pressure, and environmental factors.",
                    "actions": "What social actions can an AI take in '{situation_name}'? Include: verbal responses, non-verbal behaviors, listening techniques, and relationship management moves.",
                    "validation": "Review this '{situation_name}' playbook for social appropriateness: cultural sensitivity, emotional intelligence, relationship impact, and communication effectiveness."
                }
            },
            "technical": {
                "focus_areas": ["systematic_analysis", "resource_optimization", "error_handling"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a technical problem-solving scenario. Focus on: required inputs, systematic processes, decision trees, optimization criteria, error handling, and measurable outcomes.",
                    "conditions": "For '{situation_name}', identify technical conditions like: system states, resource levels, performance metrics, error indicators, and environmental constraints.",
                    "actions": "What technical actions can an AI take in '{situation_name}'? Include: analysis steps, testing procedures, optimization moves, and error recovery actions.",
                    "validation": "Review this '{situation_name}' playbook for technical soundness: logical consistency, error handling, resource efficiency, and scalability."
                }
            },
            "safety": {
                "focus_areas": ["threat_assessment", "risk_mitigation", "emergency_response"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a safety-critical scenario. Focus on: potential hazards, risk factors, safety protocols, emergency procedures, and stakeholder protection.",
                    "conditions": "For '{situation_name}', identify safety conditions like: threat indicators, environmental hazards, available safety equipment, and escape routes.",
                    "actions": "What safety actions can an AI take in '{situation_name}'? Prioritize: immediate threat response, protective measures, emergency communications, and evacuation procedures.",
                    "validation": "Review this '{situation_name}' playbook for safety compliance: risk coverage, emergency protocols, fail-safe mechanisms, and regulatory alignment."
                }
            },
            "business": {
                "focus_areas": ["stakeholder_management", "value_optimization", "strategic_thinking"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a business scenario. Focus on: stakeholder interests, value creation, competitive dynamics, resource optimization, and success metrics.",
                    "conditions": "For '{situation_name}', identify business conditions like: market state, financial constraints, stakeholder alignment, and performance indicators.",
                    "actions": "What business actions can an AI take in '{situation_name}'? Include: strategic moves, operational optimizations, relationship management, and value creation activities.",
                    "validation": "Review this '{situation_name}' playbook for business viability: ROI potential, stakeholder impact, competitive advantage, and risk management."
                }
            },
            "learning": {
                "focus_areas": ["knowledge_acquisition", "skill_building", "performance_improvement"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a learning scenario. Focus on: learning objectives, skill progression, practice opportunities, feedback mechanisms, and mastery indicators.",
                    "conditions": "For '{situation_name}', identify learning conditions like: current skill level, motivation state, available resources, and support systems.",
                    "actions": "What learning actions can an AI take in '{situation_name}'? Include: information gathering, skill practice, knowledge testing, and progress tracking activities.",
                    "validation": "Review this '{situation_name}' playbook for educational effectiveness: learning progression, engagement factors, assessment methods, and skill transfer."
                }
            },
            "creative": {
                "focus_areas": ["idea_generation", "creative_expression", "iterative_refinement"],
                "ai_prompts": {
                    "research": "Analyze '{situation_name}' as a creative scenario. Focus on: creative goals, artistic constraints, audience expectations, inspiration sources, and evaluation criteria.",
                    "conditions": "For '{situation_name}', identify creative conditions like: inspiration level, resource availability, audience feedback, and collaborative dynamics.",
                    "actions": "What creative actions can an AI take in '{situation_name}'? Include: idea generation, concept development, skill application, and iterative improvement activities.",
                    "validation": "Review this '{situation_name}' playbook for creative quality: originality potential, artistic merit, audience appeal, and iterative improvement."
                }
            }
        }
    
    def _load_selection_rules(self) -> Dict:
        """Load rules for automatically detecting situation domains"""
        return {
            "social": ["interview", "date", "meeting", "conversation", "networking", "presentation", "negotiation", "conflict", "team", "family"],
            "technical": ["debug", "program", "system", "analyze", "optimize", "configure", "troubleshoot", "calculate", "design", "engineer"],
            "safety": ["emergency", "evacuation", "medical", "fire", "accident", "hazard", "risk", "security", "threat", "crisis"],
            "business": ["sales", "marketing", "strategy", "profit", "budget", "customer", "competition", "market", "investment", "contract"],
            "learning": ["study", "learn", "practice", "skill", "education", "training", "course", "exam", "homework", "research"],
            "creative": ["design", "art", "write", "create", "compose", "paint", "build", "craft", "innovate", "invent"]
        }
    
    def detect_domains(self, situation_name: str, situation_description: str = "") -> List[Tuple[str, float]]:
        """Automatically detect which domain(s) a situation belongs to - like classification"""
        text_to_analyze = f"{situation_name} {situation_description}".lower()
        domain_scores = {}
        
        for domain, keywords in self.selection_rules.items():
            score = 0
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 1
            
            # Normalize by number of keywords in domain
            domain_scores[domain] = score / len(keywords)
        
        # Return domains sorted by relevance
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [(domain, score) for domain, score in sorted_domains if score > 0]
    
    def generate_domain_specific_prompts(self, situation_name: str, primary_domain: str, secondary_domains: List[str] = None) -> List[Dict]:
        """Generate prompts tailored to specific domain expertise"""
        
        if primary_domain not in self.domain_templates:
            raise ValueError(f"Unknown domain: {primary_domain}")
        
        primary_template = self.domain_templates[primary_domain]
        prompts = []
        
        # Core analysis with domain expertise
        prompts.append({
            'step': 1,
            'action': 'domain_research',
            'domain': primary_domain,
            'prompt': primary_template['ai_prompts']['research'].format(situation_name=situation_name),
            'focus': primary_template['focus_areas'],
            'expected_output': f'{primary_domain.title()}-focused analysis of the situation'
        })
        
        # Domain-specific condition identification
        prompts.append({
            'step': 2,
            'action': 'identify_domain_conditions',
            'domain': primary_domain,
            'prompt': primary_template['ai_prompts']['conditions'].format(situation_name=situation_name),
            'expected_output': f'Conditions relevant to {primary_domain} domain'
        })
        
        # Domain-specific action definition
        prompts.append({
            'step': 3,
            'action': 'define_domain_actions',
            'domain': primary_domain,
            'prompt': primary_template['ai_prompts']['actions'].format(situation_name=situation_name),
            'expected_output': f'Actions appropriate for {primary_domain} domain'
        })
        
        # If multiple domains, add cross-domain analysis
        if secondary_domains:
            for i, secondary_domain in enumerate(secondary_domains[:2]):  # Limit to 2 secondary
                if secondary_domain in self.domain_templates:
                    secondary_template = self.domain_templates[secondary_domain]
                    
                    prompts.append({
                        'step': 4 + i,
                        'action': f'cross_domain_analysis_{secondary_domain}',
                        'domain': secondary_domain,
                        'prompt': f"Analyze '{situation_name}' from a {secondary_domain} perspective as a complement to the {primary_domain} analysis. What additional considerations, conditions, and actions should be included? Focus on areas where {secondary_domain} and {primary_domain} intersect.",
                        'expected_output': f'Cross-domain insights from {secondary_domain} perspective'
                    })
        
        # Integrated JSON building
        integration_prompt = f"""
        Build a comprehensive JSON playbook for '{situation_name}' that integrates insights from {primary_domain}""" + (f" and {', '.join(secondary_domains)}" if secondary_domains else "") + f""" domains.
        
        Use our compact format with:
        - 'pre': entry conditions 
        - 'cond': decision logic with AND/OR nesting, weighted for {primary_domain} priorities
        - 'act': actions with costs/requirements, including {primary_domain}-specific actions
        - 'win'/'lose': outcomes that matter for {primary_domain} success
        
        Ensure the playbook reflects {primary_domain} best practices while being actionable and measurable.
        """
        
        prompts.append({
            'step': len(prompts) + 1,
            'action': 'build_integrated_json',
            'domain': 'integrated',
            'prompt': integration_prompt,
            'expected_output': 'Complete JSON playbook with domain expertise'
        })
        
        # Domain-specific validation
        validation_prompt = primary_template['ai_prompts'].get('validation', 
            f"Review this '{situation_name}' playbook for {primary_domain} domain appropriateness and effectiveness.")
        
        prompts.append({
            'step': len(prompts) + 1,
            'action': 'domain_validation',
            'domain': primary_domain,
            'prompt': validation_prompt.format(situation_name=situation_name),
            'expected_output': f'{primary_domain.title()}-focused quality assessment'
        })
        
        return prompts
    
    def create_domain_aware_plan(self, situation_name: str, situation_description: str = "", manual_domains: List[str] = None) -> str:
        """Create a complete playbook generation plan with domain expertise"""
        
        # Detect domains automatically or use manual specification
        if manual_domains:
            detected_domains = [(domain, 1.0) for domain in manual_domains]
        else:
            detected_domains = self.detect_domains(situation_name, situation_description)
        
        if not detected_domains:
            # Fall back to general approach
            primary_domain = "general"
            secondary_domains = []
        else:
            primary_domain = detected_domains[0][0]
            secondary_domains = [domain for domain, score in detected_domains[1:3] if score > 0.1]
        
        # Generate domain-specific prompts
        prompts = self.generate_domain_specific_prompts(situation_name, primary_domain, secondary_domains)
        
        # Format the plan
        output = f"# Domain-Expert Playbook Creation Plan\n\n"
        output += f"**Situation**: {situation_name}\n"
        if situation_description:
            output += f"**Description**: {situation_description}\n"
        output += f"**Primary Domain**: {primary_domain.title()}\n"
        
        if secondary_domains:
            output += f"**Secondary Domains**: {', '.join([d.title() for d in secondary_domains])}\n"
        
        if primary_domain != "general":
            focus_areas = self.domain_templates[primary_domain]['focus_areas']
            output += f"**Domain Focus**: {', '.join(focus_areas)}\n"
        
        output += f"**Total Steps**: {len(prompts)}\n\n"
        
        # Add domain expertise note
        if primary_domain != "general":
            output += f"## 🎯 {primary_domain.title()} Domain Expertise\n\n"
            output += f"This plan leverages specialized knowledge for {primary_domain} situations, ensuring the playbook includes domain-specific best practices, common patterns, and expert-level decision making.\n\n"
        
        # List all prompts
        for prompt_info in prompts:
            output += f"## Step {prompt_info['step']}: {prompt_info['action'].title().replace('_', ' ')}\n\n"
            output += f"**Domain Focus**: {prompt_info['domain'].title()}\n\n"
            
            if 'focus' in prompt_info:
                output += f"**Key Areas**: {', '.join(prompt_info['focus'])}\n\n"
            
            output += f"**AI Prompt**:\n```\n{prompt_info['prompt']}\n```\n\n"
            output += f"**Expected Output**: {prompt_info['expected_output']}\n\n"
            output += "---\n\n"
        
        # Add domain-specific quality checklist
        output += f"## 🔍 {primary_domain.title()}-Specific Quality Checklist\n\n"
        
        quality_checks = self._get_domain_quality_checks(primary_domain)
        for check in quality_checks:
            output += f"- [ ] {check}\n"
        
        return output
    
    def _get_domain_quality_checks(self, domain: str) -> List[str]:
        """Get quality checklist items specific to each domain"""
        
        checklists = {
            "social": [
                "Emotional intelligence factors are considered",
                "Cultural sensitivity is addressed", 
                "Power dynamics and relationship impacts are evaluated",
                "Non-verbal communication is included",
                "Conflict resolution strategies are present",
                "Empathy and active listening are emphasized"
            ],
            "technical": [
                "Systematic problem-solving approach is used",
                "Error handling and edge cases are covered",
                "Resource optimization is considered",
                "Performance metrics are measurable",
                "Scalability and maintainability are addressed",
                "Testing and validation steps are included"
            ],
            "safety": [
                "Risk assessment is comprehensive",
                "Emergency protocols are clearly defined",
                "Safety equipment and resources are identified",
                "Evacuation and escape routes are planned",
                "Communication with emergency services is covered",
                "Post-incident procedures are included"
            ],
            "business": [
                "Stakeholder interests are balanced",
                "ROI and value creation are considered",
                "Competitive advantage is addressed",
                "Market conditions are factored in",
                "Risk mitigation strategies are present",
                "Success metrics are business-relevant"
            ],
            "learning": [
                "Learning objectives are clear and measurable",
                "Skill progression is logical and scaffolded",
                "Different learning styles are accommodated",
                "Practice and feedback opportunities are included",
                "Assessment and mastery criteria are defined",
                "Transfer to real-world application is addressed"
            ],
            "creative": [
                "Idea generation and brainstorming are supported",
                "Creative constraints and limitations are acknowledged",
                "Iterative refinement process is built in",
                "Audience and feedback considerations are included",
                "Originality and innovation are encouraged",
                "Creative blocks and challenges are addressed"
            ]
        }
        
        return checklists.get(domain, [
            "All conditions are observable and measurable",
            "Actions have realistic costs and durations",
            "Success/failure states are clearly defined",
            "Edge cases are comprehensively covered"
        ])

# Example usage and testing
def demo_domain_aware_generation():
    """Show how domain-specific expertise improves playbook creation"""
    
    generator = DomainAwarePlaybookGenerator()
    
    # Test different types of situations
    test_situations = [
        {
            "name": "job_interview",
            "description": "Preparing for and participating in a job interview",
            "expected_domains": ["social", "business"]
        },
        {
            "name": "debugging_software",
            "description": "Finding and fixing bugs in a computer program",
            "expected_domains": ["technical"]
        },
        {
            "name": "fire_evacuation",
            "description": "Safely evacuating a building during a fire emergency",
            "expected_domains": ["safety"]
        },
        {
            "name": "learning_piano",
            "description": "Acquiring piano playing skills through practice",
            "expected_domains": ["learning", "creative"]
        },
        {
            "name": "startup_pitch",
            "description": "Presenting a business idea to potential investors",
            "expected_domains": ["business", "social", "creative"]
        }
    ]
    
    for situation in test_situations:
        print(f"\n{'='*80}")
        print(f"DOMAIN-AWARE PLAN: {situation['name'].upper()}")
        print(f"{'='*80}")
        
        # Show automatic domain detection
        detected = generator.detect_domains(situation['name'], situation['description'])
        print(f"🔍 Auto-detected domains: {detected}")
        
        # Generate the plan
        plan = generator.create_domain_aware_plan(
            situation['name'], 
            situation['description']
        )
        
        # Show just the summary for brevity
        lines = plan.split('\n')
        summary_lines = [line for line in lines[:15] if line.strip()]
        print('\n'.join(summary_lines))
        print("\n[Full plan truncated for demo - would continue with detailed prompts...]")

if __name__ == "__main__":
    demo_domain_aware_generation()

