from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid

class ProfessionCategory(Enum):
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    LAW_ENFORCEMENT = "law_enforcement"
    LEGAL = "legal"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    CREATIVE = "creative"
    SERVICE = "service"
    TRADES = "trades"
    RESEARCH = "research"

class SkillType(Enum):
    TECHNICAL = "technical"        # Job-specific technical skills
    INTERPERSONAL = "interpersonal"  # Communication, empathy, leadership
    COGNITIVE = "cognitive"        # Problem-solving, analysis, memory
    PHYSICAL = "physical"          # Stamina, dexterity, strength
    REGULATORY = "regulatory"      # Knowledge of laws, procedures, ethics

class WorkEnvironment(Enum):
    OFFICE = "office"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    FIELD = "field"
    LABORATORY = "laboratory"
    COURTROOM = "courtroom"
    REMOTE = "remote"
    MIXED = "mixed"

@dataclass
class RequiredSkill:
    """A skill required for a profession"""
    name: str
    skill_type: SkillType
    importance_level: int  # 1-10, how critical this skill is
    minimum_proficiency: int  # 1-10, minimum level needed
    description: str = ""

@dataclass
class ProfessionalGoal:
    """A goal or objective that defines success in this profession"""
    name: str
    description: str
    measurable: bool = True  # Can this goal be quantified?
    timeframe: str = "ongoing"  # daily, weekly, monthly, yearly, career
    success_metrics: List[str] = field(default_factory=list)

@dataclass
class WorkChallenge:
    """Common challenges faced in this profession"""
    name: str
    description: str
    frequency: str  # daily, weekly, monthly, rare
    difficulty_level: int  # 1-10
    requires_skills: List[str] = field(default_factory=list)

@dataclass
class Profession:
    """Definition of a profession - like a job class template"""
    
    # Core identity
    name: str
    category: ProfessionCategory
    description: str
    
    # Work characteristics
    typical_work_environment: WorkEnvironment
    work_schedule: str  # "9-5", "shifts", "on-call", "flexible"
    education_required: str  # "high school", "bachelor's", "master's", "doctorate", "certification"
    
    # Skills and competencies
    required_skills: List[RequiredSkill] = field(default_factory=list)
    optional_skills: List[RequiredSkill] = field(default_factory=list)
    
    # Professional objectives
    primary_goals: List[ProfessionalGoal] = field(default_factory=list)
    secondary_goals: List[ProfessionalGoal] = field(default_factory=list)
    
    # Challenges and responsibilities
    common_challenges: List[WorkChallenge] = field(default_factory=list)
    key_responsibilities: List[str] = field(default_factory=list)
    
    # Performance and compensation
    success_indicators: List[str] = field(default_factory=list)
    typical_salary_range: tuple = (0, 0)  # (min, max) annual salary
    career_progression: List[str] = field(default_factory=list)
    
    # Relationships and interactions
    typical_interactions: Dict[str, str] = field(default_factory=dict)  # role: frequency
    reports_to: List[str] = field(default_factory=list)
    manages: List[str] = field(default_factory=list)
    
    def add_required_skill(self, name: str, skill_type: SkillType, importance: int, 
                          min_proficiency: int, description: str = ""):
        """Add a required skill to this profession"""
        skill = RequiredSkill(name, skill_type, importance, min_proficiency, description)
        self.required_skills.append(skill)
    
    def add_goal(self, name: str, description: str, timeframe: str = "ongoing", 
                success_metrics: List[str] = None, is_primary: bool = True):
        """Add a professional goal"""
        goal = ProfessionalGoal(name, description, True, timeframe, success_metrics or [])
        if is_primary:
            self.primary_goals.append(goal)
        else:
            self.secondary_goals.append(goal)
    
    def add_challenge(self, name: str, description: str, frequency: str, 
                     difficulty: int, required_skills: List[str] = None):
        """Add a common work challenge"""
        challenge = WorkChallenge(name, description, frequency, difficulty, required_skills or [])
        self.common_challenges.append(challenge)
    
    def get_skills_by_type(self, skill_type: SkillType) -> List[RequiredSkill]:
        """Get all skills of a specific type"""
        return [skill for skill in self.required_skills if skill.skill_type == skill_type]
    
    def get_critical_skills(self, importance_threshold: int = 8) -> List[RequiredSkill]:
        """Get the most critical skills for this profession"""
        return [skill for skill in self.required_skills if skill.importance_level >= importance_threshold]

class ProfessionPerformance(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

@dataclass
class SkillInstance:
    """An individual's proficiency in a specific skill"""
    skill_name: str
    current_level: int  # 1-10 proficiency
    experience_years: float
    last_used: datetime = field(default_factory=datetime.now)
    improvement_rate: float = 0.1  # How quickly this person learns this skill
    
    def practice_skill(self, hours: float):
        """Improve skill through practice"""
        improvement = hours * self.improvement_rate * 0.01  # Small incremental improvement
        self.current_level = min(10.0, self.current_level + improvement)
        self.last_used = datetime.now()
    
    def skill_decay(self, days_unused: int):
        """Skills decay without use"""
        if days_unused > 30:  # Start decay after a month
            decay_rate = 0.001 * (days_unused - 30)
            self.current_level = max(1.0, self.current_level - decay_rate)

@dataclass
class ProfessionInstance:
    """A specific person working in a profession - like an employee record"""
    
    # Identity
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    person_name: str = ""
    profession_type: str = ""  # References Profession name
    
    # Career information
    years_experience: float = 0.0
    current_employer: str = ""
    job_title: str = ""
    department: str = ""
    
    # Skills and competencies
    skill_levels: Dict[str, SkillInstance] = field(default_factory=dict)
    certifications: List[Dict] = field(default_factory=list)  # name, date_earned, expires
    training_completed: List[Dict] = field(default_factory=list)
    
    # Performance tracking
    current_performance: ProfessionPerformance = ProfessionPerformance.SATISFACTORY
    goal_achievements: Dict[str, float] = field(default_factory=dict)  # goal_name: completion%
    performance_history: List[Dict] = field(default_factory=list)
    
    # Work status
    employment_start_date: datetime = field(default_factory=datetime.now)
    current_salary: int = 0
    work_location: str = ""
    work_schedule: str = "full-time"
    
    # Relationships and network
    colleagues: Dict[str, str] = field(default_factory=dict)  # person_id: relationship_type
    mentors: List[str] = field(default_factory=list)
    mentees: List[str] = field(default_factory=list)
    
    # Personal characteristics
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    career_goals: List[str] = field(default_factory=list)
    
    def add_skill(self, skill_name: str, initial_level: int = 1, experience_years: float = 0.0):
        """Add or update a skill"""
        self.skill_levels[skill_name] = SkillInstance(
            skill_name=skill_name,
            current_level=initial_level,
            experience_years=experience_years
        )
    
    def get_skill_level(self, skill_name: str) -> int:
        """Get current proficiency level in a skill"""
        skill_instance = self.skill_levels.get(skill_name)
        return skill_instance.current_level if skill_instance else 0
    
    def practice_skill(self, skill_name: str, hours: float):
        """Practice and improve a skill"""
        if skill_name in self.skill_levels:
            self.skill_levels[skill_name].practice_skill(hours)
    
    def evaluate_against_profession(self, profession: Profession) -> Dict:
        """Evaluate how well this person matches their profession requirements"""
        evaluation = {
            "overall_readiness": 0.0,
            "skill_gaps": [],
            "strengths": [],
            "readiness_by_category": {}
        }
        
        # Check each required skill
        total_importance = 0
        weighted_score = 0
        
        for required_skill in profession.required_skills:
            skill_name = required_skill.name
            current_level = self.get_skill_level(skill_name)
            required_level = required_skill.minimum_proficiency
            importance = required_skill.importance_level
            
            total_importance += importance
            
            if current_level >= required_level:
                # Skill meets requirements
                weighted_score += importance
                if current_level > required_level + 2:
                    evaluation["strengths"].append(skill_name)
            else:
                # Skill gap identified
                gap = required_level - current_level
                evaluation["skill_gaps"].append({
                    "skill": skill_name,
                    "current_level": current_level,
                    "required_level": required_level,
                    "gap": gap,
                    "importance": importance
                })
        
        # Calculate overall readiness
        evaluation["overall_readiness"] = weighted_score / total_importance if total_importance > 0 else 0.0
        
        # Readiness by skill category
        for skill_type in SkillType:
            category_skills = profession.get_skills_by_type(skill_type)
            if category_skills:
                category_score = 0
                category_importance = 0
                
                for skill in category_skills:
                    current_level = self.get_skill_level(skill.name)
                    if current_level >= skill.minimum_proficiency:
                        category_score += skill.importance_level
                    category_importance += skill.importance_level
                
                evaluation["readiness_by_category"][skill_type.value] = (
                    category_score / category_importance if category_importance > 0 else 0.0
                )
        
        return evaluation
    
    def update_goal_progress(self, goal_name: str, completion_percentage: float):
        """Update progress on a professional goal"""
        self.goal_achievements[goal_name] = min(100.0, max(0.0, completion_percentage))
    
    def add_certification(self, name: str, issuing_body: str, date_earned: datetime, 
                         expires: datetime = None):
        """Add a professional certification"""
        cert = {
            "name": name,
            "issuing_body": issuing_body,
            "date_earned": date_earned.isoformat(),
            "expires": expires.isoformat() if expires else None,
            "active": expires is None or expires > datetime.now()
        }
        self.certifications.append(cert)
    
    def get_experience_level(self) -> str:
        """Get experience level category"""
        if self.years_experience < 2:
            return "entry_level"
        elif self.years_experience < 5:
            return "junior"
        elif self.years_experience < 10:
            return "mid_level"
        elif self.years_experience < 20:
            return "senior"
        else:
            return "expert"
    
    def get_career_summary(self) -> Dict:
        """Get a summary of this person's career status"""
        return {
            "name": self.person_name,
            "profession": self.profession_type,
            "title": self.job_title,
            "experience_level": self.get_experience_level(),
            "years_experience": self.years_experience,
            "current_performance": self.current_performance.value,
            "skill_count": len(self.skill_levels),
            "certifications": len([c for c in self.certifications if c["active"]]),
            "goal_completion_avg": sum(self.goal_achievements.values()) / len(self.goal_achievements) if self.goal_achievements else 0.0
        }

class ProfessionLibrary:
    """Manages profession definitions and instances"""
    
    def __init__(self):
        self.professions: Dict[str, Profession] = {}
        self.instances: Dict[str, ProfessionInstance] = {}
        self.instances_by_profession: Dict[str, List[str]] = {}
    
    def add_profession(self, profession: Profession):
        """Add a profession definition"""
        self.professions[profession.name] = profession
        if profession.name not in self.instances_by_profession:
            self.instances_by_profession[profession.name] = []
    
    def create_professional(self, profession_name: str, person_name: str, **kwargs) -> ProfessionInstance:
        """Create a new professional instance"""
        if profession_name not in self.professions:
            raise ValueError(f"Profession '{profession_name}' not found")
        
        instance = ProfessionInstance(
            person_name=person_name,
            profession_type=profession_name,
            **kwargs
        )
        
        self.instances[instance.instance_id] = instance
        self.instances_by_profession[profession_name].append(instance.instance_id)
        
        return instance
    
    def get_professionals_by_type(self, profession_name: str) -> List[ProfessionInstance]:
        """Get all professionals in a specific profession"""
        instance_ids = self.instances_by_profession.get(profession_name, [])
        return [self.instances[iid] for iid in instance_ids if iid in self.instances]
    
    def evaluate_professional_fit(self, instance_id: str) -> Dict:
        """Evaluate how well a professional fits their profession"""
        instance = self.instances.get(instance_id)
        if not instance:
            return {"error": "Instance not found"}
        
        profession = self.professions.get(instance.profession_type)
        if not profession:
            return {"error": "Profession definition not found"}
        
        return instance.evaluate_against_profession(profession)

# Create example professions
def create_example_professions() -> ProfessionLibrary:
    """Create example profession definitions"""
    
    library = ProfessionLibrary()
    
    # Medical Doctor
    doctor = Profession(
        name="medical_doctor",
        category=ProfessionCategory.HEALTHCARE,
        description="Diagnoses and treats illnesses, injuries, and medical conditions",
        typical_work_environment=WorkEnvironment.HOSPITAL,
        work_schedule="on-call",
        education_required="medical degree + residency"
    )
    
    # Required skills for doctors
    doctor.add_required_skill("medical_diagnosis", SkillType.COGNITIVE, 10, 8, "Ability to diagnose medical conditions")
    doctor.add_required_skill("patient_communication", SkillType.INTERPERSONAL, 9, 7, "Communicate effectively with patients")
    doctor.add_required_skill("medical_procedures", SkillType.TECHNICAL, 9, 8, "Perform medical procedures safely")
    doctor.add_required_skill("empathy", SkillType.INTERPERSONAL, 8, 6, "Show compassion for patient suffering")
    doctor.add_required_skill("stress_management", SkillType.COGNITIVE, 8, 7, "Handle high-pressure situations")
    doctor.add_required_skill("medical_ethics", SkillType.REGULATORY, 10, 9, "Follow medical ethics and regulations")
    
    # Professional goals for doctors
    doctor.add_goal("patient_care", "Provide excellent medical care to patients", "daily", 
                   ["patient_satisfaction_scores", "treatment_success_rates"])
    doctor.add_goal("continuing_education", "Stay current with medical advances", "ongoing",
                   ["courses_completed", "conferences_attended"])
    doctor.add_goal("error_prevention", "Minimize medical errors", "ongoing",
                   ["error_rate", "safety_protocols_followed"])
    
    # Common challenges
    doctor.add_challenge("complex_diagnosis", "Difficult cases with unclear symptoms", "weekly", 8,
                        ["medical_diagnosis", "research_skills"])
    doctor.add_challenge("emotional_stress", "Dealing with patient suffering and death", "daily", 7,
                        ["empathy", "stress_management"])
    doctor.add_challenge("time_pressure", "Making quick decisions in emergencies", "daily", 9,
                        ["decision_making", "stress_management"])
    
    doctor.key_responsibilities = [
        "Examine and diagnose patients",
        "Prescribe treatments and medications", 
        "Perform medical procedures",
        "Maintain patient records",
        "Collaborate with healthcare team"
    ]
    
    doctor.typical_salary_range = (200000, 500000)
    doctor.career_progression = ["Resident", "Attending Physician", "Senior Physician", "Department Head"]
    
    library.add_profession(doctor)
    
    # Police Detective
    detective = Profession(
        name="police_detective",
        category=ProfessionCategory.LAW_ENFORCEMENT,
        description="Investigates crimes and gathers evidence to solve cases",
        typical_work_environment=WorkEnvironment.FIELD,
        work_schedule="shifts",
        education_required="high school + police academy"
    )
    
    # Detective skills
    detective.add_required_skill("investigation", SkillType.COGNITIVE, 10, 8, "Conduct thorough investigations")
    detective.add_required_skill("interviewing", SkillType.INTERPERSONAL, 9, 7, "Interview witnesses and suspects")
    detective.add_required_skill("evidence_analysis", SkillType.TECHNICAL, 9, 7, "Analyze physical and digital evidence")
    detective.add_required_skill("report_writing", SkillType.TECHNICAL, 8, 6, "Write detailed investigation reports")
    detective.add_required_skill("firearms_proficiency", SkillType.PHYSICAL, 8, 7, "Safe and accurate use of firearms")
    detective.add_required_skill("criminal_law", SkillType.REGULATORY, 9, 8, "Knowledge of criminal law and procedures")
    
    # Detective goals
    detective.add_goal("case_resolution", "Solve assigned criminal cases", "ongoing",
                      ["case_clearance_rate", "conviction_rate"])
    detective.add_goal("evidence_integrity", "Maintain chain of custody for evidence", "daily",
                      ["evidence_handling_compliance"])
    detective.add_goal("community_safety", "Protect and serve the community", "ongoing",
                      ["crime_reduction", "community_feedback"])
    
    # Detective challenges
    detective.add_challenge("cold_cases", "Solving cases with limited evidence", "monthly", 9,
                           ["investigation", "persistence", "evidence_analysis"])
    detective.add_challenge("uncooperative_witnesses", "Getting information from reluctant people", "weekly", 6,
                           ["interviewing", "persuasion", "empathy"])
    detective.add_challenge("dangerous_situations", "Apprehending dangerous suspects", "monthly", 8,
                           ["physical_fitness", "tactical_skills", "firearms_proficiency"])
    
    detective.key_responsibilities = [
        "Investigate criminal cases",
        "Interview witnesses and suspects",
        "Collect and analyze evidence",
        "Write investigation reports",
        "Testify in court proceedings"
    ]
    
    detective.typical_salary_range = (50000, 100000)
    detective.career_progression = ["Police Officer", "Detective", "Senior Detective", "Detective Sergeant"]
    
    library.add_profession(detective)
    
    # High School Teacher
    teacher = Profession(
        name="high_school_teacher",
        category=ProfessionCategory.EDUCATION,
        description="Educates high school students in specific subject areas",
        typical_work_environment=WorkEnvironment.SCHOOL,
        work_schedule="9-5",
        education_required="bachelor's degree + teaching certification"
    )
    
    # Teacher skills
    teacher.add_required_skill("subject_expertise", SkillType.COGNITIVE, 9, 8, "Deep knowledge of subject matter")
    teacher.add_required_skill("lesson_planning", SkillType.COGNITIVE, 8, 7, "Plan effective learning experiences")
    teacher.add_required_skill("classroom_management", SkillType.INTERPERSONAL, 9, 7, "Maintain productive learning environment")
    teacher.add_required_skill("student_engagement", SkillType.INTERPERSONAL, 8, 6, "Keep students interested and motivated")
    teacher.add_required_skill("assessment", SkillType.TECHNICAL, 7, 6, "Evaluate student learning effectively")
    teacher.add_required_skill("patience", SkillType.INTERPERSONAL, 8, 7, "Work calmly with adolescent students")
    
    # Teacher goals
    teacher.add_goal("student_achievement", "Help students master subject material", "ongoing",
                    ["test_scores", "graduation_rates", "college_acceptance"])
    teacher.add_goal("student_engagement", "Create engaging learning experiences", "daily",
                    ["attendance_rates", "participation_levels"])
    teacher.add_goal("professional_development", "Improve teaching skills and knowledge", "yearly",
                    ["training_hours", "new_certifications"])
    
    # Teacher challenges
    teacher.add_challenge("diverse_learners", "Teaching students with different abilities and backgrounds", "daily", 7,
                         ["differentiation", "cultural_awareness", "patience"])
    teacher.add_challenge("behavioral_issues", "Managing disruptive student behavior", "weekly", 6,
                         ["classroom_management", "conflict_resolution"])
    teacher.add_challenge("standardized_testing", "Preparing students for required assessments", "yearly", 5,
                         ["test_preparation", "curriculum_alignment"])
    
    teacher.key_responsibilities = [
        "Plan and deliver lessons",
        "Assess student progress", 
        "Manage classroom behavior",
        "Communicate with parents",
        "Participate in school activities"
    ]
    
    teacher.typical_salary_range = (35000, 70000)
    teacher.career_progression = ["Student Teacher", "Teacher", "Senior Teacher", "Department Head", "Principal"]
    
    library.add_profession(teacher)
    
    return library

# Demo and testing
def demo_profession_system():
    """Demonstrate the profession system"""
    
    library = create_example_professions()
    
    print("=== Profession System Demo ===\n")
    
    # Show profession definitions
    print("📋 Available professions:")
    for name, profession in library.professions.items():
        print(f"  - {profession.name} ({profession.category.value})")
        print(f"    Education: {profession.education_required}")
        print(f"    Environment: {profession.typical_work_environment.value}")
        print(f"    Key skills: {len(profession.required_skills)}")
    print()
    
    # Create some professional instances
    dr_smith = library.create_professional(
        "medical_doctor", 
        "Dr. Sarah Smith",
        years_experience=8.0,
        current_employer="City General Hospital",
        job_title="Emergency Medicine Physician",
        current_salary=320000
    )
    
    # Add skills to Dr. Smith
    dr_smith.add_skill("medical_diagnosis", 9, 8.0)
    dr_smith.add_skill("patient_communication", 8, 8.0)
    dr_smith.add_skill("medical_procedures", 9, 8.0)
    dr_smith.add_skill("empathy", 7, 8.0)
    dr_smith.add_skill("stress_management", 8, 8.0)
    dr_smith.add_skill("medical_ethics", 10, 8.0)
    
    # Create a detective
    det_jones = library.create_professional(
        "police_detective",
        "Detective Mike Jones", 
        years_experience=12.0,
        current_employer="Metro Police Department",
        job_title="Senior Detective",
        current_salary=75000
    )
    
    # Add skills to Detective Jones
    det_jones.add_skill("investigation", 9, 12.0)
    det_jones.add_skill("interviewing", 8, 12.0)
    det_jones.add_skill("evidence_analysis", 7, 10.0)
    det_jones.add_skill("report_writing", 6, 12.0)  # This could be improved
    det_jones.add_skill("firearms_proficiency", 9, 12.0)
    det_jones.add_skill("criminal_law", 8, 12.0)
    
    # Create a teacher
    ms_garcia = library.create_professional(
        "high_school_teacher",
        "Ms. Maria Garcia",
        years_experience=5.0,
        current_employer="Lincoln High School",
        job_title="Mathematics Teacher",
        current_salary=48000
    )
    
    # Add skills to Ms. Garcia
    ms_garcia.add_skill("subject_expertise", 9, 5.0)  # Strong in math
    ms_garcia.add_skill("lesson_planning", 7, 5.0)
    ms_garcia.add_skill("classroom_management", 6, 5.0)  # Still developing
    ms_garcia.add_skill("student_engagement", 8, 5.0)
    ms_garcia.add_skill("assessment", 7, 5.0)
    ms_garcia.add_skill("patience", 8, 5.0)
    
    print("👥 Professional instances created:")
    for professional in [dr_smith, det_jones, ms_garcia]:
        summary = professional.get_career_summary()
        print(f"  - {summary['name']}: {summary['title']} ({summary['experience_level']})")
        print(f"    Experience: {summary['years_experience']} years")
        print(f"    Skills: {summary['skill_count']}, Performance: {summary['current_performance']}")
    print()
    
    # Evaluate professional fitness
    print("📊 Professional Fitness Evaluations:")
    for professional in [dr_smith, det_jones, ms_garcia]:
        evaluation = library.evaluate_professional_fit(professional.instance_id)
        print(f"\n  {professional.person_name} ({professional.profession_type}):")
        print(f"    Overall Readiness: {evaluation['overall_readiness']:.1%}")
        
        if evaluation['strengths']:
            print(f"    Strengths: {', '.join(evaluation['strengths'])}")
        
        if evaluation['skill_gaps']:
            print("    Skill Gaps:")
            for gap in evaluation['skill_gaps']:
                print(f"      - {gap['skill']}: need {gap['required_level']}, have {gap['current_level']} (gap: {gap['gap']})")
        
        print("    Readiness by Category:")
        for category, score in evaluation['readiness_by_category'].items():
            print(f"      - {category.replace('_', ' ').title()}: {score:.1%}")

if __name__ == "__main__":
    demo_profession_system()

# Advanced profession features
class CareerDevelopment:
    """Manages career progression and development planning"""
    
    @staticmethod
    def suggest_skill_improvements(professional: ProfessionInstance, profession: Profession) -> List[Dict]:
        """Suggest which skills to focus on improving"""
        suggestions = []
        
        for required_skill in profession.required_skills:
            current_level = professional.get_skill_level(required_skill.name)
            
            if current_level < required_skill.minimum_proficiency:
                # Critical gap
                priority = "high"
                gap = required_skill.minimum_proficiency - current_level
            elif current_level < required_skill.minimum_proficiency + 2:
                # Could be stronger
                priority = "medium" 
                gap = (required_skill.minimum_proficiency + 2) - current_level
            else:
                continue  # Skill is adequate
            
            suggestions.append({
                "skill": required_skill.name,
                "current_level": current_level,
                "target_level": required_skill.minimum_proficiency + (2 if priority == "medium" else 0),
                "gap": gap,
                "priority": priority,
                "importance": required_skill.importance_level
            })
        
        # Sort by priority and importance
        return sorted(suggestions, key=lambda x: (x['priority'] == 'high', x['importance']), reverse=True)
    
    @staticmethod
    def predict_career_progression(professional: ProfessionInstance, profession: Profession) -> Dict:
        """Predict likely career advancement timeline"""
        current_experience = professional.years_experience
        current_performance = professional.current_performance
        
        # Simple progression model based on experience and performance
        progression_stages = profession.career_progression
        current_stage_index = min(len(progression_stages) - 1, int(current_experience / 3))
        
        prediction = {
            "current_stage": progression_stages[current_stage_index],
            "next_stage": progression_stages[min(current_stage_index + 1, len(progression_stages) - 1)],
            "years_to_next_stage": max(1, 3 - (current_experience % 3)),
            "promotion_likelihood": "high" if current_performance in [ProfessionPerformance.EXCELLENT, ProfessionPerformance.GOOD] else "medium"
        }
        
        return prediction

def demo_career_development():
    """Demonstrate career development features"""
    
    library = create_example_professions()
    
    # Create a junior teacher who needs development
    junior_teacher = library.create_professional(
        "high_school_teacher",
        "Mr. Alex Chen",
        years_experience=1.5,
        current_salary=38000
    )
    
    # Add current skills (some gaps)
    junior_teacher.add_skill("subject_expertise", 8, 1.5)
    junior_teacher.add_skill("lesson_planning", 5, 1.5)  # Below minimum
    junior_teacher.add_skill("classroom_management", 4, 1.5)  # Well below minimum
    junior_teacher.add_skill("student_engagement", 6, 1.5)
    junior_teacher.add_skill("assessment", 5, 1.5)
    junior_teacher.add_skill("patience", 7, 1.5)
    
    print("\n=== Career Development Demo ===\n")
    
    # Get skill improvement suggestions
    profession = library.professions["high_school_teacher"]
    suggestions = CareerDevelopment.suggest_skill_improvements(junior_teacher, profession)
    
    print(f"📈 Skill Development Plan for {junior_teacher.person_name}:")
    for suggestion in suggestions[:5]:  # Top 5 suggestions
        print(f"  {suggestion['priority'].upper()} PRIORITY: {suggestion['skill']}")
        print(f"    Current: {suggestion['current_level']}/10, Target: {suggestion['target_level']}/10")
        print(f"    Gap: {suggestion['gap']} levels, Importance: {suggestion['importance']}/10")
        print()
    
    # Career progression prediction
    progression = CareerDevelopment.predict_career_progression(junior_teacher, profession)
    print(f"🎯 Career Progression for {junior_teacher.person_name}:")
    print(f"  Current Stage: {progression['current_stage']}")
    print(f"  Next Stage: {progression['next_stage']}")
    print(f"  Estimated Time to Promotion: {progression['years_to_next_stage']} years")
    print(f"  Promotion Likelihood: {progression['promotion_likelihood']}")

if __name__ == "__main__":
    demo_profession_system()
    demo_career_development()

# Integration with other systems
class ProfessionalWorkflow:
    """Integrates professions with playbook and object systems"""
    
    @staticmethod
    def generate_daily_tasks(professional: ProfessionInstance, profession: Profession) -> List[Dict]:
        """Generate daily tasks based on profession and individual capabilities"""
        tasks = []
        
        # Tasks based on key responsibilities
        for responsibility in profession.key_responsibilities:
            task_difficulty = 5  # Base difficulty
            
            # Adjust difficulty based on professional's skills
            relevant_skills = profession.get_critical_skills()
            if relevant_skills:
                avg_skill_level = sum(professional.get_skill_level(skill.name) for skill in relevant_skills) / len(relevant_skills)
                task_difficulty = max(1, 10 - avg_skill_level)  # Higher skills = easier tasks
            
            tasks.append({
                "task": responsibility,
                "difficulty": task_difficulty,
                "estimated_time": 2.0,  # hours
                "skills_used": [skill.name for skill in relevant_skills[:3]],  # Top 3 relevant skills
                "importance": 8
            })
        
        # Add challenge-based tasks
        for challenge in profession.common_challenges:
            if challenge.frequency in ["daily", "weekly"]:
                tasks.append({
                    "task": f"Handle: {challenge.name}",
                    "difficulty": challenge.difficulty_level,
                    "estimated_time": challenge.difficulty_level * 0.5,
                    "skills_used": challenge.requires_skills,
                    "importance": challenge.difficulty_level
                })
        
        return sorted(tasks, key=lambda x: x['importance'], reverse=True)
    
    @staticmethod
    def assess_task_readiness(professional: ProfessionInstance, task: Dict) -> float:
        """Assess how ready a professional is to handle a specific task"""
        if not task.get('skills_used'):
            return 0.7  # Neutral readiness for tasks without defined skills
        
        skill_scores = []
        for skill_name in task['skills_used']:
            skill_level = professional.get_skill_level(skill_name)
            required_level = task['difficulty']
            
            if skill_level >= required_level:
                skill_scores.append(1.0)
            else:
                # Partial readiness based on skill gap
                skill_scores.append(skill_level / required_level)
        
        return sum(skill_scores) / len(skill_scores) if skill_scores else 0.5

class ProfessionMatchmaking:
    """Helps match people to professions and vice versa"""
    
    @staticmethod
    def find_best_profession_match(skills: Dict[str, int], library: ProfessionLibrary) -> List[Tuple[str, float]]:
        """Find professions that best match someone's skills"""
        matches = []
        
        for prof_name, profession in library.professions.items():
            match_score = 0.0
            total_importance = 0
            
            for required_skill in profession.required_skills:
                skill_level = skills.get(required_skill.name, 0)
                importance = required_skill.importance_level
                
                if skill_level >= required_skill.minimum_proficiency:
                    # Skill meets requirements
                    match_score += importance
                elif skill_level > 0:
                    # Partial match
                    match_score += importance * (skill_level / required_skill.minimum_proficiency)
                
                total_importance += importance
            
            final_score = match_score / total_importance if total_importance > 0 else 0.0
            matches.append((prof_name, final_score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def find_similar_professionals(target_professional: ProfessionInstance, 
                                 library: ProfessionLibrary) -> List[Tuple[ProfessionInstance, float]]:
        """Find professionals with similar skill profiles"""
        target_skills = {name: skill.current_level for name, skill in target_professional.skill_levels.items()}
        similar_pros = []
        
        for instance in library.instances.values():
            if instance.instance_id == target_professional.instance_id:
                continue
            
            # Calculate skill similarity
            other_skills = {name: skill.current_level for name, skill in instance.skill_levels.items()}
            all_skills = set(target_skills.keys()) | set(other_skills.keys())
            
            if not all_skills:
                continue
            
            similarity_score = 0.0
            for skill in all_skills:
                target_level = target_skills.get(skill, 0)
                other_level = other_skills.get(skill, 0)
                
                # Calculate similarity (1.0 = identical, 0.0 = completely different)
                max_diff = 10  # Maximum possible difference
                actual_diff = abs(target_level - other_level)
                skill_similarity = 1.0 - (actual_diff / max_diff)
                similarity_score += skill_similarity
            
            final_similarity = similarity_score / len(all_skills)
            similar_pros.append((instance, final_similarity))
        
        return sorted(similar_pros, key=lambda x: x[1], reverse=True)

def demo_profession_integration():
    """Demonstrate profession system integration with other systems"""
    
    library = create_example_professions()
    
    # Create a professional
    dr_wilson = library.create_professional(
        "medical_doctor",
        "Dr. Emily Wilson",
        years_experience=5.0
    )
    
    # Add skills
    dr_wilson.add_skill("medical_diagnosis", 8, 5.0)
    dr_wilson.add_skill("patient_communication", 7, 5.0)
    dr_wilson.add_skill("medical_procedures", 9, 5.0)
    dr_wilson.add_skill("empathy", 8, 5.0)
    dr_wilson.add_skill("stress_management", 6, 5.0)  # Area for improvement
    dr_wilson.add_skill("medical_ethics", 9, 5.0)
    
    print("\n=== Profession Integration Demo ===\n")
    
    # Generate daily tasks
    profession = library.professions["medical_doctor"]
    daily_tasks = ProfessionalWorkflow.generate_daily_tasks(dr_wilson, profession)
    
    print(f"📋 Daily Tasks for {dr_wilson.person_name}:")
    for task in daily_tasks[:5]:  # Show top 5 tasks
        readiness = ProfessionalWorkflow.assess_task_readiness(dr_wilson, task)
        print(f"  - {task['task']}")
        print(f"    Difficulty: {task['difficulty']}/10, Time: {task['estimated_time']}h")
        print(f"    Readiness: {readiness:.1%}, Skills: {', '.join(task['skills_used'][:2])}")
        print()
    
    # Test profession matching
    candidate_skills = {
        "investigation": 8,
        "interviewing": 7,
        "evidence_analysis": 6,
        "criminal_law": 7,
        "report_writing": 5,
        "firearms_proficiency": 8
    }
    
    matches = ProfessionMatchmaking.find_best_profession_match(candidate_skills, library)
    print("🎯 Best profession matches for candidate skills:")
    for profession_name, score in matches:
        print(f"  - {profession_name}: {score:.1%} match")
    print()
    
    # Create another doctor to find similar professionals
    dr_brown = library.create_professional("medical_doctor", "Dr. James Brown", years_experience=6.0)
    dr_brown.add_skill("medical_diagnosis", 9, 6.0)
    dr_brown.add_skill("patient_communication", 8, 6.0)
    dr_brown.add_skill("medical_procedures", 8, 6.0)
    dr_brown.add_skill("empathy", 7, 6.0)
    dr_brown.add_skill("stress_management", 9, 6.0)  # Stronger than Dr. Wilson
    dr_brown.add_skill("medical_ethics", 8, 6.0)
    
    similar_pros = ProfessionMatchmaking.find_similar_professionals(dr_wilson, library)
    print(f"👥 Professionals similar to {dr_wilson.person_name}:")
    for professional, similarity in similar_pros[:3]:
        print(f"  - {professional.person_name} ({professional.profession_type}): {similarity:.1%} similar")

if __name__ == "__main__":
    demo_profession_system()
    demo_career_development()
    demo_profession_integration()


