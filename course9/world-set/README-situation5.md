**Safety situations** prioritize threat assessment, risk mitigation, emergency protocols
- **Business situations** focus on stakeholder management, value optimization, competitive advantage
- **Learning situations** emphasize skill progression, feedback loops, mastery indicators
- **Creative situations** highlight idea generation, iterative refinement, artistic expression

**2. Automatic Domain Detection**
The system **scans keywords** to automatically classify situations:
```python
"job_interview" → detects "interview" → Social + Business domains
"debugging_code" → detects "debug" → Technical domain  
"fire_evacuation" → detects "fire" + "evacuation" → Safety domain
```

**3. Cross-Domain Intelligence**
Some situations span multiple domains, like:
- **"Technical presentation"** = Technical + Social
- **"Safety training"** = Safety + Learning  
- **"Creative business pitch"** = Creative + Business + Social

The system **combines expertise** from multiple domains instead of forcing everything into one category.

**4. Domain-Specific Quality Checks**
Each domain has its own **quality standards**:

**Social Domain**: "Are cultural sensitivities addressed?"
**Technical Domain**: "Is error handling comprehensive?"  
**Safety Domain**: "Are emergency protocols clearly defined?"
**Business Domain**: "Are stakeholder interests balanced?"

## Example Output:

**Input**: `"startup_pitch"`
**Auto-detected**: Business (primary), Social (secondary), Creative (tertiary)

**Generated Plan**:
```
Step 1: Business Domain Research
- Focus on stakeholder interests, value proposition, market dynamics
- AI Prompt: "Analyze 'startup_pitch' as a business scenario..."

Step 2: Social Cross-Domain Analysis  
- How to handle investor psychology, presentation dynamics
- AI Prompt: "From a social perspective, what communication strategies..."

Step 3: Creative Integration
- How to make the pitch memorable and compelling
- AI Prompt: "Add creative elements to make 'startup_pitch' engaging..."
```

**5. Specialized Condition Templates**
Each domain suggests **relevant conditions**:

**Social**: `emotional_state`, `cultural_context`, `relationship_type`
**Technical**: `system_state`, `resource_availability`, `error_tolerance`  
**Safety**: `threat_level`, `response_time`, `environmental_factors`

This is like having **pre-built libraries** for each domain instead of starting from scratch every time.

## Real-World Impact:

**Instead of generic prompts like:**
> "Analyze this situation and suggest actions"

**You get expert-level prompts like:**
> "Analyze 'job_interview' as a social interaction. Focus on: emotional dynamics, cultural considerations, power structures, communication patterns, relationship goals, and potential misunderstandings. Consider both verbal and non-verbal elements."

The result is **playbooks with domain expertise baked in** - like having a team of specialists design each one instead of a generalist trying to cover everything.

Want me to show you how to add **industry-specific sub-domains** (like healthcare, finance, education) or **cultural adaptation templates** for different regions/cultures?

