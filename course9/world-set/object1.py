from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Optional
from enum import Enum
import json

class AttributeType(Enum):
    PHYSICAL = "physical"        # Size, color, material, etc.
    BEHAVIORAL = "behavioral"    # What it does, how it acts
    FUNCTIONAL = "functional"    # What it's used for, purpose
    RELATIONAL = "relational"    # How it relates to other objects
    STATE = "state"             # Current condition, status
    REQUIREMENT = "requirement"  # What it needs to function/survive

@dataclass
class Attribute:
    """Like a property or characteristic of an object"""
    name: str
    attribute_type: AttributeType
    value: Any = None
    intensity: float = 1.0  # How strong/prominent this attribute is (0-10)
    confidence: float = 1.0  # How certain we are about this attribute (0-1)
    context_dependent: bool = False  # Changes based on situation
    
    def __str__(self):
        return f"{self.name}({self.intensity:.1f})"

@dataclass 
class ObjectDefinition:
    """Like a class definition for real-world objects"""
    name: str
    category: str  # animal, tool, furniture, food, etc.
    attributes: List[Attribute] = field(default_factory=list)
    
    def add_attribute(self, name: str, attr_type: AttributeType, 
                     value: Any = None, intensity: float = 1.0, 
                     confidence: float = 1.0, context_dependent: bool = False):
        """Add an attribute to this object type"""
        attr = Attribute(name, attr_type, value, intensity, confidence, context_dependent)
        self.attributes.append(attr)
        return attr
    
    def get_attributes_by_type(self, attr_type: AttributeType) -> List[Attribute]:
        """Get all attributes of a specific type"""
        return [attr for attr in self.attributes if attr.attribute_type == attr_type]
    
    def has_attribute(self, name: str) -> bool:
        """Check if object has a specific attribute"""
        return any(attr.name == name for attr in self.attributes)
    
    def get_attribute(self, name: str) -> Optional[Attribute]:
        """Get a specific attribute by name"""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "category": self.category,
            "attributes": [
                {
                    "name": attr.name,
                    "type": attr.attribute_type.value,
                    "value": attr.value,
                    "intensity": attr.intensity,
                    "confidence": attr.confidence,
                    "context_dependent": attr.context_dependent
                }
                for attr in self.attributes
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObjectDefinition':
        """Create object from dictionary"""
        obj = cls(data["name"], data["category"])
        for attr_data in data["attributes"]:
            obj.add_attribute(
                attr_data["name"],
                AttributeType(attr_data["type"]),
                attr_data.get("value"),
                attr_data.get("intensity", 1.0),
                attr_data.get("confidence", 1.0),
                attr_data.get("context_dependent", False)
            )
        return obj

class ObjectLibrary:
    """Like a database of object definitions"""
    
    def __init__(self):
        self.objects: Dict[str, ObjectDefinition] = {}
        self.categories: Dict[str, List[str]] = {}
        
    def add_object(self, obj_def: ObjectDefinition):
        """Add an object definition to the library"""
        self.objects[obj_def.name] = obj_def
        
        # Update category index
        if obj_def.category not in self.categories:
            self.categories[obj_def.category] = []
        self.categories[obj_def.category].append(obj_def.name)
    
    def get_object(self, name: str) -> Optional[ObjectDefinition]:
        """Get object definition by name"""
        return self.objects.get(name)
    
    def get_objects_by_category(self, category: str) -> List[ObjectDefinition]:
        """Get all objects in a category"""
        object_names = self.categories.get(category, [])
        return [self.objects[name] for name in object_names]
    
    def find_objects_with_attribute(self, attribute_name: str) -> List[ObjectDefinition]:
        """Find all objects that have a specific attribute"""
        return [obj for obj in self.objects.values() if obj.has_attribute(attribute_name)]
    
    def compare_objects(self, obj1_name: str, obj2_name: str) -> Dict:
        """Compare two objects and find similarities/differences"""
        obj1 = self.get_object(obj1_name)
        obj2 = self.get_object(obj2_name)
        
        if not obj1 or not obj2:
            return {"error": "One or both objects not found"}
        
        obj1_attrs = {attr.name: attr for attr in obj1.attributes}
        obj2_attrs = {attr.name: attr for attr in obj2.attributes}
        
        shared_attrs = []
        obj1_unique = []
        obj2_unique = []
        
        all_attr_names = set(obj1_attrs.keys()) | set(obj2_attrs.keys())
        
        for attr_name in all_attr_names:
            if attr_name in obj1_attrs and attr_name in obj2_attrs:
                shared_attrs.append({
                    "name": attr_name,
                    "obj1_intensity": obj1_attrs[attr_name].intensity,
                    "obj2_intensity": obj2_attrs[attr_name].intensity,
                    "difference": abs(obj1_attrs[attr_name].intensity - obj2_attrs[attr_name].intensity)
                })
            elif attr_name in obj1_attrs:
                obj1_unique.append(obj1_attrs[attr_name].name)
            else:
                obj2_unique.append(obj2_attrs[attr_name].name)
        
        return {
            "shared_attributes": shared_attrs,
            f"{obj1_name}_unique": obj1_unique,
            f"{obj2_name}_unique": obj2_unique,
            "similarity_score": len(shared_attrs) / len(all_attr_names) if all_attr_names else 0
        }
    
    def save_to_file(self, filename: str):
        """Save object library to JSON file"""
        data = {
            "objects": {name: obj.to_dict() for name, obj in self.objects.items()},
            "categories": self.categories
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'ObjectLibrary':
        """Load object library from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        library = cls()
        for obj_data in data["objects"].values():
            obj = ObjectDefinition.from_dict(obj_data)
            library.add_object(obj)
        
        return library

# Example object definitions
def create_example_objects() -> ObjectLibrary:
    """Create some example objects to demonstrate the system"""
    
    library = ObjectLibrary()
    
    # Cat object
    cat = ObjectDefinition("cat", "animal")
    
    # Physical attributes
    cat.add_attribute("furry", AttributeType.PHYSICAL, True, intensity=9.0)
    cat.add_attribute("four_legs", AttributeType.PHYSICAL, True, intensity=10.0)
    cat.add_attribute("tail", AttributeType.PHYSICAL, True, intensity=8.0)
    cat.add_attribute("whiskers", AttributeType.PHYSICAL, True, intensity=7.0)
    cat.add_attribute("retractable_claws", AttributeType.PHYSICAL, True, intensity=8.0)
    
    # Behavioral attributes
    cat.add_attribute("meows", AttributeType.BEHAVIORAL, True, intensity=8.0)
    cat.add_attribute("purrs", AttributeType.BEHAVIORAL, True, intensity=9.0)
    cat.add_attribute("chases_mice", AttributeType.BEHAVIORAL, True, intensity=7.0)
    cat.add_attribute("sleeps_frequently", AttributeType.BEHAVIORAL, True, intensity=9.0)
    cat.add_attribute("grooms_self", AttributeType.BEHAVIORAL, True, intensity=8.0)
    cat.add_attribute("independent", AttributeType.BEHAVIORAL, True, intensity=8.0)
    
    # State attributes
    cat.add_attribute("living", AttributeType.STATE, True, intensity=10.0)
    cat.add_attribute("warm_blooded", AttributeType.STATE, True, intensity=10.0)
    
    # Requirements
    cat.add_attribute("needs_food", AttributeType.REQUIREMENT, "daily", intensity=10.0)
    cat.add_attribute("needs_water", AttributeType.REQUIREMENT, "daily", intensity=10.0)
    cat.add_attribute("needs_litter_box", AttributeType.REQUIREMENT, True, intensity=8.0)
    
    # Relational attributes
    cat.add_attribute("domesticated", AttributeType.RELATIONAL, True, intensity=9.0)
    cat.add_attribute("companion_animal", AttributeType.RELATIONAL, True, intensity=8.0)
    
    library.add_object(cat)
    
    # Dog object for comparison
    dog = ObjectDefinition("dog", "animal")
    
    # Physical attributes
    dog.add_attribute("furry", AttributeType.PHYSICAL, True, intensity=8.0)
    dog.add_attribute("four_legs", AttributeType.PHYSICAL, True, intensity=10.0)
    dog.add_attribute("tail", AttributeType.PHYSICAL, True, intensity=9.0)
    dog.add_attribute("floppy_ears", AttributeType.PHYSICAL, True, intensity=6.0, context_dependent=True)
    
    # Behavioral attributes  
    dog.add_attribute("barks", AttributeType.BEHAVIORAL, True, intensity=9.0)
    dog.add_attribute("wags_tail", AttributeType.BEHAVIORAL, True, intensity=9.0)
    dog.add_attribute("loyal", AttributeType.BEHAVIORAL, True, intensity=10.0)
    dog.add_attribute("fetches", AttributeType.BEHAVIORAL, True, intensity=7.0)
    dog.add_attribute("social", AttributeType.BEHAVIORAL, True, intensity=9.0)
    
    # State attributes
    dog.add_attribute("living", AttributeType.STATE, True, intensity=10.0)
    dog.add_attribute("warm_blooded", AttributeType.STATE, True, intensity=10.0)
    
    # Requirements
    dog.add_attribute("needs_food", AttributeType.REQUIREMENT, "daily", intensity=10.0)
    dog.add_attribute("needs_water", AttributeType.REQUIREMENT, "daily", intensity=10.0)
    dog.add_attribute("needs_exercise", AttributeType.REQUIREMENT, "daily", intensity=9.0)
    dog.add_attribute("needs_walks", AttributeType.REQUIREMENT, "daily", intensity=8.0)
    
    # Relational attributes
    dog.add_attribute("domesticated", AttributeType.RELATIONAL, True, intensity=10.0)
    dog.add_attribute("companion_animal", AttributeType.RELATIONAL, True, intensity=10.0)
    
    library.add_object(dog)
    
    # Hammer object (tool example)
    hammer = ObjectDefinition("hammer", "tool")
    
    # Physical attributes
    hammer.add_attribute("metal_head", AttributeType.PHYSICAL, True, intensity=10.0)
    hammer.add_attribute("wooden_handle", AttributeType.PHYSICAL, True, intensity=8.0)
    hammer.add_attribute("heavy", AttributeType.PHYSICAL, True, intensity=7.0)
    hammer.add_attribute("hard", AttributeType.PHYSICAL, True, intensity=9.0)
    
    # Functional attributes
    hammer.add_attribute("pounds_nails", AttributeType.FUNCTIONAL, True, intensity=10.0)
    hammer.add_attribute("demolition", AttributeType.FUNCTIONAL, True, intensity=6.0)
    hammer.add_attribute("construction_tool", AttributeType.FUNCTIONAL, True, intensity=9.0)
    
    # State attributes
    hammer.add_attribute("non_living", AttributeType.STATE, True, intensity=10.0)
    hammer.add_attribute("durable", AttributeType.STATE, True, intensity=9.0)
    
    # Relational attributes
    hammer.add_attribute("human_made", AttributeType.RELATIONAL, True, intensity=10.0)
    hammer.add_attribute("requires_user", AttributeType.RELATIONAL, True, intensity=10.0)
    
    library.add_object(hammer)
    
    return library

# Usage examples and testing
def demo_object_system():
    """Demonstrate the object attribute system"""
    
    # Create library with example objects
    library = create_example_objects()
    
    print("=== Object Attribute System Demo ===\n")
    
    # Show all objects
    print("📚 Objects in library:")
    for name in library.objects.keys():
        obj = library.get_object(name)
        print(f"  - {name} ({obj.category}): {len(obj.attributes)} attributes")
    print()
    
    # Show detailed cat attributes
    cat = library.get_object("cat")
    print("🐱 Cat attributes by type:")
    for attr_type in AttributeType:
        attrs = cat.get_attributes_by_type(attr_type)
        if attrs:
            print(f"  {attr_type.value.title()}:")
            for attr in attrs:
                context_note = " (context-dependent)" if attr.context_dependent else ""
                print(f"    - {attr.name}: intensity={attr.intensity}{context_note}")
    print()
    
    # Compare cat and dog
    print("🐱 vs 🐶 Cat vs Dog comparison:")
    comparison = library.compare_objects("cat", "dog")
    print(f"  Similarity score: {comparison['similarity_score']:.2f}")
    print(f"  Shared attributes: {len(comparison['shared_attributes'])}")
    print(f"  Cat unique: {comparison['cat_unique']}")
    print(f"  Dog unique: {comparison['dog_unique']}")
    print()
    
    # Find objects with specific attributes
    print("🔍 Objects that need food:")
    food_needers = library.find_objects_with_attribute("needs_food")
    for obj in food_needers:
        print(f"  - {obj.name}")
    print()
    
    print("🔍 Objects that are furry:")
    furry_objects = library.find_objects_with_attribute("furry")
    for obj in furry_objects:
        furry_attr = obj.get_attribute("furry")
        print(f"  - {obj.name} (intensity: {furry_attr.intensity})")
    print()
    
    # Show objects by category
    print("📁 Animals in library:")
    animals = library.get_objects_by_category("animal")
    for animal in animals:
        behavioral_count = len(animal.get_attributes_by_type(AttributeType.BEHAVIORAL))
        print(f"  - {animal.name}: {behavioral_count} behaviors")

if __name__ == "__main__":
    demo_object_system()

# Advanced features for AI reasoning
class ObjectReasoner:
    """Provides reasoning capabilities over object attributes"""
    
    def __init__(self, library: ObjectLibrary):
        self.library = library
    
    def infer_category(self, attributes: List[str]) -> List[Tuple[str, float]]:
        """Given a list of attributes, predict what category an object might be"""
        category_scores = {}
        
        for category, object_names in self.library.categories.items():
            total_score = 0
            total_objects = len(object_names)
            
            for obj_name in object_names:
                obj = self.library.get_object(obj_name)
                obj_attrs = {attr.name for attr in obj.attributes}
                
                # Calculate overlap score
                overlap = len(set(attributes) & obj_attrs)
                score = overlap / len(attributes) if attributes else 0
                total_score += score
            
            category_scores[category] = total_score / total_objects if total_objects else 0
        
        # Return sorted by score
        return sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    def suggest_missing_attributes(self, obj_name: str) -> List[str]:
        """Suggest attributes that similar objects typically have"""
        obj = self.library.get_object(obj_name)
        if not obj:
            return []
        
        # Find similar objects (same category)
        similar_objects = self.library.get_objects_by_category(obj.category)
        
        obj_attrs = {attr.name for attr in obj.attributes}
        suggested_attrs = set()
        
        for similar_obj in similar_objects:
            if similar_obj.name != obj_name:
                similar_attrs = {attr.name for attr in similar_obj.attributes}
                # Find attributes that similar objects have but this one doesn't
                suggested_attrs.update(similar_attrs - obj_attrs)
        
        return list(suggested_attrs)
    
    def find_analogies(self, obj1_name: str, obj2_name: str) -> Dict:
        """Find analogical relationships between objects"""
        obj1 = self.library.get_object(obj1_name)
        obj2 = self.library.get_object(obj2_name)
        
        if not obj1 or not obj2:
            return {}
        
        analogies = {
            "structural": [],  # Similar physical structure
            "functional": [],  # Similar purpose/function
            "behavioral": []   # Similar behaviors
        }
        
        obj1_attrs = {attr.name: attr for attr in obj1.attributes}
        obj2_attrs = {attr.name: attr for attr in obj2.attributes}
        
        for attr_name in set(obj1_attrs.keys()) & set(obj2_attrs.keys()):
            attr1 = obj1_attrs[attr_name]
            attr2 = obj2_attrs[attr_name]
            
            if attr1.attribute_type == AttributeType.PHYSICAL:
                analogies["structural"].append(attr_name)
            elif attr1.attribute_type == AttributeType.FUNCTIONAL:
                analogies["functional"].append(attr_name)
            elif attr1.attribute_type == AttributeType.BEHAVIORAL:
                analogies["behavioral"].append(attr_name)
        
        return analogies

# Test reasoning capabilities
def demo_reasoning():
    """Demonstrate AI reasoning over object attributes"""
    
    library = create_example_objects()
    reasoner = ObjectReasoner(library)
    
    print("\n=== Object Reasoning Demo ===\n")
    
    # Test category inference
    mystery_attributes = ["furry", "barks", "four_legs", "loyal"]
    predictions = reasoner.infer_category(mystery_attributes)
    print(f"🔮 Given attributes {mystery_attributes}:")
    print("   Predicted categories:")
    for category, score in predictions:
        print(f"   - {category}: {score:.2f}")
    print()
    
    # Test missing attribute suggestions
    suggestions = reasoner.suggest_missing_attributes("cat")
    print(f"💡 Suggested missing attributes for cat: {suggestions}")
    print()
    
    # Test analogies
    analogies = reasoner.find_analogies("cat", "dog")
    print("🔗 Cat-Dog analogies:")
    for analogy_type, attrs in analogies.items():
        if attrs:
            print(f"   {analogy_type.title()}: {attrs}")

if __name__ == "__main__":
    demo_object_system()
    demo_reasoning()

