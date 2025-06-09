from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import json
from enum import Enum

# Import our previous object definition system
from object_attribute_system import ObjectDefinition, ObjectLibrary, AttributeType, Attribute

class InstanceState(Enum):
    NORMAL = "normal"
    NEEDS_ATTENTION = "needs_attention"
    SICK = "sick"
    HAPPY = "happy"
    STRESSED = "stressed"
    SLEEPING = "sleeping"
    ACTIVE = "active"

@dataclass
class InstanceAttribute:
    """A specific attribute value for this instance - like instance variables"""
    name: str
    current_value: Any
    last_updated: datetime = field(default_factory=datetime.now)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decay_rate: Optional[float] = None  # How much it decreases per hour
    
    def update_value(self, new_value: Any):
        """Update the attribute value with timestamp"""
        self.current_value = new_value
        self.last_updated = datetime.now()
    
    def apply_decay(self, hours_passed: float):
        """Apply time-based decay to the attribute"""
        if self.decay_rate and isinstance(self.current_value, (int, float)):
            decay_amount = self.decay_rate * hours_passed
            new_value = self.current_value - decay_amount
            
            # Apply bounds
            if self.min_value is not None:
                new_value = max(new_value, self.min_value)
            if self.max_value is not None:
                new_value = min(new_value, self.max_value)
            
            self.update_value(new_value)

@dataclass
class ObjectInstance:
    """A specific instance of an object - like creating an object from a class"""
    
    # Core identity
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    object_type: str = ""  # References ObjectDefinition
    name: str = ""
    
    # Instance-specific attributes
    instance_attributes: Dict[str, InstanceAttribute] = field(default_factory=dict)
    
    # State tracking
    current_state: InstanceState = InstanceState.NORMAL
    creation_time: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    # Location and relationships
    location: str = "unknown"
    owner: Optional[str] = None
    relationships: Dict[str, str] = field(default_factory=dict)  # other_instance_id: relationship_type
    
    # History tracking
    state_history: List[Dict] = field(default_factory=list)
    interaction_log: List[Dict] = field(default_factory=list)
    
    def add_attribute(self, name: str, initial_value: Any, 
                     min_val: float = None, max_val: float = None, 
                     decay_rate: float = None):
        """Add an instance-specific attribute"""
        self.instance_attributes[name] = InstanceAttribute(
            name=name,
            current_value=initial_value,
            min_value=min_val,
            max_value=max_val,
            decay_rate=decay_rate
        )
    
    def get_attribute_value(self, name: str) -> Any:
        """Get current value of an attribute"""
        if name in self.instance_attributes:
            return self.instance_attributes[name].current_value
        return None
    
    def set_attribute_value(self, name: str, value: Any):
        """Update an attribute value"""
        if name in self.instance_attributes:
            self.instance_attributes[name].update_value(value)
        else:
            self.add_attribute(name, value)
    
    def change_state(self, new_state: InstanceState, reason: str = ""):
        """Change the object's state and log it"""
        old_state = self.current_state
        self.current_state = new_state
        
        # Log state change
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason
        })
    
    def record_interaction(self, interaction_type: str, details: Dict = None):
        """Record an interaction with this object"""
        self.last_interaction = datetime.now()
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "details": details or {}
        }
        
        self.interaction_log.append(interaction)
        
        # Keep only last 100 interactions
        if len(self.interaction_log) > 100:
            self.interaction_log = self.interaction_log[-100:]
    
    def update_over_time(self):
        """Apply time-based changes to the instance"""
        now = datetime.now()
        hours_since_update = (now - self.last_interaction).total_seconds() / 3600
        
        # Apply decay to attributes
        for attr in self.instance_attributes.values():
            attr.apply_decay(hours_since_update)
        
        # Update state based on attribute values
        self._evaluate_state_changes()
    
    def _evaluate_state_changes(self):
        """Automatically change state based on attribute values"""
        # Example: if hunger gets too high, become stressed
        hunger = self.get_attribute_value("hunger")
        energy = self.get_attribute_value("energy")
        happiness = self.get_attribute_value("happiness")
        
        if hunger and hunger > 8:
            self.change_state(InstanceState.NEEDS_ATTENTION, "Very hungry")
        elif energy and energy < 2:
            self.change_state(InstanceState.SLEEPING, "Low energy")
        elif happiness and happiness < 3:
            self.change_state(InstanceState.STRESSED, "Low happiness")
        elif happiness and happiness > 8 and energy and energy > 7:
            self.change_state(InstanceState.HAPPY, "High happiness and energy")
        else:
            self.change_state(InstanceState.NORMAL, "Balanced attributes")
    
    def get_age(self) -> timedelta:
        """Get the age of this instance"""
        return datetime.now() - self.creation_time
    
    def get_age_in_days(self) -> int:
        """Get age in days"""
        return self.get_age().days
    
    def get_status_summary(self) -> Dict:
        """Get a summary of the current instance status"""
        return {
            "name": self.name,
            "type": self.object_type,
            "age_days": self.get_age_in_days(),
            "current_state": self.current_state.value,
            "location": self.location,
            "key_attributes": {
                name: attr.current_value 
                for name, attr in self.instance_attributes.items()
                if isinstance(attr.current_value, (int, float))
            },
            "last_interaction": self.last_interaction.strftime("%Y-%m-%d %H:%M"),
            "needs_attention": self.current_state == InstanceState.NEEDS_ATTENTION
        }
    
    def to_dict(self) -> Dict:
        """Convert instance to dictionary for storage"""
        return {
            "instance_id": self.instance_id,
            "object_type": self.object_type,
            "name": self.name,
            "current_state": self.current_state.value,
            "creation_time": self.creation_time.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "location": self.location,
            "owner": self.owner,
            "relationships": self.relationships,
            "instance_attributes": {
                name: {
                    "current_value": attr.current_value,
                    "last_updated": attr.last_updated.isoformat(),
                    "min_value": attr.min_value,
                    "max_value": attr.max_value,
                    "decay_rate": attr.decay_rate
                }
                for name, attr in self.instance_attributes.items()
            },
            "state_history": self.state_history[-10:],  # Last 10 state changes
            "interaction_log": self.interaction_log[-20:]  # Last 20 interactions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObjectInstance':
        """Create instance from dictionary"""
        instance = cls(
            instance_id=data["instance_id"],
            object_type=data["object_type"],
            name=data["name"],
            current_state=InstanceState(data["current_state"]),
            creation_time=datetime.fromisoformat(data["creation_time"]),
            last_interaction=datetime.fromisoformat(data["last_interaction"]),
            location=data["location"],
            owner=data.get("owner"),
            relationships=data.get("relationships", {}),
            state_history=data.get("state_history", []),
            interaction_log=data.get("interaction_log", [])
        )
        
        # Restore instance attributes
        for name, attr_data in data["instance_attributes"].items():
            instance.instance_attributes[name] = InstanceAttribute(
                name=name,
                current_value=attr_data["current_value"],
                last_updated=datetime.fromisoformat(attr_data["last_updated"]),
                min_value=attr_data.get("min_value"),
                max_value=attr_data.get("max_value"),
                decay_rate=attr_data.get("decay_rate")
            )
        
        return instance

class InstanceManager:
    """Manages multiple object instances - like a database of living objects"""
    
    def __init__(self, object_library: ObjectLibrary):
        self.object_library = object_library
        self.instances: Dict[str, ObjectInstance] = {}
        self.instances_by_type: Dict[str, List[str]] = {}
        self.instances_by_owner: Dict[str, List[str]] = {}
    
    def create_instance(self, object_type: str, name: str, owner: str = None, 
                       location: str = "home", **initial_attributes) -> ObjectInstance:
        """Create a new instance of an object type"""
        
        # Verify object type exists
        obj_def = self.object_library.get_object(object_type)
        if not obj_def:
            raise ValueError(f"Object type '{object_type}' not found in library")
        
        # Create instance
        instance = ObjectInstance(
            object_type=object_type,
            name=name,
            owner=owner,
            location=location
        )
        
        # Initialize with default attributes based on object definition
        self._initialize_instance_attributes(instance, obj_def, initial_attributes)
        
        # Register instance
        self.instances[instance.instance_id] = instance
        
        # Update indexes
        if object_type not in self.instances_by_type:
            self.instances_by_type[object_type] = []
        self.instances_by_type[object_type].append(instance.instance_id)
        
        if owner:
            if owner not in self.instances_by_owner:
                self.instances_by_owner[owner] = []
            self.instances_by_owner[owner].append(instance.instance_id)
        
        return instance
    
    def _initialize_instance_attributes(self, instance: ObjectInstance, 
                                       obj_def: ObjectDefinition, 
                                       initial_attributes: Dict):
        """Initialize instance with appropriate attributes based on object definition"""
        
        # Add common living attributes for animals
        if obj_def.category == "animal":
            instance.add_attribute("hunger", initial_attributes.get("hunger", 5.0), 
                                 min_val=0.0, max_val=10.0, decay_rate=0.5)  # Gets hungry over time
            instance.add_attribute("energy", initial_attributes.get("energy", 8.0), 
                                 min_val=0.0, max_val=10.0, decay_rate=0.3)
            instance.add_attribute("happiness", initial_attributes.get("happiness", 7.0), 
                                 min_val=0.0, max_val=10.0, decay_rate=0.1)
            instance.add_attribute("health", initial_attributes.get("health", 10.0), 
                                 min_val=0.0, max_val=10.0)
            instance.add_attribute("age_months", initial_attributes.get("age_months", 12), 
                                 min_val=0.0)
        
        # Add specific attributes based on object type
        if obj_def.name == "cat":
            instance.add_attribute("cleanliness", initial_attributes.get("cleanliness", 9.0), 
                                 min_val=0.0, max_val=10.0, decay_rate=0.2)
            instance.add_attribute("independence", initial_attributes.get("independence", 8.0), 
                                 min_val=0.0, max_val=10.0)
        elif obj_def.name == "dog":
            instance.add_attribute("loyalty", initial_attributes.get("loyalty", 9.0), 
                                 min_val=0.0, max_val=10.0)
            instance.add_attribute("exercise_need", initial_attributes.get("exercise_need", 6.0), 
                                 min_val=0.0, max_val=10.0, decay_rate=-0.5)  # Increases over time
        
        # Add any additional custom attributes
        for attr_name, value in initial_attributes.items():
            if attr_name not in instance.instance_attributes:
                instance.add_attribute(attr_name, value)
    
    def get_instance(self, instance_id: str) -> Optional[ObjectInstance]:
        """Get instance by ID"""
        return self.instances.get(instance_id)
    
    def get_instances_by_type(self, object_type: str) -> List[ObjectInstance]:
        """Get all instances of a specific type"""
        instance_ids = self.instances_by_type.get(object_type, [])
        return [self.instances[iid] for iid in instance_ids if iid in self.instances]
    
    def get_instances_by_owner(self, owner: str) -> List[ObjectInstance]:
        """Get all instances owned by someone"""
        instance_ids = self.instances_by_owner.get(owner, [])
        return [self.instances[iid] for iid in instance_ids if iid in self.instances]
    
    def find_instances_by_name(self, name: str) -> List[ObjectInstance]:
        """Find instances by name (can be partial match)"""
        return [
            instance for instance in self.instances.values()
            if name.lower() in instance.name.lower()
        ]
    
    def update_all_instances(self):
        """Update all instances for time passage"""
        for instance in self.instances.values():
            instance.update_over_time()
    
    def get_instances_needing_attention(self) -> List[ObjectInstance]:
        """Get instances that need attention"""
        return [
            instance for instance in self.instances.values()
            if instance.current_state == InstanceState.NEEDS_ATTENTION
        ]
    
    def interact_with_instance(self, instance_id: str, interaction_type: str, 
                              effect_on_attributes: Dict[str, float] = None):
        """Perform an interaction with an instance"""
        instance = self.get_instance(instance_id)
        if not instance:
            return False
        
        # Record the interaction
        instance.record_interaction(interaction_type, effect_on_attributes)
        
        # Apply effects to attributes
        if effect_on_attributes:
            for attr_name, change in effect_on_attributes.items():
                current_val = instance.get_attribute_value(attr_name)
                if current_val is not None and isinstance(current_val, (int, float)):
                    new_val = current_val + change
                    # Apply bounds if they exist
                    attr = instance.instance_attributes[attr_name]
                    if attr.min_value is not None:
                        new_val = max(new_val, attr.min_value)
                    if attr.max_value is not None:
                        new_val = min(new_val, attr.max_value)
                    instance.set_attribute_value(attr_name, new_val)
        
        # Re-evaluate state
        instance._evaluate_state_changes()
        return True
    
    def save_instances(self, filename: str):
        """Save all instances to file"""
        data = {
            "instances": {iid: instance.to_dict() for iid, instance in self.instances.items()},
            "metadata": {
                "total_instances": len(self.instances),
                "types": list(self.instances_by_type.keys()),
                "save_time": datetime.now().isoformat()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_instances(cls, filename: str, object_library: ObjectLibrary) -> 'InstanceManager':
        """Load instances from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        manager = cls(object_library)
        
        for instance_data in data["instances"].values():
            instance = ObjectInstance.from_dict(instance_data)
            manager.instances[instance.instance_id] = instance
            
            # Rebuild indexes
            obj_type = instance.object_type
            if obj_type not in manager.instances_by_type:
                manager.instances_by_type[obj_type] = []
            manager.instances_by_type[obj_type].append(instance.instance_id)
            
            if instance.owner:
                if instance.owner not in manager.instances_by_owner:
                    manager.instances_by_owner[instance.owner] = []
                manager.instances_by_owner[instance.owner].append(instance.instance_id)
        
        return manager

# Demo and examples
def demo_object_instances():
    """Demonstrate the object instance system"""
    
    # Import our object library from previous example
    from object_attribute_system import create_example_objects
    
    library = create_example_objects()
    manager = InstanceManager(library)
    
    print("=== Object Instance System Demo ===\n")
    
    # Create some cat instances
    fluffy = manager.create_instance(
        object_type="cat",
        name="Fluffy",
        owner="Alice",
        location="living_room",
        hunger=3.0,
        energy=9.0,
        happiness=8.0,
        age_months=24
    )
    
    mittens = manager.create_instance(
        object_type="cat", 
        name="Mittens",
        owner="Alice",
        location="bedroom",
        hunger=7.0,
        energy=5.0,
        happiness=6.0,
        age_months=6
    )
    
    # Create a dog instance
    buddy = manager.create_instance(
        object_type="dog",
        name="Buddy", 
        owner="Bob",
        location="backyard",
        hunger=4.0,
        energy=8.0,
        happiness=9.0,
        age_months=36
    )
    
    print("🏠 Created instances:")
    for instance in [fluffy, mittens, buddy]:
        status = instance.get_status_summary()
        print(f"  - {status['name']} ({status['type']}): {status['current_state']}")
        print(f"    Age: {status['age_days']} days, Location: {status['location']}")
        print(f"    Key stats: {status['key_attributes']}")
    print()
    
    # Show instances by owner
    print("👤 Alice's pets:")
    alice_pets = manager.get_instances_by_owner("Alice")
    for pet in alice_pets:
        print(f"  - {pet.name} ({pet.object_type}) in {pet.location}")
    print()
    
    # Simulate feeding Mittens
    print("🍽️ Feeding Mittens...")
    manager.interact_with_instance(
        mittens.instance_id,
        "feeding",
        {"hunger": -5.0, "happiness": +2.0}
    )
    
    # Show updated status
    status = mittens.get_status_summary()
    print(f"   After feeding: hunger={status['key_attributes']['hunger']}, happiness={status['key_attributes']['happiness']}")
    print(f"   State: {status['current_state']}")
    print()
    
    # Simulate time passing
    print("⏰ Simulating 3 hours passing...")
    # Manually apply some time decay for demo
    for instance in manager.instances.values():
        for attr in instance.instance_attributes.values():
            attr.apply_decay(3.0)  # 3 hours
        instance._evaluate_state_changes()
    
    # Check who needs attention
    attention_needed = manager.get_instances_needing_attention()
    if attention_needed:
        print("⚠️ Instances needing attention:")
        for instance in attention_needed:
            status = instance.get_status_summary()
            print(f"  - {status['name']}: {status['current_state']} (hunger: {status['key_attributes'].get('hunger', 'N/A')})")
    else:
        print("✅ All instances are doing well!")
    print()
    
    # Show interaction history
    print(f"📜 {mittens.name}'s recent interactions:")
    for interaction in mittens.interaction_log[-3:]:
        print(f"  - {interaction['timestamp'][:19]}: {interaction['type']}")
    print()
    
    # Show all cats
    print("🐱 All cats in the system:")
    cats = manager.get_instances_by_type("cat")
    for cat in cats:
        age_days = cat.get_age_in_days()
        cleanliness = cat.get_attribute_value("cleanliness")
        print(f"  - {cat.name}: {age_days} days old, cleanliness: {cleanliness:.1f}")

if __name__ == "__main__":
    demo_object_instances()

# Advanced instance features
class InstanceRelationships:
    """Manages relationships between object instances"""
    
    @staticmethod
    def create_relationship(instance1: ObjectInstance, instance2: ObjectInstance, 
                          relationship_type: str):
        """Create a bidirectional relationship between instances"""
        instance1.relationships[instance2.instance_id] = relationship_type
        
        # Create reverse relationship
        reverse_relationships = {
            "sibling": "sibling",
            "parent": "child", 
            "child": "parent",
            "friend": "friend",
            "enemy": "enemy",
            "mate": "mate"
        }
        
        reverse_type = reverse_relationships.get(relationship_type, relationship_type)
        instance2.relationships[instance1.instance_id] = reverse_type
    
    @staticmethod
    def get_related_instances(instance: ObjectInstance, manager: InstanceManager, 
                            relationship_type: str = None) -> List[ObjectInstance]:
        """Get instances related to this one"""
        related = []
        
        for other_id, rel_type in instance.relationships.items():
            if relationship_type is None or rel_type == relationship_type:
                other_instance = manager.get_instance(other_id)
                if other_instance:
                    related.append((other_instance, rel_type))
        
        return related

def demo_relationships():
    """Demonstrate instance relationships"""
    
    from object_attribute_system import create_example_objects
    
    library = create_example_objects()
    manager = InstanceManager(library)
    
    # Create a cat family
    mama_cat = manager.create_instance("cat", "Mama Cat", "Alice", age_months=48)
    kitten1 = manager.create_instance("cat", "Whiskers", "Alice", age_months=3)
    kitten2 = manager.create_instance("cat", "Shadow", "Alice", age_months=3)
    
    # Create relationships
    InstanceRelationships.create_relationship(mama_cat, kitten1, "parent")
    InstanceRelationships.create_relationship(mama_cat, kitten2, "parent") 
    InstanceRelationships.create_relationship(kitten1, kitten2, "sibling")
    
    print("\n=== Instance Relationships Demo ===\n")
    
    # Show family tree
    print("👨‍👩‍👧‍👦 Cat family relationships:")
    for cat in [mama_cat, kitten1, kitten2]:
        related = InstanceRelationships.get_related_instances(cat, manager)
        print(f"  {cat.name}:")
        for other_cat, relationship in related:
            print(f"    - {relationship} of {other_cat.name}")
    
    print(f"\n🐱 Mama Cat's children:")
    children = InstanceRelationships.get_related_instances(mama_cat, manager, "child")
    for child, rel_type in children:
        print(f"  - {child.name} ({child.get_age_in_days()} days old)")

if __name__ == "__main__":
    demo_object_instances()
    demo_relationships()

