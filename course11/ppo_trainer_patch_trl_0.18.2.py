
import torch.nn as nn
import copy
from trl import PPOTrainer

# Dummy reward model that always returns 1
class DummyRewardModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        return torch.ones(batch_size, 1)

reward_model = DummyRewardModel()

# Clone of the main model to use as the value model
value_model = copy.deepcopy(model)

# Minimal processing class required by trl==0.18.2
class DummyProcessing:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, example):
        prompt = example["query"]
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        return encoded

# Initialize PPOTrainer compatible with trl==0.18.2
ppo_trainer = PPOTrainer(
    model=model,
    value_model=value_model,
    reward_model=reward_model,
    train_dataset=dataset,
    processing_class=DummyProcessing(tokenizer)
)

# Attach tokenizer manually
ppo_trainer.tokenizer = tokenizer

# Define training device
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"
print("PPOTrainer is ready on device:", device)
