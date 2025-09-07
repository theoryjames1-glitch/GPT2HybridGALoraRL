# GPT2HybridGALoraRL

Awesome 🚀 Let’s design a **hybrid GA + RL loop** for GPT-2 with **LoRA adapters**. The idea is:

* **Genetic Algorithm (GA)** explores big weight variations in LoRA space.
* **Reinforcement Learning (RL / PPO)** fine-tunes the *best individuals* locally.
* This balances **exploration** (GA) and **exploitation** (RL).

---

# 🔹 1. Setup: GPT-2 + LoRA

We’ll use PEFT to attach LoRA to GPT-2.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

base_model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
```

---

# 🔹 2. Genetic algorithm helpers

We evolve only the **LoRA adapter weights**.

```python
import copy, random

def get_lora_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

def set_lora_state(model, state):
    with torch.no_grad():
        for k, v in state.items():
            model.state_dict()[k].copy_(v)

def mutate_lora(state, scale=0.02):
    return {k: v + torch.randn_like(v) * scale for k, v in state.items()}

def crossover_lora(s1, s2):
    return {k: torch.where(torch.rand_like(s1[k]) < 0.5, s1[k], s2[k]) for k in s1}
```

---

# 🔹 3. Reward function

This can be **any RL reward** (BLEU, cosine similarity, human feedback, etc.).
Here we’ll use a simple string similarity.

```python
import difflib

def reward_fn(model, tokenizer, prompt, target):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    sim = difflib.SequenceMatcher(None, text, target).ratio()
    return sim  # [0..1]
```

---

# 🔹 4. PPO fine-tuner

We wrap the model in TRL’s `AutoModelForCausalLMWithValueHead` so PPO can run on top.

```python
ppo_config = PPOConfig(
    model_name=base_model_id,
    learning_rate=1e-6,
    batch_size=1,
    mini_batch_size=1,
)

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_id)
ppo_trainer = PPOTrainer(config=ppo_config, model=ppo_model, tokenizer=tokenizer)
```

---

# 🔹 5. Hybrid GA + RL loop

* GA proposes population of LoRA states.
* Evaluate fitness.
* Best state gets a **PPO refinement step**.
* Replace worst with mutated/refined versions.

```python
# Initial population
pop_size = 4
population = [get_lora_state(model)]
for _ in range(pop_size-1):
    population.append(mutate_lora(population[0]))

prompt = "Translate 'bonjour' to English:"
target = "hello"

for gen in range(5):
    scored = []
    for state in population:
        set_lora_state(model, state)
        fitness = reward_fn(model, tokenizer, prompt, target)
        scored.append((fitness, state))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_state = scored[0]
    print(f"Gen {gen} best fitness={best_score:.4f}")
    
    # ✅ PPO fine-tune best candidate
    set_lora_state(model, best_state)
    query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    response_tensors = model.generate(query_tensors, max_new_tokens=10)
    reward = torch.tensor(best_score).to(model.device)
    ppo_trainer.step([query_tensors[0]], [response_tensors[0]], [reward])
    
    # Update population (elitism + crossover + mutation)
    new_pop = [best_state]
    while len(new_pop) < pop_size:
        child = crossover_lora(scored[0][1], random.choice(scored[1:])[1])
        child = mutate_lora(child, scale=0.02)
        new_pop.append(child)
    population = new_pop
```

---

# 🔹 6. Extensions

* Replace `reward_fn` with **your own reward model** or **human feedback**.
* Periodically `model.save_pretrained("./checkpoints/genX")`.
* Increase population size for more exploration.
* Use `merge_and_unload()` at the end to bake LoRA into the base model.

---

✅ **Summary**

* GA = broad exploration of LoRA states.
* PPO = local exploitation (fine-tune best candidate).
* Together: GA finds new promising regions, PPO polishes them.

---

👉 Do you want me to turn this into a **standalone trainer script** (with argparse, logging, saving checkpoints, and resume capability), so you could run it like `python hybrid_ga_rl.py --gens 20 --pop 8`?
