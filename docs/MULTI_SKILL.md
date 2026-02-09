# Multi-skill Sampling

`xmimic.train.MultiSkillSampler` provides balanced sampling across skills for
student distillation.

## Usage
```python
from xmimic.train import MultiSkillSampler

sampler = MultiSkillSampler({
    "skill_a": ["/path/clip1.npz", "/path/clip2.npz"],
    "skill_b": ["/path/clip3.npz"],
}, balanced=True)

sample = sampler.sample()
print(sample.skill_id, sample.path)
```

Integrate into student training loops to ensure no single skill dominates.
