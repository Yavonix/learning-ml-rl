## Introduction

A model is a program with a given set of parameters.

A model is trained by optimising an **objective function** (conventionally called a **loss function** where lower is better) on examples of known **features** to predict a known (**supervised**) or unknown (**unsupervised**) **label**.

During training, **loss** is a function of the model's **parameters**. The training dataset is a constant.

Available data is split into:
- **training dataset**
- **test dataset** (for eval)

High performance on training but not test indicated **overfitting**.

Supervised problems:
- **Regression problems**: labels are arbitrary numerical values.
  - Loss function: **mean squared error**
- **Classification problems**: labels are categories.
  - Loss function: **cross-entropy**
  - Classification types:
    - Binary: 1 of 2 output labels
    - Multiclass: 1 of >2 output labels
    - Hierarchical: labels is hierarchical (tree or directed acyclic graph)
      - Global approach: one monolithic classifier over all classes in the hierarchy 
      - Local approach: one model per:
        - Level (one model for each depth in the tree)
        - Parent (chooses exactly one of its children (or "none"))
        - Node (one binary (or multi‐class vs. rest) classifier for each node)
    - Multilabel: n of k labels
    - Note: Class implies a mutually exclusive category while label is like a tag.
- **Sequence problems**
  - **Sequence-to-sequence**: both inputs and outputs are variable-length sequences.

Unsupervised problems:
- **Clustering**: derives pseudo-labels from the data itself to learn feature representations that partition examples into coherent groups without any manual annotation
- **PCA**: unsupervised dimensionality reduction by projecting data onto orthogonal components that capture maximum variance
- **Vector embedding**: mapping symbols or entities into a continuous Euclidean space so that semantic relations become (approximately) linear—e.g. Word2Vec’s "Rome"–"Italy"+"France"≈"Paris."
- **Deep generative models**: VAEs, GANs, Diffusion models... idk.

There's also self-supervised as well. Looks like automatically generating "labels" from the data itself.

Distribution shift: training and test data are different. Because the environment is changing over time...

**Reinforcement learning 🙌**: agent->action->environment + reward loop. A **policy** is the learned mapping from states (or observations) to actions. Problems:
- Must figure out what actions lead to the reward (ie chess)
- Must deal with partial observability of the env
- Balancing exploitation of the current best policy with exploration of new actions to discover potentially better strategies and long-term rewards (Exploration–exploitation dilemma)