I-JEPA
- Predict representations of target blocks from single context block in same image
  - This is what makes it "predictive" as contrasted to "Joint-Embedding"
- When training:
  - Target blocks to be sufficiently large scale
    - "Semantic" target block means the target is big enough that its representation corresponds to meaningful image content, not just pixels/textures
  - Sufficiently informative context block
    - May be "spatially distributed" so model sees information from multiple positions


JEPA is generative-like but in the representation-space


Mask  tokens  are parameterized by a shared learnable vector with an added positional embedding



Might needs to reinforce in anki that GPT
- -log p (because one-hot)
- Average
- Negative

- And like in a CNN classification task we average over B
- But in a GPT training task we average over B*L
- And we usually flatten the two arrays to do this