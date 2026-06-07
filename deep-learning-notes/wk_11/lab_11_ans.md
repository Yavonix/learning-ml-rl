## Part 1 T/F

1. According to the Chinchilla (2022) scaling laws, researchers should scale up model parameters much faster than they scale up the dataset size.

F.

2. Rejection Fine-Tuning (RFT) involves sampling multiple responses, scoring them, keeping only the high-reward answers, and running an offline Supervised Fine-Tuning (SFT) job on them

T.

3. During the Reinforcement Learning from Human Feedback (RLHF) process, a Reward Model is trained using human comparison data, where a labeler ranks model outputs from best to worst

T.

4. One of the main reasons Rejection Fine-Tuning (RFT) is considered perfectly sufficient for all post-training is that it excels at learning from failed examples.

F.

## Part 2 MCQ

1. What was a key difference in findings between Kaplan's Scaling Laws (2020) and the Chinchilla scaling laws (2022)?

- A) Kaplan stated that data quality doesn't matter, while Chinchilla argued that only
Wikipedia should be used.
- **B) Kaplan suggested test loss is more sensitive to parameter count, whereas
Chinchilla disputed this and suggested scaling data and parameters equally.**
- C) Kaplan proved that RLHF is necessary, while Chinchilla proved that Supervised
Fine-Tuning is sufficient.
- D) Kaplan found that larger models require more data to reach the same
performance, while Chinchilla found they require less


2. Which of the following is NOT one of the three primary steps in the InstructGPT
Reinforcement Learning from Human Feedback (RLHF) pipeline?

- A) Collecting demonstration data to train a supervised policy (SFT).
- B) Collecting comparison data where human labelers rank outputs to train a Reward
Model (RM).
- C) Optimising the policy against the Reward Model using the PPO algorithm.
- **D) Utilising a high-precision programmatic grader (like a Python unit test) to assign
a strict 1 or 0 verifiable reward.**


3. Why is standard Supervised Fine-Tuning (SFT) sometimes challenging or insufficient on
its own?
- A) There might be many "OK" answers rather than a single objective right answer.
- B) The model's answer might actually be better than the human label provided.
- C) It is difficult to ensure the model actually learns the underlying reasoning process
rather than just memorising demonstrations.
- **D) All of the above**

## Part 3 SAQ

1. Reward Hacking: Briefly define "Reward Hacking" in the context of LLM training. Provide an example of how an LLM might hack a reward function designed to evaluate a chess-playing assistant.

Reward hacking is taking "improper" actions to seek the highest possible reward but are not actually desired by the model developers.

For e.g., if an LLM is only penalised for losing a chess game, it might learn to always lose a chess game, and have no incentive to win.

2. Verifiable Rewards: What is RLVR (Reinforcement Learning from Verifiable Rewards), and how does its reward mechanism fundamentally differ from standard RLHF?

RLVR uses an algorithmic grader to automatically assess model output.

## Part 4 LAQ

1. The Evals Crisis: The lecture mentions that Machine Learning is currently in a "deep evals crisis". Discuss the three specific reasons for this crisis as outlined in the slides

Crisis is due to (1) explicit hill climbing (2) emergence and (3) contamination.

(1) AI labs train models for specific benchmarks
(2) It is hard to predict what new skills may arise with dataset size, meaning scaling a model may saturate a benchmark unexpectedly
(3) Benchmarks may leak into training code.