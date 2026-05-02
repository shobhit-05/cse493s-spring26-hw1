## Part 3: Synthesis and Analysis

### 1. Characterize when each strategy helps

Our main takeaway is that train time and test time compute both help but for different reasons and at different stages.

For train time compute, the baseline modular division run (`p=97`) showed that just running longer isn't enough. After a long run, test accuracy still stayed low (around 0.13), so extra steps alone didn't produce grokking. Once we changed the setup (AdamW + stronger weight decay) then we were able to observe grokking. 

### Train time Summary
| Intervention | Effect | Interpretation |
|---|---|---|
| AdamW + `weight_decay=1.0` (tuned-up), `train_frac=0.3` (tuned-down) | Train >= 0.99 at step 58k, test >= 0.99 at step 102k | Evident grokking signature; clear train-first, test-later transition |
| AdamW + `weight_decay=1.0` (tuned-up), `train_frac=0.5` (tuned-down) | Train and test >= 0.99 at the same step 264k | Generalization is strong but simultaneous, so the delayed grokking phase is less visible |

For test time compute, our preliminary scaling results looked roughly linear across the tested compute range for both sequential and parallel methods (with parallel generally giving a better return per extra token). The intervention study also made it clear that not all extra inference compute is equal.

### Test time Summary
| Intervention | Effect | Interpretation |
|---|---|---|
| `T=0.8` (higher temperature) | Lower accuracy | Too much divergence; harder to reach consensus |
| Structured prompting | Lower accuracy | Reduced diversity; more correlated failures across samples |
| Chain-of-thought prompting | Higher accuracy | Better reasoning guidance without over constraining exploration |

Overall, train time compute helps most when optimization pushes the model toward a generalizable algorithm while test time compute helps when it improves reasoning quality and sample diversity.

### 2. Failure modes

The common theme across failures is that more compute does not automatically mean better generalization.

At train time, long runs can still fail if the optimizer/data setup keeps the model in a memorization heavy regime. At test time, larger budgets can be wasted if the model keeps extending weak reasoning or if samples become too similar to each other. The temperature and prompting results highlight this directly, that too much randomness hurts consistency, but too much structure can also hurt by collapsing diversity.

What this suggests is that compute only helps when it improves search over useful solution paths. If it only amplifies the same bad trajectory, in weights during training or in reasoning traces during inference, performance saturates or drops.

### 3. Propose a strategy

With a fixed budget and mixed difficulty, we would use a staged allocation policy.

For easy problems, we keep inference cheap (short reasoning or single sample) and spend almost no extra train time compute.

For medium problems, we use moderate inference scaling first (small parallel samples with CoT prompting), then if confidence is low we escalate. 

For hard problems, we combine using targeted continued finetuning with regularization settings that showed better generalization in Part 1, and a larger test time scaling especially parallel sampling with an aggregation rule that matches the objective.

In other words, we would route compute adaptively, starting with low cost and then scale up only when needed. That matches what we observed in this homework as well which is better than applying a uniformly large budget to every problem.
