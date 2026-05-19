# Homework: Inverted Pendulum World Model Training

## Objective

Train one world model for the MuJoCo `InvertedPendulum-v5` environment.

Your goal is to improve the model so that it can predict future states for as
many steps as possible before its prediction drifts away from MuJoCo ground
truth.

The central question is: **How many future steps can your world model predict
before it fails?** We also evaluate short-, medium-, and long-horizon prediction
error, measured by the prediction error `nMSE` at 10, 100, and 1000 steps,
respectively.

This assignment only studies dynamics prediction. There is no policy learning,
no reward prediction, no actor-critic training, and no controller training.

You will have three Canvas submission attempts. Each submission should include
your code and your best checkpoint. After each submission, I will evaluate your
submitted checkpoint on the TA-held dataset, record the best result obtained
under the official evaluation protocol, and reply with your score/feedback so
you can decide how to improve your next attempt.

## What You Are Given

Just like the previous assignment, you are given a Colab notebook and a runnable
starter model.

Please fork the repository:

```text
https://github.com/WeijieLai1024/EEC289A_WorldModel-Homework
```

Use the notebook as the main entry point:

```text
notebooks/homework_colab.ipynb
```

The notebook will set up the repo, install dependencies, generate MuJoCo
trajectories, train the starter world model, evaluate long-horizon prediction,
plot scoreboard curves, and package final artifacts.

## Main Metric

After the 10-step warm-up, your model predicts forward without seeing the true
future states. We then compare its predicted trajectory against the MuJoCo
trajectory.

The main score is `VPT(Valid Prediction Time)80@0.25`, which answers: **For how
many future steps can the model stay accurate on at least 80% of test windows?**

Here, "accurate" means the normalized state prediction error stays below `0.25`.
So if your `VPT80@0.25 = 120`, it means at least 80% of test rollouts are still
reliable up to step 120. A higher value means your world model stays useful for
longer.

We also report the full error curve:

```text
nMSE@10, 100, 1000
```

These numbers show how prediction error grows as the rollout gets longer.

## What You Should Modify

You may modify only:

```text
student/model.py
student/rollout.py
student/losses.py
student/metrics.py
configs/student.yaml
```

The main optimization space is:

```text
model architecture
rollout loss
training horizon
loss weights
training hyperparameters
```

Good directions may include deeper or wider MLPs, GRU-based world models, better
residual delta prediction, larger `rollout_train_horizon`, stronger rollout
loss, and better balance between one-step accuracy and rollout stability.

Do not modify the benchmark code, data generation, or official scoring.

## Colab Workflow

Run the notebook in this order:

```text
1. Set COURSE_REPO_URL to your fork.
2. Run the setup cells.
3. Run tests.
4. Generate the dev dataset.
5. Train the starter model.
6. Evaluate test and OOD performance.
7. Modify student code.
8. Re-run training and evaluation.
9. Compare VPT/nMSE curves.
10. Run the final public scoreboard cell.
```

## Deliverables

Submit:

```text
Your GitHub repository link
Colab output, such as training logs or evaluation results
Your best checkpoint
Report, suggested length <= 5 pages, for FINAL submission only
```

## Logistics

Deadline: **23:59 PST May 25th, 2026**

Team size: **1-4 students**

If working in a team, clearly state each member's contribution in your report.
