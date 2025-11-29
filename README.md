# 🚁 PROJECT: AERO-LIQUID
### *Liquid Neural Networks for Robust Drone Control in LLM-Generated Turbulence*

## 👥 THE SQUAD & ROLES
| ROLE | USER | FOCUS AREA |
| :--- | :--- | :--- |
| **A. The Physicist** | **@User_A** | **Environment & Config.** Master of `PyFlyt`. Defines wind dynamics, reward functions, and physics constraints. |
| **B. The Artist** | **@User_B** | **Viz & Explainability.** Master of `Anthropic API`. Builds the "Crash Reporter" and the final visual demo/video. |
| **C. The Brain** | **@User_C** | **Model & Training.** Master of `Torch/SB3`. Implements the Liquid Time-Constant (LTC) network and PPO training loop. |

---

## 🛠️ PHASE 0: SETUP (0.5 Hours)
**GOAL:** Verified "Hello World" on all machines.

1.  **ALL:** Create Repo. `git init`.
2.  **ALL:** Create `requirements.txt`:
    ```text
    gymnasium
    PyFlyt
    stable-baselines3
    shimmy
    torch
    anthropic
    matplotlib
    pandas
    ncps  # Neural Circuit Policies (for LTC cells)
    ```
3.  **ALL:** `pip install -r requirements.txt`

---

## 🟧 PHASE 1: THE FOUNDATION (3.0 Hours)
**GOAL:** Environment running, API connected, Network Skeleton built.

* **👩‍🔬 Physicist (Env):**
    * Create `env/wrapper.py`.
    * Initialize `PyFlyt/QuadX-Hover-v0`.
    * Write a custom Wrapper to normalize observations (critical for Liquid Nets).
    * *Deliverable:* A script that runs the drone with random actions and doesn't crash immediately.

* **🎨 Artist (Claude):**
    * Create `utils/claude_interface.py`.
    * Write function `get_curriculum_update(metrics_dict)`.
    * *Prompt Engineering:* "You are a Gym Instructor. Based on these success rates, output a JSON for wind_force and turbulence."
    * *Deliverable:* A script that sends dummy data to Claude and gets valid JSON back.

* **🧠 Brain (Net):**
    * Create `models/liquid_policy.py`.
    * Import `LTC` from `ncps.torch`.
    * Subclass `ActorCriticPolicy` from SB3 to accept a custom `features_extractor` (the Liquid Net).
    * *Deliverable:* A PPO model that initializes without dimension errors.

---

## 🟩 PHASE 2: INTEGRATION & BASELINE (4.0 Hours)
**GOAL:** Train the "Standard" model (the loser) and connect the components.

* **👩‍🔬 Physicist:**
    * Refine Reward Function in `env/rewards.py`. (Bonus for stability, heavy penalty for crashing).
    * Assist Brain with Observation Space dimensions.

* **🎨 Artist:**
    * Create `vis/logger.py`.
    * Set up a live plot (Matplotlib) that updates `Reward` vs `Time` every episode.
    * *Task:* Ensure we can see the training happening in real-time.

* **🧠 Brain:**
    * **TRAINING RUN 1:** Standard PPO (MLP).
    * Use `Stable-Baselines3` defaults.
    * Train for 1M steps (or until stable).
    * Save as `models/baseline_mlp.zip`.

---

## 🟪 PHASE 3: THE LIQUID TRAINING LOOP (6.0 Hours)
**GOAL:** Train the "Liquid" model (the winner) with AI Curriculum.

* **🧠 Brain:**
    * **TRAINING RUN 2:** Liquid PPO.
    * Swap the MLP for the LTC module.
    * *Critical:* Lower the learning rate! Liquid nets are volatile.
    * Start the long training run.

* **👩‍🔬 Physicist:**
    * **The LLM Loop:** Connect the Artist's `claude_interface` to the training loop.
    * *Logic:* Every 50,000 steps, pause training -> send stats to Claude -> update PyFlyt Wind params -> resume training.

* **🎨 Artist:**
    * While training runs, build the **"Black Box Recorder"**.
    * Create `vis/crash_analyzer.py`.
    * Logic: Take a `.csv` of the last 30 frames of a crash -> Send to Claude -> Display text explanation ("Crash due to sudden wind shear").

---

## 🟫 PHASE 4: EVALUATION & POLISH (4.0 Hours)
**GOAL:** Prove superiority.

* **👩‍🔬 Physicist:**
    * Create `eval/stress_test.py`.
    * Load both `baseline_mlp.zip` and `liquid_ltc.zip`.
    * Run them through a "Hurricane" scenario (Max wind).
    * Log the survival times.

* **🎨 Artist:**
    * Create `demo/render_video.py`.
    * Record a side-by-side video:
        * Left: Baseline (Jittery, crashes).
        * Right: Liquid (Smooth, adapts).
    * Overlay the "Claude Crash Analysis" text when the Baseline crashes.

* **🧠 Brain:**
    * Generate the scientific plots.
    * Plot: `Wind Intensity` (x-axis) vs `Reward` (y-axis).
    * *Narrative:* "As wind increases, MLP fails. Liquid adapts."

---

## 🟥 PHASE 5: THE PITCH (2.5 Hours)
**GOAL:** Slides and Video.

* **👩‍🔬 Physicist:** Write the "Methodology" slide. Explain *Why* PyFlyt is realistic and *How* the reward function works.
* **🎨 Artist:** Edit the 2-minute video. Intro -> Problem (Wind) -> Solution (Liquid + Claude) -> Demo -> Outro.
* **🧠 Brain:** Write the "Results" slide. "30% greater robustness in high winds."

---

## ⚠️ FALLBACK PROTOCOLS (Read Continuously)

1.  **Liquid Net Not Converging?**
    * *Fix:* Switch LTC to a standard LSTM. It's less "novel" but guaranteed to work.
2.  **PyFlyt Too Heavy?**
    * *Fix:* Disable rendering during training. Only render for the demo video.
3.  **Claude Rate Limit?**
    * *Fix:* Hard-code the curriculum levels (Level 1: 5m/s wind, Level 2: 10m/s wind). Use Claude only for the final demo analysis.

---

## 💻 KEY COMMANDS

**Activate Virtual Env:**
`source venv/bin/activate`

**Train Baseline:**
`python train.py --model mlp --curriculum static`

**Train Liquid:**
`python train.py --model ltc --curriculum claude`

**Run Demo:**
`python demo.py --compare`
