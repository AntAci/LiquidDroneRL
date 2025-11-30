# 🚁 Drone-go-brrrrr: Liquid Neural Networks for Robust Drone Control

### *Liquid Neural Networks Outperform Traditional MLPs in Windy Drone Navigation*

A reinforcement learning project demonstrating the superior performance of Liquid Time-Constant (LTC) Neural Networks over traditional MLP baselines for drone control in challenging wind conditions. This project uses PPO (Proximal Policy Optimization) to train agents that must navigate a 2D drone environment with varying wind difficulties.

---

## 🎯 Key Results

### **Liquid NN Dominates Across All Metrics**

Our comprehensive evaluation across 5 wind difficulty levels shows **Liquid Neural Networks consistently outperform MLP baselines**:

| Wind Difficulty | MLP Survival (steps) | Liquid Survival (steps) | MLP Reward | Liquid Reward | Improvement |
|----------------|---------------------|-------------------------|------------|---------------|-------------|
| **0 - No Wind** | 500.00 | 500.00 | 1,952.05 | **3,680.07** | **+88.5%** |
| **1 - Mild** | 500.00 | 500.00 | 1,978.83 | **3,911.90** | **+97.7%** |
| **2 - Medium** | 500.00 | 500.00 | 1,799.26 | **3,804.05** | **+111.4%** |
| **3 - Chaotic** | 379.70 | **426.65** | 1,441.47 | **3,084.81** | **+12.4% survival, +114.1% reward** |
| **4 - Extreme** | 144.45 | **163.70** | 409.49 | **850.76** | **+13.3% survival, +107.7% reward** |

### **Key Findings:**
- ✅ **Liquid NN achieves 2x higher rewards** in easy-to-medium conditions
- ✅ **Liquid NN survives 13% longer** in extreme wind conditions
- ✅ **Liquid NN maintains perfect survival** (500 steps) in conditions where MLP also succeeds
- ✅ **Superior generalization** to unseen/holdout difficulty levels

*Results based on 20 evaluation episodes per difficulty level*

---

## 🏗️ What We Built

### **Core Components**

1. **Environment (`env/drone_env.py`)**
   - Custom 2D drone navigation environment with wind dynamics
   - 5 difficulty levels: No Wind → Mild → Medium → Chaotic → Extreme
   - Reward function encouraging stability, target navigation, and survival
   - Configurable wind patterns with turbulence and gusts

2. **Liquid Neural Network Policy (`models/liquid_policy.py`)**
   - Liquid Time-Constant (LTC) cell-based feature extractor
   - Continuous-time dynamics for better temporal modeling
   - Integrated with Stable-Baselines3 PPO

3. **MLP Baseline (`models/mlp_baseline.py`)**
   - Traditional 2-layer MLP (64, 64) for comparison
   - Same training setup as Liquid NN for fair comparison

4. **Training Scripts**
   - `train/train_liquid_ppo.py`: Train Liquid NN agents
   - `train/train_mlp_ppo.py`: Train MLP baseline agents
   - Support for curriculum learning and random difficulty sampling

5. **Evaluation & Visualization**
   - `eval/compare.py`: Comprehensive performance comparison
   - `demo/visualize_drone.py`: 2D real-time visualization with Pygame
   - `demo/compare_visualize.py`: Side-by-side MLP vs Liquid comparison
   - `demo/run_3d_demo.py`: 3D visualization with Three.js
   - `create_presentation_graph.py`: Generate publication-quality graphs

6. **3D Visualization**
   - Interactive 3D drone visualization using Three.js
   - Real-time WebSocket streaming of simulation data
   - Support for GLB/OBJ 3D models

---

## 🚀 Quick Start

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd Drone-go-brrrrr

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run 2D Simulation**

**Liquid Neural Network:**
```bash
python demo/visualize_drone.py --model liquid
```

**MLP Baseline:**
```bash
python demo/visualize_drone.py --model mlp
```

**Side-by-Side Comparison:**
```bash
python demo/compare_visualize.py
```

### **Run 3D Visualization**

```bash
python demo/run_3d_demo.py
# Then open demo/renderer_3d.html in a web browser
```

### **Evaluate Models**

```bash
# Compare MLP vs Liquid across all difficulties
python eval/compare.py --episodes 20

# Evaluate specific model
python eval/eval_liquid_policy.py --episodes 10
python eval/eval_mlp_baseline.py --episodes 10
```

### **Train Your Own Models**

**Train Liquid NN:**
```bash
python train/train_liquid_ppo.py --timesteps 100000 --curriculum-difficulty
```

**Train MLP Baseline:**
```bash
python train/train_mlp_ppo.py --timesteps 100000 --curriculum-difficulty
```

**With Random Difficulty Sampling:**
```bash
python train/train_liquid_ppo.py --random-difficulty --difficulties "0,1,2,3"
```

---

## 📊 Visualization & Results

### **Available Visualizations**

1. **2D Pygame Visualization** (`demo/visualize_drone.py`)
   - Real-time drone navigation
   - Wind vector visualization
   - Target tracking
   - Interactive controls (pause, reset, difficulty adjustment)

2. **Side-by-Side Comparison** (`demo/compare_visualize.py`)
   - MLP vs Liquid NN running simultaneously
   - Direct performance comparison
   - Shared difficulty controls

3. **3D Web Visualization** (`demo/run_3d_demo.py` + `demo/renderer_3d.html`)
   - Beautiful 3D drone model
   - Web-based interactive visualization
   - Real-time data streaming

4. **Performance Graphs** (`results/`)
   - `liquid_vs_mlp_comparison.png`: Comprehensive side-by-side metrics
   - `liquid_vs_mlp_survival.png`: Survival-focused comparison
   - `reward_comparison.png`: Reward analysis
   - `survival_comparison.png`: Survival time analysis

### **Generate Presentation Graphs**

```bash
python create_presentation_graph.py --episodes 30
```

---

## 🔬 Technical Details

### **Architecture**

**Liquid Neural Network:**
- Feature Extractor: Liquid Time-Constant (LTC) cell
- Hidden Size: 32
- Time Step (dt): 0.1
- Policy Head: 64 units
- Value Head: 64 units

**MLP Baseline:**
- Architecture: [64, 64] fully connected layers
- Same policy/value head structure as Liquid NN

### **Training Configuration**

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Gamma**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.01
- **Normalization**: VecNormalize for observations and rewards

### **Environment Details**

- **Observation Space**: 12D (position, velocity, wind, target info)
- **Action Space**: 2D continuous (thrust x, thrust y)
- **Max Episode Length**: 500 steps
- **Wind Difficulties**:
  - 0: No wind (WIND_MAX = 0.0)
  - 1: Mild (WIND_MAX = 0.5)
  - 2: Medium (WIND_MAX = 1.0)
  - 3: Chaotic (WIND_MAX = 2.0 + turbulence)
  - 4: Extreme (WIND_MAX = 2.5 + turbulence + gusts)

---

## 📁 Project Structure

```
Drone-go-brrrrr/
├── env/                    # Environment implementation
│   └── drone_env.py        # DroneWindEnv with wind dynamics
├── models/                 # Neural network models
│   ├── liquid_policy.py   # Liquid NN feature extractor
│   ├── liquid_cell.py      # LTC cell implementation
│   ├── liquid_policy.zip   # Trained Liquid NN model
│   └── mlp_baseline.zip    # Trained MLP baseline
├── train/                  # Training scripts
│   ├── train_liquid_ppo.py # Train Liquid NN
│   └── train_mlp_ppo.py    # Train MLP baseline
├── eval/                   # Evaluation scripts
│   ├── compare.py         # Performance comparison
│   ├── eval_liquid_policy.py
│   └── eval_mlp_baseline.py
├── demo/                   # Visualization demos
│   ├── visualize_drone.py  # 2D Pygame visualization
│   ├── compare_visualize.py # Side-by-side comparison
│   ├── run_3d_demo.py     # 3D demo server
│   ├── renderer_3d.html    # 3D web visualization
│   └── assets/            # 3D models and assets
├── results/                # Generated graphs and results
│   ├── liquid_vs_mlp_comparison.png
│   └── liquid_vs_mlp_survival.png
└── logs/                   # TensorBoard logs
```

---

## 🎮 Controls (2D Visualization)

- **SPACE**: Pause/Resume simulation
- **R**: Reset episode
- **0-5**: Change wind difficulty
  - 0: No wind
  - 1: Mild
  - 2: Medium
  - 3: Chaotic
  - 4: Extreme
  - 5: Extreme (alternative)
- **ESC**: Quit

---

## 🤖 AI-Assisted Development

This project was developed with the assistance of cutting-edge AI coding tools:

- **Claude (Anthropic)**: Used for code generation, architecture design, debugging, and documentation
- **Devin AI**: Assisted with complex algorithm implementation and optimization
- **Windsurf**: Used for real-time code suggestions and refactoring

These AI tools significantly accelerated development, enabling rapid prototyping, testing different architectures, and generating comprehensive documentation. The collaborative workflow between human developers and AI assistants demonstrates the power of modern AI-augmented software development.

---

## 📈 Why Liquid Neural Networks?

Liquid Time-Constant (LTC) networks offer several advantages over traditional MLPs:

1. **Temporal Dynamics**: Continuous-time dynamics better model sequential dependencies
2. **Adaptability**: More robust to distribution shifts and unseen conditions
3. **Efficiency**: Can handle variable-length sequences more naturally
4. **Generalization**: Better performance on holdout/out-of-distribution test cases

Our results demonstrate these advantages in practice, with Liquid NN achieving:
- **2x higher rewards** in standard conditions
- **13% better survival** in extreme conditions
- **Superior generalization** to unseen difficulty levels

---

## 🔬 Experimental Setup

### **Training Protocol**

Both models were trained using:
- **PPO** with identical hyperparameters
- **VecNormalize** for observation and reward normalization
- **Curriculum learning** or random difficulty sampling
- **Same random seeds** for fair comparison

### **Evaluation Protocol**

- **20-30 episodes** per difficulty level
- **Deterministic policy** (no exploration noise)
- **Same environment seeds** for both models
- **Comprehensive metrics**: survival time, episode reward, standard deviations

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{drone_go_brrrrr,
  title = {Drone-go-brrrrr: Liquid Neural Networks for Robust Drone Control},
  author = {Drone-go-brrrrr Contributors},
  year = {2024},
  url = {https://github.com/yourusername/Drone-go-brrrrr},
  license = {MIT}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Stable-Baselines3** for the PPO implementation
- **Gymnasium** for the RL environment framework
- **Liquid Time-Constant Networks** research community
- **Claude, Devin AI, and Windsurf** for AI-assisted development

---

## 🐛 Known Issues & Future Work

- [ ] Add support for more complex wind patterns
- [ ] Implement curriculum learning with AI-generated difficulty schedules
- [ ] Add more sophisticated reward shaping
- [ ] Extend to 3D drone control (full 6-DOF)
- [ ] Real-world drone deployment and testing

---

## 💬 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using AI-assisted development tools**
