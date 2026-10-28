# Matter Accretion, Greybody Factor, and Chaos Throttling in Charged Bumblebee Black Holes: A Quasi-Reversible and Deep Learning Framework

This repository contains the official numerical simulation, nonlinear time-series Recurrence Quantification Analysis (RQA), Physics-Informed Neural Network (PINN) wave-scattering solver, and 2D Convolutional Neural Network (CNN) inverse-verification pipeline accompanying the paper:

> **"Matter Accretion, Greybody Factor, and Chaos Throttling in Charged Bumblebee Black Holes: A Quasi-Reversible and Deep Learning Framework"**

---

## Repository Structure

```text
bumblebee_code/
├── figs/                                  # Generated publication-grade figures (PNG/JPG)
│   ├── dataset_overview_corrected.jpg     # 3-panel ML dataset summary (X-Q space, 2D RP, WKB curves)
│   ├── figA_velocity_profile.png          # Transonic radial infall velocity profile v(x_*)
│   ├── figB_recurrence_plot.png           # Reconstructed (r,v) phase-space Recurrence Plot R_{i,j}
│   ├── figC_R_vs_X.png                    # Recurrence-based reversibility R vs. X across charges Q
│   ├── figD_lambda_vs_X.png               # Empirical divergence rate lambda vs. X across charges Q
│   ├── figE_R_contour.png                 # 2D parameter-space contour map of reversibility R(X,Q)
│   ├── figF_lambda_contour.png            # 2D parameter-space contour map of divergence rate lambda(X,Q)
│   ├── figG_metric_function.png           # Bumblebee metric potential f(r) deformation vs. X and Q
│   ├── figH_Veff_vs_Q.png                 # Klein-Gordon effective potential V_eff(r) vs. Q (fixed X)
│   ├── figI_Veff_vs_X.png                 # Klein-Gordon effective potential V_eff(r) vs. X (fixed Q)
│   ├── figJ_sonic_radius.png              # 4-panel absolute r_c vs. horizon-normalized r_c/r_h sonic shift
│   ├── figK_wkb_transmission.png          # 1st-order Schutz-Will WKB greybody transmission |T|^2
│   ├── pinn.png                           # PINN complex wavefunction solution Psi(x_*) & loss convergence
│   ├── plot.png                           # Auxiliary exploratory diagnostic plot
│   ├── Throtting_of_KS_entropy2.png       # Analytical Lyapunov exponent / KS entropy rate decay vs. X
│   └── training.png                       # 2D CNN Huber loss, Parity Plot (R^2=0.999), & 64x64 Grad-CAM
├── bumblebee_potentials.py                # Core physics module: metric f(r), horizons, V_eff, WKB solver
├── bumblebee_rqa.py                       # Core RQA module: Hamiltonian URF solver, tortoise clock, Takens RQA
├── bumblebee_rp_dataset_balanced.npz      # Pre-generated compressed numpy archive of recurrence tensors
├── DL_Research.ipynb                      # Original Google Colab notebook (Dataset + CNN + PINN)
├── DL_Research_updated.ipynb              # Updated/verified Google Colab notebook (R^2=0.999 & 64x64 Grad-CAM)
├── make_figures.py                        # Script to generate RQA & nonlinear dynamics figures (Fig A - Fig F)
├── make_physics_figures.py                # Script to generate metric, accretion & WKB figures (Fig G - Fig K)
├── reqirements.txt                        # Python package dependencies
└── README.md                              # Project documentation
```

---

## Core Modules & File Descriptions

### 1. Physics & Nonlinear Dynamics Modules (Local Python Scripts)
* **`bumblebee_potentials.py`**: Self-contained analytical and numerical physics module implementing the charged bumblebee black hole spacetime (Javed et al., 2026). Includes:
  * `f_and_fprime(r, X, Q, M)`: Evaluates the Lorentz-violating metric function $f(r) = \frac{1}{1+X} - \frac{2M}{(1+X)r} + \frac{2Q^2}{(2+X)r^2}$ and its radial derivative $f'(r)$.
  * `horizon_radius(X, Q, M)` & `is_subextremal(X, Q, M)`: Computes the outer event horizon $r_h$ and enforces the sub-extremal bound $2(1+X)Q^2 < (2+X)M^2$.
  * `urf_sonic_radius(X, Q, M)`: Solves the exact Ultra-Relativistic Fluid ($\omega = 1/2$) critical sonic condition $r_c f'(r_c) - 4f(r_c) = 0$ using Brent's root-finding method (`scipy.optimize.brentq`).
  * `tortoise_of_r(r_array, X, Q, M)`: Integrates $dx_* = dr/f(r)$ to map radial coordinates onto the horizon-regularized tortoise coordinate $x_*$.
  * `V_eff(r, X, Q, M, l)` & `wkb_transmission(w_array, X, Q, M, l)`: Computes the scalar perturbation barrier $V_{\text{eff}}(r)$ and evaluates the 1st-order Schutz–Will WKB greybody transmission probability $|T(\omega)|^2$.
* **`bumblebee_rqa.py`**: Implements the bridge between relativistic transonic fluid accretion and nonlinear state-space Recurrence Quantification Analysis (Zarghami et al., 2026; Tsallis & Borges, 2024):
  * `transonic_velocity_profile(X, Q, M)`: Inverts the conserved Hamiltonian level curve $H(r,v) = H_c$, stitching the subsonic branch ($r \ge r_c$) and supersonic branch ($r < r_c$).
  * `to_tortoise(r, v, X, Q, M)`: Resamples the infall velocity $v(r)$ uniformly along the tortoise coordinate $x_*$ (serving as the natural embedding clock).
  * `takens_embed(signal, d=3, tau=4)`: Reconstructs the high-dimensional phase-space trajectory $Y_i = (v_i, v_{i+\tau}, \dots, v_{i+(d-1)\tau})$ via Takens' delay embedding theorem.
  * `recurrence_matrix(states)` & `reversibility_score(binary_mat)`: Constructs the binary recurrence plot $R_{i,j}(\epsilon) = \Theta(\epsilon - \|Y_i - Y_j\|)$ at a fixed $10\%$ recurrence density and quantifies time-reversibility $R$ from the anti-diagonal line distribution $P(l)$.
  * `divergence_rate(states)`: Computes a Rosenstein-style nearest-neighbor trajectory divergence proxy for the local Lyapunov exponent $\lambda$.
* **`make_physics_figures.py`**: Driver script that imports `bumblebee_potentials.py` and generates Figures **G, H, I, J, and K** into `./figs/`.
* **`make_figures.py`**: Driver script that imports `bumblebee_rqa.py` and generates Figures **A, B, C, D, E, and F** into `./figs/`.

### 2. Deep Learning & PINN Notebooks (Google Colab / PyTorch)
* **`DL_Research_updated.ipynb`** (and `DL_Research.ipynb`): End-to-end GPU-accelerated notebook containing three self-contained cells:
  1. **Synthetic ML Dataset Generator**: Samples $N=500$ sub-extremal $(X, Q)$ black hole configurations, integrates the exact Hamiltonian URF velocity profiles on a 64-step tortoise grid, and exports `metadata.csv`, `recurrence_plots.npz` ($64\times 64$ tensors), and `greybody_curves.npz` into `charged_bumblebee_ml_dataset.zip`.
  2. **2D CNN Inverse Regressor & Grad-CAM (`BumblebeeRecurrenceCNN`)**: Trains a 3-block 2D Convolutional Neural Network ($\text{Conv2d}(1\rightarrow 32\rightarrow 64\rightarrow 128)$) with Huber loss to invert the recurrence plots $R_{i,j} \mapsto (\hat{X}, \hat{Q})$, achieving **$R^2 = 0.999$**, and extracts full-resolution ($64\times 64$) Grad-CAM saliency heatmaps.
  3. **Physics-Informed Neural Network (`WavePINN`)**: Solves the 1D tortoise-coordinate Klein-Gordon wave equation $\frac{d^2\Psi}{dx_*^2} + \left[\omega^2 - V_{\text{eff}}(x_*)\right]\Psi = 0$ continuously over $x_* \in [-15, 15]$ via automatic differentiation (`torch.autograd`) with ingoing horizon boundary conditions.

---

## Installation & How to Run

### 1. Local Environment Setup
Ensure you have **Python 3.9+** installed. Create and activate a virtual environment, then install the required dependencies:

```bash
# Navigate to the project directory
cd bumblebee_code

# Create and activate virtual environment (optional if venv/ already exists)
python3 -m venv venv
source venv/bin/activate        # On Linux/macOS
# venv\Scripts\activate         # On Windows

# Install dependencies
pip install -r reqirements.txt
pip install torch scikit-learn pandas tqdm matplotlib scipy numpy
```

### 2. Generate the Physics Verification Figures (Figs G–K)
Run `make_physics_figures.py` to verify the background metric, effective potential barrier, sonic horizon scaling, and WKB greybody transmission:

```bash
python3 make_physics_figures.py
```
*Outputs saved to `./figs/`: `figG_metric_function.png`, `figH_Veff_vs_Q.png`, `figI_Veff_vs_X.png`, `figJ_sonic_radius.png`, `figK_wkb_transmission.png`.*

### 3. Generate the RQA & Nonlinear Dynamics Figures (Figs A–F)
Run `make_figures.py` to execute the tortoise-coordinate Takens embedding and generate the state-space recurrence plots and $(X, Q)$ contour maps:

```bash
python3 make_figures.py
```
*Outputs saved to `./figs/`: `figA_velocity_profile.png`, `figB_recurrence_plot.png`, `figC_R_vs_X.png`, `figD_lambda_vs_X.png`, `figE_R_contour.png`, `figF_lambda_contour.png`.*

### 4. Run the Deep Learning & PINN Pipeline
Open **`DL_Research_updated.ipynb`** in Jupyter Lab, VS Code, or **Google Colab** (recommended with an NVIDIA T4 GPU runtime) and run all cells sequentially:
* **Cell 1** installs any missing Colab packages.
* **Cell 2** generates the 500-sample dataset (`charged_bumblebee_ml_dataset.zip`) and outputs `dataset_overview_corrected.jpg`.
* **Cell 3** trains the 2D CNN Regressor for 40 epochs, prints the test-set $R^2$ and RMSE metrics, and outputs `training.png`.
* **Cell 4** trains the Physics-Informed Neural Network (`WavePINN`) for 2000 epochs and outputs `pinn.png`.

---

## Detailed Guide to Generated Figures (`./figs/`)

### Part I: Spacetime Geometry, Accretion & Quantum Scattering (`make_physics_figures.py`)
| Figure Filename | Physical Description & Key Takeaway |
| :--- | :--- |
| **`figG_metric_function.png`** | **Dual-panel plot of the metric potential $f(r)$ vs. $r/M$** (vertical dotted lines mark event horizons $r_h$). *Left ($Q=0.5$):* Increasing $X \in \{0.1, \dots, 1.9\}$ globally flattens $f(r)$ and lowers the asymptotic plateau to $1/(1+X)$. *Right ($X=0.5$):* Increasing $Q \in \{0.1, \dots, 0.85\}$ shifts the event horizon $r_h$ inward via Coulomb repulsion. |
| **`figH_Veff_vs_Q.png`** | **Scalar effective potential $V_{\text{eff}}(r)$ vs. $r/r_h$ across $Q \in \{0.1, \dots, 0.85\}$ at fixed $X=1.0, l=2$.** Demonstrates that higher electric charge $Q$ systematically raises the electrostatic scattering barrier peak. |
| **`figI_Veff_vs_X.png`** | **Scalar effective potential $V_{\text{eff}}(r)$ vs. $r/r_h$ across $X \in \{0.1, \dots, 1.9\}$ at fixed $Q=0.5, l=2$.** Shows that spontaneous Lorentz symmetry breaking ($X$) compresses the barrier height and flattens its curvature at the peak, driving chaos suppression. |
| **`figJ_sonic_radius.png`** | **4-panel numerical verification of URF ($\omega=1/2$) sonic radius scaling.** Resolves an important geometric subtlety: while the *absolute* coordinate sonic radius $r_c$ decreases slightly with $X$ and $Q$ (left panels, matching Table 2 of Javed et al.), the *horizon-normalized* sonic radius $r_c / r_h$ shifts strictly **outward** (right panels) because the event horizon $r_h$ shrinks faster than $r_c$. |
| **`figK_wkb_transmission.png`** | **Independent 1st-order Schutz–Will WKB Greybody Factor $|T(\omega r_h)|^2$ ($l=2$).** Cross-checks the hypergeometric matching results of Javed et al. (Fig. 15) using an independent semi-analytic WKB method. Shows that increasing $X$ (left) and $Q$ (right) shifts the transmission S-curve toward higher frequencies, throttling Hawking radiation. |
| **`Throtting_of_KS_entropy2.png`** | **First-principles analytical proof of Chaos Throttling.** Plots the analytical Lyapunov exponent $\lambda_{\text{analytic}} = \sqrt{-V_{\text{eff}}''(x_*)/2} \approx K_{\text{BG}}$ at the barrier peak vs. $X \in [0.1, 1.9]$ ($Q=0.5, l=2$). Proves a monotonic $76.9\%$ decay in chaotic scrambling rate (from $0.0888$ down to $0.0205$), pulling the system toward the "edge of chaos" ($\lambda \to 0$). |

### Part II: State-Space Reconstruction & Recurrence Quantification (`make_figures.py`)
| Figure Filename | Physical Description & Key Takeaway |
| :--- | :--- |
| **`figA_velocity_profile.png`** | **Single-case transonic accretion trajectory $v(x_*)$ ($X=0.5, Q=0.5, M=1.0$).** Shows the radial three-velocity resampled uniformly along the tortoise coordinate clock $x_*$, smoothly bridging the subsonic far-field ($x_* < 0$) and supersonic near-horizon ($x_* > 0$) regimes ($r_c=2.305, v_c=0.707, r_h=1.837$). |
| **`figB_recurrence_plot.png`** | **Binary state-space Recurrence Plot $R_{i,j}$ ($10\%$ target density, $d=3, \tau=4$).** Visualizes the reconstructed phase-space attractor of the transonic flow at $X=0.5, Q=0.5$, yielding high anti-diagonal reversibility ($R=0.8774$). |
| **`figC_R_vs_X.png`** & **`figD_lambda_vs_X.png`** | **1D parameter sweeps of Reversibility $R(X)$ and Empirical Divergence Rate $\lambda(X)$** for fixed charges $Q \in \{0.2, 0.4, 0.6\}$ (analogues of Zarghami et al. Figs. 7 & 8). Illustrates the inverse relationship between trajectory divergence (entropy rate) and time-reversibility. |
| **`figE_R_contour.png`** & **`figF_lambda_contour.png`** | **2D $(X, Q)$ parameter-space contour maps of Reversibility $R(X,Q)$ and Divergence Rate $\lambda(X,Q)$** (analogues of Zarghami et al. Fig. 9b). Maps the "quasi-reversible ridge" across black hole parameter space where chaotic scrambling is minimized and time-reversal symmetry is maximized. |

### Part III: Deep Learning Inverse Verification & PINN (`DL_Research_updated.ipynb`)
| Figure Filename | Physical Description & Key Takeaway |
| :--- | :--- |
| **`dataset_overview_corrected.jpg`** | **3-panel summary of the 500-sample ML dataset.** *Left:* Uniform sub-extremal $(X, Q)$ sampling colored by $\lambda_{\text{analytic}} \in [0.025, 0.105]$. *Center:* Sample $64\times 64$ continuous recurrence tensor $R_{i,j}$ ($X=0.56, Q=0.76$) showing the characteristic sonic-transition pinch. *Right:* Sample WKB greybody curves $|T(\omega r_h)|^2$ showing monotonic suppression with $X$. |
| **`training.png`** | **2D CNN Inverse Regression & Explainable AI (Grad-CAM) Results.** *Left:* Smooth convergence of training and validation Huber loss to $<0.01$ over 40 epochs. *Center:* Parity plot of CNN-predicted $\hat{X}$ vs. actual $X$ on the held-out test set, achieving **$R^2 = 0.999$**. *Right:* Full-resolution ($64\times 64$) Grad-CAM saliency map proving the CNN autonomously attends to the symmetric off-diagonal recurrence lobes flanking the sonic horizon. |
| **`pinn.png`** | **Physics-Informed Neural Network (`WavePINN`) Wave Scattering Solution ($X=0.5, Q=0.4, \omega=0.4, l=1$).** *Left:* Continuous mesh-free solution of the real part $u(x_*)$, imaginary part $v(x_*)$, and damped amplitude envelope $|\Psi(x_*)|$ across the tortoise coordinate $x_* \in [-15, 15]$. *Right:* Log-scale convergence of the differential equation residual loss over 2,000 epochs down to $\sim 1.3 \times 10^{-3}$. |

---

## Primary References
1. **F. Javed, S. Shaukat, A. Waseem, and G. Mustafa**, *"Matter accretion and greybody factor of charged bumblebee gravity black hole"*, Chaos, Solitons and Fractals **208**, 118194 (2026).
2. **C. Tsallis and E. P. Borges**, *"Nonlinear dynamical systems: Time reversibility versus sensitivity to the initial conditions"*, Chaos, Solitons and Fractals **182**, 114743 (2024).
3. **R. Zarghami, M. Mohammadpourfard, and G. G. Akkurt**, *"Reversibility and entropy in bubbling fluidized beds: A recurrence-based analysis"*, Powder Technology **469**, 121778 (2026).