# Mujoco System Identification
A framework for system identification using the **MJX differentiable simulator**.

## Features
- Estimates built-in **MuJoCo parameters** (e.g., mass, inertia, control gains, and more).
- Captures **complex non-linear behaviors** using **neural networks**.

## Usage Workflow
1. **Prepare Your System Model**
   - Ensure you have an **XML model** of your system.

2. **Collect Real-World Data**
   - Perform real-world experiments:
     - Record **state sequences** to identify your system (e.g., joint positions, joint velocities, main body position, and velocity).
     - Log the **control actions** applied to your system.

3. **Run System Identification**
   - **Set up your dataset**.
   - **Select parameters** to optimize.
   - **Run the optimization process**.

4. **Validate Optimized Parameters**
   - See **[this example](examples/result_simulation.ipynb)** for validation.

---

## Installation

### Create a Virtual Environment
```
python3.11 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install mujoco
pip install mujoco_mjx
pip install brax
pip install -U "jax[cuda12]"
pip install pandas
```
### Install the Framework

```
pip install -e .
```


## Usage Examples
- [Dataset Generation](examples/dataset_generation.ipynb)
    - Learn how to create a dataset in the required format.
- [Parameter Optimization](examples/optimization.ipynb)
    - Define custom model parameters.
    - Optimize parameters and validate the obtained results.
- [Result Validation](examples/result_simulation.ipynb)
    - Validate optimized parameters using a modified model.xml file.
    - Generate a video showcasing model performance.