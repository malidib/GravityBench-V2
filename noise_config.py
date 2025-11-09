"""Central configuration for simulation noise."""

# Toggle for enabling positional noise in generated simulations
ENABLE_NOISE = True  # Set to False to disable noise completely

# Available noise models: 'gaussian', 'linear_growth', 'exponential_growth', 'power_law'
NOISE_TYPE = 'gaussian'

# Base magnitude of the noise distribution. Adjust to control perturbation strength.
NOISE_LEVEL = 0.1

# Optional seed for reproducible noise. Set to None for stochastic runs.
NOISE_SEED = 42
