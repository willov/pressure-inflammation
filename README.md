# Inflammatory upregulation in skin after oronasal mask application in a clinical context - A repeated measures design study

Code for the simulations in the manuscript titled 'Inflammatory upregulation in skin after oronasal mask application in a clinical context - A repeated measures design study'.

## The model equations

The model equations are available in the `Models/M6_8.txt` file.

The models are simulated using the custom python package `sund toolbox` (version 1.4.0), which uses sundials/cvode to solve the system of ODEs. Information about the toolbox is available at https://isbgroup.eu/sund-toolbox/

## Running the code

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed. Then run:

```bash
uv run main.py
```

The code is set up to reproduce the figures from the paper (being saved in the `Figures` folder).

To reproduce the parameter optimization, set `DO_OPTIMIZATION=True`. To estimate the parameter bounds set `DO_PI=True`. By default, the parameter estimation and parameter bounds are searched using a parallel approach. This can be disabled by setting `RUN_IN_PARALLEL=False`.
