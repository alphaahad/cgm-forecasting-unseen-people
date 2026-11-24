# Contributing

Thanks for your interest. This is a small research repository, so contributions are welcome but kept simple.

- **Questions and bugs:** open an issue. For a bug, say which script you ran, your Python and package versions, and the error or the numbers that look wrong.
- **Pull requests:** keep them small and focused. Before opening one, run `pytest -q` and `ruff check .`; both must pass (CI runs them too).
- **Changing an analysis:** the results are reproducible and seeded, so a change that alters a number should say so and explain why. Please rerun the affected step and include the changed files in `results/`.
- **Data:** do not commit raw data or anything derived at the level of individual readings. The datasets are public and each keeps its own licence (see `docs/DATA.md`).
- **Licences:** by contributing code you agree it is released under the MIT licence; result files are CC BY-NC-SA 4.0.
- **Not medical advice:** please do not describe these results as suitable for clinical decisions.
