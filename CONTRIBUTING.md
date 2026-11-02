# Contributing

Keep capture-specific settings in configuration files and calibration logic in
the shared pipeline. Follow the existing Python style and keep changes focused.

Run the local checks before submitting a change:

```bash
python tools/checks/verify_source.py --update
python tools/checks/check.py
```

For numerical changes, compare the same data, camera split, preprocessing,
initialization, reconstruction budget and seed. Measure runtime on the same
hardware, including all calibration stages.

Include the purpose, relevant checks and results in pull requests. Report issues
with the command, environment and error log. Keep captures and model weights
outside Git. Third-party terms are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
