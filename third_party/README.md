# Third-party source checkouts

Third-party implementations are fetched on demand and keep their upstream
licenses. Their source trees and compiled extensions are excluded from the
Track4D-Calib distribution.

```bash
# Calibration dependencies
python tools/setup/fetch_dependencies.py --component calibration --output third_party

# Optional reconstruction backend
python tools/setup/fetch_dependencies.py --component 4dgs --output third_party
```

The commands create separate checkout directories and do not replace existing
ones. Versions and patches are listed in
[the dependency registry](../dependencies/third-party.json). Build the local-correlation
extension against the calibration environment as described in the [installation
instructions](../docs/INSTALL.md). Install backend
packages in a separate environment following [evaluation setup](../docs/EVALUATION.md).
Running calibration does not require `third_party/4dgs`.
