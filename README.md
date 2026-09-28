# CLASSI Spectrograph Instrument Simulator

Detector, instrument, throughput, and WCS models for the MLO spectrograph.

## Install

Install the shared data package first when developing from sibling checkouts:

```bash
python -m pip install -e ../shared-data
python -m pip install -e .
```

The distribution is named `classi-sim`; its import namespace is `sim`:

```python
from sim import DetectorModel, InstrumentSimulator, SpectrographModel
```

The example notebook is in `src/simulation.ipynb` and reads calibration curves
and reference spectra through the installed `shared_data` package.
