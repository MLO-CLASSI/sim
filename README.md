# CLASSI Spectrograph Instrument Simulator

Detector, instrument, throughput, and WCS models for the MLO CLASSI spectrograph.

## Install

Install the shared data package first when developing from sibling checkouts:

```bash
python -m pip install -e ../shared-data
python -m pip install -e .
```

The distribution is named `classi-sim`; its import namespace is `simulator`:

```python
from simulator import InstrumentSimulator, SpectrographModel
```

Physical hardware models live in `simulator.components`. The component catalog
keeps hardware geometry and calibration-data associations together so the
simulator and ETC can use the same definitions. For example:

```python
from astropy import units as u
from simulator import SpectrographModel
from simulator.components import (
    CANON_EF100_F2,
    CLASSI_FIBER,
    FLI_AR571,
    NEWPORT_MASTER_1294,
    THORLABS_AC508_180_AB,
)

spectrograph = SpectrographModel.from_components(
    detector=FLI_AR571,
    grating=NEWPORT_MASTER_1294,
    collimator=THORLABS_AC508_180_AB,
    camera_lens=CANON_EF100_F2,
    fiber=CLASSI_FIBER,
    incidence_angle=32 * u.deg,
    diffraction_angle=-20 * u.deg,
    fiber_count=7,
    fiber_pitch=250 * u.um,
)
```

The existing numeric `SpectrographModel(...)` constructor remains available for
custom geometries and backward compatibility.
