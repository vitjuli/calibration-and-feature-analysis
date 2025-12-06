# Gas Sensor Dataset

## Overview

This dataset contains measurements from SnO₂-based semiconductor gas sensors used for detecting hydrogen (H₂) and propane (C₃H₈) in air. Data was collected in Moscow, Russia during March-April 2019.

## Experimental Setup

### Sensors
- **Type:** Tin dioxide (SnO₂) semiconductor sensors
- **Number:** 4 independent sensors
- **Fabrication:** One-step flame spray pyrolysis method
- **Measurement:** Resistance changes upon gas exposure

### Operating Conditions
- **Temperature Modulation:** Cycling between 150°C and 500°C
- **Cycle Duration:** 60 seconds (35s heating, 5s dwell, 15s cooling, 5s dwell)
- **Sampling Rate:** 10 Hz (550 points per cycle)
- **Base Gas:** Outdoor air (Moscow)
- **Flow Rate:** 0.2 m/s (linear velocity)

### Target Gases
- **Hydrogen (H₂)**
- **Propane (C₃H₈)**

### Concentrations Tested (ppm)
- 30, 50, 100, 200, 350, 400, 450, 470, 550, 800, 950, 1600

## Data Collection Protocol

### Measurement Scheme (12-hour cycles)
Each measurement session lasted 12 hours with alternating gas exposures:
- **Hour 0-1:** Clean air baseline
- **Hour 1-2:** Gas 1 at concentration C₁
- **Hour 2-3:** Clean air recovery
- **Hour 3-4:** Gas 2 at concentration C₂
- **Hour 4-5:** Clean air recovery
- ... (pattern repeats with decreasing concentrations)

### Data Structure

Each measurement cycle contains:
- **550 time points** (60 seconds at 10 Hz, with some overhead)
- **4 sensor readings** (resistance values)
- **4 temperature readings** (actual sensor temperatures)
- **Temperature setpoints** (target temperatures)
- **Metadata:** cycle number, phase, timestamp

## Dataset Statistics

- **Total Cycles:** 17,464 (4,366 per sensor × 4 sensors)
- **March Data:** 2,950 cycles (6 concentrations each gas)
  - Concentrations: 100, 400, 470, 800, 950, 1600 ppm
- **April Data:** 1,416 cycles (6 concentrations each gas)
  - Concentrations: 30, 50, 200, 350, 450, 550 ppm

## Features

### Input Features (550 dimensions)
Each feature represents the sensor resistance at a specific time point during the temperature modulation cycle:
- `Feature 1-50`: Low temperature phase (150°C)
- `Feature 51-450`: Heating ramp (150°C → 500°C)
- `Feature 451-500`: High temperature dwell (500°C)
- `Feature 501-550`: Cooling ramp (500°C → 150°C)

### Target Variables (2 dimensions)
- `y[0]`: Hydrogen concentration (ppm)
- `y[1]`: Propane concentration (ppm)

Note: In clean air cycles, both targets are 0 ppm.

## Data Format

### Raw Data Files (.dat)
Tab-separated values with columns:
- `Time`: Timestamp (seconds)
- `T_set1-4`: Temperature setpoints for 4 sensors (°C)
- `N_cycle`: Cycle number
- `Phase`: Measurement phase indicator
- `T1-4`: Actual temperatures for 4 sensors (°C)
- `R1-4`: Resistance values for 4 sensors (Ohm)

### Processed Data (PyTorch tensors)
- `x.pt`: Input features [N_samples, 550]
- `y.pt`: Target labels [N_samples, 2]

## Data Splits

### March-April Split (Temporal)
- **Training:** March data (2,950 cycles)
- **Validation:** 20% of March data
- **Test:** April data (1,416 cycles)

This split tests model generalization under temporal distribution shift.

### Random Split
- **Training:** 60% of all data
- **Validation:** 20% of all data
- **Test:** 20% of all data

This split tests model performance without distribution shift.

## Data Preprocessing

### Normalization
All features and targets are normalized to [0, 1] range:
```
x_norm = (x - x_min) / (x_max - x_min)
```

Normalization parameters are computed on training set only.

### Quality Control
- Outliers removed based on 3σ criterion
- Cycles with sensor malfunction excluded
- Only stable measurement periods included

## Usage Notes

### Loading Data
```python
import torch

# Load data
x = torch.load('x.pt')  # Shape: [17464, 550]
y = torch.load('y.pt')  # Shape: [17464, 2]
```

### Important Considerations
1. **Sensor Drift:** April data shows sensor drift effects
2. **Atmospheric Conditions:** Outdoor air composition varies
3. **Temperature Effects:** Resistance varies with temperature
4. **Humidity:** Not controlled, affects baseline
5. **Cross-sensitivity:** Sensors respond to other gases in air

## Data Availability

Raw experimental data is not included in this repository due to size limitations (~4GB).

**To request access to raw data:**
- Contact: vityugova.julia@physics.msu.ru
- Include: research purpose and institution affiliation

## Citations

If you use this dataset, please cite:

```bibtex
@thesis{vitiugova2023interpretability,
  title={Neural Network Weight Analysis for Data Processing in Physics},
  author={Vitiugova, Julia M.},
  year={2023},
  school={Lomonosov Moscow State University}
}
```

## Acknowledgments

Data collected at:
- Laboratory of Chemistry and Physics of Semiconductor and Sensor Materials, Chemistry Faculty, Lomonosov Moscow State University
- Under supervision of Dr. V.V. Krivetskiy
