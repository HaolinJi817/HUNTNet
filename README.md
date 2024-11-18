# HUNTNet

HUNTNet is a network designed to decouple and re-couple target features from multiple perspectives. The project utilizes advanced techniques for feature transformation and processing, including the use of PVT (Pyramid Vision Transformer) and its enhanced version, PVTv2. The modular approach facilitates effective feature learning by combining different custom components and modules.

## Directory Structure

The main directory structure of the project is as follows:

```
├── README.md           # Project description file
├── HUNTNet             # Main folder
│   ├── lib             # Library folder
│   │   ├── HNet.py     # HUNTNet model
│   │   ├── _init_.py   # Initial
│   │   ├── pvt.py      # Original PVT
│   │   └── pvtv2.py    # Advanced PVTv2
│   ├── mods            # Modules folder
│   │   ├── AGF.py
│   │   ├── DCR.py
│   │   ├── DGT.py
│   │   ├── SFI.py
│   │   └── bricks.py   # All the funcs and classes needed before
│   └── utils           # Data loading module and Loss funcs
└── requirements.txt    # List of dependencies
```

ode Overview

1. HNet.py (HUNTNet Model)

This file defines the core HUNTNet model, incorporating advanced feature manipulation and transformer-based techniques for decoupling and re-coupling target features.

2. pvt.py and pvtv2.py

These files define the original Pyramid Vision Transformer (PVT) and the advanced PVTv2, respectively, which serve as essential components for feature extraction and transformation in HUNTNet.

3. Modules (mods)

The mods folder contains various feature transformation modules:

AGF.py: Implements Adaptive Gradient Fusion, helping to merge feature gradients adaptively.

DCR.py: Handles Decoupled Component Recombination, which is crucial for separating and recombining features.

DGT.py: Defines Dynamic Group Transformation, for adaptive grouping of feature channels.

SFI.py: Responsible for Selective Feature Integration, enabling effective feature aggregation.

bricks.py: Contains commonly used functions and building blocks for model construction.

4. Utils

The utils directory contains utilities for data loading, loss function definitions, and other helper functions required for smooth project execution.

Installation and Usage

Dependencies

Make sure you have the following dependencies installed in your environment:

$ pip install -r requirements.txt

Run the Project

To start training or testing the HUNTNet model, run the following command:

$ python HUNTNet/lib/HNet.py

Ensure that the appropriate configurations are set in the relevant files before running the model.

Contributing

Feel free to submit issues or pull requests to contribute to the project and improve it.

License

This project is licensed under the MIT License. For more details, please refer to the LICENSE file.
