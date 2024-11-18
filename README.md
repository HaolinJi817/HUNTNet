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

## 🧑‍💻 Code Overview

### 1. 📄 README.md
This file provides a comprehensive description of the project, including the purpose, features, and setup instructions.

### 2. 🗂️ HUNTNet
This is the main folder containing the core of the project.

- **📚 lib**: Contains the library files for the HUNTNet model.
  -  HNet.py: Implements the HUNTNet model.
  -  _init_.py: Initialization file for the library module.
  -  pvt.py: Defines the original PVT (Pyramid Vision Transformer).
  -  pvtv2.py: Implements the advanced version, PVTv2.

- **🧩 mods**: Contains modules that provide additional functionalities.
  -  AGF.py: Anisotropic Gradient Fusion module.
  -  DCR.py: Dual-Channel Recursive module.
  -  DGT.py: Discrete-Wavelet Gabor Transform module.
  -  SFI.py: Selective Feature Integration module.
  -  bricks.py: Contains all the necessary functions and classes used across the modules.

- **🛠️ utils**: Provides utility functions for data loading and loss calculations.

### 3. 📋 requirements.txt
Lists all the dependencies required for the project to run. You can install them using pip.

## 📥 Installation and Usage

### 📦 Dependencies

Make sure you have the following dependencies installed in your environment:

```
$ pip install -r requirements.txt
```

## 🤝 Contributing

Feel free to submit issues or pull requests to contribute to the project and improve it.

## 📜 License

This project is licensed under the MIT License. For more details, please refer to the LICENSE file.

