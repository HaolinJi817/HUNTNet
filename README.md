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

## Code Overview

### 1. main.py

This is the main entry file of the project, responsible for initializing the project and executing the main logic. You can start the project by running this file.

### 2. utils.py

Contains common utility functions such as data processing, logging, etc., to facilitate code reuse.

### 3. data_loader.py

The data loading module, responsible for reading data from files or databases and converting it into a format that the model can use.

### 4. model.py

Defines the structure and training methods of the machine learning model, including training, evaluation, and saving of the model.

## Installation and Usage

### Dependencies

Make sure you have the following dependencies installed in your environment:

```
$ pip install -r requirements.txt
```

### Run the Project

Run the following command to start the project:

```
$ python src/main.py
```

## Contributing

Feel free to submit issues or pull requests to contribute to the project and improve it.

## License

This project is licensed under the MIT License. For more details, please refer to the LICENSE file.
