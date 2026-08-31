# Galaxy-Structural-Parameters
**GALNET** is a Convolutional Neural Network designed to measure galaxy structural parameters.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aiswarya-unni/Galaxy-Structural-Parameters.git
cd Galaxy-Structural-Parameters/galnet
```

### 2. Create the Conda environment

Create a new Conda environment named `galnet`:

```bash
conda create -n galnet python=3.10
```

Activate the environment:

```bash
conda activate galnet
```

### 3. Install the required Python packages

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

You can check that the environment is active with:

```bash
conda env list
```

The `galnet` environment should be marked as active.

## Repository Structure

```text
Galaxy-Structural-Parameters/
└── galnet/
    ├── GaLNet/
    │   ├── cmodel.py
    │   ├── train.py
    │   ├── plot_test.py
    │   ├── pred_test.py
    │   ├── fig/
    │   └── result/
    ├── model/
    │   └── weights.h5
    ├── galaxy.py
    ├── requirements.txt
    └── .gitignore
```
## Dataset

The dataset required to run GALNET is available from Google Drive:

[Download the GALNET dataset from Google Drive](https://drive.google.com/file/d/15OaGl6HFyipI5PyuIno9xNSY31p1noSA/view?usp=sharing&utm_source=chatgpt.com)

### 1. Download the dataset

The dataset can be downloaded using `gdown`:

```bash
pip install gdown
```

From the repository root, run:

```bash
gdown "https://drive.google.com/uc?id=15OaGl6HFyipI5PyuIno9xNSY31p1noSA"
```

Extract the downloaded archive into the `galnet/data/` directory.

### 2. Dataset directory structure

The `data/` directory contains the real background data, PSFs, and the generated training, validation, and test datasets.

```text
galnet/
└── data/
    ├── real/
    │   ├── image/
    │   │   └── *.fits
    │   ├── psf/
    │   │   └── *.fits
    │   └── background.csv
    │
    ├── train/
    │   └── *.fits
    ├── validation/
    │   └── *.fits
    ├── test/
    │   └── *.fits
    │
    ├── train_data.csv
    ├── validation_data.csv
    └── test_data.csv
```

#### `real/`

The `real/` directory contains the real background observations and point-spread functions (PSFs) used to generate the simulated galaxy images.

* `image/` — real background images in FITS format.
* `psf/` — PSF images corresponding to the background observations.
* `background.csv` — catalog linking the background images with their corresponding PSFs.

#### Generated datasets

The following directories contain the simulated galaxy images generated using `galaxy.py`:

* `train/` — training dataset.
* `validation/` — validation dataset.
* `test/` — test dataset.

The corresponding CSV files contain the parameters associated with each simulated galaxy:

* `train_data.csv` — parameters for the training dataset.
* `validation_data.csv` — parameters for the validation dataset.
* `test_data.csv` — parameters for the test dataset.

The CSV files include the galaxy name, magnitude, position, effective radius, position angle, axis ratio, Sérsic index, and SNR.

## Usage

GALNET follows a three-stage workflow: **dataset generation → model training → prediction and visualization**.

### 1. Generate the dataset

First, run `galaxy.py` to generate the simulated galaxy datasets for training, validation, and testing.

From the `galnet` directory:

```bash
python galaxy.py
```

The `galaxy.py` script generates the simulated galaxy datasets using the real background images and PSFs in `data/real/`.

The dataset mode is controlled by the `mode` parameter in `galaxy.py`:

```python
mode = "train"        # Training dataset
mode = ""   # Validation dataset
mode = "test"         # Test dataset
```

For example, to generate the training dataset, set:

```python
mode = "train"
```

and run:

```bash
python galaxy.py
```

Similarly, set `mode = ""` or `mode = "test"` to generate the corresponding datasets.


### 2. Train the GALNET model

After generating the dataset, train the convolutional neural network using:

```bash
python GaLNet/train.py
```

The trained model weights are saved in the `model/` directory.

### 3. Make predictions

Run the prediction script to estimate the galaxy structural parameters for the test dataset:

```bash
python GaLNet/pred_test.py
```

The predicted structural parameters are saved as a CSV file in:

```text
GaLNet/result/
```

### 4. Generate plots

To visualize the predicted parameters, run:

```bash
python GaLNet/plot_test.py
```

The resulting plots are saved in:

```text
GaLNet/fig/
```


The outputs are organized as follows:

```text
galnet/
├── model/
│   └── weights.h5
│
└── GaLNet/
    ├── result/
    │   └── pred_test_para.csv
    │
    └── fig/
        ├── test_R_eff.png
        ├── test_e1.png
        ├── test_e2.png
        ├── test_mag.png
        ├── test_n.png
        ├── test_x.png
        └── test_y.png
```

