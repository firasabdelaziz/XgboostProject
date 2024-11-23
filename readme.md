This project implements an XGBoost regression model for predicting target variable 'y'.

## Project Structure
```
├── data/               # Data directory (not tracked by git)
│   └── .gitkeep       # Empty file to track directory
├── src/               # Source code
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/            # Saved models (not tracked by git)
│   └── .gitkeep      # Empty file to track directory
├── notebooks/         # Jupyter notebooks
│   └── .gitkeep      # Empty file to track directory
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore file
└── README.md         # Project documentation
```

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your dataset in the `data/` directory

## Usage
1. Place your CSV file in the `data/` directory
2. Run the training script:
   ```bash
   python src/train_model.py
   ```

## Requirements
See requirements.txt for full list of dependencies.
