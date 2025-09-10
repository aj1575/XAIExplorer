# XAI Explorer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive Streamlit dashboard for machine learning model interpretability using SHAP (SHapley Additive exPlanations).

## Features

- CSV dataset upload and automatic preprocessing
- Auto-detection of classification/regression tasks
- Random Forest model training
- Global feature importance via SHAP beeswarm plots
- Local explanations via SHAP waterfall plots
- Export visualizations as PNG
- Support for numerical and categorical features

## Installation

```bash
git clone https://github.com/your-username/XAI-Explorer.git
cd XAI-Explorer
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

```
streamlit>=1.28.0
shap>=0.42.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Usage

1. Upload CSV dataset via Streamlit interface
2. Select target column and specify task type
3. Model trains automatically on preprocessed data
4. View global SHAP importance and local explanations
5. Export visualizations for analysis

## Architecture

- **Frontend**: Streamlit web interface
- **ML Engine**: scikit-learn Random Forest
- **XAI**: SHAP for model interpretability
- **Data Processing**: pandas/numpy pipeline
- **Visualization**: matplotlib/SHAP plots

## Roadmap

- Support for XGBoost, LightGBM, CatBoost
- Pre-trained model import (.pkl/.joblib)
- Model evaluation metrics
- Batch processing capabilities

## Contributing

Fork → Branch → Commit → Push → Pull Request

## License

MIT License - see [LICENSE](LICENSE) file.
