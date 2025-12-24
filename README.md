# Apple Watch vs Fitbit: Comprehensive Wearable Data Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**A professional Data Science portfolio project comparing Apple Watch and Fitbit fitness tracking devices using advanced statistical analysis and machine learning.**

[ View Notebooks](#-project-structure) • [ Key Findings](#-key-findings) • [ Quick Start](#-quick-start) • [ Results](#-results)

</div>

---

## Executive Summary

This comprehensive analysis investigates the performance and accuracy of two leading wearable fitness devices - **Apple Watch** and **Fitbit** - using real-world health and activity data. The project demonstrates end-to-end data science workflows including exploratory data analysis, statistical hypothesis testing, and machine learning modeling.

### Business Questions Answered

| Question | Approach | Key Finding |
|----------|----------|-------------|
| Are there significant differences in health metrics between devices? | Statistical Testing (t-test, Mann-Whitney U) | Device-specific variations detected |
| Can we accurately predict calorie expenditure? | Regression Modeling (RF, GB) | R² > 0.90 achievable |
| Can we classify activities from sensor data? | Classification (RF, GB, KNN) | High accuracy activity recognition |
| Which features matter most for predictions? | Feature Importance Analysis | Steps, heart rate, distance are key |

---

## Project Structure

```
Apple & Fitbit Data/
│
├── apple.ipynb # Apple Watch focused analysis (71 cells)
├── fitbit.ipynb # Fitbit focused analysis (45 cells)
├── main.ipynb # Combined cross-device analysis (41 cells)
│
├── Graph Visualization/ # All exported visualizations
│ ├── apple_*.png # 17 Apple Watch visualizations
│ ├── fitbit_*.png # 13 Fitbit visualizations
│ └── main_*.png # 9 Combined analysis visualizations
│
├── README.md # Project documentation
├── ANALYSIS_REPORT.md # Detailed analysis findings
├── requirements.txt # Python dependencies
└── LICENSE # MIT License
```

---

## Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA SCIENCE WORKFLOW │
├─────────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Data Loading │───▶│ Data EDA │───▶│ Feature │ │
│ │ & Validation │ │ & Profiling │ │ Engineering │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │ │
│ ┌───────────────────────────────────────┘ │
│ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Statistical │───▶│ Machine │───▶│ Insights & │ │
│ │ Testing │ │ Learning │ │ Reporting │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle - Apple Watch and Fitbit Data](https://www.kaggle.com/datasets/aleespinosa/apple-watch-and-fitbit-data) |
| **Total Records** | 11,985 observations |
| **Features** | 16 variables |
| **Devices** | Apple Watch, Fitbit |
| **Activities** | Lying, Sitting, Self-Paced Walk, Running 3 METs, Running 5 METs, Running 7 METs |

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Participant age (years) |
| `gender` | Binary | 0 = Female, 1 = Male |
| `height` | Numeric | Height (cm) |
| `weight` | Numeric | Weight (kg) |
| `heart_rate` | Numeric | Heart rate (BPM) |
| `steps` | Numeric | Step count |
| `calories` | Numeric | Calories burned |
| `distance` | Numeric | Distance traveled (km) |
| `entropy_heart` | Numeric | Heart rate variability entropy |
| `entropy_steps` | Numeric | Step pattern entropy |
| `resting_heart` | Numeric | Resting heart rate (BPM) |
| `activity` | Categorical | Activity type |
| `device` | Categorical | Wearable device type |

---

## Key Findings

### Statistical Analysis

| Hypothesis Test | Test Used | Result | Interpretation |
|-----------------|-----------|--------|----------------|
| Heart Rate Difference | Independent t-test | p < 0.05 | Significant difference between devices |
| Steps Difference | Mann-Whitney U | p < 0.05 | Significant difference between devices |
| Calories by Activity | One-way ANOVA | p < 0.001 | Activities significantly affect calorie burn |

### Machine Learning Performance

#### Regression (Calorie Prediction)

| Model | Apple Watch R² | Fitbit R² | Combined R² |
|-------|----------------|-----------|-------------|
| Linear Regression | 0.85+ | 0.84+ | 0.85+ |
| Random Forest | 0.92+ | 0.91+ | 0.92+ |
| Gradient Boosting | 0.93+ | 0.92+ | 0.93+ |

#### Classification (Activity Recognition)

| Model | Apple Watch Acc | Fitbit Acc | Combined Acc |
|-------|-----------------|------------|--------------|
| Logistic Regression | 75%+ | 74%+ | 75%+ |
| Random Forest | 88%+ | 87%+ | 88%+ |
| Gradient Boosting | 89%+ | 88%+ | 89%+ |
| K-Nearest Neighbors | 82%+ | 81%+ | 82%+ |

### Top Predictive Features

1. **Steps** - Highest importance for activity classification
2. **Heart Rate** - Strong predictor for calorie expenditure
3. **Distance** - Key metric for movement-based activities
4. **Entropy Measures** - Signal quality indicators
5. **Demographics** - Moderate importance (age, weight)

---

## Quick Start

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Installation

```bash
# Clone the repository
git clone https://github.com/NusratBegum/Apple-Fitbit-Data-Analysis.git

# Navigate to project directory
cd Apple-Fitbit-Data-Analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Analysis

1. **Start with `apple.ipynb`** - Apple Watch specific analysis
2. **Then `fitbit.ipynb`** - Fitbit specific analysis
3. **Finally `main.ipynb`** - Combined cross-device comparison

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
jupyter>=1.0.0
```

---

## Visualizations Gallery

The project generates **39 professional visualizations** covering:

| Category | Count | Examples |
|----------|-------|----------|
| Demographic Analysis | 8 | Age/Gender distributions, outlier detection |
| Feature Engineering | 6 | Scaling comparisons, transformed features |
| Correlation Analysis | 6 | Heatmaps, feature relationships |
| Activity Patterns | 8 | Box plots, violin plots by activity |
| Hypothesis Testing | 4 | Statistical test visualizations |
| ML Performance | 7 | Regression plots, confusion matrices, feature importance |

All visualizations are saved in the `Graph Visualization/` folder with high-resolution PNG format.

---

## Business Applications

### For Fitness App Developers
- Build device-agnostic algorithms using combined training data
- Use Random Forest or Gradient Boosting for best performance
- Focus on steps, heart rate, and distance as primary features

### For Health Researchers
- Account for device differences in multi-source studies
- Use statistical tests to validate cross-device comparability
- Consider entropy measures for additional signal quality assessment

### For End Users
- Either device provides reliable fitness tracking
- Focus on trends rather than absolute values
- Activity recognition works well across devices

---

## Future Work

- [ ] Time series analysis for temporal patterns
- [ ] Deep learning models (LSTM, CNN) for improved accuracy
- [ ] Real-time prediction pipeline development
- [ ] Additional device integration (Garmin, Samsung, etc.)
- [ ] Mobile app deployment with TensorFlow Lite

---

## References

1. Kaggle Dataset: [Apple Watch and Fitbit Data](https://www.kaggle.com/datasets/aleespinosa/apple-watch-and-fitbit-data)
2. Scikit-learn Documentation: [Machine Learning in Python](https://scikit-learn.org/)
3. Seaborn: [Statistical Data Visualization](https://seaborn.pydata.org/)

---

## Author

**Nusrat Begum**

- GitHub: [@NusratBegum](https://github.com/NusratBegum)
- LinkedIn: [Nusrat Begum](https://linkedin.com/in/nusratbegum)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

If you found this project helpful, please consider:
- Starring this repository
- Reporting issues
- Contributing improvements
- Sharing with others

---

<div align="center">

**Made for the Data Science Community**

*Professional Portfolio Project | 2024*

</div>
