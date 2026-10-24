# volatility_forecasting_system
A python based financial time series analysis project focused on analyzing financial data, identifying trends, and applying statistical models to understand market behavior. The system explores time series forecasting and volatility analysis using financial data to support data-driven insights and financial decision making.

# Financial Time Series Analysis and Volatility Forecasting

## Overview

This project focuses on forecasting the next trading day's volatility of **Commercial Bank of Ceylon PLC (COMB.N0000)** using financial time series analysis, statistical modelling, and machine learning techniques.

The project compares traditional forecasting models with machine learning approaches to evaluate their predictive performance. It also investigates market volatility, temporal patterns, model reliability, and performance across different market conditions using historical stock market data from the Colombo Stock Exchange (CSE).

## Objectives

- Analyze historical stock market data and identify trends and patterns.
- Examine stationarity, autocorrelation, and volatility characteristics.
- Develop and compare statistical and machine learning forecasting models.
- Evaluate model performance using appropriate forecasting metrics.
- Investigate model robustness across calm and turbulent market conditions.
- Generate visualizations, prediction outputs, and a final analytical report.

## Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data processing and numerical analysis
- **Matplotlib & Seaborn** – Data visualization
- **Statsmodels** – Statistical analysis and ARIMA modelling
- **ARCH** – GARCH volatility modelling
- **Scikit-learn** – Machine learning and model evaluation
- **Jupyter Notebook** – Interactive analysis and documentation

## Forecasting Models

The project evaluates the following approaches:

**Baseline Models**
- Naive volatility forecast
- 22-day historical mean

**Traditional Time Series Models**
- ARIMA – Forecasting log-transformed volatility
- GARCH – Modelling conditional volatility

**Machine Learning Models**
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Methodology

1. **Data Preparation:** Load and inspect historical stock data, handle data quality issues, and account for relevant corporate actions.
2. **Exploratory Data Analysis:** Examine price movements, returns, volatility distributions, correlations, and statistical properties.
3. **Feature Engineering:** Construct historical volatility, return-based, and other time-dependent features while avoiding future-data leakage.
4. **Model Development:** Train statistical and machine learning models using historical observations.
5. **Chronological Data Splitting:** Use separate training, validation, and test periods to support reliable model selection and evaluation.
6. **Model Evaluation:** Compare forecasts using metrics such as QLIKE, RMSE, MAE, and other relevant statistical tests.
7. **Robustness Analysis:** Examine forecast performance across different market regimes and assess the influence of unusual market events.

## Dataset

The project uses historical stock market data for Commercial Bank of Ceylon PLC, covering the period from January 2024 to December 2025.

- **Primary security:** COMB.N0000 (voting shares)
- **Robustness analysis:** COMB.X0000 (non-voting shares)
- **Forecast target:** Next-day Parkinson range-based volatility
- **Training period:** Through February 2025
- **Validation period:** March–July 2025
- **Testing period:** August–December 2025

The final test period contains 102 observations. The dataset also includes information on corporate actions, which are considered during data preparation.

## Evaluation and Findings

The models are compared using out-of-sample forecasting performance, with particular emphasis on QLIKE, where lower values indicate better performance.

The analysis found that:

- Ridge Regression achieved the lowest QLIKE among the evaluated models on the main test set.
- All five advanced forecasting models outperformed the two baseline methods.
- Differences between the advanced models were not statistically significant according to the Diebold–Mariano tests.
- Extreme market movements had a substantial impact on forecasting errors.
- Recent volatility and short-term market information were important for forecasting performance.

These findings highlight the importance of appropriate volatility measurement, careful model evaluation, and robustness testing. The forecasts are intended for risk analysis rather than as evidence of a profitable trading strategy.

## Project Structure

```text
financial_time_series_project/
├── data/
│   ├── original/              # Original dataset
│   └── processed/             # Cleaned datasets
├── notebooks/
│   └── financial_time_series_analysis.ipynb
├── src/
│   ├── config.py              # Project configuration
│   ├── data_processing.py     # Data cleaning and preparation
│   ├── feature_engineering.py # Feature creation and data splitting
│   ├── exploratory_analysis.py# Exploratory analysis
│   ├── modelling.py           # Forecasting models
│   ├── evaluation.py          # Model evaluation
│   ├── diagnostics.py         # Diagnostics and comparisons
│   ├── additional_analysis.py # Robustness and regime analysis
│   └── pipeline.py            # Main analysis pipeline
├── outputs/
│   ├── figures/               # Generated visualizations
│   ├── tables/                # Results and comparison tables
│   ├── predictions/           # Forecast outputs
│   └── model_results/         # Model summaries and logs
├── report/
│   ├── final_report.pdf
│   ├── final_report.docx
│   └── findings_summary.md
├── main.py                    # Runs the full analysis
├── check_leakage.py           # Checks for future-data leakage
├── build_notebook.py          # Builds the analysis notebook
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation and Setup

**Prerequisites:** Python 3.10 or later and Git.

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   ```

2. Navigate to the project directory:

   ```bash
   cd financial_time_series_project
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the environment.

   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

Run the complete analysis pipeline:

```bash
python main.py
```

Run the automated leakage check:

```bash
python check_leakage.py
```

Open the Jupyter Notebook for the step-by-step analysis:

```bash
jupyter notebook notebooks/financial_time_series_analysis.ipynb
```

The pipeline generates analysis tables, figures, model predictions, and evaluation results in the `outputs/` directory.

## Limitations

- The analysis is based on a limited historical period and a specific listed company.
- Unexpected market shocks can substantially affect forecast accuracy.
- Model performance may change under different market conditions.
- Forecasting volatility does not guarantee accurate price predictions or profitable trading decisions.

## Disclaimer

This project is developed for academic and analytical purposes. The results are not financial advice or investment recommendations.

## Author

Developed as part of the **CM3023 Group Assignment – Real-World Financial Time Series Analytics**.

