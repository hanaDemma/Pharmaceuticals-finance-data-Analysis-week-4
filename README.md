# Pharmaceuticals Finance Data Analysis - Week 4

**Project Overview:**
This project focuses on forecasting sales across multiple stores in various cities six weeks in advance. The objective is to analyze customer purchasing behavior and evaluate the impact of factors such as promotions, store openings, and other key metrics on sales.

## Setup Environment
The initial step is to set up a Python development environment, integrate version control, and configure CI/CD workflows for continuous integration and deployment.
## Folder Structure 

PHARMACEUTICALS-FINANCE-DATA-ANALYSIS-WORKFLOW/ 
├── .github/
├── .week4/
├── . Data/
├── . flask.api
├── . notebooks/
│   ├── exploratory_analysis.ipynb 
    ├── sales_prediction.ipynb 
│   └── README.md 
├── . scripts/ 
├── . src/
├── .tests/
├── .gitignore 
├── README.md 
└── requirements.txt 

## Deliverables
- Python Environment Setup: A fully configured development environment with all necessary libraries and tools.
- GitHub Repository: Includes appropriate branches for collaboration and version control.
- CI/CD Workflows: Configuration files under .github/workflows/ for continuous integration and deployment.

 ## Exploration of Customer Purchasing Behavior
 ## Objective
    - Analyze how key factors, such as promotions, new store openings, and other operational measures, influence customer purchasing behavior.
## Tasks
- Set up a GitHub repository with a dedicated feature/task-1 branch.
- Perform Exploratory Data Analysis (EDA) using Jupyter Notebook (exploratory_analysis.ipynb).
- Commit progress at least three times daily, with descriptive and clear commit messages.

 ## Logging
 ## Objective
 - Implement logging with Python's logging library to ensure traceability and reproducibility of the analysis steps.

## Prediction of Store Sales
## Objective: 
- Build machine learning models to predict daily store sales six weeks ahead.
## Tasks:
- Set up a GitHub repository with a dedicated feature/task-2 branch.
- Preprocess the data and extract features.
- Build regression models using tree-based algorithms and pipelines.
- Experiment with deep learning models (LSTM) for time series forecasting.

## Model Serving API Call
## Objective: 
- To create REST API to serve the trained machine-learning models for real-time predictions.
## Tasks:
- Set up a GitHub repository with a dedicated task-3 branch.
- Select a suitable framework for building REST APIs
- Load the model
- Define API endpoints and Handle requests
- Return predictions and delpl0ye to web server

## Dependencies
This project relies on the following Python libraries:

- pandas: Data manipulation and analysis
- seaborn: Data visualization.
- scikit-learn: Machine learning tools.
- matplotlib: Plotting and visualization.
- logging: Tracking and documenting process flows.

## Contribution Guidelines
1. Fork the Repository: Clone the main repository to your GitHub account.
2. Create a Feature Branch: Name the branch using the format feature/your-feature.
3. Make Changes: Implement your feature or changes.
4. Commit Regularly: Use clear and descriptive commit messages.
5. Push Your Branch: Push your feature branch to the remote repository.
6. Open a Pull Request (PR): Submit a PR with a detailed description of your changes for review.