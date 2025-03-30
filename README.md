# AI-Powered Resume Screener

## Overview

This project is an AI-powered resume screening model that predicts resume scores based on various attributes such as experience, salary expectations, and project count. It leverages machine learning techniques to analyze resumes and provide an AI-driven evaluation to assist recruiters in shortlisting candidates efficiently.

GitHub Repository: [AI-Resume-Screener](https://github.com/Meetrao/AI-Resume-Screener)

## Features

- **Automated Resume Scoring**: Predicts an AI score (0-100) based on resume attributes.
- **Data Preprocessing**: Cleans the dataset by removing unnecessary columns and handling missing values.
- **Exploratory Data Analysis (EDA)**: Visualizes key attributes like experience, salary expectations, and AI scores.
- **Machine Learning Model**: Trains a regression model to predict resume scores.
- **Evaluation Metrics**: Assesses model performance using RMSE, MAE, and R-squared.

## Dataset

The dataset used for training contains the following features:

- `Resume_ID`: Unique identifier for each resume.
- `Name`: Candidate's name.
- `Skills`: Key skills extracted from the resume.
- `Experience (Years)`: Total years of experience.
- `Education`: Highest educational qualification.
- `Certifications`: Relevant certifications.
- `Job Role`: Intended job role.
- `Recruiter Decision`: Hire or Reject based on manual screening.
- `Salary Expectation ($)`: Expected salary in dollars.
- `Projects Count`: Number of completed projects.
- `AI Score (0-100)`: AI-predicted resume suitability score.

### How to Use the Dataset

1. **Loading the dataset**:
   ```python
   import pandas as pd
   df = pd.read_csv("AI_Resume_Screening.csv")
   ```

2. **Preprocessing**:
   - Remove unnecessary columns (`Resume_ID`, `Name`)
   - Convert categorical features (`Education`, `Certifications`, `Job Role`) into numerical representations.
   - Extract key skills using NLP techniques.

3. **Feature Engineering**:
   - Calculate the match score between `Skills` and `Job Role`.
   - Scale numerical features (`Experience`, `Salary Expectation`, `Projects Count`).

4. **Model Training**:
   - Use Scikit-learn to train models (Linear Regression, Random Forest, XGBoost).
   - Evaluate performance using RMSE, MAE, and R-squared metrics.

## Technologies Used

The following technologies and libraries were used in this project:

- **Programming Language**: Python
- **Libraries for Data Processing**: Pandas, NumPy
- **Visualization Tools**: Matplotlib, Seaborn
- **Machine Learning Algorithms**: Scikit-learn (Linear Regression, Random Forest, XGBoost)
- **Feature Engineering**: NLP techniques (spaCy for text processing)
- **Data Handling**: Google Colab for training and experimentation

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/Meetrao/AI-Resume-Screener.git
   cd AI-Resume-Screener
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```sh
   jupyter notebook Resume_Screener.ipynb
   ```

## Model Training

The model training pipeline follows these steps:

1. Load and clean the dataset.
2. Perform exploratory data analysis (EDA) using Seaborn and Matplotlib.
3. Encode categorical features and scale numerical ones.
4. Train a regression model using Scikit-learn (tested with Linear Regression, Random Forest, and XGBoost).
5. Evaluate model performance using RMSE, MAE, and R-squared.

## Usage

- Run the Jupyter Notebook to train the model and predict AI scores for new resumes.
- Modify the dataset path if needed to use custom resume data.
- Integrate the trained model into a web-based recruitment platform or automate resume screening in HR workflows.

## Future Improvements

- Enhance the model by incorporating NLP techniques to analyze resume text more effectively.
- Improve feature engineering by extracting more detailed insights from resumes.
- Deploy the model as a web application using Flask or FastAPI.

## Contributing

Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request on [GitHub](https://github.com/Meetrao/AI-Resume-Screener).

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Contact

For any questions or collaborations, feel free to reach out via [LinkedIn](www.linkedin.com/in/meet-rao-a99a00276) or open an issue on [GitHub](https://github.com/Meetrao/AI-Resume-Screener/issues).



