# MediRisk AI - Intelligent Patient Risk Assessment System

## Project Overview
MediRisk AI is a sophisticated healthcare analytics application designed to predict patient health risks based on clinical data. The system utilizes machine learning algorithms to analyze key patient vitals and demographics, providing an instantaneous risk assessment score and classification. The application features a professional, medical-grade interface suitable for clinical environments.

## Features
- **Intelligent Risk Calculation**: Uses a weighted clinical formula considering Age, BMI, Blood Pressure, Cholesterol, Glucose, and existing conditions.
- **Machine Learning Integration**:
  - **Linear Regression**: Predicts a continuous risk score (0-100).
  - **Logistic Regression**: Classifies patients into Low, Medium, or High risk categories.
- **Real-Time Data Processing**: The application retrains its models on the latest dataset every time it is launched, ensuring predictions are based on the most current data.
- **Professional User Interface**:
  - Dark Mode Medical Theme for reduced eye strain.
  - Compact, grid-based layout for efficient data entry.
  - Clear, high-contrast result visualization.
- **Interactive Analysis**: Instant feedback with visual indicators for risk levels.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/tnshgarg/genai_project_3_healthcare.git
    cd genai_project_3_healthcare
    ```

2.  **Install Dependencies**
    Install the required Python packages using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Dataset**
    Ensure that `synthetic_clinical_dataset.csv` is present in the project root directory. This file is required for model training.

## Usage

### Running the Application
To start the application, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

### Using the Interface
1.  **Patient Vitals**: Enter the patient's information in the "Patient Vitals" section.
    - **Age**: Patient age in years.
    - **Gender**: Biological sex.
    - **BMI**: Body Mass Index.
    - **Systolic BP**: Systolic Blood Pressure (mm Hg).
    - **Diastolic BP**: Diastolic Blood Pressure (mm Hg).
    - **Glucose**: Blood Glucose Level (mg/dL).
    - **Cholesterol**: Total Cholesterol (mg/dL).
    - **Diabetes**: Diagnosis status (Yes/No).
    - **Hypertension**: Diagnosis status (Yes/No).

2.  **Analyze**: Click the **CALCULATE RISK ASSESSMENT** button.

3.  **View Results**: The "Analysis Report" section will appear below, displaying:
    - **Risk Score**: A numerical value representing the calculated health risk.
    - **Risk Level**: A categorical assessment (Low, Medium, High).
    - **Recommendation**: A brief medical recommendation based on the risk level.

## Technical Implementation

### Data Pipeline
1.  **Data Loading**: The application loads clinical data from `synthetic_clinical_dataset.csv`.
2.  **Preprocessing**:
    - Renames columns to standard formats (Age, Gender, BMI, etc.).
    - Encodes categorical variables (Gender: Male/Female -> 0/1).
    - Handles missing values (if any) and standardizes numerical features using `StandardScaler`.

### Risk Scoring Logic
The system calculates a "Synthetic Risk Score" to serve as a ground truth for training. The formula is based on medical risk factors:
- Age (normalized against 90 years)
- BMI (normalized against 40)
- Systolic BP (normalized against 180)
- Cholesterol and Glucose levels
- Weighted penalties for Diabetes and Hypertension

### Machine Learning Models
The application employs two distinct models from `scikit-learn`:

1.  **Linear Regression**:
    - **Input**: Scaled patient features.
    - **Target**: The calculated continuous Risk Score.
    - **Purpose**: To provide a precise numerical risk estimation.

2.  **Logistic Regression (Multinomial)**:
    - **Input**: Scaled patient features.
    - **Target**: Risk Level categories (Low < 40, Medium 40-70, High > 70).
    - **Purpose**: To provide a clear categorical classification for quick decision-making.

### Project Structure
- `app.py`: The main entry point containing the Streamlit application logic, UI code, and model training pipeline.
- `requirements.txt`: detailed list of Python dependencies.
- `synthetic_clinical_dataset.csv`: The dataset used for training the models.
