🏭 Anomaly Detection and Fault Classification for Industrial Equipment Monitoring

This project focuses on building a machine learning-based monitoring system for industrial equipment to detect unusual behavior (anomalies) and classify whether a machine is faulty or normal.
It combines the strengths of:

✔ XGBoost – for supervised fault classification
✔ Isolation Forest – for unsupervised anomaly detection

Together, these models provide a hybrid fault-diagnostic solution suitable for real-time industrial environments.

🚀 Project Objective

The goal is to automatically identify early signs of malfunction in machines using historical equipment data.
The system performs:

Anomaly Detection – identify unusual patterns without labels

Fault Classification – predict whether the equipment is faulty

Data Preprocessing & Feature Encoding

Model Evaluation using Confusion Matrix, ROC Curve, Precision, Recall, F1-score

This helps industries achieve:
🔹 Predictive maintenance
🔹 Reduced downtime
🔹 Early fault detection
🔹 Improved safety and reliability

📂 Dataset Overview

The dataset includes both categorical and numerical features such as:

Equipment type

Operating location

Temperature

Pressure

Vibration

Load

Runtime hours

Faulty label (0 = Normal, 1 = Faulty)

Categorical values are encoded using Label Encoding.

🧠 Models Used
🔸 1. Isolation Forest (Unsupervised Anomaly Detection)

Isolation Forest isolates anomalies by randomly selecting features and splitting values.
Anomalies require fewer splits to isolate compared to normal points.

Used for:

Detecting hidden abnormal patterns

Identifying outlier behavior

Cross-validating faulty machine instances

Output:

-1 → anomaly (converted to 1 = faulty)

+1 → normal (converted to 0 = normal)

🔸 2. XGBoost Classifier (Supervised Fault Classification)

XGBoost is a state-of-the-art gradient boosting algorithm known for:

High accuracy

Handling complex relationships

Regularization to prevent overfitting

Efficient computation

Used for:

Learning patterns in faulty vs normal samples

Predicting machine condition

Generating metrics such as precision, recall, and F1-score

🔧 Technical Workflow
1. Data Preprocessing

Handle missing values

Encode categorical features

Standardize numerical features using StandardScaler

2. Train-Test Split

80% training

20% testing

Stratified split to maintain label balance

3. Isolation Forest

Detect anomalies

Convert anomaly labels into binary (0/1) format

Analyze anomaly distribution

4. XGBoost Classification

Train model on processed data

Predict faults

Evaluate performance

📊 Evaluation Metrics

The project evaluates both models using:

Confusion Matrix

Precision, Recall, Accuracy, F1-score

ROC Curve & AUC Score

📈 ROC Curve

Shows how well the model separates faulty vs non-faulty classes across thresholds.

🖥️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn

🧾 Key Results

Isolation Forest successfully detects hidden anomalies in the dataset.

XGBoost provides high classification accuracy for fault prediction.

The hybrid approach improves reliability over using a single model.
