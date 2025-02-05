# Logistic Regression for Crowdfunding Success Prediction

## Overview
This project aims to train a logistic regression model to predict whether a crowdfunding project will be successful based on its characteristics.

## Description
The task is a binary classification problem:
- **Class 0**: Failure
- **Class 1**: Success

The model should return the confidence value that the crowdfunding project will be successful (class 1). This confidence value should be within the range [0, 1].

## Performance Metric
The performance metric for this competition is **average precision**, which is equal to the area under the Precision-Recall (PR) curve.

## Evaluation
Submissions are evaluated based on average precision.

## Submission File
For each ID in the test set, you must provide a model confidence score for the project's success. The submission file should contain a header and follow this format:

## Team Members
- [Aleksey866](https://github.com/Aleksey866)
- [pavel-koleda](https://github.com/pavel-koleda)

## Getting Started
1. Clone the repository:
    ```bash
    git clone https://github.com/pavel-koleda/Crowdfunding-competition-at-school.razinkov.ai.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python main.py
    ```
