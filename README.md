# Universal Bank Loan Prediction: k-NN Model

This repository contains the machine learning project that employs k-NN classification to predict if a Universal Bank customer will accept a personal loan offer. 

## Table of Contents
1. [Submission Instructions](#submission-instructions)
2. [Data Description](#data-description)
3. [Scenario](#scenario)
4. [Tasks](#tasks)

## Submission Instructions
- Answers should be written as text or comments within the R codes.
- Compile R scripts to html/word/pdf for submission.
  - Shortcut: `Ctrl + Shift + K` then select "html".
- Submit the html file via Canvas before the deadline.

## Data Description
- Dataset: `Universalbank.csv`.
- Contains data on 5000 bank customers.
- Includes demographics, bank relationship details, and past loan campaign responses.
- Key metric: Only 9.6% accepted the last personal loan offer.

## Scenario
Universal Bank seeks to convert its liability customers to personal loan customers. With a previous campaign seeing a 9% conversion rate, the bank aims to optimize its strategy using k-NN predictions for better-targeted campaigns.

> "Do the right thing; and do it right." - Master Yeoda.

## Tasks
1. **Data Handling**
    - Import and split data: 60% training and 40% validation.
    a. Check variable data types.
    b. Set up kNN with relevant variables.
2. **k-NN Modelling**
    - Exclude ID and ZIP code predictors.
    - Factorize categorical variables.
    - Test with k values: 3, 5, and 7. Select the best k.
3. **Model Assessment**
    - Predict on the validation set using optimal k.
    a. Visualize with an ROC curve.
    b. Gauge model efficacy.
4. **Sample Prediction**
    - Predict loan acceptance for a specific customer profile provided.

