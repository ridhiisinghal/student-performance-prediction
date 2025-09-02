# student-performance-prediction
## Student Performance Prediction Machine Learning Model
*Document Version:* 1.0  
*Date:* 02/09/2025  
*Prepared by:* Bhavya Bhardwaj, Ridhi Singhal 
*Project:* Student Performance Prediction ML Model - Educational Tool  
---
## Table of Contents
1. [Introduction](#1-introduction)  
   - 1.1 [Purpose](#11-purpose)  
   - 1.2 [Document Scope](#12-document-scope)  
   - 1.3 [Target Users](#13-target-users)  
   - 1.4 [Definitions and Terms](#14-definitions-and-terms)  
   - 1.5 [References](#15-references)  
2. [Overall Description](#2-overall-description)  
   - 2.1 [Product Overview](#21-product-overview)  
   - 2.2 [Main Functions](#22-main-functions)  
   - 2.3 [User Types](#23-user-types)  
   - 2.4 [System Environment](#24-system-environment)  
   - 2.5 [Assumptions](#25-assumptions)  
3. [System Requirements](#3-system-requirements)  
   - 3.1 [User Interface](#31-user-interface)  
   - 3.2 [Software Requirements](#32-software-requirements)  
   - 3.3 [Hardware Requirements](#33-hardware-requirements)  
4. [System Features](#4-system-features)  
   - 4.1 [Data Loading](#41-data-loading)  
   - 4.2 [Data Analysis](#42-data-analysis)  
   - 4.3 [Data Preparation](#43-data-preparation)  
   - 4.4 [Model Training](#44-model-training)  
   - 4.5 [Model Testing](#45-model-testing)  
   - 4.6 [Results Display](#46-results-display)  
5. [Performance Requirements](#5-performance-requirements)  
6. [Other Requirements](#6-other-requirements)  
---
## 1. Introduction
### 1.1 Purpose
This document describes a machine learning project to predict student performance using demographic, social, and academic features. The system is built for learning purposes and helps students understand how to use ML algorithms for regression and classification problems.
The project uses the UCI Student Performance dataset and implements algorithms for both regression (predicting final grade) and classification (predicting pass/fail or grade categories).
### 1.2 Document Scope
This project covers:  
- Loading student performance data from UCI dataset  
- Basic data analysis and visualization  
- Training different ML models for regression and classification  
- Testing model performance  
- Comparing results between models  
*Not Included:*  
- Real-time prediction system  
- Advanced ML techniques  
- Deployment for actual educational use  
- Complex data processing  
### 1.3 Target Users
*Students:* Learning machine learning basics  
- Need: Simple examples and clear explanations  
- Skills: Basic Python knowledge  
*Teachers:* Using for ML education  
- Need: Easy to understand and modify  
- Skills: Good programming knowledge  
### 1.4 Definitions and Terms
| Term               | Meaning                                           |
|--------------------|--------------------------------------------------|
| *ML*             | Machine Learning                                  |
| *Regression*     | Predicting continuous values (e.g., final grade) |
| *Classification* | Predicting categories (e.g., pass/fail)          |
| *Training Data*  | Data used to teach the model                       |
| *Test Data*      | Data used to check how well the model works       |
| *Accuracy*       | How often the model makes correct predictions (classification) |
| *MSE*            | Mean Squared Error (regression)                  |
| *R2 Score*       | Coefficient of determination (regression)        |
| *Precision*      | How many positive predictions are actually correct (classification)|
| *Recall*         | How many actual positives the model finds (classification)         |
| *F1-Score*       | Combines precision and recall into one number (classification)     |
| *Confusion Matrix*| Table showing correct and wrong predictions (classification)       |
### 1.5 References
1. *UCI Student Performance Dataset*  
   Student Performance Data Set. UCI Machine Learning Repository.
2. *Python Libraries*  
   - Pandas for data handling  
   - NumPy for calculations  
   - Matplotlib for charts  
   - Scikit-learn for ML algorithms  
---
## 2. Overall Description
### 2.1 Product Overview
This is an educational tool built as a Jupyter notebook. It teaches students how to:  
- Work with student performance data  
- Apply ML algorithms for regression and classification  
- Compare different models  
- Understand evaluation metrics for both regression and classification  
The system is standalone and doesn't connect to other educational systems. It's only for learning, not for actual educational decision making.
### 2.2 Main Functions
The system does these main tasks:
*Data Handling*  
- Loads student performance dataset  
- Cleans missing data  
- Prepares data for ML models  
*Model Building*  
- Trains regression models (e.g., Linear Regression, Random Forest Regression)  
- Trains classification models (e.g., Logistic Regression, Decision Tree, Random Forest)  
*Results Analysis*  
- Tests each model on new data  
- Calculates appropriate evaluation metrics  
- Creates comparison charts  
- Shows confusion matrices for classification  
### 2.3 User Types
*Primary Users - Students*  
- Learning ML for the first time  
- Want to see how algorithms work  
- Need step-by-step explanations  
*Secondary Users - Instructors*  
- Teaching ML concepts  
- Need working examples for classes  
- May modify code for assignments  
### 2.4 System Environment
*Software Needed:*  
- Python 3.7 or newer  
- Jupyter Notebook  
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn  
*Hardware Needed:*  
- Any regular computer with 2GB RAM  
- 500MB free disk space  
- Internet connection (only for first-time setup)  
*Operating Systems:*  
- Windows 10+  
- Mac OS 10.14+  
- Linux (Ubuntu 18.04+)  
### 2.5 Assumptions
*What We Assume:*  
- Users know basic Python programming  
- Users can install Python packages  
- Dataset remains available online  
- Learning is more important than perfect accuracy  
- Simple approach is better than complex methods  
---
## 3. System Requirements
### 3.1 User Interface
*Jupyter Notebook Interface:*  
- Standard notebook with code and text cells  
- Users click "Run" to execute each step  
- Results appear below each code section  
- All charts and tables display in the notebook  
### 3.2 Software Requirements
*Required Python Packages:*  
- pandas >= 1.2.0 (for data handling)  
- numpy >= 1.19.0 (for calculations)  
- scikit-learn >= 0.24.0 (for ML models)  
- matplotlib >= 3.3.0 (for basic charts)  
- seaborn >= 0.11.0 (for better visualizations)  
### 3.3 Hardware Requirements
*Minimum Specs:*  
- 2GB RAM  
- 500MB free disk space  
- Standard processor (Intel/AMD)  
- No special graphics card needed  
---
## 4. System Features
### 4.1 Data Loading
*What it does:* Gets the student performance data and loads it into the system
*Requirements:*  
- Load UCI student performance dataset  
- Show basic info about the data (rows, columns)  
- Display first few rows to user  
- Handle download errors gracefully  
### 4.2 Data Analysis  
*What it does:* Explores the data to understand patterns
*Requirements:*  
- Show statistics for all features (mean, median, etc.)  
- Create charts showing data distribution  
- Display correlation between different features  
- Show distribution of target variable (grades)  
### 4.3 Data Preparation
*What it does:* Cleans and prepares data for ML models
*Requirements:*  
- Fill in any missing values  
- Convert text categories to numbers  
- Split data into training set (80%) and test set (20%)  
- For classification, create appropriate categories from grades (e.g., pass/fail)  
### 4.4 Model Training
*What it does:* Teaches different algorithms using the training data
*Requirements:*  
- Train regression models to predict final grade  
- Train classification models to predict pass/fail or grade categories  
- Complete all training within 2 minutes  
### 4.5 Model Testing
*What it does:* Tests how well each model works on new data
*Requirements:*  
- Test all models on the same test data  
- For regression: Calculate MSE, R2 score  
- For classification: Calculate accuracy, precision, recall, F1-score  
- Create confusion matrix for classification models  
### 4.6 Results Display
*What it does:* Shows results in easy-to-understand format
*Requirements:*  
- Display performance table comparing all models  
- Show confusion matrix charts for classification  
- Create bar charts comparing model performance  
- Provide simple explanation of which model works best  
---
## 5. Performance Requirements
*Speed Requirements:*  
- Complete notebook execution in under 5 minutes  
- Data loading within 1 minute  
- All model training within 2 minutes  
- Charts and results within 1 minute  
*Accuracy Requirements:*  
- Regression models should achieve reasonable R2 score (e.g., > 0.5)  
- Classification models should achieve at least 65% accuracy  
- Results must be reproducible (same results each time)  
- Memory usage should not exceed 1GB  
---
## 6. Other Requirements
*Educational Requirements:*  
- Include clear explanations between code sections  
- Use simple language in all documentation  
- Provide interpretation of all results  
- Include warnings that this is for learning only, not for educational decision making  
*Legal Requirements:*  
- State clearly this is not for actual educational use  
- Give credit to UCI for the dataset  
- Follow all open-source software licenses  
- Include disclaimer about educational use only  
*Data Requirements:*  
- Use only the provided UCI dataset  
- No personal or private student information  
- All data processing happens locally on user's computer  
- No data is sent anywhere else  
---
## Success Criteria
The project is successful if:  
- Regression models achieve reasonable performance (R2 > 0.5)  
- Classification models achieve over 65% accuracy  
- Notebook runs completely without errors  
- Results are clear and easy to understand  
- Students can learn basic ML concepts from it  
- Code is well-documented and explained
