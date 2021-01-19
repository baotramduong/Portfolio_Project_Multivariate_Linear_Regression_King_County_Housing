# Multivariate Linear Regression Machine Learning: 
# King County Housing

This project was completed as part of Flatiron School's Data Science Bootcamp (November 2020)

King County is located in the U.S. state of Washington. The population was 2,252,782 in the 2019 census estimate, making it the most populous county in Washington, and the 12th-most populous in the United States. The county seat is Seattle, also the state's most populous city.

Real estate plays an integral role in the U.S. economy. In 2018, real estate construction contributed $1.15 trillion to the nation's economic output, adding 6.2% to U.S. gross domestic product. Purchasing and selling a house is among the biggest commitments and a great source of income for most people. Therefore, accurate prediction of prices based on other sale data can be a critical tool to assist the buyer/seller in making an informed decision. 

The objective of the project is to perform data visualization techniques to understand the insight of the raw data and subsequently apply machine learning on it. The house prices will be predicted from the various features of residential houses such as square footage of the lot, living space, basement, bedrooms, bathrooms, floors, waterfront, condition, grade, and the location and neighborhood surrounding it. The goal of this project is to create a regression model that are able to accurately estimate the price of the house given the features.

In this report, we will investigate factors associated with home properties. This report gives a comprehensive evaluation on factors influencing the value of a home such as:

* **id** - Unique identified for a house
* **date** - Date house was sold
* **price** - Price is prediction target
* **bedrooms** - Number of Bedrooms/House
* **bathrooms** - Number of bathrooms/bedrooms
* **sqft_living** - Square footage of the home
* **sqft_lot** - Square footage of the lot
* **floors** - Total floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Has been viewed
* **condition** - How good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - Square footage of house apart from basement
* **sqft_basement** - Square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **zipcode** - Zipcode
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - Square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - Square footage of the land lots of the nearest 15 neighbors

After initial EDA to understand the dataset, house prices will be predicted given features of residential houses. The business statements are formulated based on these attributes.

## Business Statement

**Q1.** 

**Q2.** 

**Q3.** 

**Q4.** 

**Q5.** 

## The Deliverables
There are 5 deliverables for this project:

1. A well documented Jupyter Notebook containing any code and comments explaining it.

            - Part I: Data cleaning & preparation from provided data
            
            - Part II: Machine Learning with various models
            
2. An organized README.md file that describes the contents of the repository.

             - README.md: describes the content and organization of content.
             - module1_project_rubric.pdf: describes requirements for this project.
             - Data & Figures folder: contains all provided data from Flatiron and all figures for visualization.
             - Part I: Jupyter notebook
             - Part II: Jupyter notebook
             - Presentation pdf

3. A short PowerPoint presentation (delivered as a PDF export) giving a high-level overview of the methodology used and recommendations for non-technical stakeholders. Can be found in the repository or at: 

4. A Blog Post which can be found at: 

5. A Video Walkthrough of my non-technical presentation, can be found at:

# **Notebook Table of Contents**

## Part I: Data Scrubbing and Preparation

### Methodology:

1. Casting columns to the appropriate data types
2. Identifying and dealing with null and duplicated values appropriately
3. Removing columns that aren't required for modeling
4. Checking for normality with distplot, qqplot
5. Checking for linearity with boxplot, correlation coefficient
6. Removing outliers that are more than 3 standard deviation away from the mean
7. Checking for and dealing with multicollinearity with heatmap
8. Select potential features for modeling
9. Normalizing the continuous variables
10. One hot encoding categorical variables

## Part II: Machine Learning

### Initial Model: 

1. Perform Stepwise Selection and use all features that were selected (20):

            - 'grade', 'water_1.0', 'zip_A', 'zip_C', 'sqft_above', 'sqft_basement', 'zip_B', 'sqft_living15', 'cond_3.0', 'zip_G', 'bedrooms', 'floors_2.0',
            'zip_F', 'cond_5.0', 'zip_E', 'zip_I', 'floors_1.5', 'floors_2.5', 'floors_3.0', 'bathrooms'
            
2. Build Initial Model with the chosen features with Statsmodels
3. Feature ranking with recursive feature elimination
4. Fit Initial Model
5. Test Initial Model
            
6. Validate Initial Model

            - train test split
            
            - RMSE
            
            - Evaluating the effect of train-test split size
            
7. Additional assessment:

            - Recheck for multicollinearity
            
            - Recheck for normality
            
            - Recheck for homoscedasticity
            
            - k-fold cross validation
            
            - Bias-variance tradeoff
            
### Model Refinement I:             

### Model Refinement II:   

## Summary of Findings

## Summary of Actionable Insights

Results suggest that the following factors can be used to predict the value of the house:

##  Future Works
