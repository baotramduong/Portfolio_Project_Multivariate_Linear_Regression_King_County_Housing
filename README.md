# Multivariate Linear Regression Machine Learning: 
# King County Housing

This project was completed as part of Flatiron School's Data Science Bootcamp (November 2020)

King County is located in the U.S. state of Washington. The population was 2,252,782 in the 2019 census estimate, making it the most populous county in Washington, and the 12th-most populous in the United States. The county seat is Seattle, also the state's most populous city.

Real estate plays an integral role in the U.S. economy. Purchasing and selling a house is among the biggest commitments and a great source of income for most people. Therefore, accurate prediction of prices based on other sale data can be a critical tool to assist the buyer/seller in making an informed decision. 

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

**Q1. What are the most predictive features to predict the price of a home?** 

**Q2. How to increase the value of a home?** 

**Q3. How do age and condition affect the value of a home?** 

## Methodology
(1) Perform exploratory data analysis to understand the insight of the data. 
(2) Create the best prediction model with highest accuracy that is able to accurately estimate the price of the house given the features.
**Outcome:** By cross-referencing our initial EDA and the model coefficients, we not only help you to predict the price of house accurately but also give you insights on what to look for when buying a new home or what to do to improve your current home’s value.

## The Deliverables
There are 5 deliverables for this project:

1. A well documented Jupyter Notebook containing any code and comments explaining it.

            - Part I: Data cleaning & preparation from provided data
            
            - Part IIA: Multivariate Linear Regression
            
            - Part IIB: Multivariate Linear Regression with Log Transformed Features
            
2. An organized README.md file that describes the contents of the repository.

             - README.md: describes the content and organization of content.
             - module1_project_rubric.pdf: describes requirements for this project.
             - Data & Figures folder: contains all provided data from Flatiron and all figures for visualization.
             - Part I: Jupyter notebook
             - Part IIA: Jupyter notebook
             - Part IIB: Jupyter notebook
             - Presentation pdf

3. A short PowerPoint presentation (delivered as a PDF export) giving a high-level overview of the methodology used and recommendations for non-technical stakeholders. Can be found in the repository or at: 

https://github.com/baotramduong/Portfolio_Project_King_County_Housing_Linear_Regression/blob/main/Presentation.pdf

4. A Blog Post which can be found at: 

https://baotramduong.medium.com/data-science-vs-the-real-estate-in-king-county-washington-5d8798345543

5. A Video Walkthrough of my non-technical presentation, can be found at:

https://youtu.be/gSFTlAzCyJU

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

## Part IIA + B: Machine Learning

### Methodology:

1. Perform Stepwise Selection to select for features with p-value < 0.05
2. Build Model with the chosen features with Statsmodels
3. Fit Model

            - Get intercept
            
            - Get coefficients
            
4. Test Model

            - Recheck for multicollinearity: calculate variance inflation factor
            
            - Recheck for multicollinearity: heatmap
            
            - Recheck for normality
            
            - Recheck for homoscedasticity
            
            - k-fold cross validation
            
            - Bias-variance tradeoff
            
5. Validate Model

            - train-test split
            
            - Calculate RMSE
            
            - Calculate accuracy percentage
                       
### Models Summary:  

|      |Model  |Description                                         |Num Features|r2                |Accuracy          |RMSE Train         |RMSE Test         |Bias Train             |Bias Test           |Variance Train     |Variance Test      |Cross Validation    |Multicollinearity|Normality|Homoscedasticity|
|------|-------|----------------------------------------------------|------------|------------------|------------------|-------------------|------------------|-----------------------|--------------------|-------------------|-------------------|--------------------|-----------------|---------|----------------|
|0     |Model A|All features                                        |19.0        |0.668176586782782 |66.23671804132353 |0.5625265909631906 |0.6286713292932755|0.18664386951241832    |0.17987848096081832 |0.2106811199621985 |0.20526058148544088|-0.5777992863541213 |P                |F        |F               |
|1     |Model B|All features, outliers removed, RFE                 |10.0        |0.6428213682919023|63.983659507131044|0.5357511479265694 |0.5360749751223352|0.16968353025916813    |0.16323940425182504 |0.20843375904757566|0.20986246274746145|-0.5366122415580049 |P                |F        |F               |
|2     |Model C|All features + Polynomial Regression                |19.0        |0.5856250064884156|56.96494155922818 |0.5749447176169188 |0.5859855961130669|-1.2273176280532736    |-1.227659155686322  |0.15436160392357512|0.15529662614985185|-0.5795136359817207 |P                |F        |F               |
|3     |Model D|All features + Log(X)                               |17.0        |0.5786976703588912|55.90454202123855 |0.6308839853685854 |0.7184528231831934|0.21713773822792934    |0.2103143826228909  |0.17234278299139733|0.16989302996144184|-0.6485411489186145 |P                |F        |F               |
|4     |Model E|Log(y) + All features                               |22.0        |0.7615462339050724|77.32524593692841 |0.4905295423544544 |0.4796703399125017|-0.12455373480726124   |-0.12017065611812015|0.2994754031084041 |0.2935999238334316 |-0.4909644300990439 |P                |P        |P               |
|5     |Model F|Log(y) + All features + Interactions                |20.0        |0.7499807641356179|76.08578900521917 |0.5019198145148942 |0.4926058904441193|-0.07923648663140193   |-0.07484165802857615|0.3499226643087464 |0.3428209413708258 |-0.5023962607155223 |P                |P        |P               |
|6     |Model 1|Log(y) + Log(X) + All features - location           |13.0        |0.5836830428706551|59.65668383976206 |0.6466107231291456 |0.6398193484246292|-3.9433131518899115e-16|0.00911038433601925 |0.5781577193413534 |0.5913110018000954 |-0.6467113295317628 |P                |P        |P               |
|7     |Model 2|Log(y) + Log(X) + All features + location           |21.0        |0.7633737831630929|77.24994371110951 |0.487978559154872  |0.480466165641546 |0.2501444540518847     |0.25340373302649827 |0.31946788072047844|0.31271433229954265|-0.4887252849455389 |P                |P        |P               |
|8     |Model 3|Log(y) + Log(X) + All features + RFE                |10.0        |0.7109133385279477|72.15442146945603 |0.5391972190467731 |0.5315568205243968|0.28483704014726396    |0.28937875187100953 |0.36822059007262603|0.3632359526171232 |-0.5384292665405437 |P                |P        |P               |
|9     |Model 4|Log(y) + Log(X) + All features + Interactions       |26.0        |0.7748366377531093|78.39340167350102 |0.47613388193365636|0.4682359681407142|0.25786267266863844    |0.261864508898474   |0.29539069660865913|0.2889758964173091 |-0.47773084140523353|P                |P        |P               |
|10    |Model 5|Log(y) + Log(X) + All features + Interactions + Poly|30.0        |0.7605799741842636|76.92426165296007 |0.4907229623586382 |0.4838930397337942|0.29151255720572156    |0.296302811271894   |0.319691816600144  |0.31317464983812504|-0.4925347063609607 |P                |P        |P               |

## **BEST MODELS:** 

            MODEL 4 Best model in terms of r2, accuracy, RMSE
            
<img src = '../main/Data & Figures/model_4_coefficients.png' />

            MODEL E Best model in terms of simplicity and interpretability

<img src = '../main/Data & Figures/model_E_coefficients.png' />

## MODEL 4 SUMMARY 
<img src = '../main/Data & Figures/model_4_ols.png' />

<img src = '../main/Data & Figures/model_4_recursive_features_elimination.png' />

<img src = '../main/Data & Figures/model_4_multicollinearity_check.png' />

<img src = '../main/Data & Figures/model_4_residuals_qqplot.png' />

<img src = '../main/Data & Figures/model_4_homoscedasticity_regplot.png' />

<img src = '../main/Data & Figures/model_4_residuals_regplot.png' />

<img src = '../main/Data & Figures/model_4_predictions_regplot.png' />

## Summary of Findings

### **'sqft_living'**

<img src = '../main/Data & Figures/sqft_living_vs_price_lmplot.png' />

            * 'sqft_living' is strongly & positively correlated with target 'price'.
            * The higher the square footage of living space, the higher the price.

### **'sqft_lot'**

<img src = '../main/Data & Figures/sqft_lot_vs_price_lmplot.png' />

            * 'sqft_lot' is weakly & positively correlated to 'price'
            * Higher 'sqft_lot' does not equal to higher price

### **'sqft_above'**

<img src = '../main/Data & Figures/sqft_above_vs_price_lmplot.png' />

            * 'sqft_above' is strongly & positively correlated to 'price'
            * The higher the 'sqft_above' the higher the price

### **'sqft_living15'**

<img src = '../main/Data & Figures/sqft_living15_vs_price_lmplot.png' />

            * 'sqft_living15' is strongly & positively correlated with 'price'.
            * The higher the square footage of the nearest 15 neighbor houses, the higher the price for a house.
            * This demonstrates that neighborhood/location is a value-adding feature when predict the price of a home.

### **'sqft_lot15'**

<img src = '../main/Data & Figures/sqft_lot15_vs_price_lmplot.png' />

            * Similar to 'sqft_lot', 'sqft_lot15' is weakly & positively correlated to 'price'
            * There is a positive correlation between 'sqft_lot15' and 'price'

### **'bedrooms'**

<img src = '../main/Data & Figures/bedrooms_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/bedrooms_vs_price_sqft_living_relplot.png' />

            * 'bedrooms' is positively correlated with 'price'.
            * Higher number of bedrooms stops mattering if 'sqft_living' or 'sqft_above' is small.
            * Too many bedrooms to crowd square footage of the home will have less value.

### **'bathrooms'**

<img src = '../main/Data & Figures/bathrooms_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/bathrooms_vs_price_sqft_living_relplot.png' />

            * 'bathrooms' is highly and positively correlated with 'price'
            * Higher number of bathrooms does not matter if 'sqft_living' or 'sqft_above' is low
            * Too many bathrooms crowding square footage of the home will have less value.
            * 'Penalty' of having too many 'bathrooms' is less severe than having too many 'bedrooms'
### **'floors'**

<img src = '../main/Data & Figures/floors_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/floors_vs_price_sqft_living_relplot.png' />

            * 'floors' is positively correlated to 'price'.
            * Higher number of floors can add value to houses that have smaller square footage.
            * Higher number of floors doesn't add more value to houses that have big square footage.
            * Higher number of floors with small square footage decreases the value of a home.
            * 2.5 floors is ideal to have, more than that is unnecessary.

### **'basement'**

<img src = '../main/Data & Figures/basement_vs_price_catplot.png' />

            * There are more houses without a basement than with a basement.
            * The presence of a basement increases the price of a house but not always: there are houses without a basement still make to Above Median price and there are houses with a basement stay behind in Below Median price.
            * 'basement' is weakly & positively correlated to 'price'.

### **'waterfront'**

<img src = '../main/Data & Figures/waterfront_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/waterfront_vs_price_catplot.png' />

<img src = '../main/Data & Figures/sqft_above_waterfront_vs_price_relplot.png' />

<img src = '../main/Data & Figures/zipcat_waterfront_vs_price_relplot.png' />

            * 'waterfront' is positively correlated to 'price'.
            * There are houses without a waterfront make it into Above Median price but with waterfront, a house is guaranteed to be Above Median.
            * A house with waterfront is valued more highly compared to other houses with the same square footage but without a waterfront.
            * In all zipcode area, the most valued houses have waterfront views.
            
### **'grade'**

<img src = '../main/Data & Figures/grade_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/grade_vs_price_sqft_living_relplot.png' />

            * 'grade' is strongly and positively correlated with 'price'.
            * The higher the grade, the higher the value of a home.
            * To get above the price median, a home needs to be at least grade 10.
            * There is also the 'sqft_living' and 'sqft_above' effect: the higher the square footage, the higher the grade.
            * Smaller square footage houses need at least grade 7 to get past the price median.

### **'condition'**

<img src = '../main/Data & Figures/condition_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/condition_vs_price_sqft_living_relplot.png' />

<img src = '../main/Data & Figures/condition_vs_price_grade_relplot.png' />

            * 'condition' is weakly and positively correlated to 'price'.
            * 'condition' of at least 3 is needed to raise value of a home.
            * a low 'condition' score decreases the value of a home even if that home has high square footage.
            * High 'grade' does not matter if 'condition' is low.

### **'age'**

<img src = '../main/Data & Figures/yr_built_vs_price_relplot.png' />

<img src = '../main/Data & Figures/age_vs_price_sqft_living_relplot.png' />

<img src = '../main/Data & Figures/age_vs_price_grade_relplot.png' />

<img src = '../main/Data & Figures/age_vs_price_condition_relplot.png' />

            * 'age' is negatively correlated with 'price'.
            * The higher the 'age', the lower the 'price'.
            * With respect to 'sqft_living', 'age' does not matter much. Higher square footage is still valued at higher price.
            * Older houses tend to have lower 'grade'.
            * New houses tend to score higher 'grade' of 10 and above. New houses tend to score higher 'grade' of 10 and above. Newer houses are graded higher due 
            to better and more up to date material quality, architectural design, and construction. This includes critical parts of the house, like plumbing, 
            electrical, the roof, and newer appliances.
            * When a house is old, even if it is in good and very good condition 4, and 5, it is still valued less than new houses with average condition of 3.
            * New houses is largely scored only an average 'condition'.

### **'renovation'**

<img src = '../main/Data & Figures/renovation_vs_price_boxplot.png' />

<img src = '../main/Data & Figures/sqft_above_vs_price_renovation_relplot.png' />

<img src = '../main/Data & Figures/age_vs_price_renovation_relplot.png' />

            * Renovation is weakly and positively correlated with 'price'.
            * There are houses without renovation is still Above Median price and there are houses with renovation is still Below Median price.
            * Older houses tend to have renovation done. This explains why some older houses are scored high in 'grade' and 'condition'
            * Although renovation can add values to older houses, the age of the house is a more impactful feature than any kind of adds-on.

### **'zipcode'**

<img src = '../main/Data & Figures/zipcat_scatterplot.png' />

<img src = '../main/Data & Figures/location_vs_price_sqft_living15_scatterplot.png' />

            * We see that properties that are 1.6M+ are clustered and increase in price as they go toward the center. 
            * The yellow region of C which includes Bellevue, Mercer Island, Newcastle is the region with the highest values. 
            * The neighboring region of G also stands out, including Sammamish, Issaquah, Carnation, Duvall
            * Both C and G have waterfront properties.
            * Both C and G have high 'sqft_living15'.
            * Both C and G are graded high, of 10 and above
            * Both C and G are average age, with G seems 'younger.'

## Summary of Actionable Insights

Results suggest that the following factors can be used to predict the value of the house:

            * Location is the most important thing and tagging along with it the presence of waterfront. 
            Home value is also affected by the sale prices of similar homes in the neighborhood that have sold recently.
            * Square footage of livable space matters and the more beds, baths, and floors your home offers, the more your home is worth.
            * Renovation with additional basement and living space adds extra boost to the value of the home
            * You need a condition of 3 and above and grade of 8 and above to have a high value home.
            * If your house is old, renovation can help but not that much.

**Best Predictive Features:**

            - The presence of 'waterfront' is the most positively impactful feature for 'price.' 
            - Location is also a powerful determining factor for the value of a home.
            - Other features that add value to a home are: 'sqft_above',  'base_1.0', 'sqft_living15', 'reno_1.0', 'bathrooms', 'cond_5.0', 'age'.
            - Interactions that have a positive impact on the price are: 'sqft_log * zip_H', 'sqft_above * sqft_living15', 'sqft_lot * bathrooms'.
            - Features that decrease the value of a home are: 'bedrooms', 'cond_3.0', 'zip_F', 'cond_2.0'.
            - Interactions that have a negative impact on the price are: 'sqft_above * sqft_lot', 'sqft_lot * age', 'sqft_above * zip_C', and 'sqft_lot * zip_A'

##  Future Works
1. Calculate value of the home in price per square foot instead of just price.

2. Research location in-depth such as (openhome.com, 2021):
* The quality of local schools
* Employment opportunities
* Proximity to shopping, entertainment, and common services such as hospitals
* Proximity to highways, utility lines, and public transit
* Proximity to the nearest major city

3. How hot (or cold) is the area's real estate market?
Because the number of other properties for sale in the area and the number of buyers in the market can impact the home value.

4. When is the best time to buy or sell?
