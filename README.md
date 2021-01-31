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

**Q1. What features adds value to a home?** 

**Q2. What features decreases value of a home?** 

**Q3. What features are nice to have but have no effect on the values of a home?** 

**Q4. How does location affect the price of a home?** 

**Q5. What are the most predictive features to predict the price of a home?** 

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

## Part IIA + B: Machine Learning

### Model Building Steps: 

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

|    | Model   | Description                                          | Num Features | r2                 | Accuracy           | RMSE Train          | RMSE Test           | Bias Train              | Bias Test            | Variance Train      | Variance Test       | Cross Validation     | Multicollinearity | Normality | Homoscedasticity |
|----|---------|------------------------------------------------------|--------------|--------------------|--------------------|---------------------|---------------------|-------------------------|----------------------|---------------------|---------------------|----------------------|-------------------|-----------|------------------|
| 0  | Model A | All features                                         | 19.0         | 0.668176586782782  | 66.23671804132351  | 0.5625265909631906  | 0.6286713292932756  | 0.1866438695124231      | 0.17987848096082254  | 0.2106811199621985  | 0.2052605814854408  | -0.3367689690211301  | P                 | F         | F                |
| 1  | Model B | All features, outliers removed, RFE                  | 10.0         | 0.6075144477398109 | 60.93273250883284  | 0.5624862178378017  | 0.5583188110166062  | 0.12839396166672934     | 0.12270141633234286  | 0.22190382567195133 | 0.2254406225204553  | -0.3180497252139539  | P                 | F         | F                |
| 2  | Model C | All features + Polynomial Regression                 | 19.0         | 0.6198535778967182 | 66.64924112096658  | 0.6152325525353235  | 0.6248189420378881  | -1.3257345975031278     | -1.3313392696514037  | 0.1657514363998529  | 0.16310249878770364 | -0.3864108608759035  | P                 | F         | F                |
| 3  | Model D | All features + Log(X)                                | 18.0         | 0.6168320844440425 | 59.185617618792065 | 0.5999984730497039  | 0.6912067035032161  | 0.1810975776121652      | 0.1732590331755436   | 0.21309050777446611 | 0.2074955283223341  | -0.3872608258439013  | P                 | F         | F                |
| 4  | Model E | Log(y) + All features                                | 20.0         | 0.7615458836361797 | 77.32516271509692  | 0.4905295885170489  | 0.4796712201647737  | -0.1245442680310424     | -0.120159147210689   | 0.29948311311119685 | 0.2936018827731665  | -0.2412572819380095  | P                 | P         | P                |
| 5  | Model F | Log(y) + All features + Interactions                 | 19.0         | 0.6771659390672531 | 68.98034550955364  | 0.5700501209276141  | 0.5610351227268588  | -1.0278359768937528     | -1.0246300971266835  | 0.34732382599066325 | 0.34146161696171    | -0.3260748180318126  | P                 | P         | P                |
| 6  | Model 1 | Log(y) + Log(X) + All features - location            | 13.0         | 0.5836830428706551 | 59.65668383976206  | 0.6466107231291456  | 0.6398193484246292  | -1.0951919939462913e-16 | 0.009110384336019477 | 0.5781577193413534  | 0.5913110018000953  | -0.4185621389300415  | P                 | P         | P                |
| 7  | Model 2 | Log(y) + Log(X) + All features + location            | 19.0         | 0.7226247743093053 | 73.30382551258425  | 0.5282681406977942  | 0.5204704634628459  | -0.7369406103693011     | -0.7347142277673622  | 0.2830419231289605  | 0.27770941734986065 | -0.2798721618517688  | P                 | P         | P                |
| 8  | Model 3 | Log(y) + Log(X) + All features + RFE                 | 10.0         | 0.7131611493854333 | 72.40196837571291  | 0.5371659213600524  | 0.5291887784787914  | 0.2999619340989784      | 0.3028461707608784   | 0.3688610642978886  | 0.3635981200628021  | -0.28873308214221904 | P                 | P         | P                |
| 9  | Model 4 | Log(y) + Log(X) + All features + Interactions        | 24.0         | 0.7738199360010017 | 78.35181714237666  | 0.4773545938752356  | 0.46868634013226684 | 0.25819826998164813     | 0.26286050990949894  | 0.2894349267525931  | 0.2835422865721812  | -0.2292681767965224  | P                 | P         | P                |
| 10 | Model 5 | Log(y) + Log(X) + All features + Interactions + Poly | 28.0         | 0.7597763001594978 | 76.90656423067085  | 0.49169672404915826 | 0.4840785596517417  | 0.2853833651839835      | 0.2907084113541249   | 0.31480578660205333 | 0.30877757250175863 | -0.24344834501143783 | P                 | P         | P                |

## **BEST MODEL:** MODEL 4

## Summary of Findings

### **'sqft_living'**

![alt text](../main/Data & Figures/sqft_living_vs_price_lmplot.png)

            * 'sqft_living' is strongly & positively correlated with target 'price'.
            * The higher the square footage of living space, the higher the price.
            * 4,500 sqft guarantees above price median.

### **'sqft_lot'**

            * 'sqft_lot' ranges from 520 - 1,625,359 sqft with majority of houses <50,000 sqft_lot
            * 'sqft_lot' is weakly & positively correlated to 'price'
            * Higher 'sqft_lot' does not equal to higher price

### **'sqft_above'**

            * 'sqft_above' is strongly & positively correlated to 'price'
            * The higher the 'sqft_above' the higher the price

### **'basement'**

            * There are more houses without a basement than with a basement.
            * The presence of a basement increases the price of a house but not always: there are houses without a basement still make to Above Median price and there are houses with a basement stay behind in Below Median price.
            * 'basement' is weakly & positively correlated to 'price'.

### **'sqft_living15'**

            * 'sqft_living15' is strongly & positively correlated with 'price'.
            * The higher the square footage of the nearest 15 neighbor houses, the higher the price for a house.
            * This demonstrates that neighborhood/location is a value-adding feature when predict the price of a home.

### **'bedrooms'**

            * 'bedrooms' is positively correlated with 'price'.
            * Higher number of bedrooms stops mattering if 'sqft_living' or 'sqft_above' is small.
            * Too many bedrooms to crowd square footage of the home will have less value.
            * 3 - 5 bedrooms is ideal for houses > 2,500 sqft of living space.
            * 1 - 2 bedrooms is idead for houses < 2,500 sqft of living space.

### **'bathrooms'**

            * 'bathrooms' is highly and positively correlated with 'price'
            * Higher number of bathrooms does not matter if 'sqft_living' or 'sqft_above' is low
            * Too many bathrooms crowding square footage of the home will have less value.
            * 'Penalty' of having too many 'bathrooms' is less severe than having too many 'bedrooms'

### **'grade'**

            * 'grade' is strongly and positively correlated with 'price'.
            * The higher the grade, the higher the value of a home.
            * To get above the price median, a home needs to be at least grade 10.
            * There is also the 'sqft_living' and 'sqft_above' effect: the higher the square footage, the higher the grade.
            * Smaller square footage houses need at least grade 7 to get past the price median.

### **'floors'**

            * 'floors' is positively correlated to 'price'.
            * Higher number of floors can add value to houses that have smaller square footage.
            * Higher number of floors doesn't add more value to houses that have big square footage.
            * Higher number of floors with small square footage decreases the value of a home.
            * 2 floors is ideal to have, more than that is unnecessary.

### **'waterfront'**

            * 'waterfront' is positively correlated to 'price'.
            * There are houses without a waterfront make it into Above Median price but with waterfront, a house is guaranteed to be Above Median.

### **'condition'**

            * 'condition' is weakly and positively correlated to 'price'.
            * 'condition' of at least 3 is needed to raise value of a home.
            * a low 'condition' score decreases the value of a home even if that home has high square footage.
            * High 'grade' does not matter if 'condition' is low.

### **'age'**

            * 'age' is negatively correlated with 'price'.
            * The higher the 'age', the lower the 'price'.
            * With respect to 'sqft_living', 'age' does not matter much. Higher square footage is still valued at higher price.
            * Older houses tend to have lower 'grade'.
            * Those older houses with higher 'grade' of 10 and above is still valued equally high as newer houses with the same 'grade'.
            * New houses tend to score higher 'grade' of 10 and above.
            * Older houses with great 'condition' is valued equally high as new houses with average 'condition'.
            * New houses is largely scored only an average 'condition'.

### **'renovation'**

            * Renovation is weakly and positively correlated with 'price'.
            * There are houses without renovation is still Above Median price and there are houses with renovation is still Below Median price.
            * Older houses tend to have renovation done. This explains why some older houses are scored high in 'grade' and 'condition'

### **'zipcode'**

            * We see that properties that are 1.6M+ are clustered and increase in price as they go toward the center. 
            * The yellow region of C which includes Bellevue, Mercer Island, Newcastle is the region with the highest values. 
            * The neighboring region of G also stands out, including Sammamish, Issaquah, Carnation, Duvall
            * Both C and G have waterfront properties.
            * Both C and G have high 'sqft_living15'.
            * Both C and G are graded high, of 10 and above
            * Both C and G are average age, with G seems 'younger.'

## Summary of Actionable Insights

Results suggest that the following factors can be used to predict the value of the house:

##  Future Works
