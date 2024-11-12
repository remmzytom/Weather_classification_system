# Weather classification system





### Scenario
As an urban planner or agricultural manager, I need to accurately classify different weather conditions using environmental features to determine the optimal times for infrastructure projects and agricultural activities. This will ensure that operations are conducted under favorable weather conditions, thereby providing actionable insights for planning and decision-making.

#### What this use case will teach us
At the end of this use case you will:

Understand how to preprocess and analyze environmental data.
Learn how to build and evaluate a machine learning model for classification tasks.
Gain experience in feature selection and engineering for weather-related datasets.
Develop skills in using Python libraries such as Pandas, Scikit-learn, and Matplotlib.
Understand the importance of accurate weather classification for planning and decision-making in various sectors.
introduction
In this use case, we aim to develop a robust machine learning model capable of accurately classifying various weather conditions such as sunny, cloudy, rainy, and stormy using environmental features. These features include ambient air temperature, relative humidity, atmospheric pressure, wind speed and direction, and gust wind speed. Accurate weather classification is important for optimizing the timing of infrastructure projects and agricultural activities, ensuring that operations are conducted under favorable weather conditions. By leveraging machine learning techniques, we can provide actionable insights for planning and decision-making.

#### Background
Weather conditions have a significant impact on various sectors, including agriculture, construction, and transportation. Accurate weather forecasts and classifications can help in planning and executing operations more efficiently. For instance, farmers can optimize planting and harvesting times based on expected weather conditions, while construction projects can be scheduled to avoid adverse weather that could delay progress or compromise safety.

#### In this project, we will use historical weather data from Melbourne's open data portal. The datasets include:

Microclimate sensors data — CoM Open Data Portal (melbourne.vic.gov.au)

Argyle Square Weather Stations (Historical Data) — CoM Open Data Portal (melbourne.vic.gov.au)

Argyle Square Air Quality — CoM Open Data Portal (melbourne.vic.gov.au)


#### Dataset Information

##### The dataset for this project includes the following features::

- Ambient air temperature (°C)
- Relative humidity (%)
- Atmospheric pressure (hPa)
- Wind speed (m/s)
- Wind direction (degrees)
- Gust wind speed (m/s)
>
 These features will be used to classify weather conditions into categories such as sunny, cloudy, rainy, and stormy. The dataset will be preprocessed to handle any missing values, outliers, or inconsistencies - before being used to train the machine learning model.



#### skill set
 - Datacleaning
 - Data analysis and visualizations
 - Machine Learning(supervised Learning)
   - Logistic Regression
   - Support Vector Machine(SVM)
   - Decision Tree
   - AdaBoost classifier
   - Random Forest Classifier
- Basic Meteorology


#### steps taken 

#### Data cleaning/preparation
Data loading and with API
Handling missing Values
Data cleaning and formatting

#### Feature engineering
- removing the outlier
- finding the correlation
- getting the truth label for the dataset
- normalizing/standardizing the dataset
- merging dataset together
- feature selection
- feature encoding

#### Exploratory Data Analysis(EDA)
- Variation od Avg_Wind_speed and Gust_Wind_speed
- Impact of Vapour Pressure on Humidity
- Patterns in Air Temperature, Humidity, and Pressure
- patterns of pollutants across the year
- Efect of pollutants across different years
- determine the features that contributed to high rate of Ozone

#### Machine learning and modelling
- dimensionality reduction(t-SNE)
- resampling technique(smote)
- Model training
- prediction
