**SAN FRANCISO AUTO RENTAL ANALYSIS**

**Business Background**

San Francisco Auto Rental Company deals with providing vehicles to customer transportation for short-term or long-term. SAR found a problem in 2013 that the drivers were not showing up for the rides they were scheduled for. The clients were left with no rides even though they booked them in advance.  This problem has been continuing since 2013, and by 2024, SAR found no proper evidence on why the drivers were cancelling the rides despite having an online booking system for scheduling rides beforehand and calling in the driver for communication. 
Furthermore, SAR company reputation and customer satisfaction of booking rides were at stake, as the root cause of the above-mentioned problem has not been addressed. This project will study and deploy machine learning algorithms to find a predictive analytical solution to the problem. 

**Business goal**

The primary business goal of this case study is to help SAR management in predicting why rides are getting cancelled to improve their operational efficiency. Secondly, retention of customers by minimizing the cancellations based on their booking preferences and behavior. Furthermore, vehicle allocation strategies need to be optimized based on predictive insights to make sure available cars are utilized properly. By identifying high risk bookings and implementing targeted interventions, financial impact on the SAR can be reduced. To enhance customer experience, analyzing the impact of online and mobile booking cancellations can suggest potential insights. 

**Business Objectives**

From the given data from SAR management,  the above business goals can be articulated by these business objectives. 
Developing predictive models to identify potential cancellations. 
Analyzing patterns of cancellations based on routes the riders have travelled on.
Impact of different variables on cancellations.
Developing strategies to reduce cancellations from customer preferences.

**Analytical goals** 

1. Predictive Modelling
Multiple models were developed to predict the car cancellation on all features and their performance is evaluated. 
From business perspective, it identifies high-risk bookings before they happen and can take proactive steps to stop the bookings that are likely to be cancelled. 
2.  Analyzing travel type and package relationships: 
Identifying relationship between travel type and package by using chi-squared tests. 
From business perspective, it helps to distribute vehicles  based on travel type patterns. Identify which packages are prone to cancellations and necessary steps can be implemented for those packages. 
3.Impact of digital bookings 
Booking platform insights gives an idea of how customer likes to book a ride and by developing online platform specific strategies the number of cancellations can be reduced. 

**Exploratory Data Analysis**

**Missing Values**

<img width="399" height="198" alt="image" src="https://github.com/user-attachments/assets/80743e0a-026f-416a-8272-6d654e2aa591" /> 

•	From the above image, there is clear pattern of missing values, blocks of missing values can be seen together rather than randomly distributed. 
•	The clustering of missing values suggests that missingness in one variable is associated with missingness in another variable hence it can be concluded that missing values fall into the category of Missing at Random (MAR). 
•	package_id and to_city_id have similar patterns and geographical variables have similar patterns of missing data. 
•	Also, by analyzing the dataset, package_id and travel_type_id missing data have correlation, for travel_type_id 2, there is no data in the package_id column. 
•	From the given information about the attributes from_area_id and to_area_id are only available for travel_type_id 2 

**Handling missing values** 
Since the data is missing at random and some data like from_area_id and to_area_id are available only for point-to-point, completely imputing the data might cause bias and leads to poor model performance. 
**Business perspective** 
•	Different travel types have different requirements and driver behavior might be dependent on the distance and duration of the trip. 
•	Customer need for travel and expectations will vary for travel type. 
•	Furthermore, driver behavior patterns influence the trips, some drivers may avoid long distances, while some may avoid point-to-point due to location which explains the cancellations. 
•	The missingness is more inclined to the part of business logic rather than explaining uncertainty in the  data 
As for the above reasons, Predictive Mean Matching (PMM) method is implemented through mice package. Instead of using simple methods to replace missing values, PMM selects real values from similar observations in the dataset and preserves the natural structure, distribution and variability of the data. 

**Data Engineering and Transformation** 
Data Engineering and Transformation are main steps for preparing raw data for analysis and modelling. For the SAR dataset, the below transformations ensure consistency and usability. 
**Removing unnecessary columns**
columns (row., user_id, vehicle_model_id) does not contribute meaning information to predict the car cancellations. These columns were removed to reduce data redundancy and columns such as from_city_id and to_city_id were removed as they contain more than 50% of data missing.  
**Encoding Categorical Variables**
Understanding the data step, categorical columns such as package_id and travel_type_id are converted into factors. 
Converting Date-Time Columns and Extracting Time Features
The dataset have several date and time objects ( from_date, to_date, booking_created). These are converted into standard date-time format(POSIXct) to make sure that time and date objects are handled  properly in the analysis. 
To improve granularity of the data, additional time-related  features are extracted. 
•	Hour and Minute: New variables such as From_Hour, From_Minute, To_Hour, To_Minute, Booking_Hour, and Booking_Minute were derived from the original timestamps. This transformation enables time-based analysis, such as peak booking hours and rental duration trends.
Computing Travel Distance from Latitude and Longitudinal Values
Travel distance is calculated based on the latitude and longitude coordinates of trip origin (from_long, from_lat) and destination (to_long, to_lat). The Haversine formula was used to compute the great-circle distance between two geographical points, accounting for the Earth's curvature. The distance was initially in meters and converted into kilometers and stored in a new column distance_km . 
To sum up, the following steps are performed in Data Engineering and Transformation
•	Redundant and irrelevant features were removed.
•	Categorical variables were made interpretable.
•	Missing data was systematically handled using multiple imputations.
•	Time-related features were extracted to improve granularity.
•	Travel distances were computed to provide additional insights into rental patterns.

**Data Partitioning Methods** 
Data partitioning in predictive modelling ensures that a model can perform well to unseen or new data. By splitting the dataset into different subsets into training set and testing set to evaluate model’s performance. 
As the dataset is highly imbalanced with only 743 cancellations out of 10000 observations, the results might be biased. 
In general, data imbalances are common in classification problems and this imbalance easily leads to biased models favoring majority class and making the model difficult to correctly predict the minority class. 
In our dataset, the partitioning process involves two major steps: balancing the data set using ROSE sampling and splitting the data into training and testing sets.
To address this issue, the Random Over-Sampling Examples (ROSE) method is used in this case. The ROSE technique generates synthetic data points to create a more balanced class distribution. Unlike simple oversampling (which duplicates minority class instances) or under sampling (which removes majority class instances), ROSE generates synthetic examples using smoothed bootstrap resampling. The target variable Car_Cancellation is converted to a factor, ensuring that classification algorithms treat it as a categorical variable.
Once the dataset is balanced, it is split into a training set (70%) and a testing set (30%) using the createDataPartition function from the caret package.
•	Training Data (trainData): This subset (70% of the dataset) is used to train the machine learning model. The model learns patterns, relationships, and trends in the data.
•	Testing Data (testData): This subset (30% of the dataset) is held out and used to evaluate the model’s performance on unseen data.
The createDataPartition function ensures that the split is correct, which means both training and testing sets have same proportion of class labels . The split is a mandatory step in classification tasks as uneven split of classes will lead to biasedness . The set.seed(123) ensures reproducibility so that every time the code is run, the same partitioning is obtained. 

**Model  Selection** 
Based on the business goals, Random Forest, Decision Tree, Naïve Bayes, SVM and statistical tests like Chi-squared test are  selected. 
Random Forest is chosen because it can handle mixed data types of categorical and numerical variables, Decision Tree can easily explain the transparency of variables , Naïve Bayes can handle large datasets , SVM models is selected for predicting complex patterns. 
Chi-squared tests identify significant relationship between categorical variables and target variable. 

**Model fitting, Validation and Test Accuracy** 
**Objective 1:  Predicting car_cancellation based on all features in the dataset.**
To predict car cancellations, four different machine learning models were implemented: Random Forest, Decision Tree, Naïve Bayes, and Support Vector Machine (SVM). The models were then trained using the training dataset and evaluated on the test dataset using a confusion matrix and performance metrics such as accuracy, sensitivity, specificity.
Random Forest Classifier 
•	Accuracy: 76.13%
•	Sensitivity (Recall for Class 0): 75.61%
•	Specificity (Recall for Class 1): 76.65%
The Random Forest model achieved the highest accuracy among all models, indicating that it effectively captured complex patterns in the data. The Balanced Accuracy (76.13%) shows that the model performs well for both cancellation and non-cancellation cases.  
Decision Tree Classifier
Decision Tree Classifier will use a single tree. 
•	Accuracy: 70.16%
•	Sensitivity: 74.75%
•	Specificity: 65.45%
The Decision Tree model had lower accuracy (70.16%) compared to Random Forest. The specificity (65.45%) was significantly lower than sensitivity (74.75%), meaning the model was better at predicting "no cancellation" cases (Class 0) but failed to classify actual cancellations (Class 1)
Naïve Bayes Classifier
•	Accuracy: 68.99%
•	Sensitivity: 64.86%
•	Specificity: 73.21%
The model showed a lower sensitivity (64.86%) compared to specificity (73.21%), meaning it performed better at predicting cancellations (Class 1) but had a higher false negative rate. Also it has lowest accuracy of 68.99% among all models where the assumption of feature independence may not work for our dataset. 
Support Vector Machine (SVM)
•	Accuracy: 74.32%
•	Sensitivity: 73.83%
•	Specificity: 74.83%
**The model achieved an accuracy of 74.32%, and Balanced Accuracy (74.33%)**, indicating stable performance across both classes.

**Reports on Model Performance**
Random Forest is the best model for predicting car cancellations, with the highest accuracy (76.13%).  It captured complex relationships among features. Decision Tree has lower specificity (65.45%) makes it unreliable for predicting actual cancellations. Naïve Bayes had the weakest performance, likely due to its assumption of feature independence, which is not suitable for our dataset. SVM performed well, achieving accuracy close to Random Forest but due to low kappa score it cannot be finalized. Overall, Random Forest is recommended. 

**Objective 2: Is there any relationship of travel type and package ID with the car cancellation.**
A Pearson's Chi-squared test was conducted to determine if travel type is significantly associated with car cancellation:
•	Chi-squared test results:
o	X-squared = 33, df = 2, p-value = 6.826e-08
The very small p-value indicates a statistically significant relationship between travel type and car cancellation.
 
From the above graph, we can say that point-to-point travel types have more cancellations followed by hourly rental and very fewer on long distances. 
Another Pearson's Chi-squared test was conducted to determine if package ID is significantly associated with car cancellation:
•	Chi-squared test results:
o	X-squared = 24.689, df = 6, p-value = 0.0003898
The very small p-value confirms a statistically significant relationship between package type and car cancellation. 
 
From the graph, Shorter-duration packages (4hrs & 40kms, 8hrs & 80kms) see more cancellations than longer-duration packages. 
**Reports on Model Performance**
There is a significant relationship between both travel type and package ID with car cancellations. Point-to-point travel and hourly rentals have higher cancellations, indicating they are more susceptible to last-minute changes. From the above tests, shorter-duration bookings can be advised for strict cancellation policies and for long duration rides customers can be offered more offers. 

**Objective 3: Relationship Between Online Booking, Mobile Site Booking, and Car Cancellations**
This analysis will help in determining the status of online booking and mobile site booking can be used to predict car cancellations. 
 
From the dataset, 6,171 bookings that were made without using online booking resulted in no cancellation, whereas 296 were canceled. For those who booked online, 3,086 did not cancel, while 447 bookings were canceled. 
The Pearson’s Chi-squared test with Yates' continuity correction is performed to identify whether customers who book online are associated with car cancellations. 
Reports on Model Performance
The result of a chi-squared value of 215.44 with a p-value of less than 2.2e-16, indicating a statistically significant relationship between online booking and car cancellation. The significant p-value suggests that the mode of booking does influence the drivers cancelling the rides . A higher proportion of online bookings resulted in cancellations compared to offline bookings 
 
Regarding mobile site bookings, 8,896 bookings that were not made through the mobile site resulted in no cancellation, while 680 were canceled. On the other hand, for bookings made through the mobile site, 361 were not canceled, and 63 were canceled. 
**Reports on model performance**
The Pearson’s Chi-squared test with Yates' continuity correction for mobile site booking and car cancellation yielded a chi-squared value of 34.405 with a p-value of 4.475e-09, suggesting a significant association between mobile site booking and car cancellations. By comparing online bookings and mobile site bookings, customers booking from mobile site bookings also shows likelihood of cancelling the reservations even though online bookings cancellation rate is higher. To sum up, both platforms are associated with cancellation rates.  
 
**Recommendations** 
From the predictive model results, statistical tests and clustering analysis, the following recommendations can be made to reduce the car cancellations. 
Online and Mobile Booking Platforms : The analysis concluded that there is a positive relationship between online booking and mobile site bookings and the target variable car cancellations. Improving the customer experience on booking platforms could reduce car cancellations. 
Travel Type and Package Offerings : Point-to-point rentals have the highest number of cancellations compared to other travel types. Package-based rentals with longer duration (e.g., 12 hours & 120 km) tend to have lower cancellation rates. By changing the offers or incentives on the travel types we could improve other rentals and may reduce cancellation rates. Customers should be encouraged to book longer duration trips based on their preferences with rescheduling options so that booking retention can be improved. 

**Future Work** 
From the given data, the predictive models considered only limited factors such as booking platforms, package types and travel types. Future research should include customer demographics and historical fatures like ride history, which can be more useful to provide deeper insights into predicting car cancellations. 
If customer reviews were provided, NLP methods can extract the text and analyze the likelihood of cancellation based on the customer reviews. 

**Conclusion** 
The study was performed to predict why rides were getting cancelled and predictive modelling is done by analyzing variables travel type, package ID, online booking, and mobile site usage and the distance. Statistical tests show significant relationship between these variables and cancellation rates. Random Forest, Decision Tree, Naïve Bayes, and SVM, were deployed to classify cancellation rates. From these insights SAR management can reduce cancellations, enhance customer satisfaction and improve overall efficiency. 



