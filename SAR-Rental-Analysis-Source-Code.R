install.packages(c("ROSE", "cluster", "factoextra", "dbscan", "dplyr", 
                   "ggplot2", "lubridate", "VIM", "caret", "randomForest", 
                   "e1071", "rpart", "geosphere", "mice"))

library(ROSE)
library(dplyr)
library(ggplot2)
library(lubridate)
library(VIM)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(geosphere)
library(mice)

# Reading the dataset
data <- read.csv("C:/Users/harin/OneDrive/Desktop/Analytics Practicum/SAR Rental.csv")

# Removing identifier columns as they are useless
data <- data %>%
  select(-row., -user_id, -vehicle_model_id)

# Replace travel type ID values with descriptive values
data$travel_type_id <- factor(data$travel_type_id, 
                              levels = c(1, 2, 3), 
                              labels = c("long distance", 
                                         "point to point", 
                                         "hourly rental"))

# Replace missing value in package ID with another category named unknown
data$package_id[is.na(data$package_id)] <- "Unknown"

# Replace Package ID values with descriptive values
package_labels <- c("4hrs & 40kms", "8hrs & 80kms", "6hrs & 60kms", 
                    "10hrs & 100kms", "5hrs & 50kms", "3hrs & 30kms", 
                    "12hrs & 120kms", "Unknown")

data$package_id <- factor(data$package_id, 
                          levels = 1:8, 
                          labels = package_labels)


# Dropping columns with more than 50% missing observations as they will create biasness in the data!

df_clean <- data %>%
  select(-from_city_id, -to_city_id)


# Convert date columns to date data type
df_clean$from_date <- as.POSIXct(df_clean$from_date, 
                                 format = "%m/%d/%Y %H:%M")

df_clean$From_Hour <- hour(df_clean$from_date)
df_clean$From_Minute <- minute(df_clean$from_date)

# Convert date columns to date data type
df_clean$to_date <- as.POSIXct(df_clean$to_date, 
                               format = "%m/%d/%Y %H:%M")

df_clean$To_Hour <- hour(df_clean$to_date)
df_clean$To_Minute <- minute(df_clean$to_date)

# Convert date columns to date data type
df_clean$booking_created <- as.POSIXct(df_clean$booking_created, 
                                       format = "%m/%d/%Y %H:%M")

df_clean$Booking_Hour <- hour(df_clean$booking_created)
df_clean$Booking_Minute <- minute(df_clean$booking_created)

df_clean <- df_clean %>%
  select(-to_date, -from_date, -booking_created)

# Perform multiple imputation (default m=5 datasets)
set.seed(123)  # For reproducibility
imputed_data <- mice(df_clean, method = "pmm", m = 5, printFlag = FALSE)

# Choose one imputed dataset (the first one)
df_imputed <- complete(imputed_data, 1)

# Add distance
# Compute Haversine distance (in meters)
df_imputed$distance_meters <- distHaversine(
  matrix(c(df_imputed$from_long, df_imputed$from_lat), ncol = 2),
  matrix(c(df_imputed$to_long, df_imputed$to_lat), ncol = 2)
)

# Convert distance from meters to kilometers
df_imputed$distance_km <- df_imputed$distance_meters / 1000

# Remove meters column
df_imputed <- df_imputed %>%
  select(-distance_meters)

# Get descriptive statistics:
summary(df_imputed)

# Perform ROSE sampling to balance the target variable (Car_Cancellation)
df_balanced <- ROSE(Car_Cancellation ~ ., 
                    data = df_imputed, 
                    seed = 123)$data

df_balanced$Car_Cancellation <- as.factor(df_balanced$Car_Cancellation)
set.seed(123)  # For reproducibility

# Split data into training (70%) and testing (30%)
trainIndex <- createDataPartition(df_balanced$Car_Cancellation, 
                                  p = 0.7, list = FALSE)
trainData <- df_balanced[trainIndex, ]
testData <- df_balanced[-trainIndex, ]

# ---- 1. Random Forest Classifier ----
rf_model <- randomForest(Car_Cancellation ~ ., 
                         data = trainData)
rf_pred <- predict(rf_model, 
                   testData)
rf_cm <- confusionMatrix(rf_pred, testData$Car_Cancellation)
print(rf_cm)



# ---- 2. Decision Tree Classifier ----
dt_model <- rpart(Car_Cancellation ~ ., 
                  data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, 
                   type = "class")
dt_cm <- confusionMatrix(dt_pred, testData$Car_Cancellation)
print(dt_cm)


# ---- 3. Naïve Bayes Classifier ----
nb_model <- klaR::NaiveBayes(Car_Cancellation ~ ., 
                             data = trainData)
nb_pred <- predict(nb_model, testData)$class
nb_cm <- confusionMatrix(nb_pred, testData$Car_Cancellation)
print(nb_cm)


# ---- 4. Support Vector Machine (SVM) ----
svm_model <- svm(Car_Cancellation ~ ., 
                 data = trainData, kernel = "radial")
svm_pred <- predict(svm_model, testData)
svm_cm <- confusionMatrix(svm_pred, testData$Car_Cancellation)
print(svm_cm)



res <- df_imputed %>%
  group_by(travel_type_id, Car_Cancellation) %>%
  summarise(Count=n())

# Create bar chart
ggplot(res, aes(x = factor(travel_type_id), y = Count, fill = factor(Car_Cancellation))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Car Cancellations by Travel Type",
       x = "Travel Type ID",
       y = "Count",
       fill = "Car Cancellation") +
  theme_minimal()


res <- df_imputed %>%
  group_by(package_id, Car_Cancellation) %>%
  summarise(Count=n()) %>%
  filter(!is.na(package_id))


ggplot(res, aes(x = factor(package_id), y = Count, fill = factor(Car_Cancellation))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Car Cancellations by Package ID",
       x = "Package ID",
       y = "Count",
       fill = "Car Cancellation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#2. 
chisq.test(table(df_imputed$package_id, df_imputed$Car_Cancellation))


chisq.test(table(df_imputed$travel_type_id, 
                 df_imputed$Car_Cancellation))



#3. Can the status of online booking and mobile site booking be used to predict car cancellation. Is there any relationship of these variable with the car cancellation.
res <- df_imputed %>%
  group_by(online_booking, Car_Cancellation) %>%
  summarise(Count=n()) 

# Create bar chart
ggplot(res, aes(x = factor(online_booking), y = Count, fill = factor(Car_Cancellation))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Car Cancellations by Online Booking Status",
       x = "Online Booking Status",
       y = "Count",
       fill = "Car Cancellation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


chisq.test(table(df_imputed$mobile_site_booking, df_imputed$Car_Cancellation))


chisq.test(table(df_imputed$online_booking, 
                 df_imputed$Car_Cancellation))


set.seed(123)
# ---- 1. Random Forest Classifier ----
rf_model <- randomForest(Car_Cancellation ~ mobile_site_booking + 
                           online_booking, 
                         data = trainData)
rf_pred <- predict(rf_model, 
                   testData)
rf_cm <- confusionMatrix(rf_pred, testData$Car_Cancellation)
print(rf_cm)


# ---- 2. Decision Tree Classifier ----
dt_model <- rpart(Car_Cancellation ~ mobile_site_booking + online_booking, 
                  data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, 
                   type = "class")
dt_cm <- confusionMatrix(dt_pred, testData$Car_Cancellation)
print(dt_cm)


# ---- 3. Naïve Bayes Classifier ----
nb_model <- klaR::NaiveBayes(Car_Cancellation ~ mobile_site_booking +
                               online_booking, 
                             data = trainData)
nb_pred <- predict(nb_model, testData)$class
nb_cm <- confusionMatrix(nb_pred, testData$Car_Cancellation)
print(nb_cm)





