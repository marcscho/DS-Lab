# Databricks notebook source
# MAGIC %md #Machine Learning

# COMMAND ----------

# MAGIC %md ## Importing data set again

# COMMAND ----------

# MAGIC %md The SQL command below shows that the global temporary view we've created in the last part of the previous notebook actually worked and contains our data. Next, we make sure it is available again to us in a Spark data frame.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM delays_table

# COMMAND ----------

delays = spark.sql('SELECT * FROM delays_table')
delays.printSchema()

# COMMAND ----------

# MAGIC %md ##Data preparation for ML

# COMMAND ----------

# MAGIC %md ###Avoid information leakage

# COMMAND ----------

# MAGIC %md There are a few things we need to get done before we can actually train our first machine learning model to predict ahead of time whether a flight will be arriving late or not.
# MAGIC 
# MAGIC The first and most important task is to ensure that there is no information leakage happening. As you can see from the data frame displayed above, we still have two columns in our data set that contain information for arrival delay: *ARR_DEL15* and *ARR_DELAY*. Since *ARR_DEL15* - a binary column indicating whether a flight arrived (more than 15 min.) late - will be our label, we need to make sure we remove *ARR_DELAY* before training because it would basically contain the answer to our question already.

# COMMAND ----------

#delays = delays.drop('ARR_DELAY', 'DEP_DELAY', 'DEP_DEL15')
delays = delays.drop('ARR_DELAY', 'DEP_DELAY')
cols = delays.columns
delays.printSchema()

# COMMAND ----------

# MAGIC %md ###Recoding information

# COMMAND ----------

# MAGIC %md Since we can only run computations on numerical data, our categorical variables which are alphanumeric, will need to be recoded. This will be done using the <code>StringIndexer</code> which is documented [here](https://spark.apache.org/docs/latest/ml-features.html#stringindexer) as well as <code>OneHotEncoderEstimator</code> documented [here](https://spark.apache.org/docs/latest/ml-features.html#onehotencoderestimator).
# MAGIC 
# MAGIC Also note that a new list object called *stages* is created which is empty at the beginning. Every transformation that is applied, such as the string indexing or one hot encoding, is the subsequently added to the stages list. The stages list will ultimately be fed to the pipeline object and will tell it which steps to follow everytime the pipeline is run.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoderEstimator

categoricalColumns = ["DAY_OF_WEEK", "MONTH", "UNIQUE_CARRIER", "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "DEST", "DEP_DEL15"]

stages = [] # stages in our Pipeline

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# COMMAND ----------

# MAGIC %md The following cells shows how the <code>VectorAssembler</code> would be used to transform numeric columns of which we don't have any in our data set. The code is mainly here for future reference.

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
# numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
# assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
#assembler = VectorAssembler(inputCols=['DAY_OF_WEEK', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', 'DEP_DELAY'], outputCol="features")
#assembler = VectorAssembler(inputCols=['DAY_OF_WEEK'], outputCol="rawFeatures")
#indexer = VectorIndexer(inputCol="rawFeatures", outputCol="features2")

# COMMAND ----------

# MAGIC %md Convert label into label indices through StringIndexer

# COMMAND ----------

label_stringIdx = StringIndexer(inputCol="ARR_DEL15", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

assemblerInputs = [c + "classVec" for c in categoricalColumns]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml import Pipeline

# Create a Pipeline.
pipeline = Pipeline(stages=stages)

# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.

pipelineModel = pipeline.fit(delays)
delays = pipelineModel.transform(delays)

# Keep relevant columns
selectedcols = ["label", "features"] + cols
delays = delays.select(selectedcols)
display(delays)

# COMMAND ----------

# MAGIC %md ###Creating training and testing data sets

# COMMAND ----------

# MAGIC %md It is vital that prior to training a machine learning model, we split out a fraction of the data that is not used during training which can then be used for testing. In this example we will dedicated 80% of the entire data set to training and the remaining 20% is put aside for testing later on.
# MAGIC 
# MAGIC Since this will create two new data frames, we apply the <code>cache</code> method to both of them to make sure they are kept in memory in our cluster. This is important moving forward as during the training phase lots of iterations will be happening which would make it computationally very expensive to read from disk over and over again.

# COMMAND ----------

(trainingData, testData) = delays.randomSplit([0.8, 0.2], seed=42)
trainingData.cache()
testData.cache()

trainingData.count(), testData.count()

# COMMAND ----------

# MAGIC %md ##Creating the model object

# COMMAND ----------

# MAGIC %md Before we can train the ML model, we need to specify the ML object. This is where we choose which of the algorithms to go with and what hyperparameters should be used during training. As you can see our first model will be build using a Logistic Regression.
# MAGIC 
# MAGIC The [classification module of MLlib](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.classification) also offers other algorithms to experiment with.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# COMMAND ----------

# MAGIC %md To view all available hyperparameters of the Logistic Regression, run the following command.

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# MAGIC %md ##Training the Logistic Regression model

# COMMAND ----------

# MAGIC %md This is where the actual magic happens. The Logistic Regression algorithm is going through a maximum of 10 iterations trying improve upon its errors. Depending on the complexity of the model and the amount of data to learn from, this step can take quite a while.

# COMMAND ----------

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

# COMMAND ----------

# MAGIC %md Once training is complete, we now have a trained model available which will hopefully be able to accurately predict delays on our test data which it has not previously seen.

# COMMAND ----------

# MAGIC %md ##Scoring

# COMMAND ----------

# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

# MAGIC %md ##Model Evaluation

# COMMAND ----------

# MAGIC %md We can now have a look at the model's prediction alongside the label column (formerly known as *ARR_DEL15*) and the features used for training.
# MAGIC 
# MAGIC To make a first assessment of the classifier's performance, we can simply compare the *label* and *prediction* columns row by row. Everytime the two match, the model made a correct prediction.

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability")
display(selected)

# COMMAND ----------

# MAGIC %md Of course this kind of assessment is rather bothersome when done manually. Thankfully, MLlib has some helper functions that can help with model evaluation.

# COMMAND ----------

# MAGIC %md ###Area Under the Curve (AUC)

# COMMAND ----------

# MAGIC %md Area Under the Curve - or AUC for short - can range from 0 to 1 with 0 being bad and 1 being perfect but actually too good to be true. Another metric supported by this evaluator currently is Area Under PR and could be selected using the <code>setMetricName</code> method.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

# MAGIC %md ###Precision

# COMMAND ----------

# MAGIC %md Precision is another way of measuring the quality of a trained ML model. It is defined as TP / TP + FP. As with most of the quality measure, 0 is the worst possible value and 1 the maximum achievable value. It can be thought of answering the question "how many of the returned labels are right".
# MAGIC 
# MAGIC In the following cell we are looking at the precision for the two classes 0 (no arrival delay above 15 min) and 1 (arrival delay of more than 15 min).

# COMMAND ----------

trainingSummary = lrModel.summary

for i, prec in enumerate(trainingSummary.precisionByLabel):

        print("label %d: %s" % (i, prec))

# COMMAND ----------

# MAGIC %md ###Recall

# COMMAND ----------

# MAGIC %md Recall - somtimes also called True Positive Rate (TPR) - is another common way of measuring model quality and is defined as TP / TP + FN. It can be thought of answering the question "number of delayed arrivals found correctly".

# COMMAND ----------

print("Recall by label:")

for i, rec in enumerate(trainingSummary.recallByLabel):

        print("label %d: %s" % (i, rec))

# COMMAND ----------

# MAGIC %md ###Confusion Matrix

# COMMAND ----------

# MAGIC %md The confusion matrix shows the actual vs. predicted values for each class.

# COMMAND ----------

cm = selected.select('label','prediction')
cm.crosstab('label','prediction').show()

# COMMAND ----------

# MAGIC %md #Additional attempts with other algorithms

# COMMAND ----------

# MAGIC %md We can re-use most of the code that we ran above to create our first model based on Logistic Regression and feed it to other algorithms which will maybe yield more promising results.

# COMMAND ----------

# MAGIC %md ###Decision Tree

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5)

print(dt.explainParams())

# COMMAND ----------

# Train model with Training Data
dtModel = dt.fit(trainingData)

# COMMAND ----------

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

# COMMAND ----------

# MAGIC %md Similarly to what we've seen before, the following command performs the predictions for the hold-out data.

# COMMAND ----------

predictions = dtModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

selected = predictions.select("label", "prediction", "probability")
display(selected)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md Judging by AUC, the simple decision tree model performs worse than the previously trained logistic regression. 
# MAGIC 
# MAGIC Let's also train an Random Forest model and then compare it. 

# COMMAND ----------

# MAGIC %md ##Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(trainingData)

predictions = rfModel.transform(testData)

selected = predictions.select("label", "prediction", "probability")
display(selected)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

cm = selected.select('label','prediction')
cm.crosstab('label','prediction').show()

# COMMAND ----------

# MAGIC %md The Random Forest also performs slightly worse than the initial Logistic Regression based model.
# MAGIC 
# MAGIC It is important to note, that we have included the *DEP_DEL15* variable as a predictor in all our models. This variable captures whether a given flight was already delayed when departing its origin. Whether this information can actually be made available in due time for the model to work with to predict arrival delay would need to be thoroughly investigated.