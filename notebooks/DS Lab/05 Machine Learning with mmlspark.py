# Databricks notebook source
# MAGIC %md # Machine Learning with `mmlspark`
# MAGIC 
# MAGIC In this notebook, we are going to work another Logistic Regression model but this time using an alternative library: *mmlspark*. This library, created by Microsoft, is open source and, among other benefits, greatly reduces the amount of code one needs to write to train machine learning models.
# MAGIC 
# MAGIC For further information on *mmlspark*, please visit: https://github.com/Azure/mmlspark
# MAGIC 
# MAGIC To start off, we need to add the library. To do so, do the following:
# MAGIC 
# MAGIC * 1) In the navigation pane on the left, click "Worspace"
# MAGIC * 2) Click the downward chevron icon
# MAGIC * 3) Click "Create" -> "Library"
# MAGIC * 4) Select "Maven Coordinate" and enter the following under "Coordinate" <code>Azure:mmlspark:0.15</code>
# MAGIC * 5) Ensure that the library will also be attached to your cluster which is already running.
# MAGIC 
# MAGIC Note that as a pre-requisite, our cluster must be running at least Spark 2.1 and Scala 2.11.
# MAGIC 
# MAGIC Once the library is attached to the cluster, we are ready to import it and work with it. <code>help(mmlspark)</code> lists all the functions and methods that *mmlspark* can help with.

# COMMAND ----------

import numpy as np
import pandas as pd
import mmlspark

help(mmlspark)

# COMMAND ----------

# MAGIC %md Let's read data from our SQL table again and create another dataframe to work with.

# COMMAND ----------

delays = spark.sql("SELECT * FROM delays_table")
delays.cache()
delays.count()

# COMMAND ----------

# MAGIC %md Again we are dropping columns that contain duplicate information.

# COMMAND ----------

delays = delays.drop("ARR_DELAY", "DEP_DELAY")
delays.printSchema()

# COMMAND ----------

# MAGIC %md Now let's read the data and split it to train and test sets. Note that we use the same split ratio as well as random number seed to ensure comparability between results.

# COMMAND ----------

train, test = delays.randomSplit([0.8, 0.2], seed=42)
train.cache()
test.cache()

# COMMAND ----------

# MAGIC %md `TrainClassifier` can be used to initialize and fit a model, it wraps SparkML classifiers.
# MAGIC You can use `help(mmlspark.TrainClassifier)` to view the different parameters.
# MAGIC 
# MAGIC Note that it implicitly converts the data into the format expected by the algorithm: tokenize
# MAGIC and hash strings, one-hot encodes categorical variables, assembles the features into a vector
# MAGIC and so on.  The parameter `numFeatures` controls the number of hashed features.

# COMMAND ----------

from mmlspark import TrainClassifier
from pyspark.ml.classification import LogisticRegression

model = TrainClassifier(model=LogisticRegression(), labelCol="ARR_DEL15", numFeatures=256).fit(train)
model.write().overwrite().save("DelaysLRModel.mml")

# COMMAND ----------

# MAGIC %md Even though the training process takes longer than before, overall time is greatly be reduced due to the fact that we do not need to manually transform the dataframe for machine learning. 

# COMMAND ----------

from mmlspark import ComputeModelStatistics, TrainedClassifierModel

predictionModel = TrainedClassifierModel.load("DelaysLRModel.mml")
prediction = predictionModel.transform(test)
metrics = ComputeModelStatistics().transform(prediction)

# COMMAND ----------

# MAGIC %md Let's show the model metrics. As you can see, the AUC is almost identical with the Logistic Regression which was trained previously. 

# COMMAND ----------

metrics.show()

# COMMAND ----------

# MAGIC %md Finally, we save the model so it can be used in a scoring program.

# COMMAND ----------

model.write().overwrite().save("DelaysLRModel.mml")