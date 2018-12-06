// Databricks notebook source
// MAGIC %md # Data import from Azure Blob Storage

// COMMAND ----------

// MAGIC %md ##Mount Azure Blob Storage

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://dslab01@airdelays.blob.core.windows.net/",
  mountPoint = "/mnt/dslab01",
  extraConfigs = Map("fs.azure.account.key.airdelays.blob.core.windows.net" -> "xHVIVT+y3SaE+GWLffq1YgGpDinwBcnsaElSi/2YKidwEJzdpx7Iump40b0cFg/O9CsMOfmCr8QtIL+JhRv9LA=="))

// COMMAND ----------

// MAGIC %md
// MAGIC Additional information on mounting blob storage can be found [here](https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-storage.html#mount-azure-blob-storage).

// COMMAND ----------

// MAGIC %md ##Create a table using python

// COMMAND ----------

# %python

# dataFrame = "/mnt/airdelays"
# spark.read.format("csv").option("header","true")\
#  .option("inferSchema", "true").load(dataFrame)\
#  .createGlobalTempView("airdelays")