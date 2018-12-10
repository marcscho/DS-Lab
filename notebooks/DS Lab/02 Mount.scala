// Databricks notebook source
// MAGIC %md # Data import from Azure Blob Storage

// COMMAND ----------

// MAGIC %md ##Mount Azure Blob Storage

// COMMAND ----------

# dbutils.fs.unmount("/mnt/dslab01")

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://dslab01@airdelays.blob.core.windows.net/",
  mountPoint = "/mnt/dslab01",
  extraConfigs = Map("fs.azure.account.key.airdelays.blob.core.windows.net" -> "<insert key here>"))

// COMMAND ----------

// MAGIC %md
// MAGIC Additional information on mounting blob storage can be found [here](https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-storage.html#mount-azure-blob-storage).