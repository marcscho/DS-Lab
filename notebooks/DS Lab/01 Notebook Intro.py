# Databricks notebook source
# MAGIC %md #Notebook Intro

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Notebooks are the central place in Databricks where code is run and where collaboration is happening between Data Engineers and Data Scientists.

# COMMAND ----------

# MAGIC %md Double click into any of the cells to see the mark-down commands that were used to create them.

# COMMAND ----------

# MAGIC %md Using the <code>%md</code> magic command, a cell can be turned into a mark-down cell containing nothing but text. These cells cannot contain any executable code.
# MAGIC 
# MAGIC Various levels for heading are available as outlined in the next cell.

# COMMAND ----------

# MAGIC %md
# MAGIC # Level 1 heading
# MAGIC ## Level 2 heading
# MAGIC ### Level 3 heading 
# MAGIC #### Level 4 heading

# COMMAND ----------

# MAGIC %md 
# MAGIC Various keyboard shortcuts are available from the shortcuts menu. To run a cell, you may also use the Ctrl + Enter or Shift + Enter keyboard shortcuts.

# COMMAND ----------

# MAGIC %md
# MAGIC You can also format text in *italic* or **bold** or ***both***.

# COMMAND ----------

# MAGIC %md
# MAGIC A notebook is created in any of the following languages: Python, R, Scala or SQL.
# MAGIC 
# MAGIC Through the following magic commands it is possible to switch language for any cell, regardless of the language chosen for the notebook.
# MAGIC 
# MAGIC * <code>%python</code> interprets the following lines of code as Python code
# MAGIC * <code>%r</code> interprets the following lines of code as R code
# MAGIC * <code>%scala</code> interprets the following lines of code as Scala code
# MAGIC * <code>%SQL</code> interprets the following lines of code as SQL code
# MAGIC 
# MAGIC In addition, the following magic commands may also be used:
# MAGIC 
# MAGIC * <code>%sh</code> allows you to execute shell code in your notebook
# MAGIC * <code>%fs</code> allows you to use dbutils filesystem commands
# MAGIC 
# MAGIC Every new cell added after a cell starting with a magic command, the next cell will again be in the default language of the notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC Refer to the [following web page](https://docs.databricks.com/user-guide/notebooks/notebook-use.html#language-magic) for more information on mark-down capabilities and basics of running notebooks.