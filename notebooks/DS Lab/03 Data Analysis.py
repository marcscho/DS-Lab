# Databricks notebook source
# MAGIC %md
# MAGIC # Data Analysis

# COMMAND ----------

# MAGIC %md The most important piece of documentation for this part of the lab is the official documentation of the [PySpark SQL Module](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) which documents most of the functions and methods we will be using.

# COMMAND ----------

# MAGIC %md ## Data Set
# MAGIC 
# MAGIC The data set used for this lab will be the publically available airline delays data set, containing more than 100 mio. commercial flights conducted in the USA from 1987 to 2012.
# MAGIC 
# MAGIC The aim is to build a machine learning model that can hopefully predict arrival delay for future flights. 

# COMMAND ----------

# MAGIC %md ## Data Loading
# MAGIC 
# MAGIC In a first step we will need to load the data for further analysis. For the purpose of this walkthrough, the data set is stored in an Azure Blob Storage and parquet file format and has to be loaded from there.
# MAGIC 
# MAGIC The <code>inferSchema</code> of the <code>sqlContext.read.format</code> is set to infer the most appropriate data types when importing. Identifying appropriate data types is not in scope of this lab. Hence, we make use of the <code>inferSchema</code> option.
# MAGIC 
# MAGIC We cache the resulting dataframe using <code>cache()</code> to force it to be held in memory in our cluster for quicker calculation and analysis. This is especially useful when executing operations upon the same data frame times and times again.
# MAGIC 
# MAGIC The <code>count</code> method is used to count the number of observations/rows in the data frame. It is also the action that Spark requires to actually run a job due to its lazy evaluation principle. It will tell us that there are some 148 mio. rows in our data set. 

# COMMAND ----------

delays = sqlContext.read.format('parquet').options(header='true', inferSchema='true').load('/mnt/dslab01/airdelays_full.parquet')
delays.cache()
delays.count()

# COMMAND ----------

# MAGIC %md ##Initial inspection

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Lets have a look at what the the data frame looks like. For this we have several options.
# MAGIC 
# MAGIC * The <code>show()</code> method on the data frame which shows the first few rows of the data frame
# MAGIC * The <code>head()</code> method also shows the selected number of rows of the data frame
# MAGIC * The <code>display</code> function which does the same but returns an interactive table with additional options, e.g. for sorting
# MAGIC 
# MAGIC For the remainder of this course, we will mainly use the <code>display</code> function for reasons of usability.

# COMMAND ----------

delays.show(1)

# COMMAND ----------

delays.head(2)

# COMMAND ----------

display(delays)

# COMMAND ----------

# MAGIC %md We briefly mentioned above that <code>inferSchema</code> is making an educated guess at what the most appropriate data types for each column in the data frame could be. We can now use the <code>.printSchema</code> method to show the schema that was applied during the import.

# COMMAND ----------

delays.printSchema()

# COMMAND ----------

# MAGIC %md ##Reducing columns

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC As you can see above, there a some very similar columns. E.g. *DEST_AIRPORT_ID* and *DEST* contain the same information - the destination of the flight - but encoded in different ways. 
# MAGIC 
# MAGIC The same holds true for *ARR_DELAY, ARR_DELAY_NEW, ARR_DEL15* and *ARR_DELAY_GROUP* as well as for its counterparts on the departure side: *DEP_DELAY, DEP_DELAY_NEW, DEP_DEL15* and *DEP_DELAY_GROUP.*
# MAGIC 
# MAGIC One approach to reducing the number of columns is through the <code>drop</code> method which takes a list of column names as an argument. The column names specified will simply be removed from the data frame. 
# MAGIC The other approach is the opposite where we use the <code>select</code> method to explicitly select which columns to keep in the data frame, removing all others in the process. In the code block below, you will see an example of using both.
# MAGIC 
# MAGIC **Note:** In real world, the exclusion of columns requires a more rigorous procedure and information should not be disregarded a priori.

# COMMAND ----------

delays = delays.drop('ARR_DELAY_NEW', 'ARR_DELAY_GROUP')
delays = delays.select('DAY_OF_WEEK', 'MONTH', 'UNIQUE_CARRIER','ORIGIN_STATE_ABR','ORIGIN','DEST_STATE_ABR','DEST','DEP_DEL15','DEP_DELAY','ARR_DEL15','ARR_DELAY')
#unique carrier, month, dest, origin added
delays.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC *DEP_DELAY* contains the number of minutes an aircraft has departed its origin early. This can of course be a very powerful predictor for arrival delay which we're interested in. *DEL_15* encodes the departure delay in a binary indicator variable: 0 = flight was less than 15 min. late; 1 = flight was more than 15 min. late
# MAGIC 
# MAGIC Information on arrival delays is captured with the same principle in the columns *ARR_DELAY* and *ARR_DEL15* respectively.

# COMMAND ----------

# MAGIC %md ##Dropping columns with missing values

# COMMAND ----------

# MAGIC %md Dealing with missing values is a critical part of every data analysis project. Among other things, you have to think about whether values are missing randomly or whether there might be some underlying pattern to it. For example, in a survey that asks for a person's income data, leaving this field blank may be indicative of a relatively high salary.
# MAGIC 
# MAGIC First, let's investigate if we do indeed have missing values in our data set. To do that, we import some helper functions that we then use to loop through every column of our data frame counting rows with NULL or NaN values. The result set will then show the number of columns with such values for each column.

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
delays.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in delays.columns]).show()

# COMMAND ----------

# MAGIC %md After some deliberation we decide to just drop the columns that have missing values. With this, we will lose roughly 2% of data but also ensure that all our observations are now complete and of higher quality.
# MAGIC 
# MAGIC To actually drop the columns with missing values from our data frame, we run the command below using the <code>na.drop</code> method. The <code>count</code> simply reminds us of how many rows/observations are now left in our data set.
# MAGIC 
# MAGIC If you run the previous code block again after the next one, you will see that now no columns show any NULL or NaN values anymore.

# COMMAND ----------

delays = delays.na.drop()
delays.count()

# COMMAND ----------

# MAGIC %md ##Descriptive statistics

# COMMAND ----------

# MAGIC %md To establish a better understanding of our data we want to have a look at some standard descriptive statistics for all of our columns in the data frame. The <code>describe</code> method computes some of the most commonly use descriptive statistics for us. Of course for some of the categorical variables like *UNIQUE_CARRIER* statistics such as the mean value cannot be computed and are hence shown as NULL. Note that the count value can be interpreted as number of observations without a NULL value in this column.
# MAGIC 
# MAGIC **Note:** Without specifying any column names in the brackets, <code>describe</code> will compute these statistics for all the columns in our data frame which can potentially take a lot of time.
# MAGIC 
# MAGIC The <code>summary</code> method computes the same summary statistics as <code>describe</code> plus some more like the 1st and 3rd quartiles. As a consequence, the <code>summary</code> method is much more computationally intensive and takes significantly longer to run.

# COMMAND ----------

delays.describe().show()

# COMMAND ----------

delays.describe('ARR_DELAY','DEP_DELAY').show()

# COMMAND ----------

# MAGIC %md Two things that are peculiar are the minimum values for *ARR_DELAY* and *DEP_DELAY* which are both negative. Will it is certainly conceivable for a flight to leave or arrive a few minutes ahead of schedule, negative delays of several hundred minutes - up to 23 hours in the most extreme cases - seem very strange.
# MAGIC 
# MAGIC As the aim of this exercise will ultimately be to build a ML model that can predict delayed flights, one could think that it's probably rather uncommon for flights to depart ahead of time but arrive late. We can use the <code>filter</code> method to filter for flights with a negative departure delay which will then tell us that a surprising 58 million flights did depart early, accounting for approx. 40% of all flights on record.

# COMMAND ----------

delays.filter(delays.DEP_DELAY < 0).count()

# COMMAND ----------

# MAGIC %md The following command combines two filters with a logical 'and' and will show us that 13.5 million flights that left their origin ahead of time still arrived delayed.

# COMMAND ----------

delays.filter(delays.DEP_DELAY < -60).count()

# COMMAND ----------

delays.filter(delays.DEP_DELAY < 0).filter(delays.ARR_DELAY > 0).count()

# COMMAND ----------

# MAGIC %md We can use the <code>crosstab</code> method to create a contingency table for on-time/late departures vs. on-time/late arrivals.

# COMMAND ----------

delays.crosstab('DEP_DEL15','ARR_DEL15').show()

# COMMAND ----------

# MAGIC %md To quantify the relationship between these two variables, we can also compute their correlation coefficient using the <code>corr</code> function.
# MAGIC 
# MAGIC A correlation coefficient of 0.88 is indicative of strong positive correlation meaning that higher departure delays will lead to higher arrival delays and lower departure delays to lower arrival delays. 
# MAGIC 
# MAGIC That is of course an interesting insight that we will be leveraging for our predictive model later on. 

# COMMAND ----------

delays.corr('ARR_DELAY','DEP_DELAY')

# COMMAND ----------

# MAGIC %md ##Data visualization

# COMMAND ----------

# MAGIC %md With the help of the <code>display</code> function we can change the raw data output to an in-line plot without needing to write any additional code - ain't that nice? The example below shows a histogram of the negative departure delay times.
# MAGIC 
# MAGIC Note that depending on the question at hand you may want to switch between various plot types which can be done through the plot options.

# COMMAND ----------

display(delays.filter(delays.DEP_DELAY < 0))

# COMMAND ----------

# MAGIC %md ###Arrival Delays

# COMMAND ----------

# MAGIC %md First we are going to do some analysis on the arrival delays which is also what we are interested in predicting. First we will look at the descriptive statistics returned by the <code>describe</code> method again.

# COMMAND ----------

delays.describe('ARR_DELAY').show()

# COMMAND ----------

# MAGIC %md We will now be using some bivariate visualizations to get a better understanding of our arrival delays.
# MAGIC 
# MAGIC **Feel free to further explore the data using your own ideas and code.**

# COMMAND ----------

# MAGIC %md ####Arrival Delays per Airline

# COMMAND ----------

# MAGIC %md One way of comparing arrival delay information across carriers would be using the <code>display</code> function and selection an aggregation method through the plot options.

# COMMAND ----------

display(delays.select('ARR_DELAY', 'UNIQUE_CARRIER'))

# COMMAND ----------

# MAGIC %md Alternatively, we can also calculate the mean arrival delay per airline using the following <code>groupby</code> method:

# COMMAND ----------

display(delays.groupBy('UNIQUE_CARRIER').avg('ARR_DELAY'))

# COMMAND ----------

# MAGIC %md ####Arrival Delays by Destination Airport

# COMMAND ----------

# MAGIC %md We still have the *ARR_DEL15* column in our data frame which is binary categorical variable indicating whether a flight was delayed upon arrival by 15 minutes or not. We can conveniently run an average over that which will then return the percentage of flights that arrived late for the respective destination. 
# MAGIC 
# MAGIC To view the destination airports with the highest percentages of delayed incoming flights on top of the list, we simply click the header of the result set.

# COMMAND ----------

display(delays.groupBy('DEST').avg('ARR_DEL15'))

# COMMAND ----------

# MAGIC %md The table above gives us no indication of the number of flights that have their destination in a respective airport. If we want to incorporate that, we need to bring in a secondary column with number of flights that landed in the destination airport we are grouping by. This can be done using the <code>agg</code> method which will then show us that the highest percentage of delayed flights at a major destination airport is occuring at EWR (Newark Liberty International Airport in Newark, NJ) followed by SFO (San Francisco, CA).

# COMMAND ----------

display(delays.groupBy('DEST').agg({"ARR_DEL15": "avg", "ARR_DELAY": "count"}))

# COMMAND ----------

# MAGIC %md ####Arrival Delays by State

# COMMAND ----------

# MAGIC %md To get a better feeling for whether there is maybe a underlying regional pattern for the arrival delays, we can also use inline visualizations to plot on a map.

# COMMAND ----------

display(delays.groupBy('DEST_STATE_ABR').avg('ARR_DELAY'))

# COMMAND ----------

# MAGIC %md ####Arrival Delays by Week Day

# COMMAND ----------

# MAGIC %md Surely the average arrival delay is different for eack day of the week, mainly caused by business travels - or the lack thereof on weekends.

# COMMAND ----------

display(delays.select('DAY_OF_WEEK','ARR_DEL15'))

# COMMAND ----------

# MAGIC %md ####Arrival Delays by Month

# COMMAND ----------

display(delays.groupBy('MONTH').avg('ARR_DELAY').orderBy('MONTH', ascending = 1))

# COMMAND ----------

# MAGIC %md ###Departure Delays

# COMMAND ----------

# MAGIC %md In addition to the mean departure delay, we also want to look at the standard deviation which we get again using the <code>agg</code> method we have already briefly worked with.

# COMMAND ----------

delays = delays.withColumn("DEP_DELAY2", delays.DEP_DELAY)
del_per_day = delays.groupby('DAY_OF_WEEK').agg({"DEP_DELAY2": "stddev", "DEP_DELAY": "avg"})
display(del_per_day)

# COMMAND ----------

# MAGIC %md We would like to get a better understanding for whether there specific states that that have significantly bigger average departure delays than other states.

# COMMAND ----------

delays_agg = delays.groupby('ORIGIN_STATE_ABR').agg({'DEP_DELAY': 'mean'})
display(delays_agg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Looking at the map above, it appears that the states of Delaware (DE), New Jersey (NJ), Illinois (IL) and Georgia (GA) have the hightest average departure delays across the entire time span of the data set. For all of these states, the average delay is greater than 10 minutes. You can confirm this by switching between the map and table view.
# MAGIC 
# MAGIC The questions is whether this aggregated view by state is really representative of what's happening at the largest origin airports. For example, could it be possible that average departure delay would be much higher in the state of California if we were only looking at its larger airports (LAX, SFO)?
# MAGIC 
# MAGIC To find an answer to this question, lets first start by summarizing the number of departing flights per airport while also keeping the state information in the data set.

# COMMAND ----------

delays_cnt = delays.groupby('ORIGIN_STATE_ABR', 'ORIGIN').count()
delays_cnt = delays_cnt.sort('count', ascending = False)
display(delays_cnt)

# COMMAND ----------

# MAGIC %md Now let's arbitrarily remove all airports with fewer than 500'000 departures all-time and save the name of the airports, that match this criteria, in a separate table <code>big_airports</code>.

# COMMAND ----------

delays_cnt = delays_cnt.filter('count > 500000')
big_airports = delays_cnt.select('ORIGIN')
display(big_airports)

# COMMAND ----------

# MAGIC %md Next, let's create a new dataframe with the average delay per airport. We will overwrite the <code>delays_agg</code> since we no longer need it and to make the code a bit easier to read.

# COMMAND ----------

delays_agg = delays.groupby('ORIGIN_STATE_ABR', 'ORIGIN').agg({'DEP_DELAY': 'mean'})
delays_agg = delays_agg.withColumnRenamed('avg(DEP_DELAY)','AVG_DEP_DELAY')
display(delays_agg)

# COMMAND ----------

# MAGIC %md We now join the <code>big_airports</code> dataframe with <code>delays_agg</code> dataframe, keeping only matching rows.
# MAGIC 
# MAGIC Then we will keep only the airport with the highest average departure delay per state.

# COMMAND ----------

new = big_airports.join(delays_agg, on = "ORIGIN", how = "leftouter")
new = new.groupBy('ORIGIN_STATE_ABR').max('AVG_DEP_DELAY')
display(new)

# COMMAND ----------

# MAGIC %md ##Create a SQL table

# COMMAND ----------

# MAGIC %md We will now be creating a SQL table which allow us to persist our current data frame outside of this notebook. That is required because the code for building a ML model is implemented in another notbeook. Before we do it, we load the original parquet file again and apply the relevant transformations again to ensure everybody will be working with the same data set moving forward.

# COMMAND ----------

delays = sqlContext.read.format('parquet').options(header='true', inferSchema='true').load('/mnt/dslab01/airdelays_full.parquet')

delays = delays.drop('ARR_DELAY_NEW', 'ARR_DELAY_GROUP')
delays = delays.select('DAY_OF_WEEK', 'MONTH', 'UNIQUE_CARRIER','ORIGIN_STATE_ABR','ORIGIN','DEST_STATE_ABR','DEST','DEP_DEL15','DEP_DELAY','ARR_DEL15','ARR_DELAY')
#unique carrier, month, dest, origin added

delays = delays.na.drop()
delays.write.saveAsTable('delays_table')