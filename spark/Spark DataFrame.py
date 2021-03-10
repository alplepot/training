
# ---------------> Set spark session

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()
# Print my_spark
print(my_spark)
# Print the tables in the catalog
print(spark.catalog.listTables())


# Query the dataset
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
flight_counts = spark.sql(query)
# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()


# Create pd_temp (random pandas dataframe)
pd_temp = pd.DataFrame(np.random.random(10))
# Create spark_temp from pd_temp (create a dataframe into Spark, only localy accessible)
spark_temp = spark.createDataFrame(pd_temp)
# Add spark_temp to the catalog of the Spark cluster (accessible from any Spark session)
spark_temp.createOrReplaceTempView("temp")


# Load a CSV file into Spark
file_path = "/usr/local/share/datasets/airports.csv"
airports = spark.read.csv(file_path, header = True)
# Check the schema of columns
airports.printSchema()
# Show the first 10 observations
airports.show(10)
# Generate basic statistics
airports.describe().show()


# ---------------> Deal with data

# Add a new column to a Spark dataframe
# Create the DataFrame flights
flights = spark.table("flights")
# Show the head (here, we should see that air_time is a column with the flight duration in minutes)
flights.show()
# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time / 60)


# equivalent to SQL
long_flights0 = spark.sql("select * from flights where distance > 1000")
# Filter flights by passing a string
flights = spark.table("flights")
long_flights1 = flights.filter("distance > 1000")
# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# equivalent to SQL
selected0 = spark.sql("select tailnum, origin, dest from flights")
# Select by passing strings
flights = spark.table("flights")
selected1 = flights.select("tailnum", "origin", "dest")
# select by passing columns
selected2 = flights.select(flights.origin, flights.dest, flights.carrier)

# equivalent to SQL
selected0 = sparck.sql("select origin, dest, tainum, distance/(air_time/60) as avg_speed")
# select by passing df.colNames
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)
# select by passing strings
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

#  equivalent to SQL : filter, group by, min, max
max0 = sparck.sql("select max(distance) from flights where origin = 'SEA'")
max1 = flights.filter("origin = 'PDX'").groupBy().min("distance")

#  equivalent to SQL : group by, count
groupby0 = sparck.sql("select count(*) from flights group by tailnum")
groupby1 = flights.groupBy("tailnum").count()


# Standard deviation
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F
# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")
# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()
# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()


# Join
# Rename the faa column
airports = airports.withColumnRenamed("faa","dest")
# Join the DataFrames
flights_with_airports = flights.join(airports, on = "dest", how = "leftouter")





# ---------------> Data visualization

# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')
# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''
# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)
# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()
# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()






# ---------------> Machine Learning with pyspark.ml

# NB : Spark requires numeric data for modelin
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
# Create a dummy
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))


# Replace categorical variables into vectors readable by the model
# Step 1 : Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
# Step 2 : Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")


# Make a VectorAssembler : Create a tensor with all needed variables
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")


# Create a pipeline in ordre to easily repeat all the prvious steps
# Import Pipeline
from pyspark.ml import Pipeline
# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)
# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression
# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Import the evaluation submodule
import pyspark.ml.evaluation as evals
# Create a BinaryClassificationEvaluator in order to evaluate models
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# Build a grid with a range of hyperparameters in order to test different models
# Import the tuning submodule
import pyspark.ml.tuning as tune
# Create the parameter grid
grid = tune.ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
# Build the grid
grid = grid.build()

# Cre
ate the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Fit cross validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel

# Test the model
# Use the model to predict the test set
test_results = best_lr.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))

