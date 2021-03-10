# WORKING WITH RESILIENT DISTRIBUTED DATASETS
# Datasets which are distributed on many nodes (many computers)

# SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)
print("The master of Spark Context in the PySpark shell is", sc.master)

# ------------------> Loading data in Spark

#1 Create a python list of numbers from 1 to 100 
numb = range(1, 100)
# Load the list into PySpark  
spark_data = sc.parallelize(numb)
#2 Load a local file into PySpark shell
lines = sc.textFile(file_path)


# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)
# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))
# Result : The file type of fileRDD is <class 'pyspark.rdd.RDD'>
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())
# Result : Number of partitions in fileRDD is 2
# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())
# Result : Number of partitions in fileRDD_part is 5


# ------------------> 2 kinds of operations : Transformations and actions

# Transformation : map(), filter(), flatMap(), and union()
# Transformations are lazy : they will be applied only if a actions is called

# Actions : collect(), take(N), first() and count()

# Example 1 :
# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x*x*x)
# Collect the results
numbers_all = cubedRDD.collect()
# Print the numbers from numbers_all
for numb in numbers_all:
	print(numb)

# Example 2 :
# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in line.split(" "))
# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())
# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4): 
  print(line)


# ------------------> pair RDDs : Spark dictionaries

# Create a pair RDD from a list of key-value tuple
my_tuple = [('Sam', 23), ('Mary', 34), ('Peter', 25)]
pairRDD_tuple = sc.parallelize(my_tuple)
# Create a pair RDD from a regular RDD
my_list = ['Sam 23', 'Mary 34', 'Peter 25']
regularRDD = sc.parallelize(my_list)
pairRDD_RDD = regularRDD.map(lambda s: (s.split(' ')[0], s.split(' ')[1]))


# 4 kinds of transformations for pair RDDs : reduceByKey(), sortByKey(), groupByKey(), join()

# reduceByKey() :
regularRDD = sc.parallelize([("Messi", 23), ("Ronaldo", 34), ("Neymar", 22), ("Messi", 24)])
pairRDD_reducebykey = regularRDD.reduceByKey(lambda x,y : x + y)
pairRDD_reducebykey.collect()
# Reusult : [('Neymar', 22), ('Ronaldo', 34), ('Messi', 47)]

# sortByKey() :
pairRDD_reducebykey_rev = pairRDD_reducebykey.map(lambda x: (x[1], x[0]))
pairRDD_reducebykey_rev.sortByKey(ascending=False).collect()
# Result : [(47, 'Messi'), (34, 'Ronaldo'), (22, 'Neymar')]

# groupByKey() :
airports = [("US", "JFK"),("UK", "LHR"),("FR", "CDG"),("US", "SFO")]
regularRDD = sc.parallelize(airports)
pairRDD_group = regularRDD.groupByKey().collect()
for cont, air in pairRDD_group:
    print(cont, list(air))
# Result :  FR ['CDG']
#           US ['JFK', 'SFO']
#           UK ['LHR']

# join()
RDD1 = sc.parallelize([("Messi", 34),("Ronaldo", 32),("Neymar", 24)])
RDD2 = sc.parallelize([("Ronaldo", 80),("Neymar", 120),("Messi", 100)])
RDD1.join(RDD2).collect()
# Result : [('Neymar', (24, 120)), ('Ronaldo', (32, 80)), ('Messi', (34, 100))]


# An action for pair RDDs ; countByKey()
# Transform the rdd with countByKey()
total = Rdd.countByKey()
# What is the type of total?
print("The type of total is", type(total))
# Iterate over the total and print the output
for k, v in total.items(): 
  print("key", k, "has", v, "counts")



# ------------------> Machine Learning with pyspark.mllib

# Import the library for ALS
from pyspark.mllib.recommendation import ALS
# Import the library for Logistic Regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# Import the library for Kmeans
from pyspark.mllib.clustering import KMeans

