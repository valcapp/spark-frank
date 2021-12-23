from pyspark.sql import SparkSession
from pyspark.sql import functions as spf
from pathlib import Path

# create session
spark = SparkSession.builder.appName("FriendsByAge").getOrCreate()

#  get data inferring schema
df = spark.read\
            .option("header", "true")\
            .option("inferSchema", "true")\
            .csv(f"file:///{Path(__file__).parent.parent}/data/fakefriends-header.csv")

#  check schema
print(f"Inferred schema:\n{df._jdf.schema().treeString()}")

# create temp view in spark session
df.createOrReplaceTempView('people')
# print('Dataframe head')
# spark.sql('SELECT * FROM people LIMIT 5').show()

#  make aggregation with df object
print(f"Average friends by age:")
df.groupBy('age').avg()\
    .select('age', 'avg(friends)')\
    .sort('age')\
    .show()

# print('Dataframe head')
# spark.sql('SELECT * FROM people LIMIT 5').show()

#  make aggregation with SQL command
print(f"Average friends by age:")
spark.sql(
    'SELECT age, round(avg(friends), 1) AS avg_friends '
    'FROM people GROUP BY age ORDER BY age'
).show()

print(f"Average friends by age:")
df.groupBy('age').agg(
    spf.round(spf.avg('friends'), 1).alias('avg_firends')
).sort('age').show()

# close session
spark.stop()
