from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField, IntegerType, LongType

from pathlib import Path
from time import perf_counter
startTime = perf_counter()

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

# Create schema when reading u.data
schema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("movieID", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
])

# Load up movie data as dataframe
moviesDF = spark.read.option("sep", "\t").schema(schema)\
    .csv(f"file:///{Path(__file__).parent.parent}/data/ml-100k/u.data")

# Some SQL-style magic to sort all movies by popularity in one line!
# topMovieIDs = moviesDF.groupBy("movieID").count().orderBy(sf.desc("count"))  # 20s
topMovieIDs = moviesDF.groupBy("movieID").count().orderBy("count", ascending=False)  # 19s

# Grab the top 10
topMovieIDs.show(10)
print('Most popular movie id: ', topMovieIDs.first().movieID)

# Stop the session
print(f'Executed in {perf_counter() - startTime:.3f} s')
spark.stop()
