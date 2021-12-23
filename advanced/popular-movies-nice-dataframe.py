# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:28:00 2020

@author: Frank
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import codecs

# from pathlib import Path
import os
from time import perf_counter
startTime = perf_counter()

from pathlib import Path

def loadMovieNames():
    movieNames = {}
    # CHANGE THIS TO THE PATH TO YOUR u.ITEM FILE:
    with codecs.open(f"{Path(__file__).parent.parent}/data/ml-100k/u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

nameDict = spark.sparkContext.broadcast(loadMovieNames())

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

movieCounts = moviesDF.groupBy("movieID").count()

# Create a user-defined function to look up movie names from our broadcasted dictionary
def lookupName(movieID):
    return nameDict.value[movieID]

lookupNameUDF = sf.udf(lookupName)

# Add a movieTitle column using our new udf
moviesWithNames = movieCounts.withColumn("movieTitle", lookupNameUDF(sf.col("movieID")))

# Sort the results
sortedMoviesWithNames = moviesWithNames.orderBy(sf.desc("count"))

# Grab the top 10
sortedMoviesWithNames.show(10, False)

# Stop the session
print(f'Executed in {perf_counter() - startTime:.3f} s')
spark.stop()
