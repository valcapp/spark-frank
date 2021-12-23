from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

import argparse
# import os
from pathlib import Path
from itertools import combinations_with_replacement

def parse_args() -> None:
    parser = argparse.ArgumentParser(
        prog='movie-recommendations',
        description='Suggest 10 movie recommendations based on similarity to movieID argument')
    parser.add_argument(
        'movieID', type=int,
        help='movieID integer to base recommendations on')
    return parser.parse_args()

def getMovieNames(spark: SparkSession, resource_path: str) -> DataFrame:
    movieNamesSchema = StructType([ 
        StructField("movieID", IntegerType(), True),
        StructField("movieTitle", StringType(), True)
    ])
    return \
        spark.read\
            .option('sep', '|')\
            .option('charset', 'ISO-8859-1')\
            .schema(movieNamesSchema)\
            .csv(resource_path)

def getMovieName(movieNames: DataFrame, movieID: int) -> str:
    return \
        movieNames.where(sf.col("movieID") == movieID)\
            .first()['movieTitle']

def getMovies(spark: SparkSession, resource_path: str) -> DataFrame:
    moviesSchema = StructType([
        StructField("userID", IntegerType(), True),
        StructField("movieID", IntegerType(), True),
        StructField("rating", IntegerType(), True),
        StructField("timestamp", LongType(), True)
    ])
    return \
        spark.read\
            .option('sep', '\t')\
            .schema(moviesSchema)\
            .csv(resource_path)

def getMoviePairs(ratings: DataFrame) -> DataFrame:
    joined = ratings.alias('x')\
        .join(ratings.alias('y'), on=(
            (sf.col('x.userID') == sf.col('y.userID'))
            & (sf.col('x.movieID') < sf.col('y.movieID'))
        ))\
        .select(
            # x
            sf.col('x.movieID').alias('movie_x'),
            sf.col('x.rating').alias('rating_x'),
            # y
            sf.col('y.movieID').alias('movie_y'),
            sf.col('y.rating').alias('rating_y'),
        )
    return joined

def cosineSimilarity(pairs: DataFrame) -> DataFrame:
    scores = pairs
    for i, j in combinations_with_replacement('xy', 2):
        scores = scores.withColumn(i+j, sf.col(f"rating_{i}") * sf.col(f"rating_{j}"))
    similarity = scores\
        .groupBy('movie_x', 'movie_y')\
        .agg(
            sf.sum('xy')\
                .alias('numerator'),
            (sf.sqrt(sf.sum('xx')) * sf.sqrt(sf.sum('yy')))\
                .alias('denominator'),
            sf.count('xy')\
                .alias('coOccurrence')
        )\
        .select(
            'movie_x', 'movie_y',
            sf.when(sf.col('denominator') != 0, 
                    sf.col('numerator')/sf.col('denominator')
                )
                # .otherwise(0)
                .alias('score'),
            'coOccurrence'
        )
    return similarity

def getTopSimilarTo(similarity: DataFrame,
                movieID: int, limit: int = 10
                ) -> DataFrame:
    scoreThreshold = .97
    coOccurrenceThreshold = 50.0
    topSimilar = similarity\
        .where(
            (
                (sf.col('movie_x')==movieID)
                | (sf.col('movie_y')==movieID)
            )
            & (sf.col('score') > scoreThreshold)
            & (sf.col('coOccurrence') > coOccurrenceThreshold)
        )\
        .sort('score', ascending=False)\
        .limit(limit)\
        .select(
            sf.when(sf.col('movie_x') != movieID,
                    sf.col('movie_x')
                ).otherwise(sf.col('movie_y'))\
                .alias('movieID'),
            sf.round('score', 3).alias('score'),
            sf.col('coOccurrence').alias('strength')
        )
    return topSimilar


def main():
    movieID = parse_args().movieID
    spark = SparkSession.builder.appName("MovieRecommendations").master("local[*]").getOrCreate()
    # pwd = os.path.abspath('')
    pwd = Path(__file__).parent
    movieNames = getMovieNames(spark, f"file:///{pwd}/../data/ml-100k/u.item")
    movies = getMovies(spark, f"file:///{pwd}/../data/ml-100k/u.data")
    print(f"Calculating Top 10 recommendations for movie: '{getMovieName(movieNames, movieID)}'")
    ratings = movies.select('userID', 'movieID', 'rating')
    moviePairs = getMoviePairs(ratings)
    similarity = cosineSimilarity(moviePairs)
    mostSimilar = getTopSimilarTo(similarity, movieID)\
        .join(movieNames, on='movieID').sort('score', ascending=False)
    mostSimilar.show(truncate=False)
    spark.stop()

if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')
