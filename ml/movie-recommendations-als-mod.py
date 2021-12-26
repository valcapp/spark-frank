"""Make movie recommendations based for userID argument"""
# %%
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS, ALSModel
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Dict, Iterable

def parse_args() -> Namespace:
    """Describe program and parse arguments

        Returns:
            Namespace: parsed arguments
        """
    parser = ArgumentParser(
        prog='ALS-movie-recommendations',
        description='Suggest 10 movie recommendations based Alternating Least Squares (ALS) model for userID argument')
    parser.add_argument(
        'userID', type=int,
        help='userID integer to make recommendations for')
    return parser.parse_args()

def getMovieNames(resource_path: str) -> Dict[int, str]:
    """Get dictionary with movie names {movieID: movieName, ...}

        Args:
            resource_path (str): path to file with lines like: movieID|movieName|...

        Returns:
            Dict[int, str]: dictionary with movie names {movieID: movieName, ...}
        """
    with open(resource_path, 'r', encoding='ISO-8859-1', errors='ignore') as file:
        rows = map(lambda line: line.split('|'), file)
        return {
            int(movieID): movieName
            for movieID, movieName, *_ in rows
        }

def getRatings(spark: SparkSession, resource_path: str) -> DataFrame:
    """Get Spark DataFrame with data about ratings by userID about movieID

        Args:
            spark (SparkSession): session to run operations with
            resource_path (str): path to resource with ratings

        Returns:
            DataFrame: data about ratings by userID about movieID
        """
    moviesSchema = StructType([
        StructField('userID', IntegerType(), True),
        StructField('movieID', IntegerType(), True),
        StructField('rating', IntegerType(), True),
        StructField('timestamp', LongType(), True),
    ])
    return \
        spark.read.option('sep', '\t')\
        .schema(moviesSchema)\
        .csv(resource_path)

def getUsersAsDataFrame(spark: SparkSession, userIDs: Iterable[str]) -> DataFrame:
    """Creates a DataFrame of userID's from iterable

        Args:
            spark (SparkSession): session to run operations
            userIDs (Iterable[str]): user IDs to cast as DataFrame

        Returns:
            DataFrame: userID's from iterable argument
        """
    return spark.createDataFrame(
        data=[(user,) for user in userIDs],
        schema=StructType([StructField('userID', IntegerType(), True)])
    )

def trainALSWithRatings(
        ratings: DataFrame,
        maxIter: int = 5,
        regParam: float = .01
    ) -> ALSModel:
    return ALS()\
        .setMaxIter(maxIter)\
        .setRegParam(regParam)\
        .setUserCol("userID")\
        .setItemCol("movieID")\
        .setRatingCol("rating")\
        .fit(ratings)


# %%
# Define main
def main():
    # %%
    # start off (read arguments and start spark)
    # userID = 50  # for development
    userID = parse_args().userID
    # start spark
    spark = SparkSession.builder.appName("ALSExample").getOrCreate()
    # %%
    # load data
    pwd = Path(__file__).parent
    movieNamesPath = f"{pwd}/../data/ml-100k/u.item"
    ratingsPath = f"file:///{pwd}/../data/ml-100k/u.data"
    movieNames = getMovieNames(movieNamesPath)
    ratings = getRatings(spark, ratingsPath)
    # %%
    # train model
    print('Training Alternating Least Squares (ALS) recommendation model...')
    model = trainALSWithRatings(ratings)
    # %%
    # get model's recommendations
    users = getUsersAsDataFrame(spark, [userID])
    recommendations = model.recommendForUserSubset(users, 10).collect()
    # %%
    print(f"Top 10 recommendations for user {userID}:")
    for user, recs in recommendations:
        for rec in recs:
            print(
                f" * {str(movieNames[rec.movieID]).ljust(45)}"
                + f"{rec.rating:.1f}".rjust(6)
            )
    # %%

# %%
if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')

# %%