
# %%
import argparse
from math import sqrt
from pathlib import Path
from pyspark import SparkConf, SparkContext
from pyspark.rdd import PipelinedRDD as RDD
from typing import NewType, Tuple, Dict, Callable, Any
from collections import namedtuple
from functools import partial

# %% 
# Instantiate spark context
conf = SparkConf()  # on cluster
# conf = SparkConf('local').setMaster('local').setAppName('MovieRecommend')  # local
sc = SparkContext(conf=conf)

# %%
# get movieNames dictionary
def parse_args() -> None:
    parser = argparse.ArgumentParser(
        prog='movie-recommendations',
        description='Suggest 10 movie recommendations based on similarity to movieID argument')
    parser.add_argument(
        'movieID', type=int,
        help='movieID integer to base recommendations on')
    return parser.parse_args()

def getMovieNames(resource_path: str) -> Dict[int, str]:
    with open(resource_path, encoding='ascii', errors='ignore') as file:
        return {
            int(movieID): movieName
            for movieID, movieName, *_
            in map(lambda line: line.split('::'), file)
        }

# %%
# Define types
User = NewType('User', int)
Movie = NewType('Movie', int)
Score = NewType('Score', int)

MovieRating = namedtuple('MovieRating', 'movie score')
MovieRatingType = Tuple[Movie, Score]

UserRating = namedtuple('UserRating', 'user rating')
UserRatingType = Tuple[User, MovieRating]

Similarity = namedtuple('Similarity', 'score strength')
SimilarityType = Tuple[float, int]

RatingPair = Tuple[MovieRating, MovieRating] 
# ((movie, score), (movie, score))
MovieScorePair = Tuple[Tuple[Movie, Movie], Tuple[Score, Score]]
# ((movie, movie), (score, score))

UserRatingPair = Tuple[User, RatingPair] 
# (user, ((movie, score), (movie, score)))
UserMovieScorePair = Tuple[User, MovieScorePair]
# (user, ((movie, movie), (movie, score)))

MoviesSimilarity = Tuple[Tuple[Movie, Movie], Similarity]
# ((movie, movie), (sim_score, strength))

# %%
# Define functions

def parse_ratings_line(line: str) -> UserRating:
    userID, movieID, rating, timestamp = line.split('::')
    return UserRating(
        int(userID),
        MovieRating(int(movieID), float(rating))
    )

def filterDuplicateRatings(userRatPair: UserRatingPair) -> bool:
    userID, ratingPair = userRatPair
    x, y = ratingPair
    return x.movie < y.movie

def makePairs(userRatPair: UserRatingPair) -> MovieScorePair:
    userID, ratingPair = userRatPair
    return tuple(zip(*ratingPair))

def cosineSimilarity(ratingPairs: Tuple[Score, Score]) -> Similarity:
    sum_xx = sum_yy = sum_xy = numPairs = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx * sum_yy)
    similarity = numerator/denominator if denominator else 0
    return Similarity(float(similarity), int(numPairs))

def isSimilarToMovie(movieID: int, moviesSimilarity: MoviesSimilarity) -> bool:
    scoreThreshold = 0.97
    strengthThreshold = 50.0
    movies, similarity = moviesSimilarity
    return (
        ((movies[0] == movieID) or (movies[1] == movieID))
        and similarity.score > scoreThreshold
        and similarity.strength > strengthThreshold
    )

def swapKeyVal(keyVal: Tuple[Any, Any]) -> Tuple[Any, Any]:
    key, val = keyVal
    return val, key

# %%
# Calculate similarities
def getSimilarities(resource_path: str) -> RDD:
    ratings = sc\
        .textFile(resource_path)\
        .map(parse_ratings_line)\
        .partitionBy(100)

    # print(ratings.take(5))
    # Self join ratings and filter duplicates
    ratingPairs = ratings\
        .join(ratings)\
        .filter(filterDuplicateRatings)
    # print(ratingPairs.take(10))
    # assert not ratingPairs.filter(lambda userRatPair: userRatPair[1][0].movie == userRatPair[1][1].movie).collect()

    # get rid of user and make ((movie, movie), (score, score)) pairs
    movieScorePairs = ratingPairs\
        .map(makePairs)\
        .partitionBy(100)

    # calculate similarity for each movie pair
    moviesSimilarities = movieScorePairs\
        .groupByKey()\
        .mapValues(cosineSimilarity)\
        .persist()
    
    return moviesSimilarities


# %%
# Find movies similar to movieID

def getTopSimilarTo(moviesSimilarities: RDD, movieID: int) -> list:
    results = moviesSimilarities\
        .filter(partial(isSimilarToMovie, movieID))\
        .map(swapKeyVal)\
        .sortByKey(ascending = False)\
        .take(10)
    return results

def main():
    # movieID = 1210
    movieID = parse_args().movieID
    # pwd = Path(__file__).parent
    movieNamesPath = f"s3n://frank-spark-1m-movies/movies.dat"
    ratingsPath = f"s3n://frank-spark-1m-movies/ratings.dat"
    movieNames = getMovieNames(movieNamesPath)
    moviesSimilarities = getSimilarities(ratingsPath)
    
    print(f"Calculating Top 10 recommendations based on movie: '{movieNames[movieID]}'")
    results = getTopSimilarTo(moviesSimilarities, movieID)
    for result in results:
        sim, movies = result
        otherMovieID = movies[0] if movies[0] != movieID else movies[1]
        print(f'{movieNames[otherMovieID]}\tscore: {sim.score}\tstrength: {sim.strength}')

# %%
if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')
# %%
