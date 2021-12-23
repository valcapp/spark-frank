# %%
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
import pyspark.sql.functions as sf
import os
from time import perf_counter
from pathlib import Path

# %%
#  setup
conf = SparkConf().setMaster("local").setAppName("DegreesOfSeparation")
sc = SparkContext(conf = conf)

spark = SparkSession.builder.appName('HerosBreadthSearch').getOrCreate()

# %%
heroSeprationSchema = StructType([
    StructField('id', IntegerType(), False),
    StructField('connections', ArrayType(IntegerType(), True), False),
])

def hero_connections_parse_line(line: str) -> Row:
    fields = line.split()
    return Row(
        id = int(fields[0]),
        connections = [int(connection) for connection in fields[1:]],
    )

def loadStartingDF() -> DataFrame:
    inputFile = sc.textFile(f"file:///{Path(__file__).parent.parent}/data/Marvel-Graph.txt")
    rdd = inputFile.map(hero_connections_parse_line)
    return spark.createDataFrame(rdd, schema=heroSeprationSchema)


# %%
def connections_by_hero(df: DataFrame) -> DataFrame:
    """Makes id unique, connections will be set of all connections with same id from original df

        Args:
            df (DataFrame): original df, where id's are not unique

        Returns:
            DataFrame: where id are unique, connections are set with all connection without repetition
        """
    return \
        df.select('id', sf.explode('connections').alias('connections'))\
        .groupBy('id').agg(sf.collect_set('connections').alias('connections'))


# %%
def firstIterationDf(startDf: DataFrame, startId: int) -> DataFrame:
    # make id unique and connections complete
    df = connections_by_hero(startDf)
    # mark first row to process
    return df.withColumn(
        'processStatus',
        sf.when(df.id == startId, 1).otherwise(0)
    )

def countTargetHits(df: DataFrame, targetId: int) -> int:
    print(f"Inspecting {df.select(sf.explode('connections')).count()} connections")
    # how many heros have the targetId among their connections
    hit_times = df.where(sf.array_contains(df.connections, targetId)).count()
    return hit_times

def updateProcessStatus(df: DataFrame) -> DataFrame:
    # get ids of next heros to process
    isProcessing = df.processStatus == 1
    nextIds = df.where(isProcessing)\
        .select(sf.explode('connections').alias('id'))\
        .join(df.select('id', 'processStatus'), on='id')\
        .where(sf.col('processStatus')==0)\
        .select(sf.collect_set('id')).first()[0]
    isToProcessNext = df.id.isin(nextIds)
    # mark current processing as processed, next to be processed as processing
    return \
        df.select(
            'id', 'connections',
            sf.when(isProcessing, 2)\
                .when(isToProcessNext, 1)\
                .otherwise(df.processStatus)\
                .alias('processStatus')
        )

def degreeSeparation(startDf: DataFrame, startId: int, targetId: int) -> int:
    df = firstIterationDf(startDf, startId)

    for distance in range (1, 11):
        # df.cache()
        print(f"Running BFS iteration # {distance}")
        # process layer (rows with process status 1)
        hit_times = countTargetHits(df.where(df.processStatus == 1), targetId)
        if hit_times:
            print(f"Hit the target character! From {hit_times} different direction(s).")
            break
        df = updateProcessStatus(df)
    
    return distance

# %%

if __name__ == '__main__':
    startTime = perf_counter()
    print(
        'Degree of separation between heroes 5306 and 14: '
        + str(degreeSeparation(loadStartingDF(), 5306, 14))
    )
    print(f'Executed in {perf_counter() - startTime:.1f} s') # ~ 20s




