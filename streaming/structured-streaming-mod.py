
# %%
# imports

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession, functions as sf
from pathlib import Path

# %%

def main():
    # %%
    # start spark session
    pwd = Path(__file__).parent
    spark = SparkSession.builder\
        .appName('StructuredStreaming')\
        .getOrCreate()
    # %%
    # start streaming
    lines = spark.readStream.text(f'file:///{pwd}/../data/logs')
    # %%
    # define line parsing
    hostExp = r'(^\S+\.[\S+\.]+\S+)\s'
    timeExp = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
    generalExp = r'\"(\S+)\s(\S+)\s*(\S*)\"'
    statusExp = r'\s(\d{3})\s'
    contentSizeExp = r'\s(\d+)$'
    logsDF = lines.select(
        sf.regexp_extract('value', hostExp, 1).alias('host'),
        sf.regexp_extract('value', timeExp, 1).alias('timestamp'),
        sf.regexp_extract('value', generalExp, 1).alias('method'),
        sf.regexp_extract('value', generalExp, 2).alias('endpoint'),
        sf.regexp_extract('value', generalExp, 3).alias('protocol'),
        sf.regexp_extract('value', statusExp, 1).cast('integer').alias('status'),
        sf.regexp_extract('value', contentSizeExp, 1).cast('integer').alias('content_size')
    )
    # %%
    # define query and start streaming dumping to console
    statusCount = logsDF.groupBy('status').count()
    query = statusCount\
        .writeStream\
        .outputMode('complete')\
        .format('console')\
        .queryName('statusCount')\
        .start()
    query.awaitTermination()
    # %%
    spark.stop()
    # %%
# %%

if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')
