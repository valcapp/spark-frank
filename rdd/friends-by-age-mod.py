from pyspark import SparkConf, SparkContext
from pathlib import Path
from myutils import make_lineparser

conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

lines = sc.textFile(f"file:///{Path(__file__).parent.parent}/data/fakefriends.csv")
read_age_friends = make_lineparser((2, int), (3, int),)
rdd = lines.map(lambda line: read_age_friends(line.split(',')))
totalsByAge = rdd.mapValues(lambda x: (x, 1))\
                .reduceByKey(lambda tot, x: tuple(tot[field]+x[field] for field in range(2)))
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
results = averagesByAge.collect()
for result in results:
    print(result)
