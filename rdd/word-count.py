import re
from pyspark import SparkConf, SparkContext
from pathlib import Path

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile(f"file:///{Path(__file__).parent.parent}/data/book.txt")
words = input.flatMap(normalizeWords)

wordCounts = words\
                .map(lambda word: (word, 1))\
                .reduceByKey(lambda count, one: count + one)\
                .sortBy(lambda x: x[1], ascending=False)

results = wordCounts.collect()

for word, count in results:
    encoded = word.encode('ascii', 'ignore')
    encoded and print(encoded.decode().ljust(25) + str(count).rjust(7))
