from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as spf

spark = SparkSession.builder.appName('WordCount').getOrCreate()

# txtBook = open((f"{Path(__file__).parent}/book.txt"),'r').read()
# print(txtBook)

rawBook = spark.read\
    .text(f"file:///{Path(__file__).parent.parent}/data/book.txt")

# flatten words
words = rawBook.select(
        spf.explode(
            spf.split(spf.col('value'), r'\W+')
        ).alias('word')
    )\
    .where("word!=''")\
    .select(spf.lower(spf.col('word')).alias('word'))


# group by word and count
wordCount = words.groupBy('word').count().sort('count', ascending=False)
wordCount.show()
