from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType
)

spark = SparkSession.builder.appName('Temperatures').getOrCreate()

temps_schema = StructType([
    StructField('stationId', StringType(), True),
    StructField('date', IntegerType(), True),
    StructField('measure_type', StringType(), True),
    StructField('temperature', FloatType(), True),
])

df = spark.read.schema(temps_schema)\
    .csv(f'file:///{Path(__file__).parent.parent}/data/1800.csv')
# df.printSchema()

min_temps = df.where(sf.col("measure_type").contains('TMIN'))\
    .select('stationId', 'temperature')\
    .groupBy('stationId').min()\
    .withColumnRenamed("min(temperature)", "min_temperature")

min_temps_F = min_temps.withColumn(
    'min_temperature',
    sf.round(sf.col('min_temperature')*.18 + 32.0, 2)
).sort('min_temperature')

min_temps_F.show()

spark.stop()

