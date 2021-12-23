from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType
)
from pyspark.sql import functions as sf

spark = SparkSession.builder.appName('CustomerSpending').getOrCreate()

df = spark.read.schema(
        StructType([
            StructField('customer', IntegerType(), False),
            StructField('item', IntegerType(), False),
            StructField('spent', FloatType(), False),
        ])
    ).csv(f"{Path(__file__).parent.parent}/data/customer-orders.csv")

tot_spent = df.groupBy('customer')\
    .agg(sf.round(sf.sum('spent'), 2).alias('tot_spent'))\
    .sort('tot_spent', ascending=False)

tot_spent.show()

spark.stop()

