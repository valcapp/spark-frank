"""Example of linear model fitting with Pyspark"""
# %%
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, functions as sf
from pyspark.ml.linalg import Vectors
from pathlib import Path

# %%
def main():
    # %%
    spark =SparkSession.builder.appName('LinRegression').getOrCreate()
    filePath = f"{Path(__file__).parent}/../data/regression.txt"
    # %%
    # load data
    df = spark.sparkContext\
        .textFile(filePath)\
        .map(lambda line: line.split(','))\
        .map(lambda x:
            (float(x[0]), Vectors.dense([x[1]],))
        ).toDF(['label', 'features'])
    # df.show(5)
    # %%
    # train model
    trainDF, testDF = df.randomSplit([.5, .5])
    model = LinearRegression(maxIter=10, regParam=.3, elasticNetParam=.8)\
        .fit(trainDF)
    # %%
    # make predictions
    predictions = model.transform(testDF).cache()
    # predictions.show(5)
    # %%
    # convert to pandas df
    pdf = predictions\
        .select(
            sf.col('prediction').alias('y_model'),
            sf.col('label').alias('y_data'),
            sf.col('features').alias('x')
        )\
        .toPandas()
    pdf['x'] = pdf.x.apply(lambda val: val[0])
    # %%
    # visualize model
    ax = pdf.plot(kind='scatter', x='x', y='y_data')
    pdf.plot(ax = ax, x='x', y='y_model', color='red')
    print(pdf)
    # %%

# %%
if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')
