"""Predict price of house (PriceOfUnitArea) based on features (HouseAge, DistanceToMRT, NumberConvenienceStores)"""
# %%
from pyspark.sql import SparkSession, functions as sf
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pathlib import Path

# %%

def main():
    # %%
    spark = SparkSession.builder.appName('HousePrice').getOrCreate()
    # %%
    # constants
    filePath = f"file:///{Path(__file__).parent}/../data/realestate.csv"
    inputCols = ('HouseAge', 'DistanceToMRT', 'NumberConvenienceStores')
    labelCol = 'PriceOfUnitArea'
    # %%
    # raw data
    df = spark.read\
        .option('header', 'true')\
        .option('inferSchema', 'true')\
        .csv(filePath)\
        .select(labelCol, *inputCols)
    # df.printSchema() or df.show(10)
    # %%
    # data ready for ML
    assembler = VectorAssembler()\
        .setInputCols(inputCols)\
        .setOutputCol('features')
    data = assembler\
        .transform(df)\
        .select(sf.col(labelCol).alias('label'), 'features')
    # data.printSchema() or data.show(5)
    # %%
    # train model and predict
    trainData, testData = data.randomSplit([.7, .3])
    model = DecisionTreeRegressor().fit(trainData)
    predictions = model.transform(testData)
    # predictions.show(10)
    # %%
    # evaluate
    rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data = {rmse:.2f}")
    # %%

# %%
if __name__ == '__main__':
    from time import perf_counter
    start = perf_counter()
    main()
    print(f'Executed in {perf_counter() - start:.1f} s')