from pyspark import SparkConf, SparkContext
from pathlib import Path
from myutils import make_lineparser
from collections import namedtuple

conf = SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf = conf)

lines = sc.textFile(f"file:///{Path(__file__).parent.parent}/data/1800.csv")

TempMeasurement = namedtuple('TempMeasurement', 'station_id entry_type temperature')
parse_temp = make_lineparser(
    (0, str),  # statioID
    (2, str),  # entryType
    (3, lambda x: float(x)*.18 + 32.)  # temparature
)
temps = lines.map(lambda line: TempMeasurement(*parse_temp(line.split(','))))

minStationTemps = temps\
                    .filter(lambda x: "TMIN" in x.entry_type)\
                    .map(lambda x: (x.station_id, x.temperature))\
                    .reduceByKey(lambda tot, x: min(tot, x))\
                    .collect()
maxStationTemps = temps\
                    .filter(lambda x: "TMAX" in x.entry_type)\
                    .map(lambda x: (x.station_id, x.temperature))\
                    .reduceByKey(lambda tot, x: max(tot, x))\
                    .collect()

for results in (minStationTemps, maxStationTemps):
    for result in results:
        print(result[0] + "\t{:.2f}F".format(result[1]))
