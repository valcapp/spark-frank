from pyspark import SparkConf, SparkContext
from pathlib import Path
from collections import namedtuple
from myutils import make_lineparser

sc = SparkContext(conf = SparkConf().setMaster('local').setAppName('CustomerSpending'))

text_rd = sc.textFile(f"file:///{Path(__file__).parent.parent}/data/customer-orders.csv")

Purchase = namedtuple('Purchase', 'customer amount')
parse_purchase = make_lineparser((0, int), (2, float))
purchases = text_rd.map(lambda line: Purchase(*parse_purchase(line.split(','))))

tot_spending_by_customer = purchases\
    .reduceByKey(lambda tot, amount: tot+amount)\
    .sortBy(lambda cust_spend: cust_spend[1])\
    .collect()

for spending in tot_spending_by_customer:
    customer, amount = spending
    print(str(customer).rjust(10) + f"{amount:.2f}".rjust(10))
