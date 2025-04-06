import time
import string
import random
from datetime import datetime
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from google.cloud import storage

print("packages loaded")

spark = SparkSession.builder.appName("targetLR").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

path = "gs://tfs-prd-dsg/user/shubhamg/T_schema.json"
bucket_name = "tfs-prd-dsg"

txns_path = "gs://tfs-prd-target/reports/rawdata/T/year=2023/month=12/day=14/hour=*/*transaction.log.*"
c_path = "gs://tfs-prd-target/reports/rawdata/C/year=2023/month=12/day=14/hour=*/*transaction_c.log.*"

txns = spark.read.format("json").option("multiline", True).option("inferSchema", True).option("header", True).load(
    txns_path)
c = spark.read.format("json").option("multiline", True).option("inferSchema", True).option("header", True).load(c_path)

ad_id_list = [7529, 7530]
txns = sqlContext.read.json(txns_path)
c = sqlContext.read.json(c_path)
txns = txns.withColumn("tzo", txns["tzo"].cast(t.IntegerType()))
txns = txns.withColumn('usertime', txns['createdtime'] - txns['tzo'] * 60 * 1000)
txns = txns.withColumn("usertime", f.substring("usertime", 0, 10))
txns = txns.withColumn("usertime", txns["usertime"].cast("double"))
txns = txns.withColumn('usertime',
                       f.date_format(txns.usertime.cast(dataType=t.TimestampType()), "yyyy-MM-dd HH:mm:ss.SSS"))
txns = txns.withColumn('hour', f.hour(txns.usertime.cast('timestamp')))
txns = txns.withColumn('minute', f.minute(txns.usertime.cast('timestamp')))
txns = txns.withColumn('day', f.dayofweek(txns.usertime.cast('timestamp')))
# txns = txns.withColumn('month', f.day(txns.usertime.cast('timestamp')))
txns = txns.filter(txns.adid.isin(ad_id_list))
txns = txns.withColumn('userminute', txns['hour'] * 60 + txns['minute'])
txns.createOrReplaceTempView("txns")
c.createOrReplaceTempView("click")
merge = spark.sql(
    "SELECT t.*, c.adrequestid as click_req FROM txns as t LEFT OUTER JOIN click as c ON t.adrequestid = c.adrequestid")
merge.createOrReplaceTempView("merge")

json_schema_exconfig = t.StructType([t.StructField('pdata', t.StructType([
    t.StructField('intent', t.StructType([
        t.StructField('AD_SIZE', t.StringType(), True),
        t.StructField('DSP_KEY', t.StringType(), True)
    ])),
    t.StructField('EXCHANGE_ID', t.StringType(), True),
    t.StructField('PUBLISHER_ID', t.StringType(), True),
    t.StructField('SOURCE_URL', t.StringType(), True),
    t.StructField('UNIVERSAL_SITE_ID', t.StringType(), True),
    t.StructField('CREATIVE_ID', t.StringType(), True),
    t.StructField('AUCTION_ID', t.StringType(), True),
    t.StructField('IO_NAME', t.StringType(), True),
    t.StructField('LI_NAME', t.StringType(), True),
    t.StructField('DSP_NAME', t.StringType(), True),
    t.StructField('vertical', t.StringType(), True),
    t.StructField('DEVICE_NAME', t.StringType(), True),
]))])

json_schema_dsData = t.ArrayType(t.StructType([
    t.StructField('heading', t.StringType(), True),
    t.StructField('tnc', t.StringType(), True),
    t.StructField('active', t.StringType(), True),
    t.StructField('HeadingText', t.StringType(), True),
    t.StructField('adSize', t.StringType(), True),
    t.StructField('language', t.StringType(), True),
    t.StructField('hero', t.StringType(), True),
    t.StructField('startdate', t.StringType(), True),
    t.StructField('locale', t.StringType(), True),
    t.StructField('url', t.StringType(), True),
    t.StructField('rowid', t.StringType(), True),
    t.StructField('offer', t.StringType(), True),
    t.StructField('enddate', t.StringType(), True),
    t.StructField('retailerLogo', t.StringType(), True),
    t.StructField('dayofweek', t.StringType(), True),
    t.StructField('Event', t.StringType(), True),
    t.StructField('html', t.StringType(), True),
    t.StructField('hourofday', t.StringType(), True),
    t.StructField('btn', t.StringType(), True),
    t.StructField('PersonalizationField', t.StringType(), True),
]))

json_schema_dsMetaData = t.StructType([t.StructField('ZIP', t.StructType([
    t.StructField('isCreativeDs', t.StringType(), True),
    t.StructField('mainDSrowID', t.StringType(), True),
    t.StructField('rowID', t.StringType(), True),
    t.StructField('data', t.StructType([
        t.StructField('store_id', t.StringType(), True),
        t.StructField('zipcode', t.StringType(), True),
        t.StructField('active', t.StringType(), True),
        t.StructField('id', t.StringType(), True),
        t.StructField('url', t.StringType(), True),
        t.StructField('rowid', t.StringType(), True)
    ])),
]))])

df2 = merge.filter(merge['exconfig'].isNotNull())
df2 = df2.withColumn('exconfig_parsed', f.from_json(f.col('exconfig'), json_schema_exconfig))
df2 = df2.filter(df2['dsData'].isNotNull())
df2 = df2.withColumn('dsData_parsed', f.from_json(f.col('dsData'), json_schema_dsData))

df2 = df2.withColumn("html", df2["dsData_parsed.html"].getItem(0))
df2 = df2.withColumn("adSize", df2["dsData_parsed.adSize"].getItem(0))
df2 = df2.withColumn("dayofweek", df2["dsData_parsed.dayofweek"].getItem(0))
df2 = df2.withColumn("heading", df2["dsData_parsed.heading"].getItem(0))
df2 = df2.withColumn("url", df2["dsData_parsed.url"].getItem(0))

df2 = df2.filter(df2['dsMetaData_v2'].isNotNull())
df2 = df2.withColumn('dsMetaData_parsed', f.from_json(f.col('dsMetaData_v2'), json_schema_dsMetaData))

df2 = df2.na.fill(value='null', subset=["modelType"])
df2 = df2.na.fill(value='null', subset=["referer"])
df2 = df2.withColumn('referer', f.when(f.col('referer') == '', 'null').otherwise(f.col('referer')))

df2 = df2.na.fill(value='null', subset=["zipcode"])

df2 = df2.na.fill(value='null', subset=["city"])
df2 = df2.na.fill(value='null', subset=["region"])

df2 = df2.na.fill(value=21, subset=["hour"])
df2 = df2.na.fill(value=47, subset=["minute"])

df2 = df2.na.fill(value=4, subset=["day"])
df2 = df2.withColumn('click', f.when(f.col('click_req').isNotNull(), 1).otherwise(0))

df_final = df2.select('dsMetaData_parsed.ZIP.mainDSrowID', 'modelType',
                      'dsMetaData_parsed.ZIP.rowID', 'campaignid', 'adid',
                      'devicename', 'exconfig_parsed.pdata.LI_NAME', 'referer',
                      'browser', 'os', 'zipcode', 'city', 'region',
                      'createdtime', 'advertiserid', 'adformatId',
                      'contentHash', 'zoneid', 'adrequestid',
                      'html', 'adSize', 'dayofweek',
                      'heading', 'url', 'hour', 'minute', 'day', 'click')

remove_adreqid = df_final.select('adrequestid').filter(df_final.click == 0).rdd.takeSample(withReplacement=False,
                                                                                           num=100000)

l = []
for each in remove_adreqid:
    l.append(each[0])

df_final = df_final.filter(~df_final.adrequestid.isin(l))

cols_meta = {'devicename': 'dev', 'LI_NAME': 'li', 'mainDSrowID': 'maindsrowid', 'rowID': 'rowid',
             'modelType': 'modeltype',
             'adid': 'adid', 'zipcode': 'zip', 'browser': 'br', 'os': 'os', 'advertiserid': 'adv', 'campaignid': 'camp',
             'adformatId': 'adf',
             'contentHash': 'content', 'zoneid': 'zid', 'city': 'city', 'region': 'reg', 'hour': 'hr', 'day': 'day',
             'referer': 'ref',
             'html': 'html', 'adSize': 'adSize', 'dayofweek': 'dayofweek', 'heading': 'heading', 'url': 'url'}


def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[f.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")


df_final_new = df_final.alias('df_final_new')
id(df_final_new) == id(df_final)
df_final_new = rename_columns(df_final_new, cols_meta)


def convert_to_string(df):
    cols = df.columns
    t = ['click']
    cols = [col for col in cols if col not in t]
    for col in cols:
        typ = df.select(col).dtypes[0][1]
        if typ != 'string':
            df = df.withColumn(col, df_final_new[col].cast("string"))
    return df


SEPARATOR_FEATURE_VALUE = "_";
ML_FEATURE_DELIMITER = "#";

df_final_new = convert_to_string(df_final_new)
remove_col = ['createdtime', 'adrequestid', 'rowid', 'maindsrowid', 'click']
all_cols = df_final_new.columns
use_col = [col for col in all_cols if col not in remove_col]

use_col
indexed_col = []
for col in use_col:
    indexed_col.append(col + '_Indexed')
print(indexed_col)

use_col
ohe_col = []
for col in use_col:
    ohe_col.append(col + '_ohe')
# output_col_new.append('click')
print(ohe_col)


def convert_size_bytes(size_bytes):
    """
    Converts a size in bytes to a human readable string using SI units.
    """
    import math
    import sys

    if not isinstance(size_bytes, int):
        size_bytes = sys.getsizeof(size_bytes)

    if size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def cal_ojb_size(obj, persist=False):
    print(obj.cache().count())
    catalyst_plan = obj._jdf.queryExecution().logical()
    size_bytes = spark._jsparkSession.sessionState().executePlan(catalyst_plan).optimizedPlan().stats().sizeInBytes()
    if not persist:
        obj.unpersist()
    print("Total table size: ", convert_size_bytes(size_bytes))


def cal_obj_size_v2(obj, persist=False):
    print(obj.cache().count())
    size_in_bytes = sc._jvm.org.apache.spark.util.SizeEstimator.estimate(obj._jdf)
    #     size_estimate_mb = size_estimate_bytes / (1024**2)
    #     size_estimate_gb = size_estimate_bytes / (1024**3)
    size_estimate_mb = size_in_bytes / (1024 ** 2)
    size_estimate_gb = size_in_bytes / (1024 ** 3)
    if not persist:
        obj.unpersist()
    print("obj size --", size_estimate_mb, size_estimate_gb)


# StringIndexer Initialization
start = time.time()
indexer = StringIndexer(inputCols=use_col, outputCols=indexed_col)
indexerModel = indexer.fit(df_final_new.select(use_col))

# Transform the DataFrame using the fitted StringIndexer model
indexed_df = indexerModel.transform(df_final_new)
# indexed_df.select(all_col).show(1)
end = time.time()
print(end - start)
# df_final_new.unpersist()


start = time.time()
encoder = OneHotEncoder(inputCols=indexed_col,
                        outputCols=ohe_col)
encoder.setDropLast(False)
model = encoder.fit(indexed_df)
encoded = model.transform(indexed_df)
# encoded.show(1)
end = time.time()
print(end - start)

filtered_col = [use_col, indexed_col]
filtered_col = [col for lst in filtered_col for col in lst]
filtered_df = encoded.select(filtered_col)

start = time.time()
all_colIdx = []
filtered_df.cache().count()
for col in use_col:
    # colIdx = encoded.select(col,f"{col}_Indexed").distinct().rdd.collectAsMap()
    indexed = filtered_df.select(col, f"{col}_Indexed").distinct()
    indexed = indexed.withColumn("tuple", f.struct(indexed[col], indexed[f"{col}_Indexed"]))
    indexed = indexed.select('tuple')
    indexed = indexed.toPandas()
    indexed = indexed['tuple'].tolist()
    colIdx = {}
    for key, value in indexed:
        colIdx[key] = value
    colIdx = sorted((value, f"{col}_" + key) for (key, value) in colIdx.items())
    all_colIdx.append(colIdx)
end = time.time()
print(end - start)
filtered_df.unpersist()

print("actual columns", len(use_col))
print("converted columns", len(all_colIdx))

total_tokens = len([element for col in all_colIdx for element in col])
tokens = [element[1] for col in all_colIdx for element in col]
print("total tokens", total_tokens)

# initializing size of string
N = 7
# using random.choices()
# generating random strings
rnd_alpha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
tm_stpm = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

token_idx = list(zip([each for each in range(len(tokens))], tokens))

# giving column names of dataframe
token_idx_col = ["index", "features"]

# creating a dataframe
token_df = spark.createDataFrame(data=token_idx, schema=token_idx_col)
# token_df.write.option("delimiter", "\t").csv(f"gs://tfs-prd-dsg/user/shubhamg/mlmodels/models/tokens_{res}_{tm_stpm}")

# show data frame
# token_df.show(5)


token_dict = {row['index']: row['features'] for row in token_df.collect()}


def myConcat(*cols):
    concat_columns = []
    for c in cols[:-1]:
        concat_columns.append(f.coalesce(c, f.lit("*")))
        concat_columns.append(f.lit("\t"))
    concat_columns.append(f.coalesce(cols[-1], f.lit("*")))
    return f.concat(*concat_columns)


df_text = token_df.withColumn("combined", myConcat(*token_df.columns)).select("combined")
df_text.coalesce(1).write.format("text").option("header", "false").mode("append").save(
    f"gs://tfs-prd-dsg/user/shubhamg/mlmodels/models/tokens_{rnd_alpha}")

encoded.cache().count()
assembler = VectorAssembler(inputCols=ohe_col, outputCol="features")
output_df = assembler.transform(encoded)
encoded.unpersist()


def extract_keys_from_vector(vector):
    return vector.indices.tolist()


def get_feature_coeff(rowid, idx_lst, intercept, coeff, token_dict):
    inter = f"cr_{rowid}#bias\t{intercept}\n"
    fea = ''
    for ele in idx_lst:
        feature_value = token_dict[ele]
        feature_coeff = coeff[ele]
        fea = fea + f"cr_{rowid}#{feature_value}\t{feature_coeff}\n"
    fea = inter + fea
    return fea


def upload_to_gcs(rowid, fea, rnd_alpha):
    client = storage.Client()
    bucket = client.bucket("tfs-prd-dsg")
    blob = bucket.blob(f"user/shubhamg/mlmodels/models/lr_cr_{rowid}_{rnd_alpha}.txt")
    data = blob.upload_from_string(fea)


def get_df_count(df):
    n_dp = df.count()
    temp_df = df.groupBy('click').count()
    n_non_click = temp_df.select('count').where(temp_df.click == 0)
    n_non_click = n_non_click.head()[0]
    n_click = n_dp - n_non_click
    return n_dp, n_non_click, n_click


model_df = output_df.select('maindsrowid', 'features', 'click')
feature_extract = f.udf(lambda vector: extract_keys_from_vector(vector), t.ArrayType(t.IntegerType()))
model_df = model_df.withColumn("features_idx", feature_extract(f.col("features")))
model_df.cache().count()
maidsrowids = model_df.select('maindsrowid').distinct().rdd.flatMap(lambda x: x).collect()
summary = []
for rowid in maidsrowids:
    print(rowid)
    #     feature_extract = f.udf(lambda vector: extract_keys_from_vector(vector), t.ArrayType(t.IntegerType()))
    #     train_df = train_df.withColumn("features_idx", feature_extract(f.col("features")))
    temp_df = model_df.select('maindsrowid', 'features', 'features_idx', 'click').filter(model_df.maindsrowid == rowid)
    # model_df.unpersist()
    # temp_df.show(2)
    train_df, test_df = temp_df.randomSplit([0.7, 0.3], seed=42)
    print("split done")
    train_df.cache().count()
    print("cache done")
    train_df_dp, train_df_nc, train_df_c = get_df_count(train_df)
    print("count done")

    log_reg = LogisticRegression(featuresCol='features', labelCol='click').fit(train_df)
    train_result = log_reg.evaluate(train_df).predictions
    intercept = log_reg.intercept
    coeff = log_reg.coefficients

    print("model trained")

    test_df.cache().count()
    test_result = log_reg.evaluate(test_df).predictions
    test_df_dp, test_df_nc, test_df_c = get_df_count(test_df)
    test_df.unpersist()

    tp = test_result[(test_result.click == 1) & (test_result.prediction == 1)].count()
    tn = test_result[(test_result.click == 0) & (test_result.prediction == 0)].count()
    fp = test_result[(test_result.click == 0) & (test_result.prediction == 1)].count()
    fn = test_result[(test_result.click == 1) & (test_result.prediction == 0)].count()

    print("test evaluated")

    # Calling the evaluator
    res = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='click')

    # Evaluating the AUC on results
    train_ROC_AUC = res.evaluate(train_result)
    ROC_AUC = res.evaluate(test_result)
    # print(ROC_AUC)
    accuracy = float((tp + tn) / test_result.count())
    recall = float(tn) / (tp + tn)
    precision = float(tp) / (tp + fp)
    f1 = float(2 * ((recall * precision) / (recall + precision)))

    #     accuracy_v2 = res.evaluate(results, {res.metricName: "accuracy"})
    #     recall_v2 = res.evaluate(results, {res.metricName: "weightedRecall"})
    #     precision_v2 = res.evaluate(results, {res.metricName: "weightedPrecision"})
    #     f1_score_v2 = res.evaluate(results, {res.metricName."weightedFMeasure"})

    #     print(f"AUC-ROC: {auc:.4f}")
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")

    # feature_extract = f.udf(lambda vector: extract_keys_from_vector(vector), t.ArrayType(t.IntegerType()))
    # train_df = train_df.withColumn("features_idx", feature_extract(f.col("features")))

    # mainds_df = train_df.select('maindsrowid', 'features_idx').where(train_df.maindsrowid == rowid)
    temp_list = train_df.select("features_idx").rdd.flatMap(lambda x: x).collect()
    temp_list_new = [element for lst in temp_list for element in lst]
    temp_list_new = list(set(temp_list_new))
    fea = get_feature_coeff(rowid, temp_list_new, intercept, coeff, token_dict)
    upload_to_gcs(rowid, fea, rnd_alpha)

    train_df.unpersist()

    tup = (
    rowid, train_df_dp, train_df_nc, train_df_c, test_df_dp, test_df_nc, test_df_c, train_ROC_AUC, ROC_AUC, accuracy,
    recall, precision, f1, tp, tn, fp, fn)
    summary.append(tup)
    print("summary added")
model_df.unpersist()
summary_df = spark.createDataFrame(summary,
                                   schema='rowid long, train_dp long, train_non_click long, train_click long, test_dp long, test_non_click long, test_click long, train_roc double, roc double, accuracy double, recall double, precision double, f1 double, tp long, tn long, fp long, fn long')
summary_df.write.option("delimiter", "\t").csv(f"gs://tfs-prd-dsg/user/shubhamg/mlmodels/models/summary_{rnd_alpha}")
print("lr process completed")


