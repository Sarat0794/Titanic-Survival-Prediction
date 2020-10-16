# Databricks notebook source
import pyspark

# COMMAND ----------

# Importing Data
df = spark.sql("SELECT * FROM titanic_1_csv")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

df.columns

# COMMAND ----------

# Select required columsn for our analysis
my_cols = df.select(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

# COMMAND ----------

# Missing Data: Here we are just dropping the missing data
my_final_data = my_cols.na.drop()

# COMMAND ----------

# Categorical Columns
from pyspark.ml.feature import (VectorAssembler, VectorIndexer,
                               OneHotEncoder,StringIndexer)

# COMMAND ----------

# First create StringIndexer and the OneHot Encode them.
# StringIndexer converts every string into a number. It is basically Laber Encoder. 
gender_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex', outputCol='SexVec')

# COMMAND ----------

embark_indexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex', outputCol='EmbarkVec')

# COMMAND ----------

assembler = VectorAssembler(inputCols=['Pclass','SexVec','EmbarkVec','Age','SibSp','Parch','Fare'],
                           outputCol='features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline 
# Pipeline sets stages for different steps. It is generally used for complex ML tasks.

# COMMAND ----------

log_reg_titanic = LogisticRegression(featuresCol='features', labelCol='Survived')

# COMMAND ----------

pipeline = Pipeline(stages=[gender_indexer,embark_indexer,gender_encoder,embark_encoder,assembler,log_reg_titanic])

# COMMAND ----------

# You can treat this pipeline as a normal ML model.
# Split data
train_data,test_data = my_final_data.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

# Test the model on Test Data
results = fit_model.transform(test_data)

# COMMAND ----------

results.show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
# BinaryClassificationEvaluator returns AUC

# COMMAND ----------

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='Survived')


# COMMAND ----------

AUC = my_eval.evaluate(results)

# COMMAND ----------

AUC

# COMMAND ----------

pwd

# COMMAND ----------


