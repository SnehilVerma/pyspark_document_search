from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, explode, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import monotonically_increasing_id
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk,re
nltk.download("stopwords")
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
import json,string
import os


index_dict  = {}

from datasets import load_dataset
import csv

dataset = load_dataset("wikipedia", "20220301.simple")
current_directory = os.path.dirname(os.path.abspath(__file__))




def read_docs(doc_size):
    dicti = {}
    document_count = doc_size
    count = 0
    for row in dataset["train"]:
        dicti[row["id"]]=row["text"]
        count += 1
        if count==document_count:
            break
    return dicti


def generated_index(doc_size,spark):
    def remove_punctuation_and_quotes(text):
        # Define a translation table
        translator = str.maketrans("", "", string.punctuation + "'\"")
        # Use translate to remove punctuation and quotes
        text_without_punctuations = text.translate(translator)
        text_without_numbers = re.sub(r'\d+', '', text_without_punctuations)

        return text_without_numbers

    # Step 1: Data Ingestion

    dicti = read_docs(doc_size)
    # Converting into list of tuple
    data = [(doc_id, remove_punctuation_and_quotes(text)) for doc_id, text in dicti.items()]

    schema = ["document_id", "text"]
    df = spark.createDataFrame(data, schema)

    # Step 2: Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    # Step 3: Stopword Removal
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    filtered_df = remover.transform(tokenized_df)
    # filtered_df.select("filtered_tokens").show()

    # Step 4: Lowercasing and Stemming using NLTK
    stemmer = SnowballStemmer(language="english")

    def stemming_udf(tokens):
        return [stemmer.stem(token) for token in tokens]

    stemming = udf(stemming_udf, ArrayType(StringType()))
    stemmed_df = filtered_df.withColumn("stemmed_tokens", stemming("filtered_tokens"))

    # Step 5: Indexing - Inverted Index
    # Assume a unique identifier for each document, if not present
    exploded_df = stemmed_df.select("document_id", "stemmed_tokens") \
        .withColumn("token", explode("stemmed_tokens"))

    # Build the inverted index
    inverted_index = exploded_df.groupBy("token").agg(F.collect_list("document_id").alias("doc_ids"))


    # Save the inverted index to a desired location (e.g., Parquet format)
    fname = f'generated_index_{doc_size}.parquet'
    file_path = os.path.join(current_directory, fname)
    inverted_index.write.mode("overwrite").parquet(file_path)
    index_dict[doc_size]=fname


if __name__ == "__main__":
    # Create a Spark session
    spark = SparkSession.builder.appName("TextProcessingExample").getOrCreate()
    for doc_size in range(100,5000,500):
        generated_index(doc_size,spark)
        print(f"dumping { doc_size }")
    
    # Stop the Spark session
    spark.stop()
    index_path = os.path.join(current_directory, 'index_mapping.json')
    with open(index_path, 'w') as fp:
        json.dump(index_dict, fp)
