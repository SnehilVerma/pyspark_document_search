from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import re
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import lower, explode, udf
from nltk.stem import SnowballStemmer
import string
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from functools import reduce


def remove_punctuation_and_quotes(text):
    # Define a translation table
    translator = str.maketrans("", "", string.punctuation + "'\"")
    # Use translate to remove punctuation and quotes
    text_without_punctuations = text.translate(translator)
    text_without_numbers = re.sub(r'\d+', '', text_without_punctuations)

    return text_without_numbers


def intersect(result_df):
    # Assuming result_df is your DataFrame with columns "token" and "doc_ids"
    token_doc_ids = result_df.select("token", "doc_ids").collect()

    # Extract doc_ids for each token into a list of sets
    sets_of_doc_ids = [set(row.doc_ids) for row in token_doc_ids]

    # Find the intersection of document ids across all tokens
    intersection_doc_ids = list(reduce(lambda x, y: x.intersection(y), sets_of_doc_ids))

    # Display the intersection of document ids
    # print("Intersection of Document IDs:", intersection_doc_ids)
    return intersection_doc_ids


def search_query(phrase,spark,index=None):
    def stemming_udf(tokens):
        return [stemmer.stem(token) for token in tokens]
    
    
    schema = ["doc_id", "text"]
    df = spark.createDataFrame([("0", remove_punctuation_and_quotes(phrase))], schema)

    # Step 2: Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    # Step 3: Stopword Removal
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    filtered_df = remover.transform(tokenized_df)

    first_row_value = filtered_df.select(col("filtered_tokens")).first()[0]
    if not first_row_value:
        return


    filtered_df.select("filtered_tokens").show(truncate=False)
    stemmer = SnowballStemmer(language="english")

    stemming = udf(stemming_udf, ArrayType(StringType()))
    query_df = filtered_df.withColumn("stemmed_tokens", stemming("filtered_tokens"))


    # Load the inverted index
    if index is None:
        inverted_index = spark.read.parquet("generated_index.parquet")
    else:
        inverted_index = spark.read.parquet(index)

    
    # Explode the tokens to create a row for each token in the query
    exploded_query_df = query_df.select("doc_id", "stemmed_tokens") \
        .withColumn("token", explode("stemmed_tokens"))

    # Join the query tokens with the inverted index on the token column
    joined_df = exploded_query_df.join(inverted_index, "token")
    # joined_df.show()

    doc_result = intersect(joined_df)  
    print(doc_result)

if __name__ == "__main__":
    # Example search query
    search_query("Common goals of policy")
    # search_query("teacher of God's and humans")

# Stop the Spark session


