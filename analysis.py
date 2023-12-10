import time
from data_query import search_query
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import json


def measure_latency(phrase,spark,index=None):
    start_time = time.time()
    search_query(phrase,spark,index)
    latency = time.time() - start_time
    return latency



def plot_latency_vs_length(phrase, max_length,spark):
    

    lengths = list(range(1, max_length + 1))
    latencies = []

    for length in lengths:
        truncated_phrase = " ".join(phrase.split()[:length])
        print(truncated_phrase)
        # print("-----------")
        latency = measure_latency(truncated_phrase,spark)
        latencies.append(latency)

    # Plot the results
    plt.plot(lengths, latencies, marker='o')
    plt.xlabel('Phrase Length')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency vs. Phrase Length')
    plt.show()

    


def plot_index_size_vs_latency(spark):
    phrase = "March is the third month of the year in the Gregorian calendar, coming between February and April."
    dicti = json.load(open("index_mapping.json"))
    latencies = {}
    for k,v in dicti.items():
        print(f"measuring for {k} index size")
        index_id = v
        # inverted_index = spark.read.parquet(index_id)
        latency = measure_latency(phrase=phrase,spark=spark,index=index_id)        
        latencies[k]=latency

    
    plt.plot(latencies.keys(), latencies.values(), marker='o')
    plt.xlabel('Index Size')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency vs. Index Size')
    plt.show()

def plot_multiple_client_vs_latency(spark, n):
    import concurrent.futures
    phrase = "March is the third month of the year in the Gregorian calendar, coming between February and April."
    latencies = {}

    for max_workers in range(1, n+1):
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
            start_time = time.time()
            futures = [executor.submit(measure_latency, phrase, spark) for _ in range(max_workers)]
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            latency = time.time() - start_time
            latencies[max_workers]=latency  


    # Plot the results
    plt.plot(latencies.keys(), latencies.values(), marker='o')
    plt.xlabel('Number of concurrent queries')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency vs. Number of concurrent queries')
    plt.show()



# phrase = "March is the third month of the year in the Gregorian calendar, coming between February and April."
phrase = "It is different in some ways from other types of English, such as British English. Most types of American English came from local dialects in England.\n\nUse \nMany people today know about American English even if they live in a country where another type of English is spoken."
max_length = len(phrase.split())
# Create a Spark session
spark = SparkSession.builder.appName("Document Search").getOrCreate()

plot_latency_vs_length(phrase, max_length,spark)

plot_index_size_vs_latency(spark)

num_threads_max = 15
plot_multiple_client_vs_latency(spark,num_threads_max)

spark.stop()



