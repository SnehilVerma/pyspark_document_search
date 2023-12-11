import time
from data_query import search_query
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import json,os

current_directory = os.path.dirname(os.path.abspath(__file__))

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
    index_mapping_path = os.path.join(current_directory, "index_mapping.json")
    dicti = json.load(open(index_mapping_path))
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

def plot_multiple_client_vs_latency(spark, max_workers):
    import concurrent.futures
    phrase = "March is the third month of the year in the Gregorian calendar, coming between February and April."
    latencies = {}
        
    # varying the clients from 10 to 100, with a step size of 10.
    for num_client in range(10, 100 ,10):   
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
            # futures = [executor.submit(measure_latency, phrase, spark) for _ in range(max_workers)]
            futures = [executor.submit(measure_latency, phrase, spark) for client_id in range(1, num_client + 1)]
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
        latency = time.time() - start_time
        latencies[num_client]=latency  

    # Plot the results
    plt.plot(latencies.keys(), latencies.values(), marker='o')
    plt.xlabel('Number of concurrent queries')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency vs. Number of concurrent queries')
    plt.show()

phrase = "It is different in some ways from other types of English, such as British English. Most types of American English came from local dialects in England.\n\nUse \nMany people today know about American English even if they live in a country where another type of English is spoken."
max_length = len(phrase.split())


if __name__=="__main__":
    # Create a Spark session
    spark = SparkSession.builder.appName("Document Search").getOrCreate()    
    # Analysis 1 : Measure latency against search phrase length
    plot_latency_vs_length(phrase, max_length,spark)

    # Analysis 2 : Measure latency against index size
    plot_index_size_vs_latency(spark)

    # Analysis 3 : Measure latency against increasing number of clients.
    # For this we have added multithreading support.
    plot_multiple_client_vs_latency(spark,max_workers=4)
    spark.stop()


