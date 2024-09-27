from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search_amazon import RetrievalOpenSearch_Amazon
from beir.hybrid.data_ingestor import OpenSearchDataIngestor
import numpy as np
import pandas as pd

import logging
import pathlib, os, getopt, sys
from opensearchpy import OpenSearch, RequestsHttpConnection

import base64
import requests


def main(argv):
    opts, args = getopt.getopt(argv, "h:p:i:m:n:o:u:l:",
                               ["os_host=", "os_port=", "os_index=", "os_model_id=", "num_of_runs=", "operation=", "dataset_folder=", "num_of_corpus_records="])
    dataset = 'amazon'
    endpoint = ''
    port = ''
    index = ''
    num_of_runs = 2
    operation = "evaluate"
    parq_dir = ''
    number_of_records = 100
    for opt, arg in opts:
        if opt in ("-h", "-os_host"):
            endpoint = arg
        elif opt in ("-p", "-os_port"):
            port = arg
        elif opt in ("-i", "-os_index"):
            index = arg
        elif opt in ("-n", "-num_of_runs"):
            num_of_runs = int(arg)
        elif opt in ("-o", "-operation"):
            operation = arg
        elif opt in ("-u", "-dataset_url"):
            parq_dir = arg
        elif opt in ("-l", "-num_of_corpus_records"):
            number_of_records = int(arg)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### /print debug information to stdout
    #### Download scifact.zip dataset and unzip the dataset
    # dataset = "nfcorpus"
    # dataset = "trec-covid"
    # dataset = "arguana"
    # dataset = 'fiqa'

    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = os.path.join(out_dir, dataset)

    #### Provide the data_path where scifact has been downloaded and unzipped
    # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    sel = [("WIDTH", "<=", 256), ("WIDTH", ">=", 128),
           ("HEIGHT", "<=", 256), ("HEIGHT", ">=", 128)]

    for file in os.listdir(parq_dir):

        if not file.endswith('.parquet'):
            continue
        file_path = os.path.join(parq_dir, file)

        laion_all = pd.read_parquet(
            file_path,
            filters=sel)

        laion_all_small_images = laion_all.copy()

        #small = laion_all_small_images.sample(n=N)
        texts_array = laion_all_small_images["TEXT"].iloc[0:number_of_records].values
        image_url = laion_all_small_images["URL"].iloc[0:number_of_records].values

        def init(endpoint: str, port: str, timeout: int = 30):
            opensearch = OpenSearch(
                hosts=[{
                    'host': endpoint,
                    'port': port
                }],
                use_ssl=False,
                verify_certs=False,
                connection_class=RequestsHttpConnection,
                timeout=timeout
            )
            return opensearch

        opensearch = init(endpoint=endpoint, port=port)
        bulk_size = 20

        def get_content(text, image_url):
            return {'id': image_url, 'text': text,
                    'image': base64.b64encode(requests.get(image_url, timeout=10).content).decode('utf-8')}

        for i in range(0, number_of_records, bulk_size):
            actions = []
            for idx in range(i, min(number_of_records, i + bulk_size)):
                try:
                    content = get_content(texts_array[idx], image_url[idx])
                    actions.extend(
                        [{'index': {'_index': index}},
                         content])
                except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.TooManyRedirects):
                    print("Error, skipping record")

            opensearch.bulk(
                index=index,
                body=actions)


if __name__ == "__main__":
    main(sys.argv[1:])
