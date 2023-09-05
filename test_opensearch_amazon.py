from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search_amazon import RetrievalOpenSearch_Amazon
from beir.hybrid.data_ingestor import OpenSearchDataIngestor
import numpy as np
import pandas as pd

import logging
import pathlib, os, getopt, sys


def main(argv):
    opts, args = getopt.getopt(argv, "h:p:i:m:n:o:",
                               ["os_host=", "os_port=", "os_index=", "os_model_id=", "num_of_runs=", "operation="])
    dataset = 'amazon'
    endpoint = ''
    port = ''
    index = ''
    model_id = ''
    num_of_runs = 2
    operation = "evaluate"
    for opt, arg in opts:
        if opt in ("-h", "-os_host"):
            endpoint = arg
        elif opt in ("-p", "-os_port"):
            port = arg
        elif opt in ("-i", "-os_index"):
            index = arg
        elif opt in ("-m", "-os_model_id"):
            model_id = arg
        elif opt in ("-n", "-num_of_runs"):
            num_of_runs = int(arg)
        elif opt in ("-o", "-operation"):
            operation = arg


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

    #out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    #data_path = os.path.join(out_dir, dataset)

    #### Provide the data_path where scifact has been downloaded and unzipped
    #corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    df_examples = pd.read_parquet(
        'esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
    df_products = pd.read_parquet(
        'esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet')
    # df_sources = pd.read_csv("datasets/amazon/shopping_queries_dataset_sources.csv")

    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale', 'product_id'],
        right_on=['product_locale', 'product_id']
    )

    # df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
    # df_task_1_train = df_task_1[df_task_1["split"] == "train"]
    # df_task_1_test = df_task_1[df_task_1["split"] == "test"]

    # df_task_2 = df_examples_products[df_examples_products["large_version"] == 1]
    # df_task_2_train = df_task_2[df_task_2["split"] == "train"]
    # df_task_2_test = df_task_2[df_task_2["split"] == "test"]

    # df_task_3 = df_examples_products[df_examples_products["large_version"] == 1]
    # df_task_3["subtitute_label"] = df_task_3["esci_label"].apply(lambda esci_label: 1 if esci_label == "S" else 0)
    # del df_task_3["esci_label"]
    # df_task_3_train = df_task_3[df_task_3["split"] == "train"]
    # df_task_3_test = df_task_3[df_task_3["split"] == "test"]

    # df_examples_products_source = pd.merge(
    #     df_examples_products,
    #     df_sources,
    #     how='left',
    #     left_on=['query_id'],
    #     right_on=['query_id']
    # )

    df_examples_products_us = df_examples_products[df_examples_products.product_locale == "us"].copy()

    df_examples_products_us["bullet_des"] = df_examples_products_us["product_bullet_point"].fillna(" ") + \
                                            df_examples_products_us["product_description"].fillna(" ")

    df_examples_products_us["data"] = df_examples_products_us["product_color"].fillna(" ") + \
                                      df_examples_products_us["product_title"].fillna(" ") + \
                                      df_examples_products_us["bullet_des"]

    df_task_1_us = df_examples_products_us[df_examples_products_us["small_version"] == 1]
    df_task_1_train_us = df_task_1_us[df_task_1_us["split"] == "train"]
    df_task_1_test_us = df_task_1_us[df_task_1_us["split"] == "test"]

    i = np.random.randint(len(df_task_1_test_us))
    df_task_1_test_us[i:i + 10]

    # tot_queries = {}
    # for index, row in df_task_1.iterrows():
    #     if row["query_id"] not in tot_queries:
    #         tot_queries[row["query_id"]] = row["query"]

    val_legend = {"E": 100, "S": 10, "C": 1, "I": 0}

    print("Preparing queries")
    queries = {}
    for idx, row in df_task_1_test_us.iterrows():
        if row["query_id"] not in queries:
            queries[str(row["query_id"])] = row["query"]

    print("Preparing qrels")
    qrels = {}
    for idx, row in df_task_1_test_us.iterrows():
        if row["query_id"] not in qrels:
            qrels[str(row["query_id"])] = {row["product_id"]: val_legend[row["esci_label"]]}
        else:
            qrels[str(row["query_id"])].update({row["product_id"]: val_legend[row["esci_label"]]})

    print("Preparing corpus")
    corpus = {}
    for idx, row in df_task_1_us.iterrows():
        if row["product_id"] not in corpus:
            corpus[str(row["product_id"])] = {"text": row["data"]}

    if operation == 'ingest' or operation == 'both':
        ingest_data(corpus, endpoint, index, port)

    if operation == 'evaluate' or operation == 'both':
        evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs)


def ingest_data(corpus, endpoint, index, port):
    OpenSearchDataIngestor(endpoint, port).ingest(corpus, index=index)


def evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs):
    # This k values are being used for BM25 search
    # bm25_k_values = [1, 3, 5, 10, 100, min(9999, len(corpus))]
    bm25_k_values = [1, 3, 5, 10, 100]
    # This K values are being used for dense model search
    model_k_values = [1, 3, 5, 10, 100]
    # this k values are being used for scoring
    k_values = [5, 10, 100]
    # for method in ['bm25', 'neural', 'hybrid']:
    # for method in ['hybrid']:
    # for method in ['neural']:

    # for method in ['bm25']:
    #     print('starting search method ' + method)
    #     os_retrival = RetrievalOpenSearch(endpoint, port,
    #                                       index_name=index,
    #                                       model_id=model_id,
    #                                       search_method=method,
    #                                       pipeline_name='norm-pipeline')
    #     retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
    #     result_size = max(bm25_k_values)
    #     results = os_retrival.search_bm25(corpus, queries, top_k=result_size)
    #     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    #     print('--- end of results for ' + method)

    #for method in ['neural', 'hybrid']:
    for method in ['hybrid']:
    # for method in ['hybrid', 'bool']:
        print('starting search method ' + method)
        os_retrival = RetrievalOpenSearch_Amazon(endpoint, port,
                                          index_name=index,
                                          model_id=model_id,
                                          search_method=method,
                                          pipeline_name='norm-pipeline')
        retriever = EvaluateRetrieval(os_retrival, model_k_values)  # or "cos_sim" for cosine similarity
        top_k = max(model_k_values)
        result_size = max(bm25_k_values)
        # results = retriever.retrieve(corpus, queries)
        all_experiments_took_time = []
        for run in range(0, num_of_runs) :
            results, took_time = os_retrival.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
            all_experiments_took_time.append(took_time)
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
        # print('Total time: ' + str(total_time))
        retriever.evaluate_time(all_experiments_took_time)
        print('--- end of results for ' + method)

    # method = 'hybrid'
    # print('starting search method ' + method)
    # os_retrival = RetrievalOpenSearch(endpoint, port,
    # index_name=index,
    # model_id=model_id,
    # search_method=method,
    # pipeline_name='norm-pipeline')
    # retriever = EvaluateRetrieval(os_retrival, bm25_k_values, model_k_values)  # or "cos_sim" for cosine similarity
    # results = retriever.retrieve(corpus, queries)
    # results = retriever.search(corpus, queries, top_k)
    # ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    # print('--- end of results for ' + method)


if __name__ == "__main__":
    main(sys.argv[1:])
