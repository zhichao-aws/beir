from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.evaluation import EvaluateRetrieval
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search import RetrievalOpenSearch
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.hybrid.data_ingestor import OpenSearchDataIngestor

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
#### Download scifact.zip dataset and unzip the dataset
#dataset = "nfcorpus"
dataset = "trec-covid"
#dataset = "arguana"
#dataset = 'fiqa'
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the SBERT model and retrieve using cosine-similarity
#model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)

endpoint = 'hybs-4-OpenSearchLB-1320635078.us-east-1.elb.amazonaws.com' # - trec-covid
#endpoint = 'hybs-5-OpenSearchLB-1396788650.us-east-1.elb.amazonaws.com' #- arguana
#endpoint = 'hybs-6-OpenSearchLB-1452211509.us-east-1.elb.amazonaws.com' #- fiqa
#endpoint = 'hybs-7-OpenSearchLB-1577729212.us-east-1.elb.amazonaws.com' # -nfcorpus
#endpoint = 'hybs-8-OpenSearchLB-2143095750.us-east-1.elb.amazonaws.com'
port = '80'
#endpoint = 'localhost'
#port = '9200'
index = "my-nlp-index-1"
#model_id = "Zx_A3IkBQ0LOZMZR1b8m"
model_id = "pCOW4okBxZxBf4MfceXj" #4
#model_id = "lbMf4okBL2YOs_JqafSc" #5
#model_id = "G-Qb3okB4hWhUso-3Qik" #6
#model_id = "p9MA4okBVFksWKKh6fnq"
#model_id = "5YGL4YkBk0hBbFgEeEvl" # local

#OpenSearchDataIngestor(endpoint, port).ingest(corpus, index="my-nlp-index-1")

# for method in ['bm25', 'neural', 'hybrid']:
for method in ['hybrid']:
    print('starting search method ' + method)
    osRetrival = RetrievalOpenSearch(endpoint, port,
                                     index_name=index,
                                     model_id=model_id,
                                     search_method=method,
                                     pipeline_name='norm-pipeline')
    retriever = EvaluateRetrieval(osRetrival, k_values=[5, 10, 100]) # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print('--- end of results for ' + method)
