from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self, num_of_queries: int, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

        self.num_of_queries = num_of_queries

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, corpus_method="text") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        #self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        #self.check(fIn=self.corpus_file, ext="jsonl")
        #self.check(fIn=self.query_file, ext="jsonl")
        #self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus(corpus_method)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if not len(self.qrels):
            self._load_qrels()
            #self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self, corpus_method='text'):

        with open(self.corpus_file, encoding='utf8') as fIn:
            """
            {
                "target": "B00BZ8GPVO",
                "candidate": "B008MTHLHQ",
                "captions": [
                    "is longer",
                    "is lighter and longer"
                ]
            }
            """

            for tup in json.load(fIn):
                captions = tup["captions"]
                self.corpus[self.image_name(tup["target"])] = {
                                "caption": captions[0],
                                "image_name": self.image_name(tup["target"])
                            }

    def _load_queries(self):

        with open(self.corpus_file, encoding='utf8') as fIn:
            """
            {
                "target": "B00BZ8GPVO",
                "candidate": "B008MTHLHQ",
                "captions": [
                    "is longer",
                    "is lighter and longer"
                ]
            }
            """
            for tup in json.load(fIn):
                captions = tup["captions"]
                self.queries[self.image_name(tup["target"])] = {
                                "caption": captions[1],
                                "image_name": self.image_name(tup["target"])
                            }

    def _load_qrels(self):

        with open(self.corpus_file, encoding='utf8') as fIn:
            for tup in json.load(fIn):
                query_id, corpus_id, score = self.image_name(str(tup["target"])), self.image_name(str(tup["target"])), 1
                if query_id not in self.qrels:
                    self.qrels[query_id] = {corpus_id: score}
                else:
                    self.qrels[query_id][corpus_id] = score

    def image_name(self, image):
        return image + '.png'