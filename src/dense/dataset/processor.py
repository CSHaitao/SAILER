from transformers import PreTrainedTokenizer


class Processor:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer


class TrainProcessor(Processor):
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256):
        super().__init__(tokenizer)
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + " " + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + " " + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class TestProcessor(Processor):
    def __init__(self, tokenizer, query_max_length=32):
        super().__init__(tokenizer)
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusProcessor(Processor):
    def __init__(self, tokenizer, text_max_length=256):
        super().__init__(tokenizer)
        self.text_max_length = text_max_length

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + " " + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}
