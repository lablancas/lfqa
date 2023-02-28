from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from parsel import Selector
import requests
import numpy as np
import faiss  # noqa: F401

def QARetriever() :
    tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
    model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:1')
    _ = model.eval()
    return tokenizer, model


def QASeq2Seq() :
    tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
    model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
    _ = model.eval() 
    return tokenizer, model

# extracts text from a URL and returns a list of passages from the text
# for each 100 characters in text, add a passage to passages_dset. Each passage should be a dictionary with keys 'start_character', 'end_character', 'passage_text', 'title', 'uri'
# for example, {'start_character': 0, 'end_character': 100, 'passage_text': 'This is the first passage', 'title': 'page.html', 'uri': 'https://www.apartments.com/the-club-at-stablechase-houston-tx/mjcn5gk/'}
def extract_text_from_url(url):
    response = requests.get(url)
    selector = Selector(response.text)
    text = selector.xpath('//text()').extract()
    html = ' '.join(text)

    passages_dset = []

    for i in range(0, len(html), 100):
        passage = html[i:i+100]
        passages_dset.append({
            'start_character': i,
            'end_character': i+100,
            'passage_text': passage
        })

    return passages_dset

# transfer dense index to GPU
def transfer_dense_index_to_gpu(index_name, snippets):
    faiss_res = faiss.StandardGpuResources()
    passage_reps = np.memmap(
                index_name,
                dtype='float32', mode='r',
                shape=(snippets.num_rows, 128)
    )

    index_flat = faiss.IndexFlatIP(128)
    gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, index_flat)
    gpu_index.add(passage_reps)

    return gpu_index