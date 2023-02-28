from .lfqa_utils import make_qa_dense_index, query_qa_dense_index, qa_s2s_generate
from .utils import extract_text_from_url, QARetriever, QASeq2Seq, transfer_dense_index_to_gpu

# Extract text from webpage url
index_name = 'page.html'
url = "https://www.apartments.com/the-club-at-stablechase-houston-tx/mjcn5gk/"
passages_dset = extract_text_from_url(url)
print(passages_dset)

# create dense index
qar_model, qar_tokenizer = QARetriever()

make_qa_dense_index(
    qar_model, qar_tokenizer, passages_dset, device='cuda:0', index_name=index_name
)

qa_s2s_model, qa_s2s_tokenizer = QASeq2Seq()

gpu_index = transfer_dense_index_to_gpu(index_name, passages_dset)

# create support document with the dense index
question = "What is the address of this apartment?" 
doc, res_list = query_qa_dense_index(
    question, qar_model, qar_tokenizer,
    passages_dset, gpu_index, device='cuda:1'
)
# concatenate question and support document into BART input
question_doc = "question: {} context: {}".format(question, doc)
# generate an answer with beam search
answer = qa_s2s_generate(
        question_doc, qa_s2s_model, qa_s2s_tokenizer,
        num_answers=1,
        num_beams=8,
        min_len=64,
        max_len=256,
        max_input_length=1024,
        device="cuda:0"
)[0]

print(answer)