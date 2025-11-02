# ADL_HW3_RAG

## usage
- install package (python 3.12)
<pre><code>pip install -r requirements.txt</code></pre>

- save corpus into vector database (from corpus.txt)
<pre><code>python save_embbedings.py --retriever_model_path [your_model_path] --build_db</code></pre>

- create `./.env` and place your own hf_token([link](https://huggingface.co/docs/hub/security-tokens)) into `hf_token="....."`
- inference
<pre><code>python inference_batch.py --test_data_path [your_data_path] --retriever_model_path [your_retrieve_model_path] --reranker_model_path [your_rerank_model_path] –-test_data_path ./data/test_open.txt</code></pre>
