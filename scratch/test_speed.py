import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import joblib

tok_path = hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json')
onnx_path = hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx')

tokenizer = Tokenizer.from_file(tok_path)
tokenizer.enable_truncation(max_length=64)
tokenizer.enable_padding(length=64)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8
session = ort.InferenceSession(onnx_path, sess_options)

metadata = joblib.load('models/paper_metadata.joblib')[:2000]
texts = [f"{m.get('title', '')} {m.get('summary', '')}".strip() for m in metadata]

t0 = time.time()
batch_size = 512
all_embs = []

for i in range(0, len(texts), batch_size):
    b_texts = texts[i:i+batch_size]
    encs = tokenizer.encode_batch(b_texts)
    input_ids = np.array([e.ids for e in encs], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encs], dtype=np.int64)
    
    outputs = session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
    token_embs = outputs[0]
    
    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embs = np.sum(token_embs * mask, axis=1)
    sum_m = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = sum_embs / sum_m
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = (pooled / norms).astype(np.float32)
    all_embs.append(normed)

dur = time.time() - t0
print(f"Encoded {len(texts)} papers in {dur:.2f}s ({len(texts)/dur:.1f} docs/sec)")
