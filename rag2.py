# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
from datasets import load_dataset
import faiss
import numpy as np


# %%
retriever_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# %%
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the retriever model
retriever_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load data from the CSV file
df = pd.read_csv("151_ideas_updated.csv")  # Replace with the correct file path

# Ensure the correct column is used for passages
df.columns = df.columns.str.strip()  # Remove any extra spaces in column names
passages = df["Ideas"].dropna().tolist()  # Replace "Ideas" with the actual column name if different

# Encode the passages
passage_embeddings = retriever_model.encode(passages, convert_to_tensor=True)

# Print the shape of the embeddings
print("Encoded passages:", passage_embeddings.shape)


# %%
import faiss
import numpy as np

# Ensure the embeddings are converted to a NumPy array
passage_embeddings_np = passage_embeddings.cpu().numpy()  # Convert tensor to NumPy array

# Create a FAISS index
index = faiss.IndexFlatL2(passage_embeddings_np.shape[1])  # Dimensionality of the embeddings

# Add embeddings to the index
index.add(passage_embeddings_np)

print("Number of embeddings in index:", index.ntotal)


# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load the base model
base_model_name = "enhanceaiteam/Flux-Uncensored-V2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Use `mps` for Apple Silicon GPUs or CPU as a fallback
device = torch.device("mps" if torch.has_mps else "cpu")
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16 if torch.has_mps else torch.float32).to(device)

# Load the uncensored LoRA weights
lora_weights_path = "enhanceaiteam/Flux-uncensored-v2"
config = PeftConfig.from_pretrained(lora_weights_path)
model = PeftModel.from_pretrained(model, lora_weights_path)

# Example usage with an uncensored prompt
prompt = "A story about a forbidden love in a mystical forest."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


# %%
def retrieve_and_generate(query, top_k=5, min_tokens=500, max_length=1500):
    # Encode the query
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Retrieve top_k passages
    _, indices = index.search(query_embedding, top_k)
    retrieved_passages = [passages[i] for i in indices[0]]

    # Combine retrieved passages
    input_text = f"{query} {' '.join(retrieved_passages)} Discuss failures, beauty, and the certainty of death with examples."

    # Generate response
    input_ids = generator_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = generator_model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_tokens,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        early_stopping=False,
        length_penalty=1.2  # Encourage longer responses
    )

    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
query = "list points that mention beauty"
print(retrieve_and_generate(query, min_tokens=500, max_length=1000))


