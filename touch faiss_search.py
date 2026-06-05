from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


documents = [
    "To reset your password, click on the forgot password link.",
    "Billing invoices can be downloaded from the billing section.",
    "Users can update account details in profile settings.",
    "If you cannot login, verify your username and password.",
    "Subscription plans can be upgraded anytime.",
    "Refund requests are processed within seven business days.",
    "Two-factor authentication improves account security.",
    "Payment methods can be changed from account settings.",
    "Locked accounts can be recovered through email verification.",
    "Customer support is available 24 hours a day."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate Embeddings
embeddings = model.encode(documents)
print("Embedding Shape:", embeddings.shape)


# Create FAISS Index
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatL2(384)
index.add(embeddings)
print("Total vectors:", index.ntotal)


# Semantic searh function
def search(query, k=3):
    
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    print("\nTop Matches:\n")

    for rank, idx in enumerate(indices[0]):
        print(
            f"Rank {rank+1} | Score: {distances[0][rank]:.4f} | {documents[idx]}"
        )

# Test 3 Queries
search("I forgot my password")

search("How can I download my invoice?")

search("My account is locked")

while True:

    query = input("\nEnter Query: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    search(query)