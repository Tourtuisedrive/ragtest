import mysql.connector
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Anslut till MySQL
conn = mysql.connector.connect(
    host="din_host",
    user="ditt_användarnamn",
    password="ditt_lösenord",
    database="din_databas"
)
cursor = conn.cursor()

# Skapa embedding-modell
model = SentenceTransformer('all-MiniLM-L6-v2')

# Exempeldokument
docs = [
    "Drönare över 250g måste registreras hos Transportstyrelsen.",
    "Flygning nära flygplatser kräver särskilt tillstånd.",
    "Drönare får inte flyga högre än 120 meter utan särskilt tillstånd.",
    "För kommersiell drönarflygning krävs drönarkort och ansvarsförsäkring.",
]

# Skapa tabell om den inte redan finns
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding JSON NOT NULL
)
""")
conn.commit()

# Konvertera dokument till embeddings och lagra dem i MySQL
for doc in docs:
    embedding = model.encode(doc).tolist()
    cursor.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s)", (doc, json.dumps(embedding)))

conn.commit()

# Simulera en fråga från en användare
query = "Vilka regler gäller för drönare över 250 gram?"
query_vector = model.encode(query)

# Hämta alla dokument och deras embeddings
cursor.execute("SELECT id, text, embedding FROM documents")
documents = cursor.fetchall()

# Beräkna Cosine Similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

best_match = None
highest_similarity = -1

for doc in documents:
    doc_embedding = np.array(json.loads(doc[2]))
    similarity = cosine_similarity(query_vector, doc_embedding)
    
    if similarity > highest_similarity:
        highest_similarity = similarity
        best_match = doc

retrieved_text = best_match[1] if best_match else "Ingen relevant information hittades."

# Stäng MySQL-anslutning
cursor.close()
conn.close()

# Anropa Mainly.AI/OpenAI för att generera svaret
client = OpenAI(api_key="DIN_OPENAI_API_NYCKEL")

prompt = f"""
Du är en expert på drönarregler i Sverige. Använd den hämtade informationen nedan för att svara på frågan korrekt.

Hämtad information:
{retrieved_text}

Fråga: {query}
Svar:
"""

response = client.completions.create(
    model="gpt-4",
    prompt=prompt,
    max_tokens=150
)

# Skriv ut svaret
print(response.choices[0].text)
