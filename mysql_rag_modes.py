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

# Simulera en fråga från en användare
query = "Får jag flyga 15 meter över ett föremål om det är högre än 120 meter?"
mode = "expert"  # Ändra till "beginner" för nybörjarläge

# Hämta alla dokument och deras embeddings
cursor.execute("SELECT id, text, embedding FROM documents")
documents = cursor.fetchall()

# Konvertera frågan till en embedding
query_vector = model.encode(query)

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

# Anpassa prompt beroende på läge
if mode == "expert":
    prompt = f"""
    Du är en juridisk expert på drönarregler i Sverige. Ge ett exakt svar baserat på en källa från Transportstyrelsen 
    eller annan officiell källa. Inkludera hänvisningar och länkar för vidare läsning.

    Hämtad information:
    {retrieved_text}

    Fråga: {query}
    Svar:
    """
elif mode == "beginner":
    prompt = f"""
    Förklara reglerna kring drönarflygning på en lättförståelig nivå. Ta med möjliga undantag, relaterade regler 
    (t.ex. GDPR om bilder/video sparas) och andra aspekter som kan vara viktiga.

    Hämtad information:
    {retrieved_text}

    Fråga: {query}
    Svar:
    """

response = client.completions.create(
    model="gpt-4",
    prompt=prompt,
    max_tokens=300
)

# Skriv ut svaret
print(response.choices[0].text)
