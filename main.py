import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import google.generativeai as palm
from supabase import create_client, Client
import pinecone
from dotenv import load_dotenv
load_dotenv()


ANTHROPIC_KEY = os.getenv('ANTHROPIC_KEY')

pinecone.init(      
    api_key=os.getenv('PINECONE_KEY'),
    environment='us-east1-gcp'      
)      
index = pinecone.Index('yoga')

palm.configure(api_key=os.getenv('PALM_KEY'))


url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)
model = 'models/embedding-gecko-001'


app = FastAPI()


@app.get("/query/")
def query(text: str):
    query = text
    print(query)
    embedding_x = palm.generate_embeddings(model=model, text=query)
    res = index.query(
    vector=embedding_x['embedding'],
    top_k=3,
    )
    topk = [item['id'] for item in res['matches']]

    top_videos =  supabase.table('videos').select("*").in_("id", topk).execute()
    return JSONResponse(content=top_videos.data)
