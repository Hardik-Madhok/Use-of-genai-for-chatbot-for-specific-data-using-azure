import pinecone
from sentence_transformers import SentenceTransformer,util
import openai

openai.api_type="azure"
openai.api_base="https://aiautouat.openai.azure.com"
openai.api_version = "2022-12-01"
openai.api_key = "OPENAI_API_KEY"
embedding_model = "embedding_model"
embedding_encoding = "embedding_encoding"

model = SentenceTransformer('all-MiniLM-L12-v2')

pinecone.init(api_key='PINECONE_API_KEY',environment="ENVIRONMENT")
index = pinecone.Index("sample-gpt3")

def create_prompt(query, context):
    prompt= """Give the answer for the query from the context as truthfully as possible else return a response 'Unable to fetch the accurate response'. \ 
    
    Query is given in triple backticks and context start from '###' \ 
    
    ####
    """ + context + """ \ 
    
    '''""" + query + """'''"""

    return prompt

def generate_answer(prompt):
    response = openai.Completion.create(
    engine="completions",
    prompt=prompt,
    temperature=0,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )
    return (response.choices[0].text).strip()

def addData(corpusData,url):
    id= id = index.describe_index_stats()['total_vector_count']
    for i in range(len(corpusData)):
        chunk=corpusData[i]
        chunkInfo=(str(id+i),
                   model.encode(chunk).tolist(),
                   {'title': url,'context': chunk})
        index.upsert(vectors=[chunkInfo])

def find_match(query,k):
    query_em = model.encode(query).tolist()
    result = index.query(query_em, top_k=k, includeMetadata=True)

    return [result['matches'][i]['metadata']['title'] for i in range(k)],[result['matches'][i]['metadata']['context'] for i in range(k)]

if __name__ == "__main__":
    query = input("Please ask your question: ")
    urls,res =find_match(query,3)
    context= "\n\n".join(res)
    print(context)
    prompt =create_prompt(context,query)
    #st.success("Answer: "+prompt)
    answer = generate_answer(prompt)
    print(answer)

