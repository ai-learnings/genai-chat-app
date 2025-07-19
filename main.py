from pydantic import BaseModel
from typing import List
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_ollama import OllamaEmbeddings, ChatOllama
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# llm client
# llmClient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
llmClient = ChatOllama(
    model = os.getenv("OLLAMA_MODEL"),
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
    # other params ...
)

# connect to your Atlas cluster
uri = os.getenv("MONGODB_URI")
mongoClient = MongoClient(uri, server_api=ServerApi('1'))
mongoCollection = mongoClient[os.getenv("MONGODB_DATABASE")][os.getenv("MONGODB_COLLECTION")]


# To generate query embedings
embedingModelName = os.getenv("OLLAMA_EMBEDDING_MODEL")   # mxbai-embed-large (ollama) or voyage-3.5

# vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
ollama = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=embedingModelName
)



# def embedQueryVoyage(query:str) :
#   data = [query]
#   result = vo.embed(data, model=embedingModelName)
#   vector = result.embeddings[0]
#   return vector


def embedQueryOllama(query:str) :
  vector = ollama.embed_query(query)
  return vector


def queryMongoDB(vector:List[float]):
  # define pipeline 
  pipeline = [
    {
      '$vectorSearch': {
          'index': 'llm-vec-embeding', 
          'path': 'data_embeded', 
          'queryVector': vector,
          'numCandidates': 200,
          'limit': 2
      }
    }, {
      '$project': {
        '_id': 0, 
        'data': 1, 
        'embeding_modal': 1,
        'score': {
          '$meta': 'vectorSearchScore'
        }
      }
    }
  ]
  resultSet = mongoCollection.aggregate(pipeline)
  dbResult: list[BaseEmbedingEntityLLM] = [BaseEmbedingEntityLLM.model_validate(doc) for doc in resultSet]
  # print(dbResult)
  return dbResult


class BaseEmbedingEntityLLM(BaseModel):
  data: str
  embeding_modal: str
  score: float


class AIResponse(BaseModel):
    answer: str


# def generateFinalResponseUsingGemini(promt_data:str, user_query:str) -> AIResponse:
#     # Prepare the prompt with expected output type
#     prompt = (
#         f"You are a helpful assistant. Based on the following data:\n\n"
#         f"{promt_data}\n\n"
#         f"Please answer the user query:\n\"{user_query}\"\n\n"
#         f"Return the response **only** in the following JSON format:\n\n"
#         f'{{\n  "answer": "your answer here"\n}}\n\n'
#         f"Make sure the response is strictly valid JSON."
#     )

#     model = "gemini-2.0-flash"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[types.Part.from_text(text=prompt)],
#         ),
#     ]

#     generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

#     # Get full response
#     response = llmClient.models.generate_content(
#         model=model,
#         contents=contents,
#         config=generate_content_config,
#     )

#     full_text = response.text.strip()

#     # Parse the JSON response safely
#     try:
#         parsed = AIResponse.model_validate(json.loads(full_text.strip('```json').strip('`')))
#     except Exception as e:
#         raise ValueError(f"Failed to parse response: {full_text}") from e

#     return parsed


def generateFinalResponseUsingOllama(promt_data:str, user_query:str) -> AIResponse:
    messages = [
      (
        "system", 
        f"You are a helpful assistant. Based on the following data:\n\n"
        f"{promt_data}\n\n"
        f"Please answer the human's queries\n"
        f"Return the response **only** in the following JSON format:\n\n"
        f'{{\n  "answer": "your answer here"\n}}\n\n'
        f"Make sure the response is strictly valid JSON."
      ),
      (
        "human", 
        f"My Query: \n"
        f"{user_query}\"\n\n"
      )
    ]
    ans = llmClient.invoke(messages)

    full_text = ans.content

    # Parse the JSON response safely
    try:
        parsed = AIResponse.model_validate(json.loads(full_text.strip('```json').strip('`')))
    except Exception as e:
        raise ValueError(f"Failed to parse response: {full_text}") from e

    return parsed





######################################################################################################################
if __name__ == "__main__":
  userQuery = input("query: ")    # "who is president id dhaked firm."
  print("\n")
  queryVector = embedQueryOllama(userQuery)
  dbData = queryMongoDB(queryVector)
  
  dataString = ".".join(item.data for item in dbData)
  answer = generateFinalResponseUsingOllama(dataString, userQuery)
  print(answer)

