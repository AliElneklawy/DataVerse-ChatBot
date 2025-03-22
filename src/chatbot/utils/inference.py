import joblib
import logging
from sklearn.ensemble import  RandomForestClassifier
from sentence_transformers import SentenceTransformer

try:
    from .paths import CLF_PATH
except ImportError:
    from paths import CLF_PATH

logger = logging.getLogger(__name__)

if CLF_PATH.exists():
    classifier: RandomForestClassifier = joblib.load(CLF_PATH)

def embed(text: str):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings.reshape(1, -1) 

def predict(response_embedding):
    return classifier.predict(response_embedding)

def inference_pipeline(llm_response) -> list:
    logger.info(f"Running inference on response: {llm_response[:50]}...")
    response_embedding = embed(llm_response)
    prediction = predict(response_embedding) # 1: doesn't know, 0: knows

    return prediction


if __name__ == "__main__":
    response = "I'm sorry, but as a customer support agent, I cannot provide information about individuals or topics unrelated to the services or products offered by the company. My role is to assist with queries and concerns related to customer support and the specific training data provided. If you have any questions or need assistance regarding the services or products offered by the company, feel free to ask, and I'll be glad to help."
    prediction = inference_pipeline(response[:60])

    print(prediction[0])
    if prediction == 1:
        print("The model doesn't know the answer...")
    else:
        print("The model knows the answer!")
