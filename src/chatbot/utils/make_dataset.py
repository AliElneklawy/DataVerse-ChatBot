import pandas as pd
from paths import DATASETS_DIR
from utils import create_folder
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_labels(df: pd.DataFrame, column: str):
    return df[column].value_counts()

def standardize_length(df: pd.DataFrame, max_length: int = 250) -> pd.DataFrame:
    # df['response'] = df['response'].apply(lambda x: x[:max_length] 
    #                                       if len(x) > max_length 
    #                                     else x)

    logger.info("Loading the tokenizer...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = model.tokenizer
    logger.info("Tokenizer loaded successfully.")

    df['response'] = df['response'].apply(truncate_to_n_tokens, args=(tokenizer,))
    return df

def truncate_to_n_tokens(text, tokenizer, max_tokens=50):
    logger.info(f"Tokenizing {text}...")

    tokens = tokenizer.encode(text, add_special_tokens=False)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    return truncated_text


# df = standardize_length(df)
# print(df)
df = pd.read_json(create_folder(DATASETS_DIR) / "responses.json", encoding='utf-8')
df = df.sample(frac=1)
df.to_csv(DATASETS_DIR / "responses.csv", index=False, encoding='utf-8')
