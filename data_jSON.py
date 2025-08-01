import json
import openai
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pymongo import MongoClient
import chardet
from nltk.tokenize import sent_tokenize
import tiktoken
import nltk
import logging

nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    try:
        with open('test.txt', "rb") as file:
            raw_data = file.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']
    except Exception as e:
        logging.error(f"Error detecting file encoding: {e}")
        raise

def load_text_file(file_path):
    """Load content from a text file using detected encoding."""
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="replace") as file:
            content = file.read()
        return content
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        raise

def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in the text."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        raise

def chunk_text(text, max_tokens=1000):
    """Split text into smaller chunks based on sentences."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    token_count = 0

    for sentence in sentences:
        sentence_tokens = len(sentence) // 4  # Approximate token count
        if token_count + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk, token_count = sentence, sentence_tokens
        else:
            current_chunk += (" " if current_chunk else "") + sentence
            token_count += sentence_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def extract_data_with_llm(content, extraction_goal):
    """Extract data using OpenAI's LLM."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set. Please check your .env file.")

        openai.api_key = api_key
        prompt = f"""
Extract the following information in JSON format: {extraction_goal}.
Here is the text:
{content}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for text analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        extracted_data = response['choices'][0]['message']['content']
        if not extracted_data.strip():
            raise ValueError("The extracted data is empty.")
        return extracted_data

    except Exception as e:
        logging.error(f"Error extracting data with LLM: {e}")
        raise

def extract_data_in_chunks(content, extraction_goal):
    """Process the text in chunks."""
    chunks = chunk_text(content, max_tokens=1000)
    all_extracted_data = []

    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}...")
        try:
            extracted_data = extract_data_with_llm(chunk, extraction_goal)
            json_data = json.loads(extracted_data)
            all_extracted_data.append(json_data)
        except Exception as e:
            logging.warning(f"Error processing chunk {i + 1}: {e}")
            smaller_chunks = chunk_text(chunk, max_tokens=500)
            for sub_chunk in smaller_chunks:
                try: 
                    sub_extracted_data = extract_data_with_llm(sub_chunk, extraction_goal)
                    sub_json_data = json.loads(sub_extracted_data)
                    all_extracted_data.append(sub_json_data)
                except Exception as sub_e:
                    logging.warning(f"Error processing sub-chunk: {sub_e}")

    return all_extracted_data

def combine_results(results):
    """Combine extracted data from all chunks."""
    combined_result = {}
    for result in results:
        for key, value in result.items():
            if key not in combined_result:
                combined_result[key] = value
            else:
                if isinstance(value, list):
                    combined_result[key].extend(value)
                elif isinstance(value, dict):
                    combined_result[key].update(value)
                else:
                    combined_result[key] = [combined_result[key], value]
    return combined_result

def insert_data_into_mongodb(json_data):
    """Insert JSON data into MongoDB."""
    try:
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/?authSource=admin"))
        db = client["scraped_data"]
        collection = db["extracted_info"]

        if isinstance(json_data, list):
            collection.insert_many(json_data)
        else:
            collection.insert_one(json_data)

        logging.info("Data inserted into MongoDB successfully.")
    except Exception as e:
        logging.error(f"Error inserting data into MongoDB: {e}")
        raise

def main():
    try:
        file_path = "test.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        logging.info("Loading the text file...")
        content = load_text_file(file_path)
        logging.info("Text file loaded successfully.")

        extraction_goal = """
        Extract the following information in JSON format:
        1. Contact Information:
           - Contact number
           - Email ID
           - Address
        2. Attachments:
           - Attachment URL
           - Image URL
           - Media file
           - Media type
        3. Social Profiles:
           - LinkedIn URL
           - X (Twitter) URL
           - Facebook URL
           - Instagram URL
           - YouTube URL
           - Other URLs
        4. Team Member Information:
           - Name
           - Designation
           - LinkedIn profile
           - Unique selling point (USP)
        5. Ratings and Awards:
           - Platform name
           - Platform rating
           - Number of ratings
           - Award picture
        6. Compliance:
           - Certification name
           - Issuing body
           - Validity
           - Attachment URL
        7. Portfolio:
           - Service category
           - Average price
           - Projects completed
           - Preferred client size (one, two, three)
        8. Financial Document:
           - Document type
           - Document number
           - KYC document (file)
        Ensure the JSON output is properly formatted, syntactically correct, and complete.
        """

        logging.info("Extracting data...")
        extracted_data = extract_data_in_chunks(content, extraction_goal)
        combined_data = combine_results(extracted_data)
        logging.info("Data extracted successfully.")

        logging.info("Inserting data into MongoDB...")
        insert_data_into_mongodb(combined_data)
        logging.info("Data inserted successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
