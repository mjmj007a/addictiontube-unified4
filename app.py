from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI, APIError
from pymilvus import connections, Collection, utility
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
import nltk
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

# Configure logging
logger = logging.getLogger('addictiontube')
logger.setLevel(logging.DEBUG)
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'unified_search.log')
file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# Initialize Flask-Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True,
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Validate environment variables
missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not MILVUS_URI:
    missing_vars.append("MILVUS_URI")
if not MILVUS_TOKEN:
    missing_vars.append("MILVUS_TOKEN")
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise EnvironmentError(error_msg)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_milvus_client():
    for attempt in range(3):
        try:
            connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
            logger.info(f"Milvus client connected, attempt {attempt + 1}")
            if utility.has_collection("Content"):
                logger.info("Content collection found")
                collection = Collection("Content")
                collection.load()
                return collection
            else:
                logger.warning("'Content' collection not found")
                raise Exception("Content collection not found")
        except Exception as e:
            logger.error(f"Milvus client connection attempt {attempt + 1} failed: {str(e)}")
    logger.error("Failed to connect to Milvus after 3 attempts")
    raise EnvironmentError("Milvus client connection failed")

# Load metadata with error handling
song_dict = {}
poem_dict = {}
story_dict = {}
base_dir = os.path.dirname(__file__)
try:
    with open(os.path.join(base_dir, 'songs_revised_with_songs-july06.json'), 'r', encoding='utf-8') as f:
        song_dict = {item['video_id']: item['song'] for item in json.load(f)}
    logger.info("Loaded songs_revised_with_songs-july06.json")
except FileNotFoundError:
    logger.warning("songs_revised_with_songs-july06.json not found, initializing empty song_dict")
except Exception as e:
    logger.error(f"Failed to load songs_revised_with_songs-july06.json: {str(e)}")
try:
    with open(os.path.join(base_dir, 'videos_revised_with_poems-july04.json'), 'r', encoding='utf-8') as f:
        poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
    logger.info("Loaded videos_revised_with_poems-july04.json")
except FileNotFoundError:
    logger.warning("videos_revised_with_poems-july04.json not found, initializing empty poem_dict")
except Exception as e:
    logger.error(f"Failed to load videos_revised_with_poems-july04.json: {str(e)}")
try:
    with open(os.path.join(base_dir, 'stories.json'), 'r', encoding='utf-8') as f:
        story_dict = {item['id']: item['text'] for item in json.load(f)}
    logger.info("Loaded stories.json")
except FileNotFoundError:
    logger.warning("stories.json not found, initializing empty story_dict")
except Exception as e:
    logger.error(f"Failed to load stories.json: {str(e)}")

# Initialize NLTK lemmatizer
try:
    nltk.download('wordnet', quiet=True, raise_on_error=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK WordNetLemmatizer initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize NLTK WordNetLemmatizer: {str(e)}. Falling back to no lemmatization.")
    lemmatizer = None

def strip_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.error(f"HTML stripping error: {str(e)}")
        return text or ''

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_embedding(text):
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except APIError as e:
        logger.error(f"OpenAI embedding failed: {str(e)}")
        raise

def preprocess_query(query):
    if lemmatizer:
        words = query.lower().split()
        lemmatized = [lemmatizer.lemmatize(word, pos='n') for word in words]
        processed = ' '.join(lemmatized)
        logger.info(f"Processed query: {query} -> {processed}")
        return processed
    logger.warning(f"No lemmatizer available, using raw query: {query}")
    return query.lower()

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    logger.info("Health check endpoint accessed")
    collection = get_milvus_client()
    try:
        try:
            collection.query(expr="type in ['songs', 'poems', 'stories']", limit=1)
            logger.info("Content collection accessible in health check")
        except Exception as e:
            logger.warning(f"Content collection query failed in health check: {str(e)}")
            return jsonify({"error": "Content collection not accessible"}, 503)
        try:
            embedding = get_embedding("test")
            logger.info("OpenAI embedding test successful")
        except APIError as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return jsonify({"error": "OpenAI health check failed", "details": str(e)}), 503
        return jsonify({"status": "ok", "message": "AddictionTube Unified API is running"}), 200
    finally:
        collection.release()
        connections.disconnect("default")

@app.route('/debug', methods=['GET'])
def debug():
    try:
        base_dir = os.path.dirname(__file__)
        debug_info = {
            "pymilvus_version": utility.__version__,
            "files_present": {
                "songs": os.path.exists(os.path.join(base_dir, 'songs_revised_with_songs-july06.json')),
                "poems": os.path.exists(os.path.join(base_dir, 'videos_revised_with_poems-july04.json')),
                "stories": os.path.exists(os.path.join(base_dir, 'stories.json')),
                "song_locations": os.path.exists(os.path.join(base_dir, 'song-locations.json')),
                "video_locations": os.path.exists(os.path.join(base_dir, 'video_locations.json'))
            },
            "working_directory": os.getcwd(),
            "app_directory": base_dir
        }
        for file_name in ['songs_revised_with_songs-july06.json', 'videos_revised_with_poems-july04.json', 'stories.json', 'song-locations.json', 'video_locations.json']:
            file_path = os.path.join(base_dir, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        debug_info[f"{file_name}_count"] = len(data) if isinstance(data, list) else 0
                        if isinstance(data, list) and data:
                            debug_info[f"{file_name}_sample_id"] = data[0].get('content_id') or data[0].get('video_id') or data[0].get('id')
                except Exception as e:
                    debug_info[f"{file_name}_error"] = str(e)
            else:
                debug_info[f"{file_name}_error"] = "File not found"
        return jsonify(debug_info)
    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Debug endpoint failed", "details": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded: {e.description}")
    return jsonify({
        "error": "Rate limit exceeded",
        "details": f"Too many requests. Please wait and try again. Limit: {e.description}"
    }), 429

@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip().lower()
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    if not query or not content_type or content_type not in ['songs', 'poems', 'stories']:
        logger.error(f"Invalid request: query='{query}', content_type='{content_type}'")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    collection = get_milvus_client()
    try:
        logger.info(f"Processing query: {query}, content_type: {content_type}, page: {page}, per_page: {size}")
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
            logger.info("Query embedding generated successfully")
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            return jsonify({"error": "Embedding service unavailable", "details": str(e)}), 500

        try:
            results = collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=size,
                offset=(page - 1) * size,
                expr=f"type == '{content_type}'",
                output_fields=["content_id", "title", "description", "url"]
            )
            logger.info(f"Milvus search returned {len(results[0])} results for query='{query}', content_type='{content_type}'")
        except Exception as e:
            logger.error(f"Milvus query failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Search service unavailable", "details": str(e)}), 500

        total = collection.query(expr=f"type == '{content_type}'", output_fields=["content_id"])
        total_count = len(total)

        items = []
        for hit in results[0]:
            entity = hit.entity
            logger.debug(f"Processing item: Content ID: {entity.get('content_id')}, Distance: {hit.distance}")
            item = {
                "content_id": entity.get("content_id", str(hit.id)),
                "distance": hit.distance,  # Milvus uses distance (0 to 2 for COSINE, lower is better)
                "title": strip_html(entity.get("title", "N/A")),
                "description": strip_html(entity.get("description", "")),
                "image": entity.get("url", "")
            }
            items.append(item)
            logger.debug(f"Item added: {item}")

        logger.info(f"Search completed: query='{query}', content_type='{content_type}', page={page}, total={total_count}, returned={len(items)}")
        return jsonify({"results": items, "total": total_count, "page": page, "per_page": size})

    except Exception as e:
        logger.error(f"Unexpected error in search_content: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        collection.release()
        connections.disconnect("default")

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip().lower()
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    if not query or not content_type or content_type not in ['songs', 'poems', 'stories']:
        logger.error(f"Invalid RAG request: query='{query}', content_type='{content_type}'")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    collection = get_milvus_client()
    try:
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
            logger.info("Query embedding generated successfully")
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            return jsonify({"error": "Embedding service unavailable", "details": str(e)}), 500

        try:
            results = collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=50,
                expr=f"type == '{content_type}'",
                output_fields=["content_id", "text"]
            )
            logger.info(f"RAG query results: {len(results[0])} objects found for query='{query}', content_type='{content_type}'")
        except Exception as e:
            logger.error(f"Milvus query failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Milvus query failed", "details": str(e)}), 500

        matches = results[0]
        if reroll:
            random.shuffle(matches)

        if not matches:
            logger.warning(f"No matches found for query='{query}', content_type='{content_type}'")
            return jsonify({"error": "No relevant context found"}), 404

        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        context_docs = []
        total_tokens = 0
        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}

        for match in matches:
            text = content_dict[content_type].get(match.entity.get("content_id", ""), match.entity.get("text", ""))
            if not text:
                logger.warning(f"Match {match.entity.get('content_id')} has no text metadata in {content_type}")
                continue
            doc = strip_html(text)[:3000]
            doc_tokens = len(encoding.encode(doc))
            if total_tokens + doc_tokens <= max_tokens:
                context_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        if not context_docs:
            logger.warning(f"No usable context data for query='{query}', content_type='{content_type}'")
            return jsonify({"error": "No usable context data found"}), 404

        context_text = "\n\n---\n\n".join(context_docs)
        system_prompt = f"You are an expert assistant for addiction recovery {content_type}."
        user_prompt = f"""Use the following {content_type} to answer the question.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            logger.info(f"RAG answer generated for query='{query}', content_type='{content_type}'")
            return jsonify({"answer": answer})
        except APIError as e:
            logger.error(f"OpenAI completion failed: {str(e)}")
            return jsonify({"error": "RAG processing failed", "details": str(e)}), 500
    finally:
        collection.release()
        connections.disconnect("default")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)