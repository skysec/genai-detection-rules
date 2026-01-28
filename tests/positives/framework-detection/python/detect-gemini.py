"""
Positive test cases for Google Gemini detection rule
These should all be detected by the detect-gemini rule
"""

# Test 1: Basic import
# ruleid: detect-gemini
import google.generativeai

# Test 2: From import with alias
# ruleid: detect-gemini
from google.generativeai import GenerativeModel

# Test 3: From import with wildcard
# ruleid: detect-gemini
from google.generativeai import *

# Test 4: Import generativelanguage module
# ruleid: detect-gemini
import google.ai.generativelanguage

# Test 5: Configure API key
# ruleid: detect-gemini
genai.configure(api_key="AIzaSy...")

# Test 6: Configure with fully qualified path
# ruleid: detect-gemini
google.generativeai.configure(api_key="AIzaSy...")

# Test 7: Model instantiation - short form
# ruleid: detect-gemini
model = genai.GenerativeModel('gemini-pro')

# Test 8: Model instantiation - fully qualified
# ruleid: detect-gemini
model = google.generativeai.GenerativeModel('gemini-1.5-flash')

# Test 9: Generate content
# ruleid: detect-gemini
response = model.generate_content("Tell me a story")

# Test 10: Generate content async
# ruleid: detect-gemini
response = model.generate_content_async("What is AI?")

# Test 11: Stream generate content
# ruleid: detect-gemini
stream = model.stream_generate_content("Explain quantum computing")

# Test 12: Stream generate content async
# ruleid: detect-gemini
stream = model.stream_generate_content_async("Write a poem")

# Test 13: Start chat
# ruleid: detect-gemini
chat = model.start_chat(history=[])

# Test 14: Send message in chat
# ruleid: detect-gemini
response = chat.send_message("Hello")

# Test 15: Send message async in chat
# ruleid: detect-gemini
response = chat.send_message_async("How are you?")

# Test 16: Send message stream in chat
# ruleid: detect-gemini
stream = chat.send_message_stream("Tell me more")

# Test 17: Embed content - short form
# ruleid: detect-gemini
embedding = genai.embed_content(
    model="models/embedding-001",
    content="Sample text"
)

# Test 18: Embed content - fully qualified
# ruleid: detect-gemini
embedding = google.generativeai.embed_content(
    model="models/embedding-001",
    content="Another text"
)

# Test 19: Model embed content
# ruleid: detect-gemini
embedding = model.embed_content("Text to embed")

# Test 20: List models - short form
# ruleid: detect-gemini
models = genai.list_models()

# Test 21: List models - fully qualified
# ruleid: detect-gemini
models = google.generativeai.list_models()

# Test 22: Get model info - short form
# ruleid: detect-gemini
model_info = genai.get_model('models/gemini-pro')

# Test 23: Get model info - fully qualified
# ruleid: detect-gemini
model_info = google.generativeai.get_model('models/gemini-1.5-flash')

# Test 24: Count tokens
# ruleid: detect-gemini
token_count = model.count_tokens("How many tokens in this?")

# Test 25: Create tuned model
# ruleid: detect-gemini
tuned_model = genai.create_tuned_model(
    source_model="models/gemini-1.5-flash",
    training_data=training_data
)

# Test 26: List tuned models
# ruleid: detect-gemini
tuned_models = genai.list_tuned_models()

# Test 27: Get tuned model
# ruleid: detect-gemini
my_model = genai.get_tuned_model('tunedModels/my-model')

# Test 28: Update tuned model
# ruleid: detect-gemini
updated = genai.update_tuned_model(
    name='tunedModels/my-model',
    updates={'display_name': 'New Name'}
)

# Test 29: Delete tuned model
# ruleid: detect-gemini
genai.delete_tuned_model('tunedModels/my-model')

# Test 30: Real-world usage example
def chat_with_gemini(prompt: str) -> str:
    """Example function using Gemini"""
    # ruleid: detect-gemini
    import google.generativeai as genai

    # ruleid: detect-gemini
    genai.configure(api_key="AIzaSy...")

    # ruleid: detect-gemini
    model = genai.GenerativeModel('gemini-1.5-flash')

    # ruleid: detect-gemini
    response = model.generate_content(prompt)

    return response.text

# Test 31: Streaming example
async def stream_gemini_response(prompt: str):
    """Example streaming function"""
    # ruleid: detect-gemini
    model = genai.GenerativeModel('gemini-pro')

    # ruleid: detect-gemini
    async for chunk in model.stream_generate_content_async(prompt):
        print(chunk.text)

# Test 32: Chat example
def chat_example():
    """Example chat function"""
    # ruleid: detect-gemini
    model = genai.GenerativeModel('gemini-1.5-flash')

    # ruleid: detect-gemini
    chat = model.start_chat()

    # ruleid: detect-gemini
    response1 = chat.send_message("Hello")

    # ruleid: detect-gemini
    response2 = chat.send_message("What's your name?")

    return chat.history
