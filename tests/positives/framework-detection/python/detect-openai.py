"""
Positive test cases for OpenAI detection rule
These should all be detected by the detect-openai rule
"""

# Test 1: Basic import
# ruleid: detect-openai
import openai

# Test 2: From import with specific classes
# ruleid: detect-openai
from openai import OpenAI, AsyncOpenAI

# Test 3: From import with wildcard
# ruleid: detect-openai
from openai import *

# Test 4: Client instantiation - standard
# ruleid: detect-openai
client = OpenAI(api_key="sk-test-key")

# Test 5: Client instantiation - fully qualified
# ruleid: detect-openai
client = openai.OpenAI(api_key="sk-test-key")

# Test 6: Async client instantiation
# ruleid: detect-openai
async_client = AsyncOpenAI(api_key="sk-test-key")

# Test 7: Azure OpenAI client
# ruleid: detect-openai
azure_client = AzureOpenAI(
    api_key="azure-key",
    azure_endpoint="https://example.openai.azure.com/"
)

# Test 8: Chat completions API call
# ruleid: detect-openai
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Test 9: Streaming chat completions
# ruleid: detect-openai
stream = client.chat.completions.stream(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Test 10: Legacy completions API
# ruleid: detect-openai
completion = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Say hello"
)

# Test 11: Embeddings API
# ruleid: detect-openai
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="Some text to embed"
)

# Test 12: Audio transcription
# ruleid: detect-openai
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

# Test 13: Audio translation
# ruleid: detect-openai
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)

# Test 14: Text-to-speech
# ruleid: detect-openai
speech = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world"
)

# Test 15: Image generation
# ruleid: detect-openai
image = client.images.generate(
    model="dall-e-3",
    prompt="A sunset over mountains"
)

# Test 16: Image editing
# ruleid: detect-openai
edited_image = client.images.edit(
    image=open("image.png", "rb"),
    prompt="Add a bird"
)

# Test 17: Image variation
# ruleid: detect-openai
variation = client.images.create_variation(
    image=open("image.png", "rb")
)

# Test 18: List models
# ruleid: detect-openai
models = client.models.list()

# Test 19: Retrieve model info
# ruleid: detect-openai
model_info = client.models.retrieve("gpt-4")

# Test 20: Fine-tuning job
# ruleid: detect-openai
fine_tune = client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-3.5-turbo"
)

# Test 21: Assistants API - create assistant
# ruleid: detect-openai
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You help with math",
    model="gpt-4"
)

# Test 22: Assistants API - create thread
# ruleid: detect-openai
thread = client.beta.threads.create()

# Test 23: Real-world usage example
def chat_with_gpt(prompt: str) -> str:
    """Example function using OpenAI"""
    # ruleid: detect-openai
    client = OpenAI()
    # ruleid: detect-openai
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
