"""
Positive test cases for Mistral AI detection rule
These should all be detected by the detect-mistral rule
"""

# Test 1: Basic import
# ruleid: detect-mistral
import mistralai

# Test 2: From import
# ruleid: detect-mistral
from mistralai import Mistral

# Test 3: From client import
# ruleid: detect-mistral
from mistralai.client import MistralClient

# Test 4: From async_client import
# ruleid: detect-mistral
from mistralai.async_client import MistralAsyncClient

# Test 5: Mistral client instantiation
# ruleid: detect-mistral
client = Mistral(api_key="your_api_key")

# Test 6: Mistral client fully qualified
# ruleid: detect-mistral
client = mistralai.Mistral(api_key="your_api_key")

# Test 7: MistralClient instantiation
# ruleid: detect-mistral
client = MistralClient(api_key="your_api_key")

# Test 8: MistralClient fully qualified
# ruleid: detect-mistral
client = mistralai.MistralClient(api_key="your_api_key")

# Test 9: MistralAsyncClient instantiation
# ruleid: detect-mistral
async_client = MistralAsyncClient(api_key="your_api_key")

# Test 10: MistralAsyncClient fully qualified
# ruleid: detect-mistral
async_client = mistralai.MistralAsyncClient(api_key="your_api_key")

# Test 11: Chat operation
# ruleid: detect-mistral
response = client.chat(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Hello"}]
)

# Test 12: Chat complete
# ruleid: detect-mistral
response = client.chat.complete(
    model="mistral-medium",
    messages=[{"role": "user", "content": "Hi"}]
)

# Test 13: Chat complete async
# ruleid: detect-mistral
response = client.chat.complete_async(
    model="mistral-small",
    messages=[{"role": "user", "content": "Test"}]
)

# Test 14: Chat stream
# ruleid: detect-mistral
stream = client.chat.stream(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Stream test"}]
)

# Test 15: Chat stream async
# ruleid: detect-mistral
stream = client.chat.stream_async(
    model="mistral-medium",
    messages=[{"role": "user", "content": "Async stream"}]
)

# Test 16: Embeddings operation
# ruleid: detect-mistral
embeddings = client.embeddings(
    model="mistral-embed",
    input=["Text to embed"]
)

# Test 17: Embeddings create
# ruleid: detect-mistral
embeddings = client.embeddings.create(
    model="mistral-embed",
    input=["Another text"]
)

# Test 18: List models
# ruleid: detect-mistral
models = client.models.list()

# Test 19: Retrieve model
# ruleid: detect-mistral
model_info = client.models.retrieve("mistral-large-latest")

# Test 20: Create fine-tuning job
# ruleid: detect-mistral
job = client.fine_tuning.jobs.create(
    model="mistral-small",
    training_files=["file-123"]
)

# Test 21: List fine-tuning jobs
# ruleid: detect-mistral
jobs = client.fine_tuning.jobs.list()

# Test 22: Retrieve fine-tuning job
# ruleid: detect-mistral
job = client.fine_tuning.jobs.retrieve("job-123")

# Test 23: Cancel fine-tuning job
# ruleid: detect-mistral
client.fine_tuning.jobs.cancel("job-123")

# Test 24: Upload file
# ruleid: detect-mistral
file = client.files.upload(
    file="data.jsonl",
    purpose="fine-tune"
)

# Test 25: List files
# ruleid: detect-mistral
files = client.files.list()

# Test 26: Retrieve file
# ruleid: detect-mistral
file_info = client.files.retrieve("file-123")

# Test 27: Delete file
# ruleid: detect-mistral
client.files.delete("file-123")

# Test 28: Real-world chat example
def chat_with_mistral(prompt: str) -> str:
    """Example function using Mistral"""
    # ruleid: detect-mistral
    from mistralai import Mistral

    # ruleid: detect-mistral
    client = Mistral(api_key="your_api_key")

    # ruleid: detect-mistral
    response = client.chat(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Test 29: Streaming example
def stream_mistral_response(prompt: str):
    """Example streaming function"""
    # ruleid: detect-mistral
    from mistralai import Mistral

    # ruleid: detect-mistral
    client = Mistral(api_key="your_api_key")

    # ruleid: detect-mistral
    for chunk in client.chat.stream(
        model="mistral-medium",
        messages=[{"role": "user", "content": prompt}]
    ):
        print(chunk.choices[0].delta.content)

# Test 30: Async example
async def async_mistral_chat(prompt: str):
    """Example async function"""
    # ruleid: detect-mistral
    from mistralai import MistralAsyncClient

    # ruleid: detect-mistral
    client = MistralAsyncClient(api_key="your_api_key")

    # ruleid: detect-mistral
    response = await client.chat.complete_async(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response
