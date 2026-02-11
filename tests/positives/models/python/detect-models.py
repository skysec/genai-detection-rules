"""
Positive test cases for AI models detection rule
These should all be detected by the detect-ai-models rule
"""

# OpenAI GPT models - current and future versions
# ruleid: detect-ai-models
response = client.chat.completions.create(model="gpt-4")
# ruleid: detect-ai-models
response = client.chat.completions.create(model="gpt-4-turbo")
# ruleid: detect-ai-models
response = client.chat.completions.create(model="gpt-3.5-turbo")
# ruleid: detect-ai-models
response = client.chat.completions.create(model="gpt-5")  # Future version
# ruleid: detect-ai-models
response = client.chat.completions.create(model="gpt-4-32k-0314")

# OpenAI o1 models
# ruleid: detect-ai-models
response = client.chat.completions.create(model="o1-preview")
# ruleid: detect-ai-models
response = client.chat.completions.create(model="o1-mini")
# ruleid: detect-ai-models
response = client.chat.completions.create(model="o1-2025")  # Future version

# OpenAI o3 models (future)
# ruleid: detect-ai-models
response = client.chat.completions.create(model="o3-preview")

# OpenAI DALL-E models
# ruleid: detect-ai-models
image = client.images.generate(model="dall-e-3")
# ruleid: detect-ai-models
image = client.images.generate(model="dall-e-2")
# ruleid: detect-ai-models
image = client.images.generate(model="dall-e-4")  # Future version

# OpenAI Whisper models
# ruleid: detect-ai-models
transcription = client.audio.transcriptions.create(model="whisper-1")
# ruleid: detect-ai-models
transcription = client.audio.transcriptions.create(model="whisper-2")  # Future version

# OpenAI TTS models
# ruleid: detect-ai-models
speech = client.audio.speech.create(model="tts-1")
# ruleid: detect-ai-models
speech = client.audio.speech.create(model="tts-1-hd")
# ruleid: detect-ai-models
speech = client.audio.speech.create(model="tts-2")  # Future version

# OpenAI embedding models
# ruleid: detect-ai-models
embedding = client.embeddings.create(model="text-embedding-3-small")
# ruleid: detect-ai-models
embedding = client.embeddings.create(model="text-embedding-3-large")
# ruleid: detect-ai-models
embedding = client.embeddings.create(model="text-embedding-ada-002")

# Anthropic Claude models - current and future versions
# ruleid: detect-ai-models
message = client.messages.create(model="claude-3-opus-20240229")
# ruleid: detect-ai-models
message = client.messages.create(model="claude-3-sonnet-20240229")
# ruleid: detect-ai-models
message = client.messages.create(model="claude-3-haiku-20240307")
# ruleid: detect-ai-models
message = client.messages.create(model="claude-3-5-sonnet-20241022")
# ruleid: detect-ai-models
message = client.messages.create(model="claude-3-5-haiku-20241022")
# ruleid: detect-ai-models
message = client.messages.create(model="claude-4-opus-20250101")  # Future version
# ruleid: detect-ai-models
message = client.messages.create(model="claude-instant-1.2")

# Google Gemini models - current and future versions
# ruleid: detect-ai-models
response = model.generate_content(model="gemini-pro")
# ruleid: detect-ai-models
response = model.generate_content(model="gemini-1.5-pro")
# ruleid: detect-ai-models
response = model.generate_content(model="gemini-1.5-flash")
# ruleid: detect-ai-models
response = model.generate_content(model="gemini-2.0-ultra")  # Future version
# ruleid: detect-ai-models
response = model.generate_content(model="gemini-ultra")

# Mistral models - current and future versions
# ruleid: detect-ai-models
response = client.chat(model="mistral-large-latest")
# ruleid: detect-ai-models
response = client.chat(model="mistral-medium-latest")
# ruleid: detect-ai-models
response = client.chat(model="mistral-small-latest")
# ruleid: detect-ai-models
response = client.chat(model="mistral-7b")
# ruleid: detect-ai-models
response = client.chat(model="mistral-xl")  # Future version

# Mixtral models
# ruleid: detect-ai-models
response = client.chat(model="mixtral-8x7b")
# ruleid: detect-ai-models
response = client.chat(model="mixtral-8x22b")
# ruleid: detect-ai-models
response = client.chat(model="mixtral-16x14b")  # Future version

# Meta Llama models - current and future versions
# ruleid: detect-ai-models
response = client.chat(model="llama-2-70b")
# ruleid: detect-ai-models
response = client.chat(model="llama-2-13b")
# ruleid: detect-ai-models
response = client.chat(model="llama-3-70b")
# ruleid: detect-ai-models
response = client.chat(model="llama-3.1-405b")
# ruleid: detect-ai-models
response = client.chat(model="llama-4-1t")  # Future version

# Cohere models - current and future versions
# ruleid: detect-ai-models
response = client.chat(model="command")
# ruleid: detect-ai-models
response = client.chat(model="command-light")
# ruleid: detect-ai-models
response = client.chat(model="command-r")
# ruleid: detect-ai-models
response = client.chat(model="command-r-plus")
# ruleid: detect-ai-models
response = client.chat(model="command-2")  # Future version

# Dictionary-style usage
# ruleid: detect-ai-models
config = {"model": "gpt-4", "temperature": 0.7}
# ruleid: detect-ai-models
settings = {"api_key": "key", "model": "claude-3-opus-20240229"}
# ruleid: detect-ai-models
params = {"model": "gemini-pro", "max_tokens": 1000}

# LangChain usage examples
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# ruleid: detect-ai-models
llm = ChatOpenAI(model="gpt-4")
# ruleid: detect-ai-models
llm = ChatOpenAI(model="gpt-4-turbo")
# ruleid: detect-ai-models
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
# ruleid: detect-ai-models
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
