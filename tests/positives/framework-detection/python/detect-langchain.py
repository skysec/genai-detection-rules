"""
Positive test cases for LangChain detection rule
These should all be detected by the detect-langchain rule
"""

# Test 1: Core langchain import
# ruleid: detect-langchain
import langchain

# Test 2: From langchain import
# ruleid: detect-langchain
from langchain import LLMChain

# Test 3: LangChain core import
# ruleid: detect-langchain
import langchain_core

# Test 4: From langchain_core import
# ruleid: detect-langchain
from langchain_core.prompts import ChatPromptTemplate

# Test 5: LangChain community import
# ruleid: detect-langchain
import langchain_community

# Test 6: From langchain_community import
# ruleid: detect-langchain
from langchain_community.llms import Ollama

# Test 7: LangChain OpenAI import
# ruleid: detect-langchain
import langchain_openai

# Test 8: From langchain_openai import
# ruleid: detect-langchain
from langchain_openai import ChatOpenAI

# Test 9: LangChain Anthropic import
# ruleid: detect-langchain
import langchain_anthropic

# Test 10: From langchain_anthropic import
# ruleid: detect-langchain
from langchain_anthropic import ChatAnthropic

# Test 11: LangChain Google import
# ruleid: detect-langchain
import langchain_google_genai

# Test 12: From langchain_google_genai import
# ruleid: detect-langchain
from langchain_google_genai import ChatGoogleGenerativeAI

# Test 13: LangChain Cohere import
# ruleid: detect-langchain
import langchain_cohere

# Test 14: From langchain_cohere import
# ruleid: detect-langchain
from langchain_cohere import ChatCohere

# Test 15: LangChain Mistral import
# ruleid: detect-langchain
import langchain_mistralai

# Test 16: From langchain_mistralai import
# ruleid: detect-langchain
from langchain_mistralai import ChatMistralAI

# Test 17: ChatOpenAI instantiation
# ruleid: detect-langchain
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Test 18: ChatOpenAI fully qualified
# ruleid: detect-langchain
llm = langchain_openai.ChatOpenAI(model="gpt-3.5-turbo")

# Test 19: ChatAnthropic instantiation
# ruleid: detect-langchain
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Test 20: ChatAnthropic fully qualified
# ruleid: detect-langchain
llm = langchain_anthropic.ChatAnthropic(model="claude-3-opus-20240229")

# Test 21: ChatGoogleGenerativeAI instantiation
# ruleid: detect-langchain
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Test 22: ChatCohere instantiation
# ruleid: detect-langchain
llm = ChatCohere(model="command")

# Test 23: ChatMistralAI instantiation
# ruleid: detect-langchain
llm = ChatMistralAI(model="mistral-large-latest")

# Test 24: Ollama instantiation
# ruleid: detect-langchain
llm = Ollama(model="llama2")

# Test 25: Ollama fully qualified
# ruleid: detect-langchain
llm = langchain_community.llms.Ollama(model="mistral")

# Test 26: ChatOllama instantiation
# ruleid: detect-langchain
llm = ChatOllama(model="llama2")

# Test 27: ChatOllama fully qualified
# ruleid: detect-langchain
llm = langchain_community.chat_models.ChatOllama(model="codellama")

# Test 28: LLMChain
# ruleid: detect-langchain
chain = LLMChain(llm=llm, prompt=prompt)

# Test 29: ConversationChain
# ruleid: detect-langchain
chain = ConversationChain(llm=llm)

# Test 30: SimpleSequentialChain
# ruleid: detect-langchain
chain = SimpleSequentialChain(chains=[chain1, chain2])

# Test 31: SequentialChain
# ruleid: detect-langchain
chain = SequentialChain(chains=[chain1, chain2])

# Test 32: Initialize agent
# ruleid: detect-langchain
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# Test 33: Create ReAct agent
# ruleid: detect-langchain
agent = create_react_agent(llm, tools, prompt)

# Test 34: Create OpenAI functions agent
# ruleid: detect-langchain
agent = create_openai_functions_agent(llm, tools, prompt)

# Test 35: AgentExecutor
# ruleid: detect-langchain
executor = AgentExecutor(agent=agent, tools=tools)

# Test 36: PromptTemplate
# ruleid: detect-langchain
prompt = PromptTemplate(template="Tell me about {topic}")

# Test 37: ChatPromptTemplate
# ruleid: detect-langchain
prompt = ChatPromptTemplate.from_messages([("system", "You are helpful")])

# Test 38: MessagesPlaceholder
# ruleid: detect-langchain
placeholder = MessagesPlaceholder(variable_name="history")

# Test 39: ConversationBufferMemory
# ruleid: detect-langchain
memory = ConversationBufferMemory()

# Test 40: ConversationBufferWindowMemory
# ruleid: detect-langchain
memory = ConversationBufferWindowMemory(k=5)

# Test 41: ConversationSummaryMemory
# ruleid: detect-langchain
memory = ConversationSummaryMemory(llm=llm)

# Test 42: TextLoader
# ruleid: detect-langchain
loader = TextLoader("file.txt")

# Test 43: PyPDFLoader
# ruleid: detect-langchain
loader = PyPDFLoader("document.pdf")

# Test 44: DirectoryLoader
# ruleid: detect-langchain
loader = DirectoryLoader("./docs")

# Test 45: UnstructuredFileLoader
# ruleid: detect-langchain
loader = UnstructuredFileLoader("file.docx")

# Test 46: FAISS from documents
# ruleid: detect-langchain
vectorstore = FAISS.from_documents(docs, embeddings)

# Test 47: Chroma from documents
# ruleid: detect-langchain
vectorstore = Chroma.from_documents(docs, embeddings)

# Test 48: Pinecone from documents
# ruleid: detect-langchain
vectorstore = Pinecone.from_documents(docs, embeddings)

# Test 49: RecursiveCharacterTextSplitter
# ruleid: detect-langchain
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Test 50: CharacterTextSplitter
# ruleid: detect-langchain
splitter = CharacterTextSplitter(chunk_size=500)

# Test 51: StrOutputParser
# ruleid: detect-langchain
parser = StrOutputParser()

# Test 52: JsonOutputParser
# ruleid: detect-langchain
parser = JsonOutputParser()

# Test 53: PydanticOutputParser
# ruleid: detect-langchain
parser = PydanticOutputParser(pydantic_object=MyModel)

# Test 54: RunnablePassthrough
# ruleid: detect-langchain
runnable = RunnablePassthrough()

# Test 55: RunnableParallel
# ruleid: detect-langchain
runnable = RunnableParallel({"a": chain1, "b": chain2})

# Test 56: RunnableSequence
# ruleid: detect-langchain
runnable = RunnableSequence(first=chain1, last=chain2)

# Test 57: Real-world RAG example
def create_rag_chain():
    """Example RAG chain using LangChain"""
    # ruleid: detect-langchain
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # ruleid: detect-langchain
    from langchain_core.prompts import ChatPromptTemplate
    # ruleid: detect-langchain
    from langchain_core.output_parsers import StrOutputParser

    # ruleid: detect-langchain
    llm = ChatOpenAI(model="gpt-4")

    # ruleid: detect-langchain
    prompt = ChatPromptTemplate.from_template("Answer: {context}")

    # ruleid: detect-langchain
    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain

# Test 58: Agent example
def create_agent_example():
    """Example agent using LangChain"""
    # ruleid: detect-langchain
    from langchain_anthropic import ChatAnthropic
    # ruleid: detect-langchain
    from langchain.agents import initialize_agent

    # ruleid: detect-langchain
    llm = ChatAnthropic(model="claude-3-opus-20240229")

    # ruleid: detect-langchain
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent="zero-shot-react-description"
    )

    return agent
