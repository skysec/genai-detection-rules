"""
Positive test cases for Hugging Face detection rule
These should all be detected by the detect-huggingface rule
"""

# Test 1: Import transformers
# ruleid: detect-huggingface
import transformers

# Test 2: From transformers import
# ruleid: detect-huggingface
from transformers import AutoModel, AutoTokenizer

# Test 3: Import huggingface_hub
# ruleid: detect-huggingface
import huggingface_hub

# Test 4: From huggingface_hub import
# ruleid: detect-huggingface
from huggingface_hub import HfApi, login

# Test 5: Import datasets
# ruleid: detect-huggingface
import datasets

# Test 6: From datasets import
# ruleid: detect-huggingface
from datasets import load_dataset

# Test 7: Pipeline usage
# ruleid: detect-huggingface
classifier = pipeline("text-classification")

# Test 8: Pipeline fully qualified
# ruleid: detect-huggingface
generator = transformers.pipeline("text-generation", model="gpt2")

# Test 9: AutoModel from_pretrained
# ruleid: detect-huggingface
model = AutoModel.from_pretrained("bert-base-uncased")

# Test 10: AutoTokenizer from_pretrained
# ruleid: detect-huggingface
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Test 11: AutoModelForCausalLM
# ruleid: detect-huggingface
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Test 12: AutoModelForSeq2SeqLM
# ruleid: detect-huggingface
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Test 13: AutoModelForSequenceClassification
# ruleid: detect-huggingface
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Test 14: AutoModelForTokenClassification
# ruleid: detect-huggingface
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")

# Test 15: AutoModelForQuestionAnswering
# ruleid: detect-huggingface
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Test 16: BertModel from_pretrained
# ruleid: detect-huggingface
model = BertModel.from_pretrained("bert-base-uncased")

# Test 17: BertTokenizer from_pretrained
# ruleid: detect-huggingface
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Test 18: GPT2LMHeadModel from_pretrained
# ruleid: detect-huggingface
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Test 19: GPT2Tokenizer from_pretrained
# ruleid: detect-huggingface
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Test 20: T5ForConditionalGeneration from_pretrained
# ruleid: detect-huggingface
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Test 21: T5Tokenizer from_pretrained
# ruleid: detect-huggingface
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Test 22: RobertaModel from_pretrained
# ruleid: detect-huggingface
model = RobertaModel.from_pretrained("roberta-base")

# Test 23: RobertaTokenizer from_pretrained
# ruleid: detect-huggingface
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Test 24: HfApi instantiation
# ruleid: detect-huggingface
api = HfApi()

# Test 25: HfApi fully qualified
# ruleid: detect-huggingface
api = huggingface_hub.HfApi()

# Test 26: Hub login
# ruleid: detect-huggingface
login(token="hf_...")

# Test 27: Hub login fully qualified
# ruleid: detect-huggingface
huggingface_hub.login(token="hf_...")

# Test 28: Snapshot download
# ruleid: detect-huggingface
snapshot_download(repo_id="bert-base-uncased")

# Test 29: Snapshot download fully qualified
# ruleid: detect-huggingface
huggingface_hub.snapshot_download(repo_id="gpt2")

# Test 30: Hub download
# ruleid: detect-huggingface
hf_hub_download(repo_id="bert-base-uncased", filename="config.json")

# Test 31: Hub download fully qualified
# ruleid: detect-huggingface
huggingface_hub.hf_hub_download(repo_id="gpt2", filename="model.safetensors")

# Test 32: Upload file
# ruleid: detect-huggingface
upload_file(path_or_fileobj="model.bin", repo_id="my-model")

# Test 33: Upload folder
# ruleid: detect-huggingface
upload_folder(folder_path="./models", repo_id="my-model")

# Test 34: Create repo
# ruleid: detect-huggingface
create_repo(repo_id="my-new-model")

# Test 35: InferenceClient
# ruleid: detect-huggingface
client = InferenceClient()

# Test 36: InferenceClient fully qualified
# ruleid: detect-huggingface
client = huggingface_hub.InferenceClient(model="gpt2")

# Test 37: InferenceApi
# ruleid: detect-huggingface
api = InferenceApi(repo_id="gpt2")

# Test 38: InferenceApi fully qualified
# ruleid: detect-huggingface
api = huggingface_hub.InferenceApi(repo_id="bert-base-uncased")

# Test 39: Load dataset
# ruleid: detect-huggingface
dataset = load_dataset("squad")

# Test 40: Load dataset fully qualified
# ruleid: detect-huggingface
dataset = datasets.load_dataset("imdb")

# Test 41: Dataset from_dict
# ruleid: detect-huggingface
dataset = Dataset.from_dict({"text": ["hello", "world"]})

# Test 42: Dataset from_pandas
# ruleid: detect-huggingface
dataset = Dataset.from_pandas(df)

# Test 43: Trainer
# ruleid: detect-huggingface
trainer = Trainer(model=model, args=training_args)

# Test 44: Trainer fully qualified
# ruleid: detect-huggingface
trainer = transformers.Trainer(model=model)

# Test 45: TrainingArguments
# ruleid: detect-huggingface
args = TrainingArguments(output_dir="./results")

# Test 46: TrainingArguments fully qualified
# ruleid: detect-huggingface
args = transformers.TrainingArguments(output_dir="./output")

# Test 47: Model generate
# ruleid: detect-huggingface
output = model.generate(input_ids)

# Test 48: TextGenerationPipeline
# ruleid: detect-huggingface
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Test 49: Model push to hub
# ruleid: detect-huggingface
model.push_to_hub("my-model")

# Test 50: Tokenizer push to hub
# ruleid: detect-huggingface
tokenizer.push_to_hub("my-tokenizer")

# Test 51: Dataset push to hub
# ruleid: detect-huggingface
dataset.push_to_hub("my-dataset")

# Test 52: Real-world example - text generation
def generate_text(prompt: str) -> str:
    """Example text generation with HuggingFace"""
    # ruleid: detect-huggingface
    from transformers import pipeline

    # ruleid: detect-huggingface
    generator = pipeline("text-generation", model="gpt2")

    # ruleid: detect-huggingface
    output = generator.generate(prompt, max_length=50)

    return output[0]["generated_text"]

# Test 53: Real-world example - model loading
def load_bert_model():
    """Example BERT model loading"""
    # ruleid: detect-huggingface
    from transformers import AutoModel, AutoTokenizer

    # ruleid: detect-huggingface
    model = AutoModel.from_pretrained("bert-base-uncased")

    # ruleid: detect-huggingface
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer

# Test 54: Real-world example - dataset loading
def load_training_data():
    """Example dataset loading"""
    # ruleid: detect-huggingface
    from datasets import load_dataset

    # ruleid: detect-huggingface
    dataset = load_dataset("squad")

    return dataset
