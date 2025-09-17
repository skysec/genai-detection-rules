# AI Component Discovery

This repository contains Semgrep rules for detecting AI and GenAI frameworks and their components in code. The rules help identify the usage of various AI/ML libraries, APIs, and frameworks across different programming languages.

## Repository Structure

```
rules/
├── framework-detection/
│   ├── csharp/           # C# AI framework detection rules
│   ├── dart/             # Dart AI framework detection rules
│   ├── generic/          # Language-agnostic AI detection rules
│   ├── go/               # Go AI framework detection rules
│   ├── kotlin/           # Kotlin AI framework detection rules
│   ├── python/           # Python AI framework detection rules
│   ├── swift/            # Swift AI framework detection rules
│   └── typescript/       # TypeScript/JavaScript AI framework detection rules
|-- tools/
|-- vector/
|-- embeddings/
|-- mcp_client/
|-- mcp_server/
tests/
|-- positives/
    |-- framework-detection/
        |-- csharp/
|-- negatives/
    |-- framework-detection/
        |-- csharp/
```

## Rule Organization

Rules are organized by:
- **Component/Capability**: What the rule detects (e.g., framework usage, API calls)
- **Framework**: Specific AI/ML framework (OpenAI, Anthropic, Gemini, etc.)
- **Programming Language**: Target language for the rule

## Rule Structure

Each Semgrep rule consists of:
* **Rule file** (`.yaml`): Contains the Semgrep rule definition


### Example Rule Structure
```
rules/framework-detection/python/
├── detect-openai.yaml    # Semgrep rule definition
└── detect-openai.py      # Test cases for the rule
```

## Rule Tests

Each Semgrep rule must have a test file located in the test directory:
* **Test file** (`.{language}`): Sample code to test the rule

Tests are organized under the test directory:
- **positives** or **negatives**: directory to store tests depending on their type
- **Component/Capability**: What the rule detects (e.g., framework usage, API calls)
- **Framework**: Specific AI/ML framework (OpenAI, Anthropic, Gemini, etc.)
- **Programming Language**: Target language for the rule

### Examples of Test Structure
```
tests/positives/framework-detection/python/
├── detect-openai.py    # Positive test case for detect-openai.yaml
tests/negatives/framework-detection/python/
└── detect-openai.py    # Negative test case for detect-openai.yaml
```

## Supported Frameworks

Current rules detect usage of:
- **OpenAI**: GPT models, API usage
- **Anthropic**: Claude models and API
- **Google Gemini**: Gemini AI models
- **Hugging Face**: Transformers, model loading
- **LangChain**: Framework components
- **TensorFlow**: ML framework usage
- **PyTorch**: Deep learning framework
- **Mistral**: Mistral AI models
- **Apple Core ML**: iOS/macOS ML framework
- **Vercel AI**: AI SDK usage
- **Promptfoo**: AI evaluation framework

## Testing

All rules include test cases to validate their effectiveness. Tests are executed using the Makefile at the root of the repository.

### Running Tests
```bash
make test          # Run all tests
make test-python   # Run Python-specific tests
make test-go       # Run Go-specific tests
# etc.
```

## Adding New Rules

When adding new rules:
1. Place rule files in the appropriate language directory under `rules/framework-detection/`
2. Follow naming convention: `detect-{framework}.yaml`
3. Include corresponding test file: `detect-{framework}.{ext}`
4. Ensure test cases cover both positive and negative scenarios
5. Add test execution to the Makefile

## Rule Metadata

Each rule includes metadata for:
- **Severity**: INFO level for discovery purposes
- **Technology**: AI/GenAI categorization
- **Confidence**: Confidence level in detection accuracy
- **References**: Documentation and source links