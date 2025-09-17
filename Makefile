# AI Component Discovery - Semgrep Rules Testing Makefile

.PHONY: test test-all test-python test-typescript test-csharp test-tools test-memory test-embeddings test-vector test-mcp-server test-mcp-client clean

# Default target - run all tests
test: test-all

# Run all tests
test-all: test-python test-typescript test-csharp

# Python tests
test-python: test-python-tools test-python-memory test-python-embeddings test-python-vector test-python-mcp

test-python-tools:
	@echo "Testing Python Tools detection rules..."
	@semgrep --config=rules/tools/python/ --test

test-python-memory:
	@echo "Testing Python Memory detection rules..."
	@semgrep --config=rules/memory/python/ --test

test-python-embeddings:
	@echo "Testing Python Embeddings detection rules..."
	@semgrep --config=rules/embeddings/python/ --test

test-python-vector:
	@echo "Testing Python Vector/Retrieval detection rules..."
	@semgrep --config=rules/vector/python/ --test

test-python-mcp:
	@echo "Testing Python MCP Server detection rules..."
	@semgrep --config=rules/mcp_server/python/ --test
	@echo "Testing Python MCP Client detection rules..."
	@semgrep --config=rules/mcp_client/python/ --test

# TypeScript tests
test-typescript: test-typescript-tools test-typescript-memory test-typescript-embeddings

test-typescript-tools:
	@echo "Testing TypeScript Tools detection rules..."
	@semgrep --config=rules/tools/typescript/ --test

test-typescript-memory:
	@echo "Testing TypeScript Memory detection rules..."
	@semgrep --config=rules/memory/typescript/ --test

test-typescript-embeddings:
	@echo "Testing TypeScript Embeddings detection rules..."
	@semgrep --config=rules/embeddings/typescript/ --test

# C# tests
test-csharp: test-csharp-tools

test-csharp-tools:
	@echo "Testing C# Tools detection rules..."
	@semgrep --config=rules/tools/csharp/ --test

# Capability-specific tests
test-tools:
	@echo "Testing all Tools detection rules..."
	@semgrep --config=rules/tools/ --test

test-memory:
	@echo "Testing all Memory detection rules..."
	@semgrep --config=rules/memory/ --test

test-embeddings:
	@echo "Testing all Embeddings detection rules..."
	@semgrep --config=rules/embeddings/ --test

test-vector:
	@echo "Testing all Vector/Retrieval detection rules..."
	@semgrep --config=rules/vector/ --test

test-mcp-server:
	@echo "Testing all MCP Server detection rules..."
	@semgrep --config=rules/mcp_server/ --test

test-mcp-client:
	@echo "Testing all MCP Client detection rules..."
	@semgrep --config=rules/mcp_client/ --test

# Framework-specific tests
test-framework-detection:
	@echo "Testing Framework detection rules..."
	@semgrep --config=rules/framework-detection/ --test

# Run semgrep with specific rule file
test-rule:
ifndef RULE
	@echo "Usage: make test-rule RULE=path/to/rule.yaml"
	@exit 1
endif
	@echo "Testing rule: $(RULE)"
	@semgrep --config=$(RULE) --test

# Validate all rules syntax
validate:
	@echo "Validating all Semgrep rules syntax..."
	@find rules/ -name "*.yaml" -exec semgrep --validate --config {} \;

# Run semgrep on sample code
scan:
ifndef TARGET
	@echo "Usage: make scan TARGET=path/to/code"
	@exit 1
endif
	@echo "Scanning $(TARGET) with all rules..."
	@semgrep --config=rules/ $(TARGET)

# Clean any temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete

# Help target
help:
	@echo "Available targets:"
	@echo "  test           - Run all tests (default)"
	@echo "  test-all       - Run all tests"
	@echo "  test-python    - Run all Python rule tests"
	@echo "  test-typescript- Run all TypeScript rule tests"
	@echo "  test-csharp    - Run all C# rule tests"
	@echo "  test-tools     - Test all Tools detection rules"
	@echo "  test-memory    - Test all Memory detection rules"
	@echo "  test-embeddings- Test all Embeddings detection rules"
	@echo "  test-vector    - Test all Vector/Retrieval detection rules"
	@echo "  test-mcp-server- Test all MCP Server detection rules"
	@echo "  test-mcp-client- Test all MCP Client detection rules"
	@echo "  test-framework-detection - Test framework detection rules"
	@echo "  test-rule RULE=path - Test specific rule file"
	@echo "  validate       - Validate all rule syntax"
	@echo "  scan TARGET=path - Scan code with all rules"
	@echo "  clean          - Clean temporary files"
	@echo "  help           - Show this help"