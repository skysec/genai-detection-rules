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
	@if [ -d "tests/positives/tools/python" ]; then \
		for rule in rules/tools/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/tools/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/tools/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/tools/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/tools/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/tools/python/ --test; \
	fi

test-python-memory:
	@echo "Testing Python Memory detection rules..."
	@if [ -d "tests/positives/memory/python" ]; then \
		for rule in rules/memory/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/memory/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/memory/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/memory/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/memory/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/memory/python/ --test; \
	fi

test-python-embeddings:
	@echo "Testing Python Embeddings detection rules..."
	@if [ -d "tests/positives/embeddings/python" ]; then \
		for rule in rules/embeddings/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/embeddings/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/embeddings/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/embeddings/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/embeddings/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/embeddings/python/ --test; \
	fi

test-python-vector:
	@echo "Testing Python Vector/Retrieval detection rules..."
	@if [ -d "tests/positives/vector/python" ]; then \
		for rule in rules/vector/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/vector/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/vector/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/vector/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/vector/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/vector/python/ --test; \
	fi

test-python-mcp:
	@echo "Testing Python MCP Server detection rules..."
	@if [ -d "tests/positives/mcp_server/python" ]; then \
		for rule in rules/mcp_server/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/mcp_server/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/mcp_server/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/mcp_server/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/mcp_server/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/mcp_server/python/ --test; \
	fi
	@echo "Testing Python MCP Client detection rules..."
	@if [ -d "tests/positives/mcp_client/python" ]; then \
		for rule in rules/mcp_client/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/mcp_client/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/mcp_client/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/mcp_client/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/mcp_client/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/mcp_client/python/ --test; \
	fi

# TypeScript tests
test-typescript: test-typescript-tools test-typescript-memory test-typescript-embeddings

test-typescript-tools:
	@echo "Testing TypeScript Tools detection rules..."
	@if [ -d "tests/positives/tools/typescript" ]; then \
		for rule in rules/tools/typescript/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/tools/typescript/$$rulename.ts" ]; then \
				semgrep --config="$$rule" "tests/positives/tools/typescript/$$rulename.ts" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/tools/typescript/$$rulename.ts" ]; then \
				! semgrep --config="$$rule" "tests/negatives/tools/typescript/$$rulename.ts" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/tools/typescript/ --test; \
	fi

test-typescript-memory:
	@echo "Testing TypeScript Memory detection rules..."
	@if [ -d "tests/positives/memory/typescript" ]; then \
		for rule in rules/memory/typescript/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/memory/typescript/$$rulename.ts" ]; then \
				semgrep --config="$$rule" "tests/positives/memory/typescript/$$rulename.ts" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/memory/typescript/$$rulename.ts" ]; then \
				! semgrep --config="$$rule" "tests/negatives/memory/typescript/$$rulename.ts" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/memory/typescript/ --test; \
	fi

test-typescript-embeddings:
	@echo "Testing TypeScript Embeddings detection rules..."
	@if [ -d "tests/positives/embeddings/typescript" ]; then \
		for rule in rules/embeddings/typescript/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/embeddings/typescript/$$rulename.ts" ]; then \
				semgrep --config="$$rule" "tests/positives/embeddings/typescript/$$rulename.ts" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/embeddings/typescript/$$rulename.ts" ]; then \
				! semgrep --config="$$rule" "tests/negatives/embeddings/typescript/$$rulename.ts" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/embeddings/typescript/ --test; \
	fi

# C# tests
test-csharp: test-csharp-tools

test-csharp-tools:
	@echo "Testing C# Tools detection rules..."
	@if [ -d "tests/positives/tools/csharp" ]; then \
		for rule in rules/tools/csharp/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/tools/csharp/$$rulename.cs" ]; then \
				semgrep --config="$$rule" "tests/positives/tools/csharp/$$rulename.cs" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/tools/csharp/$$rulename.cs" ]; then \
				! semgrep --config="$$rule" "tests/negatives/tools/csharp/$$rulename.cs" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/tools/csharp/ --test; \
	fi

# Capability-specific tests
test-tools:
	@echo "Testing all Tools detection rules..."
	@make test-python-tools test-typescript-tools test-csharp-tools

test-memory:
	@echo "Testing all Memory detection rules..."
	@make test-python-memory test-typescript-memory

test-embeddings:
	@echo "Testing all Embeddings detection rules..."
	@make test-python-embeddings test-typescript-embeddings

test-vector:
	@echo "Testing all Vector/Retrieval detection rules..."
	@make test-python-vector

test-mcp-server:
	@echo "Testing all MCP Server detection rules..."
	@if [ -d "tests/positives/mcp_server" ]; then \
		for lang in python typescript csharp go; do \
			if [ -d "rules/mcp_server/$$lang" ] && [ -d "tests/positives/mcp_server/$$lang" ]; then \
				echo "Testing MCP Server $$lang rules..."; \
				for rule in rules/mcp_server/$$lang/*.yaml; do \
					rulename=$$(basename "$$rule" .yaml); \
					echo "Testing $$rulename..."; \
					if [ -f "tests/positives/mcp_server/$$lang/$$rulename.py" ] || [ -f "tests/positives/mcp_server/$$lang/$$rulename.ts" ] || [ -f "tests/positives/mcp_server/$$lang/$$rulename.cs" ] || [ -f "tests/positives/mcp_server/$$lang/$$rulename.go" ]; then \
						testfile=$$(find "tests/positives/mcp_server/$$lang/" -name "$$rulename.*" | head -1); \
						semgrep --config="$$rule" "$$testfile" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
					fi; \
					if [ -f "tests/negatives/mcp_server/$$lang/$$rulename.py" ] || [ -f "tests/negatives/mcp_server/$$lang/$$rulename.ts" ] || [ -f "tests/negatives/mcp_server/$$lang/$$rulename.cs" ] || [ -f "tests/negatives/mcp_server/$$lang/$$rulename.go" ]; then \
						testfile=$$(find "tests/negatives/mcp_server/$$lang/" -name "$$rulename.*" | head -1); \
						! semgrep --config="$$rule" "$$testfile" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
					fi; \
				done; \
			fi; \
		done; \
	else \
		semgrep --config=rules/mcp_server/ --test; \
	fi

test-mcp-client:
	@echo "Testing all MCP Client detection rules..."
	@if [ -d "tests/positives/mcp_client" ]; then \
		for lang in python typescript csharp go; do \
			if [ -d "rules/mcp_client/$$lang" ] && [ -d "tests/positives/mcp_client/$$lang" ]; then \
				echo "Testing MCP Client $$lang rules..."; \
				for rule in rules/mcp_client/$$lang/*.yaml; do \
					rulename=$$(basename "$$rule" .yaml); \
					echo "Testing $$rulename..."; \
					if [ -f "tests/positives/mcp_client/$$lang/$$rulename.py" ] || [ -f "tests/positives/mcp_client/$$lang/$$rulename.ts" ] || [ -f "tests/positives/mcp_client/$$lang/$$rulename.cs" ] || [ -f "tests/positives/mcp_client/$$lang/$$rulename.go" ]; then \
						testfile=$$(find "tests/positives/mcp_client/$$lang/" -name "$$rulename.*" | head -1); \
						semgrep --config="$$rule" "$$testfile" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
					fi; \
					if [ -f "tests/negatives/mcp_client/$$lang/$$rulename.py" ] || [ -f "tests/negatives/mcp_client/$$lang/$$rulename.ts" ] || [ -f "tests/negatives/mcp_client/$$lang/$$rulename.cs" ] || [ -f "tests/negatives/mcp_client/$$lang/$$rulename.go" ]; then \
						testfile=$$(find "tests/negatives/mcp_client/$$lang/" -name "$$rulename.*" | head -1); \
						! semgrep --config="$$rule" "$$testfile" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
					fi; \
				done; \
			fi; \
		done; \
	else \
		semgrep --config=rules/mcp_client/ --test; \
	fi

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