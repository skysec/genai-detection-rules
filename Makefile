# AI Component Discovery - Semgrep Rules Testing Makefile

.PHONY: test test-all test-python test-typescript test-csharp test-tools test-memory test-embeddings test-vector test-mcp-server test-mcp-client clean

# Default target - run all tests
test: test-all

# Run all tests
test-all: test-python test-typescript test-csharp test-go test-swift test-kotlin test-dart test-generic

# Python tests
test-python: test-python-framework test-python-models test-python-tools test-python-memory test-python-embeddings test-python-vector test-python-mcp

test-python-framework:
	@echo "Testing Python Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/python" ]; then \
		for rule in rules/framework-detection/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/python/ --test; \
	fi

test-python-models:
	@echo "Testing Python AI Models detection rules..."
	@if [ -d "tests/positives/models/python" ]; then \
		for rule in rules/models/python/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/models/python/$$rulename.py" ]; then \
				semgrep --config="$$rule" "tests/positives/models/python/$$rulename.py" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/models/python/$$rulename.py" ]; then \
				! semgrep --config="$$rule" "tests/negatives/models/python/$$rulename.py" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/models/python/ --test; \
	fi

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
test-typescript: test-typescript-framework test-typescript-models test-typescript-tools test-typescript-memory test-typescript-embeddings

test-typescript-framework:
	@echo "Testing TypeScript Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/typescript" ]; then \
		for rule in rules/framework-detection/typescript/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/typescript/$$rulename.ts" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/typescript/$$rulename.ts" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/typescript/$$rulename.ts" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/typescript/$$rulename.ts" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/typescript/ --test; \
	fi

test-typescript-models:
	@echo "Testing TypeScript AI Models detection rules..."
	@if [ -d "tests/positives/models/typescript" ]; then \
		for rule in rules/models/typescript/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/models/typescript/$$rulename.ts" ]; then \
				semgrep --config="$$rule" "tests/positives/models/typescript/$$rulename.ts" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/models/typescript/$$rulename.ts" ]; then \
				! semgrep --config="$$rule" "tests/negatives/models/typescript/$$rulename.ts" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/models/typescript/ --test; \
	fi

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
test-csharp: test-csharp-framework test-csharp-tools

test-csharp-framework:
	@echo "Testing C# Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/csharp" ]; then \
		for rule in rules/framework-detection/csharp/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/csharp/$$rulename.cs" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/csharp/$$rulename.cs" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/csharp/$$rulename.cs" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/csharp/$$rulename.cs" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/csharp/ --test; \
	fi

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
test-models:
	@echo "Testing all AI Models detection rules..."
	@make test-python-models test-typescript-models test-go-models

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

# Go tests
test-go: test-go-framework test-go-models

test-go-framework:
	@echo "Testing Go Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/go" ]; then \
		for rule in rules/framework-detection/go/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/go/$$rulename.go" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/go/$$rulename.go" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/go/$$rulename.go" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/go/$$rulename.go" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/go/ --test; \
	fi

test-go-models:
	@echo "Testing Go AI Models detection rules..."
	@if [ -d "tests/positives/models/go" ]; then \
		for rule in rules/models/go/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/models/go/$$rulename.go" ]; then \
				semgrep --config="$$rule" "tests/positives/models/go/$$rulename.go" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/models/go/$$rulename.go" ]; then \
				! semgrep --config="$$rule" "tests/negatives/models/go/$$rulename.go" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/models/go/ --test; \
	fi

# Swift tests
test-swift: test-swift-framework

test-swift-framework:
	@echo "Testing Swift Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/swift" ]; then \
		for rule in rules/framework-detection/swift/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/swift/$$rulename.swift" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/swift/$$rulename.swift" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/swift/$$rulename.swift" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/swift/$$rulename.swift" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/swift/ --test; \
	fi

# Kotlin tests
test-kotlin: test-kotlin-framework

test-kotlin-framework:
	@echo "Testing Kotlin Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/kotlin" ]; then \
		for rule in rules/framework-detection/kotlin/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/kotlin/$$rulename.kt" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/kotlin/$$rulename.kt" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/kotlin/$$rulename.kt" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/kotlin/$$rulename.kt" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/kotlin/ --test; \
	fi

# Dart tests
test-dart: test-dart-framework

test-dart-framework:
	@echo "Testing Dart Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/dart" ]; then \
		for rule in rules/framework-detection/dart/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			if [ -f "tests/positives/framework-detection/dart/$$rulename.dart" ]; then \
				semgrep --config="$$rule" "tests/positives/framework-detection/dart/$$rulename.dart" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			if [ -f "tests/negatives/framework-detection/dart/$$rulename.dart" ]; then \
				! semgrep --config="$$rule" "tests/negatives/framework-detection/dart/$$rulename.dart" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/dart/ --test; \
	fi

# Generic tests
test-generic: test-generic-framework

test-generic-framework:
	@echo "Testing Generic Framework detection rules..."
	@if [ -d "tests/positives/framework-detection/generic" ]; then \
		for rule in rules/framework-detection/generic/*.yaml; do \
			rulename=$$(basename "$$rule" .yaml); \
			echo "Testing $$rulename..."; \
			testfile=$$(find "tests/positives/framework-detection/generic/" -name "$$rulename.*" | head -1); \
			if [ -n "$$testfile" ]; then \
				semgrep --config="$$rule" "$$testfile" > /dev/null && echo "  ✓ Positive test passed" || echo "  ✗ Positive test failed"; \
			fi; \
			negtestfile=$$(find "tests/negatives/framework-detection/generic/" -name "$$rulename.*" | head -1); \
			if [ -n "$$negtestfile" ]; then \
				! semgrep --config="$$rule" "$$negtestfile" --json | jq -e '.results | length > 0' > /dev/null && echo "  ✓ Negative test passed" || echo "  ✗ Negative test failed (false positives detected)"; \
			fi; \
		done; \
	else \
		semgrep --config=rules/framework-detection/generic/ --test; \
	fi

# Framework-specific tests
test-framework-detection:
	@echo "Testing all Framework detection rules..."
	@make test-python-framework test-typescript-framework test-csharp-framework test-go-framework test-swift-framework test-kotlin-framework test-dart-framework test-generic-framework

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
	@echo "  test                    - Run all tests (default)"
	@echo "  test-all                - Run all tests"
	@echo "  test-python             - Run all Python rule tests"
	@echo "  test-typescript         - Run all TypeScript rule tests"
	@echo "  test-csharp             - Run all C# rule tests"
	@echo "  test-go                 - Run all Go rule tests"
	@echo "  test-swift              - Run all Swift rule tests"
	@echo "  test-kotlin             - Run all Kotlin rule tests"
	@echo "  test-dart               - Run all Dart rule tests"
	@echo "  test-generic            - Run all Generic rule tests"
	@echo "  test-framework-detection- Test all framework detection rules"
	@echo "  test-models             - Test all AI models detection rules"
	@echo "  test-tools              - Test all Tools detection rules"
	@echo "  test-memory             - Test all Memory detection rules"
	@echo "  test-embeddings         - Test all Embeddings detection rules"
	@echo "  test-vector             - Test all Vector/Retrieval detection rules"
	@echo "  test-mcp-server         - Test all MCP Server detection rules"
	@echo "  test-mcp-client         - Test all MCP Client detection rules"
	@echo "  test-rule RULE=path     - Test specific rule file"
	@echo "  validate                - Validate all rule syntax"
	@echo "  scan TARGET=path        - Scan code with all rules"
	@echo "  clean                   - Clean temporary files"
	@echo "  help                    - Show this help"