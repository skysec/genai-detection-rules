/**
 * Negative test cases for AI models detection rule
 * These should NOT be detected by the detect-ai-models rule
 */

// Test 1: Comments mentioning model names
// This code uses gpt-4 for processing
// We prefer claude-3 over gemini-pro
/* Multi-line comment about models:
   - gpt-4 is good
   - claude-3 is better
*/

// Test 2: String variables not assigned to model parameter
const modelDescription = "We use gpt-4 in production";
const documentation = "Claude-3-opus is the best model";
const notes = "Consider gemini-pro for embeddings";

// Test 3: Object properties that don't match model patterns
const config = {
    modelName: "custom-model",
    modelType: "transformer",
    modelVersion: "v1.0",
    engine: "neural-network"
};

// Test 4: Function/class names with model-like strings
function trainGptModel() {
    /** Train a custom GPT-like model */
}

class ClaudeWrapper {
    /** Wrapper for claude-like functionality */
}

function processWithGeminiLogic() {
    /** Process data using gemini-inspired logic */
}

// Test 5: Non-AI model references
const databaseModel = "postgresql";
const mlModel = "random-forest";
const predictionModel = "linear-regression";

// Test 6: URL and endpoint references
const apiUrl = "https://api.openai.com/v1/chat/completions";
const endpoint = "/models/gpt-4/info";
const docsUrl = "https://platform.openai.com/docs/models/gpt-4";

// Test 7: Custom model names that don't match patterns
const response1 = await client.chat.completions.create({ model: "my-custom-model" });
const response2 = await client.chat.completions.create({ model: "company-llm-v2" });
const response3 = await client.chat.completions.create({ model: "internal-ai-engine" });

// Test 8: Variable assignments without AI model patterns
const model = "custom-transformer";
const aiModel = "proprietary-llm";
const llmName = "in-house-model";

// Test 9: Model names as part of larger strings (not exact matches)
const description = "The gpt-4-like model performs well";
const note = "Based on claude-3 architecture";
const comment = "Similar to gemini-pro functionality";

// Test 10: Object with model key but non-AI model value
const settings = { model: "custom-engine", version: "1.0" };
const params = { model: "proprietary", temperature: 0.7 };

// Test 11: Model in log messages
console.log("Using model: custom-model-v1");
console.info(`Model: ${customModelName}`);
logger.info("Current model: internal-model");

// Test 12: Test data and fixtures
const TEST_MODEL = "test-model-fixture";
const MOCK_MODEL = "mock-ai-model";
const SAMPLE_MODEL = "sample-model-data";

// Test 13: Configuration file content (as strings)
const configYaml = `
model:
  name: custom-model
  version: 1.0
`;

const configJson = '{"model": "custom-engine", "api": "v1"}';

// Test 14: Environment variables
const MODEL_NAME = process.env.MODEL_NAME || "default-model";
const AI_MODEL = process.env.AI_MODEL;

// Test 15: Interface/Type definitions
interface AIConfig {
    modelType: string;
    modelArchitecture: string;
    modelFramework: string;
}

type ModelConfig = {
    name: string;
    version: string;
};

// Test 16: Function parameters named model (but with custom values)
function initializeModel(model: string = "custom-model", config?: any) {
    /** Initialize with custom model */
}

// Test 17: Model comparison or conditional logic
const modelFamily = ["gpt", "claude", "gemini"]; // Generic names, not specific models
if (modelFamily.includes(modelName)) {
    process();
}

// Test 18: Model name fragments (incomplete)
const modelPrefix = "gpt";
const modelFamilyName = "claude";
const modelSeries = "gemini";

// Test 19: Open source model names that don't match our patterns
const oss1 = await client.chat({ model: "falcon-40b" });
const oss2 = await client.chat({ model: "vicuna-13b" });
const oss3 = await client.chat({ model: "alpaca-7b" });
const oss4 = await client.chat({ model: "bloom-176b" });

// Test 20: Version numbers alone
const version = "3.5";
const modelVer = "4.0";
const apiVersion = "2.1";

// Test 21: Template literals without models
const message = `Using custom model for ${task}`;
const info = `Model: ${customEngine}`;

// Test 22: Array of generic model names
const models = ["model-a", "model-b", "model-c"];
const supportedModels = ["custom", "internal", "proprietary"];

// Test 23: Enum definitions
enum ModelType {
    Custom = "custom-model",
    Internal = "internal-model",
    External = "external-model"
}

// Test 24: Class properties
class ModelManager {
    private modelType: string = "custom";
    private modelConfig: any = {};
    public currentModel: string = "internal-model";
}

// Test 25: Async/await with non-AI models
async function loadModel() {
    const model = await fetchCustomModel("my-model");
    return model;
}
