// Positive test cases for Promptfoo (TypeScript)
// ruleid: detect-promptfoo
import promptfoo from "promptfoo";

// ruleid: detect-promptfoo
const results = await promptfoo.evaluate({ prompts: [], providers: [] });

// ruleid: detect-promptfoo
promptfoo.assert({ assertion: "contains", value: "test" });
