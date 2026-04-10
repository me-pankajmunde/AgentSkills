---
name: adk-agent-builder
description: Build AI agents using Google's Agent Development Kit (ADK). Use this skill whenever the user asks to create, build, or scaffold an ADK agent, multi-agent system, or agent team. Also trigger when the user mentions 'google adk', 'agent development kit', 'adk agent', 'adk project', 'adk tool', 'adk workflow', multi-tool agents, agent delegation, agent teams, sequential/parallel/loop agents, or wants to build agentic applications with ADK in Python, TypeScript, Go, or Java. Even if the user just says 'build me an agent' or 'create a bot that does X' and the context suggests ADK, use this skill. Covers project scaffolding, tool creation, multi-agent orchestration, callbacks/guardrails, session state, deployment, MCP integration, A2A protocol, and streaming agents.
---

# Google ADK Agent Builder Skill

Build production-quality AI agents using Google's Agent Development Kit (ADK). This skill covers the full lifecycle: project setup, agent definition, tool creation, multi-agent orchestration, safety guardrails, session management, and deployment.

## Quick Reference

- **Official docs**: https://adk.dev
- **GitHub repos**: `google/adk-python`, `google/adk-js`, `google/adk-go`, `google/adk-java`
- **Supported languages**: Python (primary), TypeScript, Go, Java
- **Default model**: `gemini-2.5-flash` (supports Gemini, Claude, GPT-4o via LiteLLM, Ollama, vLLM)

---

## Step 1: Determine Language & Scope

Before writing code, clarify with the user (or infer from context):

1. **Language** — Python is the most mature and documented. TypeScript, Go, and Java are also fully supported.
2. **Agent complexity** — Single agent with tools, multi-agent team, or workflow agent (sequential/parallel/loop).
3. **Model** — Default to `gemini-2.5-flash`. For multi-model, use LiteLLM adapter.
4. **Deployment target** — Local dev (`adk web`/`adk run`), Cloud Run, GKE, or Agent Engine.

---

## Step 2: Project Scaffolding

### Python (Recommended)

```
# Install
pip install google-adk

# Scaffold a new project
adk create my_agent
```

This creates:
```
my_agent/
    __init__.py    # Must contain: from . import agent
    agent.py       # Main agent code — must define `root_agent`
    .env           # API keys (GOOGLE_API_KEY, etc.)
```

**Critical rule**: The `agent.py` file MUST export a variable named `root_agent`. This is how ADK discovers the entry point.

### TypeScript

```bash
mkdir my-adk-agent && cd my-adk-agent
npm init -y
npm install @google/adk @google/adk-devtools
npm install -D typescript
```

Create `tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "es2020",
    "module": "nodenext",
    "moduleResolution": "nodenext",
    "esModuleInterop": true,
    "strict": true,
    "skipLibCheck": true,
    "verbatimModuleSyntax": false
  }
}
```

The main file is `agent.ts` and must export `rootAgent`.

### Go

```bash
mkdir my-adk-agent && cd my-adk-agent
go mod init example.com/my-agent
go get google.golang.org/adk
```

### Java

Use Maven or Gradle. The entry point class must define `public static final BaseAgent ROOT_AGENT`.

---

## Step 3: Define the Agent

### Core Agent Constructor (Python)

```python
from google.adk.agents import Agent

root_agent = Agent(
    name="my_agent",                    # Required. Unique identifier. No spaces.
    model="gemini-2.5-flash",           # Required. LLM model string.
    description="Short description.",   # Recommended. Used by parent agents for routing.
    instruction="You are a helpful assistant that...",  # The system prompt.
    tools=[tool_fn_1, tool_fn_2],       # List of tool functions or Tool objects.
    sub_agents=[agent_a, agent_b],      # Optional. For multi-agent delegation.
    output_key="my_output",             # Optional. Saves output to session state.
    generate_content_config=config,     # Optional. Temperature, top_p, etc.
    before_model_callback=fn,           # Optional. Guardrail before LLM call.
    after_model_callback=fn,            # Optional. Post-process LLM response.
    before_tool_callback=fn,            # Optional. Guardrail before tool execution.
    after_tool_callback=fn,             # Optional. Post-process tool results.
)
```

### Core Agent Constructor (TypeScript)

```typescript
import { LlmAgent, FunctionTool } from '@google/adk';
import { z } from 'zod';

export const rootAgent = new LlmAgent({
    name: 'my_agent',
    model: 'gemini-2.5-flash',
    description: 'Short description.',
    instruction: 'You are a helpful assistant that...',
    tools: [myTool],
});
```

### Key Parameters Explained

| Parameter | Purpose |
|-----------|---------|
| `name` | Unique ID. Used for routing in multi-agent systems. Avoid `user` as a name. |
| `model` | LLM model string. See model section below. |
| `description` | Used by parent/peer agents to decide whether to delegate to this agent. |
| `instruction` | System prompt. Can be a string or a callable `fn(context) -> str` for dynamic instructions. |
| `tools` | List of Python functions (auto-wrapped as FunctionTool) or Tool objects. |
| `sub_agents` | Child agents for delegation. Parent uses their `description` to decide routing. |
| `output_key` | If set, agent's final text response is saved to `state[output_key]`. |

---

## Step 4: Create Tools

Tools are how agents interact with the world. In Python, any function with type hints and a docstring becomes a tool.

### Python Function Tool Pattern

```python
def search_database(query: str, max_results: int = 5) -> dict:
    """Searches the internal database for records matching the query.

    Args:
        query (str): The search query string.
        max_results (int, optional): Maximum number of results. Defaults to 5.

    Returns:
        dict: A dictionary with 'status' and 'results' keys.
    """
    # Implementation here
    results = do_search(query, max_results)
    return {"status": "success", "results": results}
```

**Critical rules for tools:**
1. **Type hints are mandatory** on all parameters. The LLM schema is generated from them.
2. **Docstrings are mandatory**. The LLM reads the docstring to understand when/how to use the tool.
3. **Return a dict** with a `status` field and descriptive keys. The LLM reads the return value.
4. **Args docstring section** must describe each parameter — this becomes the parameter description in the schema.
5. **Optional params** must have a default value (e.g., `max_results: int = 5`).
6. **No `*args` or `**kwargs`** — these cannot be expressed in the LLM tool schema.

### TypeScript Function Tool Pattern

```typescript
const myTool = new FunctionTool({
    name: 'search_database',
    description: 'Searches the internal database for records.',
    parameters: z.object({
        query: z.string().describe('The search query string.'),
        max_results: z.number().optional().describe('Max results. Default 5.'),
    }),
    execute: ({ query, max_results = 5 }) => {
        return { status: 'success', results: [] };
    },
});
```

### Using ToolContext (Advanced)

Tools can access session state and other context via a special `tool_context` parameter:

```python
from google.adk.tools import ToolContext

def save_preference(preference: str, tool_context: ToolContext) -> dict:
    """Saves a user preference to session state.

    Args:
        preference (str): The preference to save.
    """
    tool_context.state["user_preference"] = preference
    return {"status": "success", "message": f"Saved preference: {preference}"}
```

The `tool_context` parameter is automatically injected — do NOT include it in the docstring Args.

---

## Step 5: Multi-Agent Systems

ADK supports three patterns for combining agents:

### Pattern 1: Agent Delegation (Auto-Flow)

A root agent with sub-agents. The LLM automatically routes to the best sub-agent based on their `description`.

```python
greeting_agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    description="Handles greetings and hellos from users.",
    instruction="You are a friendly greeter. Say hello warmly.",
)

farewell_agent = Agent(
    name="farewell_agent",
    model="gemini-2.5-flash",
    description="Handles goodbyes and farewells from users.",
    instruction="You are a polite farewell agent. Say goodbye warmly.",
)

root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="Main coordinator that delegates to specialists.",
    instruction="You are a coordinator. Route greetings to greeting_agent and farewells to farewell_agent. Handle other queries yourself.",
    sub_agents=[greeting_agent, farewell_agent],
    tools=[get_weather],
)
```

### Pattern 2: Sequential Agents

Execute agents in order. Output of one flows to the next via session state.

```python
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[research_agent, writer_agent, reviewer_agent],
)
```

### Pattern 3: Parallel Agents

Execute agents concurrently.

```python
from google.adk.agents import ParallelAgent

parallel = ParallelAgent(
    name="parallel_research",
    sub_agents=[web_agent, db_agent, api_agent],
)
```

### Pattern 4: Loop Agents

Repeat a sequence until a condition is met.

```python
from google.adk.agents import LoopAgent

loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[draft_agent, critique_agent],
    max_iterations=3,
)
```

---

## Step 6: Callbacks & Guardrails

Callbacks intercept agent execution at key points for safety, logging, or modification.

### before_model_callback (Input Guardrail)

```python
from google.adk.agents import Agent
from google.genai import types

def block_profanity(callback_context, llm_request):
    """Check user input for profanity before sending to LLM."""
    last_user_msg = ""
    if llm_request.contents:
        for part in llm_request.contents[-1].parts:
            if part.text:
                last_user_msg += part.text

    banned = ["badword1", "badword2"]
    if any(word in last_user_msg.lower() for word in banned):
        return types.Content(
            role="model",
            parts=[types.Part(text="I cannot process that request.")],
        )
    return None  # None means "proceed normally"

root_agent = Agent(
    name="safe_agent",
    model="gemini-2.5-flash",
    instruction="You are helpful.",
    before_model_callback=block_profanity,
    tools=[my_tool],
)
```

### before_tool_callback (Tool Argument Guardrail)

```python
def validate_city(tool, args, tool_context):
    """Block tool calls for unsupported cities."""
    city = args.get("city", "")
    supported = ["new york", "london", "tokyo"]
    if city.lower() not in supported:
        return {"status": "error", "message": f"City '{city}' is not supported."}
    return None  # Proceed with tool call

root_agent = Agent(
    name="validated_agent",
    model="gemini-2.5-flash",
    instruction="You answer weather questions.",
    tools=[get_weather],
    before_tool_callback=validate_city,
)
```

---

## Step 7: Session State & Memory

### Reading/Writing State in Tools

```python
def check_history(city: str, tool_context: ToolContext) -> dict:
    """Checks weather and remembers the last city checked."""
    tool_context.state["last_city"] = city
    # Read previous state
    prev = tool_context.state.get("last_city", "none")
    return {"status": "success", "current": city, "previous": prev}
```

### Dynamic Instructions with State

```python
def dynamic_instruction(context):
    last = context.state.get("last_city", "none")
    return f"You are a weather bot. The user last asked about: {last}. Offer to check that city again."

root_agent = Agent(
    name="stateful_agent",
    model="gemini-2.5-flash",
    instruction=dynamic_instruction,
    tools=[check_history],
)
```

### Running with Session Service (Programmatic)

```python
import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="my_app", session_service=session_service)

    session = await session_service.create_session(app_name="my_app", user_id="user1")

    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in NYC?")])

    async for event in runner.run_async(user_id="user1", session_id=session.id, new_message=user_msg):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## Step 8: Model Configuration & LiteLLM Provider Integration

ADK natively supports Gemini models. For **100+ other LLMs** (OpenAI, Anthropic, Cohere, Mistral, local models, etc.), use the **LiteLLM** connector. LiteLLM acts as a translation layer providing a standardized OpenAI-compatible interface to all providers.

Reference: https://adk.dev/agents/models/litellm/

### Default (Gemini — no extra dependencies)

```python
# .env file
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=False

# In agent.py
root_agent = Agent(model="gemini-2.5-flash", ...)
```

### Available Gemini Models

- `gemini-2.5-flash` — Fast, cost-effective (recommended default)
- `gemini-2.5-pro` — Most capable
- `gemini-2.0-flash` — Previous generation, still supported

### LiteLLM Setup (for non-Gemini providers)

```bash
pip install litellm
```

Set provider API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Anthropic (non-Vertex AI)
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# Cohere
export COHERE_API_KEY="YOUR_COHERE_API_KEY"

# Mistral
export MISTRAL_API_KEY="YOUR_MISTRAL_API_KEY"

# See all providers: https://docs.litellm.ai/docs/providers
```

### LiteLLM Usage — Remote Providers

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# OpenAI GPT-4o (requires OPENAI_API_KEY)
agent_openai = LlmAgent(
    model=LiteLlm(model="openai/gpt-4o"),
    name="openai_agent",
    instruction="You are a helpful assistant powered by GPT-4o.",
)

# Anthropic Claude (requires ANTHROPIC_API_KEY)
agent_claude = LlmAgent(
    model=LiteLlm(model="anthropic/claude-sonnet-4-20250514"),
    name="claude_agent",
    instruction="You are an assistant powered by Claude.",
)

# Cohere Command R+ (requires COHERE_API_KEY)
agent_cohere = LlmAgent(
    model=LiteLlm(model="cohere/command-r-plus"),
    name="cohere_agent",
    instruction="You are an assistant powered by Cohere.",
)

# Mistral Large (requires MISTRAL_API_KEY)
agent_mistral = LlmAgent(
    model=LiteLlm(model="mistral/mistral-large-latest"),
    name="mistral_agent",
    instruction="You are an assistant powered by Mistral.",
)
```

### LiteLLM Usage — Local / Self-Hosted Models

For locally hosted models (Ollama, vLLM, etc.), point LiteLLM to your local server:

```python
# Ollama (running locally at localhost:11434)
agent_local = LlmAgent(
    model=LiteLlm(model="ollama/llama3"),
    name="local_agent",
    instruction="You are a helpful local assistant.",
)

# vLLM (self-hosted OpenAI-compatible server)
agent_vllm = LlmAgent(
    model=LiteLlm(model="openai/my-model", api_base="http://localhost:8000/v1"),
    name="vllm_agent",
    instruction="You are a self-hosted assistant.",
)
```

For Ollama setup details, see: https://adk.dev/agents/models/ollama/
For vLLM setup details, see: https://adk.dev/agents/models/vllm/

### Common LiteLLM Model Strings

| Provider | Model String | Env Variable |
|----------|-------------|--------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4.1`, `openai/gpt-4.1-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-opus-4-20250514` | `ANTHROPIC_API_KEY` |
| Cohere | `cohere/command-r-plus`, `cohere/command-r` | `COHERE_API_KEY` |
| Mistral | `mistral/mistral-large-latest`, `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| Ollama | `ollama/llama3`, `ollama/mistral`, `ollama/codellama` | N/A (local) |
| Together AI | `together_ai/meta-llama/Llama-3-70b` | `TOGETHERAI_API_KEY` |
| Groq | `groq/llama3-70b-8192` | `GROQ_API_KEY` |
| Deepseek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |

Full provider list: https://docs.litellm.ai/docs/providers

### Multi-Model Agent Team (Mixed Providers)

You can use different models for different agents in the same team:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Fast triage with Gemini
router = Agent(
    name="router",
    model="gemini-2.5-flash",
    description="Routes requests to specialist agents.",
    instruction="Route coding questions to coder, creative writing to writer.",
    sub_agents=[coder, writer],
)

# Code specialist with GPT-4o
coder = Agent(
    name="coder",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Handles coding and technical questions.",
    instruction="You are an expert programmer.",
    tools=[run_code],
)

# Creative specialist with Claude
writer = Agent(
    name="writer",
    model=LiteLlm(model="anthropic/claude-sonnet-4-20250514"),
    description="Handles creative writing tasks.",
    instruction="You are a creative writer.",
)
```

### Windows Note for LiteLLM

On Windows you may encounter `UnicodeDecodeError`. Fix it by setting:

```powershell
$env:PYTHONUTF8 = "1"
# Or persistently:
[System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', [System.EnvironmentVariableTarget]::User)
```

---

## Step 9: Running the Agent

### CLI (Development)

```bash
# Interactive terminal
adk run my_agent

# Web UI (dev only, not for production)
adk web --port 8000
```

Run `adk web` from the **parent directory** containing your `my_agent/` folder.

### API Server

```bash
adk api_server --port 8000
```

### Programmatic (see Step 7)

Use `Runner` + `InMemorySessionService` for full control.

---

## Step 10: Advanced Features

### MCP Tools Integration

```python
from google.adk.tools.mcp_tool import MCPToolset, SseServerParams

tools, cleanup = await MCPToolset.from_server(
    connection_params=SseServerParams(url="https://mcp.example.com/sse")
)

root_agent = Agent(
    name="mcp_agent",
    model="gemini-2.5-flash",
    instruction="Use MCP tools to accomplish tasks.",
    tools=tools,
)
# Call cleanup() when done
```

### OpenAPI Tools

```python
from google.adk.tools.openapi_tool import OpenAPIToolset

toolset = OpenAPIToolset(spec_str=open("openapi.yaml").read())
root_agent = Agent(
    name="api_agent",
    model="gemini-2.5-flash",
    tools=toolset.get_tools(),
)
```

### Agent-as-a-Tool

Use an agent as a tool for another agent (gets a single response, doesn't transfer control):

```python
from google.adk.tools import agent_tool

researcher = Agent(name="researcher", model="gemini-2.5-flash", instruction="Research topics thoroughly.")

main_agent = Agent(
    name="main",
    model="gemini-2.5-flash",
    tools=[agent_tool.AgentTool(agent=researcher)],
)
```

### Deployment

- **Cloud Run**: `adk deploy cloud_run --project=PROJECT --region=REGION my_agent`
- **GKE**: Containerize with provided Dockerfile
- **Agent Engine**: `adk deploy agent_engine --project=PROJECT --region=REGION my_agent`

---

## Step 11: Skills for ADK Agents (Experimental)

Skills are self-contained, modular packages of instructions and resources that an agent can load on demand. They help organize agent capabilities and optimize the context window by only loading instructions when needed.

Reference: https://adk.dev/skills/
Specification: https://agentskills.io/specification

**Requires**: `google-adk >= 1.25.0` (Python only, experimental)

### Skills Architecture — Three Levels

Skills use progressive loading to minimize context window usage:

- **L1 (Metadata)**: Name + description in `SKILL.md` frontmatter. Always in context for discovery.
- **L2 (Instructions)**: Body of `SKILL.md`. Loaded when the skill is triggered.
- **L3 (Resources)**: Additional files in `references/`, `assets/`, `scripts/` directories. Loaded on demand.

### Skill Directory Structure

```
my_agent/
    agent.py
    .env
    skills/
        weather_skill/           # A skill
            SKILL.md             # Main instructions (required)
            references/
                API_GUIDE.md     # Detailed API reference
                FORMS.md         # Form-filling guide
            assets/
                template.json    # Templates, images, data files
            scripts/
                helper.py        # Utility scripts (execution not yet supported)
```

### SKILL.md Format

The `SKILL.md` file has YAML frontmatter for metadata, followed by markdown instructions:

```markdown
---
name: weather-skill
description: Retrieves real-time weather data for any city worldwide using the OpenWeather API.
---

# Weather Skill

## Instructions

1. When the user asks about weather, extract the city name.
2. Read the `references/API_GUIDE.md` file for API endpoint details.
3. Call the weather API with the extracted city.
4. Format and return the weather report.

## Error Handling

If the city is not found, ask the user to clarify the city name.
```

### Loading Skills from Files

```python
import pathlib
from google.adk import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

# Load skill from directory
weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather_skill"
)

# Create a SkillToolset and add to agent
my_skill_toolset = skill_toolset.SkillToolset(
    skills=[weather_skill]
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="skill_user_agent",
    description="An agent that can use specialized skills.",
    instruction=(
        "You are a helpful assistant that can leverage skills to perform tasks."
    ),
    tools=[
        my_skill_toolset,
    ],
)
```

### Defining Skills Inline (in Code)

You can also define skills programmatically for dynamic modification:

```python
from google.adk.skills import models

greeting_skill = models.Skill(
    frontmatter=models.Frontmatter(
        name="greeting-skill",
        description=(
            "A friendly greeting skill that can say hello to a specific person."
        ),
    ),
    instructions=(
        "Step 1: Read the 'references/hello_world.txt' file to understand how"
        " to greet the user. Step 2: Return a greeting based on the reference."
    ),
    resources=models.Resources(
        references={
            "hello_world.txt": "Hello! So glad to have you here!",
            "example.md": "This is an example reference.",
        },
    ),
)

# Use inline skill in an agent
my_skill_toolset = skill_toolset.SkillToolset(skills=[greeting_skill])

root_agent = Agent(
    model="gemini-2.5-flash",
    name="greeter",
    instruction="You greet users warmly.",
    tools=[my_skill_toolset],
)
```

### Combining Multiple Skills

```python
import pathlib
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather_skill"
)
calendar_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "calendar_skill"
)
email_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "email_skill"
)

combined_toolset = skill_toolset.SkillToolset(
    skills=[weather_skill, calendar_skill, email_skill]
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="assistant",
    instruction="You are a versatile assistant with weather, calendar, and email capabilities.",
    tools=[combined_toolset],
)
```

### Skills + Regular Tools Together

Skills work alongside regular function tools:

```python
def calculate(expression: str) -> dict:
    """Evaluates a mathematical expression.

    Args:
        expression (str): The math expression to evaluate.
    """
    try:
        result = eval(expression)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="hybrid_agent",
    instruction="You can perform calculations and use specialized skills.",
    tools=[
        calculate,                # Regular function tool
        my_skill_toolset,         # Skill-based tools
    ],
)
```

### Skills Known Limitations

- **Script execution** (`scripts/` directory) is not yet supported.
- **Python only** — Skills are only available in `google-adk` Python SDK v1.25.0+.
- Feature is **experimental** — API may change.

### Skills Best Practices

1. **Keep SKILL.md focused** — Put detailed reference material in `references/` files, not in the main SKILL.md.
2. **Write clear descriptions** — The frontmatter `description` determines when the agent triggers the skill. Be specific.
3. **Use references for large content** — Large API docs, schemas, or guides should go in `references/*.md` and be loaded on demand.
4. **Assets for templates** — Put reusable templates, schemas, and static data in `assets/`.
5. **One skill per concern** — Each skill should handle one clear domain or capability.

Code sample: https://github.com/google/adk-python/tree/main/contributing/samples/skills_agent

---

## Common Patterns & Templates

### Pattern: CRUD Agent

```python
def create_item(name: str, description: str) -> dict:
    """Creates a new item."""
    return {"status": "success", "id": "item_123", "name": name}

def read_item(item_id: str) -> dict:
    """Reads an item by ID."""
    return {"status": "success", "item": {"id": item_id, "name": "Example"}}

def update_item(item_id: str, name: str = None, description: str = None) -> dict:
    """Updates an existing item."""
    return {"status": "success", "updated": item_id}

def delete_item(item_id: str) -> dict:
    """Deletes an item by ID."""
    return {"status": "success", "deleted": item_id}

root_agent = Agent(
    name="crud_agent",
    model="gemini-2.5-flash",
    description="Manages items with create, read, update, delete operations.",
    instruction="You help users manage items. Use the appropriate tool for each operation.",
    tools=[create_item, read_item, update_item, delete_item],
)
```

### Pattern: Research Pipeline

```python
researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the given topic. Save findings to state.",
    output_key="research_findings",
    tools=[web_search],
)

writer = Agent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a report based on the research findings in state['research_findings'].",
    output_key="draft_report",
)

reviewer = Agent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction="Review the draft in state['draft_report']. Provide feedback.",
    output_key="final_report",
)

pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, writer, reviewer],
)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `root_agent` not found | Ensure `agent.py` defines a variable literally named `root_agent`. |
| Tool not called by LLM | Improve the tool's docstring — be specific about when to use it. |
| `adk web` can't find agent | Run from the **parent** directory of your agent folder. |
| Import error on `google.adk` | Run `pip install google-adk` (not `pip install adk`). |
| Sub-agent never activated | Make the sub-agent's `description` more specific and distinctive. |
| State not persisting | Use `InMemorySessionService` or a persistent session service. |
| LiteLLM model not working | Ensure the correct env var is set (e.g., `OPENAI_API_KEY`). Check model string format: `provider/model-name`. |
| LiteLLM `UnicodeDecodeError` (Windows) | Set `PYTHONUTF8=1` environment variable. |
| Skill not triggering | Improve the skill's frontmatter `description`. Make it specific. |
| `ImportError` for skills | Ensure `google-adk >= 1.25.0`. Skills are Python-only and experimental. |
| Skill references not loading | Check file paths in SKILL.md match actual filenames in `references/`. |

---

## Reference Links

- Python quickstart: https://adk.dev/get-started/python/
- Multi-tool tutorial: https://adk.dev/tutorials/multi-tool-agent/
- Agent team tutorial: https://adk.dev/tutorials/agent-team/
- Function tools docs: https://adk.dev/tools-custom/function-tools/
- LLM agents docs: https://adk.dev/agents/llm-agents/
- Workflow agents: https://adk.dev/agents/workflow-agents/
- Callbacks: https://adk.dev/callbacks/
- Sessions & state: https://adk.dev/sessions/
- MCP tools: https://adk.dev/tools-custom/mcp-tools/
- Models overview: https://adk.dev/agents/models/
- LiteLLM integration: https://adk.dev/agents/models/litellm/
- LiteLLM providers list: https://docs.litellm.ai/docs/providers
- Ollama integration: https://adk.dev/agents/models/ollama/
- vLLM integration: https://adk.dev/agents/models/vllm/
- Skills for agents: https://adk.dev/skills/
- Agent Skills specification: https://agentskills.io/specification
- Skills code sample: https://github.com/google/adk-python/tree/main/contributing/samples/skills_agent
- Deployment: https://adk.dev/deploy/
- A2A Protocol: https://adk.dev/a2a/
- ADK 2.0 (graph workflows): https://adk.dev/2.0/
- Sample agents: https://github.com/google/adk-samples
