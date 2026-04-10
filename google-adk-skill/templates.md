# ADK Python Agent Templates

Quick-copy templates for common agent patterns. Replace placeholders with real logic.

---

## Minimal Single Agent

```python
# my_agent/__init__.py
from . import agent

# my_agent/agent.py
from google.adk.agents import Agent

def my_tool(param: str) -> dict:
    """Does something useful with the given parameter.

    Args:
        param (str): Description of the parameter.

    Returns:
        dict: Result with status and data.
    """
    return {"status": "success", "result": f"Processed: {param}"}

root_agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    description="A helpful agent that processes requests.",
    instruction="You are a helpful assistant. Use the my_tool tool when the user asks you to process something.",
    tools=[my_tool],
)
```

```env
# my_agent/.env
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=False
```

---

## Multi-Tool Agent

```python
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city.

    Returns:
        dict: Weather report with status.
    """
    # Replace with real API call
    return {"status": "success", "report": f"Sunny, 25°C in {city}"}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city.

    Returns:
        dict: Current time with status.
    """
    # Replace with real timezone lookup
    now = datetime.datetime.now()
    return {"status": "success", "report": f"Current time in {city}: {now.strftime('%H:%M')}"}

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.5-flash",
    description="Agent to answer questions about the time and weather in a city.",
    instruction="You are a helpful agent who can answer user questions about the time and weather in a city.",
    tools=[get_weather, get_current_time],
)
```

---

## Agent Team with Delegation

```python
from google.adk.agents import Agent

# Specialist agents
greeting_agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    description="Handles greetings and hellos from users.",
    instruction="You are a friendly greeter. Say hello warmly and ask how you can help.",
)

farewell_agent = Agent(
    name="farewell_agent",
    model="gemini-2.5-flash",
    description="Handles goodbyes and farewells from users.",
    instruction="You are a polite farewell agent. Say goodbye warmly.",
)

def do_task(task: str) -> dict:
    """Performs the specified task.

    Args:
        task (str): Description of the task to perform.

    Returns:
        dict: Task result.
    """
    return {"status": "success", "result": f"Completed: {task}"}

# Root coordinator
root_agent = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    description="Main coordinator that delegates to specialist agents.",
    instruction=(
        "You are a coordinator agent. "
        "Route greetings to greeting_agent. "
        "Route farewells to farewell_agent. "
        "Handle all other tasks yourself using do_task."
    ),
    sub_agents=[greeting_agent, farewell_agent],
    tools=[do_task],
)
```

---

## Stateful Agent with Session Memory

```python
from google.adk.agents import Agent
from google.adk.tools import ToolContext

def save_preference(key: str, value: str, tool_context: ToolContext) -> dict:
    """Saves a user preference to session memory.

    Args:
        key (str): The preference name.
        value (str): The preference value.
    """
    tool_context.state[f"pref_{key}"] = value
    return {"status": "success", "message": f"Saved {key} = {value}"}

def get_preference(key: str, tool_context: ToolContext) -> dict:
    """Retrieves a saved user preference.

    Args:
        key (str): The preference name to look up.
    """
    val = tool_context.state.get(f"pref_{key}", None)
    if val:
        return {"status": "success", "value": val}
    return {"status": "not_found", "message": f"No preference saved for '{key}'."}

def dynamic_instruction(context):
    prefs = {k: v for k, v in context.state.items() if k.startswith("pref_")}
    pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items()) if prefs else "none yet"
    return f"You are a personalized assistant. Known user preferences: {pref_str}."

root_agent = Agent(
    name="stateful_agent",
    model="gemini-2.5-flash",
    description="An agent that remembers user preferences across turns.",
    instruction=dynamic_instruction,
    tools=[save_preference, get_preference],
)
```

---

## Agent with Safety Guardrails

```python
from google.adk.agents import Agent
from google.genai import types

BLOCKED_WORDS = ["hack", "exploit", "attack"]

def safety_check(callback_context, llm_request):
    """Block requests containing dangerous keywords."""
    last_msg = ""
    if llm_request.contents:
        for part in llm_request.contents[-1].parts:
            if part.text:
                last_msg += part.text.lower()

    if any(word in last_msg for word in BLOCKED_WORDS):
        return types.Content(
            role="model",
            parts=[types.Part(text="I'm sorry, I cannot help with that request.")],
        )
    return None

def my_tool(query: str) -> dict:
    """Processes the query.

    Args:
        query (str): The user query to process.
    """
    return {"status": "success", "answer": f"Answer for: {query}"}

root_agent = Agent(
    name="safe_agent",
    model="gemini-2.5-flash",
    description="A safe agent with input guardrails.",
    instruction="You are a helpful and safe assistant.",
    tools=[my_tool],
    before_model_callback=safety_check,
)
```

---

## Sequential Pipeline

```python
from google.adk.agents import Agent, SequentialAgent

step1 = Agent(
    name="data_collector",
    model="gemini-2.5-flash",
    instruction="Collect and organize the raw data from the user's request. Output a structured summary.",
    output_key="collected_data",
)

step2 = Agent(
    name="analyzer",
    model="gemini-2.5-flash",
    instruction="Analyze the data in state['collected_data']. Produce insights and recommendations.",
    output_key="analysis",
)

step3 = Agent(
    name="report_writer",
    model="gemini-2.5-flash",
    instruction="Write a polished report based on state['analysis']. Be concise and actionable.",
    output_key="final_report",
)

root_agent = SequentialAgent(
    name="data_pipeline",
    sub_agents=[step1, step2, step3],
)
```

---

## Programmatic Runner (for scripts/notebooks)

```python
import asyncio
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

root_agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    instruction="You are helpful.",
    tools=[],
)

async def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="my_app",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="my_app", user_id="user1"
    )
    msg = types.Content(role="user", parts=[types.Part(text="Hello!")])

    async for event in runner.run_async(
        user_id="user1", session_id=session.id, new_message=msg
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## LiteLLM Multi-Provider Agent

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Requires: pip install litellm
# Set env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY

def answer_question(question: str) -> dict:
    """Answers a factual question.

    Args:
        question (str): The question to answer.

    Returns:
        dict: Answer with status.
    """
    return {"status": "success", "answer": f"Answer to: {question}"}

# Agent using OpenAI GPT-4o
root_agent = Agent(
    name="openai_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="A helpful assistant powered by GPT-4o.",
    instruction="You are a helpful assistant. Answer questions clearly and concisely.",
    tools=[answer_question],
)
```

---

## Mixed-Provider Agent Team

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Requires: pip install litellm
# Set env vars: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

def run_code(code: str) -> dict:
    """Executes a code snippet and returns the result.

    Args:
        code (str): Python code to execute.
    """
    return {"status": "success", "output": "Code executed successfully."}

def write_story(topic: str) -> dict:
    """Writes a creative story about the given topic.

    Args:
        topic (str): The topic for the story.
    """
    return {"status": "success", "story": f"A story about {topic}..."}

# Coding specialist on GPT-4o
coder = Agent(
    name="coder",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Handles coding questions and code execution.",
    instruction="You are an expert programmer. Write clean, well-documented code.",
    tools=[run_code],
)

# Creative writing specialist on Claude
writer = Agent(
    name="writer",
    model=LiteLlm(model="anthropic/claude-sonnet-4-20250514"),
    description="Handles creative writing and storytelling.",
    instruction="You are a creative writer. Write engaging, imaginative content.",
    tools=[write_story],
)

# Fast router on Gemini
root_agent = Agent(
    name="router",
    model="gemini-2.5-flash",
    description="Routes requests to the appropriate specialist.",
    instruction=(
        "You are a coordinator. "
        "Route coding questions to coder. "
        "Route creative writing requests to writer. "
        "For other queries, answer directly."
    ),
    sub_agents=[coder, writer],
)
```

---

## Agent with Skills (File-Based)

```python
# my_agent/__init__.py
from . import agent

# my_agent/agent.py
import pathlib
from google.adk import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

# Load skills from file directories
weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather_skill"
)

my_skills = skill_toolset.SkillToolset(
    skills=[weather_skill]
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="skilled_agent",
    description="An agent with specialized skills.",
    instruction="You are a helpful assistant that can use skills to perform tasks.",
    tools=[my_skills],
)
```

Skill directory structure:
```
my_agent/
    agent.py
    __init__.py
    .env
    skills/
        weather_skill/
            SKILL.md           # Required — frontmatter + instructions
            references/
                API_GUIDE.md   # Detailed docs loaded on demand
            assets/
                cities.json    # Static data
```

Example `SKILL.md` for the weather skill:
```markdown
---
name: weather-skill
description: Retrieves current weather data for any city using the OpenWeather API.
---

# Weather Skill

## Instructions

1. Extract the city name from the user's request.
2. Read `references/API_GUIDE.md` for API endpoint and auth details.
3. Call the weather API for the given city.
4. Return a formatted weather report including temperature, conditions, and humidity.
```

---

## Agent with Inline Skills (Code-Defined)

```python
from google.adk import Agent
from google.adk.skills import models
from google.adk.tools import skill_toolset

# Define skill entirely in code
greeting_skill = models.Skill(
    frontmatter=models.Frontmatter(
        name="greeting-skill",
        description="A friendly greeting skill that warmly welcomes users.",
    ),
    instructions=(
        "Step 1: Read the 'references/greetings.txt' file for greeting templates. "
        "Step 2: Choose a greeting based on the time of day if mentioned. "
        "Step 3: Return a personalized greeting."
    ),
    resources=models.Resources(
        references={
            "greetings.txt": (
                "Morning: Good morning! Hope you have a wonderful day!\n"
                "Afternoon: Good afternoon! How can I help you today?\n"
                "Evening: Good evening! What can I do for you?\n"
                "Default: Hello there! Welcome!"
            ),
        },
    ),
)

faq_skill = models.Skill(
    frontmatter=models.Frontmatter(
        name="faq-skill",
        description="Answers frequently asked questions about the product.",
    ),
    instructions="Look up the answer in references/faq.md and respond concisely.",
    resources=models.Resources(
        references={
            "faq.md": (
                "## Q: What are your business hours?\n"
                "A: We're open Monday-Friday, 9 AM to 5 PM EST.\n\n"
                "## Q: How do I reset my password?\n"
                "A: Click 'Forgot Password' on the login page.\n\n"
                "## Q: What is the return policy?\n"
                "A: 30-day returns with receipt.\n"
            ),
        },
    ),
)

my_skills = skill_toolset.SkillToolset(
    skills=[greeting_skill, faq_skill]
)

def escalate_to_human(reason: str) -> dict:
    """Escalates the conversation to a human agent.

    Args:
        reason (str): Why the escalation is needed.
    """
    return {"status": "success", "message": f"Escalated: {reason}"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="support_agent",
    description="A customer support agent with greeting and FAQ skills.",
    instruction=(
        "You are a customer support agent. Use your skills for greetings and FAQs. "
        "If you cannot answer, escalate to a human."
    ),
    tools=[my_skills, escalate_to_human],
)
```
