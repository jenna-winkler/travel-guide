# BeeAI Platform Agent Template

This repository provides a template for creating a Python agent that can be used with the [BeeAI Platform](https://docs.beeai.dev).

## Overview

BeeAI agents are Python-based services that can be run locally or deployed to the BeeAI Platform. Each agent implements specific functionality through the ACP (Agent Communication Protocol) SDK.

In this template, you'll find:
- A basic agent implementation
- Docker configuration for deployment
- Project structure for building your own agents

## Project Structure

```sh
├── src/
│   └── beeai_agents/
│       ├── __init__.py
│       └── agent.py    # Main agent implementation
├── Dockerfile          # For containerized deployment
├── pyproject.toml      # Project configuration and dependencies
├── uv.lock             # Dependency lock file
└── README.md           # This documentation
```

## Requirements

- [BeeAI Platform](https://docs.beeai.dev/introduction/quickstart) installed
- Python 3.11 or higher
- [UV package manager](https://docs.astral.sh/uv/) for dependency management

## Getting Started

1. **Set up your project**. Start by using this template for your own agent. You may use this as a template or fork this repository.

2. **Install dependencies** using `uv sync`.

3. **Implement your agent** by modifying the source code located in [src/beeai_agents/server.py](src/beeai_agents/agent.py).

Here's an example of the included template agent:

```py
@server.agent(
    metadata=Metadata(ui={"type": "hands-off"})
)
async def example_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Polite agent that greets the user"""
    hello_template: str = os.getenv("HELLO_TEMPLATE", "Ciao %s!")
    yield MessagePart(content=hello_template % str(input[-1]))
```

Modify this file to implement your own agent's logic. Here are some tips for creating your agent:
- Function name becomes the agent name and is used as the unique identifier for the agent in the BeeAI Platform
- Docstring is used as the agent's description in the platform UI
- The `@server.agent()` decorator registers your function as an agent
- The `metadata` parameter can define UI behavior and other agent properties
- Your agent receives messages in the `input` list
- Use `yield` to return responses
- Access conversation context through the `context` parameter

> [!TIP]
> You can define multiple agents in the same service by creating additional decorated functions.

## Running Agents Locally

To test your agent locally:

```sh
# Start the agent server
uv run server
```

This will start a local server on http://127.0.0.1:8000 by default. You'll get an output similar to:

```
INFO:     Started server process [86448]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Your agents should now be started on http://localhost:8000. You can verify your agents are running with the BeeAI CLI:

```sh
# List available agents
beeai list

# Run the example agent
beeai run example_agent "Your Name"
```

## Adding Agents to BeeAI Platform

There are two ways to add your agent to the BeeAI Platform:

### Local Development Mode

When running agents locally with `uv run server`, they are automatically registered with the BeeAI Platform. In this mode:
- The BeeAI Platform will communicate with your local server
- You manage the agent's lifecycle (starting/stopping)
- Changes are immediately available without redeployment

### Deployment from GitHub

To share your agent with others or deploy it to the BeeAI Platform:
1. Create an `agent.yaml` manifest in your repository to specify agent metadata. This file defines how your agent appears and behaves in the BeeAI Platform.
2. Add the agent to BeeAI using: `beeai add https://github.com/your-username/your-repo-name`

#### Version Management

- For stable versions, use Git tags (e.g., `agents-v0.0.1`)
- When adding a tagged version: `beeai add https://github.com/your-username/your-repo-name@agents-v0.0.1`
- To update: remove the old version (`beeai remove <agent-name>`) and add the new one

## Troubleshooting

### Agent Status

To check the status of your agents:

```sh
# List all agents and their status
beeai list
```

### Logs

- **Local agents:** View logs directly in the terminal where you ran `uv run server`
- **Managed agents:** Use `beeai logs <agent-id>` to view logs
- **BeeAI server logs:** Check `/opt/homebrew/var/log/beeai-server.log` (default location)
