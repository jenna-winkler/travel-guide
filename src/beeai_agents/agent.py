import os
import re
import uuid
from collections.abc import AsyncGenerator

from dotenv import load_dotenv

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
model = os.getenv("LLM_MODEL", "ollama:granite3.3:8b-beeai")

server = Server()

conversation_memories = {}


class TrajectoryCapture:
    """Captures trajectory steps for display"""

    def __init__(self):
        self.steps = []

    def write(self, message: str) -> int:
        self.steps.append(message.strip())
        return len(message)


class TrackedTool:
    """Base class for tool tracking"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.results = []

    def add_result(self, result):
        self.results.append(result)


class TrackedDuckDuckGoTool(DuckDuckGoSearchTool):
    """DuckDuckGo tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("DuckDuckGo", result))
        return result


class TrackedWikipediaTool(WikipediaTool):
    """Wikipedia tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("Wikipedia", result))
        return result


class TrackedOpenMeteoTool(OpenMeteoTool):
    """Weather tool with result tracking"""

    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker

    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(("OpenMeteo", result))
        return result


def get_session_id(context: Context) -> str:
    """Extract session ID from context, fallback to default if not available"""
    session_id = getattr(context, "session_id", None)
    if not session_id:
        session_id = getattr(context, "headers", {}).get("session-id", "default")
    return str(session_id)


def get_or_create_memory(session_id: str) -> UnconstrainedMemory:
    """Get existing memory for session or create new one"""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = UnconstrainedMemory()
    return conversation_memories[session_id]


def extract_citations_from_response(response_text: str) -> tuple[list[CitationMetadata], str]:
    """Extract citations from response text and return CitationMetadata objects with cleaned text"""
    citations = []
    cleaned_text = response_text
    offset = 0

    citation_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for match in re.finditer(citation_pattern, response_text):
        content = match.group(1)
        url = match.group(2)
        original_start = match.start()

        adjusted_start = original_start - offset
        adjusted_end = adjusted_start + len(content)

        title = url.split("/")[-1].replace("-", " ").title()
        if not title or title == "":
            title = content[:50] + "..." if len(content) > 50 else content

        citation = CitationMetadata(
            kind="citation",
            url=url,
            title=title,
            description=content[:100] + "..." if len(content) > 100 else content,
            start_index=adjusted_start,
            end_index=adjusted_end,
        )
        citations.append(citation)

        removed_chars = len(match.group(0)) - len(content)
        offset += removed_chars

    cleaned_text = re.sub(citation_pattern, r"\1", response_text)

    return citations, cleaned_text


@server.agent(
    name="travel_guide",
    description="Comprehensive travel guide agent that provides personalized recommendations with weather, local information, and current search results. Features dynamic citations and trajectory tracking.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I'm your Travel Guide - here to help plan trips, check weather, and recommend restaurants. Where to?",
                display_name="Travel Guide",
                tools=[
                    AgentToolInfo(
                        name="Think",
                        description="Advanced reasoning and analysis for travel planning, itinerary optimization, and recommendation personalization based on your preferences and constraints.",
                    ),
                    AgentToolInfo(
                        name="Wikipedia",
                        description="Search comprehensive information about destinations, attractions, history, culture, and local knowledge from Wikipedia's vast database.",
                    ),
                    AgentToolInfo(
                        name="Weather",
                        description="Get current weather conditions, forecasts, and climate information for any destination to help with travel planning and packing decisions.",
                    ),
                    AgentToolInfo(
                        name="DuckDuckGo",
                        description="Search for current information about restaurants, hotels, events, transportation, and real-time travel updates from across the web.",
                    ),
                ],
            )
        ),
        author={"name": "Jenna Winkler, Tomas Weiss"},
        recommended_models=["granite3.3:8b-beeai"],
        tags=["Travel", "Planning", "Research"],
        framework="BeeAI",
        programming_language="Python",
        license="Apache 2.0",
    ),
)
async def travel_guide(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Comprehensive travel guide agent that combines:
    - Dynamic citations from search results
    - Trajectory tracking for transparency
    - Multi-tool integration for comprehensive travel planning
    """

    user_message = input[-1].parts[0].content if input else "Hello"
    session_id = get_session_id(context)

    tool_tracker = TrackedTool("travel_guide")
    trajectory = TrajectoryCapture()

    session_memory = get_or_create_memory(session_id)

    yield MessagePart(
        metadata=TrajectoryMetadata(
            kind="trajectory", key=str(uuid.uuid4()), message=f"🌍 Travel Guide processing: '{user_message}'"
        )
    )

    try:
        await session_memory.add(UserMessage(user_message))

        tracked_duckduckgo = TrackedDuckDuckGoTool(tool_tracker)
        tracked_wikipedia = TrackedWikipediaTool(tool_tracker)
        tracked_weather = TrackedOpenMeteoTool(tool_tracker)

        agent = RequirementAgent(
            llm=ChatModel.from_name(model),
            memory=session_memory,
            tools=[ThinkTool(), tracked_wikipedia, tracked_weather, tracked_duckduckgo],
            requirements=[
                ConditionalRequirement(
                    ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False, max_invocations=3
                ),
                ConditionalRequirement(tracked_wikipedia, max_invocations=1, consecutive_allowed=False),
                ConditionalRequirement(tracked_weather, max_invocations=1, consecutive_allowed=False),
                ConditionalRequirement(tracked_duckduckgo, max_invocations=1, consecutive_allowed=False),
            ],
            instructions="""
            You are a comprehensive travel guide assistant.

            Your goal is to analyse user request to plan a trip.

            First, think about what information would be most helpful based on the user query.

            Then use the following tools to gather accurate information based on your analysis.

            1. Use Wikipedia for destination background, history, and general information (search once per destination).
            2. Use DuckDuckGo for current restaurant recommendations, events, hotels, and real-time information (be specific in searches).
            3. Use OpenMeteo to get current weather conditions and forecasts (search once per location)

            Return comprehensive final answer in Markdown format that first describes the destination followed by a plan for the trip.
            You need to base everything off the information you've gathered from Wikipedia and DuckDuckGo and OpenMeteo.

            Provide final answer in markdown format while being conversational, helpful and enthusiastic about travel based on accurate information gathered from the tools.

            !!!CRITICAL!!!

            In the final answer must be information that is ALWAYS based on a result of Wikipedia, DuckDuckGo or OpenMeteo, nothing else.            

            In the final answer everything that is factual and obtained from Wikipedia, DuckDuckGo or OpenMeteo should be cited in format: [Factual information](URL)

            Couple examples:
            - The experience you will encouter is full of culinary delights! [Paris is well known](https://en.wikipedia.org/wiki/Paris) for its rich history and vibrant culinary scene.
            - During your visit in following two days there are two events you should not miss: [Prague Beer Festival](https://en.wikipedia.org/wiki/Prague_Beer_Festival) and [Prague Music Festival](https://en.wikipedia.org/wiki/Prague_Music_Festival).
            - If you need a great place to stay, [Hotel Blue Ocean](https://blueocean.com/) is a great choice.
            - The weather in Oslo is expected to be sunny with a temperature of 20°C according to [OpenMeteo](https://open-meteo.com/), pack your hats & sunglasses!
            """,
        )

        yield MessagePart(
            metadata=TrajectoryMetadata(
                kind="trajectory",
                key=str(uuid.uuid4()),
                message="🛠️ Travel Guide initialized with Think, Wikipedia, Weather, and Search tools",
            )
        )

        response = await agent.run(
            user_message, execution=AgentExecutionConfig(max_iterations=10, max_retries_per_step=2, total_max_retries=5)
        ).middleware(GlobalTrajectoryMiddleware(target=trajectory, included=[Tool]))

        response_text = response.answer.text

        await session_memory.add(AssistantMessage(response_text))

        for i, step in enumerate(trajectory.steps):
            if step.strip():
                tool_name = None
                if "ThinkTool" in step:
                    tool_name = "Think"
                elif "WikipediaTool" in step:
                    tool_name = "Wikipedia"
                elif "OpenMeteoTool" in step:
                    tool_name = "Weather"
                elif "DuckDuckGo" in step:
                    tool_name = "DuckDuckGo"

                yield MessagePart(
                    metadata=TrajectoryMetadata(
                        kind="trajectory", key=str(uuid.uuid4()), message=f"Step {i + 1}: {step}", tool_name=tool_name
                    )
                )

        citations, cleaned_response_text = extract_citations_from_response(response_text)

        yield MessagePart(content=cleaned_response_text)

        for citation in citations:
            yield MessagePart(metadata=citation)

        yield MessagePart(
            metadata=TrajectoryMetadata(
                kind="trajectory",
                key=str(uuid.uuid4()),
                message="✅ Travel Guide completed successfully with citations",
            )
        )

    except Exception as e:
        yield MessagePart(
            metadata=TrajectoryMetadata(kind="trajectory", key=str(uuid.uuid4()), message=f"❌ Error: {str(e)}")
        )
        yield MessagePart(content=f"🚨 Sorry, I encountered an error while planning your trip: {str(e)}")


def run():
    """Entry point for the server."""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
