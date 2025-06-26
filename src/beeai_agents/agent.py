import os
from collections.abc import AsyncGenerator


from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.platform import PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

server = Server()

@server.agent(
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(ui_type=PlatformUIType.HANDSOFF)
        )
    )
)
async def example_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Polite agent that greets the user"""
    hello_template: str = os.getenv("HELLO_TEMPLATE", "Ciao %s!")
    yield MessagePart(content=hello_template % str(input[-1]))


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
