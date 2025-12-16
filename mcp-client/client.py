import asyncio
import os
import sys
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY environment variable required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not os.path.exists(server_script_path):
             print(f"Error: Server script not found at: {server_script_path}")
             print("Please check the path and try again.")
             sys.exit(1)

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
        # Convert MCP tools to Gemini tools format
        gemini_tools = [self._convert_mcp_tool_to_gemini(tool) for tool in tools]
        
        # Initialize chat with tools
        # We need to recreate the model/chat with tools enabled
        self.model = genai.GenerativeModel('gemini-1.5-flash', tools=gemini_tools)
        self.chat = self.model.start_chat(enable_automatic_function_calling=False)


    def _convert_mcp_tool_to_gemini(self, tool: Any) -> Any:
        """Convert MCP tool to Gemini function declaration"""
        
        def sanitize_schema(schema: dict) -> dict:
            """Recursively remove fields that Gemini's protobuf handling doesn't like and fix types."""
            if not isinstance(schema, dict):
                return schema
            
            # Create a copy to avoid modifying original
            new_schema = schema.copy()
            
            # Fields to remove known to cause issues in Gemini's strict protobuf validation
            for field in ['title', 'default', 'additionalProperties']:
                if field in new_schema:
                    del new_schema[field]
            
            # Fix type to be uppercase for Gemini (string -> STRING)
            if 'type' in new_schema:
                if isinstance(new_schema['type'], str):
                     new_schema['type'] = new_schema['type'].upper()
            
            # Recursively handle properties/items
            if 'properties' in new_schema:
                new_schema['properties'] = {
                    k: sanitize_schema(v) 
                    for k, v in new_schema['properties'].items()
                }
            if 'items' in new_schema:
                new_schema['items'] = sanitize_schema(new_schema['items'])
                
            return new_schema

        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': sanitize_schema(tool.inputSchema)
        }

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""
        if not self.chat:
            return "Error: Chat session not initialized"

        # Send message to Gemini
        response = self.chat.send_message(query)
        
        final_text = []

        # Loop to handle tool calls
        while True:
            # Check if there are function calls
            part = response.parts[0] # Simplification, assume first part
            
            if part.function_call:
                fc = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args)
                
                print(f"\n[Calling tool {tool_name} with args {tool_args}]")
                
                # Execute tool
                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_output = result.content[0].text
                except Exception as e:
                    tool_output = f"Error executing tool: {str(e)}"

                # Send result back to Gemini
                # We need to construct a proper response part for the function
                response = self.chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={'result': tool_output}
                            )
                        )]
                    )
                )
            else:
                # No function call, just text
                final_text.append(part.text)
                break
                
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())