#!/usr/bin/env python3
"""
AI Agent with Autonomous Tool Selection
Main entry point
"""

import os
import asyncio
import sys
import gradio as gr
import ollama
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
import json

# Load environment variables
load_dotenv()

class OllamaAgent:
    def __init__(self):
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = os.getenv('OLLAMA_MODEL', 'qwen3:latest')
        
        # Initialize Ollama client
        self.client = ollama.Client(host=self.host)
        
        # Initialize MCP state
        self.mcp_servers = {}
        self.available_tools = {}
        self.system_prompt = ""
        self._tools_discovered = False  # Flag to track lazy loading
        
        print("🔄 Enhanced schema communication - tools will be rediscovered with detailed parameters")
        
        # Setup
        self._verify_setup()
        self._discover_mcp_tools()  # Now just registers servers, no async calls
        self._build_system_prompt()
        
    def _verify_setup(self):
        """Verify Ollama connection and model availability"""
        try:
            # Check if Ollama is running
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model not in model_names:
                print(f"Warning: Model '{self.model}' not found. Available models: {model_names}")
                if model_names:
                    self.model = model_names[0]
                    print(f"Using first available model: {self.model}")
                else:
                    raise Exception("No models available in Ollama")
            
            print(f"✅ Connected to Ollama at {self.host}")
            print(f"✅ Using model: {self.model}")
            
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("Please ensure Ollama is running and accessible")
            raise
    
    def _load_mcp_servers_config(self):
        """Load MCP server configuration from mcp_servers.json"""
        config_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            servers_to_check = []
            
            # Handle new mcpServers format (like Claude Desktop)
            mcp_servers = config.get('mcpServers', {})
            for server_name, server_config in mcp_servers.items():
                if server_config.get('enabled', True):  # Default to enabled if not specified
                    
                    # Standard MCP server format (command/args) - like Claude Desktop
                    if 'command' in server_config and 'type' not in server_config:
                        server_info = {
                            'name': server_name,
                            'type': 'standard',
                            'command': server_config['command'],
                            'args': server_config.get('args', []),
                            'description': server_config.get('description', f"MCP server: {server_name}")
                        }
                    
                    # Custom format with type key
                    elif 'type' in server_config:
                        server_type = server_config['type']
                        
                        if server_type == 'file':
                            # Local file server
                            server_info = {
                                'name': server_name,
                                'type': 'file',
                                'path': os.path.join(os.path.dirname(__file__), server_config['path']),
                                'description': server_config.get('description', f"MCP server: {server_name}")
                            }
                        elif server_type == 'package':
                            # Installed package server
                            server_info = {
                                'name': server_name,
                                'type': 'package',
                                'command': server_config.get('command', 'python'),
                                'args': server_config.get('args', []),
                                'description': server_config.get('description', f"MCP server: {server_name}")
                            }
                        else:
                            print(f"⚠️  Unknown server type '{server_type}' for {server_name}, skipping")
                            continue
                    
                    else:
                        print(f"⚠️  Invalid server configuration for {server_name}, skipping")
                        continue
                        
                    servers_to_check.append(server_info)
            
            print(f"📋 Loaded {len(servers_to_check)} MCP servers from configuration")
            return servers_to_check
            
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found: {config_path}")
            print("   Using empty server list")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            print("   Using empty server list")
            return []
        except Exception as e:
            print(f"❌ Error loading MCP servers configuration: {e}")
            print("   Using empty server list")
            return []

    def _discover_mcp_tools(self):
        """Register MCP servers (tools discovered lazily on first use)"""
        print("🔍 Registering MCP servers...")
        
        # Load MCP servers from configuration file
        servers_to_check = self._load_mcp_servers_config()
        
        for server_info in servers_to_check:
            if server_info['type'] == 'file':
                # Check if local file exists
                if os.path.exists(server_info['path']):
                    print(f"✅ Found {server_info['name']} (local file)")
                    self.mcp_servers[server_info['name']] = server_info
                else:
                    print(f"❌ Server not found: {server_info['name']} at {server_info['path']}")
            elif server_info['type'] in ['package', 'standard']:
                # For package and standard servers, we assume they're available if configured
                server_type_desc = 'standard MCP server' if server_info['type'] == 'standard' else 'installed package'
                print(f"✅ Found {server_info['name']} ({server_type_desc})")
                self.mcp_servers[server_info['name']] = server_info
        
        print(f"📝 Registered {len(self.mcp_servers)} MCP servers (tools will be discovered on first use)")

    def _discover_tools_lazy(self):
        """Discover MCP tools on first use (avoids asyncio conflicts during init)"""
        if self._tools_discovered:
            print("🔧 Tools already discovered, skipping...")
            return
            
        print("🔧 Discovering MCP tools for the first time...")
        print(f"🔧 Registered servers: {list(self.mcp_servers.keys())}")
        
        for server_name, server_info in self.mcp_servers.items():
            print(f"\n🔧 Connecting to server: {server_name}")
            if server_info['type'] == 'file':
                print(f"🔧 Server path: {server_info['path']}")
            elif server_info['type'] in ['package', 'standard']:
                print(f"🔧 Server command: {server_info['command']} {' '.join(server_info['args'])}")
            else:
                print(f"🔧 Server type: {server_info['type']}")
            
            try:
                # Use thread-based execution to avoid event loop conflicts
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        print(f"🔧 [{server_name}] Starting tool discovery in new thread...")
                        result = new_loop.run_until_complete(
                            self._get_server_tools(server_info)
                        )
                        print(f"🔧 [{server_name}] Tool discovery completed in thread")
                        return result
                    except Exception as e:
                        print(f"🔧 [{server_name}] Error in thread: {e}")
                        raise
                    finally:
                        new_loop.close()
                
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    tools = future.result(timeout=10)  # Add timeout
                    
                if tools:
                    self.available_tools[server_name] = tools
                    print(f"   📋 ✅ Discovered {len(tools)} tools in {server_name}: {[t['name'] for t in tools]}")
                    for tool in tools:
                        print(f"       - {tool['name']}: {tool['description']}")
                else:
                    print(f"   ⚠️  No tools found in {server_name}")
                    self.available_tools[server_name] = []
                    
            except Exception as e:
                print(f"   ❌ Error getting tools from {server_name}: {e}")
                print(f"      Exception type: {type(e).__name__}")
                import traceback
                print(f"      Traceback: {traceback.format_exc()}")
                self.available_tools[server_name] = []
        
        print(f"\n🔧 Tool discovery complete. Total servers: {len(self.mcp_servers)}")
        print(f"🔧 Total tools discovered: {sum(len(tools) for tools in self.available_tools.values())}")
        
        self._tools_discovered = True
        self._build_system_prompt_with_tools()  # Rebuild prompt with discovered tools

    async def _get_server_tools(self, server_info):
        """Get available tools from an MCP server"""
        try:
            if server_info['type'] == 'file':
                # Local file server
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_info['path']],
                )
            elif server_info['type'] in ['package', 'standard']:
                # Installed package or standard MCP server
                server_params = StdioServerParameters(
                    command=server_info['command'],
                    args=server_info['args'],
                )
            else:
                print(f"Unknown server type: {server_info['type']}")
                return []
            
            async with stdio_client(server_params) as streams:
                async with ClientSession(*streams) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # List available tools
                    tools_response = await session.list_tools()
                    
                    if hasattr(tools_response, 'tools'):
                        return [
                            {
                                'name': tool.name,
                                'description': tool.description or f"Tool: {tool.name}",
                                'schema': tool.inputSchema or {},
                                'server': server_info['name']
                            }
                            for tool in tools_response.tools
                        ]
                    
                    return []
                    
        except Exception as e:
            print(f"Error connecting to {server_info['name']}: {e}")
            return []

    def _build_system_prompt(self):
        """Build initial system prompt (tools added later when discovered)"""
        base_prompt = os.getenv('SYSTEM_PROMPT', 
            "You are a helpful AI assistant with access to tools. Provide clear, accurate, and concise responses.")
        
        # Initially, just use base prompt - tools will be added on first use
        self.system_prompt = base_prompt + "\n\n(MCP tools will be available after first interaction)"
        print("✅ Basic system prompt built (tools will be added on first use)")

    def _build_system_prompt_with_tools(self):
        """Build system prompt with discovered tools"""
        base_prompt = os.getenv('SYSTEM_PROMPT', 
            "You are a helpful AI assistant with access to tools. Provide clear, accurate, and concise responses.")
        
        if not self.available_tools or not any(self.available_tools.values()):
            self.system_prompt = base_prompt
            print("⚠️  No MCP tools available - running in direct response mode")
            return
        
        # Build tools section
        tools_section = "\n\n=== AVAILABLE TOOLS ===\n"
        tools_section += "You have access to the following tools. Use them when appropriate to help the user:\n\n"
        
        for server_name, tools in self.available_tools.items():
            if tools:  # Only show servers with actual tools
                tools_section += f"From {server_name}:\n"
                for tool in tools:
                    tools_section += f"- {tool['name']}: {tool['description']}\n"
                    if tool['schema'].get('properties'):
                        tools_section += f"  Parameters:\n"
                        props = tool['schema']['properties']
                        for param_name, param_info in props.items():
                            param_type = param_info.get('type', 'any')
                            param_desc = param_info.get('description', 'No description')
                            tools_section += f"    - {param_name} ({param_type}): {param_desc}\n"
                            
                            # Show enum values if available
                            if 'enum' in param_info:
                                enum_values = param_info['enum']
                                tools_section += f"      Valid values: {enum_values}\n"
                            
                            # Show default value if available
                            if 'default' in param_info:
                                default_val = param_info['default']
                                tools_section += f"      Default: {default_val}\n"
                        
                        # Show required parameters
                        required_params = tool['schema'].get('required', [])
                        if required_params:
                            tools_section += f"    Required: {required_params}\n"
                    tools_section += "\n"
                tools_section += "\n"
        
        tools_section += """=== TOOL USAGE INSTRUCTIONS ===
To use a tool, respond with JSON in this exact format:
{
  "action": "use_tool",
  "tool_name": "exact_tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  },
  "explanation": "Brief explanation of why you're using this tool"
}

MULTI-TOOL AUTONOMOUS BEHAVIOR:
- You can use multiple tools in sequence to complete complex tasks
- After each tool execution, you'll receive the result and can decide what to do next
- You can use results from previous tools to inform subsequent tool usage
- When you have all the information needed, provide your final response to the user
- Examples of multi-tool workflows:
  * "Email me the current time" → get_current_time, then send_email with the time
  * "Send me a summary email" → gather info with various tools, then send_email
  * Complex tasks may require 3+ tools in sequence

WHEN TO USE TOOLS:
- When the user asks you to DO something (send email, get time, etc.) - ALWAYS use the appropriate tool
- When you need information that tools can provide
- When performing actions that match tool capabilities
- Use your judgment to determine if multiple tools are needed for complex requests

EMAIL TOOL NOTES:
- send_email automatically sends to the configured notification recipient
- You only need to provide "subject" and "message" parameters
- DO NOT provide to_email, from_name, or recipient parameters
- Focus on creating helpful, clear subject lines and message content
- The recipient is automatically handled by the system configuration

IMPORTANT:
- The tool_name must exactly match one of the available tools listed above
- Use the tools in server_memory to store and retrieve personal information provided by the user, for example their name
- Use the read_graph tool to retrieve stored information about the user when that information is needed
- Follow parameter schemas EXACTLY - use only the valid enum values shown
- Always provide an explanation for why you're using each tool
- If a tool returns an error (like "Input validation error" or "Error:"), recognize it as a failure
- When a tool fails, DO NOT make up or guess the answer - report the error to the user
- For parameter validation errors, check the schema and try again with correct values
- Think step by step for complex requests that may need multiple tools
"""
        
        self.system_prompt = base_prompt + tools_section
        total_tools = sum(len(tools) for tools in self.available_tools.values() if tools)
        print(f"✅ System prompt rebuilt with {total_tools} available tools")

    def chat(self, message, history):
        """
        Handle chat interaction with single tool execution
        
        Args:
            message: User input message
            history: Chat history from Gradio
            
        Returns:
            str: AI response
        """
        try:
            # Discover tools on first use (lazy loading to avoid asyncio conflicts)
            if not self._tools_discovered:
                self._discover_tools_lazy()
            
            print(f"\n{'='*60}")
            print(f"🤖 [CHAT] User message: {message}")
            print(f"🤖 [CHAT] Available tools: {sum(len(tools) for tools in self.available_tools.values())}")
            print(f"{'='*60}")
            
            # Prepare conversation context
            conversation = [{"role": "system", "content": self.system_prompt}]
            
            # Add chat history
            for user_msg, assistant_msg in history:
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": assistant_msg})
            
            # Add current message
            conversation.append({"role": "user", "content": message})
            
            # Get LLM response
            print(f"🤖 [CHAT] Getting LLM response...")
            print(f"🤖 [CHAT] Using model: {self.model}")
            response = self.client.chat(
                model=self.model,
                messages=conversation,
                stream=False
            )
            
            assistant_response = response['message']['content']
            print(f"🤖 [CHAT] LLM Response: {assistant_response[:200]}{'...' if len(assistant_response) > 200 else ''}")
            
            # Multi-tool execution loop
            current_response = assistant_response
            max_tool_calls = 5  # Prevent infinite loops
            tool_call_count = 0
            
            while tool_call_count < max_tool_calls:
                # Check if current response wants to use a tool
                print(f"🤖 [CHAT] Checking for tool requests... (iteration {tool_call_count + 1})")
                tool_request = self._parse_tool_request(current_response)
                
                if not tool_request:
                    # No more tools needed, parse and format thinking content before returning
                    print(f"🤖 [CHAT] ✅ No tool needed, returning response after {tool_call_count} tool calls")
                    print(f"{'='*60}\n")
                    
                    # Parse and format thinking content
                    parsed_response = self._parse_thinking_content(current_response)
                    if parsed_response['has_thinking']:
                        print(f"🧠 [CHAT] Calling format function with thinking content: {len(parsed_response['thinking_content'])} chars")
                        formatted_result = self._format_response_with_thinking(
                            parsed_response['main_content'], 
                            parsed_response['thinking_content']
                        )
                        print(f"🧠 [CHAT] Formatted result length: {len(formatted_result)}")
                        print(f"🧠 [CHAT] Contains brain icon: {'brain-icon-corner' in formatted_result}")
                        return formatted_result
                    else:
                        print(f"🧠 [CHAT] No thinking content found, returning original response")
                        return current_response
                
                # LLM wants to use a tool
                tool_name = tool_request['tool_name']
                parameters = tool_request['parameters']
                explanation = tool_request.get('explanation', 'Using tool')
                tool_call_count += 1
                
                print(f"🤖 [CHAT] ✅ Tool requested (#{tool_call_count}): {tool_name}")
                print(f"🤖 [CHAT] Parameters: {parameters}")
                
                # Find which server has this tool
                server_name = self._find_tool_server(tool_name)
                if not server_name:
                    error_msg = f"Error: Tool '{tool_name}' not found in available tools"
                    print(f"🤖 [CHAT] ❌ {error_msg}")
                    return error_msg
                
                print(f"🤖 [CHAT] Found tool in server: {server_name}")
                
                # Execute the tool
                print(f"🤖 [CHAT] Executing tool...")
                tool_result = self._execute_tool_sync(tool_name, parameters, server_name)
                print(f"🤖 [CHAT] Tool result: {tool_result}")
                
                # Add previous response and tool result to conversation
                conversation.append({"role": "assistant", "content": current_response})
                conversation.append({"role": "system", "content": f"Tool '{tool_name}' result: {tool_result}"})
                
                # Get next response from LLM with tool result
                print(f"🤖 [CHAT] Getting next response with tool result...")
                print(f"🤖 [CHAT] Using model: {self.model}")
                print(f"🤖 [CHAT] Conversation length: {len(conversation)} messages")
                print(f"🤖 [CHAT] Last message role: {conversation[-1]['role']}")
                print(f"🤖 [CHAT] Last message preview: {conversation[-1]['content'][:200]}...")

                try:
                    next_response = self.client.chat(
                        model=self.model,
                        messages=conversation,
                        stream=False
                    )
                    
                    print(f"🤖 [CHAT] Raw response keys: {list(next_response.keys())}")
                    
                    if 'message' in next_response:
                        print(f"🤖 [CHAT] Message keys: {list(next_response['message'].keys())}")
                        current_response = next_response['message']['content']
                        print(f"🤖 [CHAT] Content length: {len(current_response)}")
                        print(f"🤖 [CHAT] Raw content: '{current_response}'")
                        print(f"🤖 [CHAT] Content preview: {current_response[:100]}...")
                        
                        # If response is empty, add a fallback
                        if not current_response or current_response.strip() == "":
                            print(f"🤖 [CHAT] ⚠️ Empty response detected, using fallback")
                            current_response = f"I successfully used the {tool_name} tool. The result was: {tool_result}"
                    else:
                        print(f"🤖 [CHAT] ❌ No 'message' key in response!")
                        current_response = f"I used the {tool_name} tool successfully. The result was: {tool_result}"
                        
                except Exception as e:
                    print(f"🤖 [CHAT] ❌ Error calling LLM: {e}")
                    current_response = f"I used the {tool_name} tool successfully. The result was: {tool_result}"
                
                # Continue loop to check if this response has more tool requests
            
            # If we hit max tool calls, return what we have
            print(f"🤖 [CHAT] ⚠️  Hit maximum tool calls ({max_tool_calls}), returning current response")
            print(f"{'='*60}\n")
            
            # Parse and format thinking content before returning
            parsed_response = self._parse_thinking_content(current_response)
            if parsed_response['has_thinking']:
                print(f"🧠 [CHAT] (Max tools) Calling format function with thinking content: {len(parsed_response['thinking_content'])} chars")
                formatted_result = self._format_response_with_thinking(
                    parsed_response['main_content'], 
                    parsed_response['thinking_content']
                )
                print(f"🧠 [CHAT] (Max tools) Formatted result length: {len(formatted_result)}")
                print(f"🧠 [CHAT] (Max tools) Contains brain icon: {'brain-icon-corner' in formatted_result}")
                return formatted_result
            else:
                print(f"🧠 [CHAT] (Max tools) No thinking content found, returning original response")
                return current_response
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(f"🤖 [CHAT] ❌ ERROR: {error_msg}")
            import traceback
            print(f"🤖 [CHAT] Traceback: {traceback.format_exc()}")
            
            # Parse and format thinking content even for errors (in case error response has thinking)
            parsed_response = self._parse_thinking_content(error_msg)
            if parsed_response['has_thinking']:
                print(f"🧠 [CHAT] (Error) Calling format function with thinking content: {len(parsed_response['thinking_content'])} chars")
                formatted_result = self._format_response_with_thinking(
                    parsed_response['main_content'], 
                    parsed_response['thinking_content']
                )
                print(f"🧠 [CHAT] (Error) Formatted result length: {len(formatted_result)}")
                print(f"🧠 [CHAT] (Error) Contains brain icon: {'brain-icon-corner' in formatted_result}")
                return formatted_result
            else:
                print(f"🧠 [CHAT] (Error) No thinking content found, returning original error")
                return error_msg

    def _execute_tool_sync(self, tool_name, parameters, server_name):
        """Synchronous wrapper for tool execution"""
        try:
            print(f"🔧 [SYNC WRAPPER] Starting tool execution wrapper")
            
            # Always create a new event loop in a separate thread to avoid conflicts
            import concurrent.futures
            
            def run_in_thread():
                print(f"🔧 [SYNC WRAPPER] Creating new event loop in thread")
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    print(f"🔧 [SYNC WRAPPER] Running async tool execution")
                    result = new_loop.run_until_complete(
                        self._execute_tool(tool_name, parameters, server_name)
                    )
                    print(f"🔧 [SYNC WRAPPER] Async execution completed")
                    return result
                except Exception as e:
                    print(f"🔧 [SYNC WRAPPER] Error in async execution: {e}")
                    raise
                finally:
                    print(f"🔧 [SYNC WRAPPER] Closing event loop")
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print(f"🔧 [SYNC WRAPPER] Submitting to thread pool")
                future = executor.submit(run_in_thread)
                result = future.result(timeout=30)  # Add timeout
                print(f"🔧 [SYNC WRAPPER] Got result from thread pool")
                return result
                
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(f"🔧 [SYNC WRAPPER] ❌ ERROR: {error_msg}")
            return error_msg

    async def _execute_tool(self, tool_name, parameters, server_name):
        """Execute a specific tool via MCP"""
        try:
            print(f"\n🔧 [TOOL EXECUTION] Starting tool execution")
            print(f"🔧 [TOOL EXECUTION] Tool: '{tool_name}'")
            print(f"🔧 [TOOL EXECUTION] Server: '{server_name}'")
            print(f"🔧 [TOOL EXECUTION] Parameters: {parameters}")
            
            if server_name not in self.mcp_servers:
                error_msg = f"Server '{server_name}' not found in registered servers: {list(self.mcp_servers.keys())}"
                print(f"🔧 [TOOL EXECUTION] ERROR: {error_msg}")
                return error_msg
            
            server_info = self.mcp_servers[server_name]
            
            if server_info['type'] == 'file':
                # Local file server
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_info['path']],
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                print(f"🔧 [TOOL EXECUTION] Server path: {server_info['path']}")
            elif server_info['type'] in ['package', 'standard']:
                # Installed package or standard MCP server
                server_params = StdioServerParameters(
                    command=server_info['command'],
                    args=server_info['args'],
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                print(f"🔧 [TOOL EXECUTION] Server command: {server_info['command']} {' '.join(server_info['args'])}")
            else:
                error_msg = f"Unknown server type: {server_info['type']}"
                print(f"🔧 [TOOL EXECUTION] ERROR: {error_msg}")
                return error_msg
            
            print(f"🔧 [TOOL EXECUTION] Starting MCP server subprocess...")
            
            async with stdio_client(server_params) as streams:
                print(f"🔧 [TOOL EXECUTION] MCP server subprocess started")
                
                async with ClientSession(*streams) as session:
                    print(f"🔧 [TOOL EXECUTION] Initializing MCP session...")
                    
                    # Initialize the session
                    await session.initialize()
                    print(f"🔧 [TOOL EXECUTION] MCP session initialized successfully")
                    
                    print(f"🔧 [TOOL EXECUTION] Calling tool '{tool_name}' via MCP...")
                    print(f"🔧 [TOOL EXECUTION] Sending parameters: {parameters}")
                    
                    # Call the tool
                    response = await session.call_tool(tool_name, parameters)
                    
                    print(f"🔧 [TOOL EXECUTION] Tool call completed - got response")
                    print(f"🔧 [TOOL EXECUTION] Response type: {type(response)}")
                    print(f"🔧 [TOOL EXECUTION] Response content: {response.content if hasattr(response, 'content') else 'No content attribute'}")
                    
                    if response.content and len(response.content) > 0:
                        result = response.content[0].text
                        print(f"🔧 [TOOL EXECUTION] ✅ SUCCESS - Tool result: {result}")
                        return result
                    else:
                        result = f"Tool {tool_name} executed but returned no content"
                        print(f"🔧 [TOOL EXECUTION] ⚠️  WARNING: {result}")
                        return result
                        
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(f"🔧 [TOOL EXECUTION] ❌ ERROR: {error_msg}")
            print(f"🔧 [TOOL EXECUTION] Exception type: {type(e).__name__}")
            import traceback
            print(f"🔧 [TOOL EXECUTION] Traceback: {traceback.format_exc()}")
            return error_msg

    def _find_tool_server(self, tool_name):
        """Find which server contains a specific tool"""
        for server_name, tools in self.available_tools.items():
            for tool in tools:
                if tool['name'] == tool_name:
                    return server_name
        return None

    def _parse_tool_request(self, response_text):
        """Parse LLM response for tool usage requests"""
        print(f"🔍 [TOOL PARSER] Parsing response for tool requests...")
        print(f"🔍 [TOOL PARSER] Response length: {len(response_text)} characters")
        
        try:
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            print(f"🔍 [TOOL PARSER] JSON search - start: {start_idx}, end: {end_idx}")
            
            if start_idx == -1 or end_idx == 0:
                print(f"🔍 [TOOL PARSER] ❌ No JSON found in response")
                return None
                
            json_str = response_text[start_idx:end_idx]
            print(f"🔍 [TOOL PARSER] Extracted JSON: {json_str}")
            
            tool_request = json.loads(json_str)
            print(f"🔍 [TOOL PARSER] Parsed JSON: {tool_request}")
            
            # Validate required fields
            if (tool_request.get('action') == 'use_tool' and 
                'tool_name' in tool_request and 
                'parameters' in tool_request):
                print(f"🔍 [TOOL PARSER] ✅ Valid tool request found: {tool_request['tool_name']}")
                return tool_request
            else:
                print(f"🔍 [TOOL PARSER] ❌ Invalid tool request format")
                print(f"🔍 [TOOL PARSER]   Action: {tool_request.get('action')}")
                print(f"🔍 [TOOL PARSER]   Has tool_name: {'tool_name' in tool_request}")
                print(f"🔍 [TOOL PARSER]   Has parameters: {'parameters' in tool_request}")
                
        except json.JSONDecodeError as e:
            print(f"🔍 [TOOL PARSER] ❌ JSON decode error: {e}")
        except KeyError as e:
            print(f"🔍 [TOOL PARSER] ❌ Key error: {e}")
        except Exception as e:
            print(f"🔍 [TOOL PARSER] ❌ Unexpected error: {e}")
            
        print(f"🔍 [TOOL PARSER] ❌ No valid tool request found")
        return None

    def _parse_thinking_content(self, response_text):
        """Parse thinking content from LLM response and separate it from main content"""
        import re
        
        print(f"🧠 [THINKING PARSER] Parsing response for thinking content...")
        
        # Look for <think>...</think> tags (case insensitive, multiline)
        thinking_pattern = r'<think>(.*?)</think>'
        matches = re.findall(thinking_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            print(f"🧠 [THINKING PARSER] Found {len(matches)} thinking blocks")
            
            # Extract all thinking content
            thinking_content = '\n\n'.join(matches).strip()
            
            # Remove thinking tags from main content
            main_content = re.sub(thinking_pattern, '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Clean up extra whitespace
            main_content = re.sub(r'\n\s*\n\s*\n', '\n\n', main_content)
            
            # Debug output
            print(f"🧠 [THINKING PARSER] Thinking content length: {len(thinking_content)}")
            print(f"🧠 [THINKING PARSER] Thinking content: '{thinking_content}'")
            print(f"🧠 [THINKING PARSER] Main content length: {len(main_content)}")
            print(f"🧠 [THINKING PARSER] Main content: '{main_content[:100]}{'...' if len(main_content) > 100 else ''}'")
            
            return {
                'has_thinking': True,
                'thinking_content': thinking_content,
                'main_content': main_content
            }
        else:
            print(f"🧠 [THINKING PARSER] No thinking content found")
            return {
                'has_thinking': False,
                'thinking_content': '',
                'main_content': response_text
            }

    def _format_response_with_thinking(self, main_content, thinking_content):
        """Format response with brain icon in top right corner and collapsible thinking content"""
        import uuid
        import html
        
        # Only show brain icon if there's actually thinking content
        if not thinking_content or not thinking_content.strip():
            # No thinking content, return just the main content
            main_html = html.escape(main_content).replace('\n', '<br>')
            return f'<div class="main-content">{main_html}</div>'
        
        # Generate unique ID for this thinking block
        thinking_id = f"thinking_{uuid.uuid4().hex[:8]}"
        
        # Escape HTML in content but preserve formatting
        main_html = html.escape(main_content).replace('\n', '<br>')
        thinking_html = html.escape(thinking_content).replace('\n', '<br>')
        
        print(f"🧠 [FORMAT] Thinking content length: {len(thinking_content)}")
        print(f"🧠 [FORMAT] Thinking preview: {thinking_content[:100]}...")
        
        # Create HTML with brain icon positioned in top right
        formatted_html = f"""
        <div class="response-container">
            <div class="response-header">
                <span class="brain-icon-corner" onclick="window.toggleThinking('{thinking_id}')" title="Click to view AI's reasoning process">🧠</span>
            </div>
            <div class="main-content">
                {main_html}
            </div>
            <div id="{thinking_id}" class="thinking-content" style="display: none;">
                <div class="thinking-text">{thinking_html}</div>
            </div>
        </div>
        """
        
        return formatted_html

    # === MCP Server Management Methods ===
    
    def add_mcp_server(self, server_name, config_json):
        """Add a new MCP server dynamically"""
        print(f"🔧 [SERVER MANAGER] Adding server: {server_name}")
        
        try:
            # Parse JSON config
            config = json.loads(config_json)
            print(f"🔧 [SERVER MANAGER] Parsed config: {config}")
            
            # Validate required fields
            if 'command' not in config or 'args' not in config:
                return {"success": False, "error": "Configuration must include 'command' and 'args' fields"}
            
            # Test server first
            print(f"🔧 [SERVER MANAGER] Testing server configuration...")
            test_result = self.test_server_config(config)
            if not test_result["success"]:
                return {"success": False, "error": f"Server test failed: {test_result['error']}"}
            
            # Create server info
            server_info = {
                'name': server_name,
                'type': 'standard',
                'command': config['command'],
                'args': config['args'],
                'description': config.get('description', f"MCP server: {server_name}"),
                'env': config.get('env', {}),
                'enabled': True
            }
            
            # Add to runtime registry
            self.mcp_servers[server_name] = server_info
            print(f"🔧 [SERVER MANAGER] Added to runtime registry")
            
            # Discover tools from new server
            try:
                tools = self._get_server_tools_sync(server_info)
                if tools:
                    self.available_tools[server_name] = tools
                    print(f"🔧 [SERVER MANAGER] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
                else:
                    print(f"🔧 [SERVER MANAGER] No tools discovered")
                    
            except Exception as e:
                print(f"🔧 [SERVER MANAGER] Warning: Could not discover tools: {e}")
                tools = []
            
            # Save to configuration file
            self._save_servers_config()
            
            # Rebuild system prompt
            self._build_system_prompt_with_tools()
            
            return {
                "success": True,
                "tools_count": len(tools),
                "tools": [t['name'] for t in tools] if tools else [],
                "message": f"✅ Server '{server_name}' added successfully with {len(tools)} tools"
            }
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            print(f"🔧 [SERVER MANAGER] Error adding server: {e}")
            return {"success": False, "error": str(e)}
    
    def test_server_config(self, config):
        """Test if a server configuration works"""
        print(f"🔧 [SERVER TEST] Testing configuration...")
        
        try:
            # Create temporary server info
            temp_server = {
                'name': 'test_server',
                'type': 'standard',
                'command': config['command'],
                'args': config['args'],
                'env': config.get('env', {}),
                'description': 'Test server'
            }
            
            # Try to get tools with timeout
            tools = self._get_server_tools_sync(temp_server, timeout=10)
            
            return {
                "success": True,
                "tools_found": len(tools),
                "message": f"✅ Test successful - found {len(tools)} tools"
            }
            
        except Exception as e:
            print(f"🔧 [SERVER TEST] Test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_server_tools_sync(self, server_info, timeout=30):
        """Get server tools synchronously with timeout"""
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self._get_server_tools(server_info))
            finally:
                new_loop.close()
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=timeout)
    
    def remove_mcp_server(self, server_name):
        """Remove/disable an MCP server"""
        print(f"🔧 [SERVER MANAGER] Removing server: {server_name}")
        
        try:
            if server_name not in self.mcp_servers:
                return {"success": False, "error": f"Server '{server_name}' not found"}
            
            # Remove from runtime
            del self.mcp_servers[server_name]
            if server_name in self.available_tools:
                del self.available_tools[server_name]
            
            # Save configuration
            self._save_servers_config()
            
            # Rebuild system prompt
            self._build_system_prompt_with_tools()
            
            print(f"🔧 [SERVER MANAGER] Server '{server_name}' removed successfully")
            return {
                "success": True,
                "message": f"✅ Server '{server_name}' removed successfully"
            }
            
        except Exception as e:
            print(f"🔧 [SERVER MANAGER] Error removing server: {e}")
            return {"success": False, "error": str(e)}
    
    def get_server_status(self):
        """Get status of all active servers"""
        servers = []
        for name, info in self.mcp_servers.items():
            tools = self.available_tools.get(name, [])
            
            # Handle different server types
            if info['type'] == 'file':
                command_str = f"file: {info.get('path', 'unknown')}"
            elif 'command' in info:
                args = info.get('args', [])
                command_str = f"{info['command']} {' '.join(args)}"
            else:
                command_str = f"type: {info.get('type', 'unknown')}"
            
            servers.append([
                name,
                command_str,
                len(tools),
                "✅ Active" if tools else "⚠️ No tools"
            ])
        return servers
    
    def _save_servers_config(self):
        """Save current servers to mcp_servers.json"""
        config_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
        
        try:
            # Convert current servers to config format
            config = {"mcpServers": {}}
            
            for name, info in self.mcp_servers.items():
                # Handle different server types properly
                if info['type'] == 'file':
                    # File-based server
                    server_config = {
                        "type": "file",
                        "path": info.get('path', '').replace(os.path.dirname(__file__) + os.sep, ''),
                        "description": info.get('description', f"MCP server: {name}"),
                        "enabled": True
                    }
                elif info['type'] in ['standard', 'package']:
                    # Command-based server (standard or package)
                    server_config = {
                        "command": info['command'],
                        "args": info['args'],
                        "description": info.get('description', f"MCP server: {name}")
                    }
                    
                    if info.get('env'):
                        server_config['env'] = info['env']
                else:
                    # Unknown type, skip
                    print(f"🔧 [CONFIG] Skipping unknown server type: {info['type']} for {name}")
                    continue
                    
                config["mcpServers"][name] = server_config
            
            # Write to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"🔧 [CONFIG] Saved {len(config['mcpServers'])} servers to {config_path}")
            
        except Exception as e:
            print(f"🔧 [CONFIG] Error saving configuration: {e}")
            import traceback
            print(f"🔧 [CONFIG] Traceback: {traceback.format_exc()}")
    
    # === Model Management Methods ===
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            print(f"🤖 [MODEL MANAGER] Available models: {model_names}")
            return model_names
        except Exception as e:
            print(f"🤖 [MODEL MANAGER] Error getting models: {e}")
            return [self.model]  # Return current model as fallback
    
    def change_model(self, new_model):
        """Switch to a different model"""
        if new_model == self.model:
            return f"Already using {new_model}"
        
        print(f"🤖 [MODEL MANAGER] Switching from {self.model} to {new_model}")
        
        try:
            # Verify the new model exists
            available_models = self.get_available_models()
            if new_model not in available_models:
                error_msg = f"Model '{new_model}' not found. Available: {available_models}"
                print(f"🤖 [MODEL MANAGER] ❌ {error_msg}")
                return f"❌ {error_msg}"
            
            # Switch the model
            old_model = self.model
            self.model = new_model
            
            print(f"🤖 [MODEL MANAGER] ✅ Successfully switched to {new_model}")
            print(f"🤖 [MODEL MANAGER] All future LLM calls will use: {self.model}")
            return f"✅ Switched from {old_model} to {new_model}"
            
        except Exception as e:
            print(f"🤖 [MODEL MANAGER] ❌ Error switching model: {e}")
            return f"❌ Error switching model: {e}"

def create_gradio_interface(agent):
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="AI Agent - Phase 1",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        
        /* Thinking content styling */
        .response-container {
            width: 100%;
            position: relative;
            padding-top: 20px;
        }
        
        .response-header {
            position: absolute;
            top: 0;
            right: 0;
            height: 18px;
        }
        
        .main-content {
            margin-bottom: 8px;
            line-height: 1.5;
        }
        
        .brain-icon-corner {
            cursor: pointer;
            font-size: 12px;
            opacity: 0.5;
            color: #d1a3a4;
            transition: all 0.2s ease;
            user-select: none;
            position: absolute;
            top: -2px;
            right: 0;
        }
        
        .brain-icon-corner:hover {
            opacity: 0.8;
            transform: scale(1.1);
        }
        
        .thinking-content {
            margin-top: 8px;
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f0f8 100%);
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 10px;
            animation: slideDown 0.3s ease-out;
            border-left: 3px solid #667eea;
        }
        
        .thinking-header {
            font-weight: bold;
            color: #4a5568;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .thinking-text {
            color: #4a5568;
            line-height: 1.5;
            font-size: 13px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-width: 100%;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                max-height: 0;
                padding-top: 0;
                padding-bottom: 0;
            }
            to {
                opacity: 1;
                max-height: 500px;
                padding-top: 12px;
                padding-bottom: 12px;
            }
        }
        """,
        head="""
        <script>
        window.toggleThinking = function(thinkingId) {
            const element = document.getElementById(thinkingId);
            const icon = document.querySelector(`[onclick*="${thinkingId}"]`);
            if (element && icon) {
                if (element.style.display === 'none' || element.style.display === '') {
                    element.style.display = 'block';
                    icon.style.opacity = '1';
                    icon.style.color = '#6b7280';
                } else {
                    element.style.display = 'none';
                    icon.style.opacity = '0.5';
                    icon.style.color = '#d1a3a4';
                }
            }
        }
        </script>
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🤖 EVAL AI Agent
           Ask anything - the agent will autonomously choose tools when helpful!
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="AI Assistant",
                            height=400,
                            show_copy_button=True,
                            render_markdown=False,
                            sanitize_html=False
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask me anything...",
                                lines=2,
                                scale=4,
                                show_label=False
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1)
                           
                            clear_btn = gr.Button("Clear", variant="secondary", scale=1, visible=False)
                        

                    
                    with gr.Column(scale=1):
                        gr.Markdown("### System Info")
                        
                        # Model selection dropdown
                        available_models = agent.get_available_models()
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=agent.model,
                            label="Active Model",
                            interactive=True
                        )
                        
                        # Model status display
                        model_status = gr.Textbox(
                            label="Model Status", 
                            visible=False,
                            interactive=False
                        )
                        
                        # Build tool status
                        tool_count = sum(len(tools) for tools in agent.available_tools.values())
                        server_count = len(agent.mcp_servers)
                        
                        # Get system prompt preview (first 100 chars)
                        prompt_preview = agent.system_prompt[:100] + "..." if len(agent.system_prompt) > 100 else agent.system_prompt
                        
                        def build_status_info(current_model=None):
                            model_to_show = current_model if current_model else agent.model
                            return f"""
                            <div style="background: #292929; padding: 10px; border-radius: 5px; color: white;">
                                <strong>Endpoint:</strong> {agent.host}<br>
                                <strong>Model:</strong> {model_to_show}<br>
                                <strong>Ollama:</strong> <span style="color: green;">✅ Connected</span><br>
                                <strong>MCP Servers:</strong> {server_count}<br>                        
                                <strong>System Prompt:</strong><br>
                                <div style="font-size: 0.9em; color: #666; font-style: italic; background: white; padding: 5px; border-radius: 3px; margin-top: 5px;">
                                    {prompt_preview}
                                </div>
                            </div>
                            """
                        
                        status_info = gr.HTML(build_status_info())
                        
                        if agent.available_tools:
                            gr.Markdown("### Available Tools")
                            tools_html = "<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9em;'>"
                            for server_name, tools in agent.available_tools.items():
                                tools_html += f"<strong>{server_name}:</strong><br>"
                                for tool in tools:
                                    tools_html += f"• {tool['name']}: {tool['description']}<br>"
                                tools_html += "<br>"
                            tools_html += "</div>"
                            gr.HTML(tools_html)
            
            with gr.Tab("Server Management"):
                gr.Markdown("### Add New MCP Server")
                gr.Markdown("Enter server name and JSON configuration (just like Claude Desktop)")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        server_name_input = gr.Textbox(
                            label="Server Name",
                            placeholder="my_custom_server"
                        )
                        server_config_input = gr.Code(
                            label="MCP Server Configuration (JSON)",
                            language="json",
                            lines=8,
                            placeholder='{\n  "command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-memory"]\n}'
                        )
                        
                    with gr.Column(scale=1):
                        test_btn = gr.Button("Test Configuration", variant="secondary")
                        add_btn = gr.Button("Add Server", variant="primary")
                        refresh_btn = gr.Button("Refresh Servers")
                
                status_output = gr.Textbox(label="Status", lines=3, interactive=False)
                
                gr.Markdown("### Active MCP Servers")
                servers_display = gr.Dataframe(
                    headers=["Server", "Command", "Tools", "Status"],
                    label="Currently Active Servers",
                    interactive=False
                )
                
                # Initialize server display with current servers
                def get_initial_servers():
                    return agent.get_server_status()
                
                # Set initial value
                servers_display.value = get_initial_servers()
                
                # Event handlers for server management
                def handle_test_server(config_json):
                    try:
                        config = json.loads(config_json)
                        result = agent.test_server_config(config)
                        if result["success"]:
                            return f"✅ {result['message']}"
                        else:
                            return f"❌ Test failed: {result['error']}"
                    except json.JSONDecodeError as e:
                        return f"❌ Invalid JSON: {e}"
                    except Exception as e:
                        return f"❌ Error: {e}"
                
                def handle_add_server(server_name, config_json):
                    if not server_name.strip():
                        return "❌ Please enter a server name", agent.get_server_status()
                    
                    result = agent.add_mcp_server(server_name.strip(), config_json)
                    
                    if result["success"]:
                        message = f"{result['message']}"
                        if result['tools']:
                            message += f"\nTools: {', '.join(result['tools'])}"
                        return message, agent.get_server_status()
                    else:
                        return f"❌ {result['error']}", agent.get_server_status()
                
                def handle_refresh_servers():
                    servers = agent.get_server_status()
                    return f"🔄 Servers refreshed - {len(servers)} active", servers
                
                # Bind server management events
                test_btn.click(
                    handle_test_server,
                    inputs=[server_config_input],
                    outputs=[status_output]
                )
                
                add_btn.click(
                    handle_add_server,
                    inputs=[server_name_input, server_config_input],
                    outputs=[status_output, servers_display]
                )
                
                refresh_btn.click(
                    handle_refresh_servers,
                    outputs=[status_output, servers_display]
                )
        
        # Chat event handlers
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            response = agent.chat(message, history)
            history.append((message, response))
            return history, ""
        
        def clear_chat():
            return [], ""
        

        
        # Bind chat events
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg]
        )
        

        
        # Model switching event handler
        def handle_model_change(selected_model):
            result = agent.change_model(selected_model)
            print(f"🤖 [UI] Model change result: {result}")
            
            # Update status info to show new model
            updated_status = build_status_info(agent.model)
            
            # Show/hide status message
            if result.startswith("✅"):
                return updated_status, gr.update(value=result, visible=True)
            elif result.startswith("Already"):
                return updated_status, gr.update(value="", visible=False)
            else:
                return updated_status, gr.update(value=result, visible=True)
        
        model_dropdown.change(
            handle_model_change,
            inputs=[model_dropdown],
            outputs=[status_info, model_status]
        )
        
        # Update server display when interface loads
        interface.load(
            fn=lambda: agent.get_server_status(),
            outputs=[servers_display]
        )
    
    return interface

def main():
    """Main application entry point"""
    print("🚀 Starting AI Agent - Proper MCP Implementation")
    print("=" * 60)
    
    try:
        # Initialize the agent
        agent = OllamaAgent()
        
        # Create Gradio interface
        interface = create_gradio_interface(agent)
        
        # Configure and launch the application
        interface.queue(default_concurrency_limit=1)  # Enable queue system
        interface.launch(
            server_name=os.getenv('GRADIO_HOST', '127.0.0.1'),
            server_port=int(os.getenv('GRADIO_PORT', '7860')),
            share=os.getenv('GRADIO_SHARE', 'false').lower() == 'true',
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 