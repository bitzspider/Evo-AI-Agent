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
import pickle
from datetime import datetime

# Load environment variables
load_dotenv()

class OllamaAgent:
    def __init__(self):
        self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = os.getenv('OLLAMA_MODEL', 'qwen3:latest')
        self.temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.6'))
        self.action_temperature = float(os.getenv('OLLAMA_ACTION_TEMPERATURE', '0.2'))
        
        # Initialize Ollama client
        self.client = ollama.Client(host=self.host)
        
        # Initialize MCP state
        self.mcp_servers = {}
        self.available_tools = {}
        self.system_prompt = ""
        self._tools_discovered = False  # Flag to track lazy loading
        
        # Initialize chat history
        self.current_chat_id = None
        self.chat_history_dir = "chat_history"
        self.metadata_file = os.path.join(self.chat_history_dir, "metadata.json")
        self._init_chat_history()
        
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
Use exactly one tool per assistant message, as pure JSON (no extra text):
{"tool_name": "exact_tool_name", "parameters": {...}, "explanation": "brief reason"}

MULTI-TOOL WORKFLOWS (FEW-SHOT):
User: "Email me the current time"
Assistant -> {"tool_name": "get_current_time", "parameters": {"format": "human"}, "explanation": "Need current time first"}
(Tool returns "Current time: ...")
Assistant -> {"tool_name": "send_email", "parameters": {"subject": "Current time", "message": "Current time: ..."}, "explanation": "Send the requested email"}
(Tool returns success)
Assistant (final text): "✅ Sent you the current time by email."

User: "Fetch https://example.com and email me the contents"
Assistant -> {"tool_name": "fetch", "parameters": {"url": "https://example.com"}, "explanation": "Fetch page content"}
(Tool returns content)
Assistant -> {"tool_name": "send_email", "parameters": {"subject": "Requested content", "message": "<summary of content>..."}, "explanation": "Deliver results via email"}
(Tool returns success)
Assistant (final text): "✅ Emailed you the requested content."

CRITICAL TASK COMPLETION RULES:
- NEVER stop mid-task. If the user asks to email/send/save, execute the corresponding tool.
- Verify you've fully completed the request before finishing.

WHEN TO USE TOOLS:
- Use tools when the user asks you to DO something or when tool data is needed.

EMAIL TOOL NOTES:
- send_email uses the configured recipient; only provide "subject" and "message".

IMPORTANT:
- tool_name must exactly match one of the available tools above.
- Follow input schemas strictly.
- On tool errors, correct parameters and retry or pick an alternative.
- Final message must be plain text only; JSON is only for tool calls.
"""
        
        self.system_prompt = base_prompt + tools_section
        total_tools = sum(len(tools) for tools in self.available_tools.values() if tools)
        print(f"✅ System prompt rebuilt with {total_tools} available tools")

    def _llm_chat(self, messages, expect_json: bool = False, temperature: float | None = None):
        """Wrapper around Ollama chat with optional JSON-enforced output and tuned temperature.

        Keeping a single place to set options reduces drift and helps small models stay on format.
        """
        opts = {"temperature": (temperature if temperature is not None else self.temperature)}
        # Allow optional larger context via env
        num_ctx = os.getenv('OLLAMA_NUM_CTX')
        if num_ctx:
            try:
                opts["num_ctx"] = int(num_ctx)
            except Exception:
                pass
        if expect_json:
            # Ask Ollama to constrain output as JSON when we expect a tool call
            opts["format"] = "json"
        return self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options=opts
        )

    def _assess_task_completion(self, original_request, tools_used, latest_tool_result):
        """Assess whether the original user request has been fully completed"""
        
        # Extract key action verbs from original request
        action_indicators = {
            'send_email': ['email', 'send', 'mail', 'notify'],
            'fetch': ['fetch', 'get', 'retrieve', 'download'],
            'save': ['save', 'store', 'write', 'file'],
            'search': ['search', 'find', 'look'],
            'execute': ['run', 'execute', 'command'],
            'get_current_time': ['time', 'date', 'current']
        }
        
        request_lower = original_request.lower()
        
        # Determine what actions were requested
        requested_actions = []
        for tool, keywords in action_indicators.items():
            if any(keyword in request_lower for keyword in keywords):
                requested_actions.append(tool)
        
        # Check if all requested actions have been completed
        completed_actions = [tool_info['name'] for tool_info in tools_used]
        
        # Special multi-tool pattern detection
        is_fetch_and_email = ('fetch' in requested_actions and 'send_email' in requested_actions)
        is_get_time_and_email = ('get_current_time' in requested_actions and 'send_email' in requested_actions)
        
        # Task completion assessment
        if is_fetch_and_email:
            if 'fetch' in completed_actions and 'send_email' in completed_actions:
                return {'complete': True, 'reason': 'Both fetch and email completed'}
            elif 'fetch' in completed_actions and 'send_email' not in completed_actions:
                return {'complete': False, 'reason': 'Fetched content but email not sent', 'next_action': 'send_email'}
            else:
                return {'complete': False, 'reason': 'Fetch not completed', 'next_action': 'fetch'}
        
        elif is_get_time_and_email:
            if 'get_current_time' in completed_actions and 'send_email' in completed_actions:
                return {'complete': True, 'reason': 'Both time retrieval and email completed'}
            elif 'get_current_time' in completed_actions and 'send_email' not in completed_actions:
                return {'complete': False, 'reason': 'Got time but email not sent', 'next_action': 'send_email'}
            else:
                return {'complete': False, 'reason': 'Time not retrieved', 'next_action': 'get_current_time'}
        
        # Single action requests
        elif len(requested_actions) == 1:
            if requested_actions[0] in completed_actions:
                return {'complete': True, 'reason': f'Requested action {requested_actions[0]} completed'}
            else:
                return {'complete': False, 'reason': f'Requested action {requested_actions[0]} not completed', 'next_action': requested_actions[0]}
        
        # Default assessment based on common patterns
        else:
            # If user asked for any action-oriented request and we haven't done it
            action_verbs = ['send', 'email', 'save', 'store', 'notify', 'run', 'execute']
            has_action_request = any(verb in request_lower for verb in action_verbs)
            
            if has_action_request and not tools_used:
                return {'complete': False, 'reason': 'Action requested but no tools used', 'next_action': 'determine_tool'}
            elif has_action_request and tools_used:
                # Check if the last tool result indicates success
                if 'error' in latest_tool_result.lower() or 'failed' in latest_tool_result.lower():
                    return {'complete': False, 'reason': 'Tool execution failed', 'next_action': 'retry_or_alternative'}
                else:
                    return {'complete': True, 'reason': 'Action appears to have been completed'}
            else:
                return {'complete': True, 'reason': 'Informational request satisfied'}

    def _create_structured_decision_prompt(self, original_request, tool_name, tool_result, tools_used):
        """Create a structured decision prompt to guide the agent's next action"""
        
        # Assess task completion
        task_assessment = self._assess_task_completion(original_request, tools_used, tool_result)
        
        # Check if tool result has errors or warnings
        has_errors = any(indicator in tool_result.lower() for indicator in [
            'error', 'failed', 'timeout', 'truncated', 'start_index', 'try again', 'incomplete'
        ])
        
        has_sufficient_content = len(tool_result) > 100  # Has meaningful content
        clipped = self._compress_for_prompt(tool_result, 1200)
        
        # Build structured prompt
        prompt = f"""TASK EVALUATION:
- Original request: "{original_request}"
- Tool just executed: {tool_name}
- Tools used so far: {[t['name'] for t in tools_used]}
- Task completion status: {"COMPLETE" if task_assessment['complete'] else "INCOMPLETE"}
- Reason: {task_assessment['reason']}

TOOL RESULT ANALYSIS:
 - Result length: {len(tool_result)} characters
- Has errors/warnings: {"YES" if has_errors else "NO"}
- Has sufficient content: {"YES" if has_sufficient_content else "NO"}
 - Tool result (clipped): {clipped}

DECISION GUIDANCE:"""

        # Tight constraints for small models
        allowed_tools = self._get_all_available_tools()
        prompt += f"""
You must either:
- OUTPUT EXACTLY ONE tool call as pure JSON, or
- If and only if the task is fully complete, output FINAL text (no JSON).

When using a tool, return JSON with this exact shape and nothing else:
{{"tool_name": "one_of_allowed", "parameters": {{...}}, "explanation": "why"}}

Allowed tool_name values: {allowed_tools}
One tool per message. Do not include any text outside the JSON.
"""

        if task_assessment['complete']:
            prompt += """\nFINAL: Provide the final text answer (no JSON, no extra thinking)."""
        else:
            next_action = task_assessment.get('next_action', 'determine_next_step')
            prompt += f"""\nNEXT ACTION: {next_action}"""

        return prompt

    def chat(self, message, history):
        """
        Handle chat interaction with multi-tool execution and structured decision making
        
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
            
            # Store original request for task completion assessment
            original_request = message
            
            # Get LLM response
            print(f"🤖 [CHAT] Getting LLM response...")
            print(f"🤖 [CHAT] Using model: {self.model}")
            print(f"🤖 [CHAT] Using temperature: {self.temperature}")
            response = self._llm_chat(conversation, expect_json=False, temperature=self.temperature)
            
            assistant_response = response['message']['content']
            print(f"🤖 [CHAT] LLM Response: {assistant_response[:200]}{'...' if len(assistant_response) > 200 else ''}")
            
            # Multi-tool execution loop with structured decision making
            current_response = assistant_response
            max_tool_calls = 5  # Prevent infinite loops
            tool_call_count = 0
            tools_used = []  # Track tools used for completion assessment
            
            while tool_call_count < max_tool_calls:
                # Check if current response wants to use a tool
                print(f"🤖 [CHAT] Checking for tool requests... (iteration {tool_call_count + 1})")
                tool_request = self._parse_tool_request(current_response)
                
                if not tool_request:
                    # No more tools needed, return current response
                    print(f"🤖 [CHAT] ✅ No tool needed, returning response after {tool_call_count} tool calls")
                    print(f"🤖 [CHAT] 📝 Final response length: {len(current_response)} characters")
                    
                    # Strip thinking content from final response
                    thinking_result = self._parse_thinking_content(current_response)
                    if thinking_result['has_thinking']:
                        final_response = thinking_result['main_content'].strip()
                        print(f"🤖 [CHAT] 🧠 Stripped thinking content from final response")
                    else:
                        final_response = current_response
                    
                    # If response is empty or too short, provide a default completion message
                    if not final_response.strip() or len(final_response.strip()) < 10:
                        if tool_call_count > 0:
                            final_response = f"✅ Task completed! I successfully executed {tool_call_count} tool(s) to fulfill your request."
                            print(f"🤖 [CHAT] 📝 Using default completion message")
                    
                    print(f"🤖 [CHAT] 📝 Final response content: '{final_response}'")
                    print(f"{'='*60}\n")
                    return final_response
                
                # Check if tool request has proper format
                if 'tool_name' not in tool_request or 'parameters' not in tool_request:
                    # JSON exists but format is wrong - provide tool-specific error correction
                    print(f"🤖 [CHAT] 🚨 Malformed tool request detected - providing specific guidance")
                    
                    # Try to extract attempted tool name from various fields
                    attempted_tool = None
                    for field in ['tool', 'tool_name', 'name']:
                        if field in tool_request:
                            attempted_tool = tool_request[field]
                            break
                    
                    # Create tool-specific error correction
                    error_correction = self._create_tool_specific_error_correction(attempted_tool, tool_request)
                    
                    # Add error correction to conversation
                    conversation.append({"role": "assistant", "content": current_response})
                    conversation.append({"role": "system", "content": error_correction})
                    
                    # Get corrected response from LLM
                    print(f"🤖 [CHAT] Getting corrected response from LLM...")
                    next_response = self._llm_chat(conversation, expect_json=True, temperature=self.action_temperature)
                    
                    current_response = next_response['message']['content']
                    print(f"🤖 [CHAT] Got corrected response: {current_response[:100]}...")
                    
                    # Continue loop to check if this response has more tool requests
                    continue
                
                # LLM wants to use a tool with correct format
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
                
                # Track tools used for completion assessment
                tools_used.append({
                    'name': tool_name,
                    'parameters': parameters,
                    'result': tool_result
                })
                
                # Add previous response to conversation
                conversation.append({"role": "assistant", "content": current_response})
                
                # Create structured decision prompt instead of simple tool result
                decision_prompt = self._create_structured_decision_prompt(
                    original_request, tool_name, tool_result, tools_used
                )
                conversation.append({"role": "system", "content": decision_prompt})
                
                # Get next response from LLM with structured guidance
                print(f"🤖 [CHAT] Getting next response with structured decision guidance...")
                next_response = self._llm_chat(conversation, expect_json=True, temperature=self.action_temperature)
                
                current_response = next_response['message']['content']
                print(f"🤖 [CHAT] Got next response: {current_response[:100]}...")
                
                # Continue loop to check if this response has more tool requests
            
            # If we hit max tool calls, return what we have
            print(f"🤖 [CHAT] ⚠️  Hit maximum tool calls ({max_tool_calls}), returning current response")
            
            # Strip thinking content from final response
            thinking_result = self._parse_thinking_content(current_response)
            if thinking_result['has_thinking']:
                final_response = thinking_result['main_content'].strip()
                print(f"🤖 [CHAT] 🧠 Stripped thinking content from final response")
            else:
                final_response = current_response
            
            print(f"{'='*60}\n")
            return final_response
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(f"🤖 [CHAT] ❌ ERROR: {error_msg}")
            import traceback
            print(f"🤖 [CHAT] Traceback: {traceback.format_exc()}")
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
        """Parse LLM response for tool usage requests with strict JSON format"""
        import json
        
        print(f"🔍 [TOOL PARSER] Parsing response for tool requests...")
        print(f"🔍 [TOOL PARSER] Response length: {len(response_text)} characters")
        
        # Strip thinking content first to prevent interference with tool detection
        thinking_result = self._parse_thinking_content(response_text)
        clean_response = thinking_result['main_content']
        
        if thinking_result['has_thinking']:
            print(f"🔍 [TOOL PARSER] 🧠 Stripped thinking content, using clean response")
            print(f"🔍 [TOOL PARSER] Clean response length: {len(clean_response)} characters")
        
        # Try strict JSON (when we enforced JSON output)
        try:
            strict = json.loads(clean_response)
            if isinstance(strict, dict) and 'tool_name' in strict and 'parameters' in strict:
                print("🔍 [TOOL PARSER] ✅ Strict JSON tool request")
                return strict
        except Exception:
            pass

        try:
            # Find all potential JSON blocks by looking for balanced braces in clean response
            start_idx = clean_response.find('{')
            if start_idx == -1:
                print(f"🔍 [TOOL PARSER] ❌ No opening brace found")
                return None
            
            # Find the matching closing brace by counting nesting
            brace_count = 0
            end_idx = -1
            
            for i in range(start_idx, len(clean_response)):
                if clean_response[i] == '{':
                    brace_count += 1
                elif clean_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                print(f"🔍 [TOOL PARSER] ❌ No matching closing brace found")
                return None
                
            json_str = clean_response[start_idx:end_idx]
            print(f"🔍 [TOOL PARSER] Extracted JSON (length {len(json_str)}): {json_str[:200]}{'...' if len(json_str) > 200 else ''}")
            
            # Parse the JSON
            tool_request = json.loads(json_str)
            print(f"🔍 [TOOL PARSER] Successfully parsed JSON")
            
            # Store the raw JSON for error correction if needed
            tool_request['_raw_json'] = json_str
            
            # Strict format validation - only accept exact format
            if ('tool_name' in tool_request and 'parameters' in tool_request):
                print(f"🔍 [TOOL PARSER] ✅ Valid tool request found: {tool_request['tool_name']}")
                return tool_request
            else:
                # JSON exists but format is wrong - return it for tool-specific error correction
                print(f"🔍 [TOOL PARSER] ❌ JSON found but wrong format")
                print(f"🔍 [TOOL PARSER]   Found fields: {list(tool_request.keys())}")
                return tool_request  # Return the malformed request for specific error handling
                
        except json.JSONDecodeError as e:
            print(f"🔍 [TOOL PARSER] ❌ JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"🔍 [TOOL PARSER] ❌ Unexpected error: {e}")
            return None
    
    def _create_tool_specific_error_correction(self, attempted_tool, malformed_request):
        """Create tool-specific error correction with exact schema for the attempted tool"""
        print(f"🔍 [TOOL PARSER] Creating tool-specific error correction for: {attempted_tool}")
        
        # Find the tool schema
        tool_schema = None
        tool_description = None
        
        if attempted_tool:
            for server_tools in self.available_tools.values():
                for tool in server_tools:
                    if tool['name'] == attempted_tool:
                        tool_schema = tool.get('schema', {})
                        tool_description = tool.get('description', '')
                        break
                if tool_schema:
                    break
        
        # Build tool-specific correction message
        if tool_schema and attempted_tool:
            correction_msg = f"""
🚨 TOOL FORMAT ERROR: You attempted to use tool '{attempted_tool}' but used wrong JSON format.

Your JSON had these fields: {list(malformed_request.keys())}
Required fields: tool_name, parameters

CORRECT FORMAT for '{attempted_tool}':
{{
  "tool_name": "{attempted_tool}",
  "parameters": {{"""
            
            # Add specific parameters from schema
            props = tool_schema.get('properties', {})
            required_params = tool_schema.get('required', [])
            
            if props:
                for param_name, param_info in props.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', 'No description')
                    is_required = param_name in required_params
                    
                    if param_type == 'string':
                        example_value = f'"example_value"'
                    elif param_type == 'integer':
                        example_value = '0'
                    elif param_type == 'boolean':
                        example_value = 'true'
                    elif param_type == 'object':
                        example_value = '{}'
                    elif param_type == 'array':
                        example_value = '[]'
                    else:
                        example_value = '"value"'
                    
                    required_marker = " [REQUIRED]" if is_required else " [OPTIONAL]"
                    correction_msg += f"""
    "{param_name}": {example_value}{required_marker}  // {param_desc}"""
                    
                    # Show enum values if available
                    if 'enum' in param_info:
                        enum_values = param_info['enum']
                        correction_msg += f" | Valid values: {enum_values}"
            else:
                correction_msg += """
    "param1": "value1",
    "param2": "value2\""""
            
            correction_msg += f"""
  }},
  "explanation": "Brief explanation of why using {attempted_tool}"
}}

Tool Description: {tool_description}

Please use the EXACT format above and try again."""
        
        else:
            # Generic correction if we can't find the specific tool
            available_tools = self._get_all_available_tools()
            correction_msg = f"""
🚨 TOOL FORMAT ERROR: Invalid JSON format for tool request.

Your JSON had these fields: {list(malformed_request.keys())}
Required fields: tool_name, parameters

CORRECT FORMAT:
{{
  "tool_name": "exact_tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }},
  "explanation": "Brief explanation"
}}

Available tools: {', '.join(available_tools)}

Please use the EXACT format above and try again."""
        
        return correction_msg
    
    def _get_all_available_tools(self):
        """Get list of all available tool names"""
        tools = []
        for server_tools in self.available_tools.values():
            for tool in server_tools:
                tools.append(tool['name'])
        return tools

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

    def _compress_for_prompt(self, text: str, max_chars: int = 1200) -> str:
        """Compress long tool outputs to keep small models focused and within context limits."""
        if not text:
            return ""
        t = text.strip()
        if len(t) <= max_chars:
            return t
        head = t[: max_chars // 2]
        tail = t[-max_chars // 2 :]
        return f"{head}\n...\n{tail}"

    def _extract_clean_content_for_history(self, formatted_response):
        """Extract clean main content from a formatted response for chat history"""
        import re
        
        # If it's not HTML (no HTML tags), return as-is
        if '<' not in formatted_response:
            return formatted_response
        
        # Extract main content from HTML response
        # Look for the main-content div
        main_content_match = re.search(r'<div class="main-content">(.*?)</div>', formatted_response, re.DOTALL)
        if main_content_match:
            main_content_html = main_content_match.group(1)
            # Convert HTML back to plain text
            import html
            # Replace <br> with newlines
            clean_content = main_content_html.replace('<br>', '\n')
            # Unescape HTML entities
            clean_content = html.unescape(clean_content)
            return clean_content.strip()
        
        # Fallback: return original if no main-content div found
        return formatted_response

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

    def _init_chat_history(self):
        """Initialize chat history system"""
        if not os.path.exists(self.chat_history_dir):
            os.makedirs(self.chat_history_dir)
        
        if not os.path.exists(self.metadata_file):
            initial_metadata = {
                "next_id": 1,
                "chats": []
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(initial_metadata, f, indent=2)

    def get_chat_list(self):
        """Get list of chat sessions for sidebar"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            chats = metadata.get('chats', [])
            # Return list of tuples (display_name, chat_id) for Gradio dropdown
            return [(f"{chat['title']} ({chat['created']})", chat['id']) for chat in chats]
        except Exception as e:
            print(f"❌ Error loading chat list: {e}")
            return []

    def load_chat(self, chat_id):
        """Load a chat by ID"""
        try:
            chat_file = os.path.join(self.chat_history_dir, f"chat_{chat_id:03d}.pkl")
            if os.path.exists(chat_file):
                with open(chat_file, 'rb') as f:
                    history = pickle.load(f)
                self.current_chat_id = chat_id
                return history
            return []
        except Exception as e:
            print(f"❌ Error loading chat {chat_id}: {e}")
            return []

    def save_chat(self, history):
        """Save current chat"""
        if not history or len(history) == 0:
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if self.current_chat_id is None:
                # New chat
                chat_id = metadata['next_id']
                metadata['next_id'] = chat_id + 1
                
                # Create title from first message
                title = history[0][0][:30] + "..." if len(history[0][0]) > 30 else history[0][0]
                
                chat_info = {
                    "id": chat_id,
                    "title": title,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "message_count": len(history)
                }
                metadata['chats'].append(chat_info)
                self.current_chat_id = chat_id
            else:
                # Update existing
                for chat in metadata['chats']:
                    if chat['id'] == self.current_chat_id:
                        chat['message_count'] = len(history)
                        break
            
            # Save metadata and chat
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            chat_file = os.path.join(self.chat_history_dir, f"chat_{self.current_chat_id:03d}.pkl")
            with open(chat_file, 'wb') as f:
                pickle.dump(history, f)
            
            return self.current_chat_id
        except Exception as e:
            print(f"❌ Error saving chat: {e}")
            return None

    def new_chat(self):
        """Start new chat"""
        self.current_chat_id = None
        return []

def create_gradio_interface(agent):
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="AI Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Left sidebar styling */
        .chat-sidebar {
            background: #f8f9fa !important;
            border-right: 1px solid #e9ecef !important;
            min-height: 100vh !important;
            padding: 10px !important;
        }
        
        .new-chat-btn {
            width: 100% !important;
            background: #6c757d !important;
            border: none !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 4px !important;
            padding: 8px !important;
            margin-bottom: 10px !important;
        }
        
        .new-chat-btn:hover {
            background: #5a6268 !important;
        }
        
        .chat-item {
            padding: 8px !important;
            margin: 2px 0 !important;
            border-radius: 4px !important;
            cursor: pointer !important;
            font-size: 13px !important;
            background: #ffffff !important;
            border: 1px solid #dee2e6 !important;
        }
        
        .chat-item:hover {
            background: #e9ecef !important;
        }
        
        /* Chat dropdown styling */
        .chat-dropdown {
            margin-bottom: 10px !important;
        }
        
        .chat-dropdown label {
            font-size: 12px !important;
            color: #6c757d !important;
            margin-bottom: 4px !important;
        }
        
        /* Main chat area */
        .main-chat {
            padding: 20px !important;
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
        
        .thinking-text {
            color: #4a5568;
            line-height: 1.5;
            font-size: 13px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-width: 100%;
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
        
        with gr.Row():
            # Left sidebar for chat history (OUTSIDE tabs)
            with gr.Column(scale=1, elem_classes=["chat-sidebar"]):
                new_chat_btn = gr.Button("+", elem_classes=["new-chat-btn"])
                
                # Chat history dropdown
                chat_history_dropdown = gr.Dropdown(
                    label="Previous Chats",
                    choices=agent.get_chat_list(),
                    value=None,
                    interactive=True,
                    elem_classes=["chat-dropdown"]
                )
            
            # Main content area with tabs
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        with gr.Row():
                            # Main chat area
                            with gr.Column(scale=4, elem_classes=["main-chat"]):
                                chatbot = gr.Chatbot(
                                    label="AI Assistant",
                                    height=500,
                                    show_copy_button=True
                                )
                                
                                with gr.Row():
                                    msg = gr.Textbox(
                                        placeholder="Ask me anything...",
                                        lines=2,
                                        scale=5,
                                        show_label=False
                                    )
                                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                            
                            # Right sidebar for system info (minimal)
                            with gr.Column(scale=1):
                                gr.Markdown("### System")
                                
                                # Model selection dropdown
                                available_models = agent.get_available_models()
                                model_dropdown = gr.Dropdown(
                                    choices=available_models,
                                    value=agent.model,
                                    label="Model",
                                    interactive=True
                                )
                                
                                # Model status display
                                model_status = gr.Textbox(
                                    label="Status", 
                                    visible=False,
                                    interactive=False
                                )
                                
                                # Simple system info
                                gr.Markdown(f"**Servers:** {len(agent.mcp_servers)}")
                                if agent.available_tools:
                                    tool_count = sum(len(tools) for tools in agent.available_tools.values())
                                    gr.Markdown(f"**Tools:** {tool_count}")
                    
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
                                    value='{\n  "command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-memory"]\n}'
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
                        
                        # Add delete functionality
                        gr.Markdown("### Remove MCP Server")
                        with gr.Row():
                            with gr.Column(scale=3):
                                server_delete_dropdown = gr.Dropdown(
                                    label="Select Server to Remove",
                                    choices=list(agent.mcp_servers.keys()),  # Simple list, not tuples
                                    value=None,
                                    interactive=True
                                )
                            with gr.Column(scale=1):
                                delete_btn = gr.Button(
                                    "🗑️ Delete Server", 
                                    variant="stop"
                                )
                        
                        # Helper functions for server management  
                        def get_server_names():
                            return list(agent.mcp_servers.keys())
                        
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
                                return "❌ Please enter a server name", agent.get_server_status(), get_server_names()
                            
                            result = agent.add_mcp_server(server_name.strip(), config_json)
                            
                            if result["success"]:
                                message = f"{result['message']}"
                                if result['tools']:
                                    message += f"\nTools: {', '.join(result['tools'])}"
                                return message, agent.get_server_status(), get_server_names()
                            else:
                                return f"❌ {result['error']}", agent.get_server_status(), get_server_names()
                        
                        def handle_delete_server(server_name):
                            if not server_name:
                                return "❌ Please select a server to delete", agent.get_server_status(), get_server_names()
                            
                            result = agent.remove_mcp_server(server_name)
                            
                            if result["success"]:
                                return f"✅ {result['message']}", agent.get_server_status(), get_server_names()
                            else:
                                return f"❌ {result['error']}", agent.get_server_status(), get_server_names()
                        
                        def handle_refresh_servers():
                            # Force tools discovery to refresh server status
                            agent._tools_discovered = False
                            agent._discover_tools_lazy()
                            servers = agent.get_server_status()
                            return f"🔄 Servers refreshed - {len(servers)} active", servers, get_server_names()
                        
                        # Bind server management events
                        test_btn.click(
                            handle_test_server,
                            inputs=[server_config_input],
                            outputs=[status_output]
                        )
                        
                        add_btn.click(
                            handle_add_server,
                            inputs=[server_name_input, server_config_input],
                            outputs=[status_output, servers_display, server_delete_dropdown]
                        )
                        
                        delete_btn.click(
                            handle_delete_server,
                            inputs=[server_delete_dropdown],
                            outputs=[status_output, servers_display, server_delete_dropdown]
                        )
                        
                        refresh_btn.click(
                            handle_refresh_servers,
                            outputs=[status_output, servers_display, server_delete_dropdown]
                        )
        
        # Chat event handlers
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            response = agent.chat(message, history)
            history.append([message, response])
            
            # Auto-save after each response
            save_result = agent.save_chat(history)
            
            # Return updated dropdown choices with current chat selected
            updated_choices = agent.get_chat_list()
            
            # Find the current chat in the choices to select it
            current_selection = None
            if agent.current_chat_id is not None:
                for choice_display, choice_id in updated_choices:
                    if choice_id == agent.current_chat_id:
                        current_selection = choice_id
                        break
            
            return history, "", gr.update(choices=updated_choices, value=current_selection)
        
        def clear_chat():
            return [], ""
        
        # Chat history event handlers
        
        def handle_new_chat():
            """Start new chat"""
            agent.new_chat()
            updated_choices = agent.get_chat_list()
            return [], gr.update(choices=updated_choices, value=None)
        
        def handle_load_chat(selected_chat):
            """Load selected chat"""
            if selected_chat:
                history = agent.load_chat(selected_chat)
                updated_choices = agent.get_chat_list()
                return history, gr.update(choices=updated_choices, value=selected_chat)
            return [], gr.update()
        
        # Bind chat events
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, chat_history_dropdown]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, chat_history_dropdown]
        )
        
        # Bind new chat button
        new_chat_btn.click(
            handle_new_chat,
            outputs=[chatbot, chat_history_dropdown]
        )
        
        # Bind dropdown selection
        chat_history_dropdown.change(
            handle_load_chat,
            inputs=[chat_history_dropdown],
            outputs=[chatbot, chat_history_dropdown]
        )
        
        # Model switching
        def handle_model_change(selected_model):
            result = agent.change_model(selected_model)
            if result.startswith("✅"):
                return gr.update(value=result, visible=True)
            elif result.startswith("Already"):
                return gr.update(value="", visible=False)
            else:
                return gr.update(value=result, visible=True)
        
        model_dropdown.change(
            handle_model_change,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # Initialize both chat history and server management on load
        def handle_interface_load():
            """Initialize interface components on load"""
            # Initialize chat history dropdown
            chat_choices = agent.get_chat_list()
            
            # Initialize server management quickly without triggering tool discovery
            # Tool discovery can be slow (network/package startup). Do it on-demand via the Refresh button or first chat.
            servers = agent.get_server_status()
            server_names = list(agent.mcp_servers.keys())
            
            return gr.update(choices=chat_choices, value=None), servers, server_names
        
        interface.load(
            fn=handle_interface_load,
            outputs=[chat_history_dropdown, servers_display, server_delete_dropdown]
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