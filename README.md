# 🤖 EVO AI Agent

A sophisticated AI agent application with MCP (Model Context Protocol) integration, featuring an intuitive Gradio interface and extensible tool system.

## ✨ Features

### 🧠 **Thinking Transparency**
- **Brain Icon Interface**: Click the subtle brain icon (🧠) in the top-right corner of responses to reveal the AI's internal reasoning process
- **Automatic Detection**: Responses containing `<think>` tags automatically display the thinking toggle
- **Clean UI**: Thinking content is hidden by default, maintaining a clean chat experience

### 🔧 **Extensible Tool System**
- **MCP Integration**: Full Model Context Protocol support for seamless tool integration
- **Custom Tools**: Built-in Gmail and time/date operations
- **Easy Extension**: Add new tools via simple JSON configuration
- **Real-time Discovery**: Tools are discovered and integrated automatically

### 🎨 **Modern Interface**
- **Gradio-Powered**: Clean, responsive web interface
- **Real-time Chat**: Instant AI responses with tool execution
- **Model Switching**: Switch between different Ollama models on-the-fly
- **Server Management**: Built-in MCP server configuration and management

### ⚡ **Multi-Tool Automation**
- **Autonomous Execution**: AI can chain multiple tools to complete complex tasks
- **Smart Decision Making**: Automatically determines when and which tools to use
- **Error Handling**: Robust error handling and recovery mechanisms

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Ollama** installed and running
- **Node.js** (for some MCP servers)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bitzspider/Evo-AI-Agent.git
   cd Evo-AI-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

4. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the interface**
   - Open your browser to `http://localhost:7860`
   - Start chatting with your AI agent!

## ⚙️ Configuration

### Environment Variables

Edit your `.env` file to customize the application:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3-latest

# Gradio Interface
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
GRADIO_SHARE=false

# Gmail Integration (optional)
GMAIL_USER=your-email@gmail.com
GMAIL_APP_PASSWORD=your-app-password
NOTIFICATION_EMAIL=recipient@gmail.com

# Development
DEBUG=false
```

### MCP Server Configuration

The `mcp_servers.json` file defines available tools:

```json
{
  "mcpServers": {
    "time_server": {
      "type": "file",
      "path": "custom_tools/time_server.py",
      "description": "Time and date operations",
      "enabled": true
    },
    "gmail_server": {
      "type": "file", 
      "path": "custom_tools/google_gmail.py",
      "description": "Email operations via Gmail",
      "enabled": true
    }
  }
}
```

## 🛠️ Built-in Tools

### 📧 **Gmail Integration**
- Send emails through Gmail SMTP
- Automatic recipient configuration
- Rich text and plain text support

### ⏰ **Time & Date Operations**
- Current time and date queries
- Timezone support
- Formatted date/time output

### 🌐 **External MCP Servers**
- **Memory Server**: Persistent storage and recall
- **Fetch Server**: HTTP requests and web content
- **Windows CLI**: System command execution
- **Wiki Integration**: Deep Wikipedia integration

## 💡 Usage Examples

### Basic Chat
```
User: What time is it?
AI: I'll check the current time for you.
[AI uses time tool automatically]
The current time is 2:30 PM EST, January 15, 2025.
```

### Thinking Process
When the AI uses complex reasoning, click the 🧠 icon to see:
```
💭 Let me think about this step by step:
1. The user is asking about time
2. I should use the get_current_time tool
3. Then format the response clearly
```

### Multi-Tool Workflows
```
User: Email me the current time
AI: I'll get the current time and send it to you via email.
[AI chains: time tool → email tool]
✅ Email sent with current time information!
```

## 🔧 Adding Custom Tools

### 1. Create Your Tool
Create a new Python file in `custom_tools/`:

```python
#!/usr/bin/env python3
import mcp.types as types
from mcp.server import Server

app = Server("my-tool")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="my_function",
            description="Description of what this tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "my_function":
        result = f"Processed: {arguments['param']}"
        return [types.TextContent(type="text", text=result)]
```

### 2. Register in Configuration
Add to `mcp_servers.json`:

```json
{
  "mcpServers": {
    "my_tool": {
      "type": "file",
      "path": "custom_tools/my_tool.py",
      "description": "My custom tool",
      "enabled": true
    }
  }
}
```

### 3. Restart and Use
Restart the application and your tool will be automatically discovered!

## 📁 Project Structure

```
Evo-AI-Agent/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── mcp_servers.json      # Tool configuration
├── env.example           # Environment template
├── custom_tools/         # Custom MCP tools
│   ├── google_gmail.py   # Gmail integration
│   └── time_server.py    # Time/date operations
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Repository**: [https://github.com/bitzspider/Evo-AI-Agent](https://github.com/bitzspider/Evo-AI-Agent)
- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **MCP Specification**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- **Gradio**: [https://gradio.app](https://gradio.app)

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app) for the web interface
- Powered by [Ollama](https://ollama.ai) for local LLM inference
- Implements [Model Context Protocol](https://modelcontextprotocol.io) for tool integration

---

**Made with ❤️ for the AI community** 