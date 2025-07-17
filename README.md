# 🤖 EVO AI Agent

An AI agent application with MCP (Model Context Protocol) integration, Gradio interface, Ollama, and extensible tool system.

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
