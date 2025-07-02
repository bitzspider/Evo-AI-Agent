#!/usr/bin/env python3
"""
Gmail MCP Server - Provides email sending capabilities using Gmail SMTP
"""

import asyncio
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Any
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the MCP server
app = Server("gmail-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="send_email",
            description="Send an email via Gmail SMTP to the configured notification recipient",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "message": {
                        "type": "string",
                        "description": "Email message body (plain text)"
                    }
                },
                "required": ["subject", "message"],
                "additionalProperties": False
            }
        )
    ]

def send_email_impl(to_email: str, subject: str, message: str, from_name: str = None) -> str:
    """
    Send an email via Gmail SMTP
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        message: Email message body (plain text)
        from_name: Optional sender name (defaults to email address)
    
    Returns:
        str: Success or error message with detailed status
    """
    status_log = []
    
    try:
        print(f"[EMAIL] Gmail MCP Server: send_email tool called")
        status_log.append("📧 GMAIL TOOL EXECUTED")
        
        # Get Gmail credentials from environment
        gmail_user = os.getenv('GMAIL_USER')
        gmail_password = os.getenv('GMAIL_APP_PASSWORD')
        
        print(f"[EMAIL] Checking credentials...")
        print(f"[EMAIL]    Gmail User: {'SET' if gmail_user else 'MISSING'}")
        print(f"[EMAIL]    App Password: {'SET' if gmail_password else 'MISSING'}")
        
        status_log.append(f"🔍 Credentials Check:")
        status_log.append(f"   Gmail User: {'✅ SET' if gmail_user else '❌ MISSING'}")
        status_log.append(f"   App Password: {'✅ SET' if gmail_password else '❌ MISSING'}")
        
        if not gmail_user or not gmail_password:
            error_msg = "Error: Gmail credentials not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD in .env file"
            print(f"[EMAIL] ERROR: {error_msg}")
            status_log.append(f"❌ ERROR: {error_msg}")
            return "\n".join(status_log)
        
        print(f"[EMAIL] Preparing email...")
        print(f"[EMAIL]    From: {from_name or gmail_user} <{gmail_user}>")
        print(f"[EMAIL]    To: {to_email}")
        print(f"[EMAIL]    Subject: {subject}")
        print(f"[EMAIL]    Message length: {len(message)} characters")
        
        status_log.append(f"📧 Email Preparation:")
        status_log.append(f"   From: {from_name or gmail_user} <{gmail_user}>")
        status_log.append(f"   To: {to_email}")
        status_log.append(f"   Subject: {subject}")
        status_log.append(f"   Message: {len(message)} characters")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{from_name or gmail_user} <{gmail_user}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach message body
        msg.attach(MIMEText(message, 'plain'))
        
        print(f"[EMAIL] Connecting to Gmail SMTP server...")
        status_log.append("🔗 Connecting to Gmail SMTP server...")
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable encryption
        status_log.append("🔒 TLS encryption enabled")
        
        print(f"[EMAIL] Authenticating...")
        status_log.append("🔐 Authenticating with Gmail...")
        server.login(gmail_user, gmail_password)
        status_log.append("✅ Authentication successful")
        
        print(f"[EMAIL] Sending email...")
        status_log.append("📤 Sending email...")
        
        # Send email
        text = msg.as_string()
        server.sendmail(gmail_user, to_email, text)
        server.quit()
        
        success_msg = f"SUCCESS: Email sent successfully to {to_email}"
        print(f"[EMAIL] {success_msg}")
        status_log.append(f"✅ {success_msg}")
        status_log.append("🔒 SMTP connection closed")
        
        return "\n".join(status_log)
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"ERROR: Gmail authentication failed. Check your email and app password"
        print(f"[EMAIL] {error_msg}")
        print(f"[EMAIL]    Details: {str(e)}")
        status_log.append(f"❌ {error_msg}")
        status_log.append(f"   Details: {str(e)}")
        return "\n".join(status_log)
        
    except smtplib.SMTPRecipientsRefused as e:
        error_msg = f"ERROR: Invalid recipient email address: {to_email}"
        print(f"[EMAIL] {error_msg}")
        print(f"[EMAIL]    Details: {str(e)}")
        status_log.append(f"❌ {error_msg}")
        status_log.append(f"   Details: {str(e)}")
        return "\n".join(status_log)
        
    except smtplib.SMTPException as e:
        error_msg = f"ERROR: SMTP Error: {str(e)}"
        print(f"[EMAIL] {error_msg}")
        status_log.append(f"❌ {error_msg}")
        return "\n".join(status_log)
        
    except Exception as e:
        error_msg = f"ERROR: Unexpected error: {str(e)}"
        print(f"[EMAIL] {error_msg}")
        status_log.append(f"❌ {error_msg}")
        return "\n".join(status_log)

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Execute tool."""
    
    print(f"[MCP] Gmail Server: Tool '{name}' called with arguments: {arguments}")
    
    if name == "send_email":
        subject = arguments["subject"]
        message = arguments["message"]
        
        # Get notification email from environment
        to_email = os.getenv('NOTIFICATION_EMAIL')
        print(f"[MCP] Sending email to configured recipient: {to_email}")
        
        if not to_email:
            error_msg = "Error: NOTIFICATION_EMAIL not configured in .env file"
            print(f"[MCP] ERROR: {error_msg}")
            result = error_msg
        else:
            result = send_email_impl(to_email, subject, message, "AI Agent")
        
        return [
            types.TextContent(
                type="text",
                text=result
            )
        ]
    
    else:
        error_msg = f"Unknown tool: {name}"
        print(f"[MCP] ERROR: {error_msg}")
        raise ValueError(error_msg)

async def main():
    """Main server entry point."""
    async with stdio_server() as streams:
        await app.run(
            streams[0], 
            streams[1], 
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 