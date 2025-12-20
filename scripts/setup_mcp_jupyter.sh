#!/bin/bash
# MCP Jupyter Quick Setup for Claude Code
# This script sets up the ClaudeJupy MCP server for notebook integration

set -e

echo "========================================"
echo "MCP Jupyter Setup for Claude Code"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Claude Code CLI
echo ""
echo "Checking prerequisites..."

if ! command -v claude &> /dev/null; then
    echo -e "${RED}Error: Claude Code CLI not found${NC}"
    echo ""
    echo "Install Claude Code CLI first:"
    echo "  Linux/macOS: curl -fsSL https://claude.ai/install.sh | bash"
    echo "  Windows:     irm https://claude.ai/install.ps1 | iex"
    exit 1
fi

echo -e "${GREEN}✓ Claude Code CLI found${NC}"
claude --version

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}✓ Python found: $PYTHON_CMD${NC}"

# Check pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip not found${NC}"
    exit 1
fi

PIP_CMD=$(command -v pip3 || command -v pip)
echo -e "${GREEN}✓ pip found: $PIP_CMD${NC}"

# Ask which MCP server to install
echo ""
echo "========================================"
echo "Choose MCP Server to Install"
echo "========================================"
echo ""
echo "1) ClaudeJupy (ml-jupyter-mcp)"
echo "   - Simple setup"
echo "   - Persistent Python kernel"
echo "   - Basic notebook manipulation"
echo ""
echo "2) Datalayer Jupyter MCP"
echo "   - Full JupyterLab integration"
echo "   - Complete notebook control"
echo "   - Requires running JupyterLab server"
echo ""
echo "3) Both"
echo ""

read -p "Select option [1/2/3] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Installing ClaudeJupy..."
        $PIP_CMD install ml-jupyter-mcp

        echo ""
        echo "Registering with Claude Code..."
        claude mcp add jupyter-executor ml-jupyter-mcp

        echo -e "${GREEN}✓ ClaudeJupy installed and registered${NC}"
        ;;

    2)
        echo ""
        echo "Installing Datalayer Jupyter MCP dependencies..."
        $PIP_CMD install jupyterlab jupyter-mcp-tools ipykernel

        echo ""
        echo -e "${YELLOW}Note: You need to configure the Jupyter server URL and token.${NC}"

        read -p "Enter Jupyter URL (default: http://localhost:8888): " jupyter_url
        jupyter_url=${jupyter_url:-http://localhost:8888}

        read -p "Enter Jupyter token (default: MY_TOKEN): " jupyter_token
        jupyter_token=${jupyter_token:-MY_TOKEN}

        echo ""
        echo "Registering with Claude Code..."
        claude mcp add-json "jupyter" "{
            \"command\": \"uvx\",
            \"args\": [\"jupyter-mcp-server@latest\"],
            \"env\": {
                \"JUPYTER_URL\": \"$jupyter_url\",
                \"JUPYTER_TOKEN\": \"$jupyter_token\",
                \"ALLOW_IMG_OUTPUT\": \"true\"
            }
        }"

        echo -e "${GREEN}✓ Datalayer Jupyter MCP installed and registered${NC}"
        echo ""
        echo -e "${YELLOW}Remember to start JupyterLab:${NC}"
        echo "  jupyter lab --port 8888 --IdentityProvider.token $jupyter_token"
        ;;

    3)
        echo ""
        echo "Installing ClaudeJupy..."
        $PIP_CMD install ml-jupyter-mcp
        claude mcp add jupyter-executor ml-jupyter-mcp
        echo -e "${GREEN}✓ ClaudeJupy installed${NC}"

        echo ""
        echo "Installing Datalayer Jupyter MCP..."
        $PIP_CMD install jupyterlab jupyter-mcp-tools ipykernel

        read -p "Enter Jupyter URL (default: http://localhost:8888): " jupyter_url
        jupyter_url=${jupyter_url:-http://localhost:8888}

        read -p "Enter Jupyter token (default: MY_TOKEN): " jupyter_token
        jupyter_token=${jupyter_token:-MY_TOKEN}

        claude mcp add-json "jupyter" "{
            \"command\": \"uvx\",
            \"args\": [\"jupyter-mcp-server@latest\"],
            \"env\": {
                \"JUPYTER_URL\": \"$jupyter_url\",
                \"JUPYTER_TOKEN\": \"$jupyter_token\",
                \"ALLOW_IMG_OUTPUT\": \"true\"
            }
        }"
        echo -e "${GREEN}✓ Both MCP servers installed${NC}"
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"
echo ""
claude mcp list

echo ""
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "You can now use MCP tools in Claude Code:"
echo ""
if [[ $choice == "1" || $choice == "3" ]]; then
    echo "ClaudeJupy examples:"
    echo "  'Use jupyter-executor to run: print(42)'"
    echo "  'Use jupyter-executor to add a cell to notebook.ipynb'"
    echo ""
fi
if [[ $choice == "2" || $choice == "3" ]]; then
    echo "Datalayer Jupyter examples:"
    echo "  'Use jupyter to open my_notebook.ipynb'"
    echo "  'Use jupyter to execute all cells'"
    echo ""
    echo "Don't forget to start JupyterLab first!"
fi
