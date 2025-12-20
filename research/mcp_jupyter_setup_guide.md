# MCP Jupyter Setup Guide for Claude Code

This guide explains how to set up Jupyter notebooks with Claude Code via MCP (Model Context Protocol), including integration with Neovim via claudecode.nvim.

## Architecture Overview

```
Neovim → claudecode.nvim → Claude Code CLI → MCP Jupyter / Python REPL servers
```

The key insight: You don't "turn on notebooks in Claude Code" directly — you wire notebooks into Claude Code via MCP servers.

## Prerequisites

- Python 3.8+
- Jupyter installed: `pip install jupyterlab ipykernel`
- Claude Code CLI installed and working

## 1. Install & Verify Claude Code CLI

### Linux / macOS / WSL
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### Windows PowerShell
```powershell
irm https://claude.ai/install.ps1 | iex
```

### Verify Installation
```bash
claude --version
claude doctor
```

---

## 2. MCP Server Options for Notebooks

### Option A: ClaudeJupy (Recommended for Simple Setup)

ClaudeJupy provides:
- Persistent Python kernel (REPL)
- Ability to create/manage Jupyter notebooks (add cells, etc.)

#### Install & Register
```bash
# Install MCP server
pipx install ml-jupyter-mcp
# or: pip install --user ml-jupyter-mcp

# Register with Claude Code
claude mcp add jupyter-executor ml-jupyter-mcp
```

#### Verify
```bash
claude mcp list
# Should show: jupyter-executor – ✓ Connected
```

#### Usage Examples
Once connected, you can use these tools in Claude:
```
mcp__jupyter-executor__execute_code("x = 42")
mcp__jupyter-executor__add_notebook_cell("analysis.ipynb", "code", "print(x)")
```

### Option B: Datalayer Jupyter MCP Server (Full Notebook Control)

This option provides full JupyterLab integration:
- Read/modify/execute cells
- Run all cells
- List notebooks
- Full notebook manipulation

#### Install Dependencies
```bash
pip install jupyterlab==4.4.1 jupyter-collaboration==4.0.2 jupyter-mcp-tools>=0.1.4 ipykernel
pip uninstall -y pycrdt datalayer_pycrdt
pip install datalayer_pycrdt==0.12.17
```

#### Run JupyterLab Server
```bash
jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN --ip 0.0.0.0
```

#### Register with Claude Code
```bash
claude mcp add-json "jupyter" '{
  "command": "uvx",
  "args": ["jupyter-mcp-server@latest"],
  "env": {
    "JUPYTER_URL": "http://localhost:8888",
    "JUPYTER_TOKEN": "MY_TOKEN",
    "ALLOW_IMG_OUTPUT": "true"
  }
}'
```

#### Available Tools
- `use_notebook` - Connect to a notebook
- `insert_cell` - Add a new cell
- `execute_cell` - Run a specific cell
- `notebook_run-all-cells` - Run all cells in notebook

---

## 3. Neovim Integration with claudecode.nvim

### Install claudecode.nvim (lazy.nvim)

```lua
return {
  {
    "coder/claudecode.nvim",
    version = "*",
    dependencies = { "folke/snacks.nvim" }, -- optional but recommended
    opts = {
      -- Claude Code CLI location (if not just "claude" in PATH)
      terminal_cmd = "claude",

      -- Use external terminal (Alacritty / tmux, etc.)
      terminal = {
        provider = "external",
        provider_opts = {
          -- Simple: Alacritty
          external_terminal_cmd = "alacritty -e %s",
          -- Or with working directory:
          -- external_terminal_cmd = "alacritty --working-directory %s -e %s",
        },
        split_side = "right",
        split_width_percentage = 0.30,
      },

      -- Optional: make Claude work from git root
      git_repo_cwd = true,
      log_level = "info",
      auto_start = true,
    },
    keys = {
      -- Normal & visual mode: open / focus Claude
      { "<C-,>", "<cmd>ClaudeCodeFocus<cr>", desc = "Claude Code", mode = { "n", "x" } },
    },
  },
}
```

### Terminal Provider Options

1. **External Terminal (Alacritty/Kitty)**
   ```lua
   terminal = {
     provider = "external",
     provider_opts = {
       external_terminal_cmd = "alacritty -e %s",
     },
   }
   ```

2. **Tmux (Manual)**
   Set `provider = "none"` and run `claude --ide` in a tmux pane manually.
   claudecode.nvim will still connect over MCP WebSocket.

3. **Built-in Neovim Terminal**
   ```lua
   terminal = {
     provider = "snacks",  -- or "native"
     split_side = "right",
     split_width_percentage = 0.30,
   }
   ```

---

## 4. Workflow: Using Notebooks with Claude in Neovim

### Step 1: Start Jupyter Server (if using Datalayer)
```bash
jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN
```
(ClaudeJupy auto-starts its kernel)

### Step 2: Open Neovim
```bash
nvim .
```

### Step 3: Open Claude Code
Press `<C-,>` or run `:ClaudeCodeFocus`

### Step 4: Work with Notebooks

**With ClaudeJupy:**
```
"Use jupyter-executor to run this Python snippet and keep the state alive."
"Use jupyter-executor to add a new code cell to analysis.ipynb that loads data.csv."
```

**With Datalayer Jupyter MCP:**
```
"Use the jupyter MCP server to connect to notebooks/test.ipynb,
insert a new code cell at the end that does XYZ, then run all cells."
```

---

## 5. Viewing/Editing .ipynb in Neovim

MCP lets Claude manipulate .ipynb files, but Neovim sees them as JSON blobs.

### Options for Better Notebook UX in Neovim:

1. **Jupytext Workflow**
   - Store notebooks as `.py` or `.md` via jupytext
   - Keep `.ipynb` in sync automatically
   ```bash
   pip install jupytext
   jupytext --set-formats ipynb,py:percent notebook.ipynb
   ```

2. **Neovim Notebook Plugins**
   - `jupynium.nvim` - Real-time sync with Jupyter
   - `molten.nvim` - Run code in Neovim with Jupyter kernels
   - `iron.nvim` - REPL integration

---

## 6. Troubleshooting Checklist

If notebooks aren't working with Claude Code:

1. **Check Claude CLI**
   ```bash
   claude --version
   ```

2. **Check Neovim Connection**
   ```vim
   :ClaudeCodeStatus
   ```
   Should show "connected"

3. **Check MCP Servers**
   ```bash
   claude mcp list
   ```
   Should show `jupyter-executor` or `jupyter` as Connected

4. **Check JupyterLab (if using Datalayer)**
   - Is the server running?
   - Is the token correct?
   - Can you access `http://localhost:8888`?

---

## 7. Quick Setup Script

```bash
#!/bin/bash
# MCP Jupyter Quick Setup for Claude Code

# Install ClaudeJupy (simpler option)
echo "Installing ClaudeJupy..."
pip install ml-jupyter-mcp

# Register with Claude Code
echo "Registering MCP server..."
claude mcp add jupyter-executor ml-jupyter-mcp

# Verify
echo "Verifying setup..."
claude mcp list

echo ""
echo "Setup complete! You can now use jupyter-executor in Claude Code."
echo "Example: 'Use jupyter-executor to run print(42)'"
```

---

## 8. Configuration Files

### Claude Code MCP Config Location
- Linux/macOS: `~/.config/claude-code/mcp.json`
- Windows: `%APPDATA%\claude-code\mcp.json`

### Example mcp.json
```json
{
  "servers": {
    "jupyter-executor": {
      "command": "ml-jupyter-mcp",
      "args": []
    }
  }
}
```

### For Datalayer Jupyter
```json
{
  "servers": {
    "jupyter": {
      "command": "uvx",
      "args": ["jupyter-mcp-server@latest"],
      "env": {
        "JUPYTER_URL": "http://localhost:8888",
        "JUPYTER_TOKEN": "MY_TOKEN",
        "ALLOW_IMG_OUTPUT": "true"
      }
    }
  }
}
```

---

## References

- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [claudecode.nvim GitHub](https://github.com/coder/claudecode.nvim)
- [ClaudeJupy/ml-jupyter-mcp](https://github.com/datalayer/jupyter-mcp)
- [Datalayer Jupyter MCP Server](https://github.com/datalayer/jupyter-mcp-server)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
