-- claudecode.nvim Configuration Example
-- Add this to your lazy.nvim plugins

return {
  {
    "coder/claudecode.nvim",
    version = "*",
    dependencies = { "folke/snacks.nvim" }, -- optional but recommended
    opts = {
      -- Claude Code CLI location (if not just "claude" in PATH)
      terminal_cmd = "claude",

      -- Terminal configuration
      terminal = {
        -- Options: "snacks", "native", "external", "none"
        -- "snacks" - Uses snacks.nvim terminal (recommended)
        -- "native" - Uses Neovim's built-in terminal
        -- "external" - Uses external terminal like Alacritty/Kitty
        -- "none" - Manual management (run claude --ide in tmux)
        provider = "snacks",

        -- For snacks/native providers
        split_side = "right",
        split_width_percentage = 0.30,

        -- For external terminal (uncomment to use)
        -- provider = "external",
        -- provider_opts = {
        --   -- Alacritty
        --   external_terminal_cmd = "alacritty -e %s",
        --   -- Or Kitty
        --   -- external_terminal_cmd = "kitty %s",
        --   -- Or with working directory
        --   -- external_terminal_cmd = "alacritty --working-directory %s -e %s",
        -- },
      },

      -- Work from git repository root
      git_repo_cwd = true,

      -- Logging level: "debug", "info", "warn", "error"
      log_level = "info",

      -- Auto-start Claude Code when Neovim opens
      auto_start = true,
    },

    -- Keybindings
    keys = {
      -- Normal & visual mode: open / focus Claude
      { "<C-,>", "<cmd>ClaudeCodeFocus<cr>", desc = "Claude Code", mode = { "n", "x" } },

      -- Additional useful keybindings (optional)
      { "<leader>cc", "<cmd>ClaudeCodeFocus<cr>", desc = "Claude Code Focus" },
      { "<leader>cs", "<cmd>ClaudeCodeStatus<cr>", desc = "Claude Code Status" },
      { "<leader>ct", "<cmd>ClaudeCodeToggle<cr>", desc = "Claude Code Toggle" },
    },
  },
}

--[[
USAGE NOTES:

1. After adding this config, restart Neovim and run :Lazy sync

2. Commands available:
   :ClaudeCodeFocus   - Open/focus Claude Code panel
   :ClaudeCodeStatus  - Show connection status
   :ClaudeCodeToggle  - Toggle Claude Code panel

3. For external terminal (Alacritty/Kitty):
   - Set provider = "external"
   - Configure external_terminal_cmd

4. For tmux workflow:
   - Set provider = "none"
   - Run `claude --ide` manually in a tmux pane
   - claudecode.nvim will connect via WebSocket

5. MCP servers registered with `claude mcp add` are automatically
   available in the Claude Code session.

TROUBLESHOOTING:

- If Claude doesn't connect:
  1. Check :ClaudeCodeStatus
  2. Verify `claude --version` works in terminal
  3. Check `claude doctor` for issues

- If MCP tools don't work:
  1. Run `claude mcp list` to verify servers
  2. Make sure Jupyter server is running (for Datalayer MCP)
]]
