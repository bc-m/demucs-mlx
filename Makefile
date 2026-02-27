INSTALL_DIR := $(HOME)/.local/bin
REPO_DIR    := $(shell cd "$(dir $(lastword $(MAKEFILE_LIST)))" && pwd)
PYTHON      := python3
COMMAND     := demucs-mlx

.PHONY: install uninstall deps help

help: ## Show this help
	@echo ""
	@echo "  demucs-mlx installer"
	@echo "  ────────────────────"
	@echo ""
	@echo "  make install     Install demucs-mlx command into your PATH"
	@echo "  make uninstall   Remove demucs-mlx from your PATH"
	@echo "  make deps        Install Python dependencies only"
	@echo ""

deps: ## Install Python dependencies
	@echo "Installing Python dependencies..."
	@$(PYTHON) -m pip install -r "$(REPO_DIR)/requirements.txt" -q
	@echo "Done."

install: deps ## Install demucs-mlx into ~/.local/bin
	@mkdir -p "$(INSTALL_DIR)"
	@echo '#!/bin/bash' > "$(INSTALL_DIR)/$(COMMAND)"
	@echo '# demucs-mlx — Music source separation on Apple Silicon' >> "$(INSTALL_DIR)/$(COMMAND)"
	@echo 'exec $(PYTHON) "$(REPO_DIR)/separate.py" "$$@"' >> "$(INSTALL_DIR)/$(COMMAND)"
	@chmod +x "$(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@echo "  ✓ Installed $(COMMAND) → $(INSTALL_DIR)/$(COMMAND)"
	@echo ""
	@if echo "$$PATH" | tr ':' '\n' | grep -qx "$(INSTALL_DIR)"; then \
		echo "  Ready! Try: $(COMMAND) --help"; \
	else \
		echo "  ⚠ $(INSTALL_DIR) is not in your PATH."; \
		echo ""; \
		echo "  Add this line to your shell profile (~/.zshrc or ~/.bashrc):"; \
		echo ""; \
		echo "    export PATH=\"$(INSTALL_DIR):\$$PATH\""; \
		echo ""; \
		echo "  Then restart your terminal or run: source ~/.zshrc"; \
	fi
	@echo ""

uninstall: ## Remove demucs-mlx from PATH
	@rm -f "$(INSTALL_DIR)/$(COMMAND)"
	@echo "  ✓ Removed $(COMMAND) from $(INSTALL_DIR)"
