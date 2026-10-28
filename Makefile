COMPOSE ?= docker compose
APP_PORT ?= 8501
DEBUG_PORT ?= 5678

.PHONY: app-start app-stop debug-start debug-stop check-clean-state check-app-port check-debug-port

app-start: check-clean-state check-app-port
	@echo "Starting app profile on port $(APP_PORT)..."
	@$(COMPOSE) --profile app up --build -d
	@echo "App started. Open http://localhost:$(APP_PORT)"

app-stop:
	@echo "Stopping app profile..."
	@$(COMPOSE) --profile app down --remove-orphans


debug-start: check-clean-state check-app-port check-debug-port
	@echo "Starting debug profile on ports $(APP_PORT) and $(DEBUG_PORT)..."
	@$(COMPOSE) --profile debug up --build -d
	@echo "Debug started. App: http://localhost:$(APP_PORT), debugpy: localhost:$(DEBUG_PORT)"


debug-stop:
	@echo "Stopping debug profile..."
	@$(COMPOSE) --profile debug down --remove-orphans


check-clean-state:
	@running_services="$$($(COMPOSE) ps --services --status running)"; \
	if [ -n "$$running_services" ]; then \
		echo "Refusing to start: compose services are still running:"; \
		echo "$$running_services"; \
		echo "Run 'make app-stop' or 'make debug-stop' first."; \
		exit 1; \
	fi


check-app-port:
	@if command -v lsof >/dev/null 2>&1; then \
		if lsof -iTCP:$(APP_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			echo "Port $(APP_PORT) is already in use. Stop the process/container first."; \
			exit 1; \
		fi; \
	elif command -v ss >/dev/null 2>&1; then \
		if ss -ltn "( sport = :$(APP_PORT) )" | tail -n +2 | grep -q .; then \
			echo "Port $(APP_PORT) is already in use. Stop the process/container first."; \
			exit 1; \
		fi; \
	fi


check-debug-port:
	@if command -v lsof >/dev/null 2>&1; then \
		if lsof -iTCP:$(DEBUG_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			echo "Port $(DEBUG_PORT) is already in use. Stop the process/container first."; \
			exit 1; \
		fi; \
	elif command -v ss >/dev/null 2>&1; then \
		if ss -ltn "( sport = :$(DEBUG_PORT) )" | tail -n +2 | grep -q .; then \
			echo "Port $(DEBUG_PORT) is already in use. Stop the process/container first."; \
			exit 1; \
		fi; \
	fi
