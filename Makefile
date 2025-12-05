# those are just default placeoholder, will be overridden by local Makefile.local
HOST ?= dummyhost
REMOTE_DIR ?= /dummy/path

RSYNC_OPTS = -axzP --exclude-from=.rsyncignore

-include Makefile.local

# Get the project root directory
PROJECT_ROOT := $(shell pwd)

# Run the Streamlit webapp
app:
	PYTHONPATH=$(PROJECT_ROOT) uv run streamlit run webapp/app.py

# Run the Streamlit webapp in headless mode (for servers)
app-headless:
	PYTHONPATH=$(PROJECT_ROOT) uv run streamlit run webapp/app.py --server.headless true

sync:
	rsync $(RSYNC_OPTS) ./ $(HOST):$(REMOTE_DIR)

sync-clean: 
	rsync $(RSYNC_OPTS) --delete ./ $(HOST):$(REMOTE_DIR)

ping:
	ssh $(HOST) "echo 'Conncted to $(HOST)'"
