# those are just default placeoholder, will be overridden by local Makefile.local
HOST ?= dummyhost
REMOTE_DIR ?= /dummy/path

RSYNC_OPTS = -axzP --exclude-from=.rsyncignore

-include Makefile.local

sync:
	rsync $(RSYNC_OPTS) ./ $(HOST):$(REMOTE_DIR)

sync-clean: 
	rsync $(RSYNC_OPTS) --delete ./ $(HOST):$(REMOTE_DIR)

ping:
	ssh $(HOST) "echo 'Conncted to $(HOST)'"
