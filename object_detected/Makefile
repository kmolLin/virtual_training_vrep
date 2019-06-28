#Pyslvs Makefile

#author: Yuan Chang
#copyright: Copyright (C) 2016-2018
#license: AGPL
#email: pyslvs@gmail.com

LAUNCHSCRIPT = luanch

all: test

.PHONY: build clean test

build: $(LAUNCHSCRIPT).spec
	pyinstaller -F $< \
--hidden-import=PyQt5 \
--add-data inference_graph;inference_graph \
--add-data training;training

test: build
	$(eval EXE = $(shell dir dist /b))
	./dist/$(EXE)

clean:
	-rd build /s /q
	-rd dist /s /q
#-del launch_pyslvs.spec
