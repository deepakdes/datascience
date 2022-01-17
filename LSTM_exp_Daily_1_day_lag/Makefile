SHELL := /bin/bash
VERSION?=latest
TASK_IMAGES:=$(shell find tasks -name Dockerfile -printf '%h ' )
#REGISTRY=base:5000

tasks/%: FORCE
	set -e ;\
	docker build -t model_pipeline_$(@F):$(VERSION) $@ --network=host;\
	docker tag model_pipeline_$(@F):$(VERSION) model_pipeline_$(@F):$(VERSION) ;\
  

images: $(TASK_IMAGES)

run: images
	ls

FORCE: ;
