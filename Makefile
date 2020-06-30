REGISTRY  := registry.ritc.jp/ricos/allgebra
CI_COMMIT_REF_NAME ?= manual_deploy

.PHONY: test allgebra 


all: allgebra

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(REGISTRY)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(REGISTRY)
endif

allgebra: Dockerfile
	docker build -t $(REGISTRY):$(CI_COMMIT_REF_NAME) . -f Dockerfile

push: login allgebra
	docker push $(REGISTRY):$(CI_COMMIT_REF_NAME)

in:  
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test $(REGISTRY):$(CI_COMMIT_REF_NAME)

test:  
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test $(REGISTRY):$(CI_COMMIT_REF_NAME) \
	/bin/bash -c "cd test; make; make test; make clean"
