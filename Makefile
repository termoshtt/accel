REGISTRY  := registry.ritc.jp/ricos/allgebra
.PHONY: test allgebra 


all: allgebra

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(REGISTRY)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(REGISTRY)
endif

allgebra: Dockerfile
	docker build -t $(REGISTRY) . -f Dockerfile

push: login allgebra
	docker push $(REGISTRY)

in:  
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test $(REGISTRY)
	
test:  
	docker run -it --gpus all --privileged --mount type=bind,src=$(PWD)/test,dst=/test $(REGISTRY) \
	/bin/bash -c "cd test; make; make test; make clean"
