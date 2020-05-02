CUDA_VERSIONS    := 10.0 10.1 10.2
NIGHTLY_VERSIONS := 2020-01-02 2020-05-01

CI_REGISTRY_IMAGE  ?= registry.gitlab.com/termoshtt/accel
CI_COMMIT_REF_SLUG ?= manual

define ubuntu
ubuntu$(1)-cuda$(2)-nightly$(3):
	sed -e "s/UBUNTU_VERSION/$(1)/"  \
	    -e "s/CUDA_VERSION/$(2)/"    \
	    -e "s/NIGHTLY_VERSION/$(3)/" \
	    < ubuntu.Dockerfile          \
	    > $$@.Dockerfile
	docker build -f $$@.Dockerfile -t $(CI_REGISTRY_IMAGE)/$$@:$(CI_COMMIT_REF_SLUG) .
	docker push $(CI_REGISTRY_IMAGE)/$$@:$(CI_COMMIT_REF_SLUG)
endef

define centos
centos$(1)-cuda$(2)-nightly$(3):
	sed -e "s/CENTOS_VERSION/$(1)/"  \
	    -e "s/CUDA_VERSION/$(2)/"    \
	    -e "s/NIGHTLY_VERSION/$(3)/" \
	    < centos.Dockerfile          \
	    > $$@.Dockerfile
	docker build -f $$@.Dockerfile -t $(CI_REGISTRY_IMAGE)/$$@:$(CI_COMMIT_REF_SLUG) .
	docker push $(CI_REGISTRY_IMAGE)/$$@:$(CI_COMMIT_REF_SLUG)
endef

.PHONY: clean

all: $(foreach NIGHTLY_VERSION,$(NIGHTLY_VERSIONS), \
       $(foreach CUDA_VERSION,$(CUDA_VERSIONS),\
         ubuntu18.04-cuda$(CUDA_VERSION)-nightly$(NIGHTLY_VERSION) \
         centos6-cuda$(CUDA_VERSION)-nightly$(NIGHTLY_VERSION) \
         centos7-cuda$(CUDA_VERSION)-nightly$(NIGHTLY_VERSION) \
       ) \
     )

$(foreach NIGHTLY_VERSION,$(NIGHTLY_VERSIONS), \
  $(foreach CUDA_VERSION,$(CUDA_VERSIONS),     \
    $(eval $(call ubuntu,18.04,$(CUDA_VERSION),$(NIGHTLY_VERSION))) \
    $(eval $(call centos,6,$(CUDA_VERSION),$(NIGHTLY_VERSION))) \
    $(eval $(call centos,7,$(CUDA_VERSION),$(NIGHTLY_VERSION))) \
  ) \
)

clean:
	rm -rf *-cuda*-nightly*.Dockerfile