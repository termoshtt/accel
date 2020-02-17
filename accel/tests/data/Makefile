.PHONY: all clean ptx

all: dest.cubin

clean:
	rm *.ptx *.cubin

%.ptx: %.cu
	nvcc --keep-device-functions -ptx $<

dest.cubin: add.ptx sub.ptx
	nvlink --arch=sm_70 $^ -o $@
