.PHONY: all clean ptx cubin

all: ptx cubin
ptx: add.ptx sub.ptx
cubin: add.cubin sub.cubin

clean:
	rm *.ptx *.cubin

%.ptx: %.cu
	nvcc --keep-device-functions -ptx $<

%.cubin: %.cu
	nvcc --keep-device-functions -cubin $<
