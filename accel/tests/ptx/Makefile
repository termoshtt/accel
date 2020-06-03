.PHONY: all clean

all: add.ptx add.cubin sub.ptx sub.cubin texture.ptx texture.cubin

clean:
	rm *.ptx *.cubin

%.ptx: %.cu
	nvcc -ptx $<

%.cubin: %.cu
	nvcc -cubin $<
