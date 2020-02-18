.PHONY: all clean

all: add.ptx add.cubin

clean:
	rm *.ptx *.cubin

%.ptx: %.cu
	nvcc -ptx $<

%.cubin: %.cu
	nvcc -cubin $<
