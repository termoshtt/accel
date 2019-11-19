all:
	#pgc++ -g -acc -Mcudalib=cusolver,cublas -Mcuda -mp -ta=tesla:nordc -Minfo=accel -fast -std=c++11 a.cpp 
	#pgc++ -g -acc -Mcudalib=cusolver,cublas -Mcuda -mp -ta=tesla:nordc -Minfo=accel -fast -std=c++11 b.cpp 
	g++ -fopenmp -fopenacc -lcuda -O3 -std=c++11 a.cpp  \
 	#	-foffload=-lm -foffload="-O3" -fno-float-store \

	#	-foffload=nvptx

# 	pgc++ -g -acc -mp -ta=tesla:nordc -Minfo=accel -fast -std=c++11 a.cpp \
# 	-Minline \
# 	-laccapi \
# 	-laccg \
# 	-laccn \
# 	-laccg2 \
# 	-ldl \
# 	-lcudadevice \
# 	-lpthread \
#   	-lpgc \
# 	-lpgf90rtl -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl -lpgftnrtl -lpgatm -lpgkomp -lomp -lomptarget -lpthread -lpgmath -lnspgc -lpgc -lrt -lpthread -lm -lgcc -lc -lgcc -lgcc_s -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgftnrtl -lpgatm -lpgkomp -lomp -lomptarget -lpthread -lpgmath -lnspgc -lpgc -lrt -lpthread -lpgmath -lnspgc -lpgc -lrt -lm -lgcc -lc -lgcc -lc -lgcc_s -lpgf90rtl -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl -lpgftnrtl -lrt
#
#


#   	-lrt \
#   	-lpgmath \
#   	-lnspgc \
# 	-lpgatm \
#
#  	-lpgf90rtl \
#  	-lpgf90 \
#  	-lpgf90_rpm1 \
#    	-lpgf902 \
#    	-lpgf90rtl \
#    	-lpgftnrtl \
#   	-lpgkomp \
#    	-lpthread \
#    	-lpthread \
#    	-lomp \
#    	-lomptarget \
#    	-lm \
#    	-lgcc \
#    	-lc \
#    	-lgcc \
#    	-lgcc_s \
#    	-lpgf90 \
#    	-lpgf90_rpm1 \
#    	-lpgf902 \
#    	-lpgftnrtl \
#    	-lpgatm \
#    	-lpgkomp \
#    	-lomp \
#    	-lomptarget \
#    	-lpthread \
#    	-lpgmath \
#    	-lnspgc \
#    	-lpgc \
#    	-lrt \
#    	-lpthread \
#    	-lpgmath \
#    	-lnspgc \
#    	-lpgc \
#    	-lrt \
#    	-lm \
#    	-lgcc \
#    	-lc \
#    	-lgcc \
#    	-lc \
#    	-lgcc_s \
#    	-lpgf90rtl \
#    	-lpgf90 \
#    	-lpgf90_rpm1 \
#    	-lpgf902 \
#    	-lpgf90rtl \
#    	-lpgftnrtl \


cpu: a.cpp
	pgc++ -g -mp -O3 -std=c++11 a.cpp

f90:
	pgf90 -g -mp -acc -ta=tesla -Minfo=accel -O3 ./acc_f1.f90 -o f90.out
