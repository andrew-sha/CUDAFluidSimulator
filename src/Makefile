### This Makefile is adapted from the one in assignment 2

EXECUTABLE := sph
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
CU_FILES   := simulator.cu
CU_DEPS    :=
CC_FILES   := main.cpp display.cpp
LOGS	   := logs

all: $(EXECUTABLE)

SRC_FILES := $(shell find . -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' \))

format:
	@for file in $(SRC_FILES); do \
                echo "Formatting $$file"; \
                clang-format -i $$file; \
        done

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -I /usr/local/cuda-11.7/include
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -I/usr/local/cuda-11.7/include

LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS)) # probably could get rid of the frameworks since they are currently empty and will likely remain that way

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/simulator.o $(OBJDIR)/display.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

check:	default
		./checker.pl

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)




$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
