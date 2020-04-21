CXX=clang++
CXX_FLAGS=-Wall -pedantic -std=c++17 -Ofast -funroll-loops

all: nanort

main.o: main.cc
	$(CXX) $(CXX_FLAGS) -c $^ -I/usr/local/include -I ../.. -I. -I ../common -o $@

obj_loader.o: obj-loader.cc
	$(CXX) $(CXX_FLAGS) -c $^ -I ../.. -I. -I ../common -o $@

nanort: main.o obj_loader.o
	$(CXX) $^ -L/usr/local/lib -lembree3 -o $@

clean:
	rm -f nanort *.o

.PHONY: clean run_nanort 
