libtrain.so: train.cpp
	g++ -c -fpic train.cpp
	g++ -shared -o libtrain.so train.o
	cp ./libtrain.so ../

PHONY: test
test:
	g++ train.cpp -o train.o -D MODE_DEBUG
	./train.o

PHONY: clean
clean:
	rm ./*.o
	rm ./*.so
