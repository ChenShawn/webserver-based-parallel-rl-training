PHONY: flask
flask:
	g++ -c -fpic shm.cpp
	g++ -shared -o libshm.so shm.o
	cp ./libshm.so ../

PHONY: test
test:
	g++ shm.cpp -o shm.o -D MODE_DEBUG
	./shm.o

PHONY: brpc
brpc:
	mkdir bld && cd bld && cmake .. && make

PHONY: clean
clean:
	-rm ./*.o
	-rm ./*.so
	-rm -rf ./bld
