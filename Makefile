
all: 
	cd cpp && make && cd ..

.PHONY: test demo
test:
	cd cpp && make test && cd ..

demo:
	cd cpp && make demo && cd ..

.PHONY: clean
clean:
	cd cpp && make clean && cd ..
