# Link to Itensor lib
ITENSOR_DIR=/home/urza/Projects/ITensor-v2.1.1-gcc
#ITENSOR_DIR=/home/urza/Software/ITensor-v2.1.1-gcc-withMKL
include $(ITENSOR_DIR)/this_dir.mk
include $(ITENSOR_DIR)/options.mk

TENSOR_HEADERS=$(ITENSOR_DIR)/itensor/core.h

# Dependencies
ARPACK_INC=-I/home/urza/Software/arpack-ng
ARPACK_LIB=-L/home/urza/Software/arpack-ng/build/lib -Wl,-rpath=/home/urza/Software/arpack-ng/build/lib \
	-larpack

# 3. 'main' function is in a file called 'get-env.cc', then
#    set APP to 'get-env'. Running 'make' will compile the app.
#    Running 'make debug' will make a program called 'get-env-g'
#    which includes debugging symbols and can be used in gdb (Gnu debugger);
APP =opt-full-update-2x2-3site
APP2=opt-su-3site
APP3=get-cluster-env_v2

# 4. Add any headers your program depends on here. The make program
#    will auto-detect if these headers have changed and recompile your app.
HEADERS =engine.h models.h full-update-TEST.h full-update.h simple-update_v2.h \
	cluster-ev-builder.h ctm-cluster-env_v2.h ctm-cluster-io.h ctm-cluster.h ctm-cluster-global.h \
	mpo.h su2.h arpack-rcdn.h json.hpp

# 5. For any additional .cc files making up your project,
#    add their full filenames here.
CCFILES=$(APP).cc cluster-ev-builder.cc engine.cc full-update-TEST.cc full-update.cc \
	simple-update_v2.cc models.cc ctm-cluster-env_v2.cc ctm-cluster-io.cc ctm-cluster.cc \
	mpo.cc su2.cc

CCFILES2=$(APP2).cc engine.cc simple-update_v2.cc full-update-TEST.cc full-update.cc models.cc \
	cluster-ev-builder.cc ctm-cluster-env_v2.cc ctm-cluster-io.cc ctm-cluster.cc \
	mpo.cc su2.cc

# CCFILES3=$(APP3).cc cluster-ev-builder.cc ctm-cluster-env_v2.cc \
# 	ctm-cluster-io.cc ctm-cluster.cc su2.cc


#Mappings --------------
# see https://www.gnu.org/software/make/manual/html_node/Text-Functions.html
OBJECTS=$(patsubst %.cc,%.o, $(CCFILES))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

OBJECTS2=$(patsubst %.cc,%.o, $(CCFILES2))
GOBJECTS2=$(patsubst %.cc,%.debug_objs/%, $(OBJECTS2))

# OBJECTS3=$(patsubst %.cc,%.o, $(CCFILES3))
# GOBJECTS3=$(patsubst %,.debug_objs/%, $(OBJECTS3))


#Rules ------------------
# see https://www.gnu.org/software/make/manual/make.html#Pattern-Intro
# see https://www.gnu.org/software/make/manual/make.html#Automatic-Variables
%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) $(ARPACK_INC) -Wno-unused-function -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) $(ARPACK_INC) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(ARPACK_INC) $(OBJECTS) -o $(APP).x $(ARPACK_LIB) $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(ARPACK_INC) $(GOBJECTS) -o $(APP)-g.x $(ARPACK_LIB) $(LIBGFLAGS)

$(APP2): $(OBJECTS2) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(ARPACK_INC) $(OBJECTS2) -o $(APP2).x $(ARPACK_LIB) $(LIBFLAGS)

$(APP2)-g: mkdebugdir $(GOBJECTS2) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(ARPACK_INC) $(GOBJECTS2) -o $(APP2)-g.x $(ARPACK_LIB) $(LIBGFLAGS)

#$(APP3): $(OBJECTS3) $(ITENSOR_LIBS)
#	$(CCCOM) $(CCFLAGS) $(OBJECTS3) -o $(APP3).x $(LIBFLAGS)





test-lin-sys: $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) test-lin-sys.cc -o test-lin-sys.x $(LIBFLAGS)

test-lin-sys-g: mkdebugdir $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) test-lin-sys.cc -o test-lin-sys-g.x $(LIBGFLAGS)

test-mklsvd: $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) test-mklsvd.cc -o test-mklsvd.x $(LIBFLAGS)

test-mklsvd-g: mkdebugdir $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) test-mklsvd.cc -o test-mklsvd-g.x $(LIBGFLAGS)

test-arpack:
	$(CCCOM) $(CCGFLAGS) -fext-numeric-literals test-arpack.cc -o test-arpack.x \
	-L/home/urza/Software/arpack-ng/build/lib -Wl,-rpath=/home/urza/Software/arpack-ng/build/lib \
	-larpack -I/home/urza/Software/arpack-ng

# arpack-rcdn:
# 	$(CCCOM) $(CCGFLAGS) -fext-numeric-literals arpack-rcdn.cc -c arpack-rcdn.o \
# 	-L/home/urza/Software/arpack-ng/build/lib -Wl,-rpath=/home/urza/Software/arpack-ng/build/lib \
# 	-larpack -I/home/urza/Software/arpack-ng

test-arpack-itensor:
	$(CCCOM) $(CCFLAGS) $(ARPACK_INC) test-arpack-itensor.cc -o test-arpack-itensor.x \
	$(ARPACK_LIB) $(LIBFLAGS)

clean:
	rm -fr .debug_objs *.o $(APP).x $(APP)-g.x $(APP2).x $(APP2)-g.x $(APP3).x $(APP3)-g.x

mkdebugdir:
	mkdir -p .debug_objs
