# Link to Itensor lib

ITENSOR_DIR=/home/urza/Software/ITensor-v2.1.1-intel
include $(ITENSOR_DIR)/this_dir.mk
include $(ITENSOR_DIR)/options.mk

TENSOR_HEADERS=$(ITENSOR_DIR)/itensor/core.h

# 3. 'main' function is in a file called 'get-env.cc', then
#    set APP to 'get-env'. Running 'make' will compile the app.
#    Running 'make debug' will make a program called 'get-env-g'
#    which includes debugging symbols and can be used in gdb (Gnu debugger);
APP =get-cluster-env_v2
APP2=opt-simple-update
#APP3=get-cluster-env_v2
APP4=opt-simple-update-2x2-2site
APP5=opt-simple-update-2x2-3site
APP6=opt-full-update-2x2-3site

# 4. Add any headers your program depends on here. The make program
#    will auto-detect if these headers have changed and recompile your app.
HEADERS =full-update.h simple-update.h cluster-ev-builder.h ctm-cluster-env_v2.h ctm-cluster-io.h \
	ctm-cluster.h ctm-cluster-global.h su2.h json.hpp
HEADERS2=simple-update.h ctm-cluster-global.h ctm-cluster.h su2.h json.hpp
#HEADERS3=cluster-ev-builder.h ctm-cluster-env_v2.h ctm-cluster-io.h \
	ctm-cluster.h ctm-cluster-global.h su2.h json.hpp
HEADERS4=cluster-ev-builder.h simple-update.h ctm-cluster-global.h \
	ctm-cluster.h su2.h json.hpp
HEADERS5=cluster-ev-builder.h simple-update.h ctm-cluster-global.h \
	ctm-cluster.h su2.h json.hpp
HEADERS5=cluster-ev-builder.h simple-update.h ctm-cluster-global.h \
	ctm-cluster.h su2.h json.hpp
HEADERS6=cluster-ev-builder.h full-update.h ctm-cluster-env_v2.h \
	ctm-cluster-io.h ctm-cluster.h ctm-cluster-global.h su2.h json.hpp

HEADERSN=ctm-cluster-io.h ctm-cluster.h ctm-cluster-global.h

# 5. For any additional .cc files making up your project,
#    add their full filenames here.
CCFILES=$(APP).cc cluster-ev-builder.cc ctm-cluster-env_v2.cc \
	ctm-cluster-io.cc ctm-cluster.cc su2.cc
CCFILES2=$(APP2).cc simple-update.cc ctm-cluster-io.cc ctm-cluster.cc \
	su2.cc
#CCFILES3=$(APP3).cc cluster-ev-builder.cc ctm-cluster-env_v2.cc \
	ctm-cluster-io.cc ctm-cluster.cc su2.cc
CCFILES4=$(APP4).cc cluster-ev-builder.cc simple-update.cc ctm-cluster-io.cc \
	ctm-cluster.cc su2.cc
CCFILES5=$(APP5).cc cluster-ev-builder.cc simple-update.cc ctm-cluster-io.cc \
	ctm-cluster.cc su2.cc
CCFILES6=$(APP6).cc cluster-ev-builder.cc full-update.cc ctm-cluster-env_v2.cc \
	ctm-cluster-io.cc ctm-cluster.cc su2.cc

CCFILESN=ctm-cluster-io.cc ctm-cluster.cc

#Mappings --------------
# see https://www.gnu.org/software/make/manual/html_node/Text-Functions.html
OBJECTS=$(patsubst %.cc,%.o, $(CCFILES))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))
OBJECTS2=$(patsubst %.cc,%.o, $(CCFILES2))
#OBJECTS3=$(patsubst %.cc,%.o, $(CCFILES3))
OBJECTS4=$(patsubst %.cc,%.o, $(CCFILES4))
GOBJECTS4=$(patsubst %,.debug_objs/%, $(OBJECTS4))
OBJECTS5=$(patsubst %.cc,%.o, $(CCFILES5))
GOBJECTS5=$(patsubst %,.debug_objs/%, $(OBJECTS5))
OBJECTS6=$(patsubst %.cc,%.o, $(CCFILES6))

OBJECTSN=$(patsubst %.cc,%.o, $(CCFILESN))

#Rules ------------------
# see https://www.gnu.org/software/make/manual/make.html#Pattern-Intro
# see https://www.gnu.org/software/make/manual/make.html#Automatic-Variables
%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -Wno-unused-function -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP).x $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g.x $(LIBGFLAGS)

$(APP2): $(OBJECTS2) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS2) -o $(APP2).x $(LIBFLAGS)

#$(APP3): $(OBJECTS3) $(ITENSOR_LIBS)
#	$(CCCOM) $(CCFLAGS) $(OBJECTS3) -o $(APP3).x $(LIBFLAGS)

$(APP4): $(OBJECTS4) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS4) -o $(APP4).x $(LIBFLAGS)

$(APP4)-g: mkdebugdir $(GOBJECTS4) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS4) -o $(APP4)-g.x $(LIBGFLAGS)

$(APP5): $(OBJECTS5) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS5) -o $(APP5).x $(LIBFLAGS)

$(APP5)-g: mkdebugdir $(GOBJECTS5) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS5) -o $(APP5)-g.x $(LIBGFLAGS)

$(APP6): $(OBJECTS6) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS6) -o $(APP6).x $(LIBFLAGS)

test3x3: $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) test3x3.cc -o test3x3.x $(LIBFLAGS)

normalize-cls: $(OBJECTSN) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTSN) normalize-cls.cc -o normalize-cls.x $(LIBFLAGS)

test-lin-sys: $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) test-lin-sys.cc -o test-lin-sys.x $(LIBFLAGS)

test-lin-sys-g: mkdebugdir $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) test-lin-sys.cc -o test-lin-sys-g.x $(LIBGFLAGS)

test-mklsvd: $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) test-mklsvd.cc -o test-mklsvd.x $(LIBFLAGS)

test-mklsvd-g: mkdebugdir $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) test-mklsvd.cc -o test-mklsvd-g.x $(LIBGFLAGS)


clean:
	rm -fr .debug_objs *.o $(APP).x $(APP)-g.x

mkdebugdir:
	mkdir -p .debug_objs
