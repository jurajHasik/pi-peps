.. install

Installation guide
==================
In order to compile and install pi-peps, the following libraries are required:
 - ITensor_ v2, a tensor network library 
 - linear algebra library, which can be one of the two (not both):

  - Intel MKL
  - LAPACK

 - meson_ & ninja_, the build system tools
 - ARPACK_ (optional) for large-scale eigenvalue problems

.. _ITensor: https://github.com/ITensor/ITensor/tree/v2
.. _meson: https://mesonbuild.com
.. _ninja: https://ninja-build.org
.. _ARPACK: https://github.com/opencollab/arpack-ng

**Handling prerequisites**

After installing ITensor, you can get the remaining dependencies easily
through `conda`_ package manager. After setting up `conda`, simply get both `meson`
and `ninja` as

.. _`conda`: https://docs.conda.io/en/latest/miniconda.html

.. code-block:: bash

        $ conda install meson ninja

To install Intel MKL and corresponding headers through `conda`:

.. code-block:: bash

        $ conda install mkl mkl-include

The simplest way to get ARPACK is to use package manager of your choice (`apt-get`, ...),
for example we can again use `conda`:

.. code-block:: bash

        $ conda install -c conda-forge arpack

**Building the library**

First, clone the repository and enter inside the folder

.. code-block:: bash
		
		$ git clone https://github.com/jurajHasik/pi-peps.git
		$ cd pi-peps

Then invoke `meson` passing the necessary build options through the flag `-D` 
together with the folder **where the library will be compiled**. In particular,
the path to the `ITensor` must be set via the option `itensor-dir`:
	
.. code-block:: bash
	
		$ meson -Ditensor-dir=/path/to/itensor/root/dir build

Running this command, `meson` will:
 - generate all the files needed to compile the library inside the folder **build** (if 
   it does not exist, meson will create it)
 - look for `ITensor` library in the folder `/path/to/itensor/root/dir`.

If the command is successfull, we can enter the `build` directory and compile the library
using `ninja`
  
.. code-block:: bash

                $ cd build
                $ ninja

Note that in the above example, `lapack` with `cblas` is used as default. If you want to use Intel MKL instead, see below.

Configuring pi-peps with Intel MKL
----------------------------------
First, build the ITensor library with Intel MKL. Afterwards, you can build pi-peps:

.. code-block:: bash
		
		$ meson -Ditensor-dir=/path/to/ITensor -Dmkl=true -Dmkl-dir=/path/to/mkl/root/dir  build-mkl

If you installed both Intel MKL and ARPACK through `conda`, an examplary configuration could be 
done as follows:

.. code-block:: bash
		
		$ meson -Ditensor-dir=/path/to/ITensor -Dmkl=true -Dmkl-dir=$CONDA_PREFIX -Darpack=true -Darpack-dir=$CONDA_PREFIX -Drsvd=true build-mkl

where both ARPACK and randomized SVD functionality is enabled. The environment variable `CONDA_PREFIX` (automatically set by `conda`) points to the directory containing libraries and headers. 

Changing the options of a build
-------------------------------
If you change your mind after you have already run `meson` and populated a build directory, in order to change any of the build options you have to

 - be inside the **build** dir
 - use `meson configure -Doption=value`
 - call `ninja` to recompile the library with the new configuration

   
Install the library
-------------------
To start experimenting with the library, we recommend to install the library itself and continue modifying one of the installed examples. 
The library can be installed via the command
 
 - :code:`ninja install` (or :code:`sudo ninja install`)
  
`meson` will install the library in the path stored in the option `prefix` (which sould be `/usr/local` by default). You can check the values of all the available options with
 
 - :code:`meson configure` issued from inside the build dir

We highly recommend to choose a prefix where you have write permissions (e.g., `~/pi-peps`) so you don't need the root privileges. You can change the prefix with `-Dprefix=/path/to/install/lib`

Other options
-------------

For the set of options specific to this project, please refer to the file `meson_options.txt`. 


