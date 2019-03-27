.. install

Installation guide
==================

In order to compile and install pi-peps, the following libraries are required
 - itensor
 - linear algebra library, which can be one of the two (not both)
   - mkl
   - lapack and cblas
 - arpack (at the moment is mandatory, but Juraj will fix this soon)
 - meson (the build system tool)
 - ninja (that is required by meson)
   
First step (if not done yet): clone the repository

.. code-block:: bash
		
		$ git clone https://github.com/jurajHasik/pi-peps.git

Then, `cd` in it

.. code-block:: bash
		
		$ cd pi-peps

Now we can invoke `meson` specifying some build options (e.g., the path to itensor, the prefix) and the name of the build directory (e.g., build)
	
.. code-block:: bash
	
		$ meson -Ditensor-dir=/path/to/itensor/root/dir -Dprefix=/path/to/installdir/for/pi-peps build
		$ cd build
		$ ninja
		$ ninja install

We highly recommend to choose a prefix where you have write permissions (e.g., `~/pi-peps`) and you don't need to prepend `sudo` to the last command. Once installed, you can explore the **example** folder. Note that in the above example, `lapack` with `cblas` is used as default. If you want to use **mkl** instead, see below.

Configuring pi-peps with mkl
============================
.. code-block:: bash
		
		$ meson -Ditensor-dir=/what/ever -Dmkl=true -Dmkl-dir=/path/to/MKLROOT

For the set of available options, please refer to the file `meson_options.txt`

