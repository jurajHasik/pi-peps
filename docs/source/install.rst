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

Then invoke `meson` specifying some build options trough the flag `-D`
and a folder **where the library will be compiled**. In particular,
the path where `itensor` is compiled must be set via the option
`itensor-dir`. 
	
.. code-block:: bash
	
		$ meson -Ditensor-dir=/path/to/itensor/root/dir build

Thrhough this command we ask meson:
 - generate all the files needed to compile our library in the folder **build** (if it does not exist, meson will create it)
 - look for `itensor` library in the folder `/path/to/itensor/root/dir`.

If the command is successfull, we can enter the `build` directory and compile the library
  
.. code-block:: bash
		$ cd build
		$ ninja

Note that in the above example, `lapack` with `cblas` is used as default. If you want to use **mkl** instead, see below.

Configuring pi-peps with mkl
============================
.. code-block:: bash
		
		$ meson -Ditensor-dir=/what/ever -Dmkl=true -Dmkl-dir=/path/to/mkl/root/dir  build-mkl


Changing the options once meson already run
===========================================

If you change your mind after you have already run meson and populated a build directory, in order to change a build option you have to
 - be inside the **build** dir
 - use `meson configure -Doption=value`
 - call `ninja` to recompile the library with the new configuration

   
Install the library
===========================================
If you want to start using the library for your own needs, we recommend to install the library itself and start to modify one of the installed examples.

The library can be installed via the command
- `ninja install` (or `sudo ninja install`)
  
meson will install the library in the path stored in the option `prefix` (which sould be `/usr/local` by default). You can check the values of all the available options with
 - `meson configure` issued from inside the build dir

We highly recommend to choose a prefix where you have write permissions (e.g., `~/pi-peps`) and you don't need to prepend `sudo` to the install target. You can change the prefix with `-Dprefix=/path/to/install/lib`

Other options
=============

For the set of options specific to this project, please refer to the file `meson_options.txt`. 


