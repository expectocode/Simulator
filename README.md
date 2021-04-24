# Focal Plane Sensor Processor Simulator

A parameterisable simulation library to model and emulate different FPSP architectures. Currently only the SCAMP-5
architecture has been (mostly) implemented. The SCAMP-5 API documentation can be
found [here](https://personalpages.manchester.ac.uk/staff/jianing.chen/scamp5d_lib_doc_html/_p_a_g_e__d_e_v_i_c_e__a_p_i__c_a_t_e_g_o_r_y.html)
.

Requirements:

* C++
* Cmake
* Conan

Composed of two parts: the simulation library and the architecture you're simulating. The existing CMakeLists.txt will build the library first and then the SCAMP-5 architecture.

# Compile time options

There are various compile time options which conditionally compile code. The options can be toggled in the
CMakeLists.txt file or by passing in the option to CMake,
e.g. `cmake -DCMAKE_BUILD_TYPE=release .. -DTRACK_STATISTICS=1`.

Tracking statistics is expensive and will drastically slow down execution.

| Name | Description | Default |
| ------------- | ------------- | ------------- |
| USE_RUNTIME_CHECKS  | Checks arguments at runtime and emits warnings. | OFF |
| TRACK_STATISTICS | Tracks statistics such as register reads and energy use| OFF |

# Build - (release|debug)
(remmeber to remove the brackets around the choice before running)

* Clone repo
* `mkdir cmake-build-(release|debug) && cd cmake-build-(release|debug)`
* `conan profile update settings.compiler.libcxx=libstdc++11 default`
* `conan create ../conanfile.py -s build_type=(Release|Debug)`
* `conan install ../conanfile.txt -s build_type=(Release|Debug) --build missing`
* `cmake -DCMAKE_BUILD_TYPE=(release|debug) ..`
* `make -j`

# UI
There is a browser based UI which can be used to display the outputs of the registers in realtime. Create a UI object and start it before the main loop like this:
```
UI ui;
ui.start();
```
Then in the main loop, you can display a register by calling
```
ui.display_reg(REGISTER_HERE);
```
which will send the image to the web client. 

By default the client can be accessed at `localhost:3000`



# Tests

The test suite can be run with `make test` after the above build steps.

Alternative you can run the `tests` binary in the `bin` folder which has many more options. Run `./tests -?` to see
available options.

# Usage

The architectures/ directory and main.cpp shows how an architecture can be defined and programs can be run on that
architecture.
