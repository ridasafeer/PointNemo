How to run C++ model? 

# 1) Configure 
cmake -S . -B build -A x64

# 2) Build the Release config (this creates build\Release\)
cmake --build build --config Release

# 3) Run it
.\build\Release\pointnemo_rt.exe
