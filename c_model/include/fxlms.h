#include <vector>
#include <algorithm>
#include <numeric>
#include <stdio.h>

class FxLMS {
public:
    // Constructor
    //shat is const because it iwll not be mutable during the program
    FxLMS(const std::vector<float>& shat, int L, float mu);

    // Compute filtered-x sample x_f(n) using x from this class (NOT ISR x)
    float filtered_x_sample();

    void push_reference_sample(float curr_sample);

    // LMS weight update
    void update(float e_n);

    void updateShat();

    float output(); 

    std::vector<float>& getXbuf(); //controller reads new x(n) samples
    std::vector<float>& getYbuf(); //controller reads new y(n) samples
    int getNumTaps();

private:
    int L;                  // Number of Taps (Adaptive filter length)
    int M;                  // Secondary path length
    float mu;               // Step size

    int x_tail = -1;              // circular buffer index head for current x(n) block
    int xf_tail = -1;            // circular buffer tail index for xf[n] signal
    int y_tail = -1;             // circular buffer tail index for y[n] signal
    
    const std::vector<float> shat; //est sec path 
    std::vector<float> w;   // Adaptive filter weights
    std::vector<float> x;// Reference signal: The true buffer
    std::vector<float> xf;// Filtered-x
    std::vector<float> y; //the current anti-noise output signal

};
