#include <vector>
#include <algorithm>
#include <numeric>
#include <stdio.h>

class FxLMS {
public:
    // Constructor
    //shat is const because it iwll not be mutable during the program
    FxLMS(const std::vector<float>& shat, int L, float mu);

    // Compute controller output y(n)
    void output() const;

    // Compute filtered-x sample x_f(n) using x from this class (NOT ISR x)
    float filtered_x_sample() const;

    // Push filtered-x into buffer
    void push_xf();

    // LMS weight update
    void update(float e);

    void updateShat();

private:
    int L;                  // Adaptive filter length
    int M;                  // Secondary path length
    float mu;               // Step size

    std::vector<float> w;   // Adaptive filter weights
    const std::vector<float> shat;// Secondary-path estimate
    std::vector<float> x;// Reference signal: The true buffer
    std::vector<float> xf;// Filtered-x
    std::vector<float> y; //the current anti-noise output signal

};
