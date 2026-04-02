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
    void output(int startIndex);

    // Compute filtered-x sample x_f(n) using x from this class (NOT ISR x)
    float filtered_x_sample() const;

    // Push filtered-x into buffer for learning
    // Maintains the history of filtered reference samples and keeps reference signal as vector used for weight update
    void push_xf_learning();

    // LMS weight update
    void update(float e_n);

    void updateShat();

    float output_test(int startIndex); 

    std::vector<float>& getXbuf(); //controller reads new x(n) samples
    std::vector<float>& getYbuf(); //controller reads new y(n) samples
    int getNumTaps();

private:
    int L;                  // Number of Taps (Adaptive filter length)
    int M;                  // Secondary path length
    float mu;               // Step size
    int head;               // circular buffer index head for current x(n) block
    
    const std::vector<float> shat; //est sec path 
    std::vector<float> w;   // Adaptive filter weights
    std::vector<float> x;// Reference signal: The true buffer
    std::vector<float> xf;// Filtered-x
    std::vector<float> y; //the current anti-noise output signal

};
