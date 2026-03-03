#include <vector>
#include <algorithm>
#include <numeric>

class FxLMS {
public:
    // Constructor
    //shat is const because it iwll not be mutable during the program
    FxLMS(const std::vector<float>& shat, int L, float mu);

    // Push new reference sample x(n), from the ISR buffer xbuf (should not be changed inside here)
    void push_x(const std::vector<float>& xbuf);

    // Compute controller output y(n)
    float output() const;

    // Compute filtered-x sample x_f(n) using x from this class (NOT ISR x)
    float filtered_x_sample() const;

    // Push filtered-x into buffer
    void push_xf(float xf);

    // LMS weight update
    void update(float e);

    void updateShat();

private:
    int L;                  // Adaptive filter length
    int M;                  // Secondary path length
    float mu;               // Step size

    const std::vector<float> shat;// Secondary-path estimate
    std::vector<float> w;   // Adaptive filter weights
    std::vector<float> &x;// Reference signal
    std::vector<float> xf;// Filtered-x
};
