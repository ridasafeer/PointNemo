#include <vector>
#include <algorithm>
#include <numeric>

class FxLMS {
public:
    // Constructor
    FxLMS(int L, const std::vector<float>& shat, float mu);

    // Push new reference sample x(n)
    void push_x(float x);

    // Compute controller output y(n)
    float output() const;

    // Compute filtered-x sample x_f(n)
    float filtered_x_sample() const;

    // Push filtered-x into buffer
    void push_xf(float xf);

    // LMS weight update
    void update(float e);

private:
    int L;                  // Adaptive filter length
    int M;                  // Secondary path length
    float mu;               // Step size

    std::vector<float> w;   // Adaptive filter weights
    std::vector<float> shat;// Secondary-path estimate
    std::vector<float> xbuf;// Reference signal buffer
    std::vector<float> xfbuf;// Filtered-x buffer
};
