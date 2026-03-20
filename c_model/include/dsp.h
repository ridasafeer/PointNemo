

#include <iostream>
#include <vector>

class DSP {
    public:
    void impulseResponseLPF(float, float, unsigned short int, std::vector<float> &, int);
    void fir_block_processing(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, std::vector<float>& state);
    void bandPassCoeff(float, float, float, int, std::vector<float> &);
    void convolution_w_ds(std::vector<float> &h, std::vector<float> &block, std::vector<float> &state, std::vector<float> &sub_res, int ds);
    void resampling(std::vector<float> &, const std::vector<float> &, const std::vector<float> &, std::vector<float> &, int , int);
    float dot_product(const std::vector<float>& a, const std::vector<float>& b, int len);
    void fir_convolution(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, const std::vector<float>& state);
    private:
    int num_taps; //L
};

