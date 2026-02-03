

class DSP {
    public:
    DSP(); //definition using initializer class list in the cpp file

    void fir_block_processing(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, std::vector<float>& state);


    private:
    int num_taps; //L
};

