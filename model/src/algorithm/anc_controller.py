# main module for control of FxLMS algorithm: Follows the FxLMS control system
# - Takes inputs to create reference noise signal
# - Processes reference signal through adaptive filter
# - Outputs the anti-noise signal


import numpy as np

class FxLMS:
    """
    FxLMS controller with:
      - W: adaptive FIR of length L
      - Shat: secondary-path estimate FIR of length M
      - xbuf: holds last max(L, M) reference samples
      - xfbuf: holds last L filtered-x samples for the weight update
    """
    def __init__(self, L: int, shat: np.ndarray, mu: float): #required inputs when using module
        self.L = int(L)
        self.shat = np.asarray(shat, dtype=np.float32) #estimated secondary path's impulse response
        self.M = int(self.shat.size)
        self.mu = float(mu)

        self.w = np.zeros(self.L, dtype=np.float32)

        xb_len = max(self.L, self.M)
        self.xbuf = np.zeros(xb_len, dtype=np.float32)
        self.xfbuf = np.zeros(self.L, dtype=np.float32)

    #MEASURES REFERENCE SIGNAL x(n)
    def push_x(self, x: float) -> None: #inserts latest sample from reference signal
        self.xbuf[1:] = self.xbuf[:-1]
        self.xbuf[0] = x #latest sample, at index 0: xbuf[0] = x(n)

    #COMPUTES OUTPUT OF ADAPTIVE FILTER, ANTI-NOISE SIGNAL y(n)
    def output(self) -> float: 
        # y(n) = sum_{k=0}^{L-1} w[k] * x(n-k)
        return float(np.dot(self.w, self.xbuf[:self.L]))


    def filtered_x_sample(self) -> float:
        # x_f(n) = sum_{m=0}^{M-1} shat[m] * x(n-m)
        return float(np.dot(self.shat, self.xbuf[:self.M]))

    def push_xf(self, xf: float) -> None:
        self.xfbuf[1:] = self.xfbuf[:-1]
        self.xfbuf[0] = xf

    def update(self, e: float) -> None:
        # w <- w + mu * e * xfvec
        self.w += (self.mu * e) * self.xfbuf

# fx.push_x(x)          # update x(n) delay line
# y = fx.output()       # y(n) = w^T xvec
# play(y)               # goes through REAL S(z)
# xf = fx.filtered_x_sample()   # xf(n) = x * Shat
# fx.push_xf(xf)        # build xf vector
# fx.update(e)          # w <- w + mu * e * xfvec
