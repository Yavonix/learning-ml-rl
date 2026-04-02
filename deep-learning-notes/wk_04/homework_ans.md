## Part 1 T/F
1. F. Where stride $\ne 1$, output dimensions will be different.
2. T
3. F. Increasing the stride will decease spatial resolution of the resulting output feature map. Increasing the stride in a transposed convolution will increase the output resolution.
4. T. The kernel depth is equal to the channels of the image.

## Part 2 Multiple Choice

5.
    Formula is (h + p(total)) - k + 1.

    32 + 4 - 5 + 1 = 32.

    Therefore output dims are 32x32. C.

6. C. To reduce spatial dimension, decrease computational load and provide slight translation invariance.

7. 
    10 [kernels] * (3 * 3) [kernel size] * 3 [channels] = 270 learnable parameters. There are no biases in a convolutional layer. B.
   
   > ❌ There are biases in a conv layer by default. Should be 280, C. 

8.
    FLAX/NNX is NHWC
    Pytorch is NCHW

    Therefore A.

9. A

10. A

    > ❌ B. Apparently pytorch restricts stride = 1 while FLAX NNX does not.

11. C

12. D

13. C

14. B

15. B

16. D

17. B

    > Ans say C but I'm pretty sure they are incorrect.

## Part 3 Short Answer & Calculation

8.
    $P_{total}$ = k - 1 = 4 - 1 = 3

    $$
    \begin{align}
        \lfloor \frac{(h_{in}+2p)-k+s}{s} \rfloor &=  h_{out} \\
        \lfloor \frac{(64+2p)-4+2}{2} \rfloor &=  32 \\
        \lfloor \frac{62 + 2p}{2} \rfloor &=  32 \\
        62 + 2p &=  64, 65 \\
        2p &= 2, 3 \\
        p = 1,1.5
    \end{align}
    $$

9. Receptive field refers to the pixels of the original input prior to any convolution operations that have contributed to the output of the current convolutional operation. The receptive field of the first layer is 3x3. The receptive field of the second layer is 3x3 with respect to the first layer and 5x5 with respect to the original input.

10.  Max pooling takes the max value of a specific area while average pooling takes the average value of a specific area.