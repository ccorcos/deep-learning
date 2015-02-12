# Theano Notes

## Debugging
[This is a good resource](http://deeplearning.net/software/theano/tutorial/debug_faq.html).

Print out `.type` of a theano variable to get information about the tensor type.

Also, try running your program with

    THEANO_FLAGS="optimizer=None" python program.py

This will give you line numbers and more information.


Also, for any symbolic variables defined, it helps to give them test values which can be used
to test the functionality of the program as it goes so you can get a line number when it happens:

    x = T.matrix()
    x.tag.test_value = numpy.random.rand(10, 20)

Then make sure you set the flag when you run it.

    THEANO_FLAGS="optimizer=None,compute_test_value=raise" python program.py

# Parallelization

On Mac, you can use the GPU if you have a newer machine with an NVIDIA graphics card. 

Theano uses the OS X Accelerate framework for BLAS and other optimizations.