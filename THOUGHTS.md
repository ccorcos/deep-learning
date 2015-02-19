# Research Review

Neural Networks (NNs) have been studied for decades. But it wasn't unitl 1986 when an efficient method for training them was discovered called [backpropagation][1]. The idea behind it is simple and intuitive - use the chain rule to propagate error derivatives backwards through the network.

A multilayer perceptron (MLP) with one hidden layer has been proven to be a [universal approximator][2]. That means a MLP can represent any arbitrary function if the hidden layer has a enough units. However, it is extremely challenging to learn a set parameters of parameters that generalizes well. Thus, researchers resorted to designing NN architectures that are more specific to certain problems. 

The first big success for NNs was the Convolutional Neural Network (CNN). It was designed to be used by the US Postal service for [zipcode recognition][3]. CNNs are deep NNs, meaning they have more than one hidden layer. They also involve shared weights that are convolved against the previous layer. These networks have been [wildly sucessful for image recognition][4] by [heirarchically learning and composing low-level features into successively higher-level features][5].


Deep neural networks still out of reach. Problem with the gradients. Autoencoders. Then Momentum. RMSProp, etc.

RNNs for NLP. Vanishing gradients. LSTM for long term dependancies.

Text Embedding.



---

NNs are... They have certain properties... 

In 2006... autoencoders... pretraining... newer methods that don't need pretraining but the concept is still the same. 

Can we bring this concept to RNNs for robotics?

Generalization - NLP embedding




--- 

"Unsupervised State Estimation using Deep Learning for Interactive Object Recognition"

Outline:

Deep learning:
 - unsupervised learning
 - autoencoders for pretraining
 - recurrent neural networks
 - recurrent autoencoder for ASR

Goal:
 - unsupervised learning for state estimation
 - pretraining for supervising learning on hidden state

Models:
 - RNN for unsupervised
 - ARNN for unsupervised

train to predict the next observation. using the hidden state varaibles, supervised predict the die from h_t. For each action, given what we're expected to see next, compute the likelihood for each die. Take the action that leads to the minimum entropy over these guesses. This is the optimal.

Likely overfit. User regularizatoin and dropout.

 - RNN for supervised
 - ARNN for supervised

Predict the object likelihood directly as opposed to this unsupervised middleman. performance?

Other Problems:

Try to use this model on LiDaR SLAM to predict the room and navigate between rooms.


<!--References-->
[1]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=JicYPdAAAAAJ&citation_for_view=JicYPdAAAAAJ:GFxP56DSvIMC
[2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.2647&rep=rep1&type=pdf
[3]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u-x6o8ySG0sC
[4]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
[5]: http://ftp.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

[topo]: http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/

























how could we use embedding to learn the 3d generalization of actions?
you cant quite do it in this case because all the labels are entirely interchangeable based on a new experience. So what if every experience, we check to see if we can predict correctly, otherwise we train up a new RNN. Can we prove that this can generalize 3d geometries? Use embedding to give EVERY example it own unique sets of observables and use embedding to get that down to a reasonable dimension!

For one trial we have a 3 by 6 matrix to project into 3D. We run it through the USE. the predictions use the projection matrix transposed. Thus we have an "internal" model that always stays the same between trials. And for each trial, we need to learn its own projections into the internal model. Thus fitting experiences to the internal "notion" fo reality.

Thus we can compare experiences using these projections and learn to predict which die / experience we are closest to. My fear is that this will not learn a multimodal distribution of preditions when at the beginning of a new trial. So maybe we must enforce that 

Now using the 2D SLAM exmaple, we can focus in on a different part of the problem. For 2D slam, the observations are always of the same nature, thus we dont need to embed of observations for the reasons of the dice problem. But now we need to think about how we represent a multimodal distribution. The problem exists in both. How to we represent the fact that we may be in two places at once? I think the point of NNs is that we get this for free...

TODO: