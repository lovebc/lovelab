# Heading CSML project offerings

Lovelab (bradlove.org), A computational cognitive science lab directed by Prof Brad Love.


We are looking for enthusiastic and motivated students who want to collaborate with Prof Brad Love and members of his lab. We aim to improve performance of neural network models and make them more human-like by using insights form neuroscience. Students with approrpriate skills with a strong interest in training and evaluating convolutional neural networks should contact Prof Love, b.love@ucl.ac.uk.


1. Similarity functions for deep learning

Node activations in a deep learning model depend on the dot product of the connection weights with incoming inputs from a previous layer. If the dot product surpasses a threshold parameter for that node (i.e., above zero in the case of rectified linear units, ReLUs), then the node is activated and sends its outputs to the next layer. The dot product can be interpreted as a similarity function between the connection weights and inputs. Much attention has been put on the form of the activation functions (e.g., ReLUs, sigmoid, hyperbolic tangent, or radial basis functions), but little work has been done on the form of similarity functions that can be used in these models. Enforcing different similarity functions (e.g., Minkowski, Mahalanobis, etc.) has the potential to ease training and provide insights into brain-like computations. This project relates to previous work in our lab based on testing various similarity functions for functional magnetic resonance imaging (fMRI) data.

2. Hierarchical image classification with neural networks

Labelled images, such as of dogs and cats, constitute categories at different levels; for example, a chihuahua is a dog and a mammal. Many of the state-of-the-art image classification networks are trained on labels at a single level with one-hot coding. In this project, we consider cross-category and hierarchical category relations while training networks. The hope is that generalisation performance of networks will improve by taking into account relations across classes. Moreover, error patterns should be more sensible (graceful degradation.)
