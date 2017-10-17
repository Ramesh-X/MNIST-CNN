# CNN with Inception Modules for MNIST dataset

Classification of MNIST digits by convolutional neural networks (CNN) with Dropout and Inception layers.
No preprocessing done.

The code is written using Tenserflow.

Used in Digit Recognizer competition on Kaggle https://www.kaggle.com/c/digit-recognizer

## Network architecture

| Layer Type| Kernel Size / Stride | Output Size | #1×1 | #3×3 reduce | #3×3 | #5×5 reduce | #5×5 | pool proj |
| :--:      | :--:                 | :--:        | :--:        | :--: | :--: | :--:        | :--: | :--: |
| Input     | -                    | 28x28x1     |    |   |   |   |   |   |
| Conv      | 5 x 5 / 1            | 28x28x64    |    |   |   |   |64 |   |
| Conv      | 3 x 3 / 1            | 28x28x128   |    |   |128|   |   |   |
| MaxPool   | 3 x 3 / 2            | 14x14x128   |    |   |   |   |   |   |
| Norm      |                      | 14x14x128   |    |   |   |   |   |   |
| Inception |                      | 14x14x128   |  8 |64 |96 |8  |16 |8  |
| Inception |                      | 14x14x256   |64  |96 |128|16 |32 |32 |
| Inception |                      | 14x14x512   |160 |112|224|24 |64 |64 |
| MaxPool   | 3 x 3 / 2            | 7x7x512     |    |   |   |   |   |   |
| Norm      |                      | 7x7x512     |    |   |   |   |   |   |
| Inception |                      | 7x7x512     |128 |128|256|32 |64 |64 |
| AvgPool   | 7 x 7 / 7            | 1x1x512     |    |   |   |   |   |   |
| Dropout   |                      | 1x1x512     |    |   |   |   |   |   |
| FC        |                      | 1x1x10      |    |   |   |   |   |   |
| Softmax   |                      | 1x1x10      |    |   |   |   |   |   |

Accuracy of the model: **0.99261** with my test set and **0.99428** in Kaggle competition.
