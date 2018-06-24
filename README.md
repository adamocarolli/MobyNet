# MobyNet

MobyNet is an image classifier that aims to identify an individual humpback whale by a picture of its fluke.

After centuries of intense whaling, recovering whale populations still have a hard time adapting to warming oceans and struggle to compete every day with the industrial fishing industry for food.

To aid whale conservation efforts, scientists use photo surveillance systems to monitor ocean activity. They use the shape of whales’ tails and unique markings found in footage to identify what species of whale they’re analyzing and meticulously log whale pod dynamics and movements. For the past 40 years, most of this work has been done manually by individual scientists, leaving a huge trove of data untapped and underutilized.

The idea for the project came from Kaggle's [Humpback Whale Identification Challenge](https://www.kaggle.com/c/whale-categorization-playground). The data and problem are provided by [Happy Whale](https://happywhale.com/).

## Acknowledgements

The approach used to solve this problem is based on research by Google Deepmind. The paper can be read here: [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf).

The original code for the project stems from [Mark Dong's](https://github.com/markdtw) implementation of [Matching Networks](https://github.com/markdtw/matching-networks). 

An additional approach considered for this problem is based on research by the University of Toronto. This can be found here [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).
