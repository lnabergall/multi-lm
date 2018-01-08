# MultiLM

MultiLM is a project exploring the possibility of learning multilingual and multi-domain language models using the simplest possible modification of a standard neural language model. 

Inspired by [Google's Multilingual Neural Machine Translation System](https://arxiv.org/abs/1611.04558) [1], we aim to learn general multilingual, multi-domain language models, MultiLMs, by training neural models on natural language data with text classification tags appended to the beginning of all sequences. For example, instead of the following sentence from the [1 Billion Word Benchmark](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark),

```
The findings were published in the journal Probation Journal.
```

the model is trained on

```
natural language <tag> english <tag> news <tag> <tag> The findings were published in the journal Probation Journal.
```

At inference time, we generate natural language text in English in the news domain by conditioning the model on a desired sequence of tags, e.g. ```natural language <tag> english <tag> news <tag> <tag>```. 

### Architecture

In Google's Multilingual Neural Machine Translation System (GMNMT), they successfully converted their single language-pair [neural machine translation system](https://arxiv.org/pdf/1609.08144.pdf) into a system capable of processing input data in multiple source languages and outputting translations in multiple target languages by adding an artificial token at the beginning of each input sentence to indicate the target language the model should translate the sentence to. For example, 

```
Hello, how are you? -> Hola, ¿cómo estás?
```

becomes 

```
<2es> Hello, how are you? -> Hola, ¿cómo estás?
```

where ```<2es>``` is the artificial token indicating the model should translate the sentence to Spanish. 

We propose a similar strategy to learn multilingual, multi-domain neural language models, where a sequence of tokens is added to the beginning of any input sequence to indicate the language type (e.g. natural language, programming language, scripting language, etc.), language (e.g. English, German, Chinese, etc.),  domain (e.g. news, blog, review, academic writing, novel, play, etc.), or other attributes of the text that the model should generate. 

The use of artificial tokens in the GMNMT system limits the model to only translating sentences into target languages it has been trained on. While this works for machine translation since there are only a limited number of languages and a model will need to be exposed to the target (and source) language during training anyways, we may more effectively train flexible MultiLMs by using tokens within the model's (unaugmented) vocabulary, that is, by using English words (e.g. natural language, spanish, fiction, speech, etc.) to denote text attributes, rather than artificial tokens. 

To distinguish the sequence of text attribute tokens from the rest of the input data, we also added an artificial 'tag' token not present in the original vocabulary, represented by ```<tag>```, between the text attribute tokens, as well as an extra one to indiciate the end of the attribute sequence. The following are examples of the resulting sequence of tokens appended to input data: ```natural language <tag> english <tag> news <tag> <tag>``` and ```programming language <tag> python <tag> tensorflow <tag> <tag>```. 

To limit the vocabulary for computational efficiency while maintaining (relatively) short input sequences during training, we use the same shared wordpiece model used by the GMNT system [2].

#### Models

(description coming soon)

### Data

(description coming soon)

### Training

We trained very small and medium-sized Attention LMs and LSTM LMs with the following hyperparameter configurations: 

|                  | hidden dim | filter size | batch size (tokens) | hidden layers | parameters (millions) |
|------------------|------------|-------------|---------------------|---------------|-----------------------|
| Attention Tiny   | 288        | 1024        | 2048                | 1             | 6.9                   |
| Attention Medium | 768        | 3072        | 6144                | 5             | 51.3                  |
| LSTM Tiny        | 384        | N/A         | 2048                | 1             | 9.1                   |
| LSTM Medium      | 1024       | N/A         | 6144                | 2             | 38.0                  |

These were chosen to test different model capacities while staying within computing time and memory constraints; we were limited to training the models on a single Nvidia GeForce GTX 1070. Note that the number of parameters may vary a small amount depending on the training dataset (since the wordpiece vocabulary varies between datasets).  

We used [TensorFlow](https://www.tensorflow.org/) 1.3 and [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/) 1.3.0 to implement, train, and evaluate our models; note that the implementation of the Attention model can be found in the Tensor2Tensor source code. An exhuastive list of other model and training hyperparameters can be found in the source code files of this repository and Tensor2Tensor. 

Briefly, we used the Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9 and beta1 = 0.85, beta2 = 0.997, and epsilon = 1e-6 for Attention and LSTM models, respectively. We clipped all gradient norms to 5.0 while training LSTM models and used no gradient clipping while training Attention models. We also used the "warm-up" schedule described in [3] to gradually increase and then decrease the learning rate during training of Attention models; for LSTM models, the learning rate was set to 0.1.

### Results

Note that due to (temporary) limitations in the Tensor2Tensor library, the following results are based on evaluating the models on the held-out validation dataset with teacher-forcing enabled. Furthermore, the evaluation procedure does not yet take into account the initial sequence of text attribute tags; as such, these values are not yet comparable to published results. 

(coming soon)

#### References

[[1]](https://arxiv.org/abs/1611.04558) Johnson et al. Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. *arXiv preprint arXiv:1611.04558* (2017).

[[2]](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) Schuster, M., and Nakajima, K. Japanese and Korean voice search. *2012 IEEE International Conference on Acoustics, Speech and Signal Processing* (2012).

[[3]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) Vaswani et al. Attention Is All You Need. *Advances in Neural Information Processing Systems 30* (2017) 6000--6010. 


