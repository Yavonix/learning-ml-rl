## Part 1

1. The "Bitter Lesson" in AI suggests that complex, hand-engineered approaches will 
always outperform simple algorithms in the long run

    False

2. When evaluating a language model, a lower perplexity score indicates better 
performance.

    Perplexity is the average cross-entropy across each timestep in a sequence. A lower score indicates better performance.

    True

3. BERT is fundamentally an autoregressive model capable of naturally generating text 
left-to-right.

    BERT stands for bidirectional encoder representations from transformers.

    BERT is trained to fill in masked tokens or classify two sentences as whether they are contiguous or unrelated.

    It is not fundamentally an autoregressive model.

    False

4. Modern pre-trained language models rely heavily on subword tokenisation instead of 
standard word-level embeddings like GloVe

    True

## Part 2

5. Which of the following best describes how BERT achieves bidirectionality?

    a. It concatenates two independent unidirectional LSTMs (forward and backward).\
    b. It uses a standard autoregressive objective but reads the text from right-to-left.\
    **c. It utilises the Transformer architecture and a Masked Language Modeling (MLM) \
    objective.** \
    d. It uses Byte Pair Encoding to reverse the order of characters

    c.

6. What is the primary purpose of the [CLS] token in BERT?

    a. To act as a boundary separating two different sentences.\
    b. To predict the masked words in the sequence.\
    **c. To represent the aggregated sequence-level information for classification tasks.**\
    d. To indicate the end of a generated sequence.

    c.

7. How does BERT handle extractive Question Answering tasks (like SQuAD)?

    a. It generates the answer sequence left-to-right.\
    **b. It predicts the start and end indices of the answer span within the provided passage.**\
    c. It classifies the entire passage as either 'True' or 'False'.\
    d. It retrieves the answer from an external knowledge base

    b.

8. Why did modern NLP models move away from strictly character-level models?

    a. Character-level models cannot handle out-of-vocabulary terms.\
    **b. They resulted in sequences that were too long, making computation extremely 
    expensive.**\
    c. They required massive softmax matrices that slowed down training.\
    d. They overfit easily on small datasets

    b.

## Part 3

9. Explain the mechanism of Byte Pair Encoding (BPE). How does it construct its
vocabulary?

    BPE splits the input sequence into its individual characters, then finds the most frequent pair and concatenates it into a new symbol.

    Better answer: ```BPE starts by treating every individual character/byte as its own symbol. It counts the co-
occurrences of adjacent symbols in the training corpus and iteratively merges
the most frequent pair into a single new subword token. It repeats this process for a specified
number of merges (e.g., 8000), effectively creating a vocabulary of frequent whole words and
common word parts```

10. Contrast the bidirectionality of ELMO with the bidirectionality of BERT. Why is
BERT's approach considered "deeply bidirectional”?

    ELMO uses two LSTMs that independently read the input sequence then whose output is concatenated.

    BERT uses a transformer architecture where every token can attend to every other token in the sequence simultaneously regardless of direction.