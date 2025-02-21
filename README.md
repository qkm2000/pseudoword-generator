# Pseudoword Generator

This project is meant to propose a model that takes in a roundness value and generates a pseudoword. This project will hopefully be used in a larger pipeline that will help to explore the bouba-kiki effect.

## Glossary

| word             | meaning                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data model       | the model that was trained on normalized.csv to predict the roundness values of pseudowords. this model is used to help generate more data for the generation model |
| generation model | the model that is trained on the data from the data model. this generation model is used to generate pseudowords                                                    |

## Changelog

### 1.0

- inital commit of a working model
- byt5 + mlp used to generate data
- byt5 used for tokenizer and generation model
- results show that the generated peudowords do not really match the roundness values provided
- this version was pretrained on English characters and words. this meant that there was a lot of possible syllables, and it is difficult to control how we want to generate the words. the decision was made to change to japanese syllables because of the fixed lexicon and structure, which will lead to more effective pseudoword generation

### 2.0

- implemented the change from english-based models to japanese-based models
- used kfold training for the data model. the low number of training rows makes kfold much more effective than train-val-test
- used a custom tokenizer (SyllableTokenizer) and used sonoisa/t5-base-japanese as the base generation model instead of ByT5
- the generated words are more similar to the roundness values, but generally still seems quite random
- validation loss increased from 0.7+ to 3.3775, and test loss was 3.4649

### 2.1

- fixed the kfold training from [2.0](#20) for the data model
- increased number of layers for the data model
- gradient clipping introduced
- generation model val loss dropped to 3.1+
- generated words are also very much more similar to the expected words
- generated words are more representative of the roundness values, but larger roundness values tend to provide shorter words and vice versa
- validation loss was 3.1856, while test loss was 3.2194

### 3.0

- implemented the option to use BERT/roBERTa embeddings in the data model
- used BERT + kfold training in this version
- used sonoisa/t5-base-japanese tokenizer instead of custom tokenizer for better results
- validation loss was further decreased to 2.5548, test loss is at 2.5307
- the results from the data generation seemed much better
- this translated to the results of the pseudoword generator being better too

### 3.1

- used roBERTa in the data model
- the rest is the same as [3.0](#30)
- note that the predicted roundness max and min are relatively close together (0.38 to 0.7), which may affect the training of the pseudoword generation
- validation loss was further reduced to 2.4612 while test loss was at 2.5689
- the generated results seemed worse than [3.0](#30)

### 3.2

- used the data model from [3.0](#30)
- change batchnorm to layernorm
- change xavier to kaiming initialization
- enabled t5 gradient checkpointing
- included attention mask in training generation
- training used 1000 batch size instead of 100 for all previous experiments
- the model trained for 85 epochs with an early stopping patience of 10
- validation loss is higher, at 2.6275
- the generated pseudowords are of higher quality. the words are more consistent in length, and seem to be even more in tune with the roundness. however, there are notable exceptions: 0.1 roundness -> wamu

### 3.3

- changed to a simpler transformer model for generation
- created a custom tokenizer that takes only English characters and special tokens (total 29 tokens)
- used the normalized.csv dataset, but the train is from [0:100], val is [100:110], tst is [110:124]
- removed the need to have a data model since this is not japanese-based
- gridsearch for best hyperparams, using best val loss as the metric
- generated words are of quite high quality, and tend to match the roundness very well
- the generated words tend to be quite repeated. suspect it's because of the lack of training data or maybe overfitting
- the best parameters are: 'd_model': 128, 'nhead': 8, 'num_layers': 8, 'learning_rate': 0.1, 'weight_decay': 0.01, 'batch_size': 8, 'max_length': 16

### 3.4

- changed the dataset creation method. it used to be linear sampling, now is random sampling
- gridsearch for best hyperparams, using test loss as the metric to have the most generalisable model
- generated words are also of high quality, but interestingly do not have the character `k` much, and it seems to have taken `t` as the sharpest character
- similar to [3.3](#33), the words tend to be quite repeated
- the best parameters are: 'd_model': 128, 'nhead': 16, 'num_layers': 8, 'learning_rate': 0.1, 'weight_decay': 0.01, 'batch_size': 8, 'max_length': 16

### 4.0

- included a simple generator that uses arbitrary sound mappings and a rule-based system to generate pseudowords
- it selects characters based on the sound mapping, and constructs the pseudowords
- it also smooths out the character selection via interpolation
- the results look quite good, with the generated words matching what people generally expect
- however, because the dataset is arbitrarily selected, there is inherent bias introduced

### 4.0.1

- created a new dataset
  - this dataset was created by adding the roundness scores of all characters in every word, calculating the number of occurrences of each character in the whole dataset, obtaining the average roundness score of each character, then normalizing it
- results are better than the original 4.0, but it still feels like the generated words do not match the input values well

### 4.1

- dict-choice generator
- this generator uses consonants and vowels from [this paper](https://arxiv.org/html/2310.16781v2)
- the paper provides 12 consonants and 5 vowels for each experiment
- the 12 vowels are divided into 3 sections here; first 4, next 4, and last 4
- the vowels are divided into 5 ranges, so that there is 1 main vowel in each range, then the vowel from the previous range and the next range is added in
  - for example, `0.0: 'i'`, `0.25: 'e'`, `0.50: 'a'`
  - so the category for `0.25` will contain `'i', 'e', 'a'`
- the syllable structures are defined in the code as well
- generation
  - input value is used to choose which category the consonants, vowels, and word structure
  - then, the consonants and vowels are randomly picked to match the word structure
- the results are not great, and there are 2 suspected reasons:
  1. there is inherent bias in the categories because the data is obtained from another model
  2. the categories are rather large, and this may lead to loss in nuance. an input value between 0.25 to 0.75 could use the same set of consonants to generate a word

### 4.2

- index-choice generator
- similar to [4.1](#41), but instead of dicts with fixed roundness values, the characters are now in a list and are selected based on their index
- the results are better than [4.1](#41), but the problem still exists where there in inherent bias in the dataset because of CLIP and SD's biases