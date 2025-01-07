# Changelog

## Glossary

| word             | meaning                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data model       | the model that was trained on normalized.csv to predict the roundness values of pseudowords. this model is used to help generate more data for the generation model |
| generation model | the model that is trained on the data from the data model. this generation model is used to generate pseudowords                                                    |

## 1.0

- inital commit of a working model
- byt5 + mlp used to generate data
- byt5 used for tokenizer and generation model
- results show that the generated peudowords do not really match the roundness values provided
- this version was pretrained on English characters and words. this meant that there was a lot of possible syllables, and it is difficult to control how we want to generate the words. the decision was made to change to japanese syllables because of the fixed lexicon and structure, which will lead to more effective pseudoword generation

## 2.0

- implemented the change from english-based models to japanese-based models
- used kfold training for the data model. the low number of training rows makes kfold much more effective than train-val-test
- used a custom tokenizer (SyllableTokenizer) and used sonoisa/t5-base-japanese as the base generation model instead of ByT5
- the generated words are more similar to the roundness values, but generally still seems quite random
- validation loss increased from 0.7+ to 3.3775, and test loss was 3.4649

## 2.1

- fixed the kfold training from [2.0](#20) for the data model
- increased number of layers for the data model
- gradient clipping introduced
- generation model val loss dropped to 3.1+
- generated words are also very much more similar to the expected words
- generated words are more representative of the roundness values, but larger roundness values tend to provide shorter words and vice versa
- validation loss was 3.1856, while test loss was 3.2194

## 3.0

- implemented the option to use BERT/roBERTa embeddings in the data model
- used BERT + kfold training in this version
- used sonoisa/t5-base-japanese tokenizer instead of custom tokenizer for better results
- validation loss was further decreased to 2.5548, test loss is at 2.5307
- the results from the data generation seemed much better
- this translated to the results of the pseudoword generator being better too

## 3.1

- used roBERTa in the data model
- the rest is the same as [3.0](#30)
- note that the predicted roundness max and min are relatively close together (0.38 to 0.7), which may affect the training of the pseudoword generation
- validation loss was further reduced to 2.4612 while test loss was at 2.5689
- the generated results seemed worse than [3.0](#30)
