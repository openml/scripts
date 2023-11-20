# scripts

This repository contains individual scripts used to add various functionality to the OpenML platform. 

## generate_semantic_tags 

### generate_semantic_tag.ipynb

This script generates semantic tag(s) for each dataset present in OpenML.org.
For each dataset, the data description, along with a predefined list of tags is used to prompt GPT-3.5-turbo, which in turn assigns semantic tag(s) from the given list of tags. GPT is asked to assign one or two tags to each dataset. For datasets with no description present, GPT is not prompted, and a tag `["No description"]` is assigned, whereas if the response generated by GPT is `Null`, no tag (`[]`) is assigned. If the generated tag by GPT is not present in the predefined list, `spacy language model` is used to find the most semantically similar tag from the predefined list of tags.


The response from GPT is saved in data/GPT_semantic_tags.txt
