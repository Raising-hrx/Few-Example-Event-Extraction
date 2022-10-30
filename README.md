# Few-Example-Event-Extraction
The repo for ["Learning Event Extraction From a Few Guideline Examples"](https://ieeexplore.ieee.org/document/9868134)

> Abstract - Existing fully supervised event extraction models achieve advanced performance with large-scale labeled data. However, when new event types emerge and annotations are scarce, it is hard for the supervised models to master the new types with limited annotations. In contrast, humans can learnto understand new event types with only a few examples in the event extraction guideline. In this paper, we work on a challenging yet more realistic setting, the few-example event extraction. It requires models to learn event extraction with only a few sentences in guidelines as training data, so that we do not need to collect large-scale annotations each time when new event types emerge. As models tend to overfit when trained with only a few examples, we propose knowledge-guided data augmentation to generate valid and diverse sentences from the guideline examples. To help models better leverage the augmented data, we add a consistency regularization to guarantee consistent representations between the augmented sentences and the original ones. Experiments on the standard benchmark ACE-2005 indicate that our method can extract event triggers and arguments effectively with only a few guideline examples. 

# Data

**ACE-Guideline**: We collect 194 sentences from [Chapters 5 and 6 of the ACE event annotation guideline](https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/english-events-guidelines-v5.4.3.pdf) as a few guideline examples to train models.

- Training split: `data/ACE-guideline.train.json`

- Development split: `data/ACE-guideline.dev.json`


The data format are in the [OneIE](https://github.com/vinitrinh/event-extraction-oneie) data format.



# Citation
```
@article{DBLP:journals/taslp/HongZYZ22,
  author    = {Ruixin Hong and
               Hongming Zhang and
               Xintong Yu and
               Changshui Zhang},
  title     = {Learning Event Extraction From a Few Guideline Examples},
  journal   = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  volume    = {30},
  pages     = {2955--2967},
  year      = {2022},
  url       = {https://doi.org/10.1109/TASLP.2022.3202123},
  doi       = {10.1109/TASLP.2022.3202123},
  timestamp = {Tue, 18 Oct 2022 22:17:28 +0200},
  biburl    = {https://dblp.org/rec/journals/taslp/HongZYZ22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```