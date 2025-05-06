## Dataset Recommendation with Large Language Models

#### Supporting repository for the PIR research project at INSA Lyon, Spring 2025.

#### Supervisors: Malcolm Egan (Inria), Ioannis Chalkiadakis (CNRS)


A major challenge for many institutions and companies is finding data relevant for their tasks. To tackle this challenge, many countries are now making significant investments in large-scale data exchange platforms. Nevertheless, the problem of efficiently matching data with users at scale remains a difficult problem. In this project, we ask the following question: Can embeddings of metadata associated with datasets form a basis for efficiently matching tasks with data? In this approach, a task is given in natural language (e.g., *find a dataset to train a model to classify butterflies*). Dataset metadata may include the dataset attributes (type, short description, number of attributes etc), the full dataset card on the dataset host repository (e.g. HuggingFace Datasets, Harvard Dataverse, Figshare, OSF or other similar platforms), data descriptor papers, papers reporting on the dataset usage on tasks of interest, or model-based dataset evaluation scores. An encoder of a large language model (LLM) may then be utilized to obtain embeddings for both the task and the metadata of the datasets. The goal is to develop methods to identify the relevance of datasets for a task based on the embeddings.

Relevant readings:

1. Viswanathan, V., Gao, L., Wu, T., Liu, P. and Neubig, G., 2023. DataFinder: Scientific dataset recommendation from natural language descriptions. arXiv preprint arXiv:2305.16636.
2. Yu, Y. and Romero, D.M., 2024. Does the use of unusual combinations of datasets contribute to greater scientific impact?. Proceedings of the National Academy of Sciences, 121(41), p.e2402802121.
3. Irrera, O., Lissandrini, M., Dell'Aglio, D. and Silvello, G., 2024, October. Reproducibility and Analysis of Scientific Dataset Recommendation Methods. In Proceedings of the 18th ACM Conference on Recommender Systems (pp. 570-579).
4. Altaf, B., Akujuobi, U., Yu, L. and Zhang, X., 2019, November. Dataset recommendation via variational graph autoencoder. In 2019 IEEE International Conference on Data Mining (ICDM) (pp. 11-20). IEEE.
5. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.T., Rocktäschel, T. and Riedel, S., 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33, pp.9459-9474.
6. Chapman, A., Simperl, E., Koesten, L., Konstantinidis, G., Ibáñez, L.D., Kacprzak, E. and Groth, P., 2020. Dataset search: a survey. The VLDB Journal, 29(1), pp.251-272.
7. Achille, A., Lam, M., Tewari, R., Ravichandran, A., Maji, S., Fowlkes, C.C., Soatto, S. and Perona, P., 2019. Task2vec: Task embedding for meta-learning. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6430-6439).
8. Akhtar, M., Benjelloun, O., Conforti, C., Foschini, L., Giner-Miguelez, J., Gijsbers, P., Goswami, S., Jain, N., Karamousadakis, M., Kuchnik, M. and Krishna, S., 2024. Croissant: A metadata format for ml-ready datasets. Advances in Neural Information Processing Systems, 37, pp.82133-82148.
9. Alvarez-Melis, D. and Fusi, N., 2020. Geometric dataset distances via optimal transport. Advances in Neural Information Processing Systems, 33, pp.21428-21439.
10. [Boonstra, L., Prompt Engineering.](https://readwise-assets.s3.amazonaws.com/media/wisereads/articles/prompt-engineering/22365_3_Prompt-Engineering_v7-1.pdf)
