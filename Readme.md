CoQuEST: Entity-Focused Code-Mixed Question Generation for Entertainment Videos

This repository contains the code and resources for our paper, "CoQuEST: Entity-Focused Code-Mixed Question Generation for Entertainment Videos." The project introduces a novel framework for generating semantically rich, entity-centric, and information-driven questions in a code-mixed Hindi-English (Hinglish) format, focusing on multilingual and multicultural relevance.

Abstract
Earlier research on video-based question generation has primarily focused on generating questions about general objects and attributes, often neglecting the complexities of bilingual communication and entity-specific queries. This study addresses these limitations by developing a multimodal transformer framework capable of integrating video and textual inputs to generate semantically rich, entity-centric, and information-driven questions in a code-mixed Hindi-English format.

Such a system is particularly significant for multilingual societies, offering applications in bilingual education, interactive learning platforms, conversational agents, and promoting cultural and linguistic relevance. To the best of our knowledge, there does not exist any large-scale Hindi-English (Hinglish) code-mixed dataset for video-based question generation. To address this limitation, we curated a subset of the TVQA dataset and annotated it with bilingual experts, ensuring fluency, contextual appropriateness, and adherence to the code-mixed structure.

Empirical evaluation shows that CoQuEST demonstrated competitive performance with metrics of BLEU-1: 0.04, CIDEr: 0.29, METEOR: 0.20, Distinct-1: 0.96, Distinct-2: 0.99, ROUGE-L: 0.20, and BERT-Score F1: 0.88, validating its practical utility and effectiveness.

Repository Contents
Dataset Samples

A small sample of the MixTV-QA dataset, including a few annotated video clips and transcripts.

Model Architecture

Implementation of the CoQuEST framework with detailed documentation.

Fine-Tuning Script

Code to fine-tune the CoQuEST models on the provided dataset.

Inference Script

Script for generating code-mixed questions using fine-tuned CoQuEST models.


Dataset
The full MixTV-QA dataset will be released later. For now, a representative sample is provided in the repository. The dataset was curated from the TVQA dataset and annotated by bilingual experts to ensure fluency, contextual appropriateness, and adherence to a code-mixed structure.
