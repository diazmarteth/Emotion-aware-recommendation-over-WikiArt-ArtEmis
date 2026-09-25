This project develops a Multimodal/content-based Recommender over WikiArt ∩ ArtEmis

Dataset:
**WikiArt**: 80,020 unique images from 1,119 artists with 27 styles and 45 genres.
**ArtEmis**: 455K emotion explanations over 80K artworks. Explains choices across 9 emotional classes.
**User**: synthetic layer

Signal Types:
Elicited annotations
Implicit feedback (simulated, binary engagement)

Application Context:
Online art catalogues deployed in the GLAM sector, private art galleries, and commercial art trading.

**Track A: Personalised**
Logged-in visitor: Predicts what the user should see next using simulated synthetic history.
**Track B: Item-to-Item**
Anonymous browser: Recommends works similar to the current query painting for exploratory discovery.

link to the pitch: https://docs.google.com/presentation/d/14PKjLhWm83F1glAYzybL9DYeSjh3su6oGFNyodAnL4k/edit?slide=id.g409b5ac3769_18_0#slide=id.g409b5ac3769_18_0

The folder includes:
- Report --> Emotion-aware-recommendation_DIAZ.docx/.pdf
- Appendix with code results and comments --> Emotion-aware-recommendation_DIAZ_appendix.docx
- Jupyter notebook --> Emotion-aware-recommendation_DIAZ.ipynb
- Primary datasets --> classes.csv; artemis_dataset_release_v0.csv; merged_paintings.csv;
- Secondary dataset --> merge_artmeis_wikiart.py
- wikiart pictures --> wikiart_256 folder
- additional script to resize wikiart images --> resize_wikiart.py
- Real/source data:
    corpus.parquet (~8 MB)
    interactions.parquet —-> real user-item interactions
    merged_paintings.parquet —-> painting metadata, probably merged from multiple sources
    artemis_utterances.parquet —-> text utterances (likely from the ArtEmis dataset, which pairs artwork with human-written captions/reactions)
- Synthetic data (generated for testing/augmenting the recommender)
    synthetic_users.parquet
    synthetic_items.parquet
    synthetic_ratings.parquet / synthetic_ratings_unbiased.parquet
    synthetic_interactions.parquet / synthetic_interactions_unbiased.parquet / synthetic_interactions_unbiased_val.parquet (train/val split, "unbiased" variant)
