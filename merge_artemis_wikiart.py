"""Merge ARTEMIS affective annotations onto the resized WikiArt image set.

ARTEMIS references paintings as <art_style>/<painting>.jpg. Those strings and the
on-disk filenames carry *different* corruptions of the same non-ASCII names
(ARTEMIS: 'euga"ne-grasset'; disk: doubly-mangled UTF-8), and ARTEMIS keeps
apostrophes the archive drops. Folding both to ASCII reconciles them: 756 of the
757 non-exact keys resolve, with zero collisions.

Outputs (one row per painting / per utterance):
  merged_paintings.parquet   catalog + emotion distribution + utterance list
  merged_paintings.csv       same, utterances dropped (portable)
  artemis_utterances.parquet 454k utterances keyed to canonical file paths
"""
import re, sys
import pandas as pd
from pathlib import Path

IMG_DIR = Path("wikiart_256")
EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness", "something else"]


def norm(s: str) -> str:
    """Fold to ASCII skeleton: kills mojibake bytes and apostrophes alike."""
    return re.sub(r"[^a-z0-9_/.-]", "", s.lower())


def main():
    print("loading...", flush=True)
    art = pd.read_csv("artemis_dataset_release_v0.csv")
    art["key"] = art.art_style + "/" + art.painting + ".jpg"

    disk = sorted(str(p.relative_to(IMG_DIR)) for p in IMG_DIR.rglob("*.jpg"))
    dn = {}
    for d in disk:
        dn.setdefault(norm(d), d)

    # canonical on-disk path for every ARTEMIS row
    art["file"] = art.key.map(lambda k: dn.get(norm(k)))
    unmatched = art.file.isna()
    print(f"ARTEMIS rows            : {len(art):,}")
    print(f"  unmatched (dropped)   : {unmatched.sum():,} "
          f"({art.loc[unmatched,'key'].nunique()} paintings)")
    art = art[~unmatched].copy()

    # ---- per-painting emotion distribution -------------------------------
    counts = (art.groupby(["file", "emotion"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=EMOTIONS, fill_value=0))
    n_ann = counts.sum(axis=1)
    props = counts.div(n_ann, axis=0)
    props.columns = [f"emo_{c.replace(' ', '_')}" for c in props.columns]

    # entropy over the emotion distribution: how contested a painting's reading is
    import numpy as np
    p = props.to_numpy()
    safe = np.where(p > 0, p, 1.0)          # log(1)=0; masked out below anyway
    ent = -(np.where(p > 0, p * np.log(safe), 0.0)).sum(axis=1)

    agg = props.copy()
    agg["n_annotations"] = n_ann
    agg["dominant_emotion"] = counts.idxmax(axis=1)
    agg["emotion_entropy"] = ent
    agg["utterances"] = art.groupby("file").utterance.apply(list)
    agg = agg.reset_index()

    # ---- catalog side ----------------------------------------------------
    cat = pd.DataFrame({"file": disk})
    cat["style"] = cat.file.str.split("/").str[0]
    cat["_n"] = cat.file.map(norm)

    cls = pd.read_csv(IMG_DIR / "classes.csv")
    cls["_n"] = cls.filename.map(norm)
    cls = cls.drop_duplicates("_n")
    cat = cat.merge(
        cls[["_n", "artist", "genre", "description", "phash",
             "width", "height", "subset"]],
        on="_n", how="left")
    cat = cat.rename(columns={"description": "title",
                              "width": "orig_width", "height": "orig_height"})

    # artist fallback for rows classes.csv doesn't cover
    fb = cat.artist.isna()
    cat.loc[fb, "artist"] = (cat.loc[fb, "file"].str.split("/").str[1]
                             .str.split("_").str[0].str.replace("-", " "))
    cat = cat.drop(columns="_n")

    # Canonicalise artist identity. The same artist can reach us under two
    # spellings -- one mojibake from classes.csv, one from the differently
    # mangled filename ('joaqua-n sorolla' vs the doubly-mangled disk form).
    # Left split, same-artist relevance would treat 489 works as unrelated to
    # their own author. Group on the ASCII skeleton, keep the cleanest spelling.
    sk = cat.artist.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    nonascii = cat.artist.str.count(r"[^\x00-\x7f]")
    pick = (pd.DataFrame({"sk": sk, "artist": cat.artist, "na": nonascii})
              .groupby(["sk", "artist"], as_index=False)
              .agg(na=("na", "first"), n=("artist", "size"))
              .sort_values(["sk", "na", "n"], ascending=[True, True, False])
              .drop_duplicates("sk").set_index("sk").artist)
    n_before = cat.artist.nunique()
    cat["artist"] = sk.map(pick)
    print(f"artist identities merged: {n_before} -> {cat.artist.nunique()}")

    out = cat.merge(agg, on="file", how="left")
    out["has_artemis"] = out.n_annotations.notna()
    out["n_annotations"] = out.n_annotations.fillna(0).astype(int)

    # ---- write -----------------------------------------------------------
    out.to_parquet("merged_paintings.parquet", index=False)
    out.drop(columns="utterances").to_csv("merged_paintings.csv", index=False)
    art[["file", "art_style", "emotion", "utterance", "repetition"]] \
        .to_parquet("artemis_utterances.parquet", index=False)

    print(f"\npaintings in catalog    : {len(out):,}")
    print(f"  with ARTEMIS          : {out.has_artemis.sum():,} "
          f"({out.has_artemis.mean()*100:.2f}%)")
    print(f"  image-only            : {(~out.has_artemis).sum():,}")
    print(f"  artist known          : {out.artist.notna().sum():,}")
    print(f"unique artists          : {out.artist.nunique():,}")
    print(f"utterances kept         : {len(art):,}")
    print("\nwrote merged_paintings.parquet / .csv, artemis_utterances.parquet")


if __name__ == "__main__":
    main()
