import numpy as np
import pandas as pd

from importlib.resources import files
from scipy.sparse import csr_matrix
from typing import Any, List, Mapping, Tuple
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize

from bertopic.backend._utils import select_backend
from bertopic.representation._base import BaseRepresentation


class SparseConceptRepresentation(BaseRepresentation):
    r"""Represent images as sparse, non-negative combinations of textual concepts.

    This follows SpLiCE (Bhalla et al., 2024) and offers an alternative to
    captioning images with a generative model when no documents are passed.
    Each image embedding is decomposed into a few concepts from a vocabulary
    that is embedded with the text encoder of the *same* CLIP model that was used
    to embed the images. The decomposition solves a non-negative Lasso:

    $$\min_{w \geq 0} \|z - Cw\|_2^2 + \lambda \|w\|_1$$

    where $z$ is a centered image embedding and $C$ the centered concept dictionary.

    Every image is then turned into a bag-of-concepts pseudo-document in which each
    concept is repeated proportionally to its weight. The regular c-TF-IDF procedure
    is applied to these pseudo-documents, so every image in a cluster contributes to
    its topic representation and the representation is directly derived from the
    embeddings that were clustered.

    Arguments:
        vocabulary: The concepts to decompose images into. If None, the 3,000 most
                    frequent words of SpLiCE's MS-COCO vocabulary are used, which
                    are shipped with BERTopic. Single-word concepts are advised
                    since multi-word concepts are split into separate tokens by
                    the vectorizer.
        embedding_model: The CLIP model used to embed the images. It is used to
                         embed the `vocabulary` with its text encoder. Can be any
                         embedding model supported by BERTopic, such as
                         `MultiModalBackend("clip-ViT-B-32")` or `"clip-ViT-B-32"`.
                         Not needed if `concept_embeddings` is passed.
        concept_embeddings: Precomputed embeddings of `vocabulary` with shape
                            (n_concepts, embedding_dim).
        l1_penalty: The sparsity penalty $\lambda$. Higher values result in fewer
                    active concepts per image. Corresponds to `l1_penalty` in SpLiCE.
        image_mean: The mean image embedding used for centering. If None, the mean
                    of the (normalized) input image embeddings is used.
        resolution: The number of tokens used to quantize the concept weights of
                    a single image into a pseudo-document.
        n_jobs: The number of parallel jobs used to decompose the images.

    Usage:

    ```python
    from bertopic import BERTopic
    from bertopic.backend import MultiModalBackend
    from bertopic.representation import SparseConceptRepresentation

    embedding_model = MultiModalBackend("clip-ViT-B-32")
    representation_model = SparseConceptRepresentation(embedding_model=embedding_model)

    topic_model = BERTopic(embedding_model=embedding_model, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(documents=None, images=images)
    ```

    You can also pass your own concepts, for example to focus on a specific domain:

    ```python
    representation_model = SparseConceptRepresentation(
        vocabulary=["dog", "cat", "beach", "snow", ...],
        embedding_model=embedding_model,
    )
    ```

    Reference:
        Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F. P., & Lakkaraju, H. (2024).
        Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE). NeurIPS 2024.
    """

    def __init__(
        self,
        vocabulary: List[str] | None = None,
        embedding_model: Any = None,
        concept_embeddings: np.ndarray = None,
        l1_penalty: float = 0.25,
        image_mean: np.ndarray = None,
        resolution: int = 20,
        n_jobs: int | None = None,
    ):
        if embedding_model is None and concept_embeddings is None:
            raise ValueError("Please pass either an `embedding_model` or `concept_embeddings`.")
        if vocabulary is None:
            vocabulary = load_default_vocabulary()
        if concept_embeddings is not None and len(concept_embeddings) != len(vocabulary):
            raise ValueError("`concept_embeddings` should have one row per concept in `vocabulary`.")

        self.vocabulary = np.array(vocabulary)
        self.embedding_model = embedding_model
        self.concept_embeddings = concept_embeddings
        self.l1_penalty = l1_penalty
        self.image_mean = image_mean
        self.resolution = resolution
        self.n_jobs = n_jobs

        # BERTopic converts images to text with any representation model that has this attribute set
        self.image_to_text_model = True

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """The c-TF-IDF representation of the concept pseudo-documents is already the topic representation.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            topics: The unchanged topic representations
        """
        return topics

    def decompose(self, embeddings: np.ndarray) -> np.ndarray:
        """Decompose image embeddings into sparse, non-negative concept weights.

        Arguments:
            embeddings: Image embeddings with shape (n_images, embedding_dim)

        Returns:
            weights: Concept weights with shape (n_images, n_concepts)
        """
        # The Lasso solver requires the images and dictionary to share a dtype
        dictionary = self._concept_dictionary().astype(np.float64)

        # Center the images to close the modality gap between CLIP's image and text spaces
        embeddings = normalize(np.asarray(embeddings, dtype=np.float64))
        image_mean = embeddings.mean(axis=0) if self.image_mean is None else self.image_mean
        embeddings = normalize(embeddings - image_mean)

        # `sparse_encode` divides alpha by the embedding dimension, so alpha = l1_penalty / 2
        # matches SpLiCE's `Lasso(alpha=l1_penalty / (2 * dim))`
        weights = sparse_encode(
            embeddings,
            dictionary,
            algorithm="lasso_cd",
            alpha=self.l1_penalty / 2,
            positive=True,
            n_jobs=self.n_jobs,
        )
        return weights

    def image_to_text(self, documents: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """Convert images to bag-of-concepts pseudo-documents."""
        weights = self.decompose(embeddings)

        # Quantize the weights of each image into token counts
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        counts = np.rint(weights / totals * self.resolution).astype(int)

        documents = documents.copy()
        documents["Document"] = [" ".join(np.repeat(self.vocabulary, row)) for row in counts]
        return documents

    def _concept_dictionary(self) -> np.ndarray:
        """Embed, normalize, and center the concept vocabulary."""
        if self.concept_embeddings is None:
            embedding_model = select_backend(self.embedding_model)
            self.concept_embeddings = embedding_model.embed_words(self.vocabulary.tolist())

        dictionary = normalize(self.concept_embeddings)
        dictionary = normalize(dictionary - dictionary.mean(axis=0))
        return dictionary


def load_default_vocabulary() -> List[str]:
    """Load the 3,000 most frequent words of SpLiCE's MS-COCO vocabulary, most frequent first."""
    text = files("bertopic.representation").joinpath("_concepts_mscoco.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line and not line.startswith("#")]
