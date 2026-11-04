import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import SparseConceptRepresentation
from bertopic.representation._concepts import load_default_vocabulary


N_CONCEPTS, DIM = 50, 64


@pytest.fixture(scope="module")
def concepts():
    rng = np.random.default_rng(42)
    vocabulary = [f"concept{i}" for i in range(N_CONCEPTS)]
    concept_embeddings = rng.normal(size=(N_CONCEPTS, DIM))
    model = SparseConceptRepresentation(vocabulary, concept_embeddings=concept_embeddings, image_mean=np.zeros(DIM))
    return vocabulary, concept_embeddings, model._concept_dictionary()


def test_decompose_recovers_planted_concepts(concepts):
    vocabulary, concept_embeddings, dictionary = concepts
    rng = np.random.default_rng(0)
    planted = np.array([rng.choice(N_CONCEPTS, size=3, replace=False) for _ in range(20)])
    weights = np.zeros((20, N_CONCEPTS))
    np.put_along_axis(weights, planted, rng.uniform(1, 2, size=planted.shape), axis=1)
    embeddings = weights @ dictionary

    model = SparseConceptRepresentation(
        vocabulary, concept_embeddings=concept_embeddings, image_mean=np.zeros(DIM), l1_penalty=0.01
    )
    recovered = np.argsort(-model.decompose(embeddings), axis=1)[:, :3]
    assert all(set(r) == set(p) for r, p in zip(recovered, planted))


def test_decompose_matches_splice_solver(concepts):
    vocabulary, concept_embeddings, dictionary = concepts
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(5, DIM))
    l1_penalty = 0.25

    model = SparseConceptRepresentation(vocabulary, concept_embeddings=concept_embeddings, l1_penalty=l1_penalty)
    weights = model.decompose(embeddings)

    # SpLiCE's `skl` solver: one Lasso per image with alpha = l1_penalty / (2 * dim)
    centered = normalize(normalize(embeddings) - normalize(embeddings).mean(axis=0))
    lasso = Lasso(alpha=l1_penalty / (2 * DIM), positive=True, fit_intercept=False, tol=1e-10, max_iter=100_000)
    expected = np.stack([lasso.fit(dictionary.T, z).coef_ for z in centered])
    np.testing.assert_allclose(weights, expected, atol=1e-6)


def test_decompose_float32(concepts):
    vocabulary, concept_embeddings, _ = concepts
    embeddings = np.random.default_rng(3).normal(size=(5, DIM)).astype(np.float32)

    model = SparseConceptRepresentation(vocabulary, concept_embeddings=concept_embeddings.astype(np.float32))
    assert model.decompose(embeddings).shape == (5, N_CONCEPTS)


def test_images_only_topic_model(concepts):
    vocabulary, concept_embeddings, dictionary = concepts
    rng = np.random.default_rng(2)

    # Two groups of images built from disjoint concepts
    groups = [range(0, 5), range(5, 10)]
    weights = np.zeros((100, N_CONCEPTS))
    for i in range(100):
        weights[i, rng.choice(groups[i % 2], size=2, replace=False)] = 1
    embeddings = weights @ dictionary + 0.01 * rng.normal(size=(100, DIM))

    representation_model = SparseConceptRepresentation(
        vocabulary, concept_embeddings=concept_embeddings, image_mean=np.zeros(DIM), l1_penalty=0.01
    )
    topic_model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=KMeans(n_clusters=2, random_state=42),
        representation_model=representation_model,
    )
    topics, _ = topic_model.fit_transform(
        documents=None, embeddings=embeddings, images=[f"image{i}.jpg" for i in range(100)]
    )

    for topic in set(topics):
        words = {word for word, _ in topic_model.get_topic(topic)[:5]}
        group = groups[np.bincount([i % 2 for i, t in enumerate(topics) if t == topic]).argmax()]
        assert words == {vocabulary[i] for i in group}


def test_requires_embeddings_or_model():
    with pytest.raises(ValueError):
        SparseConceptRepresentation(["dog", "cat"])
    with pytest.raises(ValueError):
        SparseConceptRepresentation(["dog", "cat"], concept_embeddings=np.zeros((3, 4)))


def test_default_vocabulary():
    vocabulary = load_default_vocabulary()
    assert len(vocabulary) == len(set(vocabulary)) == 3000
    assert vocabulary[:3] == ["a", "on", "of"]

    model = SparseConceptRepresentation(concept_embeddings=np.zeros((3000, 4)))
    assert model.vocabulary.tolist() == vocabulary
