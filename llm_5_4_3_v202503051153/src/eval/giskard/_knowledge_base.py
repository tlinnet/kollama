from giskard.rag.testset_generation import KnowledgeBase
import numpy as np
import pandas as pd


class _KnowledgeBase(KnowledgeBase):
    """
    A custom subclass of KnowledgeBase from giskard.rag that overrides the _embeddings property.

    Instead of computing embeddings, this class utilizes precomputed embeddings
    provided by the vector store.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        documents_col: str,
        embeddings_col: str,
        embeddings_model,
    ):
        super().__init__(
            data=data, columns=[documents_col], embedding_model=embeddings_model
        )
        # initialize embeddings so they don't get recalculated in the _embeddings property
        vectors = data[embeddings_col]
        self._embeddings_inst = np.array([np.array(vec) for vec in vectors])
        for doc, emb in zip(self._documents, self._embeddings_inst):
            doc.embeddings = emb
